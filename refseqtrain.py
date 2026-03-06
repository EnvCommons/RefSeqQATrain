"""
RefSeqTrain Environment - RefSeq/genomics question answering with web search

A training environment with 1,000 RefSeq QA pairs across 10 genomics domains.
Agents research questions using web search and URL fetching, then submit answers
for LLM-based grading. Questions cover diverse genomics domains including
transcript metadata, protein metadata, gene-transcript relationships,
coding sequences, exon structure, chromosomal location, gene nomenclature,
cross-species comparisons, protein features, and functional annotation.
"""

import json
import re
import openai
from pydantic import BaseModel, Field
from typing import Dict, List

from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

from constants import REFSEQTRAIN_JSONL


# Grader prompt template for LLM-based answer evaluation
GRADER_PROMPT_TEMPLATE = """You are a helpful assistant that evaluates the correctness of an answer.

Consider the question, the expected correct answer, and the submitted answer.
Your task is to determine if the submitted answer is correct.

Be rigorous but reasonable in your evaluation:
- Accept answers that are semantically/numerically equivalent, even if phrased slightly differently (unless the question explicitly specifies required elements or details)
- Accept reasonable approximations for numerical values (unless the question explicitly specifies required precision)
- Accept answers that clearly and uniquely capture the core concept even if they are presented in a slightly different way

Your output should include the following fields:
- rationale: A short explanation of your evaluation.
- result: MUST be one of the following words: "correct", "incorrect", or "unsure".

## QUESTION ##
{question}

## EXPECTED ANSWER ##
{correct_answer}

## SUBMITTED ANSWER ##
{answer}

## EVALUATION ##"""


# Pydantic schemas
class RefSeqQATaskSpec(BaseModel):
    id: str
    question: str
    answer: str
    accession: str
    source_url: str
    key_passage: str
    domain: str
    question_type: str


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query to find RefSeq/gene information on NCBI")


class FetchUrlInput(BaseModel):
    url: str = Field(..., description="URL to fetch (e.g., NCBI Gene, nucleotide, or protein page)")
    page: int = Field(default=1, description="Page number to retrieve (1-indexed). Each page contains ~10,000 characters.")


class SubmitAnswerParams(BaseModel):
    explanation: str = Field(
        ...,
        description="Your reasoning showing how you found and verified the answer (2-4 sentences)"
    )
    answer: str = Field(
        ...,
        description="The precise answer to the RefSeq question"
    )


def load_refseqtrain_data() -> Dict[str, List[Dict]]:
    """Load RefSeqTrain JSONL dataset."""
    print(f"Loading RefSeqTrain data from: {REFSEQTRAIN_JSONL}")

    if not REFSEQTRAIN_JSONL.exists():
        raise FileNotFoundError(
            f"RefSeqTrain JSONL not found at {REFSEQTRAIN_JSONL}. "
            f"Please ensure the dataset file exists."
        )

    tasks = []
    with open(REFSEQTRAIN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append({
                    "id": data["id"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "accession": data["accession"],
                    "source_url": data["source_url"],
                    "key_passage": data["key_passage"],
                    "domain": data["domain"],
                    "question_type": data["question_type"],
                })
            except Exception as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    print(f"Successfully loaded {len(tasks)} tasks")
    return {"train": tasks}


# Load dataset once at module level
ALL_DATA = load_refseqtrain_data()


class RefSeqTrain(Environment):
    """
    RefSeqTrain environment: RefSeq/genomics QA with web search and LLM grading.

    Agent workflow:
    1. Receives a question about a specific RefSeq/Gene record
    2. Uses web_search tool to find relevant NCBI information
    3. Uses fetch_url tool to get detailed record content from NCBI
    4. Submits answer with explanation for LLM-based grading
    5. Receives reward (1.0 correct, 0.0 incorrect) and feedback
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = RefSeqQATaskSpec.model_validate(task_spec)

        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets parameter for LLM grading. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "tavily_api_key required in secrets parameter for web search. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            raise ValueError(f"Unknown split: {split}. Available splits: train")

        return [
            {
                "id": task["id"],
                "question": task["question"],
                "answer": task["answer"],
                "accession": task["accession"],
                "source_url": task["source_url"],
                "key_passage": task["key_passage"],
                "domain": task["domain"],
                "question_type": task["question_type"],
            }
            for task in ALL_DATA["train"]
        ]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=self.config.question + "\n\n" + "Use the submit_answer tool to submit your answer when ready.")]

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web for RefSeq/gene information using Tavily.
        Returns search results with titles, URLs, and snippets.
        """
        try:
            response = await self.tavily_client.search(
                query=params.query,
                search_depth="basic",
                max_results=5
            )

            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text="No search results found. Try a different query.")],
                    metadata={"query": params.query, "results": []},
                    reward=0.0,
                    finished=False
                )

            display_parts = [f"Search results for: {params.query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("content", "")
                display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

            display_text = "\n".join(display_parts)

            return ToolOutput(
                blocks=[TextBlock(type="text", text=display_text)],
                metadata={
                    "query": params.query,
                    "results": results,
                    "count": len(results)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Web search failed: {str(e)}")],
                metadata={"query": params.query, "error": str(e)},
                reward=0.0,
                finished=False
            )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the text content from a specific URL using Tavily's extract method.
        Use this to read NCBI Gene, nucleotide, or protein pages.
        Content is paginated - use the page parameter to retrieve additional pages.
        """
        PAGE_SIZE = 10000

        try:
            response = await self.tavily_client.extract(urls=[params.url])

            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text=f"No content extracted from {params.url}")],
                    metadata={"url": params.url, "results": []},
                    reward=0.0,
                    finished=False
                )

            result = results[0]
            raw_content = result.get("raw_content", "")
            total_length = len(raw_content)

            total_pages = max(1, (total_length + PAGE_SIZE - 1) // PAGE_SIZE)
            page = max(1, min(params.page, total_pages))

            start_idx = (page - 1) * PAGE_SIZE
            end_idx = min(start_idx + PAGE_SIZE, total_length)
            page_content = raw_content[start_idx:end_idx]

            if total_pages == 1:
                display_text = f"Content from {params.url}:\n\n{page_content}"
            else:
                display_text = f"Content from {params.url} (Page {page}/{total_pages}):\n\n{page_content}"
                if page < total_pages:
                    display_text += f"\n\n[Use fetch_url with page={page + 1} to see more content]"

            return ToolOutput(
                blocks=[TextBlock(type="text", text=display_text)],
                metadata={
                    "url": params.url,
                    "page": page,
                    "total_pages": total_pages,
                    "total_length": total_length,
                    "page_start": start_idx,
                    "page_end": end_idx
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Failed to fetch URL: {str(e)}")],
                metadata={"url": params.url, "error": str(e)},
                reward=0.0,
                finished=False
            )

    async def _grade_answer(self, answer: str) -> Dict:
        """Grade answer using gpt-5-mini LLM grader."""
        grader_prompt = GRADER_PROMPT_TEMPLATE.format(
            question=self.config.question,
            correct_answer=self.config.answer,
            answer=answer
        )

        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}],
        )

        grading_text = response.choices[0].message.content or ""

        lower_text = grading_text.lower()
        result_match = re.search(r'result[:\s]+(\w+)', lower_text)
        if result_match:
            result_value = result_match.group(1).strip()
            is_correct = result_value == "correct"
        else:
            is_correct = "correct" in lower_text and "incorrect" not in lower_text and "unsure" not in lower_text

        return {
            "is_correct": is_correct,
            "grading_response": grading_text
        }

    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Submit your final answer to the RefSeq question.

        This tool grades your answer using an LLM judge and returns a reward.
        The episode ends after calling this tool.
        """
        grading_result = await self._grade_answer(params.answer)

        reward = 1.0 if grading_result["is_correct"] else 0.0
        result_status = "Correct" if grading_result["is_correct"] else "Incorrect"

        display_text = f"""{result_status}

Grading Analysis:
{grading_result['grading_response']}

Reward: {reward:.1f}

Expected Answer: {self.config.answer}
Your Answer: {params.answer}

Source: {self.config.source_url}"""

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "task_id": self.config.id,
                "is_correct": grading_result["is_correct"],
                "grading_response": grading_result["grading_response"],
                "submitted_answer": params.answer,
                "submitted_explanation": params.explanation,
                "correct_answer": self.config.answer,
                "question": self.config.question,
                "source_url": self.config.source_url,
                "accession": self.config.accession,
                "domain": self.config.domain,
                "question_type": self.config.question_type,
            },
            reward=reward,
            finished=True
        )
