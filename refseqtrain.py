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
import httpx
from pydantic import BaseModel, Field
from typing import Dict, List
from urllib.parse import urlparse, parse_qs

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

    async def _fetch_ncbi_authoritative(self, url: str) -> str | None:
        """
        Retrieve authoritative NCBI record content over plain HTTP.

        Tavily's extract() scrapes rendered HTML, but NCBI's nuccore/gene/protein
        pages serve only a JavaScript-disabled shell (the FEATURES/exon table is
        client-rendered), and Tavily drops non-HTML E-utilities bodies entirely
        ("No content extracted"). For any NCBI URL we instead resolve the
        accession and pull the GenBank flat file directly from E-utilities
        efetch, which returns the full feature table (incl. exon records) as
        text/plain over a normal HTTP GET.

        Returns the flat-file text, or None if this isn't an NCBI URL / no
        accession could be resolved (caller then falls back to Tavily).
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "ncbi.nlm.nih.gov" not in host:
            return None

        db = "nuccore"
        acc = None

        # E-utilities efetch links: honour their own db/id/rettype query params.
        if "eutils" in host and "efetch" in parsed.path:
            qs = parse_qs(parsed.query)
            ids = qs.get("id", [])
            acc = ids[0].split(",")[0] if ids else None
            db = (qs.get("db", [db])[0]) or db
        else:
            # Web pages like /nuccore/NM_004958.4, /gene/2475, /protein/NP_...
            # Resolve the database from the path and the accession from the
            # path tail (strip any ?report=... / #... fragments).
            m = re.search(r"/(nuccore|gene|protein|nucleotide)/([^/?#]+)", parsed.path)
            if m:
                path_db = m.group(1)
                acc = m.group(2)
                db = {"nucleotide": "nuccore"}.get(path_db, path_db)

        if not acc:
            return None

        # For sequence records (nuccore/protein) pull the full GenBank flat file
        # so the FEATURES/exon table is present. For a numeric Gene ID there is
        # no flat file; rettype=gene_table returns the per-transcript exon
        # listing (genomic/gene intervals, exon counts and lengths), which is
        # the authoritative gene-level exon source.
        eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params_q = {"db": db, "id": acc, "retmode": "text"}
        params_q["rettype"] = "gene_table" if db == "gene" else "gb"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(eutils, params=params_q)
            resp.raise_for_status()
            text = resp.text
        return text or None

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the text content from a specific URL.

        For NCBI URLs (nuccore/nucleotide, gene, protein pages, or E-utilities
        efetch links) this retrieves the authoritative GenBank flat file via
        NCBI E-utilities, so the FEATURES table — including exon records, CDS
        ranges, and other annotations — is delivered as plain text. (The public
        NCBI web pages only serve a JavaScript-disabled shell, so scraping their
        HTML yields no record content.) For all other URLs it uses Tavily's
        extract method to pull rendered text.

        Content is paginated - use the page parameter to retrieve additional pages.
        """
        PAGE_SIZE = 10000

        try:
            # Prefer the authoritative E-utilities flat file for NCBI records
            raw_content = None
            try:
                raw_content = await self._fetch_ncbi_authoritative(params.url)
            except Exception:
                raw_content = None

            if raw_content is None:
                response = await self.tavily_client.extract(
                    urls=[params.url],
                    extract_depth="advanced",
                    format="text",
                )
                results = response.get("results", [])
                if not results:
                    # Tavily produced no result object at all — usually a fetch
                    # failure (DNS/timeout/blocked) or an unsupported URL.
                    return ToolOutput(
                        blocks=[TextBlock(type="text", text=(
                            f"Could not fetch {params.url}: the extractor returned "
                            f"no result. The URL may be unreachable, blocked, or "
                            f"invalid. Try a different source or an NCBI record URL."
                        ))],
                        metadata={"url": params.url, "results": []},
                        reward=0.0,
                        finished=False
                    )
                raw_content = results[0].get("raw_content", "") or ""
                if not raw_content.strip():
                    # A result came back but with no usable text — typically a
                    # JavaScript-gated page that renders content client-side, so
                    # the extractor saw only an empty shell. Surface that so the
                    # agent can pick a different (e.g. API/flat-file) source.
                    return ToolOutput(
                        blocks=[TextBlock(type="text", text=(
                            f"No readable text could be extracted from {params.url}. "
                            f"The page appears to be JavaScript-gated or otherwise "
                            f"served no content to the extractor. Try the record's "
                            f"NCBI page or a direct data/API endpoint instead."
                        ))],
                        metadata={"url": params.url, "results": results, "empty_content": True},
                        reward=0.0,
                        finished=False
                    )

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
