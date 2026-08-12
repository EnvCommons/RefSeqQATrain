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


from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, terminal, tool
from openreward.toolsets import WebToolset
from openreward.toolsets._web_common import WebFetchParams

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


MAX_FETCH_CHARS = 100_000


class NcbiWebToolset(WebToolset):
    """WebToolset whose web_fetch prefers NCBI's authoritative records.

    Subclassing rather than defining web_fetch on the environment: the framework
    rejects a tool name defined in both an environment and its toolset, so the
    override has to live here. web_search is inherited unchanged.

    NCBI's nuccore/gene/protein pages serve a JavaScript-only shell, and their
    E-utilities responses are not HTML, so a generic extractor returns nothing
    usable for exactly the records these tasks are about. Those URLs go to
    E-utilities; everything else falls through to the configured backend.
    """

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        try:
            raw = await self.env._fetch_ncbi_authoritative(params.url)
        except Exception:
            raw = None

        if raw is None:
            return await super().web_fetch(params)

        text = raw[:MAX_FETCH_CHARS]
        if len(raw) > MAX_FETCH_CHARS:
            text += "\n... (truncated)"
        return ToolOutput(
            blocks=[TextBlock(type="text", text=f"Content from {params.url}:\n\n{text}")],
            metadata={"url": params.url, "source": "ncbi-eutilities", "total_length": len(raw)},
            reward=0.0,
            finished=False,
        )


class SubmitAnswerParams(BaseModel):
    # A terminal tool takes at most one field — the assistant's final message,
    # which carries both the answer and whatever reasoning it chose to show.
    answer: str = Field(
        ...,
        description="The assistant's final message, containing the answer to the RefSeq question"
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
    3. Uses web_fetch tool to get detailed record content from NCBI
    4. Writes its final answer as a plain message (no tool call)
    5. The harness routes that message to the hidden @terminal tool, which
       grades it with an LLM judge and returns a reward (1.0 / 0.0)
    """

    # web_search / web_fetch come from the SDK rather than being hand-rolled here.
    # Which provider answers is process configuration (OPENREWARD_SEARCH_BACKEND,
    # default "backsearch"), so changing search provider needs no change here.
    #
    # The toolset owns the error split too: an unfetchable page stays tool output
    # the agent can act on, while a missing key or exhausted quota raises so the
    # rollout ends with a blank reward rather than a score that reads as a bad answer.
    toolsets = [NcbiWebToolset]

    # Search hits keep their snippets, as the prompt promises. Off in the SDK by
    # default, which would force a fetch per candidate just to triage results.
    web_include_snippets = True

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = RefSeqQATaskSpec.model_validate(task_spec)

        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets parameter for LLM grading. "
                "Pass secrets={'openai_api_key': 'sk-...'} when creating session."
            )

        # Read live by WebToolset on every tool call, so the search backend takes its
        # credentials from the session rather than the server process. The configured
        # backend picks the key it needs: `api_key` for backsearch, `tavily_api_key`
        # for tavily. No up-front check — which key is required depends on the backend.
        self.search_secrets = secrets

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)

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
        return [TextBlock(type="text", text=self.config.question + "\n\n" + "Research the question with the tools available, then reply with your final answer as an ordinary message (no tool call). That message is graded.")]

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
        eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params_q = {"db": db, "id": acc, "retmode": "text"}
        params_q["rettype"] = "gene_table" if db == "gene" else "gb"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(f"{eutils_base}/efetch.fcgi", params=params_q)
            resp.raise_for_status()
            text = resp.text

            # The gene_table report omits the cytogenetic/genetic Map Location
            # and chromosome -- the source for chromosomal_location questions.
            # Prepend the gene esummary (the same field the dataset's expected
            # answers are built from: esummary 'maplocation') so those tasks
            # have a tool path to the answer.
            if db == "gene":
                try:
                    sresp = await client.get(
                        f"{eutils_base}/esummary.fcgi",
                        params={"db": "gene", "id": acc, "retmode": "json"},
                    )
                    sresp.raise_for_status()
                    g = sresp.json().get("result", {}).get(acc, {})
                    if g and "error" not in g:
                        header = (
                            "NCBI Gene summary\n"
                            f"Gene ID: {acc}\n"
                            f"Symbol: {g.get('name', '')}\n"
                            f"Description: {g.get('description', '')}\n"
                            f"Organism: {g.get('organism', {}).get('scientificname', '')}\n"
                            f"Chromosome: {g.get('chromosome', '')}\n"
                            f"Map Location: {g.get('maplocation', '')}\n\n"
                        )
                        text = header + text
                except Exception:
                    pass  # fall back to the gene_table content alone

        return text or None

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

    @terminal
    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Grade the assistant's final message against the expected RefSeq answer.

        Terminal tool: it is hidden from the agent, which simply writes its
        answer as an ordinary message. The harness routes that text here for
        LLM grading, and the episode ends.
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
