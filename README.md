# RefSeqTrain

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/RefSeqTrain)

## Description

RefSeqTrain is an ORS training environment for genomics question answering about NCBI [RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) and [Gene](https://www.ncbi.nlm.nih.gov/gene/) database records. Each question asks about a specific verifiable fact from a gene, transcript, or protein record (e.g. sequence lengths, exon counts, chromosomal locations, CDS ranges, protein domains). Questions are designed to be specific enough that they can only be answered by looking up the correct NCBI record, and answers require navigating the RefSeq and Gene databases via web search.

## Capabilities

- Question answering from NCBI RefSeq and Gene database records
- Web search and information retrieval from genomic databases
- Multi-step research: searching, reading NCBI records, and extracting precise facts
- Cross-species genomic queries across human, mouse, rat, zebrafish, fruit fly, and nematode

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There is one split: **train** with 1,000 tasks spanning 10 genomics domains:

| Domain | Count | Description |
|--------|-------|-------------|
| `transcript_metadata` | 100 | mRNA lengths, accession types (NM/XM/NR) |
| `protein_metadata` | 100 | Protein lengths, accession types (NP/XP) |
| `gene_transcript_relationships` | 100 | Isoform counts, transcript variants |
| `coding_sequence` | 100 | CDS ranges, reading frames |
| `exon_structure` | 100 | Exon counts, exon architecture |
| `chromosomal_location` | 100 | Chromosome, band, coordinates |
| `gene_nomenclature` | 100 | Full names, symbols, aliases |
| `cross_species` | 100 | Orthologs across model organisms |
| `protein_features` | 100 | Domains, signal peptides, annotations |
| `functional_annotation` | 100 | Gene summaries, pathways, map locations |

Each task provides a question and metadata (accession, source NCBI URL, domain, question type). The agent prompt contains only the question; the agent must find the answer through web search and NCBI record retrieval.

## Reward Structure

Reward is sparse and binary, emitted only when the agent calls `submit_answer` (which ends the episode). The `web_search` and `web_fetch` tools always return reward 0.0 and do not end the episode.

On submission, the agent's answer is evaluated by an LLM grader (gpt-5-mini) that checks semantic equivalence against the reference answer. The grader accounts for equivalent numeric formats, abbreviations, and minor formatting differences. Empty or whitespace-only submissions receive reward 0.0 without invoking the grader.

- **1.0**: Submitted answer is semantically equivalent to the reference answer
- **0.0**: Submitted answer is incorrect, missing, or empty

## Data

Data consists of a single JSONL file containing 1,000 QA pairs generated from NCBI RefSeq and Gene database records. Each row contains a question, answer, source NCBI URL, accession, key passage from the record, domain, and question type. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web. Returns up to 5 results with titles, URLs, and snippets. |
| `web_fetch` | Fetch full text content from a URL. Supports pagination for long documents. |
| `submit_answer` | Submit a final answer with explanation for LLM grading. Ends the episode. |

Note that the `web_fetch` and `web_search` tools require Tavily, but are optional. If you want to use a different provider for search you can exclude these tools and use external tools instead.

## Time Horizon

Multi-turn. Agents can perform multiple web searches and URL fetches before submitting a final answer.

## Environment Difficulty

[To be determined]

## Other Environment Requirements

- OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.
- Search credentials — whichever the configured backend needs: `api_key` for the default backsearch backend, or `tavily_api_key` when the server runs with `OPENREWARD_SEARCH_BACKEND=tavily`. Both fall back to the server process environment (`OPENREWARD_API_KEY` / `TAVILY_API_KEY`).

## Safety

Agents in RefSeqTrain answer genomics questions using web search in a standard environment. The environment focuses on factual information retrieval from publicly available NCBI genomic records and does not involve access to non-public data or sensitive personal genomic information. The environment does not present direct safety risks.

## Citations

RefSeqTrain uses data derived from the NCBI RefSeq database. Please cite the original RefSeq publication:

```bibtex
@article{oleary2016refseq,
  title={Reference sequence (RefSeq) database at NCBI: current status, taxonomic expansion, and functional annotation},
  author={O'Leary, Nuala A and Wright, Mathew W and Brister, J Rodney and others},
  journal={Nucleic Acids Research},
  volume={44},
  number={D1},
  pages={D733--D745},
  year={2016},
  publisher={Oxford University Press}
}
```

```bibtex
@dataset{GRRefSeqTrain,
  author    = {General Reasoning Inc. Team},
  title     = {RefSeqTrain},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/RefSeqTrain}
}
```
