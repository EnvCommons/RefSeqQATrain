# RefSeqTrain

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/RefSeqTrain)

## Description

RefSeqTrain is a training environment for genomics question answering about NCBI RefSeq and Gene database records. Agents are given questions about specific verifiable facts from RefSeq gene, transcript, and protein records and must use web search to find and verify answers from NCBI.

## Capabilities

- Researching gene, transcript, and protein metadata from NCBI RefSeq
- Extracting specific information from genomic records (sequence lengths, exon counts, chromosomal locations, CDS ranges, etc.)
- Navigating NCBI Gene, nucleotide, and protein databases
- Cross-species genomic queries across human, mouse, rat, zebrafish, and other model organisms

## Compute Requirements

No sandbox or special compute requirements. Uses external web search (Tavily API) for NCBI retrieval.

## License

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

Training tasks are distributed across 10 genomics domains:

| Domain | Description |
|--------|-------------|
| `transcript_metadata` | mRNA lengths, accession types (NM/XM/NR) |
| `protein_metadata` | Protein lengths, accession types (NP/XP) |
| `gene_transcript_relationships` | Isoform counts, transcript variants |
| `coding_sequence` | CDS ranges, reading frames |
| `exon_structure` | Exon counts, exon architecture |
| `chromosomal_location` | Chromosome, band, coordinates, strand |
| `gene_nomenclature` | Full names, symbols, aliases |
| `cross_species` | Orthologs across model organisms |
| `protein_features` | Domains, signal peptides, annotations |
| `functional_annotation` | Gene summaries, RefSeq status, pathways |

## Reward Structure

Sparse, binary reward:
- **1.0** for correct answers (as judged by LLM grader)
- **0.0** for incorrect or unsure answers

Grading uses semantic equivalence checking via gpt-5-mini.

## Data

Ground-truth data consists of QA pairs derived from NCBI RefSeq and Gene database records. Each task includes a question, expected answer, source NCBI URL, accession, key passage, and domain. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using Tavily. Returns titles, URLs, and snippets. |
| `fetch_url` | Fetch full text content from a URL. Supports pagination for long documents. |
| `submit_answer` | Submit a final answer with explanation. Triggers LLM grading and ends the episode. |

## Time Horizon

Multi-turn. Agents typically perform several web searches and URL fetches before submitting an answer.

## Other Environment Requirements

This environment requires the following API keys passed via the `secrets` parameter:
- `openai_api_key`: For LLM-based answer grading
- `tavily_api_key`: For web search and URL content extraction

## Safety

RefSeqTrain focuses on factual information retrieval from publicly available NCBI genomic records. The environment does not involve access to non-public data or sensitive personal genomic information.

## Citations

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
