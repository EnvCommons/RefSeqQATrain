"""
RefSeqTrain Dataset Generator

Generates 1,000 RefSeq QA pairs across 10 genomics domains by:
1. Discovering genes via NCBI E-utilities (esearch)
2. Fetching structured data via E-utilities (esummary, efetch)
3. Parsing precise facts from XML/JSON responses
4. Using gpt-4.1 to generate verifiable QA pairs from structured facts
5. Validating and deduplicating results
6. Writing to JSONL format

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_dataset.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ============= Configuration =============

TARGET_PER_DOMAIN = 100  # 100 per domain = 1000 total
OUTPUT_JSONL = Path(__file__).parent / "data" / "refseqtrain.jsonl"
PROGRESS_JSONL = Path(__file__).parent / "refseqtrain_progress.jsonl"
REFERENCE_FILE = Path(__file__).parent / "reference.txt"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_LIMIT_DELAY = 0.5  # Stay well under 3 req/sec without API key

DOMAINS = [
    "transcript_metadata",
    "protein_metadata",
    "gene_transcript_relationships",
    "coding_sequence",
    "exon_structure",
    "chromosomal_location",
    "gene_nomenclature",
    "cross_species",
    "protein_features",
    "functional_annotation",
]

# Search queries for gene discovery via E-utilities esearch
# Each domain uses specific Entrez queries to find relevant genes
DOMAIN_GENE_QUERIES = {
    "transcript_metadata": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cancer[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND kinase[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND receptor[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND kinase[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND transporter[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND ion channel[All Fields]"),
        ("gene", "Danio rerio[Orgn] AND genetype protein coding[Properties] AND alive[prop]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND enzyme[All Fields]"),
        ("gene", "Rattus norvegicus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND neurotransmitter[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND growth factor[All Fields]"),
    ],
    "protein_metadata": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND tumor suppressor[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND apoptosis[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND transcription factor[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND apoptosis[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cytokine[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND protease[All Fields]"),
        ("gene", "Drosophila melanogaster[Orgn] AND genetype protein coding[Properties] AND alive[prop]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND DNA repair[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cell cycle[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND immune[All Fields]"),
    ],
    "gene_transcript_relationships": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND alternative splicing[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND isoform[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND multiple transcript variants[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND alternative splicing[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND splice variant[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND exon skipping[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND long non-coding[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND signal transduction[All Fields]"),
        ("gene", "Rattus norvegicus[Orgn] AND genetype protein coding[Properties] AND alive[prop]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND neurodegenerative[All Fields]"),
    ],
    "coding_sequence": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND oncogene[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND metabolism[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND hormone[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND metabolism[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND phosphatase[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND ubiquitin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND chromatin[All Fields]"),
        ("gene", "Danio rerio[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND development[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND mitochondrial[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND membrane[All Fields]"),
    ],
    "exon_structure": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND developmental[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cardiac[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND neuronal[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND developmental[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND collagen[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND adhesion[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND structural[All Fields]"),
        ("gene", "Rattus norvegicus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND brain[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND extracellular[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND matrix[All Fields]"),
    ],
    "chromosomal_location": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND hereditary[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND syndrome[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND congenital[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND hereditary[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND autosomal[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND X-linked[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND imprinting[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND blood[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND retinal[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND deafness[All Fields]"),
    ],
    "gene_nomenclature": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND solute carrier[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND claudin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cadherin[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cadherin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND aquaporin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND integrin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND myosin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND keratin[All Fields]"),
        ("gene", "Drosophila melanogaster[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND wing[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND histone[All Fields]"),
    ],
    "cross_species": [
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND tumor suppressor[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND kinase[All Fields]"),
        ("gene", "Rattus norvegicus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND receptor[All Fields]"),
        ("gene", "Danio rerio[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND development[All Fields]"),
        ("gene", "Drosophila melanogaster[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND signaling[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND immune[All Fields]"),
        ("gene", "Rattus norvegicus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND cardiac[All Fields]"),
        ("gene", "Danio rerio[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND pigment[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND neural[All Fields]"),
        ("gene", "Caenorhabditis elegans[Orgn] AND genetype protein coding[Properties] AND alive[prop]"),
    ],
    "protein_features": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND signal peptide[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND transmembrane[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND secreted[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND signal peptide[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND domain[All Fields] AND kinase[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND zinc finger[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND immunoglobulin[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND EGF domain[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND coiled-coil[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND SH2 domain[All Fields]"),
    ],
    "functional_annotation": [
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND drug target[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND pharmacogenomics[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND biomarker[All Fields]"),
        ("gene", "Mus musculus[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND knockout[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND therapeutic[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND clinical[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND FDA approved[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND disease[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND pathway[All Fields]"),
        ("gene", "Homo sapiens[Orgn] AND genetype protein coding[Properties] AND alive[prop] AND expression[All Fields]"),
    ],
}


# ============= Pydantic Models =============

class QAPair(BaseModel):
    question: str = Field(..., description="A verifiable question about a RefSeq/Gene record")
    answer: str = Field(..., description="Short precise answer (under 200 chars)")
    accession: str = Field(..., description="RefSeq accession or Gene ID (e.g., NM_000546.6, Gene ID 672)")
    source_url: str = Field(..., description="The NCBI URL")
    key_passage: str = Field(..., description="Key data passage supporting the answer")
    domain: str = Field(..., description="The RefSeq domain category")
    question_type: str = Field(..., description="Type of question")


# ============= Reference Examples =============

def load_reference_examples() -> str:
    """Load reference examples from reference.txt for the generation prompt."""
    with open(REFERENCE_FILE, "r") as f:
        content = f.read()

    examples = []
    blocks = content.split("---")
    for block in blocks:
        block = block.strip()
        if "Question:" in block and "Answer:" in block:
            q_match = re.search(r'Question:\s*(.+?)(?=\n\nAnswer:)', block, re.DOTALL)
            a_match = re.search(r'Answer:\s*(.+?)(?=\n\nSource:)', block, re.DOTALL)
            s_match = re.search(r'Source:\s*(.+?)(?=\n)', block)
            if q_match and a_match and s_match:
                examples.append(
                    f"Question: {q_match.group(1).strip()}\n"
                    f"Answer: {a_match.group(1).strip()}\n"
                    f"Source: {s_match.group(1).strip()}"
                )

    return "\n\n".join(examples[:6])


# ============= NCBI E-utilities Helpers =============

async def eutils_request(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, str],
    ncbi_api_key: str | None = None,
) -> str:
    """Make an E-utilities request with rate limiting."""
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key

    url = f"{EUTILS_BASE}/{endpoint}"
    for attempt in range(5):
        response = await client.get(url, params=params, timeout=30)
        if response.status_code == 429:
            wait = 2 ** attempt
            print(f"    Rate limited, waiting {wait}s...")
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        await asyncio.sleep(RATE_LIMIT_DELAY)
        return response.text
    response.raise_for_status()
    return response.text


async def esearch_gene_ids(
    client: httpx.AsyncClient,
    query: str,
    retmax: int = 50,
    ncbi_api_key: str | None = None,
) -> list[str]:
    """Search for Gene IDs using esearch."""
    params = {
        "db": "gene",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
    }
    text = await eutils_request(client, "esearch.fcgi", params, ncbi_api_key)
    data = json.loads(text)
    return data.get("esearchresult", {}).get("idlist", [])


async def esummary_gene(
    client: httpx.AsyncClient,
    gene_id: str,
    ncbi_api_key: str | None = None,
) -> dict[str, Any] | None:
    """Get gene summary via esummary."""
    params = {
        "db": "gene",
        "id": gene_id,
        "retmode": "json",
    }
    try:
        text = await eutils_request(client, "esummary.fcgi", params, ncbi_api_key)
        data = json.loads(text)
        result = data.get("result", {})
        gene_data = result.get(gene_id)
        if not gene_data or "error" in gene_data:
            return None
        return gene_data
    except Exception as e:
        print(f"    esummary error for gene {gene_id}: {e}")
        return None


async def efetch_nuccore(
    client: httpx.AsyncClient,
    accession: str,
    ncbi_api_key: str | None = None,
) -> str | None:
    """Fetch GenBank record for a nucleotide accession."""
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "gb",
        "retmode": "text",
    }
    try:
        text = await eutils_request(client, "efetch.fcgi", params, ncbi_api_key)
        if len(text) < 100:
            return None
        return text
    except Exception as e:
        print(f"    efetch nuccore error for {accession}: {e}")
        return None


async def efetch_protein(
    client: httpx.AsyncClient,
    accession: str,
    ncbi_api_key: str | None = None,
) -> str | None:
    """Fetch GenPept record for a protein accession."""
    params = {
        "db": "protein",
        "id": accession,
        "rettype": "gp",
        "retmode": "text",
    }
    try:
        text = await eutils_request(client, "efetch.fcgi", params, ncbi_api_key)
        if len(text) < 100:
            return None
        return text
    except Exception as e:
        print(f"    efetch protein error for {accession}: {e}")
        return None


# ============= Fact Extraction from Gene Summary =============

def extract_gene_facts(gene_data: dict[str, Any]) -> dict[str, Any]:
    """Extract structured facts from an esummary gene record."""
    facts = {}

    facts["gene_id"] = str(gene_data.get("uid", ""))
    facts["symbol"] = gene_data.get("name", "")
    facts["full_name"] = gene_data.get("description", "")
    facts["organism"] = gene_data.get("organism", {}).get("scientificname", "")
    facts["common_name"] = gene_data.get("organism", {}).get("commonname", "")
    facts["chromosome"] = gene_data.get("chromosome", "")
    facts["map_location"] = gene_data.get("maplocation", "")
    facts["aliases"] = gene_data.get("otheraliases", "")
    facts["other_designations"] = gene_data.get("otherdesignations", "")
    facts["summary"] = gene_data.get("summary", "")
    facts["gene_type"] = gene_data.get("currentid", "")

    # Extract genomic info
    genomic_info = gene_data.get("genomicinfo", [])
    if genomic_info and len(genomic_info) > 0:
        gi = genomic_info[0]
        facts["exon_count"] = gi.get("exoncount", "")
        facts["chr_start"] = gi.get("chrstart", "")
        facts["chr_stop"] = gi.get("chrstop", "")

    # Extract RefSeq transcript accessions from locationhist
    location_hist = gene_data.get("locationhist", [])
    if location_hist:
        facts["annotation_release"] = location_hist[0].get("annotationrelease", "")

    return facts


def extract_genbank_facts(gb_text: str) -> dict[str, Any]:
    """Extract structured facts from a GenBank flat file."""
    facts = {}

    # LOCUS line: e.g. "LOCUS       NM_000546               2512 bp    mRNA    linear   PRI ..."
    locus_match = re.search(r'LOCUS\s+(\S+)\s+(\d+)\s+bp', gb_text)
    if locus_match:
        facts["accession_locus"] = locus_match.group(1)
        facts["sequence_length_bp"] = locus_match.group(2)

    # DEFINITION
    def_match = re.search(r'DEFINITION\s+(.+?)(?=\nACCESSION)', gb_text, re.DOTALL)
    if def_match:
        facts["definition"] = " ".join(def_match.group(1).split())

    # VERSION
    ver_match = re.search(r'VERSION\s+(\S+)', gb_text)
    if ver_match:
        facts["version"] = ver_match.group(1)

    # CDS range
    cds_match = re.search(r'^\s+CDS\s+((?:join\()?[\d.<>]+\.\.[\d.<>]+(?:,[\d.<>]+\.\.[\d.<>]+)*\)?)', gb_text, re.MULTILINE)
    if cds_match:
        facts["cds_range"] = cds_match.group(1)

    # Protein product from CDS
    protein_match = re.search(r'/protein_id="([^"]+)"', gb_text)
    if protein_match:
        facts["protein_id"] = protein_match.group(1)

    # Product name
    product_match = re.search(r'/product="([^"]+)"', gb_text)
    if product_match:
        facts["product"] = product_match.group(1)

    # Gene symbol
    gene_match = re.search(r'/gene="([^"]+)"', gb_text)
    if gene_match:
        facts["gene_symbol"] = gene_match.group(1)

    # Count exons
    exon_count = len(re.findall(r'^\s+exon\s+', gb_text, re.MULTILINE))
    if exon_count > 0:
        facts["exon_count_from_features"] = exon_count

    # Organism
    org_match = re.search(r'/organism="([^"]+)"', gb_text)
    if org_match:
        facts["organism"] = org_match.group(1)

    return facts


def extract_protein_facts(gp_text: str) -> dict[str, Any]:
    """Extract structured facts from a GenPept flat file."""
    facts = {}

    # LOCUS line: e.g. "LOCUS       NP_000537                393 aa"
    locus_match = re.search(r'LOCUS\s+(\S+)\s+(\d+)\s+aa', gp_text)
    if locus_match:
        facts["protein_accession_locus"] = locus_match.group(1)
        facts["protein_length_aa"] = locus_match.group(2)

    # DEFINITION
    def_match = re.search(r'DEFINITION\s+(.+?)(?=\nACCESSION)', gp_text, re.DOTALL)
    if def_match:
        facts["protein_definition"] = " ".join(def_match.group(1).split())

    # VERSION
    ver_match = re.search(r'VERSION\s+(\S+)', gp_text)
    if ver_match:
        facts["protein_version"] = ver_match.group(1)

    # Signal peptide
    sig_match = re.search(r'^\s+sig_peptide\s+([\d.<>]+\.\.[\d.<>]+)', gp_text, re.MULTILINE)
    if sig_match:
        facts["signal_peptide_range"] = sig_match.group(1)

    # mat_peptide
    mat_match = re.search(r'^\s+mat_peptide\s+([\d.<>]+\.\.[\d.<>]+)', gp_text, re.MULTILINE)
    if mat_match:
        facts["mature_peptide_range"] = mat_match.group(1)

    # Region/domain annotations
    regions = []
    for m in re.finditer(r'^\s+Region\s+([\d.<>]+\.\.[\d.<>]+)\s*\n\s+/region_name="([^"]+)"', gp_text, re.MULTILINE):
        regions.append({"range": m.group(1), "name": m.group(2)})
    if regions:
        facts["regions"] = regions

    # Coded by
    coded_match = re.search(r'/coded_by="([^"]+)"', gp_text)
    if coded_match:
        facts["coded_by"] = coded_match.group(1)

    return facts


# ============= QA Generation =============

GENERATION_PROMPT_TEMPLATE = """You are creating a RefSeq/genomics question-answering benchmark. Given structured data extracted from an NCBI Gene or RefSeq record, generate ONE high-quality question-answer pair.

Requirements:
1. The question must be NATURAL and describe what gene/transcript/protein is being asked about
2. The answer must be a SPECIFIC, VERIFIABLE FACT from the provided data
3. The answer should be concise (under 200 characters)
4. The question MUST have DISTINCTIVE SPECIFICITY -- include the gene symbol, accession number, organism, or Gene ID

CRITICAL -- the answer must be something that can be verified by visiting NCBI:
- Transcript/mRNA lengths in base pairs
- Protein lengths in amino acids
- Exon counts
- Chromosomal locations and bands
- CDS coordinate ranges
- Gene full names and aliases
- Organism information
- Number of transcript variants
- Protein domain annotations
- Signal peptide ranges

{style_instruction}

BAD patterns (DO NOT USE):
- NEVER start with "According to..."
- NEVER use "Based on the NCBI record..."
- NEVER ask general biology questions -- only ask about specific NCBI record data

Here are reference examples of the style we need:

{reference_examples}

Now generate a QA pair from this NCBI data:

Domain: {domain}
{data_description}

Respond with a JSON object with these exact fields:
- "question": a natural question with distinctive specificity (string)
- "answer": concise answer (string, under 200 chars)
- "accession": the primary accession or Gene ID referenced (e.g., "NM_000546.6" or "Gene ID 7157")
- "source_url": the NCBI URL (e.g., "https://www.ncbi.nlm.nih.gov/gene/7157" or "https://www.ncbi.nlm.nih.gov/nuccore/NM_000546.6")
- "key_passage": the key data from the record that supports the answer (string, max 500 chars)
- "question_type": one of "sequence_length", "exon_count", "chromosomal_location", "gene_name", "protein_product", "cds_range", "variant_count", "organism", "functional", "domain_annotation"

IMPORTANT:
- The answer must be directly derivable from the provided data
- Pick a fact that is SPECIFIC to this record
- If the data is insufficient, respond with {{"error": "insufficient data"}}"""

STYLE_INSTRUCTIONS = [
    'YOUR QUESTION MUST START WITH "What" and ask about a specific metadata field. Example: "What is the mRNA length in base pairs for the human TP53 RefSeq transcript NM_000546.6?" Do NOT use "According to..." or "Based on...".',
    'YOUR QUESTION MUST START WITH "How many" and ask about a count. Example: "How many exons does the human BRCA1 gene (Gene ID 672) contain according to NCBI Gene?" Do NOT use "According to..." or "Based on...".',
    'YOUR QUESTION MUST START WITH "On which" or "Where". Example: "On which chromosomal band is the human EGFR gene (Gene ID 1956) located according to NCBI Gene?" Do NOT use "According to..." or "Based on...".',
    'Your question should naturally include a RefSeq accession number. Example: "For the RefSeq transcript NM_000546.6, what is the CDS nucleotide range encoding the TP53 protein?" Do NOT use "According to..." or "Based on...".',
    'YOUR QUESTION MUST NOT include an accession number. Instead describe the gene by name and organism. Example: "What is the official full gene name for the human gene with symbol LRRK2 according to NCBI Gene?" Do NOT use "According to..." or "Based on...".',
    'YOUR QUESTION MUST START WITH "Which" or "For". Example: "Which protein accession corresponds to the product of human TP53 transcript variant 1 (NM_000546.6)?" or "For the mouse ortholog of human BRCA1, what is the Gene ID?" Do NOT use "According to..." or "Based on...".',
]

_style_counter = 0


def format_data_description(domain: str, gene_facts: dict, gb_facts: dict | None = None, protein_facts: dict | None = None) -> str:
    """Format extracted facts into a readable description for the LLM."""
    parts = []

    parts.append(f"Gene ID: {gene_facts.get('gene_id', 'N/A')}")
    parts.append(f"Gene Symbol: {gene_facts.get('symbol', 'N/A')}")
    parts.append(f"Full Name: {gene_facts.get('full_name', 'N/A')}")
    parts.append(f"Organism: {gene_facts.get('organism', 'N/A')} ({gene_facts.get('common_name', '')})")
    parts.append(f"Chromosome: {gene_facts.get('chromosome', 'N/A')}")
    parts.append(f"Map Location: {gene_facts.get('map_location', 'N/A')}")
    parts.append(f"Exon Count: {gene_facts.get('exon_count', 'N/A')}")
    if gene_facts.get("aliases"):
        parts.append(f"Aliases: {gene_facts['aliases']}")
    if gene_facts.get("summary"):
        summary = gene_facts["summary"][:500]
        parts.append(f"Summary: {summary}")

    if gb_facts:
        parts.append(f"\nTranscript Record:")
        if gb_facts.get("version"):
            parts.append(f"  Accession/Version: {gb_facts['version']}")
        if gb_facts.get("sequence_length_bp"):
            parts.append(f"  Sequence Length: {gb_facts['sequence_length_bp']} bp")
        if gb_facts.get("definition"):
            parts.append(f"  Definition: {gb_facts['definition']}")
        if gb_facts.get("cds_range"):
            parts.append(f"  CDS Range: {gb_facts['cds_range']}")
        if gb_facts.get("protein_id"):
            parts.append(f"  Protein Product: {gb_facts['protein_id']}")
        if gb_facts.get("product"):
            parts.append(f"  Product Name: {gb_facts['product']}")
        if gb_facts.get("exon_count_from_features"):
            parts.append(f"  Exon Count (from features): {gb_facts['exon_count_from_features']}")

    if protein_facts:
        parts.append(f"\nProtein Record:")
        if protein_facts.get("protein_version"):
            parts.append(f"  Accession/Version: {protein_facts['protein_version']}")
        if protein_facts.get("protein_length_aa"):
            parts.append(f"  Protein Length: {protein_facts['protein_length_aa']} aa")
        if protein_facts.get("protein_definition"):
            parts.append(f"  Definition: {protein_facts['protein_definition']}")
        if protein_facts.get("signal_peptide_range"):
            parts.append(f"  Signal Peptide Range: {protein_facts['signal_peptide_range']}")
        if protein_facts.get("mature_peptide_range"):
            parts.append(f"  Mature Peptide Range: {protein_facts['mature_peptide_range']}")
        if protein_facts.get("coded_by"):
            parts.append(f"  Coded By: {protein_facts['coded_by']}")
        if protein_facts.get("regions"):
            region_strs = [f"    {r['name']} ({r['range']})" for r in protein_facts["regions"][:10]]
            parts.append(f"  Protein Domains/Regions:\n" + "\n".join(region_strs))

    return "\n".join(parts)


async def generate_qa_from_facts(
    oai_client: AsyncOpenAI,
    domain: str,
    gene_facts: dict,
    gb_facts: dict | None,
    protein_facts: dict | None,
    reference_examples: str,
) -> QAPair | None:
    """Use LLM to generate a QA pair from extracted facts."""
    global _style_counter
    style_instruction = STYLE_INSTRUCTIONS[_style_counter % len(STYLE_INSTRUCTIONS)]
    _style_counter += 1

    data_description = format_data_description(domain, gene_facts, gb_facts, protein_facts)

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        style_instruction=style_instruction,
        reference_examples=reference_examples,
        domain=domain,
        data_description=data_description,
    )

    try:
        response = await oai_client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or ""
        result = json.loads(result_text)

        if "error" in result:
            return None

        # Validate fields
        required_fields = ["question", "answer", "accession", "source_url", "key_passage", "question_type"]
        if not all(k in result for k in required_fields):
            return None

        # Validate answer length
        if len(result["answer"]) > 200:
            return None

        # Validate question length (distinctive specificity)
        if len(result["question"]) < 50:
            return None

        # Validate key_passage non-empty
        if not result["key_passage"] or len(result["key_passage"]) < 10:
            return None

        # Validate question_type
        valid_types = ["sequence_length", "exon_count", "chromosomal_location", "gene_name",
                       "protein_product", "cds_range", "variant_count", "organism",
                       "functional", "domain_annotation"]
        question_type = result["question_type"]
        if question_type not in valid_types:
            question_type = "functional"

        return QAPair(
            question=result["question"],
            answer=result["answer"],
            accession=result["accession"],
            source_url=result["source_url"],
            key_passage=result["key_passage"][:500],
            domain=domain,
            question_type=question_type,
        )
    except Exception as e:
        print(f"    QA generation error: {e}")
        return None


# ============= Domain Processing =============

async def fetch_gene_data(
    http_client: httpx.AsyncClient,
    gene_id: str,
    domain: str,
    ncbi_api_key: str | None,
) -> tuple[dict, dict | None, dict | None]:
    """Fetch all relevant data for a gene based on domain needs."""
    # Always get gene summary
    gene_data = await esummary_gene(http_client, gene_id, ncbi_api_key)
    if not gene_data:
        return {}, None, None

    gene_facts = extract_gene_facts(gene_data)
    gb_facts = None
    protein_facts = None

    # For domains needing transcript data, fetch the first RefSeq transcript
    needs_transcript = domain in [
        "transcript_metadata", "coding_sequence", "exon_structure",
        "gene_transcript_relationships",
    ]
    # For domains needing protein data
    needs_protein = domain in [
        "protein_metadata", "protein_features",
    ]

    if needs_transcript or needs_protein:
        # Get a RefSeq mRNA accession from the gene
        # Try via elink from gene to nuccore
        try:
            params = {
                "db": "gene",
                "id": gene_id,
                "rettype": "gene_table",
                "retmode": "text",
            }
            if ncbi_api_key:
                params["api_key"] = ncbi_api_key

            # Use esearch in nuccore for gene symbol to find NM_ accessions
            symbol = gene_facts.get("symbol", "")
            organism = gene_facts.get("organism", "")
            if symbol and organism:
                nm_ids = await esearch_gene_ids(
                    http_client,
                    f"{symbol}[Gene Name] AND {organism}[Orgn] AND NM_[Accession]",
                    retmax=5,
                    ncbi_api_key=ncbi_api_key,
                )
                # esearch in nuccore db
                search_params = {
                    "db": "nuccore",
                    "term": f"{symbol}[Gene Name] AND {organism}[Orgn] AND refseq[Filter] AND biomol_mrna[Properties]",
                    "retmax": "5",
                    "retmode": "json",
                }
                if ncbi_api_key:
                    search_params["api_key"] = ncbi_api_key

                text = await eutils_request(http_client, "esearch.fcgi", search_params, ncbi_api_key)
                nuc_data = json.loads(text)
                nuc_ids = nuc_data.get("esearchresult", {}).get("idlist", [])

                if nuc_ids:
                    # Fetch the first transcript
                    gb_text = await efetch_nuccore(http_client, nuc_ids[0], ncbi_api_key)
                    if gb_text:
                        gb_facts = extract_genbank_facts(gb_text)

                        # If we need protein and found protein_id in transcript
                        if needs_protein and gb_facts.get("protein_id"):
                            gp_text = await efetch_protein(http_client, gb_facts["protein_id"], ncbi_api_key)
                            if gp_text:
                                protein_facts = extract_protein_facts(gp_text)
        except Exception as e:
            print(f"    Transcript/protein fetch error for gene {gene_id}: {e}")

    elif needs_protein:
        # Fetch protein directly
        symbol = gene_facts.get("symbol", "")
        organism = gene_facts.get("organism", "")
        if symbol and organism:
            try:
                search_params = {
                    "db": "protein",
                    "term": f"{symbol}[Gene Name] AND {organism}[Orgn] AND refseq[Filter]",
                    "retmax": "5",
                    "retmode": "json",
                }
                text = await eutils_request(http_client, "esearch.fcgi", search_params, ncbi_api_key)
                prot_data = json.loads(text)
                prot_ids = prot_data.get("esearchresult", {}).get("idlist", [])

                if prot_ids:
                    gp_text = await efetch_protein(http_client, prot_ids[0], ncbi_api_key)
                    if gp_text:
                        protein_facts = extract_protein_facts(gp_text)
            except Exception as e:
                print(f"    Protein fetch error for gene {gene_id}: {e}")

    return gene_facts, gb_facts, protein_facts


async def process_domain(
    oai_client: AsyncOpenAI,
    http_client: httpx.AsyncClient,
    domain: str,
    target_count: int,
    reference_examples: str,
    existing_questions: set[str],
    existing_accessions: set[str],
    ncbi_api_key: str | None,
    progress_path: Path = PROGRESS_JSONL,
) -> list[QAPair]:
    """Process a single domain: discover genes, fetch data, generate QA pairs."""
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"Target: {target_count} QA pairs")
    print(f"{'='*60}")

    queries = DOMAIN_GENE_QUERIES.get(domain, [])
    all_gene_ids: list[str] = []
    seen_gene_ids: set[str] = set()

    # Step 1: Discover gene IDs
    print(f"  Discovering genes...")
    for db, query in queries:
        if len(all_gene_ids) >= target_count * 5:
            break
        try:
            gene_ids = await esearch_gene_ids(http_client, query, retmax=50, ncbi_api_key=ncbi_api_key)
            for gid in gene_ids:
                if gid not in seen_gene_ids:
                    seen_gene_ids.add(gid)
                    all_gene_ids.append(gid)
        except Exception as e:
            print(f"    Search error for query: {e}")
            continue

    print(f"  Found {len(all_gene_ids)} candidate genes")

    qa_pairs = []
    processed = 0

    for gene_id in all_gene_ids:
        if len(qa_pairs) >= target_count:
            break

        processed += 1
        print(f"  [{processed}] Processing gene {gene_id}...")

        # Step 2: Fetch structured data
        gene_facts, gb_facts, protein_facts = await fetch_gene_data(
            http_client, gene_id, domain, ncbi_api_key
        )
        if not gene_facts or not gene_facts.get("symbol"):
            print(f"    -> No gene data, skipping")
            continue

        symbol = gene_facts.get("symbol", "")
        print(f"    -> Gene: {symbol} ({gene_facts.get('organism', '')})")

        # Step 3: Generate QA
        qa = await generate_qa_from_facts(
            oai_client, domain, gene_facts, gb_facts, protein_facts, reference_examples
        )
        if not qa:
            print(f"    -> Failed to generate QA, skipping")
            continue

        # Step 4: Dedup check
        if qa.question in existing_questions:
            print(f"    -> Duplicate question, skipping")
            continue

        # Also check accession hasn't been overused
        accession_key = f"{domain}:{qa.accession}"
        if accession_key in existing_accessions:
            print(f"    -> Duplicate accession for domain, skipping")
            continue

        existing_questions.add(qa.question)
        existing_accessions.add(accession_key)
        qa_pairs.append(qa)
        save_one_qa(qa, progress_path)
        print(f"    -> Generated QA #{len(qa_pairs)}: {qa.question[:60]}...")
        print(f"       Answer: {qa.answer}")

        await asyncio.sleep(0.5)  # Rate limiting between genes

    print(f"  Domain complete: {len(qa_pairs)}/{target_count} QA pairs generated")
    return qa_pairs


# ============= Progress Management =============

def save_one_qa(qa: QAPair, path: Path) -> None:
    """Append a single QA pair to the progress JSONL file."""
    record = {
        "question": qa.question,
        "answer": qa.answer,
        "accession": qa.accession,
        "source_url": qa.source_url,
        "key_passage": qa.key_passage,
        "domain": qa.domain,
        "question_type": qa.question_type,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_progress(path: Path) -> list[QAPair]:
    """Load previous progress from JSONL."""
    if not path.exists():
        return []
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pairs.append(QAPair(
                question=data["question"],
                answer=data["answer"],
                accession=data["accession"],
                source_url=data["source_url"],
                key_passage=data["key_passage"],
                domain=data["domain"],
                question_type=data["question_type"],
            ))
    return pairs


# ============= Main =============

async def main():
    # Validate API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    ncbi_api_key = os.environ.get("NCBI_API_KEY")  # Optional but recommended
    if ncbi_api_key:
        print(f"Using NCBI API key for elevated rate limits")
        global RATE_LIMIT_DELAY
        RATE_LIMIT_DELAY = 0.11  # 10 req/sec with API key
    else:
        print("No NCBI_API_KEY set -- using default rate limit (3 req/sec)")

    oai_client = AsyncOpenAI(api_key=openai_api_key)

    # Load reference examples
    reference_examples = load_reference_examples()

    # Load any previous progress
    all_qa_pairs = load_progress(PROGRESS_JSONL)
    existing_questions = {qa.question for qa in all_qa_pairs}
    existing_accessions = {f"{qa.domain}:{qa.accession}" for qa in all_qa_pairs}

    if all_qa_pairs:
        print(f"Loaded {len(all_qa_pairs)} existing QA pairs from progress file")
        domain_counts: dict[str, int] = {}
        for qa in all_qa_pairs:
            domain_counts[qa.domain] = domain_counts.get(qa.domain, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")

    # Process domains sequentially to avoid rate limiting
    async with httpx.AsyncClient() as http_client:
        for domain in DOMAINS:
            existing_count = sum(1 for qa in all_qa_pairs if qa.domain == domain)
            remaining = TARGET_PER_DOMAIN - existing_count

            if remaining <= 0:
                print(f"\nSkipping {domain} (already have {existing_count}/{TARGET_PER_DOMAIN})")
                continue

            try:
                domain_pairs = await process_domain(
                    oai_client, http_client, domain, remaining,
                    reference_examples, existing_questions, existing_accessions,
                    ncbi_api_key
                )
                all_qa_pairs.extend(domain_pairs)
            except Exception as e:
                print(f"  Domain processing error for {domain}: {e}")

    # Final output with IDs
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total QA pairs: {len(all_qa_pairs)}")

    # Group by domain for ID assignment
    domain_counters: dict[str, int] = {}
    output_records = []

    for qa in all_qa_pairs:
        domain_counters[qa.domain] = domain_counters.get(qa.domain, 0) + 1
        idx = domain_counters[qa.domain]
        record = {
            "id": f"refseqtrain_train_{qa.domain}_{idx:03d}",
            "question": qa.question,
            "answer": qa.answer,
            "accession": qa.accession,
            "source_url": qa.source_url,
            "key_passage": qa.key_passage,
            "domain": qa.domain,
            "question_type": qa.question_type,
        }
        output_records.append(record)

    # Write final output
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved to {OUTPUT_JSONL}")

    # Print domain distribution
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counters.items()):
        print(f"  {domain}: {count}")

    # Print question type distribution
    type_counts: dict[str, int] = {}
    for qa in all_qa_pairs:
        type_counts[qa.question_type] = type_counts.get(qa.question_type, 0) + 1
    print(f"\nQuestion type distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    # Print answer length stats
    answer_lengths = [len(qa.answer) for qa in all_qa_pairs]
    if answer_lengths:
        print(f"\nAnswer length stats:")
        print(f"  Min: {min(answer_lengths)}")
        print(f"  Max: {max(answer_lengths)}")
        print(f"  Mean: {sum(answer_lengths)/len(answer_lengths):.1f}")

    # Print sample
    print(f"\nSample QA pairs:")
    for qa in all_qa_pairs[:3]:
        print(f"\n  Q: {qa.question}")
        print(f"  A: {qa.answer}")
        print(f"  Accession: {qa.accession}")
        print(f"  Source: {qa.source_url}")
        print(f"  Domain: {qa.domain}")
        print(f"  Type: {qa.question_type}")


if __name__ == "__main__":
    asyncio.run(main())
