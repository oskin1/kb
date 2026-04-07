# KB-RAG — Domain-Configurable Knowledge Base with RAG + Knowledge Graph

A local RAG pipeline that combines **vector semantic search**, **BM25 keyword search**, and a **knowledge graph** (entities + typed facts) with temporal validity tracking. Configure it for any domain — no code changes needed.

## Features

- **Hybrid search** — cosine + BM25 via Reciprocal Rank Fusion, cross-encoder re-ranking
- **Knowledge graph** — LLM-extracted entities and typed relationships between them
- **Temporal validity** — track when facts become true/obsolete, query "as of" a date
- **Contradiction detection** — automatically invalidates stale facts when newer ones arrive
- **Domain-configurable** — define your own entity types, relationship ontology, and routing keywords in YAML
- **Multilingual** — bge-m3 embeddings support 100+ languages natively
- **Local-first** — Qdrant (Docker) + Ollama embeddings, no data leaves your machine
- **Multiple ingest sources** — files, URLs, arXiv papers, Telegram

## Quick Start

### 1. Prerequisites

```bash
# Start Qdrant (vector database)
docker compose up -d

# Ollama (for embeddings)
brew install ollama   # or see https://ollama.com
ollama pull bge-m3

# Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `docker compose up -d` starts Qdrant on `localhost:6333` with persistent storage. To stop: `docker compose down`. Data is preserved in the `qdrant_storage` Docker volume.

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your OpenAI (or Anthropic) API key
# Only needed for entity/fact extraction. Basic ingest+search works without it.
```

Edit `config/domain.yaml` to define your domain:
- **domains** — keyword lists for auto-routing + relation types for fact extraction
- **entity_types** — what kinds of entities to extract
- **prompts** — LLM instructions tuned to your field

Edit `config/config.yaml` for infrastructure settings (Qdrant host, embedding model, chunking params, LLM provider).

### 3. Initialize

```bash
python scripts/init_collections.py
```

### 4. Ingest documents

```bash
python scripts/ingest.py raw/papers/my_paper.pdf
python scripts/ingest.py raw/papers/ --meta domain=myfield project=myproject
python scripts/ingest.py report.pdf --valid-at 2025-Q1
```

### 5. Search

```bash
# Hybrid search (cosine + BM25 + re-ranking)
python scripts/query.py "your search query"

# Smart router — searches docs + insights + facts, auto-detects domain
python scripts/route_query.py "your query"

# Knowledge graph facts
python scripts/query_facts.py "your query" --relation CONTAINS

# Keyword-only or cosine-only
python scripts/query.py "exact term" --bm25
python scripts/query.py "broad concept" --no-hybrid
```

## Project Structure

```
kb-rag/
├── config/
│   ├── config.yaml          # Infrastructure: Qdrant, embedding, LLM, chunking
│   └── domain.yaml          # YOUR DOMAIN: keywords, relations, entity types, prompts
├── scripts/
│   ├── init_collections.py  # One-time Qdrant setup
│   ├── ingest.py            # Document ingestion (PDF, HTML, TXT, MD, CSV)
│   ├── query.py             # Hybrid search with re-ranking
│   ├── route_query.py       # Smart multi-collection router
│   ├── query_facts.py       # Knowledge graph search
│   ├── entity_extract.py    # LLM entity extraction (auto after ingest)
│   ├── fact_extract.py      # LLM relationship extraction (auto after entities)
│   ├── invalidate.py        # Contradiction detection & fact invalidation
│   ├── domain_config.py     # Shared domain config loader (internal)
│   ├── add_insight.py       # Store atomic insights
│   ├── write_note.py        # Structured research notes
│   ├── write_proposal.py    # Actionable recommendations
│   ├── write_project.py     # Named research threads
│   ├── export_insights.py   # Export insights to markdown
│   ├── fetch_url.py         # Download web pages for ingestion
│   ├── tg_ingest.py         # Telegram integration handler
│   └── arxiv_search.py      # arXiv paper search + ingest
├── raw/                     # Your source documents (gitignored)
│   ├── papers/
│   ├── reports/
│   ├── data/
│   └── web/
├── notes/                   # Research notes output
├── proposals/               # Proposal documents output
├── projects/                # Project tracking files
├── .env.example             # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

## Domain Configuration Example

Here's how you'd configure this for a **medical research** KB:

```yaml
# config/domain.yaml
domains:
  clinical:
    keywords: [patient, diagnosis, treatment, symptom, drug, dosage, trial, therapy]
    relations:
      - TREATS
      - CAUSES
      - CONTRAINDICATES
      - DIAGNOSED_BY
      - ADMINISTERED_VIA
      - SIDE_EFFECT_OF

  genomics:
    keywords: [gene, protein, mutation, expression, pathway, allele, chromosome]
    relations:
      - ENCODES
      - REGULATES
      - MUTATED_IN
      - EXPRESSED_IN
      - INTERACTS_WITH

  general:
    keywords: []
    relations:
      - RELATED_TO
      - PART_OF
      - USED_IN

entity_types:
  - drug
  - disease
  - gene
  - protein
  - symptom
  - organ
  - procedure
  - organization
  - person
  - concept
```

## Pipeline Architecture

```
Document → ingest.py → chunk + embed → Qdrant (research_docs)
                ↓
        entity_extract.py → LLM extracts entities → Qdrant (entities)
                ↓
        fact_extract.py → LLM extracts relationships → Qdrant (facts)
                ↓
        invalidate.py → detect contradictions → mark stale facts

Query → route_query.py → auto-detect domain
            ↓
        parallel search: research_docs + insights + facts
            ↓
        hybrid (cosine + BM25) → RRF merge → cross-encoder re-rank
            ↓
        merged ranked results with source labels
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `init_collections.py` | Create Qdrant collections (run once) |
| `ingest.py <file>` | Ingest document(s) + auto entity/fact extraction |
| `query.py "text"` | Hybrid search with re-ranking |
| `route_query.py "text"` | Smart multi-collection search |
| `query_facts.py "text"` | Search knowledge graph |
| `entity_extract.py --all` | Extract entities from all unprocessed docs |
| `fact_extract.py --all` | Extract facts from all docs with entities |
| `invalidate.py --audit` | Scan all facts for contradictions |
| `add_insight.py "text"` | Store a single insight |
| `write_note.py file.md` | Save + index a research note |
| `fetch_url.py <url>` | Download web page for ingestion |
| `arxiv_search.py "query"` | Search arXiv, optionally ingest papers |

## Temporal Features

```bash
# Ingest with temporal validity
python scripts/ingest.py prices_2025.csv --valid-at 2025-Q1

# Supersede old data
python scripts/ingest.py prices_2026.csv --valid-at 2026-Q1 --supersedes <old_doc_id>

# Query as of a specific date
python scripts/query.py "market prices" --as-of 2025-06-01
```

## License

MIT
