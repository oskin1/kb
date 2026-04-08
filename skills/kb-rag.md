# KB-RAG — Agent Operating Guide

You are an AI research agent with access to a local knowledge base powered by KB-RAG. This document tells you how to use the infrastructure to ingest, search, analyse, and build knowledge.

All scripts live in `scripts/` and require `--kb-root <PATH>` pointing to the KB directory. Run them with `python3 scripts/<script>.py`.

---

## Core Concepts

- **KB root** — a directory holding your data, configs, notes, and exports. Every command needs `--kb-root <path>`.
- **Collections** — Qdrant stores data in five collections:
  - `research_docs` — chunked source documents (papers, web pages, reports)
  - `data_tables` — structured/numerical data
  - `insights` — agent-generated conclusions, community summaries
  - `entities` — named entities extracted from docs (with `community_id` after clustering)
  - `facts` — typed relationships between entities (subject --[RELATION]--> object)
- **Domain** — a configurable namespace (e.g. "clinical", "genomics") with its own keywords and relation ontology. Defined in `config/domain.yaml`. The `general` domain is the catch-all.

---

## Workflow: From Document to Searchable Knowledge

Follow this pipeline in order. Each step builds on the previous one.

### 1. Ingest documents

```bash
# Single file
python3 scripts/ingest.py raw/papers/paper.pdf --kb-root ~/my-kb

# Entire directory
python3 scripts/ingest.py raw/papers/ --kb-root ~/my-kb --meta domain=clinical project=cancer

# With temporal validity
python3 scripts/ingest.py report.pdf --kb-root ~/my-kb --valid-at 2025-Q1

# Fetch and ingest a web page
python3 scripts/fetch_url.py https://example.com/article --kb-root ~/my-kb --ingest

# Search and ingest arXiv papers
python3 scripts/arxiv_search.py "transformer architecture" --kb-root ~/my-kb --ingest
```

Ingest parses (PDF, HTML, TXT, MD, CSV), chunks (512 chars, 64 overlap), embeds with bge-m3, and stores in Qdrant.

### 2. Extract entities

```bash
# All unprocessed documents
python3 scripts/entity_extract.py --all --kb-root ~/my-kb

# Specific document
python3 scripts/entity_extract.py --doc-id <DOC_ID> --kb-root ~/my-kb --verbose

# List stored entities
python3 scripts/entity_extract.py --list-entities --kb-root ~/my-kb
```

Uses LLM to extract named entities (people, concepts, materials, etc.) with deduplication via embedding similarity + LLM resolution.

### 3. Extract facts

```bash
# All docs with entities but no facts yet
python3 scripts/fact_extract.py --all --kb-root ~/my-kb

# Specific document
python3 scripts/fact_extract.py --doc-id <DOC_ID> --kb-root ~/my-kb --verbose

# List stored facts
python3 scripts/fact_extract.py --list-facts --kb-root ~/my-kb
python3 scripts/fact_extract.py --list-facts --kb-root ~/my-kb --subject "Entity Name"
```

Extracts typed relationships between entities using the domain's relation ontology.

### 4. Detect contradictions

```bash
# Audit all facts
python3 scripts/invalidate.py --audit --kb-root ~/my-kb

# Check facts from a specific document
python3 scripts/invalidate.py --doc-id <DOC_ID> --kb-root ~/my-kb --dry-run
```

Finds contradicting fact pairs and marks the older one as superseded.

### 5. Cluster the knowledge graph

```bash
python3 scripts/cluster.py --kb-root ~/my-kb --verbose
```

Builds a graph from entities + facts, runs Leiden community detection, writes `community_id` to each entity, and generates per-community summary insights. Re-run after adding new documents.

### 6. Visualize

```bash
python3 scripts/visualize.py --kb-root ~/my-kb
# Output: ~/my-kb/exports/graph.html
```

Generates an interactive HTML knowledge graph (vis.js). Open in a browser to explore communities, search entities, and inspect connections.

---

## Searching the KB

### Smart router (recommended for most queries)

```bash
python3 scripts/route_query.py "your question" --kb-root ~/my-kb
```

This is your primary search tool. It:
1. Auto-detects the domain from query keywords
2. Searches `research_docs`, `insights`, and `facts` in parallel
3. Expands via community peers (entities in the same cluster)
4. Merges and ranks all results by score

Key flags:
- `--domain <name>` — force a specific domain instead of auto-detect
- `--top N` — results per collection (default 5)
- `--merged-top N` — total results shown (default 10)
- `--facts-only` — search only the facts collection
- `--relation <TYPE>` — filter facts by relation (e.g. CONTAINS, TREATS)
- `--no-expand` — disable community-aware expansion
- `--collections docs,insights,facts` — choose which collections to search

### Hybrid document search

```bash
python3 scripts/query.py "search terms" --kb-root ~/my-kb
```

Searches a single collection with cosine + BM25 + cross-encoder re-ranking.

Key flags:
- `--collection {research_docs|data_tables|insights}` — target collection
- `--bm25` — keyword-only search (no vector similarity)
- `--no-hybrid` — cosine-only search (no BM25)
- `--no-rerank` — skip cross-encoder re-ranking (faster)
- `--as-of <DATE>` — temporal filter (e.g. `2025-Q1`, `2024-06-01`)
- `--domain`, `--tags`, `--project`, `--lang` — metadata filters

### Knowledge graph search

```bash
# Semantic search over facts
python3 scripts/query_facts.py "topic" --kb-root ~/my-kb

# Filter by entity or relation
python3 scripts/query_facts.py --subject "Entity Name" --kb-root ~/my-kb
python3 scripts/query_facts.py --relation CONTAINS --domain clinical --kb-root ~/my-kb
```

---

## Storing Agent Output

When you reach conclusions, discover gaps, or want to preserve analysis, store it in the KB so future queries can find it.

### Quick insight

```bash
python3 scripts/add_insight.py "Your conclusion or finding" --kb-root ~/my-kb \
  --type conclusion --domain clinical --confidence established \
  --tags "tag1,tag2" --project myproject
```

Types: `summary`, `conclusion`, `hypothesis`, `gap`, `connection`.
Confidence: `established`, `probable`, `speculative`, `needs-verification`.

### Research note (longer form)

```bash
python3 scripts/write_note.py notes/analysis.md --kb-root ~/my-kb \
  --title "My Analysis" --domain clinical --project cancer
```

Writes markdown to the notes directory and ingests it into the insights collection. Use `--stdin` to pipe content directly.

### Project tracking

```bash
# Create a project
python3 scripts/write_project.py --name my-research \
  --summary "Investigating X and Y" --kb-root ~/my-kb

# Append a log entry
python3 scripts/write_project.py --name my-research \
  --log "Found key relationship between A and B" --kb-root ~/my-kb

# List projects
python3 scripts/write_project.py --list --kb-root ~/my-kb
```

### Export insights to markdown

```bash
python3 scripts/export_insights.py --kb-root ~/my-kb --project myproject --output report.md
```

---

## Decision Guide: Which Tool When

| I want to... | Use |
|---|---|
| Answer a question using the KB | `route_query.py "question"` |
| Find a specific document chunk | `query.py "terms" --collection research_docs` |
| Explore entity relationships | `query_facts.py --subject "Name"` or `visualize.py` |
| Understand the overall knowledge structure | `cluster.py --verbose` then `visualize.py` |
| Add a PDF/web page to the KB | `ingest.py <file>` or `fetch_url.py <url> --ingest` |
| Process new docs through the full pipeline | `ingest.py` then `entity_extract.py --all` then `fact_extract.py --all` then `cluster.py` |
| Record a finding | `add_insight.py "text"` |
| Write up analysis | `write_note.py file.md` |
| Check for stale/contradicting facts | `invalidate.py --audit` |
| See what's in the KB | `query.py "" --list-docs` and `entity_extract.py --list-entities` |

---

## Typical Agent Session

A research session usually follows this pattern:

```
1. ORIENT     — route_query.py to see what the KB already knows
2. INGEST     — fetch_url.py / ingest.py to add new sources
3. EXTRACT    — entity_extract.py --all, fact_extract.py --all
4. VALIDATE   — invalidate.py --audit (catch contradictions)
5. CLUSTER    — cluster.py (update communities after new data)
6. ANALYSE    — route_query.py / query_facts.py (deeper dives)
7. RECORD     — add_insight.py / write_note.py (persist findings)
8. VISUALIZE  — visualize.py (inspect graph structure if needed)
```

You don't need every step every time. If you're just querying, step 1 is enough. If you just ingested new documents, run steps 3-5 to integrate them into the knowledge graph.

---

## Configuration Reference

All config lives in `<kb-root>/config/`.

**config.yaml** controls infrastructure:
- `qdrant.host/port` — Qdrant connection
- `embedding.model` — embedding model (default: bge-m3, 1024d)
- `llm.provider/model` — LLM for extraction (openai or anthropic)
- `chunking.chunk_size/chunk_overlap` — document chunking params
- `reranker.enabled/model` — cross-encoder re-ranking
- `clustering.resolution` — Leiden resolution (higher = more smaller communities)
- `clustering.generate_summaries` — whether to LLM-summarise each community
- `visualization.max_nodes` — node cap for HTML export
- `visualization.output_dir` — output directory for exports

**domain.yaml** controls domain intelligence:
- `domains.<name>.keywords` — trigger words for auto-routing
- `domains.<name>.relations` — allowed relation types for fact extraction
- `entity_types` — what kinds of entities to extract
- `prompts` — LLM system/user prompts for extraction and resolution

---

## Important Notes

- All data stays local. Embeddings run via Ollama. Only entity/fact extraction calls an external LLM API.
- Entity extraction and fact extraction require an API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `<kb-root>/.env`).
- Clustering and visualization have no LLM dependency (unless `generate_summaries` is enabled).
- The `--verbose` flag on extraction scripts shows per-chunk progress — use it when debugging or monitoring long runs.
- `cluster.py` is idempotent. Re-running it overwrites community assignments and replaces old community summaries.
- Community expansion in `route_query.py` is automatic when entities have `community_id`. If clustering hasn't been run, expansion is silently skipped.
