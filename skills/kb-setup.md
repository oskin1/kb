# KB-RAG — Setup Guide

Step-by-step instructions for setting up the KB-RAG infrastructure from scratch. Follow these in order. Each step includes a verification check — do not proceed until the check passes.

The repo lives at the path stored in `$KB_REPO`. The user's knowledge base will be created at a path they choose (`$KB_ROOT`). These are separate directories: the repo holds scripts and default configs, the KB holds data.

---

## Prerequisites

Before starting, confirm these are available on the system:
- Docker (for Qdrant vector database)
- Python 3.11+ (for scripts)
- Ollama (for local embeddings)
- An OpenAI or Anthropic API key (only needed for entity/fact extraction — basic ingest and search work without one)

---

## Step 1: Start Qdrant

Qdrant is the vector database that stores all KB data. It runs as a Docker container.

```bash
cd $KB_REPO
docker compose up -d
```

**Verify:**
```bash
curl -s http://localhost:6333/healthz
```
Expected: `ok` or a JSON health response. If the port is different, check `docker-compose.yaml`.

**Troubleshooting:**
- If Docker is not running: `open -a Docker` (macOS) or `sudo systemctl start docker` (Linux)
- If port 6333 is taken: edit `docker-compose.yaml` to change the host port, then update `config.yaml` to match
- To stop Qdrant later: `docker compose down` (data persists in the `qdrant_storage` volume)
- To wipe all data: `docker compose down -v`

---

## Step 2: Install Ollama and pull the embedding model

Ollama runs embedding models locally. KB-RAG uses `bge-m3` (multilingual, 1024-dimensional).

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server (if not already running):
```bash
ollama serve &
```

Pull the embedding model:
```bash
ollama pull bge-m3
```

**Verify:**
```bash
ollama list | grep bge-m3
```
Expected: a line showing `bge-m3` with its size.

**Troubleshooting:**
- If `ollama serve` fails with "address already in use": Ollama is already running, which is fine
- If `ollama pull` hangs: check network connectivity; the model is ~1.2 GB

---

## Step 3: Create Python virtual environment and install dependencies

```bash
cd $KB_REPO
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Verify:**
```bash
python3 -c "import qdrant_client, ollama, networkx, graspologic; print('All imports OK')"
```
Expected: `All imports OK`

**Troubleshooting:**
- If `graspologic` fails to install: ensure you have a C compiler (`xcode-select --install` on macOS)
- If `PyMuPDF` (fitz) fails: try `pip install PyMuPDF --no-build-isolation`
- Always activate the venv before running any script: `source $KB_REPO/.venv/bin/activate`

---

## Step 4: Scaffold the knowledge base directory

Choose where your KB will live. This is separate from the repo.

```bash
python3 scripts/init.py --kb-root $KB_ROOT
```

This creates:
```
$KB_ROOT/
├── config/
│   ├── config.yaml      # Infrastructure settings
│   └── domain.yaml      # Domain-specific entity types, relations, prompts
├── raw/
│   ├── papers/
│   ├── reports/
│   ├── data/
│   └── web/
├── notes/
├── proposals/
└── projects/
```

**Verify:**
```bash
ls $KB_ROOT/config/config.yaml $KB_ROOT/config/domain.yaml
```
Expected: both files listed without errors.

---

## Step 5: Configure the LLM API key

This is only needed for entity extraction, fact extraction, contradiction detection, and community summary generation. Basic ingest and search work without it.

```bash
cp $KB_REPO/.env.example $KB_ROOT/.env
```

Edit `$KB_ROOT/.env` and set your API key:

```
# For OpenAI (default)
OPENAI_API_KEY=sk-...

# For Anthropic (uncomment and set if using Anthropic)
# ANTHROPIC_API_KEY=sk-ant-...
```

If using Anthropic instead of OpenAI, also edit `$KB_ROOT/config/config.yaml`:
```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
```

**Verify:**
```bash
grep -v "^#" $KB_ROOT/.env | grep -q "API_KEY=" && echo "Key configured" || echo "No key set"
```
Expected: `Key configured`

---

## Step 6: Customize domain configuration (optional)

Edit `$KB_ROOT/config/domain.yaml` to match your knowledge domain. The defaults work for general-purpose use, but domain-specific configuration improves entity/fact extraction quality.

Key sections to customize:
- `domains.<name>.keywords` — words that trigger auto-routing to this domain
- `domains.<name>.relations` — allowed relationship types for fact extraction
- `entity_types` — what kinds of entities to extract (concept, person, material, etc.)
- `prompts` — LLM instructions for extraction (tune these for your field)

Skip this step if you're just testing. You can always edit it later and re-extract.

---

## Step 7: Initialize Qdrant collections

This creates the vector collections with the correct schema. Safe to re-run (skips existing collections).

```bash
python3 scripts/init_collections.py --kb-root $KB_ROOT
```

**Verify:** The output should show `[created]` for each collection:
```
  [created] research_docs (1024d, cosine)
  [created] data_tables (1024d, cosine)
  [created] insights (1024d, cosine)
  [created] entities (1024d, cosine)
```

**Troubleshooting:**
- If it fails to connect: check that Qdrant is running (`docker compose ps`)
- If collections show `[skip]`: they already exist, which is fine

---

## Step 8: Test with a document

Place a document in the raw directory and run the full pipeline:

```bash
# Ingest
python3 scripts/ingest.py $KB_ROOT/raw/papers/test_document.pdf --kb-root $KB_ROOT

# Extract entities
python3 scripts/entity_extract.py --all --kb-root $KB_ROOT --verbose

# Extract facts
python3 scripts/fact_extract.py --all --kb-root $KB_ROOT --verbose

# Cluster
python3 scripts/cluster.py --kb-root $KB_ROOT --verbose

# Search
python3 scripts/route_query.py "test query related to your document" --kb-root $KB_ROOT
```

**Verify:**
- Ingest should report chunks stored
- Entity extraction should list entities found
- Fact extraction should list relationships found
- Clustering should show communities
- Search should return results from your document

---

## Step 9: Generate visualization (optional)

```bash
python3 scripts/visualize.py --kb-root $KB_ROOT
```

Open `$KB_ROOT/exports/graph.html` in a browser to see the interactive knowledge graph.

---

## Quick Health Check

Run this sequence anytime to verify the full stack is operational:

```bash
# 1. Qdrant is up
curl -sf http://localhost:6333/healthz > /dev/null && echo "Qdrant: OK" || echo "Qdrant: DOWN"

# 2. Ollama has the model
ollama list 2>/dev/null | grep -q bge-m3 && echo "Ollama bge-m3: OK" || echo "Ollama bge-m3: MISSING"

# 3. Python deps are installed
python3 -c "import qdrant_client, ollama, networkx" 2>/dev/null && echo "Python deps: OK" || echo "Python deps: MISSING"

# 4. KB root exists and has config
[ -f "$KB_ROOT/config/config.yaml" ] && echo "KB config: OK" || echo "KB config: MISSING"

# 5. Collections exist
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient('localhost', port=6333)
names = [col.name for col in c.get_collections().collections]
expected = {'research_docs', 'data_tables', 'insights', 'entities'}
missing = expected - set(names)
print('Collections: OK' if not missing else f'Collections: MISSING {missing}')
" 2>/dev/null || echo "Collections: CANNOT CHECK"
```

All five should report OK. If any fail, revisit the corresponding step above.

---

## Infrastructure Summary

| Component | What | Where | Managed by |
|-----------|------|-------|------------|
| Qdrant | Vector database | Docker container, port 6333 | `docker compose up/down` |
| Ollama | Local embeddings (bge-m3) | System service | `ollama serve` |
| Python venv | Script runtime | `$KB_REPO/.venv/` | `source .venv/bin/activate` |
| KB directory | Data, config, outputs | `$KB_ROOT/` | `init.py` scaffolds it |
| LLM API | Entity/fact extraction | External (OpenAI/Anthropic) | API key in `$KB_ROOT/.env` |

---

## Tearing Down

```bash
# Stop Qdrant (preserves data)
cd $KB_REPO && docker compose down

# Stop Qdrant and delete all data
cd $KB_REPO && docker compose down -v

# Remove Python venv
rm -rf $KB_REPO/.venv
```

The KB directory (`$KB_ROOT`) contains your raw documents, notes, and configs — it's independent of the infrastructure and safe to keep.
