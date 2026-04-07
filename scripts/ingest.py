#!/usr/bin/env python3
"""
ingest.py — Ingest documents into the Qdrant knowledge base.

Usage:
    python ingest.py <file_or_dir> --kb-root /path/to/kb
                                   [--collection research_docs|data_tables|insights]
                                   [--meta key=value ...]
                                   [--valid-at YYYY-MM-DD|YYYY-QN|YYYY-MM|YYYY]
                                   [--invalid-at YYYY-MM-DD|YYYY-QN|YYYY-MM|YYYY]
                                   [--supersedes <doc_id>]

Examples:
    python ingest.py raw/papers/document.pdf --kb-root ~/my-kb
    python ingest.py raw/papers/ --kb-root ~/my-kb --collection research_docs
    python ingest.py raw/papers/document.pdf --kb-root ~/my-kb --meta project=myproject domain=mydomain
    python ingest.py raw/data/data_2025.csv --kb-root ~/my-kb --valid-at 2025-Q1
    python ingest.py raw/papers/new_doc.pdf --kb-root ~/my-kb --supersedes 1494731e-xxxx

Temporal shorthand: 2024 → 2024-01-01, 2024-Q3 → 2024-07-01, 2024-03 → 2024-03-01
Sidecar YAML (optional, same path as file but .yaml extension) overrides auto metadata.
"""

import sys
import argparse
import uuid
import yaml
import json
from datetime import date
from pathlib import Path

import fitz  # pymupdf
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.core.node_parser import SentenceSplitter
import ollama as ollama_client

from kb_root import add_kb_root_arg, resolve_kb_root, load_config

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm", ".csv"}

# Quarter shorthand → month start
QUARTER_MAP = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}


def parse_date_shorthand(value: str) -> str | None:
    """
    Parse flexible date input into ISO YYYY-MM-DD string.
    Accepts: 2024, 2024-Q3, 2024-03, 2024-03-15
    Returns None if value is None or empty.
    """
    if not value:
        return None
    value = value.strip()

    # Quarter shorthand: 2024-Q3 or 2024Q3
    import re
    m = re.match(r"^(\d{4})[- ]?(Q[1-4])$", value, re.IGNORECASE)
    if m:
        year, quarter = m.group(1), m.group(2).upper()
        month = QUARTER_MAP[quarter]
        return f"{year}-{month}-01"

    # Year only: 2024
    m = re.match(r"^(\d{4})$", value)
    if m:
        return f"{value}-01-01"

    # Year-Month: 2024-03
    m = re.match(r"^(\d{4})-(\d{2})$", value)
    if m:
        return f"{value}-01"

    # Full date: 2024-03-15 — return as-is
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if m:
        return value

    raise ValueError(f"Unrecognised date format: '{value}'. "
                     "Use YYYY, YYYY-QN, YYYY-MM, or YYYY-MM-DD.")

def load_sidecar(file_path: Path) -> dict:
    sidecar = file_path.with_suffix(".yaml")
    if sidecar.exists():
        with open(sidecar) as f:
            return yaml.safe_load(f) or {}
    return {}

def extract_text(file_path: Path) -> tuple[str, dict]:
    """Extract raw text and auto metadata from file. Returns (text, meta)."""
    ext = file_path.suffix.lower()
    meta = {}

    if ext == ".pdf":
        doc = fitz.open(str(file_path))
        pdf_meta = doc.metadata
        meta["title"] = pdf_meta.get("title", "").strip() or file_path.stem
        meta["authors"] = [a.strip() for a in pdf_meta.get("author", "").split(";") if a.strip()]
        # Extract text, skip last pages (usually references)
        pages = list(doc)
        text_pages = pages[:-2] if len(pages) > 4 else pages
        text = "\n".join(p.get_text() for p in text_pages)
        doc.close()

    elif ext in {".html", ".htm"}:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        title_tag = soup.find("title")
        meta["title"] = title_tag.get_text().strip() if title_tag else file_path.stem
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    elif ext == ".csv":
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        meta["title"] = file_path.stem

    else:  # .txt, .md
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        meta["title"] = file_path.stem

    return text, meta

def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts using Ollama batch API (single round-trip)."""
    resp = ollama_client.embed(model=model, input=texts)
    return resp["embeddings"]

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    from llama_index.core import Document
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    return [n.get_content() for n in nodes]

def apply_supersedes(client, collection: str, old_doc_id: str, new_doc_id: str,
                     new_valid_at: str | None, cfg: dict):
    """
    Mark all chunks of old_doc_id as superseded:
      - Set superseded_by = new_doc_id
      - Set invalid_at = new_valid_at (or today if new_valid_at is None)
    Scrolls through all matching chunks.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SetPayload

    expiry_date = new_valid_at or date.today().isoformat()
    offset = None
    total_updated = 0

    while True:
        results, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="_doc_id", match=MatchValue(value=old_doc_id))]
            ),
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not results:
            break

        ids = [str(r.id) for r in results]
        client.set_payload(
            collection_name=collection,
            payload={
                "superseded_by": new_doc_id,
                "invalid_at": expiry_date,
            },
            points=ids,
        )
        total_updated += len(ids)

        if offset is None:
            break

    print(f"  ✓ superseded {total_updated} chunks of doc {old_doc_id[:8]}... "
          f"(invalid_at={expiry_date})")


def ingest_file(file_path: Path, collection: str, extra_meta: dict, cfg: dict,
                valid_at: str | None = None, invalid_at: str | None = None,
                supersedes: str | None = None):
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]

    print(f"\n→ Ingesting: {file_path.name}")
    if valid_at:
        print(f"  valid_at:  {valid_at}")
    if invalid_at:
        print(f"  invalid_at: {invalid_at}")
    if supersedes:
        print(f"  supersedes: {supersedes[:8]}...")

    # Extract text + auto metadata
    text, auto_meta = extract_text(file_path)
    if not text.strip():
        print("  [skip] empty content")
        return

    # Load sidecar, merge: extra_meta > sidecar > auto_meta
    sidecar = load_sidecar(file_path)
    payload_base = {**auto_meta, **sidecar, **extra_meta}

    # Fill required fields with defaults
    payload_base.setdefault("title", file_path.stem)
    payload_base.setdefault("authors", [])
    payload_base.setdefault("domain", "cross")
    payload_base.setdefault("subdomain", "")
    payload_base.setdefault("source_type", "paper")
    payload_base.setdefault("source", str(file_path))
    payload_base.setdefault("confidence", "probable")
    payload_base.setdefault("tags", [])
    payload_base.setdefault("project", "general")
    payload_base.setdefault("date", "")
    payload_base.setdefault("language", "en")
    payload_base["_added"] = date.today().isoformat()

    # Phase 1 — temporal validity fields
    payload_base["valid_at"] = valid_at        # None = timeless/unknown
    payload_base["invalid_at"] = invalid_at    # None = still valid
    payload_base["superseded_by"] = None       # set later if --supersedes used

    # Chunk
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    doc_id = str(uuid.uuid4())
    print(f"  chunks: {len(chunks)}, doc_id: {doc_id[:8]}...")

    # Embed
    print(f"  embedding {len(chunks)} chunks via {embed_model}...")
    vectors = embed_texts(chunks, embed_model)

    # Build points
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            **payload_base,
            "text": chunk,
            "_doc_id": doc_id,
            "_chunk_index": i,
            "_chunk_total": len(chunks),
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        ))

    # Upsert to Qdrant
    client.upsert(collection_name=collection, points=points)
    print(f"  ✓ stored {len(points)} chunks in [{collection}]")

    # Phase 1 — handle supersedes: mark old doc as superseded
    if supersedes:
        auto_invalidate = cfg.get("temporal", {}).get("auto_invalidate_on_supersedes", True)
        if auto_invalidate:
            apply_supersedes(client, collection, supersedes, doc_id, valid_at, cfg)
        else:
            print(f"  [info] --supersedes set but auto_invalidate_on_supersedes=false in config")

    return doc_id


def run_entity_extraction(doc_id: str, cfg: dict, verbose: bool = False):
    """
    Phase 2 — run entity extraction on a freshly ingested doc.
    Imported from entity_extract.py to avoid subprocess overhead.
    Silently skips if OPENAI_API_KEY is not set (extraction is optional).
    """
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [skip] entity extraction: OPENAI_API_KEY not set")
        return

    try:
        # Import here to avoid circular dependency at module level
        sys.path.insert(0, str(Path(__file__).parent))
        from entity_extract import process_doc, get_llm_client, get_qdrant_client
        llm = get_llm_client(cfg)
        qdrant = get_qdrant_client(cfg)
        process_doc(doc_id, cfg, llm, qdrant, dry_run=False, verbose=verbose)
    except Exception as e:
        print(f"  [warn] entity extraction failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant KB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Temporal date formats: YYYY, YYYY-QN, YYYY-MM, YYYY-MM-DD
  Examples: 2024, 2024-Q3, 2024-03, 2024-03-15

Supersedes example (replacing an old economics report):
  python ingest.py new_prices.csv --kb-root ~/my-kb --valid-at 2025-Q1 --supersedes <old_doc_id>
        """,
    )
    parser.add_argument("path", help="File or directory to ingest")
    add_kb_root_arg(parser)
    parser.add_argument("--collection", default="research_docs",
                        choices=["research_docs", "data_tables", "insights"],
                        help="Target collection (default: research_docs)")
    parser.add_argument("--meta", nargs="*", default=[],
                        help="Extra metadata as key=value pairs")
    parser.add_argument("--no-entities", action="store_true",
                        help="Skip entity extraction after ingest (phase 2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose entity extraction output")
    parser.add_argument("--valid-at", dest="valid_at", default=None,
                        help="When this content became true (YYYY, YYYY-QN, YYYY-MM, YYYY-MM-DD)")
    parser.add_argument("--invalid-at", dest="invalid_at", default=None,
                        help="When this content was superseded (same format as --valid-at)")
    parser.add_argument("--supersedes", default=None, metavar="DOC_ID",
                        help="doc_id of older document this replaces; marks it as superseded")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)

    # Parse date shorthands
    try:
        valid_at = parse_date_shorthand(args.valid_at)
        invalid_at = parse_date_shorthand(args.invalid_at)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    extra_meta = {}
    for kv in args.meta or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            if "," in v:
                extra_meta[k] = [x.strip() for x in v.split(",")]
            else:
                extra_meta[k] = v

    path = Path(args.path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            doc_id = ingest_file(path, args.collection, extra_meta, cfg,
                                  valid_at=valid_at, invalid_at=invalid_at,
                                  supersedes=args.supersedes)
            # Phase 2 — entity extraction (only for research_docs)
            if doc_id and not args.no_entities and args.collection == "research_docs":
                run_entity_extraction(doc_id, cfg, verbose=args.verbose)
        else:
            print(f"Unsupported file type: {path.suffix}")
            sys.exit(1)
    elif path.is_dir():
        files = [f for f in sorted(path.rglob("*")) if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        print(f"Found {len(files)} files in {path}")
        if args.supersedes:
            print("  [warning] --supersedes with a directory: applies to ALL files in dir")
        for f in files:
            doc_id = ingest_file(f, args.collection, extra_meta, cfg,
                                  valid_at=valid_at, invalid_at=invalid_at,
                                  supersedes=args.supersedes)
            if doc_id and not args.no_entities and args.collection == "research_docs":
                run_entity_extraction(doc_id, cfg, verbose=args.verbose)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
