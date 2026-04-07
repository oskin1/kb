#!/usr/bin/env python3
"""
write_note.py — Write a structured research note as markdown AND ingest into Qdrant.

This is the primary tool for persisting session output: analysis, proposals,
conclusions, hypotheses. Produces a human-readable .md file in kb/notes/ AND
stores each section as a searchable insight in the `insights` collection.

Usage:
    python scripts/write_note.py <note_file.md> [options]
    python scripts/write_note.py --stdin [options]   # pipe markdown from stdin

The note file is plain markdown. Metadata is set via CLI or a YAML front matter block.

Front matter example (optional, overrides CLI args):
    ---
    title: "HoneyComb paper — implications for our KB"
    domain: cross
    project: kb-infra
    tags: [LLM, agent, materials-science, KB-design]
    confidence: probable
    source: "session 2026-03-16"
    ---

    # HoneyComb paper — implications for our KB
    ...

Options:
    --title         Note title (required if no front matter)
    --domain        Domain tag (default: cross)
    --subdomain     Subdomain (optional)
    --project       Project name (default: general)
    --tags          Comma-separated tags
    --confidence    established|probable|speculative|needs-verification (default: probable)
    --source        Source reference (default: session YYYY-MM-DD)
    --type          summary|conclusion|hypothesis|proposal|note (default: note)
    --no-ingest     Save markdown only, skip Qdrant ingest
    --output        Output path override (default: kb/notes/YYYY-MM-DD_<slug>.md)

Examples:
    python scripts/write_note.py /tmp/session_note.md --title "HoneyComb insights" --domain cross --project kb-infra --tags LLM,KB --ingest
    python scripts/write_note.py --stdin --title "Quick note" < note.md
"""

import argparse
import re
import sys
import uuid
import yaml
from datetime import date
from pathlib import Path

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
NOTES_DIR = Path(__file__).parent.parent / "notes"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60].strip("-")


def parse_front_matter(content: str) -> tuple[dict, str]:
    """Parse YAML front matter from markdown. Returns (meta, body)."""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm_text = content[3:end].strip()
            body = content[end + 4:].strip()
            try:
                return yaml.safe_load(fm_text) or {}, body
            except yaml.YAMLError:
                pass
    return {}, content


def build_output_path(title: str, today: str) -> Path:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    candidate = NOTES_DIR / f"{today}_{slug}.md"
    if candidate.exists():
        candidate = NOTES_DIR / f"{today}_{slug}_{uuid.uuid4().hex[:4]}.md"
    return candidate


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    resp = ollama_client.embed(model=model, input=texts)
    return resp["embeddings"]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    return [n.get_content() for n in nodes]


def ingest_note(body: str, meta: dict, output_path: Path, cfg: dict) -> int:
    """Ingest note body into insights collection. Returns chunk count."""
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]

    chunks = chunk_text(body, chunk_size, chunk_overlap)
    if not chunks:
        return 0

    vectors = embed_texts(chunks, embed_model)
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "text": chunk,
            "title": meta.get("title", output_path.stem),
            "authors": [],
            "domain": meta.get("domain", "cross"),
            "subdomain": meta.get("subdomain", ""),
            "source_type": "own-analysis",
            "source": meta.get("source", f"session {today}"),
            "confidence": meta.get("confidence", "probable"),
            "tags": meta.get("tags", []),
            "project": meta.get("project", "general"),
            "date": today,
            "language": meta.get("language", "en"),
            "insight_type": meta.get("type", "note"),
            "note_file": str(output_path),
            "source_doc_ids": meta.get("source_doc_ids", []),
            "_doc_id": doc_id,
            "_chunk_index": i,
            "_chunk_total": len(chunks),
            "_added": today,
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    client.upsert(collection_name="insights", points=points)
    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Write a research note + ingest into KB")
    parser.add_argument("file", nargs="?", help="Markdown file to write/ingest")
    parser.add_argument("--stdin", action="store_true", help="Read content from stdin")
    parser.add_argument("--title", help="Note title")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--subdomain", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--tags", default=None, help="Comma-separated tags")
    parser.add_argument("--confidence", default=None,
                        choices=["established", "probable", "speculative", "needs-verification"])
    parser.add_argument("--source", default=None)
    parser.add_argument("--type", dest="note_type", default=None,
                        choices=["summary", "conclusion", "hypothesis", "proposal", "note"])
    parser.add_argument("--no-ingest", action="store_true", help="Skip Qdrant ingest")
    parser.add_argument("--output", help="Override output path")
    args = parser.parse_args()

    # Read content
    if args.stdin:
        content = sys.stdin.read()
    elif args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Error: provide a file path or --stdin")
        sys.exit(1)

    # Parse front matter
    fm, body = parse_front_matter(content)

    # Merge: CLI args > front matter > defaults
    today = date.today().isoformat()
    meta = {
        "title":          args.title      or fm.get("title") or "Untitled Note",
        "domain":         args.domain     or fm.get("domain", "cross"),
        "subdomain":      args.subdomain  or fm.get("subdomain", ""),
        "project":        args.project    or fm.get("project", "general"),
        "confidence":     args.confidence or fm.get("confidence", "probable"),
        "source":         args.source     or fm.get("source", f"session {today}"),
        "type":           args.note_type  or fm.get("type", "note"),
        "language":       fm.get("language", "en"),
        "source_doc_ids": fm.get("source_doc_ids", []),
    }

    # Tags: CLI > front matter
    if args.tags:
        meta["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    elif fm.get("tags"):
        t = fm["tags"]
        meta["tags"] = t if isinstance(t, list) else [x.strip() for x in str(t).split(",")]
    else:
        meta["tags"] = []

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.file and not args.stdin:
        # If source file is already in notes/, use it in place; otherwise copy to notes/
        src = Path(args.file).resolve()
        if str(src).startswith(str(NOTES_DIR.resolve())):
            output_path = src
        else:
            output_path = build_output_path(meta["title"], today)
    else:
        output_path = build_output_path(meta["title"], today)

    # Build final markdown with front matter header
    fm_block = yaml.dump(
        {k: v for k, v in meta.items() if v},
        allow_unicode=True, default_flow_style=False
    ).strip()
    final_content = f"---\n{fm_block}\n---\n\n{body.strip()}\n"

    # Write markdown file
    output_path.write_text(final_content, encoding="utf-8")
    print(f"✓ Note saved: {output_path}")

    # Ingest into Qdrant
    if not args.no_ingest:
        cfg = load_config()
        chunks = ingest_note(body, meta, output_path, cfg)
        print(f"✓ Ingested {chunks} chunks into [insights]")
        print(f"  Title:   {meta['title']}")
        print(f"  Domain:  {meta['domain']}" + (f" / {meta['subdomain']}" if meta.get('subdomain') else ""))
        print(f"  Project: {meta['project']}")
        print(f"  Tags:    {', '.join(meta['tags']) if meta['tags'] else '—'}")
    else:
        print(f"  (ingest skipped)")


if __name__ == "__main__":
    main()
