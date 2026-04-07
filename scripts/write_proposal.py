#!/usr/bin/env python3
"""
write_proposal.py — Write a structured proposal and ingest into Qdrant insights.

Usage:
    python scripts/write_proposal.py /tmp/proposal.md --kb-root /path/to/kb
    python scripts/write_proposal.py --stdin --kb-root ~/kb --title "Add formulas collection"
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

from kb_root import add_kb_root_arg, resolve_kb_root, load_config, proposals_dir


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60].strip("-")


def parse_front_matter(content: str) -> tuple[dict, str]:
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            try:
                fm = yaml.safe_load(content[3:end].strip()) or {}
                return fm, content[end + 4:].strip()
            except yaml.YAMLError:
                pass
    return {}, content


def build_output_path(kb_root: Path, title: str, today: str) -> Path:
    pd = proposals_dir(kb_root)
    pd.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    candidate = pd / f"{today}_{slug}.md"
    if candidate.exists():
        candidate = pd / f"{today}_{slug}_{uuid.uuid4().hex[:4]}.md"
    return candidate


def ingest_proposal(body: str, meta: dict, output_path: Path, cfg: dict) -> int:
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([Document(text=body)])
    chunks = [n.get_content() for n in nodes]
    if not chunks:
        return 0

    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()

    vectors = ollama_client.embed(model=embed_model, input=chunks)["embeddings"]
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
            "language": "en",
            "insight_type": "proposal",
            "proposal_status": meta.get("status", "open"),
            "proposal_file": str(output_path),
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
    parser = argparse.ArgumentParser(description="Write a proposal and ingest into KB")
    parser.add_argument("file", nargs="?", help="Markdown file")
    add_kb_root_arg(parser)
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--title", help="Proposal title")
    parser.add_argument("--project", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--subdomain", default=None)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--confidence", default=None,
                        choices=["established", "probable", "speculative", "needs-verification"])
    parser.add_argument("--status", default=None,
                        choices=["open", "accepted", "in-progress", "done", "rejected"])
    parser.add_argument("--no-ingest", action="store_true")
    parser.add_argument("--output", help="Override output path")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)

    if args.stdin:
        content = sys.stdin.read()
    elif args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Error: provide a file path or --stdin"); sys.exit(1)

    fm, body = parse_front_matter(content)
    today = date.today().isoformat()

    meta = {
        "title":          args.title      or fm.get("title", "Untitled Proposal"),
        "project":        args.project    or fm.get("project", "general"),
        "domain":         args.domain     or fm.get("domain", "cross"),
        "subdomain":      args.subdomain  or fm.get("subdomain", ""),
        "confidence":     args.confidence or fm.get("confidence", "probable"),
        "status":         args.status     or fm.get("status", "open"),
        "source":         fm.get("source", f"session {today}"),
        "created":        fm.get("created", today),
        "source_doc_ids": fm.get("source_doc_ids", []),
    }
    if args.tags:
        meta["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    elif fm.get("tags"):
        t = fm["tags"]
        meta["tags"] = t if isinstance(t, list) else [x.strip() for x in str(t).split(",")]
    else:
        meta["tags"] = []

    output_path = Path(args.output) if args.output else build_output_path(kb_root, meta["title"], today)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fm_out = {k: v for k, v in meta.items() if v}
    fm_out["updated"] = today
    final = f"---\n{yaml.dump(fm_out, allow_unicode=True, default_flow_style=False).strip()}\n---\n\n{body.strip()}\n"
    output_path.write_text(final, encoding="utf-8")
    print(f"✓ Proposal saved: {output_path}")

    if not args.no_ingest:
        cfg = load_config(kb_root)
        chunks = ingest_proposal(body, meta, output_path, cfg)
        print(f"✓ Ingested {chunks} chunks into [insights]")
        print(f"  Title:   {meta['title']}")
        print(f"  Project: {meta['project']} | Status: {meta['status']}")
        print(f"  Tags:    {', '.join(meta['tags']) if meta['tags'] else '—'}")


if __name__ == "__main__":
    main()
