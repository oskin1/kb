#!/usr/bin/env python3
"""
add_insight.py — Store an agent-generated insight into the insights collection.

Usage:
    python add_insight.py "Your insight text here" --kb-root /path/to/kb \
        --type conclusion --domain mydomain --confidence established
"""

import argparse
import uuid
from datetime import date

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from kb_root import add_kb_root_arg, resolve_kb_root, load_config

def main():
    parser = argparse.ArgumentParser(description="Add an insight to the KB")
    parser.add_argument("text", help="The insight text")
    add_kb_root_arg(parser)
    parser.add_argument("--type", dest="insight_type", default="conclusion",
                        choices=["summary", "conclusion", "hypothesis", "gap", "connection"],
                        help="Insight type")
    parser.add_argument("--domain", default="cross")
    parser.add_argument("--subdomain", default="")
    parser.add_argument("--confidence", default="probable",
                        choices=["established", "probable", "speculative", "needs-verification"])
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--project", default="general")
    parser.add_argument("--source", default="", help="Source reference or session date")
    parser.add_argument("--source-docs", nargs="*", default=[], help="Source doc_ids from research_docs")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]

    print("Embedding insight...")
    vector = ollama_client.embed(model=embed_model, input=[args.text])["embeddings"][0]

    today = date.today().isoformat()
    payload = {
        "text": args.text,
        "title": f"[{args.insight_type}] {args.text[:60]}...",
        "authors": [],
        "domain": args.domain,
        "subdomain": args.subdomain,
        "source_type": "own-analysis",
        "source": args.source or f"session {today}",
        "confidence": args.confidence,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "project": args.project,
        "date": today,
        "language": args.lang,
        "insight_type": args.insight_type,
        "source_doc_ids": args.source_docs,
        "_doc_id": str(uuid.uuid4()),
        "_chunk_index": 0,
        "_chunk_total": 1,
        "_added": today,
    }

    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
    client.upsert(collection_name="insights", points=[point])
    print(f"✓ Insight stored in [insights]: \"{args.text[:80]}...\"")

if __name__ == "__main__":
    main()
