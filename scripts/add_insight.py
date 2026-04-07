#!/usr/bin/env python3
"""
add_insight.py — Store an agent-generated insight into the insights collection.

Usage:
    python add_insight.py "Your insight text here" \
        --type conclusion \
        --domain mydomain \
        --subdomain subtopic \
        --confidence established \
        --tags tag1,tag2 \
        --project myproject \
        --source "session 2026-03-15" \
        --source-docs <doc_id1> <doc_id2>
"""

import argparse
import uuid
import yaml
from datetime import date
from pathlib import Path

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Add an insight to the KB")
    parser.add_argument("text", help="The insight text")
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

    cfg = load_config()
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
