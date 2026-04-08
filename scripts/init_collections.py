#!/usr/bin/env python3
"""
init_collections.py — Create Qdrant collections with correct schema.
Run once on fresh setup, safe to re-run (skips existing collections).

Usage:
    python scripts/init_collections.py --kb-root /path/to/kb

Collections:
  research_docs  — papers, web, textbooks
  data_tables    — structured/numerical data
  insights       — agent-generated conclusions
  entities       — named entities extracted from docs (Phase 2)
"""

import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType, TextIndexParams, TokenizerType

from kb_root import add_kb_root_arg, resolve_kb_root, load_config

# Fields and their index types per collection type
STANDARD_FIELDS = {
    "domain":         PayloadSchemaType.KEYWORD,
    "subdomain":      PayloadSchemaType.KEYWORD,
    "source_type":    PayloadSchemaType.KEYWORD,
    "confidence":     PayloadSchemaType.KEYWORD,
    "tags":           PayloadSchemaType.KEYWORD,
    "project":        PayloadSchemaType.KEYWORD,
    "language":       PayloadSchemaType.KEYWORD,
    "date":           PayloadSchemaType.KEYWORD,
    "_added":         PayloadSchemaType.KEYWORD,
    "_doc_id":        PayloadSchemaType.KEYWORD,
    # Phase 1 — temporal validity
    "valid_at":       PayloadSchemaType.KEYWORD,
    "invalid_at":     PayloadSchemaType.KEYWORD,
    "superseded_by":  PayloadSchemaType.KEYWORD,
}

# Phase 2 — entities collection extra fields
ENTITY_FIELDS = {
    "entity_id":   PayloadSchemaType.KEYWORD,
    "name":        PayloadSchemaType.KEYWORD,
    "type":        PayloadSchemaType.KEYWORD,      # material|process|property|organization|person|concept
    "aliases":     PayloadSchemaType.KEYWORD,
    "doc_ids":     PayloadSchemaType.KEYWORD,      # source docs this entity appears in
    "domain":      PayloadSchemaType.KEYWORD,
    "tags":        PayloadSchemaType.KEYWORD,
    "_added":      PayloadSchemaType.KEYWORD,
    "community_id": PayloadSchemaType.INTEGER,   # graph clustering (Phase 5)
}


def create_collection(client, name: str, dims: int, fields: dict):
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        print(f"  [skip] {name} already exists")
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )
    print(f"  [created] {name} ({dims}d, cosine)")

    for field, schema_type in fields.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema_type,
        )

    # Phase 4 — BM25 text index on primary text fields
    text_fields = {
        "research_docs": ["text"],
        "entities":      ["name", "summary"],
        "facts":         ["fact"],
    }
    for tf in text_fields.get(name, []):
        client.create_payload_index(
            collection_name=name,
            field_name=tf,
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
            ),
        )
        print(f"  [text-indexed] {name}.{tf}")

    print(f"  [indexed] {len(fields)} payload fields for {name}")


def init_collections(cfg):
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    dims = cfg["embedding"]["dimensions"]

    # Standard collections
    for name in cfg["collections"].values():
        create_collection(client, name, dims, STANDARD_FIELDS)

    # Phase 2 — entities collection
    entities_name = cfg.get("collections_extra", {}).get("entities", "entities")
    create_collection(client, entities_name, dims, ENTITY_FIELDS)

    print("\nDone. Collections:")
    for c in client.get_collections().collections:
        info = client.get_collection(c.name)
        print(f"  {c.name}: {info.points_count} points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Qdrant collections with correct schema")
    add_kb_root_arg(parser)
    args = parser.parse_args()
    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    print(f"Connecting to Qdrant at {cfg['qdrant']['host']}:{cfg['qdrant']['port']}...")
    init_collections(cfg)
