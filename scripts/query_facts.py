#!/usr/bin/env python3
"""
query_facts.py — Semantic search over the facts collection.

Usage:
    python scripts/query_facts.py "your search query"
    python scripts/query_facts.py --subject "Entity Name"
    python scripts/query_facts.py --relation CONTAINS
    python scripts/query_facts.py --relation CONTAINS --subject "Entity Name"
    python scripts/query_facts.py "topic" --domain mydomain --top 10
    python scripts/query_facts.py "topic" --as-of 2024-01-01
"""

import argparse
import sys
from pathlib import Path

import ollama as ollama_client
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
KB_DIR = Path(__file__).parent.parent


def load_env():
    env_file = KB_DIR / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)


def load_config():
    load_env()
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def embed_text(text: str, model: str) -> list[float]:
    resp = ollama_client.embeddings(model=model, prompt=text)
    return resp["embedding"]


def parse_as_of(value: str):
    import re
    QUARTER_MAP = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    m = re.match(r"^(\d{4})[- ]?(Q[1-4])$", value, re.IGNORECASE)
    if m:
        year, q = m.group(1), m.group(2).upper()
        return f"{year}-{QUARTER_MAP[q]}-01"
    m = re.match(r"^(\d{4})$", value)
    if m:
        return f"{value}-01-01"
    m = re.match(r"^(\d{4})-(\d{2})$", value)
    if m:
        return f"{value}-01"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if m:
        return value
    raise ValueError(f"Unrecognised date format: '{value}'")


def passes_temporal(payload: dict, as_of: str) -> bool:
    valid_at = payload.get("valid_at")
    invalid_at = payload.get("invalid_at")
    if valid_at is None:
        return True
    if valid_at > as_of:
        return False
    if invalid_at is not None and invalid_at <= as_of:
        return False
    return True


def search(query: str | None, cfg: dict,
           subject: str | None, relation: str | None, domain: str | None,
           as_of: str | None, top: int):
    qdrant = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")
    embed_model = cfg["embedding"]["model"]

    # Build filter
    must = []
    if subject:
        must.append(FieldCondition(key="subject_name", match=MatchValue(value=subject)))
    if relation:
        must.append(FieldCondition(key="relation_type",
                                    match=MatchValue(value=relation.upper())))
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    filt = Filter(must=must) if must else None

    fetch_limit = top * 3 if as_of else top * 2

    if query:
        # Hybrid: cosine + BM25 on fact field
        from qdrant_client.models import MatchText
        print(f"Hybrid search (cosine + BM25)...")
        vector = embed_text(query, embed_model)
        cosine_results = qdrant.query_points(
            collection_name=facts_col,
            query=vector,
            query_filter=filt,
            limit=fetch_limit,
            with_payload=True,
        ).points

        # BM25 on fact field
        text_cond = FieldCondition(key="fact", match=MatchText(text=query))
        bm25_filter = Filter(must=(list(filt.must) if filt and filt.must else []) + [text_cond])
        bm25_raw, _ = qdrant.scroll(
            collection_name=facts_col,
            scroll_filter=bm25_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )

        # RRF merge
        k = 60
        scores: dict = {}
        payloads_map: dict = {}
        for rank, r in enumerate(cosine_results, 1):
            rid = str(r.id)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            payloads_map[rid] = r.payload
        for rank, r in enumerate(bm25_raw, 1):
            rid = str(r.id)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            if rid not in payloads_map:
                payloads_map[rid] = r.payload

        class _MP:
            def __init__(self, rid):
                self.id = rid
                self.score = scores[rid]
                self.payload = payloads_map[rid]

        results = [_MP(rid) for rid in sorted(scores, key=lambda x: scores[x], reverse=True)]

    else:
        # Filter-only mode (no semantic query)
        results, _ = qdrant.scroll(
            collection_name=facts_col,
            scroll_filter=filt,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )

    if not results:
        print("No facts found.")
        return

    # Temporal post-filter
    if as_of:
        results = [r for r in results if passes_temporal(r.payload if hasattr(r, 'payload') else r, as_of)]

    if not results:
        print(f"No facts valid as of {as_of}.")
        return

    as_of_label = f" | As-of: {as_of}" if as_of else ""
    q_label = f'"{query}"' if query else "(filter only)"
    print(f"\n{'─'*72}")
    print(f"  Query: {q_label}  |  Top {min(top, len(results))}{as_of_label}")
    print(f"{'─'*72}\n")

    for i, r in enumerate(results[:top], 1):
        p = r.payload if hasattr(r, 'payload') else r
        score_str = f"  score={r.score:.4f}" if hasattr(r, 'score') else ""
        superseded = "  ⚠ SUPERSEDED" if p.get("superseded_by") else ""
        valid_at = p.get("valid_at")
        invalid_at = p.get("invalid_at")
        temporal = f"  [{valid_at} → {invalid_at or 'now'}]" if valid_at else ""

        print(f"[{i}]{score_str}{superseded}")
        print(f"    ({p.get('subject_name','?')}) "
              f"──[{p.get('relation_type','?')}]──▶ "
              f"({p.get('object_name','?')})")
        print(f"    {p.get('fact','')}")
        print(f"    conf={p.get('confidence','?')} | domain={p.get('domain','?')}"
              f"{temporal}")
        print(f"    src={p.get('source_doc_id','?')[:8]}...")
        print(f"\n{'─'*72}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Search the facts collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Semantic search query")
    parser.add_argument("--subject", help="Filter by subject entity name")
    parser.add_argument("--relation", help="Filter by relation_type (e.g. CONTAINS)")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--as-of", dest="as_of", help="Temporal filter (YYYY, YYYY-QN, etc.)")
    parser.add_argument("--top", type=int, default=10, help="Number of results (default: 10)")
    args = parser.parse_args()

    if not args.query and not args.subject and not args.relation and not args.domain:
        parser.print_help()
        return

    cfg = load_config()

    as_of = None
    if args.as_of:
        try:
            as_of = parse_as_of(args.as_of)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    search(args.query, cfg,
           subject=args.subject, relation=args.relation,
           domain=args.domain, as_of=as_of, top=args.top)


if __name__ == "__main__":
    main()
