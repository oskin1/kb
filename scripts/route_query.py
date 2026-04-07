#!/usr/bin/env python3
"""
route_query.py — Smart query router for the KB.

Automatically selects the best collection(s) and domain filter based on
the query text. Searches research_docs, insights, AND facts in parallel,
then merges and ranks results.

Usage:
    python scripts/route_query.py "your search query" --kb-root /path/to/kb
    python scripts/route_query.py "specific topic" --kb-root ~/kb --domain mydomain
    python scripts/route_query.py "broad query" --kb-root ~/kb --top 8
    python scripts/route_query.py "topic" --kb-root ~/kb --facts-only
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from kb_root import add_kb_root_arg, resolve_kb_root, load_config, kb_root_from_cfg
from domain_config import detect_domain


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(text: str, model: str) -> list[float]:
    resp = ollama_client.embeddings(model=model, prompt=text)
    return resp["embedding"]


# ── Per-collection search ─────────────────────────────────────────────────────

def search_collection(client: QdrantClient, collection: str, vector: list[float],
                       domain: str | None, top: int) -> list[dict]:
    """Search a single collection. Returns list of result dicts."""
    must = []
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    filt = Filter(must=must) if must else None

    try:
        results = client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=filt,
            limit=top,
            with_payload=True,
        ).points
    except Exception as e:
        print(f"  [warn] {collection}: {e}", file=sys.stderr)
        return []

    out = []
    for r in results:
        out.append({
            "score": r.score,
            "collection": collection,
            "payload": r.payload,
        })
    return out


def search_facts(client: QdrantClient, vector: list[float],
                 domain: str | None, relation: str | None, top: int) -> list[dict]:
    """Search facts collection with optional relation filter."""
    must = []
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if relation:
        must.append(FieldCondition(key="relation_type",
                                    match=MatchValue(value=relation.upper())))
    filt = Filter(must=must) if must else None

    try:
        results = client.query_points(
            collection_name="facts",
            query=vector,
            query_filter=filt,
            limit=top,
            with_payload=True,
        ).points
    except Exception as e:
        print(f"  [warn] facts: {e}", file=sys.stderr)
        return []

    out = []
    for r in results:
        out.append({
            "score": r.score,
            "collection": "facts",
            "payload": r.payload,
        })
    return out


# ── Rendering ─────────────────────────────────────────────────────────────────

COLLECTION_LABELS = {
    "research_docs": "📄 PAPER/WEB",
    "insights":      "💡 INSIGHT",
    "facts":         "🔗 FACT",
}

def render_result(i: int, result: dict):
    col = result["collection"]
    p = result["payload"]
    score = result["score"]
    label = COLLECTION_LABELS.get(col, col.upper())

    print(f"[{i}] {label}  score={score:.4f}")

    if col == "facts":
        superseded = "  ⚠ SUPERSEDED" if p.get("superseded_by") else ""
        print(f"    ({p.get('subject_name','?')}) "
              f"──[{p.get('relation_type','?')}]──▶ "
              f"({p.get('object_name','?')}){superseded}")
        print(f"    {p.get('fact','')}")
        print(f"    conf={p.get('confidence','?')} | domain={p.get('domain','?')}")
        src = (p.get('source_doc_id') or '')[:8]
        if src:
            print(f"    src={src}...")
    elif col == "insights":
        print(f"    {p.get('title', '—')}")
        print(f"    domain={p.get('domain','—')} | src={p.get('source','—')}")
        text = p.get("text", "")
        preview = text[:300].replace("\n", " ").strip()
        if len(text) > 300:
            preview += "…"
        if preview:
            print(f"    {preview}")
    else:  # research_docs / data_tables
        print(f"    Title: {p.get('title','—')}")
        print(f"    domain={p.get('domain','—')} | {p.get('source_type','—')}")
        print(f"    Chunk: {p.get('_chunk_index',0)+1}/{p.get('_chunk_total','?')}")
        text = p.get("text", "")
        preview = text[:300].replace("\n", " ").strip()
        if len(text) > 300:
            preview += "…"
        if preview:
            print(f"    {preview}")

    print(f"\n{'─'*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smart query router — searches all relevant KB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Search query text")
    add_kb_root_arg(parser)
    parser.add_argument("--domain",
                        help="Override domain filter (auto-detected if not set)")
    parser.add_argument("--top", type=int, default=5,
                        help="Results per collection (default: 5)")
    parser.add_argument("--merged-top", type=int, default=10,
                        help="Total merged results to show (default: 10)")
    parser.add_argument("--collections", default="docs,insights,facts",
                        help="Comma-separated collections to search: "
                             "docs, insights, facts (default: all three)")
    parser.add_argument("--facts-only", action="store_true",
                        help="Search only the facts collection")
    parser.add_argument("--relation",
                        help="Filter facts by relation type (e.g. CONTAINS, USED_IN)")
    parser.add_argument("--no-auto-domain", action="store_true",
                        help="Disable automatic domain detection")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)

    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]

    # Resolve domain
    if args.domain:
        domain = args.domain if args.domain != "cross" else None
    elif args.no_auto_domain:
        domain = None
    else:
        domain = detect_domain(kb_root, args.query)

    # Resolve collections
    if args.facts_only:
        collections = ["facts"]
    else:
        coll_map = {"docs": "research_docs", "insights": "insights", "facts": "facts"}
        collections = []
        for c in args.collections.split(","):
            c = c.strip()
            collections.append(coll_map.get(c, c))

    domain_label = domain or "auto/cross"
    print(f"\n🔍 Route query: \"{args.query}\"")
    print(f"   domain={domain_label} | collections={', '.join(collections)}")
    if args.relation:
        print(f"   relation filter: {args.relation.upper()}")
    print()
    print("Embedding query...")

    vector = embed_query(args.query, embed_model)

    # Search collections in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for col in collections:
            if col == "facts":
                f = pool.submit(search_facts, client, vector, domain,
                                args.relation, args.top)
            else:
                f = pool.submit(search_collection, client, col, vector,
                                domain, args.top)
            futures[f] = col

        for future in as_completed(futures):
            all_results.extend(future.result())

    if not all_results:
        print("No results found.")
        return

    # Merge & sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    total = min(args.merged_top, len(all_results))

    print(f"\n{'─'*70}")
    print(f"  Merged results — top {total} of {len(all_results)} "
          f"(domain={domain_label})")
    print(f"{'─'*70}\n")

    for i, result in enumerate(all_results[:total], 1):
        render_result(i, result)


if __name__ == "__main__":
    main()
