#!/usr/bin/env python3
"""
query.py — Hybrid search (cosine + BM25 + re-ranking) over the Qdrant knowledge base.

Usage:
    python query.py "your search query"
    python query.py "exact term" --bm25                   # keyword-only
    python query.py "data query" --collection data_tables
    python query.py "topic" --domain mydomain --project myproject --top 10
    python query.py "query" --lang ru --confidence established
    python query.py "" --list-docs                        # list indexed documents
    python query.py "topic" --as-of 2024-01-01            # temporal filter
    python query.py "topic" --no-rerank                   # fast, skip re-ranking

Default mode: hybrid (cosine + BM25 via RRF) → cross-encoder re-ranking.
Re-ranking uses BAAI/bge-reranker-v2-m3 (multilingual RU+EN, local via sentence-transformers).
--no-rerank: skip re-ranking, return RRF results directly (faster).
--bm25: keyword-only (good for exact terms, compound names, element symbols).
--no-hybrid: cosine-only (original behaviour).

Filters stack (AND logic): all specified filters must match.

--as-of: returns only chunks valid on that date:
    valid_at IS NULL  (timeless)  → always included
    valid_at <= date AND (invalid_at IS NULL OR invalid_at > date)  → included
Without --as-of: all chunks returned, superseded ones tagged [SUPERSEDED].
"""

import argparse
import yaml
from datetime import date as date_type
from pathlib import Path
from functools import lru_cache

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import (Filter, FieldCondition, MatchValue, MatchAny,
                                   IsNullCondition, IsEmptyCondition,
                                   Range, DatetimeRange, MatchText)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def embed_query(text: str, model: str) -> list[float]:
    return ollama_client.embed(model=model, input=[text])["embeddings"][0]

def parse_as_of(value: str) -> str | None:
    """Parse --as-of date, same shorthand as ingest.py."""
    if not value:
        return None
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
    raise ValueError(f"Unrecognised --as-of date format: '{value}'")


def build_filter(domain=None, subdomain=None, source_type=None,
                 confidence=None, language=None, project=None, tags=None,
                 as_of: str | None = None) -> Filter | None:
    """
    Build Qdrant filter from search parameters.

    as_of: ISO date string. If provided, adds temporal validity filter:
      - Includes chunks where valid_at IS NULL (timeless)
      - OR valid_at <= as_of AND (invalid_at IS NULL OR invalid_at > as_of)
    This is implemented as a should (OR) condition between the two cases.
    """
    must_conditions = []

    if domain:
        must_conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if subdomain:
        must_conditions.append(FieldCondition(key="subdomain", match=MatchValue(value=subdomain)))
    if source_type:
        must_conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if confidence:
        must_conditions.append(FieldCondition(key="confidence", match=MatchValue(value=confidence)))
    if language:
        must_conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
    if project:
        must_conditions.append(FieldCondition(key="project", match=MatchValue(value=project)))
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        must_conditions.append(FieldCondition(key="tags", match=MatchAny(any=tag_list)))

    if as_of:
        # Case A: valid_at is null (timeless content — always include)
        timeless_condition = Filter(
            must=[IsNullCondition(is_null={"key": "valid_at"})]
        )

        # Case B: valid_at <= as_of AND (invalid_at is null OR invalid_at > as_of)
        # Qdrant KEYWORD index stores dates as strings; lexicographic comparison works for ISO dates
        # We use MatchValue with range hack: store as keyword, filter in Python post-fetch
        # NOTE: Qdrant range filters work on float/int. For date strings we filter post-query.
        # So: include everything that passes must_conditions, then filter in Python by as_of.
        # This is the pragmatic approach until we switch to datetime payload type.
        # The as_of filter is applied as a post-filter in the search() function.
        pass  # post-filter applied in search()

    if not must_conditions:
        return None
    return Filter(must=must_conditions)


def _passes_temporal_filter(payload: dict, as_of: str | None) -> tuple[bool, bool]:
    """
    Returns (include, is_superseded) for a chunk given temporal filter.
    include: True if chunk should appear in results
    is_superseded: True if chunk is currently superseded (for display tagging)
    """
    if as_of is None:
        # No temporal filter: include everything
        is_superseded = bool(payload.get("superseded_by") or payload.get("invalid_at"))
        return True, is_superseded

    valid_at = payload.get("valid_at")       # str or None
    invalid_at = payload.get("invalid_at")   # str or None

    # Timeless content (valid_at is None) — always include
    if valid_at is None:
        return True, False

    # valid_at must be <= as_of
    if valid_at > as_of:
        return False, False

    # invalid_at must be None (still valid) or > as_of
    if invalid_at is not None and invalid_at <= as_of:
        return False, True   # was already expired before as_of

    return True, False

def list_docs(client: QdrantClient, collection: str):
    """Print a summary of indexed documents."""
    seen = {}
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in results:
            doc_id = r.payload.get("_doc_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "title": r.payload.get("title", "—"),
                    "domain": r.payload.get("domain", "—"),
                    "source_type": r.payload.get("source_type", "—"),
                    "date": r.payload.get("date", "—"),
                    "chunks": r.payload.get("_chunk_total", "?"),
                    "added": r.payload.get("_added", "—"),
                    # Phase 1 temporal
                    "valid_at": r.payload.get("valid_at"),
                    "invalid_at": r.payload.get("invalid_at"),
                    "superseded_by": r.payload.get("superseded_by"),
                }
        if offset is None:
            break

    if not seen:
        print(f"  [{collection}] is empty.")
        return

    print(f"\n  [{collection}] — {len(seen)} documents:\n")
    for i, (doc_id, info) in enumerate(seen.items(), 1):
        superseded = bool(info.get("superseded_by") or info.get("invalid_at"))
        tag = " [SUPERSEDED]" if superseded else ""
        print(f"  {i:3}. {info['title']}{tag}")
        # Temporal line
        temporal_parts = []
        if info["valid_at"]:
            temporal_parts.append(f"valid from {info['valid_at']}")
        if info["invalid_at"]:
            temporal_parts.append(f"until {info['invalid_at']}")
        elif info["valid_at"]:
            temporal_parts.append("current")
        temporal_str = f" | {', '.join(temporal_parts)}" if temporal_parts else ""
        print(f"       {info['domain']} | {info['source_type']} | {info['date']} | "
              f"{info['chunks']} chunks | added {info['added']}{temporal_str}")
        print(f"       doc_id: {doc_id[:8]}...")
        if info.get("superseded_by"):
            print(f"       superseded_by: {info['superseded_by'][:8]}...")
        print()

def bm25_search(query: str, collection: str, top: int,
                filt: Filter | None, cfg: dict) -> list:
    """
    Full-text (BM25) search on the `text` payload field.
    Returns Qdrant ScoredPoint list with synthetic scores based on rank.
    """
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])

    # Build text filter condition and merge with existing filter
    text_condition = FieldCondition(key="text", match=MatchText(text=query))
    if filt and filt.must:
        combined = Filter(must=list(filt.must) + [text_condition])
    else:
        combined = Filter(must=[text_condition])

    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=combined,
        limit=top,
        with_payload=True,
        with_vectors=False,
    )
    # Assign synthetic descending scores (scroll has no score; rank-order only)
    class _FakePoint:
        def __init__(self, r, rank, total):
            self.id = r.id
            self.payload = r.payload
            self.score = (total - rank) / total  # synthetic 0-1 score

    return [_FakePoint(r, i, len(results)) for i, r in enumerate(results)]


def reciprocal_rank_fusion(cosine_results: list, bm25_results: list,
                            top: int, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank_i)) for each list the doc appears in.
    k=60 is the standard constant (Robertson & Zaragoza, 2009).
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    ids_order: list[str] = []

    for rank, r in enumerate(cosine_results, 1):
        rid = str(r.id)
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
        payloads[rid] = r.payload
        if rid not in ids_order:
            ids_order.append(rid)

    for rank, r in enumerate(bm25_results, 1):
        rid = str(r.id)
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
        if rid not in payloads:
            payloads[rid] = r.payload
        if rid not in ids_order:
            ids_order.append(rid)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    class _MergedPoint:
        def __init__(self, rid):
            self.id = rid
            self.score = scores[rid]
            self.payload = payloads[rid]

    return [_MergedPoint(rid) for rid in sorted_ids[:top]]


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    """Load and cache the CrossEncoder model (loaded once per process)."""
    try:
        from sentence_transformers import CrossEncoder
        print(f"  Loading re-ranker model ({model_name})…", flush=True)
        model = CrossEncoder(model_name, max_length=512)
        print(f"  Re-ranker loaded.", flush=True)
        return model
    except Exception as e:
        print(f"  ⚠ Re-ranker unavailable ({e}) — falling back to RRF order.")
        return None


def rerank_results(query: str, results: list, top: int, cfg: dict) -> tuple[list, bool]:
    """
    Cross-encoder re-ranking using BAAI/bge-reranker-v2-m3.
    Scores each (query, chunk_text) pair; returns (reranked_list, did_rerank).
    Falls back to RRF order if model unavailable.
    """
    reranker_cfg = cfg.get("reranker", {})
    model_name = reranker_cfg.get("model", "BAAI/bge-reranker-v2-m3")
    max_chars = reranker_cfg.get("max_chunk_chars", 512)

    model = _load_reranker(model_name)
    if model is None or not results:
        return results[:top], False

    try:
        pairs = [(query, r.payload.get("text", "")[:max_chars]) for r in results]
        scores = model.predict(pairs)  # returns ndarray of floats

        scored = sorted(zip(scores, results), key=lambda x: float(x[0]), reverse=True)

        # Attach rerank score to each result for display
        for score, r in scored:
            r._rerank_score = float(score)

        return [r for _, r in scored[:top]], True

    except Exception as e:
        print(f"  ⚠ Re-ranking failed ({e}) — falling back to RRF order.")
        return results[:top], False


def search(query: str, collection: str, top: int, filt: Filter | None, cfg: dict,
           as_of: str | None = None, show_text: bool = True,
           mode: str = "hybrid", rerank: bool = True):
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]

    # When re-ranking, fetch a larger candidate pool; also expand for temporal filtering
    rerank_enabled = rerank and cfg.get("reranker", {}).get("enabled", True)
    candidate_mult = cfg.get("reranker", {}).get("candidate_multiplier", 3) if rerank_enabled else 2
    fetch_limit = top * candidate_mult * (2 if as_of else 1)

    if mode == "bm25":
        print(f"\nBM25 search...")
        results = bm25_search(query, collection, fetch_limit, filt, cfg)
    elif mode == "hybrid":
        print(f"\nHybrid search (cosine + BM25)...")
        vector = embed_query(query, embed_model)
        cosine_results = client.query_points(
            collection_name=collection,
            query=vector,
            limit=fetch_limit,
            query_filter=filt,
            with_payload=True,
        ).points
        bm25_results = bm25_search(query, collection, fetch_limit, filt, cfg)
        results = reciprocal_rank_fusion(cosine_results, bm25_results, top=fetch_limit)
    else:
        # cosine-only
        print(f"\nEmbedding query...")
        vector = embed_query(query, embed_model)
        results = client.query_points(
            collection_name=collection,
            query=vector,
            limit=fetch_limit,
            query_filter=filt,
            with_payload=True,
        ).points

    if not results:
        print("No results found.")
        return

    # Phase 1 — apply temporal post-filter
    if as_of:
        filtered = []
        for r in results:
            include, _ = _passes_temporal_filter(r.payload, as_of)
            if include:
                filtered.append(r)
        results = filtered

    if not results:
        print(f"No results valid as of {as_of}.")
        return

    # Re-ranking: score candidates with cross-encoder, keep top N
    did_rerank = False
    if rerank_enabled and mode != "bm25":
        print(f"Re-ranking {min(len(results), top * candidate_mult)} candidates…")
        results, did_rerank = rerank_results(query, results, top, cfg)
    else:
        results = results[:top]

    as_of_label = f" | As-of: {as_of}" if as_of else ""
    mode_label = {"hybrid": "hybrid", "bm25": "BM25", "cosine": "cosine"}.get(mode, mode)
    rerank_label = " + rerank" if did_rerank else (" (--no-rerank)" if rerank_enabled is False else "")
    print(f"\n{'─'*70}")
    print(f"  Query: \"{query}\"  |  {collection}  |  {mode_label}{rerank_label}  |  Top {top}{as_of_label}")
    print(f"{'─'*70}\n")

    for i, r in enumerate(results, 1):
        p = r.payload
        _, is_superseded = _passes_temporal_filter(p, None)  # for display tag
        superseded_tag = "  ⚠ SUPERSEDED" if is_superseded else ""

        # Temporal validity display
        temporal_parts = []
        if p.get("valid_at"):
            temporal_parts.append(f"from {p['valid_at']}")
        if p.get("invalid_at"):
            temporal_parts.append(f"until {p['invalid_at']}")
        elif p.get("valid_at"):
            temporal_parts.append("current")
        temporal_str = f"  [{', '.join(temporal_parts)}]" if temporal_parts else ""

        # Score display: show rerank score if available, RRF score as secondary
        rerank_score = getattr(r, "_rerank_score", None)
        if rerank_score is not None:
            score_str = f"Rerank: {rerank_score:.4f}  RRF: {r.score:.4f}"
        else:
            score_str = f"Score: {r.score:.4f}"

        print(f"[{i}] {score_str}{superseded_tag}")
        print(f"    Title:   {p.get('title', '—')}")
        print(f"    Domain:  {p.get('domain', '—')} / {p.get('subdomain', '—')}")
        print(f"    Source:  {p.get('source_type', '—')} | {p.get('source', '—')}")
        print(f"    Conf:    {p.get('confidence', '—')} | Lang: {p.get('language', '—')}")
        print(f"    Tags:    {', '.join(p.get('tags', []))}")
        print(f"    Chunk:   {p.get('_chunk_index', '?')+1}/{p.get('_chunk_total', '?')}"
              f"{temporal_str}")
        if show_text:
            print()
            text = p.get("text", "")
            preview = text[:400].replace("\n", " ").strip()
            if len(text) > 400:
                preview += "…"
            print(f"    {preview}")
        print(f"\n{'─'*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Query the Qdrant knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
--as-of date formats: YYYY, YYYY-QN, YYYY-MM, YYYY-MM-DD
  Example: --as-of 2024-Q3  → only content valid on 2024-07-01
  Timeless content (valid_at=null) is always included.
        """,
    )
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("--collection", default="research_docs",
                        choices=["research_docs", "data_tables", "insights"])
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--subdomain", help="Filter by subdomain")
    parser.add_argument("--source-type", dest="source_type", help="Filter by source_type")
    parser.add_argument("--confidence", help="Filter by confidence level")
    parser.add_argument("--lang", help="Filter by language (en/ru)")
    parser.add_argument("--project", help="Filter by project name")
    parser.add_argument("--tags", help="Filter by tags (comma-separated, OR logic)")
    parser.add_argument("--list-docs", action="store_true", help="List indexed documents")
    parser.add_argument("--show-text", action="store_true", default=True,
                        help="Show chunk text preview (default: on)")
    parser.add_argument("--no-text", dest="show_text", action="store_false",
                        help="Hide chunk text preview")
    # Phase 4 search mode
    search_mode = parser.add_mutually_exclusive_group()
    search_mode.add_argument("--bm25", action="store_true",
                              help="Keyword-only search (exact terms, no embeddings)")
    search_mode.add_argument("--no-hybrid", action="store_true",
                              help="Cosine-only search (original behaviour)")
    # Re-ranking
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder re-ranking (faster, returns RRF order)")
    # Phase 1 temporal filter
    parser.add_argument("--as-of", dest="as_of", default=None, metavar="DATE",
                        help="Only return content valid on this date (YYYY, YYYY-QN, YYYY-MM, YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = load_config()
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])

    # Parse --as-of date
    as_of = None
    if args.as_of:
        try:
            as_of = parse_as_of(args.as_of)
        except ValueError as e:
            print(f"Error: {e}")
            import sys; sys.exit(1)

    if args.list_docs:
        collections = ["research_docs", "data_tables", "insights"] \
            if args.collection == "research_docs" and not args.query \
            else [args.collection]
        for col in collections:
            list_docs(client, col)
        return

    if not args.query:
        parser.print_help()
        return

    filt = build_filter(
        domain=args.domain,
        subdomain=args.subdomain,
        source_type=args.source_type,
        confidence=args.confidence,
        language=args.lang,
        project=args.project,
        tags=args.tags,
        as_of=as_of,
    )

    mode = "bm25" if args.bm25 else ("cosine" if args.no_hybrid else "hybrid")
    rerank = not args.no_rerank

    search(args.query, args.collection, args.top, filt, cfg,
           as_of=as_of, show_text=args.show_text, mode=mode, rerank=rerank)

if __name__ == "__main__":
    main()
