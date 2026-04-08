#!/usr/bin/env python3
"""
cluster.py — Build a graph from KB entities+facts, run Leiden community
detection, write community_id back to entity payloads, and optionally
generate per-community summary insights.

Usage:
    python scripts/cluster.py --kb-root /path/to/kb
    python scripts/cluster.py --kb-root ~/kb --domain clinical --verbose
    python scripts/cluster.py --kb-root ~/kb --no-summaries
"""

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import date

import networkx as nx
import ollama as ollama_client
from graspologic.partition import leiden
from qdrant_client import QdrantClient
from qdrant_client.models import (Filter, FieldCondition, MatchValue,
                                   PointStruct, PayloadSchemaType)

from kb_root import add_kb_root_arg, resolve_kb_root, load_config, kb_root_from_cfg


# ── Confidence ranking (for deduplicating parallel edges) ────────────────────

_CONFIDENCE_RANK = {"established": 3, "probable": 2, "speculative": 1}


# ── Qdrant data fetching ────────────────────────────────────────────────────

def fetch_all_entities(qdrant: QdrantClient, entities_col: str,
                       domain: str | None = None) -> list[dict]:
    """Scroll all entities from Qdrant. Returns list of payload dicts."""
    filt = None
    if domain:
        filt = Filter(must=[FieldCondition(key="domain",
                                           match=MatchValue(value=domain))])
    entities = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=entities_col,
            scroll_filter=filt,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for r in results:
            entities.append(r.payload)
        if offset is None:
            break
    return entities


def fetch_all_facts(qdrant: QdrantClient, facts_col: str,
                    domain: str | None = None) -> list[dict]:
    """Scroll all non-superseded facts from Qdrant."""
    must = [FieldCondition(key="superseded_by",
                           match=MatchValue(value=""))]
    if domain:
        must.append(FieldCondition(key="domain",
                                    match=MatchValue(value=domain)))

    # First pass: facts with superseded_by == "" (explicitly empty)
    facts = _scroll_all(qdrant, facts_col, Filter(must=must))

    # Also include facts where superseded_by is absent (null / never set).
    # Qdrant treats missing fields as not matching MatchValue, so we fetch
    # all facts and filter client-side for those without superseded_by.
    all_facts = _scroll_all(qdrant, facts_col,
                            Filter(must=must[1:]) if domain else None)
    seen = {f["fact_id"] for f in facts}
    for f in all_facts:
        if f["fact_id"] not in seen and not f.get("superseded_by"):
            facts.append(f)

    return facts


def _scroll_all(qdrant: QdrantClient, collection: str,
                filt: Filter | None) -> list[dict]:
    results_all = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for r in results:
            results_all.append(r.payload)
        if offset is None:
            break
    return results_all


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph(entities: list[dict], facts: list[dict]) -> nx.Graph:
    """Build undirected NetworkX graph from entities (nodes) and facts (edges)."""
    G = nx.Graph()
    entity_ids = set()

    for e in entities:
        eid = e["entity_id"]
        entity_ids.add(eid)
        G.add_node(eid, name=e.get("name", ""),
                   type=e.get("type", ""),
                   summary=e.get("summary", ""),
                   domain=e.get("domain", ""))

    # Track best edge per (subject, object) pair by confidence
    best_edges: dict[tuple[str, str], dict] = {}
    for f in facts:
        sid = f.get("subject_entity_id")
        oid = f.get("object_entity_id")
        if not sid or not oid or sid not in entity_ids or oid not in entity_ids:
            continue
        if sid == oid:
            continue

        key = (min(sid, oid), max(sid, oid))  # undirected key
        rank = _CONFIDENCE_RANK.get(f.get("confidence", ""), 0)
        existing = best_edges.get(key)
        if existing is None or rank > existing["_rank"]:
            best_edges[key] = {
                "relation_type": f.get("relation_type", "RELATED_TO"),
                "fact_id": f.get("fact_id", ""),
                "confidence": f.get("confidence", ""),
                "fact": f.get("fact", ""),
                "subject_entity_id": sid,
                "object_entity_id": oid,
                "_rank": rank,
            }

    for (n1, n2), attrs in best_edges.items():
        del attrs["_rank"]
        G.add_edge(n1, n2, **attrs)

    return G


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_graph(G: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Run Leiden community detection. Returns {node_id: community_id}."""
    connected = [n for n in G.nodes() if G.degree(n) > 0]
    isolated = [n for n in G.nodes() if G.degree(n) == 0]

    assignments = {}

    if connected:
        subgraph = G.subgraph(connected)
        partitions = leiden(subgraph, resolution=resolution)
        for node_id, comm_id in partitions.items():
            assignments[node_id] = comm_id

    # Each isolated node gets its own community
    max_id = max(assignments.values(), default=-1)
    for node in isolated:
        max_id += 1
        assignments[node] = max_id

    return assignments


def split_oversized(G: nx.Graph, assignments: dict[str, int],
                    max_size: int, max_pct: float) -> dict[str, int]:
    """Re-cluster communities exceeding size thresholds."""
    total = len(G)
    threshold = max(max_size, int(total * max_pct))

    communities = defaultdict(list)
    for node, cid in assignments.items():
        communities[cid].append(node)

    new_assignments = dict(assignments)
    next_id = max(assignments.values(), default=0) + 1

    for cid, nodes in communities.items():
        if len(nodes) <= threshold:
            continue
        sub = G.subgraph(nodes)
        try:
            sub_partitions = leiden(sub, resolution=2.0)
            unique = set(sub_partitions.values())
            if len(unique) <= 1:
                continue
            for node, sub_cid in sub_partitions.items():
                new_assignments[node] = next_id + sub_cid
            next_id += len(unique)
        except Exception:
            continue

    return new_assignments


def cohesion_score(G: nx.Graph, node_ids: list[str]) -> float:
    """Ratio of actual to possible intra-community edges."""
    n = len(node_ids)
    if n <= 1:
        return 1.0
    sub = G.subgraph(node_ids)
    actual = sub.number_of_edges()
    possible = n * (n - 1) / 2
    return round(actual / possible, 2)


def stabilize_ids(assignments: dict[str, int]) -> dict[str, int]:
    """Renumber community IDs by descending community size (0 = largest)."""
    communities = defaultdict(list)
    for node, cid in assignments.items():
        communities[cid].append(node)

    sorted_comms = sorted(communities.values(), key=len, reverse=True)
    node_to_new = {}
    for new_id, nodes in enumerate(sorted_comms):
        for n in nodes:
            node_to_new[n] = new_id
    return node_to_new


# ── Write-back ───────────────────────────────────────────────────────────────

def write_community_ids(qdrant: QdrantClient, entities_col: str,
                        assignments: dict[str, int]) -> int:
    """Set community_id on each entity point in Qdrant."""
    # Ensure the payload index exists
    try:
        qdrant.create_payload_index(
            collection_name=entities_col,
            field_name="community_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass  # index already exists

    count = 0
    batch = []
    for entity_id, comm_id in assignments.items():
        batch.append((entity_id, comm_id))
        if len(batch) >= 50:
            _flush_batch(qdrant, entities_col, batch)
            count += len(batch)
            batch = []
    if batch:
        _flush_batch(qdrant, entities_col, batch)
        count += len(batch)

    return count


def _flush_batch(qdrant: QdrantClient, entities_col: str,
                 batch: list[tuple[str, int]]):
    for entity_id, comm_id in batch:
        qdrant.set_payload(
            collection_name=entities_col,
            payload={"community_id": comm_id},
            points=[entity_id],
        )


# ── Community summaries ──────────────────────────────────────────────────────

SUMMARY_SYSTEM = (
    "You summarise clusters of entities from a research knowledge base. "
    "Given a list of entities and their relationships, produce a concise "
    "2-3 sentence summary describing what this cluster represents — its "
    "theme, key concepts, and how the entities relate to each other. "
    "Respond with the summary text only, no labels or formatting."
)


def generate_community_summary(llm, community_id: int,
                                entities: list[dict],
                                facts: list[dict],
                                max_entities: int = 20) -> str:
    """Generate an LLM summary for a community."""
    ents = entities[:max_entities]
    ent_lines = [f"- {e['name']} ({e.get('type','?')}): {e.get('summary','')}"
                 for e in ents]
    fact_lines = [f"- {f.get('subject_name','?')} [{f.get('relation_type','?')}] "
                  f"{f.get('object_name','?')}: {f.get('fact','')}"
                  for f in facts[:30]]

    prompt = (
        f"Community {community_id} contains {len(entities)} entities.\n\n"
        f"ENTITIES:\n" + "\n".join(ent_lines) + "\n\n"
        f"RELATIONSHIPS:\n" + ("\n".join(fact_lines) if fact_lines else "(none)")
    )
    return llm.call_with_system(SUMMARY_SYSTEM, prompt)


def store_community_summaries(qdrant: QdrantClient, cfg: dict,
                              communities: dict[int, list[str]],
                              entity_map: dict[str, dict],
                              all_facts: list[dict],
                              llm) -> int:
    """Generate and store a summary insight per community."""
    embed_model = cfg["embedding"]["model"]
    max_ents = cfg.get("clustering", {}).get("summary_max_entities", 20)

    # Delete stale community summaries
    try:
        old, _ = qdrant.scroll(
            collection_name="insights",
            scroll_filter=Filter(must=[
                FieldCondition(key="insight_type",
                               match=MatchValue(value="community_summary"))
            ]),
            limit=500,
            with_payload=False,
        )
        if old:
            qdrant.delete(collection_name="insights",
                          points_selector=[p.id for p in old])
    except Exception:
        pass

    # Build fact lookup by entity_id
    facts_by_entity: dict[str, list[dict]] = defaultdict(list)
    for f in all_facts:
        facts_by_entity[f.get("subject_entity_id", "")].append(f)
        facts_by_entity[f.get("object_entity_id", "")].append(f)

    today = date.today().isoformat()
    count = 0

    for comm_id, node_ids in sorted(communities.items()):
        if len(node_ids) < 2:
            continue  # skip singleton communities

        ents = [entity_map[nid] for nid in node_ids if nid in entity_map]
        comm_facts = []
        seen_fids = set()
        for nid in node_ids:
            for f in facts_by_entity.get(nid, []):
                fid = f.get("fact_id")
                if fid and fid not in seen_fids:
                    # Only include facts where both endpoints are in this community
                    if (f.get("subject_entity_id") in set(node_ids) and
                            f.get("object_entity_id") in set(node_ids)):
                        comm_facts.append(f)
                        seen_fids.add(fid)

        print(f"  Community {comm_id} ({len(ents)} entities) — generating summary...")
        summary = generate_community_summary(llm, comm_id, ents, comm_facts,
                                              max_entities=max_ents)

        # Embed the summary
        vector = ollama_client.embed(model=embed_model,
                                      input=[summary])["embeddings"][0]

        entity_names = [e.get("name", "") for e in ents]
        payload = {
            "text": summary,
            "title": f"[community_summary] Cluster {comm_id}: {', '.join(entity_names[:3])}...",
            "authors": [],
            "domain": "cross",
            "subdomain": "",
            "source_type": "own-analysis",
            "source": f"cluster.py {today}",
            "confidence": "established",
            "tags": ["community", "cluster"],
            "project": "general",
            "date": today,
            "language": "en",
            "insight_type": "community_summary",
            "community_id": comm_id,
            "entity_ids": node_ids,
            "source_doc_ids": [],
            "_doc_id": str(uuid.uuid4()),
            "_chunk_index": 0,
            "_chunk_total": 1,
            "_added": today,
        }

        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        qdrant.upsert(collection_name="insights", points=[point])
        count += 1

    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Graph clustering — Leiden community detection on KB entities & facts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_kb_root_arg(parser)
    parser.add_argument("--domain", help="Only cluster entities/facts in this domain")
    parser.add_argument("--no-summaries", action="store_true",
                        help="Skip LLM community summary generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed community info")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    cluster_cfg = cfg.get("clustering", {})

    qdrant = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    entities_col = cfg.get("collections_extra", {}).get("entities", "entities")
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")

    # 1. Fetch data
    print("Fetching entities...")
    entities = fetch_all_entities(qdrant, entities_col, args.domain)
    print(f"  {len(entities)} entities")

    print("Fetching facts...")
    facts = fetch_all_facts(qdrant, facts_col, args.domain)
    print(f"  {len(facts)} active facts")

    if len(entities) < 2:
        print("Too few entities to cluster. Run entity_extract.py first.")
        sys.exit(0)

    # 2. Build graph
    print("\nBuilding graph...")
    G = build_graph(entities, facts)
    components = nx.number_connected_components(G)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{components} connected components")

    # 3. Cluster
    resolution = cluster_cfg.get("resolution", 1.0)
    print(f"\nRunning Leiden (resolution={resolution})...")
    assignments = cluster_graph(G, resolution=resolution)

    max_size = cluster_cfg.get("max_community_size", 10)
    max_pct = cluster_cfg.get("max_community_pct", 0.25)
    assignments = split_oversized(G, assignments, max_size, max_pct)
    assignments = stabilize_ids(assignments)

    # 4. Report
    communities: dict[int, list[str]] = defaultdict(list)
    for node, cid in assignments.items():
        communities[cid].append(node)

    entity_map = {e["entity_id"]: e for e in entities}
    print(f"\n{len(communities)} communities detected:\n")
    print(f"  {'ID':>4}  {'Size':>5}  {'Cohesion':>8}  Top entities")
    print(f"  {'─'*4}  {'─'*5}  {'─'*8}  {'─'*40}")
    for cid in sorted(communities):
        nodes = communities[cid]
        coh = cohesion_score(G, nodes)
        names = [entity_map[n]["name"] for n in nodes[:4] if n in entity_map]
        suffix = "…" if len(nodes) > 4 else ""
        print(f"  {cid:>4}  {len(nodes):>5}  {coh:>8.2f}  {', '.join(names)}{suffix}")

    # 5. Write community_id to Qdrant
    print(f"\nWriting community_id to {len(assignments)} entities...")
    updated = write_community_ids(qdrant, entities_col, assignments)
    print(f"  {updated} entities updated")

    # 6. Community summaries
    if not args.no_summaries and cluster_cfg.get("generate_summaries", True):
        from entity_extract import LLMClient
        print("\nGenerating community summaries...")
        llm = LLMClient(cfg)
        stored = store_community_summaries(qdrant, cfg, communities,
                                           entity_map, facts, llm)
        print(f"  {stored} summaries stored in [insights]")

    print("\nDone.")


if __name__ == "__main__":
    main()
