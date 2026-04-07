#!/usr/bin/env python3
"""
fact_extract.py — Extract typed facts (relationships) between entities and store
in the `facts` Qdrant collection.

For each chunk, uses the entity list already extracted (from entity_extract.py)
plus an LLM to identify relationships between them, deduplicates against existing
facts, extracts temporal validity, and stores results.

Usage:
    python scripts/fact_extract.py --doc-id <doc_id>
    python scripts/fact_extract.py --doc-id <doc_id> --dry-run --verbose
    python scripts/fact_extract.py --all              # all docs with entities but no facts
    python scripts/fact_extract.py --list-facts [--subject "Tungsten Carbide"]
    python scripts/fact_extract.py --list-facts [--relation CONTAINS]

Called automatically by entity_extract.py after entity processing (unless --no-facts).
"""

import argparse
import json
import os
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import ollama as ollama_client
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (Filter, FieldCondition, MatchValue,
                                   PointStruct, SetPayload)

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


# ─── LLM client (reuse from entity_extract) ───────────────────────────────────

def get_llm_client(cfg: dict):
    sys.path.insert(0, str(Path(__file__).parent))
    from entity_extract import LLMClient
    return LLMClient(cfg)


def get_qdrant_client(cfg: dict) -> QdrantClient:
    return QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])


# ─── Prompts ──────────────────────────────────────────────────────────────────

from domain_config import get_relations_for_domain


# Domain-aware relation ontology — loaded from config/domain.yaml
def _ontology_for_domain(domain: str) -> list[str]:
    return get_relations_for_domain(domain)


FACT_EXTRACTION_PROMPT = """\
<CHUNK>
{chunk_text}
</CHUNK>
<ENTITIES>
{entities_json}
</ENTITIES>
<ALLOWED_RELATION_TYPES>
{relation_types}
</ALLOWED_RELATION_TYPES>

Extract all facts (relationships) between the listed ENTITIES that are stated or \
clearly implied in the CHUNK.

Guidelines:
1. Each fact must involve exactly TWO distinct entities from the list above.
2. relation_type: MUST be chosen from ALLOWED_RELATION_TYPES only. \
Pick the closest match. Use RELATED_TO only if truly nothing else fits.
3. fact: a complete descriptive sentence with all quantitative details present in the chunk.
4. confidence: "established" (stated as fact), "probable" (implied), "speculative" (inferred).
5. Only extract facts clearly supported by this chunk — do not invent or hallucinate.
6. If no clear inter-entity relationships exist in this chunk, return [].

Respond with JSON array only:
[
  {{
    "subject": "<entity name exactly as in ENTITIES>",
    "relation_type": "RELATION_TYPE",
    "object": "<entity name exactly as in ENTITIES>",
    "fact": "Full descriptive sentence.",
    "confidence": "established|probable|speculative"
  }}
]
"""

FACT_DEDUP_PROMPT = """\
<EXISTING_FACTS>
{existing_facts}
</EXISTING_FACTS>
<NEW_FACT>
{new_fact}
</NEW_FACT>

Does NEW_FACT express the same information as any fact in EXISTING_FACTS?

Guidelines:
1. Facts are duplicates if they assert the same relationship between the same entities, \
even with different wording.
2. A fact with additional quantitative detail is NOT a duplicate of a vaguer version — \
prefer keeping the more detailed one.
3. An updated value (e.g., newer price, revised measurement) IS a contradiction, not a duplicate.

Respond with JSON only:
{{"is_duplicate": true/false, "existing_fact_id": "<uuid or null>", "reason": "<brief>"}}
"""

TEMPORAL_EXTRACTION_PROMPT = """\
<CHUNK>
{chunk_text}
</CHUNK>
<REFERENCE_DATE>
{reference_date}
</REFERENCE_DATE>
<FACT>
{fact}
</FACT>

Extract temporal validity for this fact.
- valid_at: when did this relationship become true? ISO date or null.
- invalid_at: when did it stop being true? ISO date or null (null = still valid).

Rules:
1. For timeless scientific facts (physical properties, chemical reactions), return both null.
2. For economic/market data stated as current, use reference_date as valid_at.
3. For historical events, extract the year/date stated in the chunk.
4. For relative times ("3 years ago"), calculate from reference_date.
5. Do NOT infer dates from context — only use dates explicitly tied to this fact.

Respond with JSON only:
{{"valid_at": "YYYY-MM-DD or null", "invalid_at": "YYYY-MM-DD or null"}}
"""


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
    return json.loads(raw)


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_text(text: str, model: str) -> list[float]:
    resp = ollama_client.embeddings(model=model, prompt=text)
    return resp["embedding"]


# ─── Facts collection init ────────────────────────────────────────────────────

FACTS_FIELDS = {
    "fact_id":            "keyword",
    "subject_entity_id":  "keyword",
    "subject_name":       "keyword",
    "relation_type":      "keyword",
    "object_entity_id":   "keyword",
    "object_name":        "keyword",
    "confidence":         "keyword",
    "domain":             "keyword",
    "source_doc_id":      "keyword",
    "valid_at":           "keyword",
    "invalid_at":         "keyword",
    "superseded_by":      "keyword",
    "_added":             "keyword",
}


def ensure_facts_collection(qdrant: QdrantClient, facts_col: str, dims: int):
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
    existing = {c.name for c in qdrant.get_collections().collections}
    if facts_col in existing:
        return
    qdrant.create_collection(
        collection_name=facts_col,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )
    for field in FACTS_FIELDS:
        qdrant.create_payload_index(
            collection_name=facts_col,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print(f"  [created] {facts_col} collection with {len(FACTS_FIELDS)} indexes")


# ─── Qdrant helpers ───────────────────────────────────────────────────────────

def get_entity_id(qdrant: QdrantClient, entities_col: str, name: str) -> Optional[str]:
    """Look up entity_id by exact name match."""
    results, _ = qdrant.scroll(
        collection_name=entities_col,
        scroll_filter=Filter(must=[
            FieldCondition(key="name", match=MatchValue(value=name))
        ]),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if results:
        return results[0].payload.get("entity_id", str(results[0].id))
    return None


def get_entity_domain(qdrant: QdrantClient, entities_col: str, entity_id: str) -> str:
    results = qdrant.retrieve(
        collection_name=entities_col,
        ids=[entity_id],
        with_payload=True,
    )
    if results:
        return results[0].payload.get("domain", "cross")
    return "cross"


def search_similar_facts(qdrant: QdrantClient, facts_col: str,
                          subject_id: str, object_id: str,
                          embedding: list[float], top: int = 5) -> list[dict]:
    """
    Search for existing facts between the same entity pair.
    Constrained to same subject+object pair (per Graphiti section 2.2.2).
    """
    results = qdrant.query_points(
        collection_name=facts_col,
        query=embedding,
        query_filter=Filter(must=[
            FieldCondition(key="subject_entity_id", match=MatchValue(value=subject_id)),
            FieldCondition(key="object_entity_id", match=MatchValue(value=object_id)),
        ]),
        limit=top,
        with_payload=True,
        score_threshold=0.80,
    ).points
    return [dict(r.payload) | {"_score": r.score} for r in results]


def store_fact(qdrant: QdrantClient, facts_col: str, fact_id: str,
               subject_id: str, subject_name: str,
               relation_type: str,
               object_id: str, object_name: str,
               fact_text: str, confidence: str, domain: str,
               source_doc_id: str, valid_at: Optional[str], invalid_at: Optional[str],
               embedding: list[float], dry_run: bool = False) -> str:
    if dry_run:
        print(f"    [dry-run] would store: ({subject_name})-[{relation_type}]->({object_name})")
        return fact_id

    payload = {
        "fact_id":           fact_id,
        "subject_entity_id": subject_id,
        "subject_name":      subject_name,
        "relation_type":     relation_type,
        "object_entity_id":  object_id,
        "object_name":       object_name,
        "fact":              fact_text,
        "confidence":        confidence,
        "domain":            domain,
        "source_doc_id":     source_doc_id,
        "valid_at":          valid_at,
        "invalid_at":        invalid_at,
        "superseded_by":     None,
        "_added":            date.today().isoformat(),
    }
    qdrant.upsert(
        collection_name=facts_col,
        points=[PointStruct(id=fact_id, vector=embedding, payload=payload)],
    )
    return fact_id


def get_chunk_entities(qdrant: QdrantClient, research_col: str,
                        doc_id: str, chunk_index: int) -> list[str]:
    """Get entity_ids stored on a chunk's mentioned_entities payload field."""
    results, _ = qdrant.scroll(
        collection_name=research_col,
        scroll_filter=Filter(must=[
            FieldCondition(key="_doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="_chunk_index", match=MatchValue(value=chunk_index)),
        ]),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return []
    return results[0].payload.get("mentioned_entities", [])


def get_entity_names(qdrant: QdrantClient, entities_col: str,
                      entity_ids: list[str]) -> dict[str, str]:
    """Fetch {entity_id: name} for a list of entity_ids."""
    if not entity_ids:
        return {}
    results = qdrant.retrieve(
        collection_name=entities_col,
        ids=entity_ids,
        with_payload=True,
    )
    return {r.payload.get("entity_id", str(r.id)): r.payload.get("name", "?")
            for r in results}


# ─── LLM steps ────────────────────────────────────────────────────────────────

def extract_facts_from_chunk(llm, chunk_text: str, entities: dict[str, str],
                             domain: str = "cross") -> list[dict]:
    """
    entities: {entity_id: name}
    domain: used to inject domain-specific relation ontology into the prompt
    Returns list of {subject, relation_type, object, fact, confidence}
    """
    if len(entities) < 2:
        return []  # need at least 2 entities to have a relationship

    entities_json = json.dumps(
        [{"name": name} for name in entities.values()],
        ensure_ascii=False
    )
    allowed_types = _ontology_for_domain(domain)
    relation_types = "\n".join(f"  {r}" for r in allowed_types)
    prompt = FACT_EXTRACTION_PROMPT.format(
        chunk_text=chunk_text,
        entities_json=entities_json,
        relation_types=relation_types,
    )
    raw = llm.call(prompt)
    try:
        result = parse_json(raw)
        if not isinstance(result, list):
            return []
        valid = []
        entity_names_lower = {n.lower(): n for n in entities.values()}
        for f in result:
            if not all(k in f for k in ("subject", "relation_type", "object", "fact")):
                continue
            # Normalise subject/object to exact entity names (case-insensitive match)
            subj = f["subject"]
            obj = f["object"]
            subj_norm = entity_names_lower.get(subj.lower(), subj)
            obj_norm = entity_names_lower.get(obj.lower(), obj)
            if subj_norm == obj_norm:
                continue  # self-referential, skip
            f["subject"] = subj_norm
            f["object"] = obj_norm
            f["confidence"] = f.get("confidence", "probable")
            rel = f["relation_type"].upper().replace(" ", "_")
            # Validate against ontology; unknown types fall back to RELATED_TO
            allowed_set = {r.upper() for r in _ontology_for_domain(domain)}
            f["relation_type"] = rel if rel in allowed_set else "RELATED_TO"
            valid.append(f)
        return valid
    except (json.JSONDecodeError, ValueError):
        return []


def dedup_fact(llm, new_fact_text: str, candidates: list[dict]) -> dict:
    """Returns {is_duplicate, existing_fact_id, reason}"""
    if not candidates:
        return {"is_duplicate": False, "existing_fact_id": None, "reason": "no candidates"}

    existing = json.dumps([
        {"fact_id": c.get("fact_id"), "fact": c.get("fact", "")}
        for c in candidates
    ], ensure_ascii=False)
    prompt = FACT_DEDUP_PROMPT.format(
        existing_facts=existing,
        new_fact=new_fact_text,
    )
    raw = llm.call(prompt)
    try:
        result = parse_json(raw)
        return {
            "is_duplicate": bool(result.get("is_duplicate", False)),
            "existing_fact_id": result.get("existing_fact_id"),
            "reason": result.get("reason", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"is_duplicate": False, "existing_fact_id": None, "reason": "parse error"}


def extract_temporal(llm, chunk_text: str, fact_text: str,
                      reference_date: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (valid_at, invalid_at) as ISO strings or None."""
    prompt = TEMPORAL_EXTRACTION_PROMPT.format(
        chunk_text=chunk_text[:1000],  # truncate for cost
        reference_date=reference_date,
        fact=fact_text,
    )
    raw = llm.call(prompt)
    try:
        result = parse_json(raw)
        valid_at = result.get("valid_at") or None
        invalid_at = result.get("invalid_at") or None
        # Validate ISO format
        import re
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if valid_at and not iso_re.match(valid_at):
            valid_at = None
        if invalid_at and not iso_re.match(invalid_at):
            invalid_at = None
        return valid_at, invalid_at
    except (json.JSONDecodeError, ValueError):
        return None, None


# ─── Core pipeline ────────────────────────────────────────────────────────────

def process_doc(doc_id: str, cfg: dict, llm, qdrant: QdrantClient,
                dry_run: bool = False, verbose: bool = False) -> int:
    """
    Extract facts from all chunks of doc_id.
    Returns count of facts stored/skipped.
    """
    embed_model = cfg["embedding"]["model"]
    research_col = cfg["collections"]["research_docs"]
    entities_col = cfg.get("collections_extra", {}).get("entities", "entities")
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")
    reference_date = date.today().isoformat()

    # Fetch all chunks sorted by index
    chunks = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=research_col,
            scroll_filter=Filter(must=[
                FieldCondition(key="_doc_id", match=MatchValue(value=doc_id)),
            ]),
            limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        chunks.extend(results)
        if offset is None:
            break

    if not chunks:
        print(f"  [warn] no chunks found for doc_id {doc_id[:8]}...")
        return 0

    chunks.sort(key=lambda r: r.payload.get("_chunk_index", 0))
    doc_title = chunks[0].payload.get("title", doc_id[:8])
    doc_domain = chunks[0].payload.get("domain", "cross")
    total = len(chunks)

    print(f"\n→ Fact extraction: {doc_title} ({total} chunks)")

    stored = 0
    skipped_dedup = 0
    skipped_no_entities = 0

    for i, chunk_result in enumerate(chunks):
        chunk_text = chunk_result.payload.get("text", "")
        chunk_index = chunk_result.payload.get("_chunk_index", i)
        entity_ids = chunk_result.payload.get("mentioned_entities", [])

        if len(entity_ids) < 2:
            skipped_no_entities += 1
            continue

        # Fetch entity names for this chunk
        entity_map = get_entity_names(qdrant, entities_col, entity_ids)
        if len(entity_map) < 2:
            skipped_no_entities += 1
            continue

        if verbose:
            print(f"  chunk {chunk_index+1}/{total} ({len(entity_map)} entities)...",
                  end=" ", flush=True)

        # LLM fact extraction (domain-aware ontology)
        raw_facts = extract_facts_from_chunk(llm, chunk_text, entity_map, domain=doc_domain)

        if not raw_facts:
            if verbose:
                print("(no facts)")
            continue

        if verbose:
            print(f"({len(raw_facts)} candidates)")

        for raw in raw_facts:
            subj_name = raw["subject"]
            obj_name = raw["object"]
            relation = raw["relation_type"]
            fact_text = raw["fact"]
            confidence = raw["confidence"]

            # Resolve entity IDs by name
            subj_id = get_entity_id(qdrant, entities_col, subj_name)
            obj_id = get_entity_id(qdrant, entities_col, obj_name)

            if not subj_id or not obj_id:
                if verbose:
                    print(f"    [skip] entity not found: {subj_name!r} or {obj_name!r}")
                continue

            # Embed the fact
            embed_str = f"{subj_name} {relation} {obj_name}: {fact_text}"
            embedding = embed_text(embed_str, embed_model)

            # Dedup: search existing facts between same entity pair
            candidates = search_similar_facts(qdrant, facts_col, subj_id, obj_id, embedding)
            if candidates:
                dedup = dedup_fact(llm, fact_text, candidates)
                if dedup["is_duplicate"]:
                    skipped_dedup += 1
                    if verbose:
                        print(f"    [dedup] ({subj_name})-[{relation}]->({obj_name})")
                    continue

            # Temporal extraction
            valid_at, invalid_at = extract_temporal(llm, chunk_text, fact_text, reference_date)

            # Determine domain from entities
            domain = get_entity_domain(qdrant, entities_col, subj_id) or doc_domain

            # Store
            fact_id = str(uuid.uuid4())
            store_fact(
                qdrant, facts_col, fact_id,
                subj_id, subj_name, relation, obj_id, obj_name,
                fact_text, confidence, domain, doc_id,
                valid_at, invalid_at, embedding, dry_run,
            )
            stored += 1

            if verbose:
                temporal_str = f" [{valid_at}→{invalid_at or 'now'}]" if valid_at else ""
                print(f"    + ({subj_name})-[{relation}]->({obj_name}){temporal_str}")

            # Phase 5 — Edge Invalidation: check new fact against existing ones for same pair
            if not dry_run:
                try:
                    from invalidate import run_targeted
                    new_fact_payload = {
                        "_qdrant_id": fact_id,
                        "subject_entity_id": subj_id,
                        "object_entity_id": obj_id,
                        "fact": fact_text,
                        "valid_at": valid_at,
                    }
                    n_inv = run_targeted(cfg, qdrant, llm, new_fact_payload, dry_run=False)
                    if n_inv and verbose:
                        print(f"    ↳ invalidated {n_inv} stale fact(s)")
                except Exception as e:
                    if verbose:
                        print(f"    [warn] invalidation check failed: {e}")

    print(f"  ✓ {stored} facts stored, {skipped_dedup} deduped, "
          f"{skipped_no_entities} chunks skipped (< 2 entities)")
    return stored


def get_docs_with_entities_no_facts(qdrant: QdrantClient, cfg: dict) -> list[dict]:
    """Return docs that have mentioned_entities but no facts stored yet."""
    research_col = cfg["collections"]["research_docs"]
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")

    # Collect all doc_ids that have at least one chunk with entities
    docs_with_entities: dict[str, str] = {}
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=research_col,
            limit=200, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for r in results:
            doc_id = r.payload.get("_doc_id")
            if doc_id and r.payload.get("mentioned_entities"):
                docs_with_entities[doc_id] = r.payload.get("title", doc_id[:8])
        if offset is None:
            break

    # Collect doc_ids that already have facts
    docs_with_facts: set[str] = set()
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=facts_col,
            limit=200, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for r in results:
            src = r.payload.get("source_doc_id")
            if src:
                docs_with_facts.add(src)
        if offset is None:
            break

    return [{"doc_id": did, "title": title}
            for did, title in docs_with_entities.items()
            if did not in docs_with_facts]


# ─── List / display ───────────────────────────────────────────────────────────

def list_facts(qdrant: QdrantClient, cfg: dict,
               subject: Optional[str] = None,
               relation: Optional[str] = None,
               domain: Optional[str] = None,
               limit: int = 50):
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")

    must = []
    if subject:
        must.append(FieldCondition(key="subject_name", match=MatchValue(value=subject)))
    if relation:
        must.append(FieldCondition(key="relation_type", match=MatchValue(value=relation.upper())))
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    filt = Filter(must=must) if must else None

    results, _ = qdrant.scroll(
        collection_name=facts_col,
        scroll_filter=filt,
        limit=limit,
        with_payload=True, with_vectors=False,
    )

    total, _ = qdrant.scroll(
        collection_name=facts_col,
        scroll_filter=filt,
        limit=1, with_payload=False, with_vectors=False,
    )

    if not results:
        print("  [facts] is empty (or no matches).")
        return

    print(f"\n  [facts] — showing {len(results)} facts\n")

    # Group by relation_type
    by_rel: dict[str, list] = {}
    for r in results:
        rel = r.payload.get("relation_type", "UNKNOWN")
        by_rel.setdefault(rel, []).append(r.payload)

    for rel, facts in sorted(by_rel.items()):
        print(f"  {rel} ({len(facts)})")
        for f in facts:
            subj = f.get("subject_name", "?")
            obj = f.get("object_name", "?")
            conf = f.get("confidence", "?")
            valid_at = f.get("valid_at")
            temporal = f" [{valid_at}]" if valid_at else ""
            superseded = " ⚠SUPERSEDED" if f.get("superseded_by") else ""
            print(f"    • ({subj}) → ({obj})  [{conf}]{temporal}{superseded}")
            print(f"      {f.get('fact','')[:140]}")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract typed facts between entities and store in the facts collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fact_extract.py --doc-id 1494731e-xxxx --verbose
  python scripts/fact_extract.py --all
  python scripts/fact_extract.py --list-facts
  python scripts/fact_extract.py --list-facts --subject "Tungsten Carbide"
  python scripts/fact_extract.py --list-facts --relation CONTAINS
        """,
    )
    parser.add_argument("--doc-id", dest="doc_id", help="Process a specific doc_id")
    parser.add_argument("--all", action="store_true",
                        help="Process all docs with entities but without facts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract but don't write to Qdrant")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-chunk and per-fact progress")
    parser.add_argument("--list-facts", action="store_true",
                        help="List stored facts and exit")
    parser.add_argument("--subject", help="Filter --list-facts by subject entity name")
    parser.add_argument("--relation", help="Filter --list-facts by relation_type")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max facts to show with --list-facts (default: 50)")
    args = parser.parse_args()

    cfg = load_config()
    qdrant = get_qdrant_client(cfg)
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")

    # Ensure facts collection exists
    ensure_facts_collection(qdrant, facts_col, cfg["embedding"]["dimensions"])

    if args.list_facts:
        list_facts(qdrant, cfg, subject=args.subject,
                   relation=args.relation, domain=args.domain, limit=args.limit)
        return

    llm = get_llm_client(cfg)

    if args.doc_id:
        process_doc(args.doc_id, cfg, llm, qdrant,
                    dry_run=args.dry_run, verbose=args.verbose)

    elif args.all:
        pending = get_docs_with_entities_no_facts(qdrant, cfg)
        if not pending:
            print("All docs already have facts extracted.")
            return
        print(f"Found {len(pending)} docs with entities but no facts.")
        total_facts = 0
        for doc in pending:
            n = process_doc(doc["doc_id"], cfg, llm, qdrant,
                            dry_run=args.dry_run, verbose=args.verbose)
            total_facts += n
        print(f"\n✓ Done. {total_facts} total facts stored across {len(pending)} docs.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
