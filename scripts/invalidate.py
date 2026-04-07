#!/usr/bin/env python3
"""
invalidate.py — Phase 5: Edge Invalidation for the facts collection.

Scans for pairs of facts about the same entity pair and uses an LLM to detect
contradictions. When a contradiction is found and the newer fact temporally supersedes
the older one, the old fact is marked invalid (invalid_at set, superseded_by set).

Usage:
    python scripts/invalidate.py --audit --kb-root /path/to/kb
    python scripts/invalidate.py --doc-id <id> --kb-root ~/kb
    python scripts/invalidate.py --audit --kb-root ~/kb --dry-run

Called automatically by fact_extract.py after storing each new fact (lightweight mode:
only checks the new fact against existing facts for the same subject+object pair).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from kb_root import add_kb_root_arg, resolve_kb_root, load_config


# ─── LLM client ───────────────────────────────────────────────────────────────

def get_llm_client(cfg: dict):
    from entity_extract import LLMClient
    return LLMClient(cfg)


# ─── Prompt ───────────────────────────────────────────────────────────────────

CONTRADICTION_PROMPT = """\
<EXISTING_FACT>
{existing_fact}
(valid: {existing_valid} to {existing_invalid})
</EXISTING_FACT>
<NEW_FACT>
{new_fact}
(valid: {new_valid})
</NEW_FACT>

Do these two facts contradict each other about the same relationship between the same entities?

A contradiction means they cannot both be true at the same time. Examples:
- Two different measured values for the same property (e.g. different melting points) → contradiction
- An updated price or production rate → contradiction
- A revised composition or specification → contradiction
- One fact adds more detail to the other without conflicting → NOT a contradiction
- The same fact stated with different wording → NOT a contradiction

Return JSON only:
{{"is_contradiction": true/false, "reason": "<one sentence>"}}
"""


# ─── Qdrant helpers ───────────────────────────────────────────────────────────

def get_qdrant(cfg: dict) -> QdrantClient:
    return QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])


def fetch_facts_for_pair(qdrant: QdrantClient, facts_col: str,
                         subject_id: str, object_id: str) -> list[dict]:
    """Return all still-valid facts for a given subject+object entity pair."""
    results, _ = qdrant.scroll(
        collection_name=facts_col,
        scroll_filter=Filter(must=[
            FieldCondition(key="subject_entity_id", match=MatchValue(value=subject_id)),
            FieldCondition(key="object_entity_id", match=MatchValue(value=object_id)),
        ]),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    facts = []
    for r in results:
        p = dict(r.payload)
        p["_qdrant_id"] = str(r.id)
        facts.append(p)
    return facts


def fetch_all_facts(qdrant: QdrantClient, facts_col: str,
                    doc_id: Optional[str] = None) -> list[dict]:
    """Fetch all facts, optionally filtered by source doc."""
    offset = None
    facts = []
    while True:
        filt = None
        if doc_id:
            filt = Filter(must=[
                FieldCondition(key="source_doc_id", match=MatchValue(value=doc_id))
            ])
        results, offset = qdrant.scroll(
            collection_name=facts_col,
            scroll_filter=filt,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in results:
            p = dict(r.payload)
            p["_qdrant_id"] = str(r.id)
            facts.append(p)
        if offset is None:
            break
    return facts


def mark_invalidated(qdrant: QdrantClient, facts_col: str,
                     old_fact_id: str, new_fact_id: str,
                     new_valid_at: Optional[str], dry_run: bool):
    """Set invalid_at and superseded_by on old fact."""
    expiry = new_valid_at or date.today().isoformat()
    if dry_run:
        print(f"      [dry-run] would invalidate {old_fact_id[:8]}... "
              f"(invalid_at={expiry}, superseded_by={new_fact_id[:8]}...)")
        return
    qdrant.set_payload(
        collection_name=facts_col,
        payload={
            "invalid_at": expiry,
            "superseded_by": new_fact_id,
        },
        points=[old_fact_id],
    )


# ─── Temporal ordering ────────────────────────────────────────────────────────

def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _newer(fact_a: dict, fact_b: dict) -> Optional[dict]:
    """Return the fact with the later valid_at, or None if indeterminate."""
    da = _parse_date(fact_a.get("valid_at")) or _parse_date(fact_a.get("t_created"))
    db = _parse_date(fact_b.get("valid_at")) or _parse_date(fact_b.get("t_created"))
    if da is None and db is None:
        return None
    if da is None:
        return fact_b
    if db is None:
        return fact_a
    return fact_a if da >= db else fact_b


# ─── Core check ───────────────────────────────────────────────────────────────

def check_pair(llm, fact_a: dict, fact_b: dict) -> Optional[dict]:
    """
    Ask LLM whether fact_a and fact_b contradict each other.
    Returns {"older": <fact>, "newer": <fact>, "reason": str} or None.
    """
    newer = _newer(fact_a, fact_b)
    if newer is None:
        # Can't determine order — treat fact_b as newer (it was ingested later)
        newer = fact_b
    older = fact_b if newer is fact_a else fact_a

    prompt = CONTRADICTION_PROMPT.format(
        existing_fact=older.get("fact", ""),
        existing_valid=older.get("valid_at") or "unknown",
        existing_invalid=older.get("invalid_at") or "present",
        new_fact=newer.get("fact", ""),
        new_valid=newer.get("valid_at") or "unknown",
    )
    try:
        raw = llm.call(prompt)
        result = json.loads(raw)
        if result.get("is_contradiction"):
            return {
                "older": older,
                "newer": newer,
                "reason": result.get("reason", ""),
            }
    except (json.JSONDecodeError, Exception):
        pass
    return None


# ─── Main audit / targeted check ──────────────────────────────────────────────

def run_invalidation(cfg: dict, qdrant: QdrantClient, llm,
                     doc_id: Optional[str] = None,
                     dry_run: bool = False,
                     verbose: bool = False) -> int:
    """
    Full audit or doc-scoped invalidation pass.
    Returns count of invalidations performed (or would perform in dry-run).
    """
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")
    all_facts = fetch_all_facts(qdrant, facts_col, doc_id=doc_id)

    if not all_facts:
        print("No facts found.")
        return 0

    # Group by (subject_entity_id, object_entity_id) pair
    pair_map: dict[tuple, list[dict]] = defaultdict(list)
    for f in all_facts:
        key = (f.get("subject_entity_id", ""), f.get("object_entity_id", ""))
        if key[0] and key[1]:
            pair_map[key].append(f)

    invalidations = 0

    for (subj_id, obj_id), facts in pair_map.items():
        # Only check pairs with >1 fact
        if len(facts) < 2:
            continue

        # Skip facts already invalidated
        active = [f for f in facts if not f.get("invalid_at") and not f.get("superseded_by")]
        if len(active) < 2:
            continue

        if verbose:
            subj_name = active[0].get("subject_name", subj_id[:8])
            obj_name = active[0].get("object_name", obj_id[:8])
            print(f"  Checking pair: ({subj_name}) ↔ ({obj_name})  [{len(active)} facts]")

        # Compare all pairs within active facts
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                fa, fb = active[i], active[j]
                # Skip if either already marked during this run
                if fa.get("superseded_by") or fb.get("superseded_by"):
                    continue

                result = check_pair(llm, fa, fb)
                if result:
                    older = result["older"]
                    newer = result["newer"]
                    reason = result["reason"]
                    old_id = older["_qdrant_id"]
                    new_id = newer["_qdrant_id"]

                    print(f"    ⚠ Contradiction: {older.get('fact', '')[:80]}...")
                    print(f"      superseded by: {newer.get('fact', '')[:80]}...")
                    print(f"      reason: {reason}")

                    mark_invalidated(qdrant, facts_col, old_id, new_id,
                                     newer.get("valid_at"), dry_run)
                    # Mark in-memory so we don't double-process this run
                    older["superseded_by"] = new_id
                    invalidations += 1

    print(f"\n{'[dry-run] ' if dry_run else ''}Invalidation complete: "
          f"{invalidations} fact(s) {'would be ' if dry_run else ''}invalidated.")
    return invalidations


def run_targeted(cfg: dict, qdrant: QdrantClient, llm,
                 new_fact: dict, dry_run: bool = False) -> int:
    """
    Lightweight check: called by fact_extract after storing a new fact.
    Only scans existing facts for the same subject+object pair.
    Returns count of invalidations.
    """
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")
    subj_id = new_fact.get("subject_entity_id")
    obj_id = new_fact.get("object_entity_id")
    new_fact_id = new_fact.get("_qdrant_id") or new_fact.get("fact_id")

    if not subj_id or not obj_id or not new_fact_id:
        return 0

    existing = fetch_facts_for_pair(qdrant, facts_col, subj_id, obj_id)
    invalidations = 0
    for old in existing:
        old_id = old["_qdrant_id"]
        if old_id == new_fact_id:
            continue
        if old.get("invalid_at") or old.get("superseded_by"):
            continue

        result = check_pair(llm, old, new_fact)
        if result and result["older"]["_qdrant_id"] == old_id:
            print(f"    ⚠ Invalidating stale fact: {old.get('fact', '')[:80]}...")
            mark_invalidated(qdrant, facts_col, old_id, new_fact_id,
                             new_fact.get("valid_at"), dry_run)
            invalidations += 1

    return invalidations


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Edge invalidation for facts collection.")
    add_kb_root_arg(parser)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true",
                      help="Scan entire facts collection for contradictions")
    mode.add_argument("--doc-id", metavar="DOC_ID",
                      help="Check facts sourced from a specific document")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be invalidated without writing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-pair progress")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    qdrant = get_qdrant(cfg)
    llm = get_llm_client(cfg)

    run_invalidation(
        cfg=cfg,
        qdrant=qdrant,
        llm=llm,
        doc_id=args.doc_id if not args.audit else None,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
