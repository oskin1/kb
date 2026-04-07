#!/usr/bin/env python3
"""
entity_extract.py — Extract named entities from a document and store in the `entities` collection.

For each chunk window, uses an LLM to extract entities (materials, processes, properties, etc.),
resolves duplicates against existing entities via embedding + LLM check, then upserts to Qdrant.

Usage:
    python scripts/entity_extract.py --doc-id <doc_id> --kb-root /path/to/kb
    python scripts/entity_extract.py --doc-id <doc_id> --kb-root ~/kb --dry-run
    python scripts/entity_extract.py --all --kb-root ~/kb
    python scripts/entity_extract.py --list-entities --kb-root ~/kb

Called automatically by ingest.py after chunking (unless --no-entities flag is passed).

Output: list of entity_ids stored/updated in the `entities` Qdrant collection.
        Also updates each chunk payload in research_docs with `mentioned_entities: [uuid, ...]`.
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
from qdrant_client import QdrantClient
from qdrant_client.models import (Filter, FieldCondition, MatchValue, PointStruct,
                                   MatchAny, SetPayload)

from kb_root import add_kb_root_arg, resolve_kb_root, load_config, kb_root_from_cfg
from domain_config import get_entity_types, get_prompts

# ─── Prompts (loaded from config/domain.yaml) ────────────────────────────────


def _get_extraction_prompt(kb_root: Path) -> str:
    entity_types = get_entity_types(kb_root)
    prompts = get_prompts(kb_root)
    custom = prompts.get("entity_extraction_user")
    if custom:
        # Inject entity types list into the prompt template
        return custom.replace("{entity_types}", str(sorted(entity_types)))
    return """\
<PREVIOUS_CHUNKS>
{previous_chunks}
</PREVIOUS_CHUNKS>
<CURRENT_CHUNK>
{current_chunk}
</CURRENT_CHUNK>

Extract named entities from the CURRENT_CHUNK.
For each entity provide name, type (one of """ + str(sorted(entity_types)) + """), and summary.
Respond with a JSON array or [].
"""


def _get_resolution_prompt(kb_root: Path) -> str:
    prompts = get_prompts(kb_root)
    custom = prompts.get("entity_resolution_user")
    if custom:
        return custom
    return """\
<EXISTING_ENTITIES>
{existing_entities}
</EXISTING_ENTITIES>
<NEW_ENTITY>
{new_entity}
</NEW_ENTITY>

Determine if NEW_ENTITY is a duplicate of any entity in EXISTING_ENTITIES.
Respond with JSON: {{"is_duplicate": true/false, "existing_entity_id": "<uuid or null>", "canonical_name": "<best name>"}}
"""

# ─── Config + clients ─────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper supporting anthropic and openai providers."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        provider = cfg["llm"]["provider"]
        self.model = cfg["llm"]["model"]
        self.temperature = cfg["llm"].get("temperature", 0.0)
        self.max_tokens = cfg["llm"].get("max_tokens", 2000)
        self.provider = provider

        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
                sys.exit(1)
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not set", file=sys.stderr)
                sys.exit(1)
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        else:
            print(f"Error: unknown LLM provider '{provider}'", file=sys.stderr)
            sys.exit(1)

    def call(self, prompt: str) -> str:
        kb_root = kb_root_from_cfg(self.cfg)
        prompts = get_prompts(kb_root)
        system = prompts.get(
            "entity_extraction_system",
            "You are a precise entity extractor for a research knowledge base. "
            "Always respond with valid JSON only, no markdown fences, no explanation."
        )
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()


def get_llm_client(cfg: dict) -> LLMClient:
    return LLMClient(cfg)

def get_qdrant_client(cfg: dict) -> QdrantClient:
    return QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])

# ─── LLM calls ────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> any:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1] if raw.count("```") >= 2 else raw
        raw = raw.lstrip("json").strip().rstrip("```").strip()
    return json.loads(raw)


def extract_entities_from_window(llm: LLMClient,
                                  current_chunk: str, previous_chunks: list[str],
                                  kb_root: Path) -> list[dict]:
    """Call LLM to extract entities from a chunk window. Returns list of {name, type, summary}."""
    entity_types = get_entity_types(kb_root)
    prev_text = "\n---\n".join(previous_chunks) if previous_chunks else "(none)"
    prompt = _get_extraction_prompt(kb_root).format(
        previous_chunks=prev_text,
        current_chunk=current_chunk,
    )
    raw = llm.call(prompt)
    try:
        result = parse_json_response(raw)
        if not isinstance(result, list):
            return []
        valid = []
        for e in result:
            if isinstance(e, dict) and "name" in e and "type" in e and "summary" in e:
                e["type"] = e["type"].lower()
                if e["type"] not in entity_types:
                    e["type"] = "concept"
                valid.append(e)
        return valid
    except (json.JSONDecodeError, KeyError, ValueError):
        return []


def resolve_entity(llm: LLMClient,
                   new_entity: dict, candidates: list[dict],
                   kb_root: Path) -> dict:
    """
    Check if new_entity is a duplicate of any candidate.
    Returns: {is_duplicate, existing_entity_id, canonical_name}
    """
    if not candidates:
        return {"is_duplicate": False, "existing_entity_id": None,
                "canonical_name": new_entity["name"]}

    existing_text = json.dumps([
        {"entity_id": c["entity_id"], "name": c["name"], "summary": c.get("summary", "")}
        for c in candidates
    ], ensure_ascii=False)
    new_text = json.dumps({"name": new_entity["name"], "summary": new_entity["summary"]},
                          ensure_ascii=False)

    prompt = _get_resolution_prompt(kb_root).format(
        existing_entities=existing_text,
        new_entity=new_text,
    )
    raw = llm.call(prompt)
    try:
        result = parse_json_response(raw)
        return {
            "is_duplicate": bool(result.get("is_duplicate", False)),
            "existing_entity_id": result.get("existing_entity_id"),
            "canonical_name": result.get("canonical_name", new_entity["name"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"is_duplicate": False, "existing_entity_id": None,
                "canonical_name": new_entity["name"]}

# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_text(text: str, model: str) -> list[float]:
    resp = ollama_client.embeddings(model=model, prompt=text)
    return resp["embedding"]

# ─── Qdrant entity operations ─────────────────────────────────────────────────

def search_similar_entities(qdrant: QdrantClient, entities_col: str,
                             embedding: list[float], top: int = 5) -> list[dict]:
    """Cosine search in entities collection. Returns list of payload dicts with entity_id."""
    results = qdrant.query_points(
        collection_name=entities_col,
        query=embedding,
        limit=top,
        with_payload=True,
        score_threshold=0.75,   # only return genuinely similar candidates
    ).points
    candidates = []
    for r in results:
        p = dict(r.payload)
        p["entity_id"] = p.get("entity_id", str(r.id))
        candidates.append(p)
    return candidates


def upsert_entity(qdrant: QdrantClient, entities_col: str,
                  entity_id: str, name: str, entity_type: str,
                  summary: str, doc_id: str, domain: str,
                  tags: list[str], embedding: list[float],
                  dry_run: bool = False,
                  aliases: list[str] | None = None) -> str:
    """Insert a new entity. Returns entity_id."""
    if dry_run:
        print(f"    [dry-run] would store: {name} ({entity_type})")
        return entity_id

    payload = {
        "entity_id": entity_id,
        "name": name,
        "type": entity_type,
        "summary": summary,
        "aliases": aliases or [],
        "doc_ids": [doc_id],
        "domain": domain,
        "tags": tags,
        "_added": date.today().isoformat(),
    }
    qdrant.upsert(
        collection_name=entities_col,
        points=[PointStruct(id=entity_id, vector=embedding, payload=payload)],
    )
    return entity_id


def update_entity_doc_ref(qdrant: QdrantClient, entities_col: str,
                           entity_id: str, doc_id: str,
                           new_canonical_name: str | None = None,
                           dry_run: bool = False,
                           alias_candidate: str | None = None):
    """Add doc_id to an existing entity's doc_ids list (and optionally update name/aliases).

    alias_candidate: a name variant (e.g. Russian translation, abbreviation) to add to aliases
    if it differs from the current canonical name and isn't already listed.
    """
    if dry_run:
        return

    # Fetch current payload
    results = qdrant.retrieve(
        collection_name=entities_col,
        ids=[entity_id],
        with_payload=True,
    )
    if not results:
        return

    current = results[0].payload
    doc_ids = current.get("doc_ids", [])
    if doc_id not in doc_ids:
        doc_ids.append(doc_id)

    aliases = current.get("aliases", [])
    updates = {"doc_ids": doc_ids}

    if new_canonical_name and new_canonical_name != current.get("name"):
        # Old canonical name becomes an alias
        if current.get("name") and current["name"] not in aliases:
            aliases.append(current["name"])
        updates["name"] = new_canonical_name

    # Add alias_candidate (e.g. Russian name) if new and not already stored
    if alias_candidate and alias_candidate != current.get("name") \
            and alias_candidate != new_canonical_name \
            and alias_candidate not in aliases:
        aliases.append(alias_candidate)

    if aliases != current.get("aliases", []):
        updates["aliases"] = aliases

    qdrant.set_payload(
        collection_name=entities_col,
        payload=updates,
        points=[entity_id],
    )


def update_chunk_entity_refs(qdrant: QdrantClient, research_col: str,
                              doc_id: str, chunk_index: int,
                              entity_ids: list[str], dry_run: bool = False):
    """Add mentioned_entities field to a chunk payload."""
    if dry_run or not entity_ids:
        return

    # Find the specific chunk point
    results, _ = qdrant.scroll(
        collection_name=research_col,
        scroll_filter=Filter(must=[
            FieldCondition(key="_doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="_chunk_index", match=MatchValue(value=chunk_index)),
        ]),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    if not results:
        return

    qdrant.set_payload(
        collection_name=research_col,
        payload={"mentioned_entities": entity_ids},
        points=[str(results[0].id)],
    )

# ─── Core pipeline ────────────────────────────────────────────────────────────

def process_doc(doc_id: str, cfg: dict, llm: LLMClient, qdrant: QdrantClient,
                dry_run: bool = False, verbose: bool = False) -> list[str]:
    """
    Extract entities from all chunks of doc_id.
    Returns list of entity_ids stored/updated.
    """
    kb_root = kb_root_from_cfg(cfg)
    embed_model = cfg["embedding"]["model"]
    research_col = cfg["collections"]["research_docs"]
    entities_col = cfg.get("collections_extra", {}).get("entities", "entities")

    # Fetch all chunks for this doc, sorted by chunk_index
    chunks = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=research_col,
            scroll_filter=Filter(must=[
                FieldCondition(key="_doc_id", match=MatchValue(value=doc_id)),
            ]),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        chunks.extend(results)
        if offset is None:
            break

    if not chunks:
        print(f"  [warn] no chunks found for doc_id {doc_id[:8]}...")
        return []

    chunks.sort(key=lambda r: r.payload.get("_chunk_index", 0))
    doc_title = chunks[0].payload.get("title", doc_id[:8])
    doc_domain = chunks[0].payload.get("domain", "cross")
    doc_tags = chunks[0].payload.get("tags", [])
    total = len(chunks)

    print(f"\n→ Entity extraction: {doc_title} ({total} chunks)")

    # Context window: current chunk + up to 4 previous (Graphiti uses n=4)
    CONTEXT_WINDOW = 4
    all_entity_ids = []  # all entities encountered across this doc

    for i, chunk_result in enumerate(chunks):
        chunk_text = chunk_result.payload.get("text", "")
        chunk_index = chunk_result.payload.get("_chunk_index", i)

        prev_texts = [c.payload.get("text", "") for c in chunks[max(0, i - CONTEXT_WINDOW):i]]

        if verbose:
            print(f"  chunk {chunk_index+1}/{total}...", end=" ", flush=True)

        # LLM extraction
        candidates = extract_entities_from_window(llm, chunk_text, prev_texts, kb_root)

        if not candidates:
            if verbose:
                print("(no entities)")
            continue

        if verbose:
            print(f"({len(candidates)} candidates)")

        chunk_entity_ids = []

        for ent in candidates:
            name = ent["name"]
            ent_type = ent["type"]
            summary = ent["summary"]

            # Embed entity name for similarity search
            embed_text_str = f"{name}: {summary}"
            embedding = embed_text(embed_text_str, embed_model)

            # Search for similar existing entities
            similar = search_similar_entities(qdrant, entities_col, embedding)

            # LLM resolution
            resolution = resolve_entity(llm, ent, similar, kb_root)

            if resolution["is_duplicate"] and resolution["existing_entity_id"]:
                existing_id = resolution["existing_entity_id"]
                canonical = resolution["canonical_name"]
                if verbose:
                    print(f"    merge: '{name}' → '{canonical}' ({existing_id[:8]}...)")
                # Pass the extracted name as an alias candidate (handles cross-language merges)
                update_entity_doc_ref(qdrant, entities_col, existing_id, doc_id,
                                      canonical, dry_run, alias_candidate=name)
                chunk_entity_ids.append(existing_id)
                if existing_id not in all_entity_ids:
                    all_entity_ids.append(existing_id)
            else:
                # New entity
                new_id = str(uuid.uuid4())
                canonical = resolution["canonical_name"]
                # If extracted name differs from canonical (e.g. Russian name, abbreviation),
                # keep it as an alias so BM25 can find it
                initial_aliases = [name] if name != canonical else []
                if verbose:
                    print(f"    new:   '{canonical}' ({ent_type})" +
                          (f"  [alias: {name}]" if initial_aliases else ""))
                upsert_entity(qdrant, entities_col, new_id, canonical, ent_type,
                              summary, doc_id, doc_domain, doc_tags, embedding, dry_run,
                              aliases=initial_aliases)
                chunk_entity_ids.append(new_id)
                if new_id not in all_entity_ids:
                    all_entity_ids.append(new_id)

        # Write entity refs back to chunk payload
        update_chunk_entity_refs(qdrant, research_col, doc_id, chunk_index,
                                  chunk_entity_ids, dry_run)

    print(f"  ✓ {len(all_entity_ids)} entities stored/updated for '{doc_title}'")

    # Phase 3 — run fact extraction automatically unless disabled
    if not dry_run:
        _run_fact_extraction(doc_id, cfg, qdrant, verbose=verbose)

    return all_entity_ids


def _run_fact_extraction(doc_id: str, cfg: dict, qdrant: QdrantClient,
                          verbose: bool = False):
    """Auto-run fact extraction after entity extraction."""
    try:
        from fact_extract import process_doc as fact_process_doc, get_llm_client as fact_llm
        from fact_extract import ensure_facts_collection
        facts_col = cfg.get("collections_extra", {}).get("facts", "facts")
        ensure_facts_collection(qdrant, facts_col, cfg["embedding"]["dimensions"])
        llm = fact_llm(cfg)
        fact_process_doc(doc_id, cfg, llm, qdrant, dry_run=False, verbose=verbose)
    except Exception as e:
        print(f"  [warn] fact extraction failed: {e}")


def get_unprocessed_docs(qdrant: QdrantClient, cfg: dict) -> list[dict]:
    """Return docs in research_docs that have no mentioned_entities on any chunk."""
    research_col = cfg["collections"]["research_docs"]
    seen_docs = {}
    processed_docs = set()

    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=research_col,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in results:
            doc_id = r.payload.get("_doc_id")
            if not doc_id:
                continue
            if doc_id not in seen_docs:
                seen_docs[doc_id] = r.payload.get("title", doc_id[:8])
            if r.payload.get("mentioned_entities"):
                processed_docs.add(doc_id)
        if offset is None:
            break

    return [{"doc_id": did, "title": title}
            for did, title in seen_docs.items()
            if did not in processed_docs]


def list_entities(qdrant: QdrantClient, cfg: dict, domain: str | None = None,
                  entity_type: str | None = None):
    """Print all stored entities."""
    entities_col = cfg.get("collections_extra", {}).get("entities", "entities")

    filters = []
    if domain:
        filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if entity_type:
        filters.append(FieldCondition(key="type", match=MatchValue(value=entity_type)))
    filt = Filter(must=filters) if filters else None

    seen = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=entities_col,
            scroll_filter=filt,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        seen.extend(results)
        if offset is None:
            break

    if not seen:
        print("  [entities] is empty.")
        return

    # Group by type
    by_type: dict[str, list] = {}
    for r in seen:
        t = r.payload.get("type", "unknown")
        by_type.setdefault(t, []).append(r.payload)

    print(f"\n  [entities] — {len(seen)} total\n")
    for etype, ents in sorted(by_type.items()):
        print(f"  {etype.upper()} ({len(ents)})")
        for e in sorted(ents, key=lambda x: x.get("name", "")):
            doc_count = len(e.get("doc_ids", []))
            aliases = e.get("aliases", [])
            alias_str = f"  aka: {', '.join(aliases)}" if aliases else ""
            print(f"    • {e.get('name', '?')}{alias_str}  [{doc_count} doc(s)]")
            print(f"      {e.get('summary', '')[:120]}")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract named entities from KB documents into the entities collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/entity_extract.py --doc-id 1494731e-xxxx --kb-root ~/my-kb
  python scripts/entity_extract.py --doc-id 1494731e-xxxx --kb-root ~/my-kb --dry-run --verbose
  python scripts/entity_extract.py --all --kb-root ~/my-kb
  python scripts/entity_extract.py --list-entities --kb-root ~/my-kb
  python scripts/entity_extract.py --list-entities --kb-root ~/my-kb --type material
        """,
    )
    add_kb_root_arg(parser)
    parser.add_argument("--doc-id", dest="doc_id", help="Process a specific doc_id")
    parser.add_argument("--all", action="store_true",
                        help="Process all docs without mentioned_entities")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run extraction but don't write to Qdrant")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-chunk and per-entity progress")
    parser.add_argument("--list-entities", action="store_true",
                        help="List all stored entities and exit")
    parser.add_argument("--domain", help="Filter --list-entities by domain")
    parser.add_argument("--type", dest="entity_type",
                        help="Filter --list-entities by type")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    qdrant = get_qdrant_client(cfg)
    entities_col = cfg.get("collections_extra", {}).get("entities", "entities")

    # Ensure entities collection exists
    existing_cols = {c.name for c in qdrant.get_collections().collections}
    if entities_col not in existing_cols:
        print(f"Creating [{entities_col}] collection...")
        from init_collections import create_collection, ENTITY_FIELDS
        create_collection(qdrant, entities_col, cfg["embedding"]["dimensions"], ENTITY_FIELDS)

    if args.list_entities:
        list_entities(qdrant, cfg, domain=args.domain, entity_type=args.entity_type)
        return

    llm = get_llm_client(cfg)

    if args.doc_id:
        process_doc(args.doc_id, cfg, llm, qdrant,
                    dry_run=args.dry_run, verbose=args.verbose)

    elif args.all:
        pending = get_unprocessed_docs(qdrant, cfg)
        if not pending:
            print("All docs already processed.")
            return
        print(f"Found {len(pending)} unprocessed docs.")
        for doc in pending:
            process_doc(doc["doc_id"], cfg, llm, qdrant,
                        dry_run=args.dry_run, verbose=args.verbose)
        print(f"\n✓ Done. Processed {len(pending)} docs.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
