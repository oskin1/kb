#!/usr/bin/env python3
"""
write_project.py — Create or update a project file and ingest into Qdrant.

Usage:
    python scripts/write_project.py --name kb-infra --summary "KB pipeline" --kb-root ~/kb
    python scripts/write_project.py /tmp/project.md --kb-root ~/kb
    python scripts/write_project.py --name kb-infra --log "Added write_note.py" --kb-root ~/kb
    python scripts/write_project.py --list --kb-root ~/kb
"""

import argparse
import sys
import uuid
import yaml
from datetime import date
from pathlib import Path

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from kb_root import add_kb_root_arg, resolve_kb_root, load_config, projects_dir


def parse_front_matter(content: str) -> tuple[dict, str]:
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            try:
                fm = yaml.safe_load(content[3:end].strip()) or {}
                return fm, content[end + 4:].strip()
            except yaml.YAMLError:
                pass
    return {}, content


def slugify(name: str) -> str:
    import re
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_-]+", "-", name)
    return name[:60].strip("-")


def list_projects(kb_root: Path):
    pd = projects_dir(kb_root)
    pd.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(pd.glob("*.md")) if f.name != "TEMPLATE.md"]
    if not files:
        print("No projects yet.")
        return
    print(f"{'Name':<30} {'Status':<12} {'Domain':<12} Updated")
    print("-" * 70)
    for f in files:
        content = f.read_text(encoding="utf-8")
        fm, _ = parse_front_matter(content)
        print(f"{fm.get('name', f.stem):<30} {fm.get('status', '?'):<12} {fm.get('domain', '?'):<12} {fm.get('updated', '?')}")


def ingest_project(body: str, meta: dict, output_path: Path, cfg: dict) -> int:
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    embed_model = cfg["embedding"]["model"]
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]

    # Remove old chunks for this project first
    try:
        client.delete(
            collection_name="insights",
            points_selector=Filter(
                must=[FieldCondition(key="project_file", match=MatchValue(value=str(output_path)))]
            )
        )
    except Exception:
        pass

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([Document(text=body)])
    chunks = [n.get_content() for n in nodes]
    if not chunks:
        return 0

    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()

    vectors = ollama_client.embed(model=embed_model, input=chunks)["embeddings"]
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "text": chunk,
            "title": f"[project] {meta.get('name', output_path.stem)}",
            "authors": [],
            "domain": meta.get("domain", "cross"),
            "subdomain": "",
            "source_type": "own-analysis",
            "source": f"project {meta.get('name', '')}",
            "confidence": "established",
            "tags": meta.get("tags", []),
            "project": meta.get("name", "general"),
            "date": today,
            "language": "en",
            "insight_type": "summary",
            "project_file": str(output_path),
            "_doc_id": doc_id,
            "_chunk_index": i,
            "_chunk_total": len(chunks),
            "_added": today,
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    client.upsert(collection_name="insights", points=points)
    return len(points)


def build_new_project(name: str, summary: str, domain: str, status: str, today: str) -> str:
    meta = {
        "name": name,
        "status": status,
        "domain": domain,
        "started": today,
        "updated": today,
    }
    body = f"""# {name}

## Summary
{summary}

## Goals
-

## Active Threads
_Not yet defined._

## Key Documents
_None yet._

## Log
- {today}: Project created.
"""
    return f"---\n{yaml.dump(meta, allow_unicode=True, default_flow_style=False).strip()}\n---\n\n{body}"


def append_log(content: str, entry: str, today: str) -> str:
    """Append a log entry under the ## Log section."""
    log_marker = "## Log"
    if log_marker in content:
        idx = content.index(log_marker) + len(log_marker)
        return content[:idx] + f"\n- {today}: {entry}" + content[idx:]
    return content + f"\n\n## Log\n- {today}: {entry}\n"


def main():
    parser = argparse.ArgumentParser(description="Create or update a project")
    parser.add_argument("file", nargs="?", help="Full project markdown file")
    add_kb_root_arg(parser)
    parser.add_argument("--name", help="Project name (used as filename slug)")
    parser.add_argument("--summary", help="One-paragraph summary (for new projects)")
    parser.add_argument("--domain", default="cross")
    parser.add_argument("--status", default="active",
                        choices=["active", "paused", "complete", "archived"])
    parser.add_argument("--log", help="Append a log entry to existing project")
    parser.add_argument("--no-ingest", action="store_true")
    parser.add_argument("--list", action="store_true", help="List all projects")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)

    if args.list:
        list_projects(kb_root); return

    today = date.today().isoformat()
    pd = projects_dir(kb_root)
    pd.mkdir(parents=True, exist_ok=True)

    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        fm, body = parse_front_matter(content)
        name = args.name or fm.get("name") or Path(args.file).stem
        output_path = pd / f"{slugify(name)}.md"
        fm["updated"] = today
        fm.setdefault("started", today)
        fm.setdefault("status", args.status)
        fm.setdefault("domain", args.domain)
        fm["name"] = name
        final = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).strip()}\n---\n\n{body.strip()}\n"
        output_path.write_text(final, encoding="utf-8")

    elif args.name:
        output_path = pd / f"{slugify(args.name)}.md"

        if output_path.exists():
            content = output_path.read_text(encoding="utf-8")
            fm, body = parse_front_matter(content)
            fm["updated"] = today
            if args.status:
                fm["status"] = args.status
            if args.log:
                body = append_log(body, args.log, today)
            final = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).strip()}\n---\n\n{body.strip()}\n"
        else:
            if not args.summary:
                print("Error: --summary required when creating a new project"); sys.exit(1)
            final = build_new_project(args.name, args.summary, args.domain, args.status, today)
            fm = {"name": args.name, "domain": args.domain}

        output_path.write_text(final, encoding="utf-8")

    else:
        print("Error: provide a file path or --name"); sys.exit(1)

    print(f"✓ Project saved: {output_path}")

    if not args.no_ingest:
        _, body = parse_front_matter(output_path.read_text(encoding="utf-8"))
        cfg = load_config(kb_root)
        chunks = ingest_project(body, fm, output_path, cfg)
        print(f"✓ Ingested {chunks} chunks into [insights]")
        print(f"  Name:   {fm.get('name', output_path.stem)}")
        print(f"  Status: {fm.get('status', '?')} | Domain: {fm.get('domain', '?')}")


if __name__ == "__main__":
    main()
