#!/usr/bin/env python3
"""
Export insights collection to a markdown file for review.

Usage:
    python scripts/export_insights.py --kb-root /path/to/kb [options]
"""

import argparse
from pathlib import Path
from datetime import date

from kb_root import add_kb_root_arg, resolve_kb_root, load_config


def main():
    parser = argparse.ArgumentParser(description="Export insights to markdown")
    add_kb_root_arg(parser)
    parser.add_argument("--project", help="Filter by project")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--output", help="Output markdown file")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    output = Path(args.output) if args.output else kb_root / "insights_export.md"

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    collection = cfg["collections"]["insights"]

    conditions = []
    if args.project:
        conditions.append(FieldCondition(key="project", match=MatchValue(value=args.project)))
    if args.domain:
        conditions.append(FieldCondition(key="domain", match=MatchValue(value=args.domain)))

    qdrant_filter = Filter(must=conditions) if conditions else None

    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qdrant_filter,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        print("No insights found.")
        return

    # Group by date
    by_date = {}
    for hit in results:
        p = hit.payload
        d = p.get("_added", "unknown")
        by_date.setdefault(d, []).append(p)

    lines = [
        f"# Knowledge Base Insights Export",
        f"Generated: {date.today().isoformat()}",
        f"Total: {len(results)} insights",
        "",
    ]

    if args.project:
        lines.append(f"**Project filter:** {args.project}")
    if args.domain:
        lines.append(f"**Domain filter:** {args.domain}")
    lines.append("")

    for d in sorted(by_date.keys(), reverse=True):
        lines.append(f"## {d}")
        for p in by_date[d]:
            lines.append(f"\n### {p.get('title', 'Untitled')}")
            lines.append(f"- **Type:** {p.get('source_type')} | "
                        f"**Domain:** {p.get('domain')} / {p.get('subdomain')}")
            lines.append(f"- **Confidence:** {p.get('confidence')} | "
                        f"**Project:** {p.get('project')}")
            tags = p.get("tags", [])
            if tags:
                lines.append(f"- **Tags:** {', '.join(tags)}")
            sources = p.get("source", "")
            if sources:
                lines.append(f"- **Source:** {sources}")
            lines.append(f"\n{p.get('text', '')}\n")

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(results)} insights → {output}")


if __name__ == "__main__":
    main()
