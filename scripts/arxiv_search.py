#!/usr/bin/env python3
"""
arxiv_search.py — Search arXiv for papers and optionally ingest them into the KB.

Usage:
    python scripts/arxiv_search.py "search query" --kb-root /path/to/kb --max 10
    python scripts/arxiv_search.py "topic" --kb-root ~/kb --domain mydomain --ingest
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError

from kb_root import add_kb_root_arg, resolve_kb_root

SCRIPTS_DIR = Path(__file__).parent
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> list[dict]:
    """Query arXiv API and return list of paper dicts."""
    sort_map = {
        "relevance": "relevance",
        "date": "submittedDate",
    }
    sort_order = "descending"
    sort_by_api = sort_map.get(sort_by, "relevance")

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by_api,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API}?{urlencode(params)}"

    try:
        with urlopen(url, timeout=30) as resp:
            xml_data = resp.read()
    except URLError as e:
        print(f"❌ Error querying arXiv: {e}", file=sys.stderr)
        sys.exit(1)

    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        id_elem = entry.find(f"{{{ATOM_NS}}}id")
        raw_id = id_elem.text.strip() if id_elem is not None else ""
        arxiv_id = raw_id.split("/")[-1]
        if "v" in arxiv_id and arxiv_id[-1].isdigit():
            base = arxiv_id.rsplit("v", 1)[0]
            if arxiv_id.rsplit("v", 1)[-1].isdigit():
                arxiv_id = base

        title_elem = entry.find(f"{{{ATOM_NS}}}title")
        title = " ".join((title_elem.text or "").split()) if title_elem is not None else "No title"

        summary_elem = entry.find(f"{{{ATOM_NS}}}summary")
        abstract = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""

        published_elem = entry.find(f"{{{ATOM_NS}}}published")
        date = published_elem.text[:10] if published_elem is not None else "unknown"

        authors = []
        for author_elem in entry.findall(f"{{{ATOM_NS}}}author"):
            name_elem = author_elem.find(f"{{{ATOM_NS}}}name")
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "date": date,
            "abstract": abstract,
            "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
            "url_html": f"https://arxiv.org/html/{arxiv_id}",
        })

    return papers


def print_results(papers: list[dict]) -> None:
    if not papers:
        print("No results found.")
        return

    for i, p in enumerate(papers):
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += f" et al. ({len(p['authors'])} authors)"

        abstract_preview = p["abstract"][:280]
        if len(p["abstract"]) > 280:
            abstract_preview += "..."

        print(f"\n[{i}] {p['title']}")
        print(f"    Authors : {authors_str}")
        print(f"    Date    : {p['date']}")
        print(f"    URL     : {p['url_abs']}")
        print(f"    Abstract: {abstract_preview}")


def ingest_papers(
    papers: list[dict],
    indices: list[int],
    collection: str,
    domain: str,
    project: str,
    kb_root_str: str,
) -> None:
    """Fetch and ingest selected papers via fetch_url.py."""
    fetch_script = SCRIPTS_DIR / "fetch_url.py"

    for idx in indices:
        if idx < 0 or idx >= len(papers):
            print(f"⚠️  Index {idx} out of range, skipping.")
            continue

        p = papers[idx]
        url = p["url_html"]
        print(f"\n→ Ingesting [{idx}]: {p['title']}")
        print(f"  URL: {url}")

        cmd = [
            sys.executable,
            str(fetch_script),
            url,
            "--kb-root", kb_root_str,
            "--collection", collection,
            "--ingest",
            "--meta",
            f"domain={domain}",
            f"project={project}",
            "source_type=paper",
            f"arxiv_id={p['id']}",
            f"title={p['title'][:100]}",
        ]

        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  HTML fetch failed, trying abstract page...")
            cmd[3] = p["url_abs"]
            result = subprocess.run(cmd, text=True)
        if result.returncode == 0:
            print(f"  ✅ Done")
        else:
            print(f"  ❌ Failed to ingest [{idx}]")


def parse_indices(response: str, total: int) -> list[int]:
    response = response.strip().lower()
    if response == "all":
        return list(range(total))
    if not response:
        return []
    indices = []
    for part in response.split(","):
        part = part.strip()
        try:
            n = int(part)
            indices.append(n)
        except ValueError:
            print(f"⚠️  Ignoring invalid index: '{part}'")
    return indices


def main():
    parser = argparse.ArgumentParser(description="Search arXiv and optionally ingest papers")
    parser.add_argument("query", help="Search query string")
    add_kb_root_arg(parser)
    parser.add_argument("--max", type=int, default=10, metavar="N",
                        help="Max number of results (default: 10)")
    parser.add_argument("--domain", default="cross",
                        help="Metadata domain for ingest (default: cross)")
    parser.add_argument("--project", default="general",
                        help="Metadata project for ingest (default: general)")
    parser.add_argument("--collection", default="research_docs",
                        help="Target KB collection (default: research_docs)")
    parser.add_argument("--ingest", action="store_true",
                        help="Prompt to select papers to ingest after showing results")
    parser.add_argument("--sort", choices=["relevance", "date"], default="relevance",
                        help="Sort order: relevance (default) or date")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)

    print(f"🔍 Searching arXiv for: \"{args.query}\" (max={args.max}, sort={args.sort})")
    papers = search_arxiv(args.query, max_results=args.max, sort_by=args.sort)
    print(f"   Found {len(papers)} results\n")

    print_results(papers)

    if args.ingest and papers:
        print("\n" + "─" * 60)
        response = input("Ingest which? (comma-separated indices, or 'all', or Enter to skip): ")
        indices = parse_indices(response, len(papers))
        if indices:
            ingest_papers(papers, indices, args.collection, args.domain, args.project, str(kb_root))
        else:
            print("Skipped.")


if __name__ == "__main__":
    main()
