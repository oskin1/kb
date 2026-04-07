#!/usr/bin/env python3
"""
fetch_url.py — Download a web page and save it to kb/raw/web/ for ingestion.

Usage:
    python scripts/fetch_url.py <url> --kb-root /path/to/kb [options]
    python scripts/fetch_url.py https://example.com/article --kb-root ~/kb --ingest
"""

import argparse
import re
import subprocess
import sys
import yaml
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from kb_root import add_kb_root_arg, resolve_kb_root, raw_dir

SCRIPTS_DIR = Path(__file__).parent


def url_to_filename(url: str) -> str:
    """Generate a safe filename from a URL."""
    parsed = urlparse(url)
    slug = parsed.netloc + parsed.path
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > 80:
        slug = slug[:80]
    return slug + ".html"


def fetch_url(url: str, output_path: Path) -> dict:
    """
    Download URL, clean HTML, save to output_path.
    Returns auto-extracted metadata dict.
    """
    print(f"  Fetching: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        print(f"  [warn] Non-HTML content-type: {content_type}")

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # Extract metadata
    meta = {}
    title_tag = soup.find("title")
    meta["title"] = title_tag.get_text().strip() if title_tag else output_path.stem

    desc = (
        soup.find("meta", property="og:description")
        or soup.find("meta", attrs={"name": "description"})
    )
    if desc and desc.get("content"):
        meta["description"] = desc["content"].strip()[:500]

    for prop in ["article:published_time", "og:updated_time", "datePublished"]:
        d = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if d and d.get("content"):
            meta["date"] = d["content"][:10]
            break

    meta["source"] = url
    meta["source_type"] = "web"
    meta["_fetched"] = date.today().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Saved: {output_path} ({len(html)//1024} KB)")

    return meta


def write_sidecar(output_path: Path, auto_meta: dict, extra_meta: dict):
    """Write merged metadata to .yaml sidecar alongside the HTML file."""
    sidecar_path = output_path.with_suffix(".yaml")
    merged = {**auto_meta, **extra_meta}
    merged.setdefault("domain", "cross")
    merged.setdefault("source_type", "web")
    merged.setdefault("confidence", "probable")
    merged.setdefault("language", "en")
    merged.setdefault("project", "general")
    with open(sidecar_path, "w") as f:
        yaml.dump(merged, f, allow_unicode=True, default_flow_style=False)
    print(f"  Sidecar: {sidecar_path.name}")
    return sidecar_path


def run_ingest(output_path: Path, collection: str, extra_meta: dict, kb_root_str: str):
    """Run ingest.py on the downloaded file."""
    ingest_script = SCRIPTS_DIR / "ingest.py"
    python = sys.executable

    meta_args = []
    for k, v in extra_meta.items():
        if isinstance(v, list):
            meta_args += ["--meta", f"{k}={','.join(v)}"]
        else:
            meta_args += ["--meta", f"{k}={v}"]

    cmd = [python, str(ingest_script), str(output_path),
           "--kb-root", kb_root_str,
           "--collection", collection] + meta_args

    print(f"\n  Running ingest...")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Fetch URL and save to kb/raw/web/")
    parser.add_argument("url", help="URL to fetch")
    add_kb_root_arg(parser)
    parser.add_argument("--output", help="Output filename (default: auto from URL)")
    parser.add_argument("--meta", nargs="*", default=[],
                        help="Metadata as key=value pairs")
    parser.add_argument("--ingest", action="store_true",
                        help="Auto-ingest after download")
    parser.add_argument("--collection", default="research_docs",
                        choices=["research_docs", "data_tables", "insights"],
                        help="Target collection for ingest (default: research_docs)")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)

    # Parse extra metadata
    extra_meta = {}
    for kv in args.meta or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            if "," in v:
                extra_meta[k] = [x.strip() for x in v.split(",")]
            else:
                extra_meta[k] = v

    # Determine output path
    filename = args.output if args.output else url_to_filename(args.url)
    if not filename.endswith((".html", ".htm")):
        filename += ".html"
    output_path = raw_dir(kb_root, "web") / filename

    # Fetch
    try:
        auto_meta = fetch_url(args.url, output_path)
    except requests.exceptions.RequestException as e:
        print(f"  [error] Failed to fetch: {e}")
        sys.exit(1)

    # Write sidecar
    write_sidecar(output_path, auto_meta, extra_meta)

    # Ingest if requested
    if args.ingest:
        ok = run_ingest(output_path, args.collection, extra_meta, str(kb_root))
        if ok:
            print(f"\n  ✓ Ingested into [{args.collection}]")
        else:
            print(f"\n  [error] Ingest failed")
            sys.exit(1)
    else:
        print(f"\n  Done. To ingest:")
        print(f"    python scripts/ingest.py {output_path} --kb-root {kb_root}")

    print(f"\n  Title:  {auto_meta.get('title', '(unknown)')}")
    print(f"  Source: {args.url}")
    if "date" in auto_meta:
        print(f"  Date:   {auto_meta['date']}")


if __name__ == "__main__":
    main()
