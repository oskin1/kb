#!/usr/bin/env python3
"""
tg_ingest.py — Telegram-triggered ingest handler.

Usage:
    python scripts/tg_ingest.py --file /tmp/document.pdf --kb-root /path/to/kb [--caption "domain=mydomain"]
    python scripts/tg_ingest.py --url https://example.com/article --kb-root ~/kb [--caption "domain=mydomain"]
"""

import argparse
import re
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import date

from kb_root import add_kb_root_arg, resolve_kb_root, raw_dir

SCRIPTS_DIR = Path(__file__).parent

# Map file extensions to raw subdirs
EXT_DIR_MAP = {
    ".pdf": "papers",
    ".txt": "papers",
    ".md": "papers",
    ".html": "web",
    ".htm": "web",
    ".csv": "data",
    ".xlsx": "data",
    ".xls": "data",
}

# Map file extensions to default collections
EXT_COLLECTION_MAP = {
    ".pdf": "research_docs",
    ".txt": "research_docs",
    ".md": "research_docs",
    ".html": "research_docs",
    ".htm": "research_docs",
    ".csv": "data_tables",
    ".xlsx": "data_tables",
    ".xls": "data_tables",
}

def parse_caption(caption: str) -> dict:
    if not caption:
        return {}

    meta = {}
    kv_matches = re.findall(r"(\w+)\s*[=:]\s*([^\s,;]+)", caption)
    known_keys = {"domain", "subdomain", "project", "confidence", "language",
                  "source_type", "tags", "collection", "title",
                  "valid_at", "invalid_at", "supersedes"}
    for k, v in kv_matches:
        if k in known_keys:
            if k == "tags" and "," in v:
                meta[k] = [x.strip() for x in v.split(",")]
            else:
                meta[k] = v

    return meta


def copy_to_raw(src: Path, kb_root: Path) -> Path:
    """Copy an inbound file to the appropriate raw/ subdir."""
    ext = src.suffix.lower()
    subdir = EXT_DIR_MAP.get(ext, "papers")
    dest_dir = raw_dir(kb_root, subdir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        stem = src.stem
        dest = dest_dir / f"{stem}_{date.today().isoformat()}{ext}"
    shutil.copy2(src, dest)
    return dest


def run_ingest(file_path: Path, collection: str, meta: dict, kb_root_str: str) -> tuple[bool, str]:
    """Run ingest.py and capture output."""
    python = sys.executable
    ingest_script = SCRIPTS_DIR / "ingest.py"

    valid_at = meta.pop("valid_at", None)
    invalid_at = meta.pop("invalid_at", None)
    supersedes = meta.pop("supersedes", None)

    meta_args = []
    for k, v in meta.items():
        if k == "collection":
            continue
        if isinstance(v, list):
            meta_args += ["--meta", f"{k}={','.join(v)}"]
        else:
            meta_args += ["--meta", f"{k}={v}"]

    if valid_at:
        meta_args += ["--valid-at", valid_at]
    if invalid_at:
        meta_args += ["--invalid-at", invalid_at]
    if supersedes:
        meta_args += ["--supersedes", supersedes]

    cmd = [python, str(ingest_script), str(file_path),
           "--kb-root", kb_root_str,
           "--collection", collection] + meta_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def run_fetch_url(url: str, collection: str, meta: dict, kb_root_str: str) -> tuple[bool, str, Path | None]:
    """Run fetch_url.py and return (ok, output, saved_path)."""
    python = sys.executable
    fetch_script = SCRIPTS_DIR / "fetch_url.py"

    meta_args = []
    for k, v in meta.items():
        if k == "collection":
            continue
        if isinstance(v, list):
            meta_args += ["--meta", f"{k}={','.join(v)}"]
        else:
            meta_args += ["--meta", f"{k}={v}"]

    cmd = [python, str(fetch_script), url,
           "--kb-root", kb_root_str,
           "--collection", collection,
           "--ingest"] + meta_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    saved_path = None
    for line in output.splitlines():
        m = re.search(r"Saved:\s+(.+?\.html)", line)
        if m:
            saved_path = Path(m.group(1).strip())
            break

    return result.returncode == 0, output, saved_path


def format_report(ok: bool, mode: str, name: str, meta: dict, output: str) -> str:
    lines = []
    status = "✅ Ingested" if ok else "❌ Ingest failed"

    lines.append(f"{status}: *{name}*")
    if meta.get("domain"):
        lines.append(f"Domain: {meta['domain']}" + (f" / {meta['subdomain']}" if meta.get('subdomain') else ""))
    if meta.get("project") and meta["project"] != "general":
        lines.append(f"Project: {meta['project']}")
    if meta.get("collection"):
        lines.append(f"Collection: {meta['collection']}")

    chunks_m = re.search(r"chunks:\s+(\d+)", output)
    if chunks_m:
        lines.append(f"Chunks: {chunks_m.group(1)}")

    if not ok:
        err_lines = [l for l in output.splitlines() if l.strip()]
        if err_lines:
            lines.append(f"Error: {err_lines[-1][:200]}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Telegram ingest handler")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Local file path (from MediaPath)")
    group.add_argument("--url", help="URL to fetch and ingest")
    add_kb_root_arg(parser)
    parser.add_argument("--caption", default="", help="Telegram caption (metadata hints)")
    parser.add_argument("--collection", default=None,
                        help="Override target collection")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    kb_root_str = str(kb_root)
    meta = parse_caption(args.caption)

    if args.file:
        src = Path(args.file)
        if not src.exists():
            print(f"❌ File not found: {src}")
            sys.exit(1)

        ext = src.suffix.lower()
        if ext not in EXT_DIR_MAP:
            print(f"❌ Unsupported file type: {ext}")
            print(f"Supported: {', '.join(EXT_DIR_MAP.keys())}")
            sys.exit(1)

        collection = (
            args.collection
            or meta.pop("collection", None)
            or EXT_COLLECTION_MAP.get(ext, "research_docs")
        )

        dest = copy_to_raw(src, kb_root)
        print(f"Copied to: {dest}")

        ok, output = run_ingest(dest, collection, meta, kb_root_str)
        report = format_report(ok, "file", src.name, {**meta, "collection": collection}, output)
        print(report)
        sys.exit(0 if ok else 1)

    elif args.url:
        collection = (
            args.collection
            or meta.pop("collection", None)
            or "research_docs"
        )

        if args.url.startswith("arxiv:"):
            arxiv_query = args.url[len("arxiv:"):].strip()
            if not arxiv_query:
                print("❌ No query provided after 'arxiv:' prefix")
                sys.exit(1)

            arxiv_script = SCRIPTS_DIR / "arxiv_search.py"
            cmd = [
                sys.executable, str(arxiv_script),
                arxiv_query,
                "--kb-root", kb_root_str,
                "--ingest",
                "--collection", collection,
            ]
            if meta.get("domain"):
                cmd += ["--domain", meta["domain"]]
            if meta.get("project"):
                cmd += ["--project", meta["project"]]

            print(f"🔍 arXiv search: {arxiv_query}")
            result = subprocess.run(cmd, text=True)
            sys.exit(result.returncode)

        ok, output, saved_path = run_fetch_url(args.url, collection, meta, kb_root_str)
        name = saved_path.name if saved_path else args.url
        report = format_report(ok, "url", name, {**meta, "collection": collection}, output)
        print(report)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
