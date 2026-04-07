#!/usr/bin/env python3
"""
init.py — Scaffold a new knowledge base directory.

Usage:
    python init.py --kb-root /path/to/my-kb

Creates the standard directory structure and copies default config files
from the repo into the new KB.  Safe to re-run: existing files are never
overwritten (use --force to replace configs).
"""

import argparse
import shutil
from pathlib import Path

from kb_root import DEFAULT_CONFIG_DIR


DIRS = [
    "config",
    "raw/papers",
    "raw/reports",
    "raw/data",
    "raw/web",
    "notes",
    "proposals",
    "projects",
]

DEFAULT_CONFIGS = [
    "config.yaml",
    "domain.yaml",
]


def init_kb(kb_root: Path, force: bool = False) -> None:
    kb_root = kb_root.expanduser().resolve()
    print(f"Initialising KB at {kb_root}\n")

    # Create directories
    for d in DIRS:
        p = kb_root / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"  dir  {p}")

    # Copy default configs
    for name in DEFAULT_CONFIGS:
        src = DEFAULT_CONFIG_DIR / name
        dst = kb_root / "config" / name
        if dst.exists() and not force:
            print(f"  skip {dst}  (already exists, use --force to overwrite)")
        else:
            shutil.copy2(src, dst)
            print(f"  copy {dst}")

    print(f"\nDone.  Next steps:")
    print(f"  1. Edit {kb_root / 'config' / 'domain.yaml'} for your domain")
    print(f"  2. Drop documents into {kb_root / 'raw' / ''}")
    print(f"  3. Run: python scripts/init_collections.py --kb-root {kb_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new knowledge base directory",
    )
    parser.add_argument(
        "--kb-root", required=True, metavar="PATH",
        help="Where to create the knowledge base",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config files",
    )
    args = parser.parse_args()
    init_kb(Path(args.kb_root), force=args.force)


if __name__ == "__main__":
    main()
