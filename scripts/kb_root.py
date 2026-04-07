"""
kb_root.py — Shared KB root resolution for all scripts.

Every script accepts --kb-root <path> pointing to the user's knowledge base
directory.  This module provides helpers to wire that argument into argparse
and to derive standard paths (config, raw, notes, …) relative to the KB root.
"""

from pathlib import Path
import argparse
import yaml

# Repo-shipped default configs (used by init and as fallback reference)
_REPO_DIR = Path(__file__).parent.parent
DEFAULT_CONFIG_DIR = _REPO_DIR / "config"


def add_kb_root_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --kb-root argument to an existing ArgumentParser."""
    parser.add_argument(
        "--kb-root",
        required=True,
        metavar="PATH",
        help="Path to the knowledge base directory",
    )


def resolve_kb_root(args: argparse.Namespace) -> Path:
    """Return an absolute Path for the KB root from parsed args."""
    return Path(args.kb_root).expanduser().resolve()


# ── Derived paths ────────────────────────────────────────────────────────────

def config_path(kb_root: Path) -> Path:
    return kb_root / "config" / "config.yaml"


def domain_config_path(kb_root: Path) -> Path:
    return kb_root / "config" / "domain.yaml"


def raw_dir(kb_root: Path, sub: str = "") -> Path:
    return kb_root / "raw" / sub if sub else kb_root / "raw"


def notes_dir(kb_root: Path) -> Path:
    return kb_root / "notes"


def proposals_dir(kb_root: Path) -> Path:
    return kb_root / "proposals"


def projects_dir(kb_root: Path) -> Path:
    return kb_root / "projects"


# ── Environment ──────────────────────────────────────────────────────────────

def load_env(kb_root: Path) -> None:
    """Load .env from the KB root if present."""
    env_file = kb_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(kb_root: Path) -> dict:
    """Load config.yaml from the KB root.

    Stores ``_kb_root`` (str) inside the returned dict so downstream code
    that already receives ``cfg`` can recover the KB root path.
    """
    load_env(kb_root)
    with open(config_path(kb_root)) as f:
        cfg = yaml.safe_load(f)
    cfg["_kb_root"] = str(kb_root)
    return cfg


def kb_root_from_cfg(cfg: dict) -> Path:
    """Extract the KB root Path stored by load_config."""
    return Path(cfg["_kb_root"])
