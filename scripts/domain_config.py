"""
domain_config.py — Load domain configuration from config/domain.yaml.

Provides domain keywords, relation ontology, entity types, and LLM prompts
that are customized per-deployment. All domain-specific intelligence lives
in config/domain.yaml — no hardcoded domain knowledge in the codebase.
"""

from pathlib import Path

import yaml

from kb_root import domain_config_path

# Module-level cache keyed by kb_root to avoid re-reading per call.
_cache: dict[Path, dict] = {}


def load_domain_config(kb_root: Path) -> dict:
    if kb_root not in _cache:
        with open(domain_config_path(kb_root)) as f:
            _cache[kb_root] = yaml.safe_load(f)
    return _cache[kb_root]


def get_domains(kb_root: Path) -> dict:
    """Return the domains dict from config."""
    return load_domain_config(kb_root).get("domains", {})


def get_entity_types(kb_root: Path) -> set[str]:
    """Return the set of valid entity types."""
    return set(load_domain_config(kb_root).get("entity_types", ["concept"]))


def get_domain_keywords(kb_root: Path) -> dict[str, list[str]]:
    """Return {domain_name: [keywords]} for query routing."""
    domains = get_domains(kb_root)
    return {
        name: cfg.get("keywords", [])
        for name, cfg in domains.items()
        if cfg.get("keywords")
    }


def get_relation_ontology(kb_root: Path) -> dict[str, list[str]]:
    """Return {domain_name: [RELATION_TYPE, ...]} for fact extraction."""
    domains = get_domains(kb_root)
    return {
        name: cfg.get("relations", [])
        for name, cfg in domains.items()
    }


def get_relations_for_domain(kb_root: Path, domain: str) -> list[str]:
    """Return allowed relation types for a given domain.

    Merges the domain's own relations with the fallback 'general' relations.
    If domain is not found, returns all relations from all domains.
    """
    ontology = get_relation_ontology(kb_root)
    general = ontology.get("general", ["RELATED_TO"])

    if domain in ontology:
        return ontology[domain] + [r for r in general if r not in ontology[domain]]

    # Unknown domain: return union of all
    all_rels = []
    seen = set()
    for rels in ontology.values():
        for r in rels:
            if r not in seen:
                all_rels.append(r)
                seen.add(r)
    return all_rels


def get_prompts(kb_root: Path) -> dict:
    """Return the prompts section from domain config."""
    return load_domain_config(kb_root).get("prompts", {})


def detect_domain(kb_root: Path, query: str) -> str | None:
    """Auto-detect domain from query text using keyword matching.

    Returns domain name or None (meaning cross/no strong signal).
    """
    keywords = get_domain_keywords(kb_root)
    if not keywords:
        return None

    q_lower = query.lower()
    scores = {}
    for domain, kws in keywords.items():
        score = sum(1 for kw in kws if kw in q_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return None

    top_domain, top_score = max(scores.items(), key=lambda x: x[1])
    if list(scores.values()).count(top_score) > 1:
        return None  # tied
    return top_domain
