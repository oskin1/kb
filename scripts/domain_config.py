"""
domain_config.py — Load domain configuration from config/domain.yaml.

Provides domain keywords, relation ontology, entity types, and LLM prompts
that are customized per-deployment. All domain-specific intelligence lives
in config/domain.yaml — no hardcoded domain knowledge in the codebase.
"""

from pathlib import Path
from functools import lru_cache

import yaml

DOMAIN_CONFIG_PATH = Path(__file__).parent.parent / "config" / "domain.yaml"


@lru_cache(maxsize=1)
def load_domain_config() -> dict:
    with open(DOMAIN_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_domains() -> dict:
    """Return the domains dict from config."""
    return load_domain_config().get("domains", {})


def get_entity_types() -> set[str]:
    """Return the set of valid entity types."""
    return set(load_domain_config().get("entity_types", ["concept"]))


def get_domain_keywords() -> dict[str, list[str]]:
    """Return {domain_name: [keywords]} for query routing."""
    domains = get_domains()
    return {
        name: cfg.get("keywords", [])
        for name, cfg in domains.items()
        if cfg.get("keywords")
    }


def get_relation_ontology() -> dict[str, list[str]]:
    """Return {domain_name: [RELATION_TYPE, ...]} for fact extraction."""
    domains = get_domains()
    return {
        name: cfg.get("relations", [])
        for name, cfg in domains.items()
    }


def get_relations_for_domain(domain: str) -> list[str]:
    """Return allowed relation types for a given domain.

    Merges the domain's own relations with the fallback 'general' relations.
    If domain is not found, returns all relations from all domains.
    """
    ontology = get_relation_ontology()
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


def get_prompts() -> dict:
    """Return the prompts section from domain config."""
    return load_domain_config().get("prompts", {})


def detect_domain(query: str) -> str | None:
    """Auto-detect domain from query text using keyword matching.

    Returns domain name or None (meaning cross/no strong signal).
    """
    keywords = get_domain_keywords()
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
