"""Keyword-to-skill mapper.

Searches two skill directories:
1. ``cuoptopt-agent/skills/``  — optimization-domain skills (GPU arch, CUDA, algorithms)
2. ``skills/``                  — existing cuOpt usage/developer skills

Ranking uses TF-IDF cosine similarity between the query and each skill's
``description`` field (from YAML front-matter) plus the first 200 chars of
its body.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import NamedTuple


class LoadedSkill(NamedTuple):
    name: str
    path: Path
    score: float
    content: str


# ---------------------------------------------------------------------------
# YAML front-matter helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _extract_description(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text[:300]
    fm = m.group(1)
    for line in fm.splitlines():
        if line.lower().startswith("description:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return text[:300]


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "is",
    "are", "with", "when", "use", "no", "api", "skill",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = max(len(tokens), 1)
    return {t: c / n for t, c in counts.items()}


def _idf(docs: list[list[str]]) -> dict[str, float]:
    N = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (df_t + 1)) + 1 for t, df_t in df.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_skills(
    query: str,
    repo_root: Path,
    top_n: int = 6,
) -> list[LoadedSkill]:
    """Return the top-N most relevant skills for *query*.

    Search order (earlier dirs shadow later ones if names collide):
    1. ``<repo_root>/cuoptopt-agent/skills/``
    2. ``<repo_root>/skills/``
    """
    skill_dirs = [
        repo_root / "cuoptopt-agent" / "skills",
        repo_root / "skills",
    ]

    skill_files: list[Path] = []
    for sdir in skill_dirs:
        if sdir.is_dir():
            skill_files.extend(sorted(sdir.glob("*/SKILL.md")))

    if not skill_files:
        return []

    # Build corpus text: description + first 200 chars of body
    corpus_texts: list[str] = []
    for p in skill_files:
        raw = p.read_text(encoding="utf-8", errors="replace")
        desc = _extract_description(raw)
        body_start = raw[raw.find("---", 3) + 3 :] if "---" in raw[3:] else raw
        corpus_texts.append(desc + " " + body_start[:200])

    query_tokens = _tokenize(query)
    corpus_tokens = [_tokenize(t) for t in corpus_texts]

    idf = _idf(corpus_tokens + [query_tokens])
    query_tfidf = {t: tf * idf.get(t, 1.0) for t, tf in _tf(query_tokens).items()}

    results: list[LoadedSkill] = []
    for path, tokens, raw_text in zip(skill_files, corpus_tokens, corpus_texts):
        doc_tfidf = {t: tf * idf.get(t, 1.0) for t, tf in _tf(tokens).items()}
        score = _cosine(query_tfidf, doc_tfidf)
        content = path.read_text(encoding="utf-8", errors="replace")
        results.append(LoadedSkill(
            name=path.parent.name,
            path=path,
            score=score,
            content=content,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


def format_skills_for_prompt(skills: list[LoadedSkill]) -> str:
    """Return a formatted string of all loaded skill contents for injection into an LLM prompt."""
    parts: list[str] = []
    for skill in skills:
        parts.append(f"### Skill: {skill.name}\n\n{skill.content}")
    return "\n\n---\n\n".join(parts)
