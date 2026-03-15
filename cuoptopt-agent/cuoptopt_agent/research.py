"""Literature research agent.

Searches Google Scholar (via ``scholarly``) and arxiv in parallel, returning
a ranked list of paper summaries for injection into the implementation prompt.

Rate-limit handling:
- scholarly: exponential backoff with jitter; falls back gracefully if blocked.
- arxiv: official API with configurable page size.
"""

from __future__ import annotations

import concurrent.futures
import time
import random
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str
    source: str  # "scholar" | "arxiv"


# ---------------------------------------------------------------------------
# Google Scholar
# ---------------------------------------------------------------------------

def _search_scholar(query: str, max_results: int, retries: int = 3) -> list[Paper]:
    try:
        from scholarly import scholarly  # type: ignore[import]
    except ImportError:
        logger.warning("scholarly not installed; skipping Google Scholar search.")
        return []

    papers: list[Paper] = []
    delay = 2.0

    for attempt in range(retries):
        try:
            results = scholarly.search_pubs(query)
            for _ in range(max_results):
                try:
                    pub = next(results)
                    bib = pub.get("bib", {})
                    papers.append(Paper(
                        title=bib.get("title", "Unknown"),
                        authors=bib.get("author", []),
                        year=bib.get("pub_year"),
                        abstract=bib.get("abstract", ""),
                        url=pub.get("pub_url", ""),
                        source="scholar",
                    ))
                except StopIteration:
                    break
            return papers
        except Exception as exc:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Scholar attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                logger.warning("Scholar search failed after %d attempts: %s", retries, exc)
                return papers

    return papers


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def _search_arxiv(query: str, max_results: int) -> list[Paper]:
    try:
        import arxiv  # type: ignore[import]
    except ImportError:
        logger.warning("arxiv not installed; skipping arxiv search.")
        return []

    papers: list[Paper] = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in search.results():
            papers.append(Paper(
                title=result.title,
                authors=[str(a) for a in result.authors],
                year=result.published.year if result.published else None,
                abstract=result.summary,
                url=result.entry_id,
                source="arxiv",
            ))
    except Exception as exc:
        logger.warning("arxiv search failed: %s", exc)

    return papers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_literature(
    query: str,
    max_scholar: int = 5,
    max_arxiv: int = 5,
) -> list[Paper]:
    """Search Google Scholar and arxiv in parallel; return merged results.

    Results are deduplicated by title (case-insensitive) and ordered
    Scholar-first, then arxiv.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_scholar = ex.submit(_search_scholar, query, max_scholar)
        f_arxiv = ex.submit(_search_arxiv, query, max_arxiv)
        scholar_results = f_scholar.result()
        arxiv_results = f_arxiv.result()

    seen_titles: set[str] = set()
    merged: list[Paper] = []
    for p in scholar_results + arxiv_results:
        key = p.title.lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            merged.append(p)

    return merged


def format_papers_for_prompt(papers: list[Paper]) -> str:
    """Render papers as a numbered markdown list for LLM injection."""
    if not papers:
        return "_No relevant literature found._"

    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_str += " et al."
        year_str = f" ({p.year})" if p.year else ""
        lines.append(
            f"{i}. **{p.title}**{year_str} — {authors_str}\n"
            f"   URL: {p.url}\n"
            f"   Abstract: {p.abstract[:400]}{'...' if len(p.abstract) > 400 else ''}"
        )

    return "\n\n".join(lines)
