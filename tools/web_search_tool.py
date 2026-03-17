"""
tools/web_search_tool.py
------------------------
Tool 2 — Web Medical Guideline Search Tool

Uses the `duckduckgo-search` library (free, no API key required) to retrieve
real web search results from trusted medical sources:
  - WHO         (who.int)
  - CDC         (cdc.gov)
  - Mayo Clinic (mayoclinic.org)
  - MedlinePlus (medlineplus.gov)

Unlike the DuckDuckGo Instant Answer API, this library scrapes the actual
search results page, returning the same snippets a human would read.
"""

from __future__ import annotations

from typing import List, Dict
from langchain.tools import Tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trusted medical domains — results are biased toward these in the query
TRUSTED_DOMAINS = [
    "who.int",
    "cdc.gov",
    "mayoclinic.org",
    "medlineplus.gov",
]

MAX_RESULTS = 3          # how many snippets to return to the agent
SNIPPET_MAX_CHARS = 250  # max chars per snippet (keeps token usage low)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_biased_query(query: str) -> str:
    """Append a site-filter to steer results toward trusted medical domains."""
    site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS)
    return f"{query} ({site_filter})"


def _search_duckduckgo(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Run a real DuckDuckGo text search using the `duckduckgo-search` library.

    Returns a list of dicts with keys: title, snippet, url.
    Falls back to an informative placeholder if the search fails or returns nothing.
    """
    try:
        from ddgs import DDGS  # lazy import — fails loudly if missing
    except ImportError:
        return [{
            "title": "Missing dependency",
            "snippet": (
                "The `ddgs` package is not installed. "
                "Run: pip install ddgs"
            ),
            "url": "",
        }]

    results: List[Dict] = []

    try:
        ddgs = DDGS(verify=False)
        raw = ddgs.text(query, max_results=max_results)
        for item in raw:
            snippet = (item.get("body") or "")[:SNIPPET_MAX_CHARS]
            results.append({
                "title": item.get("title", "Untitled"),
                "snippet": snippet,
                "url": item.get("href", ""),
            })
    except Exception as exc:
        results.append({
            "title": "Search error",
            "snippet": f"DuckDuckGo search failed: {exc}",
            "url": "",
        })

    # Fallback if no results were returned
    if not results:
        safe_q = query.replace(" ", "+")
        results.append({
            "title": "No results found",
            "snippet": (
                "No results were returned. Search trusted sources directly:\n"
                f"  • WHO: https://www.who.int/search?query={safe_q}\n"
                f"  • CDC: https://search.cdc.gov/search/?query={safe_q}\n"
                f"  • Mayo Clinic: https://www.mayoclinic.org/search/search-results?q={safe_q}\n"
                f"  • MedlinePlus: https://medlineplus.gov/search/?query={safe_q}"
            ),
            "url": "",
        })

    return results


# ---------------------------------------------------------------------------
# Tool entry point (called by the LangChain ReAct agent)
# ---------------------------------------------------------------------------

def _run_web_search(query: str) -> str:
    """Format search results into a string the agent can reason over."""
    biased_query = _build_biased_query(query)
    hits = _search_duckduckgo(biased_query)

    lines = [f"Web Medical Search Results for: '{query}'\n"]
    for i, hit in enumerate(hits, 1):
        lines.append(
            f"[{i}] {hit['title']}\n"
            f"    Snippet : {hit['snippet']}\n"
            f"    URL     : {hit['url']}\n"
        )

    lines.append(
        "\n⚠️  Always verify information on official medical websites. "
        "This tool provides general guidance, NOT medical diagnosis."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain Tool object (imported by tools/__init__.py)
# ---------------------------------------------------------------------------

web_search_tool = Tool(
    name="WebMedicalSearch",
    func=_run_web_search,
    description=(
        "Search trusted medical web sources (WHO, CDC, Mayo Clinic, MedlinePlus) "
        "for treatment guidelines, prevention advice, and when to see a doctor. "
        "Input should be a clear medical query string "
        "(e.g. 'WHO flu treatment guidelines adults'). "
        "Use this tool AFTER searching the local healthcare database to find "
        "up-to-date official recommendations not in the local dataset."
    ),
)
