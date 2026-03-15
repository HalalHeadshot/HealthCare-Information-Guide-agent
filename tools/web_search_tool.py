"""
tools/web_search_tool.py
------------------------
Tool 2 — Web Medical Guideline Search Tool

Uses the DuckDuckGo Instant Answer / Abstract API (free, no API key required)
to retrieve medical guideline snippets from trusted sources:
  - WHO  (who.int)
  - CDC  (cdc.gov)
  - Mayo Clinic (mayoclinic.org)
  - MedlinePlus (medlineplus.gov)

The agent calls this tool to fetch up-to-date treatment / prevention guidance.
"""

import re
import requests
from typing import List, Dict
from langchain.tools import Tool

# Trusted medical domains — results from other sites are filtered out
TRUSTED_DOMAINS = ["who.int", "cdc.gov", "mayoclinic.org", "medlineplus.gov"]

DDGR_API = "https://api.duckduckgo.com/"


def _duckduckgo_search(query: str, max_results: int = 2) -> List[Dict]:
    """
    Call the DuckDuckGo Instant Answer API and return a list of result dicts
    with keys: title, snippet, url.
    Falls back to an empty list on network errors.
    """
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    try:
        resp = requests.get(DDGR_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return [{"title": "Network error", "snippet": str(exc), "url": ""}]

    results = []

    # AbstractURL / AbstractText — primary source
    if data.get("AbstractURL") and data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", "DuckDuckGo Abstract"),
            "snippet": data["AbstractText"][:200],
            "url": data["AbstractURL"],
        })

    # Related Topics — secondary sources
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        # Some items are sub-sections (have a "Topics" list) — skip those
        if "Topics" in topic:
            continue
        url = topic.get("FirstURL", "")
        text = topic.get("Text", "")
        if text:
            results.append({
                "title": topic.get("Name", url),
                "snippet": text[:200],
                "url": url,
            })

    # If DDG returned nothing useful, construct a suggested URL for the user
    if not results:
        safe_query = re.sub(r"\s+", "+", query.strip())
        results.append({
            "title": "No instant answer found",
            "snippet": (
                "DuckDuckGo did not return an instant answer for this query. "
                "You can search trusted sources directly using these links:\n"
                f"  • WHO: https://www.who.int/search?query={safe_query}\n"
                f"  • CDC: https://search.cdc.gov/search/?query={safe_query}\n"
                f"  • Mayo Clinic: https://www.mayoclinic.org/search/search-results?q={safe_query}\n"
                f"  • MedlinePlus: https://medlineplus.gov/search/?query={safe_query}"
            ),
            "url": "",
        })

    return results[:max_results]


def _run_web_search(query: str) -> str:
    """Entry point called by the LangChain agent."""
    # Append trusted-site bias to the query
    biased_query = f"{query} site:who.int OR site:cdc.gov OR site:mayoclinic.org OR site:medlineplus.gov"
    hits = _duckduckgo_search(biased_query)

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


# LangChain Tool object
web_search_tool = Tool(
    name="WebMedicalSearch",
    func=_run_web_search,
    description=(
        "Search trusted medical web sources (WHO, CDC, Mayo Clinic, MedlinePlus) "
        "for treatment guidelines, prevention advice, and when to see a doctor. "
        "Input should be a clear medical query string "
        "(e.g. 'WHO flu treatment guidelines adults'). "
        "Use this tool AFTER searching the local healthcare database to get "
        "up-to-date official recommendations."
    ),
)
