# utils/web_search.py

"""
Simple web search interface — OPTIONAL.
If no SERPAPI_KEY is provided, returns empty results.
"""

import logging
import requests
from config.config import SERPAPI_KEY


def serpapi_search(query: str, num_results: int = 5) -> list:
    """
    Run a Google search via SerpAPI. Returns a list of {title, snippet, link}.
    If no API key → returns empty list.
    """
    results = []
    try:
        if not SERPAPI_KEY or SERPAPI_KEY == "YOUR_SERPAPI_KEY":
            logging.warning("SERPAPI_KEY missing — skipping live web search.")
            return results

        resp = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "engine": "google", "api_key": SERPAPI_KEY},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic_results", [])
        for item in organic[:num_results]:
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link"),
            })
    except Exception as e:
        logging.exception("SerpAPI search failed: %s", e)

    return results


def summarize_web_results(results: list, llm_call) -> str:
    """
    Use LLM to summarize search output.
    """
    if not results:
        return ""

    prompt = "Summarize the search results below in 3–4 sentences:\n\n"

    for r in results:
        prompt += f"- {r['title']} → {r['snippet']}\n"

    try:
        return llm_call(prompt, max_tokens=200, temperature=0.0)
    except Exception as e:
        logging.exception("LLM summarization failed: %s", e)
        return ""
