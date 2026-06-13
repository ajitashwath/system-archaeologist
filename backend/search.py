from __future__ import annotations

import os
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def build_queries(system_name: str) -> list[str]:
    return [
        f"{system_name} architecture how it works",
        f"{system_name} tech stack technology behind",
        f"{system_name} engineering blog system design",
        f"{system_name} open source github infrastructure",
    ]


def gather_evidence(system_name: str, max_results_per_query: int = 4) -> list[str]:
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — skipping web search, LLM will use training knowledge.")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not installed. Run: pip install tavily-python")
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    queries = build_queries(system_name)
    evidence: list[str] = []
    seen_urls: set[str] = set()

    for query in queries:
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_results_per_query,
                include_answer=False,
            )
            results: list[dict[str, Any]] = response.get("results", [])

            for result in results:
                url = result.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = result.get("title", "").strip()
                content = result.get("content", "").strip()

                if content:

                    snippet = content[:300].replace("\n", " ").strip()
                    if title:
                        evidence.append(f"[{title}] {snippet} (source: {url})")
                    else:
                        evidence.append(f"{snippet} (source: {url})")

        except Exception as exc:
            logger.warning("Tavily query failed for '%s': %s", query, exc)
            continue

    logger.info("Gathered %d evidence items for '%s'", len(evidence), system_name)
    return evidence
