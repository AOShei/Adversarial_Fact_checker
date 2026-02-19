import asyncio
import time
import urllib.parse
import concurrent.futures
from functools import partial
from typing import Dict, Any, List, Tuple, Optional

from app.llm import call_llm
from app.logger import get_logger

logger = get_logger(__name__)

# ── Fix macOS Python SSL certificate issue ──
# truststore makes Python use the OS certificate store (Keychain on macOS).
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

SEARCH_TIMEOUT_SECONDS = 20   # per-search timeout (seconds)
MAX_TEXT_RESULTS = 5
MAX_WIKI_RESULTS = 2
MAX_QUERIES_PER_CLAIM = 2
MAX_RESULTS_RETURNED = 12

# Shared thread pool for blocking search calls
_search_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=6, thread_name_prefix="websearch"
)

# ── Global concurrency limiter for search requests ──
_search_semaphore: Optional[asyncio.Semaphore] = None
_search_semaphore_loop_id: Optional[int] = None
_MAX_CONCURRENT_SEARCHES = 4


def _get_search_semaphore() -> asyncio.Semaphore:
    """Lazily create the semaphore on the running event loop."""
    global _search_semaphore, _search_semaphore_loop_id
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None
    if _search_semaphore is None or _search_semaphore_loop_id != loop_id:
        _search_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
        _search_semaphore_loop_id = loop_id
        logger.debug("Created search semaphore (max=%d, loop=%s)", _MAX_CONCURRENT_SEARCHES, loop_id)
    return _search_semaphore


# ---------------------------------------------------------------------------
# Blocking search helpers (run inside the thread pool)
# ---------------------------------------------------------------------------

def _ddg_html_search(query: str, max_results: int = 5) -> List[dict]:
    """Search DuckDuckGo via the lightweight HTML-only interface (no JS).

    This avoids the curl_cffi / impersonation issues of the `ddgs` library
    and reliably returns results with title, URL, and snippet.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/17.0 Safari/605.1.15"
                )
            },
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results: List[dict] = []
        for div in soup.select("div.result, div.web-result"):
            a = div.find("a", class_="result__a") or div.find("a", href=True)
            if not a:
                continue
            href = str(a.get("href", ""))
            # DDG wraps URLs in a tracking redirect; extract the real URL
            if "uddg=" in href:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = urllib.parse.unquote(parsed.get("uddg", [href])[0])
            if not href.startswith("http"):
                continue
            title = a.get_text(strip=True)
            snippet_el = (
                div.find("a", class_="result__snippet")
                or div.find("div", class_="result__snippet")
            )
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append({"title": title, "url": href, "body": snippet[:500]})
            if len(results) >= max_results:
                break
        return results
    except Exception as exc:
        logger.warning("DDG HTML search failed | query='%s' error=%s", query[:60], exc)
        return []


def _wikipedia_search(query: str, max_results: int = 2) -> List[dict]:
    """Search Wikipedia for summaries (blocking)."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        titles = wikipedia.search(query, results=max_results + 3)
        results = []
        for title in titles:
            if len(results) >= max_results:
                break
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({
                    "title": page.title,
                    "url":   page.url,
                    "body":  page.summary[:500],
                })
            except Exception:
                continue
        return results
    except Exception as exc:
        logger.warning("Wikipedia search failed | query='%s' error=%s", query[:60], exc)
        return []


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

async def _run_search_async(
    label: str, fn, query: str, **kwargs
) -> Tuple[str, str, list]:
    """Run a blocking search function in the thread pool with semaphore + timeout."""
    sem = _get_search_semaphore()
    short_q = query[:60]
    try:
        async with sem:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(_search_pool, partial(fn, query, **kwargs)),
                timeout=SEARCH_TIMEOUT_SECONDS,
            )
            logger.debug("%s search done | query='%s' results=%d", label, short_q, len(result))
            return label, query, result
    except asyncio.TimeoutError:
        logger.warning("%s search timed out (%ds) | query='%s'", label, SEARCH_TIMEOUT_SECONDS, short_q)
        return label, query, []
    except Exception as exc:
        logger.warning("%s search error | query='%s' error=%s", label, short_q, exc)
        return label, query, []


# ---------------------------------------------------------------------------
# Public agent entry-point
# ---------------------------------------------------------------------------

async def run_web_search_agent(claim: str, provider: str, config: Dict[str, Any]) -> str:
    start = time.perf_counter()
    claim_short = claim[:80]
    logger.info("Web search start | claim='%s'", claim_short)

    # 1. Generate an optimized search query using the LLM
    query_prompt = f"""
    Task: Convert the following factual claim into a simple, effective web search query to verify it.

    Rules:
    - Strip any framing like "The webpage states", "The report says", "According to the article" etc.
    - Focus on the core factual assertion and key entities/numbers/dates.
    - Use plain keywords, not full sentences.
    - Do NOT include words like "webpage", "article", "report", "states", "claims".

    Claim: "{claim}"
    Output: Return ONLY the search query string. No quotes, no explanations.
    """
    try:
        search_query = (await call_llm(query_prompt, provider, config)).strip().replace('"', '')
        logger.info("Web search query ready | claim='%s' query='%s' elapsed=%.2fs",
                     claim_short, search_query[:60], time.perf_counter() - start)
    except Exception as e:
        logger.warning("Web search query generation failed | claim='%s' error=%s", claim_short, e)
        search_query = claim

    # Build query candidates
    query_candidates = [search_query]
    if claim.strip() and claim.strip().lower() != search_query.strip().lower():
        query_candidates.append(claim.strip())
    query_candidates = query_candidates[:MAX_QUERIES_PER_CLAIM]

    formatted_output = f"Search Query Used: {search_query}\n\n"
    if len(query_candidates) > 1:
        formatted_output += f"Expanded Query Used: {query_candidates[1]}\n\n"

    # 2. Dispatch DuckDuckGo HTML + Wikipedia searches concurrently
    tasks = []
    for query in query_candidates:
        tasks.append(_run_search_async("Web", _ddg_html_search, query, max_results=MAX_TEXT_RESULTS))
        tasks.append(_run_search_async("Wikipedia", _wikipedia_search, query, max_results=MAX_WIKI_RESULTS))

    logger.info("Web search dispatching %d tasks (DDG HTML + Wikipedia) | claim='%s'", len(tasks), claim_short)
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Aggregate & deduplicate results
    aggregated_results: List[str] = []
    seen_urls: set = set()

    for result in results_list:
        if isinstance(result, BaseException):
            logger.warning("Search task exception: %s", result)
            continue
        label, used_query, search_results = result
        for r in search_results:
            url = r.get("url", "")
            title = r.get("title", "No Title")
            body = r.get("body", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                aggregated_results.append(f"[{label}] {title}: {body} (Source: {url})")
                if len(aggregated_results) >= MAX_RESULTS_RETURNED:
                    break
        if len(aggregated_results) >= MAX_RESULTS_RETURNED:
            break

    elapsed = time.perf_counter() - start
    logger.info(
        "Web search done  | claim='%s' results=%d elapsed=%.2fs",
        claim_short, len(aggregated_results), elapsed,
    )

    if not aggregated_results:
        return f"No relevant results found for query: {search_query}"

    return formatted_output + "\n".join(aggregated_results)
