from ddgs import DDGS
import concurrent.futures
from typing import Dict, Any, List

# Local imports (assuming these are available in your project structure)
from app.llm import call_llm

def run_web_search_agent(claim: str, provider: str, config: Dict[str, Any]) -> str:
    # 1. Generate an optimized search query using the LLM
    query_prompt = f"""
    Task: Convert the following factual claim into a simple, effective web search query to verify it.
    Claim: "{claim}"
    Output: Return ONLY the search query string. No quotes, no explanations.
    """
    try:
        search_query = call_llm(query_prompt, provider, config).strip().replace('"', '')
    except Exception as e:
        # Fallback to claim if LLM fails
        search_query = claim

    formatted_output = f"Search Query Used: {search_query}\n\n"
    aggregated_results = []
    seen_urls = set()

    def perform_search(search_type: str, **kwargs):
        """Helper to run a specific DDGS search safely."""
        try:
            ddgs = DDGS()
            if search_type == "text":
                return "Web", ddgs.text(search_query, **kwargs)
            elif search_type == "news":
                return "News", ddgs.news(search_query, **kwargs)
            elif search_type == "wiki":
                # Specifically targeting Wikipedia via text backend if supported, 
                # or just appending 'site:wikipedia.org' to query
                # The docs mentioned 'wikipedia' backend for text()
                return "Wikipedia", ddgs.text(search_query, backend="wikipedia", **kwargs)
        except Exception:
            return search_type, []

    # 2. Execute searches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            # Standard Web Search
            executor.submit(perform_search, "text", max_results=3),
            # News Search
            executor.submit(perform_search, "news", max_results=2),
            # Wikipedia Search
            executor.submit(perform_search, "wiki", max_results=1)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                label, results = future.result()
                if results:
                    for r in results:
                        # Normalize result keys
                        url = r.get('href', r.get('url'))
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('abstract', '')) # 'abstract' is common in wiki results
                        
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            aggregated_results.append(f"[{label}] {title}: {body} (Source: {url})")
            except Exception:
                continue

    if not aggregated_results:
        return f"No relevant results found for query: {search_query}"
        
    return formatted_output + "\n".join(aggregated_results)
