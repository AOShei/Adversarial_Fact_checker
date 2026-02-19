import asyncio
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from openai import AsyncAzureOpenAI
from app.logger import get_logger

logger = get_logger(__name__)

# Per-LLM-call timeout in seconds (prevents indefinite hangs)
LLM_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Singleton async clients – created once, reused across all concurrent calls
# ---------------------------------------------------------------------------
_azure_client: Optional[AsyncAzureOpenAI] = None
_azure_client_key: Optional[str] = None  # track config to detect changes
_azure_client_loop_id: Optional[int] = None  # track event loop identity


def _normalize_azure_endpoint(endpoint: str) -> str:
    """Normalize Azure endpoint to resource root (scheme + host + '/')."""
    raw_endpoint = (endpoint or "").strip()
    if not raw_endpoint:
        return ""
    parsed = urlparse(raw_endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/"
    return raw_endpoint.rstrip("/") + "/"


def _get_azure_client(config: Dict[str, Any]) -> AsyncAzureOpenAI:
    """Return a reusable async Azure client, constructing one if needed.

    The client is recreated when the running event loop changes
    (e.g. a new ``asyncio.run()`` call) to avoid stale connections.
    """
    global _azure_client, _azure_client_key, _azure_client_loop_id
    cache_key = f"{config.get('azure_endpoint')}|{config.get('azure_key')}|{config.get('azure_version')}"

    try:
        current_loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        current_loop_id = None

    needs_rebuild = (
        _azure_client is None
        or _azure_client_key != cache_key
        or _azure_client_loop_id != current_loop_id
    )
    if needs_rebuild:
        endpoint = _normalize_azure_endpoint(config["azure_endpoint"])
        _azure_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=config["azure_key"],
            api_version=config["azure_version"],
        )
        _azure_client_key = cache_key
        _azure_client_loop_id = current_loop_id
        logger.info("Created new AsyncAzureOpenAI client for %s (loop=%s)", endpoint, current_loop_id)
    assert _azure_client is not None
    return _azure_client


def _extract_responses_text(response: Any) -> str:
    """Extract plain text from Responses API objects across SDK shapes."""
    direct_text = getattr(response, "output_text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        text_parts: list[str] = []
        for item in output_items:
            contents = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(contents, list):
                continue
            for content in contents:
                content_type = content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
                if content_type in {"output_text", "text"}:
                    text_value = content.get("text") if isinstance(content, dict) else getattr(content, "text", None)
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts)

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    return str(response)


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def call_llm(prompt: str, provider: str, config: Dict[str, Any]) -> str:
    """Async LLM call with timeout, logging, and fail-soft error handling."""

    prompt_len = len(prompt)
    start = time.perf_counter()
    logger.info("LLM call start | provider=%s prompt_len=%d", provider, prompt_len)

    if provider == "Google Gemini":
        return await _call_gemini(prompt, config, start)
    elif provider == "Microsoft Azure":
        return await _call_azure(prompt, config, start)
    else:
        logger.error("Invalid provider: %s", provider)
        return "Error: Invalid Provider"


async def _call_gemini(prompt: str, config: Dict[str, Any], start: float) -> str:
    """Gemini path – kept for compatibility but Azure is primary."""
    api_key = config.get("gemini_key")
    if not api_key:
        return "Error: Gemini API Key missing."
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_name = config.get("gemini_model", "gemini-2.5-flash")

        # google-genai sync SDK – run in thread with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - start
        text = response.text or ""
        logger.info("LLM call done  | provider=Gemini elapsed=%.2fs resp_len=%d", elapsed, len(text))
        return text
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.warning("LLM TIMEOUT | provider=Gemini elapsed=%.2fs", elapsed)
        return f"Gemini Error: Request timed out after {LLM_TIMEOUT_SECONDS}s"
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("LLM ERROR | provider=Gemini elapsed=%.2fs error=%s", elapsed, e, exc_info=True)
        return f"Gemini Error: {str(e)}"


async def _call_azure(prompt: str, config: Dict[str, Any], start: float) -> str:
    """Azure OpenAI path – async SDK with timeout."""
    required = ["azure_endpoint", "azure_key", "azure_version", "azure_deployment"]
    if not all(config.get(k) for k in required):
        return "Error: Missing Azure configuration details."

    client = _get_azure_client(config)
    deployment_name = config["azure_deployment"]
    system_prompt = "You are a precise analytical agent. Return only the requested information."

    # Try Responses API first (required for GPT-5.x deployments)
    try:
        responses_result = await asyncio.wait_for(
            client.responses.create(
                model=deployment_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        text = _extract_responses_text(responses_result).strip()
        if text:
            elapsed = time.perf_counter() - start
            logger.info(
                "LLM call done  | provider=Azure(Responses) elapsed=%.2fs resp_len=%d",
                elapsed, len(text),
            )
            return text
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.warning("LLM TIMEOUT | provider=Azure(Responses) elapsed=%.2fs", elapsed)
        return f"Azure Error: Request timed out after {LLM_TIMEOUT_SECONDS}s"
    except Exception as responses_error:
        responses_error_message = str(responses_error)
        logger.warning("Azure Responses API failed, falling back to Chat Completions: %s", responses_error_message)
        if (
            "Responses API is enabled only for api-version 2025-03-01-preview and later" in responses_error_message
            and deployment_name.lower().startswith("gpt-5")
        ):
            return "Azure Error: GPT-5 deployment requires AZURE_OPENAI_API_VERSION=2025-03-01-preview or later."

    # Fallback: Chat Completions API
    try:
        chat_result = await asyncio.wait_for(
            client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        text = chat_result.choices[0].message.content or ""
        elapsed = time.perf_counter() - start
        logger.info(
            "LLM call done  | provider=Azure(ChatCompletions) elapsed=%.2fs resp_len=%d",
            elapsed, len(text),
        )
        return text
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.warning("LLM TIMEOUT | provider=Azure(ChatCompletions) elapsed=%.2fs", elapsed)
        return f"Azure Error: Request timed out after {LLM_TIMEOUT_SECONDS}s"
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("LLM ERROR | provider=Azure elapsed=%.2fs error=%s", elapsed, e, exc_info=True)
        return f"Azure Error: {str(e)}"
