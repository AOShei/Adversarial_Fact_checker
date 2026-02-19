import json
import time
from typing import List, Dict, Any
from app.llm import call_llm
from app.logger import get_logger

logger = get_logger(__name__)


async def run_factual_claim_extractor(report: str, provider: str, config: Dict[str, Any]) -> List[str]:
    prompt = f"""
    Analyze the following report and extract a list of ALL distinct, checkable factual claims.

    CRITICAL INSTRUCTION: EXTRACT VERIFIABLE FACTS
    - Extract the SUBSTANCE of each factual claim as a standalone, verifiable assertion.
    - Do NOT prefix claims with "The report states", "The webpage says", "The article claims", "According to the document", or any similar source-attribution framing.
    - Each claim should read as a direct factual statement that can be independently checked.

    FORMATTING RULES:
    - WRONG: "The webpage states that RCP8.5 is the highest emission scenario."
    - RIGHT: "RCP8.5 is the highest emission scenario among the Representative Concentration Pathways."
    - WRONG: "The article says New Zealand's average temperature has risen by 1.1°C."
    - RIGHT: "New Zealand's average temperature has risen by 1.1°C since 1909."

    ADDITIONAL GUIDELINES:
    - **Resolve Pronouns & Entities:** Replace all pronouns with the named entity they refer to. Include dates, locations, and specifics from the source text.
    - **Split Attributed Claims:** If someone is quoted making a factual claim, create TWO claims:
      1. Attribution: "Donald Trump stated that dangerous criminals have entered the US."
      2. Substance: "Dangerous criminals from prisons and mental institutions have illegally entered the US."
    - **Expand Abbreviations:** If the document defines an acronym, include the full term in the claim. E.g. "Representative Concentration Pathway RCP4.5 could be realistic if immediate global action is taken to mitigate climate change."
    - **Make Claims Search-Ready:** Include enough context (subject, entity, domain, timeframe) so each claim can be independently searched and verified.
    - Do NOT extract meta-claims about the document itself (e.g. "The page links to a PDF" or "The article discusses climate change").
    - Only extract claims that assert something factually verifiable about the real world.

    Return ONLY a raw JSON list of strings. Do not use Markdown formatting.

    Report:
    {report[:5000]}
    """
    start = time.perf_counter()
    logger.info("Claim extraction started | report_len=%d", len(report))

    response = await call_llm(prompt, provider, config)
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    try:
        claims = json.loads(clean_resp)
        elapsed = time.perf_counter() - start
        logger.info("Claim extraction done | claims=%d elapsed=%.2fs", len(claims), elapsed)
        return claims
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("Claim extraction parse failure | elapsed=%.2fs error=%s", elapsed, exc)
        return ["Error parsing claims. Raw output: " + clean_resp]
