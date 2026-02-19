import time
from typing import Dict, Any
from app.llm import call_llm
from app.logger import get_logger

logger = get_logger(__name__)


async def run_supporting_evidence_extractor(report: str, claim: str, provider: str, config: Dict[str, Any]) -> str:
    prompt = f"""
    Given the report below, extract exact quotes that support the claim: "{claim}".
    If no direct evidence exists in the text, say "No direct evidence in report."
    
    Report:
    {report[:5000]}
    """
    start = time.perf_counter()
    claim_short = claim[:80]
    logger.info("Evidence extraction start | claim='%s'", claim_short)
    result = await call_llm(prompt, provider, config)
    elapsed = time.perf_counter() - start
    logger.info("Evidence extraction done  | claim='%s' elapsed=%.2fs", claim_short, elapsed)
    return result
