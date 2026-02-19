import time
import datetime
from typing import Dict, Any
from app.llm import call_llm
from app.logger import get_logger

logger = get_logger(__name__)


async def run_devils_advocate(claim: str, web_evidence: str, provider: str, config: Dict[str, Any]) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    prompt = f"""
    Role: Devil's Advocate.
    Current Date: {today}
    
    Claim: "{claim}"
    External Evidence: {web_evidence}
    
    Task: Write a critical summary (max 3 sentences) attacking the validity of this claim.
    
    INSTRUCTIONS:
    - Use the provided "External Evidence" as your primary source.
    - If external evidence is empty or says "No relevant results found", use your own expert knowledge to critique the claim.
    - If the evidence confirms an event happened (even if you think it's in the future), accept that evidence as current reality.
    - Focus on factual inaccuracies, missing nuance, or logical weaknesses.
    """
    start = time.perf_counter()
    claim_short = claim[:80]
    logger.info("Devil's advocate start | claim='%s'", claim_short)
    result = await call_llm(prompt, provider, config)
    elapsed = time.perf_counter() - start
    logger.info("Devil's advocate done  | claim='%s' elapsed=%.2fs", claim_short, elapsed)
    return result


async def run_advocate(claim: str, web_evidence: str, provider: str, config: Dict[str, Any]) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    prompt = f"""
    Role: The Advocate.
    Current Date: {today}
    
    Claim: "{claim}"
    External Evidence: {web_evidence}
    
    Task: Write a supportive summary (max 3 sentences) defending this claim.
    
    INSTRUCTIONS:
    - Use the provided "External Evidence" as your primary source.
    - If external evidence is empty or says "No relevant results found", use your own expert knowledge to support the claim.
    - If the evidence confirms an event happened (even if you think it's in the future), accept that evidence as current reality.
    - Focus on factual support, corroborating data, and logical consistency.
    """
    start = time.perf_counter()
    claim_short = claim[:80]
    logger.info("Advocate start | claim='%s'", claim_short)
    result = await call_llm(prompt, provider, config)
    elapsed = time.perf_counter() - start
    logger.info("Advocate done  | claim='%s' elapsed=%.2fs", claim_short, elapsed)
    return result
