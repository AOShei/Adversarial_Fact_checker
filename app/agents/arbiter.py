import json
import time
import datetime
from typing import Tuple, Dict, Any
from app.llm import call_llm
from app.logger import get_logger

logger = get_logger(__name__)


async def run_arbiter(claim: str, dev_adv: str, adv: str, provider: str, config: Dict[str, Any]) -> Tuple[int, str]:
    today = datetime.date.today().strftime("%B %d, %Y")
    
    prompt = f"""
    Role: The Arbiter.
    Current Date: {today}
    
    Claim: "{claim}"
    
    Argument Against: {dev_adv}
    Argument For: {adv}
    
    Task: Rate the claim 1-6 based on truthfulness and provide a 1-sentence justification.
    
    CRITICAL INSTRUCTIONS:
    - You are an expert fact-checker. Use BOTH the provided arguments AND your own extensive knowledge.
    - If the provided arguments lack strong evidence but the claim is well-known and factually accurate based on your training knowledge, you SHOULD still rate it as True or Probably True.
    - Do NOT default to "Difficult to say" simply because external search results were unavailable. Use your own knowledge to evaluate the claim.
    - If the "Argument For" provides strong evidence (links, dates) that the claim is correct, you MUST rule it as True.
    - Only use score 6 ("Difficult to say") for genuinely obscure claims where you have no basis for judgment.
    
    DISCLAIMER INSTRUCTION:
    - If the claim is verifying that someone *said* something (e.g., "Trump stated...", "Biden claimed..."):
      - You MUST append this note to your justification: "(Note: This verdict verifies the statement was made, not the factual accuracy of the content.)"
    
    Scale:
    1=Confirmed True, 2=Probably True, 3=Possibly True, 4=Doubtful, 5=Improbable, 6=Difficult to say.
    
    Return format: JSON with keys 'score' (int) and 'justification' (string).
    """
    start = time.perf_counter()
    claim_short = claim[:80]
    logger.info("Arbiter start | claim='%s'", claim_short)

    response = await call_llm(prompt, provider, config)
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_resp)
        score = int(data.get('score', 6))
        if score < 1 or score > 6:
            score = 6
        justification = data.get('justification', 'Parse Error')
        elapsed = time.perf_counter() - start
        logger.info("Arbiter done  | claim='%s' score=%d elapsed=%.2fs", claim_short, score, elapsed)
        return score, justification
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("Arbiter parse failure | claim='%s' elapsed=%.2fs error=%s", claim_short, elapsed, exc)
        return 6, "Error parsing Arbiter response"
