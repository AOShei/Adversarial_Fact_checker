import json
import time
import datetime
from typing import Tuple, Dict, Any
from app.llm import call_llm
from app.logger import get_logger

logger = get_logger(__name__)


async def run_arbiter(
    claim: str,
    dev_adv: str,
    adv: str,
    provider: str,
    config: Dict[str, Any],
    web_evidence: str = "",
    source_metadata: str = "",
) -> Tuple[int, str, str, str]:
    today = datetime.date.today().strftime("%B %d, %Y")

    # Build optional context sections
    evidence_section = ""
    if web_evidence.strip():
        evidence_section = f"\n    Web Evidence / Sources:\n    {web_evidence}\n"

    metadata_section = ""
    if source_metadata.strip():
        metadata_section = f"\n    Source Metadata (user-provided):\n    {source_metadata}\n"

    prompt = f"""
    Role: The Arbiter.
    Current Date: {today}
    
    Claim: "{claim}"
    
    Argument Against: {dev_adv}
    Argument For: {adv}
    {evidence_section}{metadata_section}
    Task: You must provide TWO separate ratings for this claim using the NATO Intelligence Evaluation System:

    1. **Credibility of Information (Truthfulness)** — Rate the CLAIM itself on a scale of 1-6:
       1 = Confirmed: Confirmed by other independent sources; logical in itself; consistent with other information on the subject.
       2 = Probably True: Not confirmed; logical in itself; consistent with other information on the subject.
       3 = Possibly True: Not confirmed; reasonably logical in itself; agrees with some other information.
       4 = Doubtful: Not confirmed; possible but not logical; no other information on the subject.
       5 = Improbable: Not confirmed; not logical in itself; contradicted by other information.
       6 = Truth Cannot Be Judged: No basis exists for evaluating the validity of the information.

    2. **Reliability of Source** — Rate the SOURCES used to support/refute the claim on a scale of A-F:
       A = Completely Reliable: No doubt of authenticity, trustworthiness, or competency; history of complete reliability.
       B = Usually Reliable: Minor doubt about authenticity, trustworthiness, or competency; history of valid information.
       C = Fairly Reliable: Doubt about authenticity, trustworthiness, or competency but has provided valid information in the past.
       D = Not Usually Reliable: Significant doubt about authenticity, trustworthiness, or competency; history of invalid information.
       E = Unreliable: Lacking in authenticity, trustworthiness, and competency; history of invalid information.
       F = Reliability Cannot Be Judged: No basis exists for evaluating the reliability of the source.

    CRITICAL INSTRUCTIONS:
    - You are an expert fact-checker. Use BOTH the provided arguments AND your own extensive knowledge.
    - If the provided arguments lack strong evidence but the claim is well-known and factually accurate based on your training knowledge, you SHOULD still rate it as True or Probably True.
    - Do NOT default to "Truth Cannot Be Judged" simply because external search results were unavailable. Use your own knowledge to evaluate the claim.
    - If the "Argument For" provides strong evidence (links, dates) that the claim is correct, you MUST rule it as Confirmed.
    - Only use score 6 for genuinely obscure claims where you have no basis for judgment.
    - For Source Reliability: evaluate the quality, reputation, and track record of the sources cited in the web evidence and source metadata. If no sources are available, use "F".
    - Provide a separate 1-sentence justification for EACH rating.
    
    DISCLAIMER INSTRUCTION:
    - If the claim is verifying that someone *said* something (e.g., "Trump stated...", "Biden claimed..."):
      - You MUST append this note to your truthfulness justification: "(Note: This verdict verifies the statement was made, not the factual accuracy of the content.)"
    
    Return format: JSON with keys:
      'score' (int 1-6),
      'justification' (string — truthfulness justification),
      'source_reliability_score' (string, one of: "A", "B", "C", "D", "E", "F"),
      'source_reliability_justification' (string — source reliability justification).
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

        reliability = str(data.get('source_reliability_score', 'F')).upper().strip()
        if reliability not in ("A", "B", "C", "D", "E", "F"):
            reliability = "F"
        reliability_justification = data.get('source_reliability_justification', '')

        elapsed = time.perf_counter() - start
        logger.info(
            "Arbiter done  | claim='%s' score=%d reliability=%s elapsed=%.2fs",
            claim_short, score, reliability, elapsed,
        )
        return score, justification, reliability, reliability_justification
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("Arbiter parse failure | claim='%s' elapsed=%.2fs error=%s", claim_short, elapsed, exc)
        return 6, "Error parsing Arbiter response", "F", ""
