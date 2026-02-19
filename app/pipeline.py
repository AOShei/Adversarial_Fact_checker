import asyncio
import time
from typing import List, Dict, Any, Callable, Optional
from app.models import ClaimAnalysis
from app.agents import (
    run_supporting_evidence_extractor,
    run_web_search_agent,
    run_devils_advocate,
    run_advocate,
    run_arbiter
)
from app.logger import get_logger

logger = get_logger(__name__)

# Overall pipeline timeout (5 minutes)
PIPELINE_TIMEOUT_SECONDS = 300


async def process_single_claim(
    claim: str,
    claim_index: int,
    report_text: str,
    provider: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Runs the full analysis pipeline for a single claim with intra-claim parallelism.
    Stage 1: evidence extraction + web search   (parallel)
    Stage 2: devil's advocate + advocate          (parallel)
    Stage 3: arbiter                              (sequential – needs debate output)
    """
    start = time.perf_counter()
    claim_short = claim[:80]
    logger.info("Claim %d start | '%s'", claim_index, claim_short)

    try:
        analysis = ClaimAnalysis(claim=claim)

        # Stage 1 – evidence + web search in parallel
        report_evidence, web_evidence = await asyncio.gather(
            run_supporting_evidence_extractor(report_text, claim, provider, config),
            run_web_search_agent(claim, provider, config),
        )
        analysis.report_evidence = report_evidence
        analysis.web_evidence = web_evidence
        logger.info("Claim %d stage-1 done (evidence+search) | elapsed=%.2fs", claim_index, time.perf_counter() - start)

        # Stage 2 – debate in parallel
        devils_summary, advocate_summary = await asyncio.gather(
            run_devils_advocate(claim, web_evidence, provider, config),
            run_advocate(claim, web_evidence, provider, config),
        )
        analysis.devils_advocate_summary = devils_summary
        analysis.advocate_summary = advocate_summary
        logger.info("Claim %d stage-2 done (debate) | elapsed=%.2fs", claim_index, time.perf_counter() - start)

        # Stage 3 – arbiter
        score, justification = await run_arbiter(claim, devils_summary, advocate_summary, provider, config)
        analysis.arbiter_score = score
        analysis.arbiter_justification = justification

        elapsed = time.perf_counter() - start
        logger.info("Claim %d DONE  | score=%d elapsed=%.2fs | '%s'", claim_index, score, elapsed, claim_short)
        return analysis.model_dump()

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("Claim %d FAILED | elapsed=%.2fs error=%s", claim_index, elapsed, e, exc_info=True)
        error_analysis = ClaimAnalysis(
            claim=claim,
            arbiter_score=6,
            arbiter_justification=f"Processing Error: {str(e)}"
        )
        return error_analysis.model_dump()


async def batch_process_claims(
    claims: List[str],
    report_text: str,
    provider: str,
    config: Dict[str, Any],
    max_workers: int = 5,
    on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Process all claims concurrently, bounded by a semaphore.

    Args:
        on_progress: Optional callback(completed_count, total, result_dict)
                     called each time a claim finishes.
    Returns:
        List of completed ClaimAnalysis dicts (order may differ from input).
    """
    total = len(claims)
    semaphore = asyncio.Semaphore(max_workers)
    results: List[Dict[str, Any]] = []
    completed_count = 0

    pipeline_start = time.perf_counter()
    logger.info("Batch start | claims=%d max_workers=%d", total, max_workers)

    async def _bounded_process(claim: str, idx: int) -> Dict[str, Any]:
        nonlocal completed_count
        async with semaphore:
            result = await process_single_claim(claim, idx, report_text, provider, config)
        completed_count += 1
        if on_progress:
            on_progress(completed_count, total, result)
        return result

    tasks = [_bounded_process(claim, i + 1) for i, claim in enumerate(claims)]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=False),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - pipeline_start
        logger.error("Batch TIMEOUT after %.2fs — %d/%d claims completed", elapsed, completed_count, total)
        # Collect whatever finished; tasks that timed out will have been cancelled
    except Exception as e:
        elapsed = time.perf_counter() - pipeline_start
        logger.error("Batch ERROR after %.2fs — %s", elapsed, e, exc_info=True)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("Batch done | completed=%d/%d elapsed=%.2fs", len(results), total, elapsed)
    return list(results)
