import concurrent.futures
from typing import List, Dict, Any, Generator
from app.models import ClaimAnalysis
from app.agents import (
    run_supporting_evidence_extractor,
    run_web_search_agent,
    run_devils_advocate,
    run_advocate,
    run_arbiter
)

def process_single_claim(claim: str, report_text: str, provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the full analysis pipeline for a single claim.
    """
    try:
        analysis = ClaimAnalysis(claim=claim)
        
        # 1. Internal Evidence
        analysis.report_evidence = run_supporting_evidence_extractor(report_text, claim, provider, config)
        
        # 2. Web Search
        analysis.web_evidence = run_web_search_agent(claim, provider, config)
        
        # 3. Debate
        analysis.devils_advocate_summary = run_devils_advocate(claim, analysis.web_evidence, provider, config)
        analysis.advocate_summary = run_advocate(claim, analysis.web_evidence, provider, config)
        
        # 4. Arbiter
        score, just = run_arbiter(claim, analysis.devils_advocate_summary, analysis.advocate_summary, provider, config)
        analysis.arbiter_score = score
        analysis.arbiter_justification = just
        
        return analysis.model_dump()
    except Exception as e:
        # Return a partial failure object instead of crashing
        error_analysis = ClaimAnalysis(
            claim=claim, 
            arbiter_score=6, 
            arbiter_justification=f"Processing Error: {str(e)}"
        )
        return error_analysis.model_dump()

def batch_process_claims(claims: List[str], report_text: str, provider: str, config: Dict[str, Any], max_workers: int = 25) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields completed analysis results as they finish.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to claims
        future_to_claim = {
            executor.submit(process_single_claim, claim, report_text, provider, config): claim 
            for claim in claims
        }
        
        for future in concurrent.futures.as_completed(future_to_claim):
            yield future.result()
