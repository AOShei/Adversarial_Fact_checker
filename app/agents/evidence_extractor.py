from typing import Dict, Any
from app.llm import call_llm

def run_supporting_evidence_extractor(report: str, claim: str, provider: str, config: Dict[str, Any]) -> str:
    prompt = f"""
    Given the report below, extract exact quotes that support the claim: "{claim}".
    If no direct evidence exists in the text, say "No direct evidence in report."
    
    Report:
    {report[:5000]}
    """
    return call_llm(prompt, provider, config)
