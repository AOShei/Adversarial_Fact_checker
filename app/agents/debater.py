from typing import Dict, Any
from app.llm import call_llm
import datetime

def run_devils_advocate(claim: str, web_evidence: str, provider: str, config: Dict[str, Any]) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    prompt = f"""
    Role: Devil's Advocate.
    Current Date: {today}
    
    Claim: "{claim}"
    External Evidence: {web_evidence}
    
    Task: Write a critical summary (max 3 sentences) attacking the validity of this claim based on the evidence or logical fallacies.
    
    CRITICAL INSTRUCTION:
    - Base your arguments ONLY on the provided "External Evidence".
    - Do NOT use your internal training data to judge the timing of events. 
    - If the evidence confirms an event happened (even if you think it's in the future), you must accept that evidence as the current reality.
    """
    return call_llm(prompt, provider, config)

def run_advocate(claim: str, web_evidence: str, provider: str, config: Dict[str, Any]) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    prompt = f"""
    Role: The Advocate.
    Current Date: {today}
    
    Claim: "{claim}"
    External Evidence: {web_evidence}
    
    Task: Write a supportive summary (max 3 sentences) defending this claim based on the evidence.
    
    CRITICAL INSTRUCTION:
    - Base your arguments ONLY on the provided "External Evidence".
    - Do NOT use your internal training data to judge the timing of events.
    - If the evidence confirms an event happened (even if you think it's in the future), you must accept that evidence as the current reality.
    """
    return call_llm(prompt, provider, config)
