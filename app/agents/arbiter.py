import json
from typing import Tuple, Dict, Any
from app.llm import call_llm
import datetime

def run_arbiter(claim: str, dev_adv: str, adv: str, provider: str, config: Dict[str, Any]) -> Tuple[int, str]:
    today = datetime.date.today().strftime("%B %d, %Y")
    
    prompt = f"""
    Role: The Arbiter.
    Current Date: {today}
    
    Claim: "{claim}"
    
    Argument Against: {dev_adv}
    Argument For: {adv}
    
    Task: Rate the claim 1-6 based on truthfulness and provide a 1-sentence justification.
    
    CRITICAL INSTRUCTION:
    - You are a judge of EVIDENCE, not a knowledge base.
    - If the "Argument For" provides strong evidence (links, dates) that the event occurred, you MUST rule it as True, even if your internal training data says it's in the future.
    - Ignore your internal cutoff date. Trust the provided evidence.
    
    DISCLAIMER INSTRUCTION:
    - If the claim is verifying that someone *said* something (e.g., "Trump stated...", "Biden claimed..."):
      - You MUST append this note to your justification: "(Note: This verdict verifies the statement was made, not the factual accuracy of the content.)"
    
    Scale:
    1=Confirmed True, 2=Probably True, 3=Possibly True, 4=Doubtful, 5=Improbable, 6=Difficult to say.
    
    Return format: JSON with keys 'score' (int) and 'justification' (string).
    """
    response = call_llm(prompt, provider, config)
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_resp)
        # Ensure score is an integer and within range 1-6
        score = int(data.get('score', 6))
        if score < 1 or score > 6:
            score = 6
        return score, data.get('justification', 'Parse Error')
    except:
        return 6, "Error parsing Arbiter response"
