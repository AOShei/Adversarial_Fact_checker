import json
from typing import List, Dict, Any, Union
from app.llm import call_llm

def run_factual_claim_extractor(report: str, provider: str, config: Dict[str, Any]) -> List[str]:
    prompt = f"""
    Analyze the following report and extract a list of ALL distinct, checkable factual claims.
    
    CRITICAL INSTRUCTION: REWRITE CLAIMS FOR CONTEXT
    - Do NOT just copy sentences. You must REWRITE them to be standalone facts.
    - **SPLIT ATTRIBUTED CLAIMS:**
      - If a person is quoted making a factual claim, split it into TWO claims:
        1. The Attribution: "Donald Trump stated that dangerous criminals have entered the US."
        2. The Substance: "Dangerous criminals from prisons and mental institutions have illegally entered the US."
    - **Resolve Pronouns & Entities:**
      - Text: "He said they are eating the dogs." -> Claim: "Former President Donald Trump stated that Haitian immigrants are eating dogs in Springfield, Ohio."
      - Text: "The bill passed last week." -> Claim: "The [Specific Bill Name] passed the Senate on [Specific Date]."
    - **Add Missing Context:**
      - If the text says "The 2020 election was stolen", the claim should be "Donald Trump claims the 2020 US Presidential Election was stolen." (Attribute opinions/allegations).
      - If the text describes an event, include the DATE and LOCATION if available in the document.
    
    Return ONLY a raw JSON list of strings. Do not use Markdown formatting.
    
    Report:
    {report[:5000]}
    """
    response = call_llm(prompt, provider, config)
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_resp)
    except:
        return ["Error parsing claims. Raw output: " + clean_resp]
