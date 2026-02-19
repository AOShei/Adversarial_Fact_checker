from pydantic import BaseModel
from typing import Optional

class ClaimAnalysis(BaseModel):
    claim: str
    report_evidence: str = ""
    web_evidence: str = ""
    devils_advocate_summary: str = ""
    advocate_summary: str = ""
    arbiter_score: int = 6
    arbiter_justification: str = ""
