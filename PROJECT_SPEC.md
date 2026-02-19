# MVP Specification: Agentic Adversarial Truth Analysis Framework

## 1. Overview

The Agentic Adversarial Truth Analysis Framework is a multi-agent AI system designed to reduce hallucination and confirmation bias when analyzing text-based reports. It extracts factual claims from a document, gathers internal and external evidence, and simulates an adversarial debate between opposing AI agents. A final "Arbiter" agent evaluates the debate to score the truthfulness of the original claim.

## 2. Technology Stack

* **Language:** Python 3.10+
* **Frontend & UI:** Streamlit (Native Markdown support, interactive expanders, status spinners)
* **Database:** SQLite (Local, lightweight storage for run history)
* **Data Validation:** Pydantic (Ensures structured JSON outputs from LLMs)
* **Web Search:** DuckDuckGo Search (`duckduckgo-search` for keyless prototyping)
* **AI Providers (A/B Testing):**
* Google Gemini (`gemini-2.5-flash` and `gemini-3-flash-preview`) via Google AI Studio
* Microsoft Azure OpenAI (e.g., `gpt-5.X`)



## 3. Pipeline & Agent Roles

The framework executes sequentially for every extracted claim.

1. **Factual Claim Extractor:** Consumes the Markdown report and outputs a distinct list of factual claims and core assumptions.
2. **Supporting Evidence Extractor:** Cross-references a single claim against the original report to find exact quotes or data points the author used as support.
3. **Web Search Agent:** Converts the claim into an optimized search query and retrieves the top public web snippets.
4. **Devil’s Advocate:** Analyzes the claim and web evidence to write a critical summary highlighting logical flaws, fallacies, or contradicting data.
5. **The Advocate:** Analyzes the same claim and web evidence to write a supportive summary highlighting corroborating data and logical soundness.
6. **The Arbiter:** Reviews the claim, the internal evidence, and the adversarial debate summaries. It assigns a final truthfulness score and a brief justification.

## 4. Arbiter Scoring Scale

The Arbiter is strictly constrained to output an integer from the following scale:

| Score | Rating | Definition |
| --- | --- | --- |
| **1** | Confirmed True | Logical, consistent with other relevant information, confirmed by independent sources. |
| **2** | Probably True | Logical, consistent with other relevant information, maybe not confirmed. |
| **3** | Possibly True | Reasonably logical, agrees with some relevant information, not confirmed. |
| **4** | Doubtful | Not logical but possible, no other information on the subject, not confirmed. |
| **5** | Improbable | Not logical, contradicted by other relevant information. |
| **6** | Difficult to say | The validity of the information cannot be determined. |

---

## 5. Implementation Code (`app.py`)

Save the following code as `app.py`. Ensure you have installed the required dependencies:

`pip install streamlit pydantic duckduckgo-search google-genai openai`

Note we plan on using Azure AI foundry so the exact install and usage may differ.

for google genai here is a usage example:

```
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)
```

---

The following code is provided as a basic template to structure the ideas but we will need refactor this and make the proejct more maintainable. So split things up into multiple files, create a directory for agent instructions, make sure to setup the database and document the table schema, set up environmental variables, etc...

```python
import streamlit as st
import sqlite3
import json
import time
from pydantic import BaseModel
from typing import List
from duckduckgo_search import DDGS
from google import genai
from openai import AzureOpenAI

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('analysis_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_text TEXT,
            analysis_results TEXT,
            provider TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(report_text, results, provider):
    conn = sqlite3.connect('analysis_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO reports (report_text, analysis_results, provider) VALUES (?, ?, ?)', 
              (report_text, json.dumps(results), provider))
    conn.commit()
    conn.close()

# ==========================================
# 2. DATA MODELS
# ==========================================
class ClaimAnalysis(BaseModel):
    claim: str
    report_evidence: str = ""
    web_evidence: str = ""
    devils_advocate_summary: str = ""
    advocate_summary: str = ""
    arbiter_score: int = 6
    arbiter_justification: str = ""

# ==========================================
# 3. LLM ABSTRACTION LAYER
# ==========================================
def call_llm(prompt: str, provider: str, config: dict) -> str:
    """Routes the prompt to the selected API provider."""
    system_instruction = "You are a precise analytical agent. Return only the requested information."
    
    if provider == "Google Gemini":
        if not config.get("gemini_key"):
            return "Error: Gemini API Key missing."
        try:
            genai.configure(api_key=config["gemini_key"])
            # Utilizing gemini-2.5-flash as specified
            model = genai.GenerativeModel('gemini-2.5-flash') 
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    elif provider == "Microsoft Azure":
        required = ["azure_endpoint", "azure_key", "azure_version", "azure_deployment"]
        if not all(config.get(k) for k in required):
            return "Error: Missing Azure configuration details."
        try:
            client = AzureOpenAI(
                azure_endpoint=config["azure_endpoint"],
                api_key=config["azure_key"],
                api_version=config["azure_version"]
            )
            response = client.chat.completions.create(
                model=config["azure_deployment"],
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Azure Error: {str(e)}"
    
    return "Error: Invalid Provider"

# ==========================================
# 4. AGENT FUNCTIONS
# ==========================================
def run_factual_claim_extractor(report: str, provider, config) -> List[str]:
    prompt = f"""
    Analyze the following report and extract a list of 3-5 distinct, checkable factual claims.
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

def run_supporting_evidence_extractor(report: str, claim: str, provider, config) -> str:
    prompt = f"""
    Given the report below, extract exact quotes that support the claim: "{claim}".
    If no direct evidence exists in the text, say "No direct evidence in report."
    
    Report:
    {report[:5000]}
    """
    return call_llm(prompt, provider, config)

def run_web_search_agent(claim: str) -> str:
    try:
        results = DDGS().text(claim, max_results=3)
        if not results:
            return "No web results found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {e}"

def run_devils_advocate(claim: str, web_evidence: str, provider, config) -> str:
    prompt = f"""
    Role: Devil's Advocate.
    Claim: "{claim}"
    External Evidence: {web_evidence}
    Task: Write a critical summary (max 3 sentences) attacking the validity of this claim based on the evidence or logical fallacies.
    """
    return call_llm(prompt, provider, config)

def run_advocate(claim: str, web_evidence: str, provider, config) -> str:
    prompt = f"""
    Role: The Advocate.
    Claim: "{claim}"
    External Evidence: {web_evidence}
    Task: Write a supportive summary (max 3 sentences) defending this claim based on the evidence.
    """
    return call_llm(prompt, provider, config)

def run_arbiter(claim: str, dev_adv: str, adv: str, provider, config) -> tuple[int, str]:
    prompt = f"""
    Role: The Arbiter.
    Claim: "{claim}"
    Argument Against: {dev_adv}
    Argument For: {adv}
    Task: Rate the claim 1-6 based on truthfulness and provide a 1-sentence justification.
    1=Confirmed True, 2=Probably True, 3=Possibly True, 4=Doubtful, 5=Improbable, 6=Difficult to say.
    Return format: JSON with keys 'score' (int) and 'justification' (string).
    """
    response = call_llm(prompt, provider, config)
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_resp)
        return data.get('score', 6), data.get('justification', 'Parse Error')
    except:
        return 6, "Error parsing Arbiter response"

# ==========================================
# 5. STREAMLIT FRONTEND
# ==========================================
st.set_page_config(page_title="Adversarial Truth Arbiter", layout="wide")
init_db()

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.radio("Select AI Provider", ["Google Gemini", "Microsoft Azure"])
    
    config = {}
    if provider == "Google Gemini":
        config["gemini_key"] = st.text_input("Gemini API Key", type="password")
        st.info("Using model: gemini-2.5-flash")
    else:
        config["azure_endpoint"] = st.text_input("Azure Endpoint", placeholder="https://your-resource.openai.azure.com/")
        config["azure_key"] = st.text_input("Azure API Key", type="password")
        config["azure_deployment"] = st.text_input("Deployment Name", placeholder="gpt-4o")
        config["azure_version"] = st.text_input("API Version", value="2024-02-15-preview")

st.title("⚖️ Agentic Adversarial Truth Analysis")
st.markdown(f"**Active Provider:** {provider}")

report_input = st.text_area("Paste your Markdown Report here:", height=150)

if st.button("Run Analysis Pipeline", type="primary"):
    if not report_input.strip():
        st.warning("Please provide a report.")
    elif (provider == "Google Gemini" and not config.get("gemini_key")) or \
         (provider == "Microsoft Azure" and not config.get("azure_key")):
        st.error(f"Please configure {provider} API keys in the sidebar.")
    else:
        results_list = []
        
        with st.status("Running Agentic Pipeline...", expanded=True) as status:
            st.write("🕵️‍♂️ **Stage 1:** Factual Claim Extractor running...")
            claims = run_factual_claim_extractor(report_input, provider, config)
            
            if isinstance(claims, list) and len(claims) > 0 and not claims[0].startswith("Error"):
                for idx, claim in enumerate(claims):
                    st.write(f"--- Analyzing Claim {idx+1}/{len(claims)}: *{claim}*")
                    analysis = ClaimAnalysis(claim=claim)
                    
                    st.write("📖 Extracting internal evidence...")
                    analysis.report_evidence = run_supporting_evidence_extractor(report_input, claim, provider, config)
                    
                    st.write("🌐 Web Agent searching public data...")
                    analysis.web_evidence = run_web_search_agent(claim)
                    
                    st.write("🤺 Agents debating (Devil's Advocate vs Advocate)...")
                    analysis.devils_advocate_summary = run_devils_advocate(claim, analysis.web_evidence, provider, config)
                    analysis.advocate_summary = run_advocate(claim, analysis.web_evidence, provider, config)
                    
                    st.write("⚖️ Arbiter judgment...")
                    score, just = run_arbiter(claim, analysis.devils_advocate_summary, analysis.advocate_summary, provider, config)
                    analysis.arbiter_score = score
                    analysis.arbiter_justification = just
                    
                    results_list.append(analysis.dict())
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                save_to_db(report_input, results_list, provider)
            else:
                st.error(f"Failed to extract claims: {claims}")
                status.update(label="Failed", state="error")

        # RESULTS DISPLAY
        if results_list:
            st.header("📊 Analysis Results")
            score_colors = {
                1: "green", 2: "lightgreen", 3: "orange", 
                4: "red", 5: "darkred", 6: "gray"
            }
            score_labels = {
                1: "Confirmed True", 2: "Probably True", 3: "Possibly True",
                4: "Doubtful", 5: "Improbable", 6: "Difficult to say"
            }
            
            for idx, res in enumerate(results_list):
                score = res['arbiter_score']
                color = score_colors.get(score, "gray")
                label = score_labels.get(score, "Unknown")
                
                with st.expander(f"Claim {idx+1}: {res['claim']}", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Arbiter's Verdict:** {res['arbiter_justification']}")
                    with c2:
                        st.markdown(f"### :{color}[{score} - {label}]")
                    
                    st.divider()
                    t1, t2, t3 = st.tabs(["🤺 The Debate", "📄 Internal Evidence", "🌐 External Evidence"])
                    with t1:
                        ca, cb = st.columns(2)
                        with ca:
                            st.success(f"**Advocate:**\n\n{res['advocate_summary']}")
                        with cb:
                            st.error(f"**Devil's Advocate:**\n\n{res['devils_advocate_summary']}")
                    with t2:
                        st.info(res['report_evidence'])
                    with t3:
                        st.text(res['web_evidence'])

```

