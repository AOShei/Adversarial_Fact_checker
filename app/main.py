import sys
import os
# Add the project root to sys.path to allow importing from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from app.database import init_db, save_to_db
from app.models import ClaimAnalysis
from app.agents import (
    run_factual_claim_extractor,
)
from app.pipeline import batch_process_claims

# Load environment variables
load_dotenv()

# ==========================================
# STREAMLIT FRONTEND
# ==========================================
st.set_page_config(page_title="Adversarial Fact Checker", layout="wide", page_icon="⚖️")

# Custom CSS for a cleaner, more professional look
st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        color: #1E1E1E; 
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Table selection highlight */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

init_db()

# Initialize Session State for Results Persistence
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = [] # Stores full dictionaries
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Configuration")
    provider = st.radio("Select AI Provider", ["Google Gemini", "Microsoft Azure"])
    
    config: Dict[str, Any] = {}
    if provider == "Google Gemini":
        config["gemini_key"] = os.getenv("GEMINI_API_KEY")
        if not config["gemini_key"]:
            st.warning("Missing GEMINI_API_KEY in .env file.")
        else:
            st.success("API Key Loaded")
        config["gemini_model"] = st.radio("Model Version", ["gemini-2.5-flash", "gemini-3-flash-preview"])
    else:
        config["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        config["azure_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        config["azure_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        config["azure_version"] = os.getenv("AZURE_OPENAI_API_VERSION")
        
        missing = [k for k, v in config.items() if not v]
        if missing:
             st.warning(f"Missing Azure config: {', '.join(missing)}")
        else:
             st.success("Configuration Loaded")

st.title("Adversarial Truth Arbiter")

# --- CONTEXT DIALOG (MODAL) ---
@st.dialog("Claim Analysis Details", width="large")
def show_claim_details(claim_data: Dict[str, Any]):
    score = claim_data.get('arbiter_score', 6)
    score_meta = {
        1: {"color": "#27ae60", "label": "Confirmed True"},   # Green
        2: {"color": "#2ecc71", "label": "Probably True"},    # Light Green
        3: {"color": "#f39c12", "label": "Possibly True"},    # Orange
        4: {"color": "#e67e22", "label": "Doubtful"},         # Dark Orange
        5: {"color": "#c0392b", "label": "Improbable"},       # Red
        6: {"color": "#7f8c8d", "label": "Uncertain"}         # Grey
    }
    meta = score_meta.get(score, score_meta[6])
    
    st.markdown(f"### Claim: *{claim_data['claim']}*")
    
    # Verdict Badge
    st.markdown(
        f"""<div style="background-color: {meta['color']}; padding: 10px; border-radius: 8px; color: white; text-align: center; font-weight: bold; margin-bottom: 20px;">
        VERDICT: {score} - {meta['label']}
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.markdown(f"**Justification:** {claim_data['arbiter_justification']}")
    st.divider()
    
    t1, t2, t3 = st.tabs(["🤺 Debate", "📄 Internal Evidence", "🌐 External Evidence"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Advocate:**\n\n{claim_data['advocate_summary']}")
        with c2:
            st.error(f"**Devil's Advocate:**\n\n{claim_data['devils_advocate_summary']}")
            
    with t2:
        st.info(claim_data['report_evidence'])
        
    with t3:
        st.caption("Web Sources:")
        st.markdown(claim_data['web_evidence'])


# --- MAIN UI ---
report_input = st.text_area("Paste Report / Text to Analyze:", height=150)

# Run Button Logic
if st.button("Run Analysis", type="primary"):
    if not report_input.strip():
        st.warning("Please provide text to analyze.")
    elif ((provider == "Google Gemini" and not config.get("gemini_key")) or 
          (provider == "Microsoft Azure" and not config.get("azure_key"))):
        st.error(f"Please configure {provider} API keys in your .env file.")
    else:
        # Reset State
        st.session_state.analysis_results = []
        st.session_state.processing_complete = False
        
        with st.status("Initializing Analysis Pipeline...", expanded=True) as status:
            st.write("🔍 **Phase 1:** Extracting Factual Claims...")
            claims = run_factual_claim_extractor(report_input, provider, config)
            
            if isinstance(claims, list) and len(claims) > 0 and isinstance(claims[0], str) and not claims[0].startswith("Error"):
                total_claims = len(claims)
                st.write(f"✅ Found {total_claims} claims. Starting batch processing...")
                
                progress_bar = status.progress(0)
                processed_count = 0
                
                # Run Parallel Batch Processing
                for result in batch_process_claims(claims, report_input, provider, config):
                    st.session_state.analysis_results.append(result)
                    processed_count += 1
                    progress_bar.progress(processed_count / total_claims)
                    # Optional: Update status text with latest claim processed
                    # st.write(f"Processed: {result['claim'][:50]}...")
                
                progress_bar.progress(1.0)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.session_state.processing_complete = True
                save_to_db(report_input, st.session_state.analysis_results, provider)
            else:
                status.update(label="Extraction Failed", state="error")
                st.error(f"Could not extract claims. Output: {claims}")

# --- RESULTS DISPLAY (Dataframe with Selection) ---
if st.session_state.processing_complete and st.session_state.analysis_results:
    st.markdown("## Analysis Results")
    st.caption("Select a row to view detailed evidence and reasoning.")
    
    # Prepare data for dataframe display (flattened for table)
    display_data = []
    score_labels = {
        1: "Confirmed True", 2: "Probably True", 3: "Possibly True",
        4: "Doubtful", 5: "Improbable", 6: "Uncertain"
    }
    
    for idx, res in enumerate(st.session_state.analysis_results):
        display_data.append({
            "Claim": res["claim"],
            "Verdict": f"{res['arbiter_score']} - {score_labels.get(res['arbiter_score'], 'Unknown')}",
            "Score": res["arbiter_score"] # Hidden sortable column
        })
        
    # Display Interactive Dataframe
    event = st.dataframe(
        display_data,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch", 
        hide_index=True,
        column_config={
            "Claim": st.column_config.TextColumn("Claim", width="large"),
            "Verdict": st.column_config.TextColumn("Verdict", width="medium"),
            "Score": None # Hide the raw score column
        }
    )
    
    # Handle Selection Event
    if event and event.selection and event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_claim_data = st.session_state.analysis_results[selected_index]
        show_claim_details(selected_claim_data)
