import sys
import os
import json
import time
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st

# Add the project root to sys.path to allow importing from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import init_db, save_to_db, get_history
from app.models import ClaimAnalysis
from app.agents import run_factual_claim_extractor
from app.pipeline import batch_process_claims
from app.logger import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# ==========================================
# STREAMLIT FRONTEND
# ==========================================
st.set_page_config(page_title="Adversarial Fact Checker", layout="wide", page_icon="⚖️")

# Custom CSS for a cleaner, more professional look
st.markdown("""
<style>
    /* Reduce default top padding so the title sits higher */
    .block-container {
        padding-top: 1.5rem !important;
    }
    header[data-testid="stHeader"] {
        height: 2rem;
    }

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

# Initialize Session State
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


def get_selected_rows(dataframe_event: Any) -> List[int]:
    event_dict = dict(dataframe_event) if dataframe_event else {}
    selection = event_dict.get("selection", {})
    rows = selection.get("rows", []) if isinstance(selection, dict) else []
    return rows if isinstance(rows, list) else []

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Configuration")
    provider = st.radio("Select AI Provider", ["Microsoft Azure", "Google Gemini"], index=0)
    
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

st.title("Adversarial Fact Checker")
st.caption("Claims are scored using the NATO Intelligence Evaluation System to determine truthfulness and source reliability.")

# --- CONTEXT DIALOG (MODAL) ---
@st.dialog("Claim Analysis Details", width="large")
def show_claim_details(claim_data: Dict[str, Any]):
    score = claim_data.get('arbiter_score', 6)
    reliability = claim_data.get('source_reliability_score', 'F')

    score_meta = {
        1: {"color": "#27ae60", "label": "Confirmed"},       # Green
        2: {"color": "#2ecc71", "label": "Probably True"},    # Light Green
        3: {"color": "#f39c12", "label": "Possibly True"},    # Orange
        4: {"color": "#e67e22", "label": "Doubtful"},         # Dark Orange
        5: {"color": "#c0392b", "label": "Improbable"},       # Red
        6: {"color": "#7f8c8d", "label": "Truth Cannot Be Judged"}  # Grey
    }
    reliability_meta = {
        "A": {"color": "#27ae60", "label": "Completely Reliable"},
        "B": {"color": "#2ecc71", "label": "Usually Reliable"},
        "C": {"color": "#f39c12", "label": "Fairly Reliable"},
        "D": {"color": "#e67e22", "label": "Not Usually Reliable"},
        "E": {"color": "#c0392b", "label": "Unreliable"},
        "F": {"color": "#7f8c8d", "label": "Reliability Cannot Be Judged"},
    }
    s_meta = score_meta.get(score, score_meta[6])
    r_meta = reliability_meta.get(reliability, reliability_meta["F"])
    
    st.markdown(f"### Claim: *{claim_data['claim']}*")
    
    # NATO Combined Rating Badge
    st.markdown(
        f"""<div style="background-color: {s_meta['color']}; padding: 10px; border-radius: 8px; color: white; text-align: center; font-weight: bold; margin-bottom: 10px;">
        NATO RATING: {reliability}{score} — {r_meta['label']} / {s_meta['label']}
        </div>""", 
        unsafe_allow_html=True
    )

    # Individual badges side by side
    col_truth, col_rel = st.columns(2)
    with col_truth:
        st.markdown(
            f"""<div style="background-color: {s_meta['color']}; padding: 8px; border-radius: 6px; color: white; text-align: center; font-size: 0.9rem;">
            Truthfulness: {score} — {s_meta['label']}
            </div>""",
            unsafe_allow_html=True
        )
    with col_rel:
        st.markdown(
            f"""<div style="background-color: {r_meta['color']}; padding: 8px; border-radius: 6px; color: white; text-align: center; font-size: 0.9rem;">
            Source Reliability: {reliability} — {r_meta['label']}
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(f"**Truthfulness Justification:** {claim_data.get('arbiter_justification', '')}")
    st.markdown(f"**Source Reliability Justification:** {claim_data.get('source_reliability_justification', 'N/A')}")
    st.divider()
    
    t1, t2, t3 = st.tabs(["🗪 Debate", "📄 Internal Evidence", "🌐 External Evidence"])
    
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

# --- TABS: NEW ANALYSIS vs HISTORY ---
tab_new, tab_history = st.tabs(["Analyze", "History"])

# --- TAB 1: NEW ANALYSIS ---
with tab_new:
    report_input = st.text_area("Paste Report / Text to Analyze:", height=150, key="report_input")

    # Optional source metadata — helps the LLM understand provenance
    with st.expander("Source Details (optional)", expanded=False):
        meta_cols = st.columns(3)
        with meta_cols[0]:
            source_publisher = st.text_input("Publisher / Organisation", placeholder="e.g. WHO, Reuters, Fox News", key="source_publisher")
        with meta_cols[1]:
            source_author = st.text_input("Author(s)", placeholder="e.g. Jane Smith", key="source_author")
        with meta_cols[2]:
            source_date = st.text_input("Publication Date", placeholder="e.g. March 2024", key="source_date")

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
            
            # Build report text with metadata preamble when provided
            metadata_parts = []
            if source_publisher:
                metadata_parts.append(f"Publisher / Organisation: {source_publisher}")
            if source_author:
                metadata_parts.append(f"Author(s): {source_author}")
            if source_date:
                metadata_parts.append(f"Publication Date: {source_date}")

            if metadata_parts:
                metadata_preamble = (
                    "--- SOURCE METADATA (user-provided, treat as context for evaluating claims) ---\n"
                    + "\n".join(metadata_parts)
                    + "\n--- END SOURCE METADATA ---\n\n"
                )
                full_report_text = metadata_preamble + report_input
            else:
                full_report_text = report_input

            with st.status("Initializing Analysis Pipeline...", expanded=True) as status:
                st.write("🔍 **Phase 1:** Extracting Factual Claims...")
                progress_bar = st.progress(0, text="Extracting claims...")
                progress_status = st.empty()
                phase1_start = time.perf_counter()
                logger.info("Analysis started | provider=%s input_len=%d", provider, len(full_report_text))

                # Shared collectors for progress callback
                _results_collector: List[Dict[str, Any]] = []
                _progress_lines: List[str] = []
                score_labels = {
                    1: "Confirmed", 2: "Probably True", 3: "Possibly True",
                    4: "Doubtful", 5: "Improbable", 6: "Truth Cannot Be Judged"
                }
                reliability_labels = {
                    "A": "Completely Reliable", "B": "Usually Reliable",
                    "C": "Fairly Reliable", "D": "Not Usually Reliable",
                    "E": "Unreliable", "F": "Reliability Cannot Be Judged"
                }

                def _on_progress(completed: int, total: int, result: Dict[str, Any]) -> None:
                    """Callback fired each time a claim finishes."""
                    _results_collector.append(result)
                    score = result.get("arbiter_score", 6)
                    reliability = result.get("source_reliability_score", "F")
                    s_label = score_labels.get(score, "Unknown")
                    r_label = reliability_labels.get(reliability, "Unknown")
                    claim_text = result.get("claim", "")[:90]
                    _progress_lines.append(
                        f"✅ Claim {completed}/{total}: *{claim_text}* — **{reliability}{score} ({r_label} / {s_label})**"
                    )
                    # ── Live progress bar update ──
                    if total > 0:
                        progress_bar.progress(
                            completed / total,
                            text=f"Analyzed {completed}/{total} claims...",
                        )
                        progress_status.markdown(
                            f"Latest: *{claim_text}* — **{reliability}{score} ({r_label} / {s_label})**"
                        )

                async def _run_full_pipeline() -> tuple:
                    """Single async entry-point: extract claims then batch-process.
                    Keeps one event loop alive so the AsyncAzureOpenAI client stays valid."""
                    claims = await run_factual_claim_extractor(full_report_text, provider, config)
                    if (
                        isinstance(claims, list)
                        and len(claims) > 0
                        and isinstance(claims[0], str)
                        and not claims[0].startswith("Error")
                    ):
                        results = await batch_process_claims(
                            claims, full_report_text, provider, config,
                            on_progress=_on_progress,
                            source_metadata=metadata_preamble if metadata_parts else "",
                        )
                        return claims, results
                    return claims, None

                # --- Single asyncio.run() for the whole pipeline ---
                try:
                    pipeline_result = asyncio.run(_run_full_pipeline())
                    claims: List[str] = pipeline_result[0]
                    all_results: List[Dict[str, Any]] | None = pipeline_result[1]
                except Exception as pipeline_err:
                    logger.error("Pipeline failed: %s", pipeline_err, exc_info=True)
                    claims = []
                    all_results = None
                    st.warning(f"Pipeline error: {pipeline_err}")

                phase1_elapsed = time.perf_counter() - phase1_start

                if all_results is not None:
                    total_claims = len(claims)
                    progress_bar.progress(1.0, text=f"✅ All {total_claims} claims analyzed.")
                    progress_status.empty()

                    # Show per-claim progress lines
                    for line in _progress_lines:
                        st.write(line)

                    st.write(f"⏱️ **Pipeline completed in {phase1_elapsed:.1f}s**")
                    status.update(label=f"Analysis Complete — {total_claims} claims in {phase1_elapsed:.1f}s", state="complete", expanded=False)

                    st.session_state.analysis_results = list(all_results) if all_results else _results_collector
                    st.session_state.processing_complete = True
                    logger.info("Analysis complete | claims=%d elapsed=%.2fs", total_claims, phase1_elapsed)
                    save_to_db(report_input, st.session_state.analysis_results, provider)
                else:
                    status.update(label="Extraction Failed", state="error")
                    logger.error("Claim extraction failed | output=%s", claims)
                    st.error(f"Could not extract claims. Output: {claims}")

    # Results for New Analysis
    if st.session_state.processing_complete and st.session_state.analysis_results:
        col_header, col_clear = st.columns([8, 2])
        with col_header:
            st.markdown("## Analysis Results")
            st.caption("Select a row to view detailed evidence and reasoning.")
        with col_clear:
            st.markdown("")  # spacer to align button vertically
            if st.button("Clear Results", type="secondary"):
                st.session_state.analysis_results = []
                st.session_state.processing_complete = False
                st.session_state.report_input = ""
                st.session_state.source_publisher = ""
                st.session_state.source_author = ""
                st.session_state.source_date = ""
                st.rerun()
        
        display_data = []
        score_labels = {
            1: "Confirmed", 2: "Probably True", 3: "Possibly True",
            4: "Doubtful", 5: "Improbable", 6: "Truth Cannot Be Judged"
        }
        reliability_labels = {
            "A": "Completely Reliable", "B": "Usually Reliable",
            "C": "Fairly Reliable", "D": "Not Usually Reliable",
            "E": "Unreliable", "F": "Reliability Cannot Be Judged"
        }
        
        for idx, res in enumerate(st.session_state.analysis_results):
            r = res.get('source_reliability_score', 'F')
            s = res['arbiter_score']
            display_data.append({
                "Claim": res["claim"],
                "Verdict": f"{r}{s} - {reliability_labels.get(r, 'Unknown')} / {score_labels.get(s, 'Unknown')}",
                "Score": res["arbiter_score"] # Hidden sortable column
            })
            
        event = st.dataframe(
            display_data,
            on_select="rerun",
            selection_mode="single-row",
            width="stretch", # Auto
            hide_index=True,
            column_config={
                "Claim": st.column_config.TextColumn("Claim", width="large"),
                "Verdict": st.column_config.TextColumn("Verdict", width="medium"),
                "Score": None
            }
        )
        
        selected_rows = get_selected_rows(event)
        if selected_rows:
            selected_index = selected_rows[0]
            selected_claim_data = st.session_state.analysis_results[selected_index]
            show_claim_details(selected_claim_data)

# --- TAB 2: HISTORY (FLATTENED CLAIM LIST) ---
with tab_history:
    st.header("Claim History")
    
    # Search Bar
    search_query = st.text_input("Search Claims:", placeholder="Enter keywords...")

    # Fetch all history (or limit to a reasonable number like 100 recent reports to avoid DB load)
    # Note: For production, we'd want pagination. For MVP, fetching last 50 reports is fine.
    history_rows = get_history(limit=50) 
    
    if not history_rows:
        st.info("No history found.")
    else:
        # Flatten the data: [ {Timestamp, Claim, Verdict, FullData}, ... ]
        flattened_history = []
        score_labels = {
            1: "Confirmed", 2: "Probably True", 3: "Possibly True",
            4: "Doubtful", 5: "Improbable", 6: "Truth Cannot Be Judged"
        }
        reliability_labels = {
            "A": "Completely Reliable", "B": "Usually Reliable",
            "C": "Fairly Reliable", "D": "Not Usually Reliable",
            "E": "Unreliable", "F": "Reliability Cannot Be Judged"
        }

        for row in history_rows:
            # row: (id, report_text, analysis_results_json, provider, timestamp)
            row_id, _, analysis_json, _, timestamp = row
            try:
                results = json.loads(analysis_json)
                if isinstance(results, list):
                    for res in results:
                        # Filter by search query if present
                        if search_query and (search_query.lower() not in res['claim'].lower()):
                            continue

                        r = res.get('source_reliability_score', 'F')
                        s = res.get('arbiter_score', 6)
                        flattened_history.append({
                            "Timestamp": timestamp,
                            "Claim": res["claim"],
                            "Verdict": f"{r}{s} - {reliability_labels.get(r, 'Unknown')} / {score_labels.get(s, 'Unknown')}",
                            "Score": res.get('arbiter_score', 6),
                            "FullData": res # Store full object for the modal
                        })
            except:
                continue
        
        if not flattened_history:
            st.warning("No claims found matching your search.")
        else:
            st.caption(f"Showing {len(flattened_history)} historical claims.")
            
            # Display Flattened Table
            history_event = st.dataframe(
                flattened_history,
                on_select="rerun",
                selection_mode="single-row",
                width="stretch",
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a", width="small"),
                    "Claim": st.column_config.TextColumn("Claim", width="large"),
                    "Verdict": st.column_config.TextColumn("Verdict", width="medium"),
                    "Score": None,
                    "FullData": None # Hidden
                }
            )
            
            # Handle History Selection
            selected_history_rows = get_selected_rows(history_event)
            if selected_history_rows:
                h_index = selected_history_rows[0]
                h_data = flattened_history[h_index]["FullData"]
                show_claim_details(h_data)
