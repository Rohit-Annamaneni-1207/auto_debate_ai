# streamlit_app.py
import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

# ---- Defensive imports of your project's modules ----
# These imports assume you run this app from the project root described earlier.
# If your modules live in different places, adjust the import paths accordingly.
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    st.stop(f"Missing dependency `sentence_transformers`. Install with `pip install sentence-transformers`. Error: {e}")

# Try to import the project's rag & agent utilities.
IMPORT_ERROR = None
try:
    # RAG pipeline utilities
    from src.tools.rag_pipeline.document_chunking import load_chunk_document  # optional diagnostic
    from src.tools.rag_pipeline.faiss_utils import (
        create_load_index,
        add_embeddings,
        load_index,
        search_index,
        clear_index,
    )
    from src.tools.rag_pipeline.process_file import process_file_add_to_index
except Exception as e1:
    try:
        # some projects keep modules directly under src.tools..., try alternate imports
        from tools.rag_pipeline.document_chunking import load_chunk_document
        from tools.rag_pipeline.faiss_utils import (
            create_load_index,
            add_embeddings,
            load_index,
            search_index,
            clear_index,
        )
        from tools.rag_pipeline.process_file import process_file_add_to_index
    except Exception as e2:
        IMPORT_ERROR = (
            "Could not import RAG pipeline modules from your project.\n"
            "Tried `src.tools.rag_pipeline.*` and `tools.rag_pipeline.*`.\n"
            f"Errors:\n - {e1}\n - {e2}\n\n"
            "Make sure you're running this Streamlit app from the project root\n"
            "and that the module paths match. You can adapt the import paths\n"
            "at the top of this file to your project layout."
        )

# Agents / orchestrator / debate
try:
    # demo/orchestrator
    from src.agents.orchestrator_agent import OrchestratorAgent
except Exception as e:
    try:
        from agents.orchestrator_agent import OrchestratorAgent
    except Exception as e2:
        # maybe orchestrator is in demo.py as class - fallback import
        try:
            from demo import OrchestratorAgent  # unlikely, but try
        except Exception:
            if IMPORT_ERROR:
                IMPORT_ERROR += f"\nAlso could not import OrchestratorAgent: {e} / {e2}"
            else:
                IMPORT_ERROR = f"Could not import OrchestratorAgent: {e} / {e2}"

try:
    # debate moderator
    from debate import DebateModerator
except Exception as e:
    try:
        from src.debate import DebateModerator
    except Exception as e2:
        if IMPORT_ERROR:
            IMPORT_ERROR += f"\nAlso could not import DebateModerator: {e} / {e2}"
        else:
            IMPORT_ERROR = f"Could not import DebateModerator: {e} / {e2}"

# If any import problems, show helpful message and stop.
if IMPORT_ERROR:
    st.set_page_config(page_title="RAG + Multi-Agent UI (Imports error)", layout="wide")
    st.title("RAG + Multi-Agent UI — Import error")
    st.error(IMPORT_ERROR)
    st.stop()

# ---- App config ----
st.set_page_config(page_title="RAG Multi-Agent Demo UI", layout="wide")
st.title("RAG Multi-Agent UI — Upload, Index, Orchestrate, Debate")

# ---- Globals for data directories ----
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
KB1_DOCS = DATA_DIR / "knowledge_base_1" / "documents"
KB2_DOCS = DATA_DIR / "knowledge_base_2" / "documents"

# Ensure directories exist
for d in [KB1_DOCS, KB2_DOCS]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Helper utilities ----
def save_uploaded_file_to_kb(uploaded_file, kb_num: int) -> str:
    """Save uploaded file object (streamlit) to target KB documents folder and return path."""
    if kb_num == 1:
        target_dir = KB1_DOCS
    else:
        target_dir = KB2_DOCS
    target_path = target_dir / uploaded_file.name
    # Save file
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(target_path)

def list_documents(kb_num: int) -> List[str]:
    d = KB1_DOCS if kb_num == 1 else KB2_DOCS
    return [str(p.name) for p in sorted(d.iterdir()) if p.is_file()]

@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def ensure_env_ok():
    """Check for OPENAI env vars (ChatOpenAI wrapper in project expects them)."""
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("OPENAI_API_BASE"):
        # Some code uses this; only warn if it's required by your ChatOpenAI wrapper.
        missing.append("OPENAI_API_BASE (used by ChatOpenAI in project)")
    return missing

# ---- UI Layout ----
tabs = st.tabs(["Indexing", "Orchestrator (demo)", "Debate", "Diagnostics / Logs"])

# ------ Tab: Indexing ------
with tabs[0]:
    st.header("1) Upload documents and build knowledge base indexes")
    st.markdown(
        """
        - Upload files (PDF / TXT) to either **Knowledge Base 1** or **Knowledge Base 2**.
        - After uploading, press **Process (chunk & embed)** to add the documents to the FAISS index.
        - You can clear indices if you want to reset them.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Knowledge Base 1 (KB1)")
        st.write("Documents (KB1):")
        kb1_list = list_documents(1)
        st.write(kb1_list if kb1_list else "*No files uploaded yet*")
        uploaded_kb1 = st.file_uploader("Upload file to KB1", type=["pdf", "txt"], key="upload_kb1")
        if uploaded_kb1:
            saved_path = save_uploaded_file_to_kb(uploaded_kb1, kb_num=1)
            st.success(f"Saved to: {saved_path}")
            # refresh list
            kb1_list = list_documents(1)

        if st.button("Process KB1 documents (chunk → embed → faiss)", key="process_kb1"):
            missing = ensure_env_ok()
            if missing:
                st.warning(f"Environment variables missing: {missing}. The pipeline may still run if not required by your connectors, but Chat/LLM calls will not succeed without proper env.")
            EMB = get_embedding_model()
            with st.spinner("Processing KB1 — chunking files and adding to index..."):
                try:
                    process_file_add_to_index(embedding_model=EMB, idx_num=1)
                    st.success("KB1 processed and index updated.")
                except Exception as e:
                    st.error(f"Error while processing KB1: {e}")

        if st.button("Clear KB1 index & metadata", key="clear_kb1"):
            try:
                clear_index(idx_num=1)
                st.success("Cleared KB1 embeddings/index/metadata.")
            except Exception as e:
                st.error(f"Failed to clear KB1: {e}")

        # --- DELETE FILE FROM KB1 ---
        st.markdown("### Delete a file from KB1")

        kb1_files = list_documents(1)
        if kb1_files:
            kb1_to_delete = st.selectbox(
                "Select a file to delete (KB1)",
                options=["-- select --"] + kb1_files,
                key="kb1_delete_select"
            )

            kb1_confirm = st.checkbox(
                "I confirm I want to permanently delete this file from KB1",
                key="kb1_delete_confirm"
            )

            if st.button("Delete Selected File from KB1", key="kb1_delete_button",
                         disabled=(kb1_to_delete == "-- select --" or not kb1_confirm)):
                try:
                    file_path = KB1_DOCS / kb1_to_delete
                    file_path.unlink()
                    st.success(f"Deleted file permanently: {kb1_to_delete}")
                except Exception as e:
                    st.error(f"Failed to delete file: {e}")
        else:
            st.info("No files available in KB1 to delete.")

    with col2:
        st.subheader("Knowledge Base 2 (KB2)")
        st.write("Documents (KB2):")
        kb2_list = list_documents(2)
        st.write(kb2_list if kb2_list else "*No files uploaded yet*")
        uploaded_kb2 = st.file_uploader("Upload file to KB2", type=["pdf", "txt"], key="upload_kb2")
        if uploaded_kb2:
            saved_path = save_uploaded_file_to_kb(uploaded_kb2, kb_num=2)
            st.success(f"Saved to: {saved_path}")
            kb2_list = list_documents(2)

        if st.button("Process KB2 documents (chunk → embed → faiss)", key="process_kb2"):
            missing = ensure_env_ok()
            if missing:
                st.warning(f"Environment variables missing: {missing}. The pipeline may still run if not required by your connectors, but Chat/LLM calls will not succeed without proper env.")
            EMB = get_embedding_model()
            with st.spinner("Processing KB2 — chunking files and adding to index..."):
                try:
                    process_file_add_to_index(embedding_model=EMB, idx_num=2)
                    st.success("KB2 processed and index updated.")
                except Exception as e:
                    st.error(f"Error while processing KB2: {e}")

        if st.button("Clear KB2 index & metadata", key="clear_kb2"):
            try:
                clear_index(idx_num=2)
                st.success("Cleared KB2 embeddings/index/metadata.")
            except Exception as e:
                st.error(f"Failed to clear KB2: {e}")

        # --- DELETE FILE FROM KB2 ---
        st.markdown("### Delete a file from KB2")

        kb2_files = list_documents(2)
        if kb2_files:
            kb2_to_delete = st.selectbox(
                "Select a file to delete (KB2)",
                options=["-- select --"] + kb2_files,
                key="kb2_delete_select"
            )

            kb2_confirm = st.checkbox(
                "I confirm I want to permanently delete this file from KB2",
                key="kb2_delete_confirm"
            )

            if st.button("Delete Selected File from KB2", key="kb2_delete_button",
                         disabled=(kb2_to_delete == "-- select --" or not kb2_confirm)):
                try:
                    file_path = KB2_DOCS / kb2_to_delete
                    file_path.unlink()
                    st.success(f"Deleted file permanently: {kb2_to_delete}")
                except Exception as e:
                    st.error(f"Failed to delete file: {e}")
        else:
            st.info("No files available in KB2 to delete.")

    st.markdown("---")
    st.markdown(
        "Tips: After uploading run **Process** to ensure files are chunked & embedded. You only need to process a KB after adding or changing files."
    )

# ------ Tab: Orchestrator (demo) ------
with tabs[1]:
    st.header("2) Orchestrator demo — multi-agent solve → critique → refine → synthesize")
    st.markdown(
        """
        This follows the `demo.py` flow: the question is augmented with retrieved context from both KB1 & KB2,
        then two worker agents solve, critique each other, refine, and an orchestrator synthesizes a final answer.
        """
    )

    problem_input = st.text_area("Problem statement / Question (Orchestrator)", height=160, value="What is an acoustic representation?")
    top_k = st.number_input("Top-K RAG results per KB used for context", min_value=1, max_value=20, value=5, step=1)
    run_orch = st.button("Run Orchestrator", key="run_orch")

    if run_orch:
        missing = ensure_env_ok()
        if missing:
            st.warning(f"Environment variables missing: {missing}. LLM calls will likely fail if these are required. Continue only if you understand.")
        EMB = get_embedding_model()
        if not problem_input.strip():
            st.error("Please provide a problem statement.")
        else:
            with st.spinner("Retrieving RAG context and running orchestrator..."):
                try:
                    # RAG search on both KBs
                    results1 = search_index(problem_input, EMB, top_k=top_k, idx_num=1)
                    if not results1:
                        st.info("No KB1 results (maybe KB1 index missing).")
                except Exception as e:
                    st.error(f"Failed to search KB1. Error: {e}")
                    results1 = []
                try:
                    results2 = search_index(problem_input, EMB, top_k=top_k, idx_num=2)
                    if not results2:
                        st.info("No KB2 results (maybe KB2 index missing).")
                except Exception as e:
                    st.error(f"Failed to search KB2. Error: {e}")
                    results2 = []

                combined = []
                if results1:
                    st.subheader("Top KB1 retrieved snippets")
                    for r in results1:
                        text_snippet = r.get('text')[:400].replace('\n', ' ')
                        st.markdown(f"- (score: {r.get('score', None):.4f}) {text_snippet}")
                        combined.append(r.get("text", ""))
                else:
                    st.info("No KB1 results (maybe KB1 index missing).")

                if results2:
                    st.subheader("Top KB2 retrieved snippets")
                    for r in results2:
                        text_snippet = r.get('text')[:400].replace('\n', ' ')
                        st.markdown(f"- (score: {r.get('score', None):.4f}) {text_snippet}")
                        combined.append(r.get("text", ""))
                else:
                    st.info("No KB2 results (maybe KB2 index missing).")

                # build augmented problem
                retrieved_text = "\n\n".join(combined[: max(1, top_k * 2)])
                augmented_problem = problem_input + "\n\nRetrieved Text for context:\n" + (retrieved_text or "No retrieved context available.")

                # Run orchestrator
                try:
                    orchestrator = OrchestratorAgent()
                except Exception as e:
                    st.error(f"Failed to instantiate OrchestratorAgent. Make sure imports and environment are correct. Error: {e}")
                    orchestrator = None

                if orchestrator:
                    try:
                        # The orchestrator.invoke expects a dict with "problem"
                        result = orchestrator.invoke({"problem": augmented_problem})
                        # result is expected to be a dict containing "final_answer" and debug info
                        st.success("Orchestrator finished.")
                    except Exception as e:
                        st.error(f"Orchestrator run failed: {e}")
                        result = None

                    # Display results if available
                    if result:
                        st.subheader("Final synthesized answer")
                        final_ans = result.get("final_answer") or result.get("final", result.get("output", ""))
                        st.write(final_ans)

                        st.markdown("---")
                        st.subheader("Worker-level details (debug)")

                        # Worker responses and critiques are expected keys (based on earlier project design)
                        w1_resp = result.get("worker1_response")
                        w2_resp = result.get("worker2_response")
                        w1_crit = result.get("worker1_critique")
                        w2_crit = result.get("worker2_critique")
                        iteration = result.get("iteration", None)

                        if w1_resp:
                            st.markdown("**Worker 1 final response:**")
                            st.write(w1_resp)
                        if w1_crit:
                            st.markdown("**Worker 1 critique of Worker 2:**")
                            st.write(w1_crit)

                        if w2_resp:
                            st.markdown("**Worker 2 final response:**")
                            st.write(w2_resp)
                        if w2_crit:
                            st.markdown("**Worker 2 critique of Worker 1:**")
                            st.write(w2_crit)

                        if iteration is not None:
                            st.caption(f"Orchestrator completed with iteration count: {iteration}")

# ------ Tab: Debate ------
with tabs[2]:
    st.header("3) Debate flow (Proponent vs Opponent, separate KBs)")
    st.markdown(
        """
        - Proponent (for) will use **KB1** for its retrieved context.
        - Opponent (against) will use **KB2** for its retrieved context.
        - Set number of rounds, then press **Start Debate**.
        """
    )

    topic = st.text_area("Debate topic / proposition", height=140, value="Artificial Intelligence will ultimately benefit humanity more than harm it")
    num_rounds = st.number_input("Number of rounds (each round: proponent then opponent)", min_value=1, max_value=10, value=3, step=1)
    top_k_debate = st.number_input("Top-K RAG results per KB (debate)", min_value=1, max_value=20, value=5, step=1)
    run_debate = st.button("Start Debate", key="start_debate")

    if run_debate:
        missing = ensure_env_ok()
        if missing:
            st.warning(f"Environment variables missing: {missing}. LLM calls will likely fail if these are required. Continue only if you understand.")
        EMB = get_embedding_model()
        if not topic.strip():
            st.error("Please provide a debate topic.")
        else:
            with st.spinner("Running RAG retrieval for both sides..."):
                try:
                    prop_ctx_list = search_index(topic, EMB, top_k=top_k_debate, idx_num=1)
                except Exception as e:
                    st.error(f"Failed to search index KB1. Error: {e}")
                    prop_ctx_list = []

                try:
                    opp_ctx_list = search_index(topic, EMB, top_k=top_k_debate, idx_num=2)
                except Exception as e:
                    st.error(f"Failed to search index KB2. Error: {e}")
                    opp_ctx_list = []

            prop_context = "\n\n".join([r.get("text", "") for r in prop_ctx_list]) or ""
            opp_context = "\n\n".join([r.get("text", "") for r in opp_ctx_list]) or ""

            st.subheader("Retrieved context for Proponent (KB1)")
            if prop_context:
                st.write(prop_context[:3000])
            else:
                st.info("No retrieved context for KB1. Make sure KB1 is processed and not empty.")

            st.subheader("Retrieved context for Opponent (KB2)")
            if opp_context:
                st.write(opp_context[:3000])
            else:
                st.info("No retrieved context for KB2. Make sure KB2 is processed and not empty.")

            # Instantiate moderator
            try:
                moderator = DebateModerator(topic=topic, embedding_model=EMB, num_rounds=num_rounds)
            except Exception as e:
                st.error(f"Failed to instantiate DebateModerator: {e}")
                moderator = None

            if moderator:
                # initial state structure from earlier design
                initial_state = {
                    "topic": topic,
                    "current_round": 1,
                    "max_rounds": num_rounds,
                    "proponent_arguments": [],
                    "opponent_arguments": [],
                    "proponent_last_argument": "",
                    "opponent_last_argument": "",
                    "proponent_context": prop_context,
                    "opponent_context": opp_context,
                    "debate_history": [],
                    "final_summary": ""
                }

                with st.spinner("Running debate rounds... (this may take a while depending on number of rounds and LLM speed)"):
                    try:
                        outcome = moderator.invoke(initial_state)
                        st.success("Debate complete.")
                    except Exception as e:
                        st.error(f"Debate execution failed: {e}")
                        outcome = None

                if outcome:
                    # display debate history nicely (expected from moderator.generate_summary)
                    history = outcome.get("debate_history", [])
                    if history:
                        st.subheader("Debate rounds transcript")
                        for i, entry in enumerate(history, start=1):
                            round_no = entry.get("round", i)
                            speaker = entry.get("speaker", "SPEAKER")
                            argument = entry.get("argument", "")
                            with st.expander(f"Round {round_no} — {speaker}", expanded=False):
                                st.write(argument)
                    else:
                        st.info("No debate_history returned. The moderator may produce output in other keys — showing raw outcome below.")
                        st.write(outcome)

                    # final judge summary
                    final_summary = outcome.get("final_summary") or outcome.get("judge_summary") or outcome.get("final")
                    if final_summary:
                        st.subheader("Judge's summary / final assessment")
                        st.write(final_summary)
                    else:
                        st.info("No final summary found in outcome; raw outcome printed below.")
                        st.write(outcome)

# ------ Tab: Diagnostics / Logs ------
with tabs[3]:
    st.header("Diagnostics")
    st.markdown("Environment, files, and some quick checks.")

    st.subheader("Environment variables")
    key_list = ["OPENAI_API_KEY", "OPENAI_API_BASE", "TAVILLY_API_KEY"]
    for k in key_list:
        val = os.getenv(k)
        st.write(f"{k}: {'SET' if val else 'NOT SET'}")

    st.subheader("Data directories")
    st.write(f"Project root: {ROOT}")
    st.write(f"KB1 documents: {KB1_DOCS} — contains {len(list(KB1_DOCS.iterdir())) if KB1_DOCS.exists() else 0} files")
    st.write(f"KB2 documents: {KB2_DOCS} — contains {len(list(KB2_DOCS.iterdir())) if KB2_DOCS.exists() else 0} files")

    st.subheader("Quick index existence check")
    try:
        idx1 = (DATA_DIR / "knowledge_base_1" / "embeddings" / "index.faiss")
        meta1 = (DATA_DIR / "knowledge_base_1" / "metadata" / "metadata.json")
        idx2 = (DATA_DIR / "knowledge_base_2" / "embeddings" / "index.faiss")
        meta2 = (DATA_DIR / "knowledge_base_2" / "metadata" / "metadata.json")
        st.write(f"KB1 index present: {idx1.exists()}, metadata present: {meta1.exists()}")
        st.write(f"KB2 index present: {idx2.exists()}, metadata present: {meta2.exists()}")
    except Exception as e:
        st.write("Could not check index files:", e)

    st.markdown("---")
    st.markdown("If you get errors when calling the LLMs, ensure `OPENAI_API_KEY` and `OPENAI_API_BASE` (if required by your ChatOpenAI implementation) are set in the environment where you run Streamlit.")
