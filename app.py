import os
import html
import tempfile
import traceback
import contextlib
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import configuration constants, caching functions, and backend pipeline functions
# Why: Keeping backend, caching, and frontend decoupled lets us reuse existing logic without duplication
from src.config import (
    APP_NAME,
    APP_TAGLINE,
    LANGUAGE,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    is_quota_error,
    get_quota_error_message,
)
from src.cache import (
    calculate_file_hash,
    get_cached_analysis,
    save_cached_analysis,
    save_cached_stage,
    get_cached_vector_store,
    save_cached_vector_store,
)
from src.pdf_reader import extract_text_from_pdf
from src.chunker import split_documents
from src.vector_store import build_vector_store
from src.analyzer import (
    detect_document_type,
    summarize_document,
    find_risky_clauses,
    strip_summary_filler,
)
from src.rag_chain import answer_question, normalize_citations

# Configure the Streamlit page metadata and layout
# Why: Wide layout provides ample space for tabs, cards, and comparison views
st.set_page_config(
    page_title=f"{APP_NAME} - {APP_TAGLINE}",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """
    Reads styles.css and injects it into the Streamlit app.
    
    Why: Keeping styles in a dedicated CSS file keeps Python code clean and allows rich custom styling
    like multi-color gradients, custom badges, hover transitions, and responsive card layouts.
    """
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """
    Initializes required session_state variables on the first run.
    
    Why: Streamlit reruns the whole script on every user interaction. Session state caches the
    expensive analysis results (doc_info, summary, risks, FAISS index) so they only compute once.
    """
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "doc_info" not in st.session_state:
        st.session_state.doc_info = {"doc_type": "Legal Document", "category": "other", "parties": []}
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "risky_clauses" not in st.session_state:
        st.session_state.risky_clauses = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None
    if "processed_language" not in st.session_state:
        st.session_state.processed_language = LANGUAGE
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = None
    if "active_file_id" not in st.session_state:
        st.session_state.active_file_id = None


def clear_analysis_state():
    """
    Clears all document analysis state from st.session_state.
    
    Why: Ensures that previous analysis results, detected types, summaries,
    risky clauses, and vector indexes never linger when an analysis fails
    or when switching documents.
    """
    st.session_state.analysis_complete = False
    st.session_state.doc_info = {"doc_type": "Legal Document", "category": "other", "parties": []}
    st.session_state.summary = ""
    st.session_state.risky_clauses = []
    st.session_state.vector_store = None
    st.session_state.processed_file_id = None
    st.session_state.processed_file_name = None
    st.session_state.processed_language = LANGUAGE
    st.session_state.messages = []
    st.session_state.analysis_error = None


def clean_html(raw_html: str) -> str:
    """
    Strips leading and trailing whitespace from every line of an HTML string.
    
    Why: Streamlit's Markdown parser interprets any line with 4 or more leading spaces
    as an indented code block (<pre><code>). Stripping leading whitespace from every line
    guarantees that the browser renders HTML elements instead of raw source code.
    """
    return "\n".join(line.strip() for line in raw_html.strip().splitlines() if line.strip())


def render_disclaimer():
    """
    Renders a permanent, prominent legal disclaimer across all views.
    
    Why: Mandatory legal notice to protect users and comply with professional ethics.
    """
    disclaimer_html = clean_html("""
        <div class="disclaimer-banner">
            <span style="font-size: 1.25rem;">⚠️</span>
            <p class="disclaimer-text">
                <strong>AI-generated explanation.</strong> This is not legal advice. 
                Laws and contract interpretations vary; please consult a qualified lawyer before signing any legal agreement.
            </p>
        </div>
    """)
    st.markdown(disclaimer_html, unsafe_allow_html=True)


def render_hero_header():
    """
    Displays the hero banner with app title and tagline.
    
    Why: Creates a clean, professional appearance.
    """
    hero_html = clean_html(f"""
        <div class="hero-header">
            <h1 class="hero-title">{APP_NAME}</h1>
            <p class="hero-tagline">
                {APP_TAGLINE}
            </p>
        </div>
    """)
    st.markdown(hero_html, unsafe_allow_html=True)


def render_empty_state(file_name: str | None = None):
    """
    Renders the welcoming landing view when no document has been analyzed yet,
    or a call-to-action prompt when a document has been selected.
    """
    if file_name:
        hint_title = f"Ready to analyze {html.escape(file_name)}?"
        hint_desc = (
            "Click <strong>'Analyze Document'</strong> in the sidebar to begin plain-language analysis."
        )
    else:
        hint_title = "Ready to understand your document?"
        hint_desc = (
            "Upload any legal agreement or contract (rental agreement, employment contract, loan sanction, "
            "insurance policy, NDA, or terms of service) from the sidebar and click <strong>'Analyze Document'</strong> to begin."
        )

    empty_html = clean_html(f"""
        <div class="feature-grid">
            <div class="feature-card feature-card-indigo">
                <div class="feature-icon-chip-indigo">📑</div>
                <h3 class="feature-title">Plain-Language Summary</h3>
                <p class="feature-desc">
                    Summarizes complex contracts into 7 structured sections: purpose, parties, dates, financial terms, obligations, remedies, and practical takeaways.
                </p>
            </div>
            <div class="feature-card feature-card-rose">
                <div class="feature-icon-chip-rose">🚨</div>
                <h3 class="feature-title">Risky Clause Detection</h3>
                <p class="feature-desc">
                    Identifies potential risks, unbalanced terms, harsh penalties, and one-sided clauses across agreements.
                </p>
            </div>
            <div class="feature-card feature-card-teal">
                <div class="feature-icon-chip-teal">💬</div>
                <h3 class="feature-title">Interactive Q&A</h3>
                <p class="feature-desc">
                    Ask questions about the document and receive accurate answers grounded strictly in the text with page citations.
                </p>
            </div>
        </div>
        
        <div class="upload-hint-box">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">👈</div>
            <div class="upload-hint-title">{hint_title}</div>
            <p style="margin: 0; font-size: 0.95rem;">
                {hint_desc}
            </p>
        </div>
    """)
    st.markdown(empty_html, unsafe_allow_html=True)


def process_uploaded_document(uploaded_file, language: str = LANGUAGE):
    """
    Saves the uploaded PDF to a temporary file, checks local cache for instant load,
    or runs full pipeline with stage caching: detects document type, generates structured summary,
    detects risky clauses, indexes passages into FAISS, and caches each stage immediately.
    
    Why stage caching:
    Saves detected doc type, summary, and risky clauses to cache as soon as each finishes.
    If vector store embedding fails or hits a rate limit, a retry skips all completed Gemini calls
    and only redoes the vector store indexing.
    """
    # 1. Clear ALL analysis state immediately before any cache lookup or API call
    clear_analysis_state()

    temp_file_path = None
    try:
        file_bytes = uploaded_file.getvalue()
        file_hash = calculate_file_hash(file_bytes)

        # 1. Check local JSON cache first
        cached_data = get_cached_analysis(file_hash, language) or {}
        cached_doc_info = cached_data.get("doc_info")
        cached_summary = cached_data.get("summary")
        cached_risks = cached_data.get("risky_clauses")

        # Load cached vector store from disk if already built
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
        vector_store = get_cached_vector_store(file_hash, embeddings)

        # If ALL stages are already completed in cache, load instantly with 0 API calls
        if cached_doc_info and cached_summary and cached_risks and vector_store is not None:
            with st.spinner("⚡ Loading analysis from local cache (0 API calls)..."):
                st.session_state.doc_info = cached_doc_info
                st.session_state.summary = strip_summary_filler(cached_summary)
                st.session_state.risky_clauses = cached_risks
                st.session_state.vector_store = vector_store
                st.session_state.messages = []
                st.session_state.processed_file_name = uploaded_file.name
                st.session_state.processed_language = language
                st.session_state.analysis_error = None
                # Set processed_file_id and analysis_complete as the LAST step
                st.session_state.processed_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                st.session_state.analysis_complete = True
                st.toast("⚡ Analysis loaded from local cache (0 Gemini calls)", icon="✅")
                return

        # 2. Need processing for missing stages
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_file_path = tmp.name

        # Extract text from PDF
        with st.spinner("📄 Reading your document and extracting pages..."):
            documents = extract_text_from_pdf(temp_file_path)

        # Stage 1: Document type & category
        if cached_doc_info:
            doc_info = cached_doc_info
        else:
            with st.spinner("🔍 Identifying document type and category..."):
                doc_info = detect_document_type(documents)
                # Stage cache immediately
                save_cached_stage(file_hash=file_hash, language=language, doc_info=doc_info)

        doc_type = doc_info.get("doc_type", "Legal Document")
        category = doc_info.get("category", "other")

        # Stage 2: Plain-language summary
        if cached_summary:
            summary_text = strip_summary_filler(cached_summary)
        else:
            with st.spinner(f"📝 Generating plain-language summary for {doc_type} in {language}..."):
                summary_text = strip_summary_filler(
                    summarize_document(documents, language=language, doc_type=doc_type)
                )
                # Stage cache immediately
                save_cached_stage(file_hash=file_hash, language=language, summary=summary_text)

        # Stage 3: Risky clauses
        if cached_risks:
            risks_data = cached_risks
        else:
            with st.spinner(f"🚨 Scanning for risky clauses tailored to {doc_type}..."):
                risks_data = find_risky_clauses(
                    documents,
                    language=language,
                    doc_type=doc_type,
                    category=category
                )
                # Stage cache immediately
                save_cached_stage(file_hash=file_hash, language=language, risky_clauses=risks_data)

        # Stage 4: Vector store
        if vector_store is None:
            chunks = split_documents(documents)

            @contextlib.contextmanager
            def spinner_progress(part: int, total: int):
                with st.spinner(f"🧠 Indexing part {part} of {total}..."):
                    yield

            vector_store = build_vector_store(chunks, progress_callback=spinner_progress)
            # Save the FAISS index to disk immediately after it is built
            save_cached_vector_store(file_hash, vector_store)

        # Set new state ONLY after the whole pipeline succeeds
        st.session_state.doc_info = doc_info
        st.session_state.summary = summary_text
        st.session_state.risky_clauses = risks_data
        st.session_state.vector_store = vector_store
        st.session_state.messages = []
        st.session_state.processed_file_name = uploaded_file.name
        st.session_state.processed_language = language
        st.session_state.analysis_error = None
        # Set processed_file_id and analysis_complete as the LAST step
        st.session_state.processed_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        st.session_state.analysis_complete = True

    except ValueError as ve:
        clear_analysis_state()
        err_msg = f"⚠️ Document Error: {str(ve)}"
        st.session_state.analysis_error = err_msg
    except Exception as e:
        clear_analysis_state()
        traceback.print_exc()
        if is_quota_error(e):
            quota_msg = get_quota_error_message(e)
            if quota_msg:
                err_msg = quota_msg
            else:
                err_msg = (
                    "⚠️ The AI service is currently experiencing high demand or encountered a temporary connection issue. "
                    "Please wait a moment and try clicking 'Analyze Document' again."
                )
        else:
            err_msg = (
                "⚠️ The AI service is currently experiencing high demand or encountered a temporary connection issue. "
                "Please wait a moment and try clicking 'Analyze Document' again."
            )
        st.session_state.analysis_error = err_msg
    finally:
        # Clean up temporary file safely
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


def render_summary_tab(summary: str):
    """
    Renders the document summary inside a styled container.
    
    Why: Formats the summary text safely, ensuring clean typography and styling.
    """
    if not summary:
        st.info("No summary available. Please analyze a document first.")
        return
        
    cleaned_summary = strip_summary_filler(summary)
    with st.container(border=True):
        st.markdown(cleaned_summary)


def render_risky_clauses_tab(risky_clauses: list[dict]):
    """
    Renders the risky clauses cards, sorted with High risk first.
    
    Why: Sorting High risk first ensures critical pitfalls (like large forfeitures)
    are immediately visible to the user before medium/low items.
    """
    if not risky_clauses:
        st.success("🎉 No high-risk or one-sided clauses were flagged in this document!")
        return

    # Sort High first, then Medium, then Low
    risk_priority = {"High": 0, "Medium": 1, "Low": 2}
    sorted_clauses = sorted(
        risky_clauses,
        key=lambda x: risk_priority.get(str(x.get("risk_level", "Low")).capitalize(), 3)
    )

    total_count = len(sorted_clauses)
    high_count = sum(1 for c in sorted_clauses if str(c.get("risk_level", "")).capitalize() == "High")
    
    # Header counter with zero leading line indentation
    header_html = clean_html(f"""
        <div class="risk-count-header">
            <div>
                <span class="risk-count-title">⚠️ {total_count} clauses need your attention</span>
                <span style="font-size: 0.85rem; color: #64748B; margin-left: 0.5rem;">
                    ({high_count} marked as High Risk)
                </span>
            </div>
            <span style="font-size: 0.85rem; color: #64748B;">Sorted by risk severity</span>
        </div>
    """)
    st.markdown(header_html, unsafe_allow_html=True)

    # Render each clause card with sanitized HTML (Security Note)
    for clause_item in sorted_clauses:
        risk_level = str(clause_item.get("risk_level", "Medium")).capitalize()
        badge_class = "badge-medium"
        card_class = "risk-card-medium"
        
        if risk_level == "High":
            badge_class = "badge-high"
            card_class = "risk-card-high"
        elif risk_level == "Low":
            badge_class = "badge-low"
            card_class = "risk-card-low"

        # Sanitize all dynamic text with html.escape to prevent HTML/XSS injection
        safe_clause = html.escape(str(clause_item.get("clause", "Not specified")))
        safe_page = html.escape(str(clause_item.get("page", "?")))
        safe_why = html.escape(str(clause_item.get("why_it_matters", "Review this clause carefully.")))
        safe_ask = html.escape(str(clause_item.get("what_to_ask", "Ask for clarification.")))

        # Clean HTML string with zero leading indentation on any line
        card_html = clean_html(f"""
            <div class="risk-card {card_class}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div>
                        <span class="badge {badge_class}">{risk_level} Risk</span>
                        <span class="badge badge-page">Page {safe_page}</span>
                    </div>
                </div>
                
                <div class="clause-quote">
                    "{safe_clause}"
                </div>
                
                <div class="clause-section-label">Why it matters:</div>
                <div class="clause-section-content">{safe_why}</div>
                
                <div class="what-to-ask-box">
                    <div class="what-to-ask-title">
                        <span>💬</span> <span>What to ask before signing:</span>
                    </div>
                    <div style="font-size: 0.92rem; color: #0F766E; font-weight: 500;">
                        {safe_ask}
                    </div>
                </div>
            </div>
        """)
        st.markdown(card_html, unsafe_allow_html=True)


def render_qa_tab(language: str = LANGUAGE):
    """
    Renders the interactive conversational Q&A chat interface.
    
    Why: Uses st.chat_message and st.chat_input so users can clarify clauses, follow up on terms,
    and receive cited answers grounded strictly in the document text.
    Uses a dedicated chat_area container above st.chat_input so the input always stays at the bottom.
    Handles transient errors with an automatic retry after ~5s and adds an assistant error message
    on failure so user questions are never left orphaned without a response.
    """
    vector_store = st.session_state.vector_store
    if not vector_store:
        st.info("Please upload and analyze a document before asking questions.")
        return

    qa_intro = clean_html(
        "<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 1rem;'>"
        "Ask questions about your document. "
        "Answers cite exact page numbers and refuse to guess missing facts."
        "</p>"
    )
    st.markdown(qa_intro, unsafe_allow_html=True)

    # 1. Create ONE container at the top of the tab for messages
    chat_area = st.container()

    # 2. Render all messages from st.session_state.messages inside chat_area
    with chat_area:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 3. Call st.chat_input AFTER the container, so the input is always below the conversation
    user_prompt = st.chat_input("Ask a question about this legal document...")

    # 4. When the user submits
    if user_prompt:
        # 5. Make sure each question can only be added to session_state once (no duplicates when a rerun happens)
        if (
            len(st.session_state.messages) >= 2
            and st.session_state.messages[-2].get("role") == "user"
            and st.session_state.messages[-2].get("content") == user_prompt
            and st.session_state.messages[-1].get("role") == "assistant"
        ):
            return

        already_has_user_msg = (
            bool(st.session_state.messages)
            and st.session_state.messages[-1].get("role") == "user"
            and st.session_state.messages[-1].get("content") == user_prompt
        )

        # Append the user message to session_state first
        if not already_has_user_msg:
            st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Then, inside chat_area, show the user message and an assistant message with a spinner
        # ("Finding answer from document...") while the answer is computed
        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Finding answer from document..."):
                    history_context = [
                        (m["content"], "") if m["role"] == "user" else ("", m["content"])
                        for m in st.session_state.messages[:-1]
                    ]

                    bot_reply = None
                    last_error = None

                    # Attempt 1
                    try:
                        bot_reply = answer_question(
                            vectorstore=vector_store,
                            question=user_prompt,
                            language=language,
                            chat_history=history_context
                        )
                    except Exception as e:
                        last_error = e
                        err_str = str(e).lower()
                        # Check for 503 or transient errors to retry once after ~5 seconds
                        is_transient = (
                            "503" in err_str
                            or "unavailable" in err_str
                            or "transient" in err_str
                            or "overloaded" in err_str
                            or "connection" in err_str
                            or "timeout" in err_str
                            or "timed out" in err_str
                        )
                        if is_transient:
                            import time
                            time.sleep(5)
                            try:
                                bot_reply = answer_question(
                                    vectorstore=vector_store,
                                    question=user_prompt,
                                    language=language,
                                    chat_history=history_context
                                )
                                last_error = None
                            except Exception as retry_err:
                                last_error = retry_err

                    if bot_reply:
                        bot_reply = normalize_citations(bot_reply)
                        st.markdown(bot_reply)
                        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    else:
                        traceback.print_exc()
                        if last_error and is_quota_error(last_error):
                            quota_msg = get_quota_error_message(last_error)
                            friendly_error = quota_msg or "Daily AI usage limit reached. Please try again later."
                        else:
                            friendly_error = (
                                "I'm sorry, I encountered a temporary issue answering your question. "
                                "Please ask your question again in a moment."
                            )
                        st.error(friendly_error)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ {friendly_error}"
                        })

        # Call st.rerun() so the whole conversation is redrawn once from session_state in the correct order
        st.rerun()


def main():
    """
    Main application orchestrator.
    
    Why: Sets up the sidebar controls, checks session states, and switches between
    the empty landing state and the full 3-tab analysis view.
    """
    # 1. Load styles and session state
    load_custom_css()
    init_session_state()

    # 2. Disclaimer on every screen
    render_disclaimer()

    # 3. Sidebar setup
    with st.sidebar:
        st.markdown(clean_html(f'<div class="sidebar-title">⚖️ {APP_NAME}</div>'), unsafe_allow_html=True)
        st.markdown(
            clean_html(f'<div class="sidebar-tagline">{APP_TAGLINE}</div>'),
            unsafe_allow_html=True
        )

        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a rental agreement, employment contract, loan sanction, insurance policy, NDA, or terms."
        )

        # Check if the user changed the file
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}" if uploaded_file else None
        selection_changed = (current_file_id != st.session_state.active_file_id)

        # When the user picks a new file, hide the old results right away
        if selection_changed:
            clear_analysis_state()
            st.session_state.active_file_id = current_file_id

        # Show "Click 'Analyze Document' to begin analysis" only when a file is selected
        # and has NOT been analyzed yet (processed_file_id differs from current file id).
        # Hide it when results for the current file are on screen.
        file_needs_analysis = (
            uploaded_file is not None
            and st.session_state.get("processed_file_id") != current_file_id
            and not st.session_state.get("analysis_error")
        )
        if file_needs_analysis:
            st.info("👆 Click **'Analyze Document'** to begin analysis.")

        analyze_button = st.button(
            "🔍 Analyze Document",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None)
        )

        st.markdown("---")
        st.markdown("### How It Works")
        step_flow_html = clean_html("""
            <div class="step-flow">
                <div class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-text"><strong>Upload:</strong> Provide your PDF legal document.</div>
                </div>
                <div class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-text"><strong>Understand:</strong> Review the summary and flagged risky clauses.</div>
                </div>
                <div class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-text"><strong>Ask:</strong> Ask questions and get cited answers.</div>
                </div>
            </div>
        """)
        st.markdown(step_flow_html, unsafe_allow_html=True)

    # 4. Handle "Analyze Document" click
    if analyze_button and uploaded_file is not None:
        process_uploaded_document(uploaded_file, LANGUAGE)
        st.rerun()

    # 5. Render Main Content Area
    # Only render tabs if analysis_complete is True for the CURRENT file
    is_analysis_valid_for_current = (
        st.session_state.get("analysis_complete", False) is True
        and st.session_state.get("vector_store") is not None
        and uploaded_file is not None
        and st.session_state.get("processed_file_id") == current_file_id
    )

    if is_analysis_valid_for_current:
        doc_info = st.session_state.get("doc_info", {})
        doc_type = doc_info.get("doc_type", "Legal Document")
        parties = doc_info.get("parties", [])
        parties_str = f" • Parties: {', '.join(parties)}" if parties else ""
        doc_name = st.session_state.get("processed_file_name", "Active Document")

        # Show mini header above analysis tabs with detected doc-type badge
        # Header is strictly built from the same state as the results
        results_header = clean_html(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.25rem;">
                        <h2 style="font-size: 1.4rem; font-weight: 800; color: #1E1B4B; margin: 0;">Analysis Results</h2>
                        <span class="doc-type-chip">📑 Detected: {html.escape(doc_type)}</span>
                    </div>
                    <span style="font-size: 0.85rem; color: #64748B;">
                        Document: {html.escape(doc_name)}{html.escape(parties_str)}
                    </span>
                </div>
            </div>
        """)
        st.markdown(results_header, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📑 Summary", "🚨 Risky Clauses", "💬 Ask Questions"])

        with tab1:
            render_summary_tab(st.session_state.summary)

        with tab2:
            render_risky_clauses_tab(st.session_state.risky_clauses)

        with tab3:
            render_qa_tab(LANGUAGE)

    elif st.session_state.get("analysis_error"):
        # If anything fails, leave the results area empty and show only the error message
        st.error(st.session_state.analysis_error)

    else:
        # Show Hero Header and Welcome Landing State / Prompt to analyze
        render_hero_header()
        render_empty_state(uploaded_file.name if uploaded_file else None)


if __name__ == "__main__":
    main()
