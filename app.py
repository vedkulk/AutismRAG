import logging

import streamlit as st

from rag_pipeline import RAGPipeline, RAGConfig
from dfs import sweep_orphans


# ── Logging policy ─────────────────────────────────────────────────────────
# Patient data must never persist on disk. Strip any FileHandler that some
# library may have attached and route everything to stderr only.
def _enforce_in_memory_logging() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.StreamHandler())
    root.setLevel(logging.INFO)


_enforce_in_memory_logging()


# ── Must be the very first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="ASD Clinical Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap');

    /* ── Global ───────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background: #f5f6f7; }
    #MainMenu, footer { visibility: hidden; }
    /* Hide the header toolbar items but keep the sidebar toggle visible */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    /* Hide deploy button and other toolbar clutter, keep collapse button */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e6ea !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1.75rem 1.25rem 2rem !important;
    }

    /* Sidebar brand */
    .sidebar-brand {
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #e2e6ea;
        margin-bottom: 1.5rem;
    }
    .sidebar-brand .name {
        font-family: 'IBM Plex Serif', serif;
        font-size: 1.1rem;
        color: #1a2332;
        display: block;
        line-height: 1.3;
    }
    .sidebar-brand .name em {
        color: #1d6fa4;
        font-style: italic;
    }
    .sidebar-brand .tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: #8a9bb0;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 5px;
        display: block;
    }

    /* Sidebar section label */
    .sb-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8a9bb0;
        margin: 1.5rem 0 0.5rem;
        display: block;
    }

    /* Report status badge */
    .report-pill {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: #f0f7ff;
        border: 1px solid #c8dff0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 8px;
    }
    .report-pill .live-dot {
        width: 8px; height: 8px;
        background: #1d6fa4;
        border-radius: 50%;
        flex-shrink: 0;
        margin-top: 4px;
    }
    .report-pill .rname {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #1d6fa4;
        font-weight: 500;
        word-break: break-all;
    }
    .report-pill .rstats {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: #5a88a8;
        margin-top: 2px;
    }

    /* Sidebar info box */
    .sb-info {
        background: #f8f9fa;
        border: 1px solid #e2e6ea;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        color: #8a9bb0;
        line-height: 1.7;
        margin-top: 0.75rem;
    }

    /* ── Main content area ────────────────────────────── */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    /* Remove Streamlit's built-in top gap */
    section[data-testid="stMain"] > div.block-container {
        padding-top: 0 !important;
    }
    [data-testid="stAppViewContainer"] > section > div {
        padding-top: 0 !important;
    }

    /* ── Chat container ───────────────────────────────── */
    /* Constrain the entire main column so every Streamlit element (static
       msg-row, streaming placeholder, expanders, welcome) shares the same
       width/padding. Eliminates the "left margin jog" the streaming bubble
       used to do when only it was wrapped in .chat-wrap. */
    section.main > div.block-container,
    [data-testid="stMain"] [data-testid="stMainBlockContainer"],
    .main .block-container {
        max-width: 820px !important;
        padding: 1.25rem 1.5rem 6rem !important;
    }
    .chat-wrap {
        /* Legacy class — layout now lives on the block container. */
        padding: 0;
        margin: 0;
        max-width: none;
    }

    /* ── Welcome screen ───────────────────────────────── */
    .welcome {
        max-width: 560px;
        margin: 1.5rem auto 0;
        text-align: center;
    }
    .welcome-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    .welcome h1 {
        font-family: 'IBM Plex Serif', serif !important;
        font-size: 1.75rem !important;
        font-weight: 400 !important;
        color: #1a2332 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.5rem !important;
    }
    .welcome h1 em { color: #1d6fa4; font-style: italic; }
    .welcome p {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        color: #5a6a7a;
        line-height: 1.7;
        margin-bottom: 2rem;
    }
    .welcome-steps {
        display: flex;
        flex-direction: column;
        gap: 0;
        text-align: left;
        background: #ffffff;
        border: 1px solid #e2e6ea;
        border-radius: 10px;
        overflow: hidden;
    }
    .welcome-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        color: #3a4a5a;
        line-height: 1.55;
        padding: 0.85rem 1.25rem;
        border-bottom: 1px solid #e2e6ea;
    }
    .welcome-step:last-child { border-bottom: none; }
    .welcome-step .num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        background: #eef4fa;
        color: #1d6fa4;
        border-radius: 4px;
        padding: 2px 7px;
        flex-shrink: 0;
        margin-top: 1px;
        font-weight: 500;
    }

    /* ── Chat messages ────────────────────────────────── */
    .msg-row {
        display: flex;
        gap: 0.9rem;
        margin-bottom: 1.5rem;
        align-items: flex-start;
    }
    .msg-row.user { flex-direction: row-reverse; }

    .avatar {
        width: 34px; height: 34px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        margin-top: 2px;
        letter-spacing: 0.05em;
    }
    .avatar.ai {
        background: #eef4fa;
        border: 1.5px solid #c8dff0;
        color: #1d6fa4;
    }
    .avatar.user {
        background: #f0f4f8;
        border: 1.5px solid #d8dfe6;
        color: #4a6070;
    }

    .bubble {
        max-width: 86%;
        padding: 0.85rem 1.1rem;
        border-radius: 12px;
        font-size: 0.9rem;
        line-height: 1.75;
    }
    /* AI answer is fixed width and matches the sources/retrieval-meta width
       so the column reads as one stack. User bubble keeps its shrink-to-fit
       right-aligned behavior. */
    .bubble.ai {
        background: #ffffff;
        border: 1px solid #e2e6ea;
        border-top-left-radius: 4px;
        color: #2a3540;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        flex: 1 1 auto;
        width: calc(100% - 34px - 0.9rem);
        max-width: calc(100% - 34px - 0.9rem);
        box-sizing: border-box;
    }
    .bubble.user {
        background: #1d6fa4;
        border: none;
        border-top-right-radius: 4px;
        color: #ffffff;
        box-shadow: 0 1px 3px rgba(29,111,164,0.2);
    }

    /* Markdown inside bubbles */
    .bubble p { margin: 0 0 0.55rem; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble.ai strong { color: #1a2332; font-weight: 600; }
    .bubble.user strong { color: #ffffff; font-weight: 600; }
    .bubble code {
        background: #eef4fa;
        color: #1d6fa4;
        padding: 1px 5px;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82em;
    }
    .bubble ul, .bubble ol {
        padding-left: 1.25rem;
        margin: 0.4rem 0;
    }
    .bubble li { margin-bottom: 0.2rem; }

    /* Streaming cursor */
    .cursor {
        display: inline-block;
        width: 2px; height: 1em;
        background: #1d6fa4;
        margin-left: 3px;
        vertical-align: text-bottom;
        animation: blink 0.9s step-end infinite;
    }
    @keyframes blink {
        0%,100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* ── Timestamp ────────────────────────────────────── */
    .msg-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        color: #aab8c8;
        margin-top: 0.25rem;
        padding: 0 0.25rem;
    }
    .msg-row.user .msg-meta { text-align: right; }

    /* ── Sources expander ─────────────────────────────── */
    /* Width/margins are owned by the later rule near the bottom of the
       stylesheet so the bubble, sources bar, and retrieval-meta share one
       fixed width. Keep only the visual properties here that aren't about
       sizing. */
    [data-testid="stExpander"] {
        overflow: hidden !important;
    }
    [data-testid="stExpander"] details summary {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #8a9bb0;
        letter-spacing: 0.08em;
        padding: 0.55rem 0.9rem;
    }
    [data-testid="stExpander"] details summary:hover { color: #1d6fa4; }
    [data-testid="stExpander"] details[open] summary { color: #1d6fa4; }

    /* Source chunk card */
    .src-card {
        background: #ffffff;
        border: 1px solid #e2e6ea;
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.82rem;
        color: #4a6070;
        line-height: 1.65;
    }
    .src-card-header {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.45rem;
    }
    .badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 2px 7px;
        border-radius: 3px;
        display: inline-block;
        font-weight: 500;
    }
    .badge-report { background: #eef4fa; color: #1d6fa4; border: 1px solid #c8dff0; }
    .badge-kb     { background: #f0faf4; color: #2a8a56; border: 1px solid #b8dfc8; }
    .src-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: #a0b0c0;
        margin-left: auto;
    }

    /* ── Chat input area ──────────────────────────────── */
    /* Streamlit always emits an empty [data-testid="stBottom"] container,
       even when no chat_input is mounted. Keep it transparent so an empty
       page never shows a leftover bar at the bottom. */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent !important;
        border-top: none !important;
        box-shadow: none !important;
    }
    /* Bar styling moves onto the chat input itself — only renders when
       st.chat_input(...) was actually called this run. */
    [data-testid="stChatInput"] {
        background: #f5f6f7 !important;
        border-top: 1px solid #e2e6ea !important;
    }
    /* Container: flex row, vertically centred */
    [data-testid="stChatInput"] > div {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        background: transparent !important;
        border: none !important;
    }
    .stChatInput textarea,
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: 1px solid #d0d8e0 !important;
        border-radius: 8px !important;
        color: #1a2332 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
        caret-color: #1d6fa4 !important;
        /* No right padding needed — button is now outside the textarea */
        padding: 0.7rem 1rem !important;
        max-height: 160px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        line-height: 1.5 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #1d6fa4 !important;
        box-shadow: 0 0 0 3px rgba(29,111,164,0.1) !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #a0b0c0 !important;
    }
    /* Kill the inner baseweb wrapper border that was bleeding through as
       a dark outline around the chat input. */
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* Submit button — fixed square, vertically centred beside textarea */
    [data-testid="stChatInput"] button {
        position: static !important;
        top: auto !important; right: auto !important; bottom: auto !important;
        transform: none !important;
        background: #1d6fa4 !important;
        border-radius: 7px !important;
        color: #ffffff !important;
        border: none !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #165d8a !important;
    }
    [data-testid="stChatInput"] button svg {
        width: 16px !important;
        height: 16px !important;
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.15s;
    }
    .stButton > button[kind="primary"] {
        background: #1d6fa4; border: none; color: #ffffff;
    }
    .stButton > button[kind="primary"]:hover {
        background: #165d8a;
        box-shadow: 0 2px 8px rgba(29,111,164,0.25);
    }
    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid #d0d8e0;
        color: #5a6a7a;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #1d6fa4;
        color: #1d6fa4;
        background: #f0f7ff;
    }

    /* ── File uploader ────────────────────────────────── */
    /* Nuclear override — Streamlit uses deeply nested divs with inline dark styles */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        background-color: #f0f4f8 !important;
        color: #7a8a9a !important;
    }
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #c8d4e0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stFileUploader"]:hover,
    [data-testid="stFileUploader"]:focus-within {
        border-color: #1d6fa4 !important;
    }
    [data-testid="stFileUploader"] small {
        font-size: 0.68rem !important;
        color: #a0b0c0 !important;
    }
    /* Override the Browse button specifically */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #ffffff !important;
        border: 1px solid #c8d4e0 !important;
        border-radius: 5px !important;
        color: #1d6fa4 !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploaderDropzone"] button * {
        color: #1d6fa4 !important;
        background-color: transparent !important;
    }
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #eef4fa !important;
        border-color: #1d6fa4 !important;
    }

    /* ── Alerts ───────────────────────────────────────── */
    .stAlert {
        background: #f8f9fa !important;
        border: 1px solid #e2e6ea !important;
        border-radius: 7px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        color: #4a6070 !important;
    }

    /* ── Spinner ──────────────────────────────────────── */
    [data-testid="stSpinner"] > div > div {
        border-top-color: #1d6fa4 !important;
    }

    /* ── Typing / loading indicator ──────────────────── */
    .typing-row {
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        margin-bottom: 1.5rem;
    }
    .typing-bubble {
        background: #ffffff;
        border: 1px solid #e2e6ea;
        border-radius: 12px;
        border-top-left-radius: 4px;
        padding: 0.85rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex;
        align-items: center;
        gap: 5px;
        height: 44px;
    }
    .typing-dot {
        width: 7px;
        height: 7px;
        background: #c8d4e0;
        border-radius: 50%;
        animation: typingBounce 1.3s ease-in-out infinite;
    }
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.18s; }
    .typing-dot:nth-child(3) { animation-delay: 0.36s; }
    @keyframes typingBounce {
        0%, 60%, 100% { transform: translateY(0); background: #c8d4e0; }
        30%            { transform: translateY(-6px); background: #1d6fa4; }
    }

    /* ── Sources expander ─────────────────────────────── */
    /* Force light theme on the expander; some Streamlit builds default to
       a dark surface when the panel is open. Also align horizontally with
       the AI bubble: left edge at avatar+gap (34px + 0.9rem ≈ 48px),
       right edge near the bubble's. */
    /* Sources bar and retrieval-meta share the AI bubble's exact fixed
       width: column-content width minus avatar(34px) + gap(0.9rem). All
       three line up as one column. */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e6ea !important;
        border-radius: 8px !important;
        margin-left: calc(34px + 0.9rem) !important;
        margin-right: 0 !important;
        width: calc(100% - 34px - 0.9rem) !important;
        max-width: calc(100% - 34px - 0.9rem) !important;
        box-sizing: border-box !important;
        margin-top: -0.75rem !important;
        margin-bottom: 0.4rem !important;
    }
    .retrieval-meta {
        margin-left: calc(34px + 0.9rem);
        margin-right: 0;
        width: calc(100% - 34px - 0.9rem);
        max-width: calc(100% - 34px - 0.9rem);
        box-sizing: border-box;
        padding: 4px 0 0;
        margin-bottom: 1.25rem;
        font-size: 0.78rem;
        color: #9e9e9e;
        letter-spacing: 0.3px;
        text-align: left;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details[open],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        background: #ffffff !important;
        color: #1a2332 !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #3a4a5a !important;
        padding: 0.6rem 0.85rem !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #f5f6f7 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0 0.85rem 0.75rem !important;
    }

    /* ── Disclaimer ───────────────────────────────────── */
    .disclaimer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        color: #b0bcc8;
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        letter-spacing: 0.05em;
        line-height: 1.7;
    }

    /* ── Divider between welcome and chat ─────────────── */
    .chat-date-divider {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        margin-top: 0.25rem;
    }
    .chat-date-divider hr {
        flex: 1;
        border: none;
        border-top: 1px solid #e2e6ea;
        margin: 0;
    }
    .chat-date-divider span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: #b0bcc8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "rag_pipeline":   None,
        "messages":       [],   # list of {role, content, sources}
        "report_loaded":  False,
        "report_meta":    None,
        "orphans_swept":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Wipe any tempdirs left by prior crashed sessions — runs once per app boot.
    if not st.session_state["orphans_swept"]:
        try:
            sweep_orphans()
        except Exception:
            pass
        st.session_state["orphans_swept"] = True


def get_pipeline() -> RAGPipeline:
    if st.session_state["rag_pipeline"] is None:
        config = RAGConfig.from_ini("config.ini")
        st.session_state["rag_pipeline"] = RAGPipeline(config=config)
    return st.session_state["rag_pipeline"]


def end_session() -> None:
    """Wipe everything patient-related. Called by the End Session button and
    implicitly when the user closes the tab (best effort, via atexit hooks
    inside the pipeline)."""
    pipeline = st.session_state.get("rag_pipeline")
    if pipeline is not None:
        try:
            pipeline.end_session()
        except Exception:
            pass
    st.session_state["rag_pipeline"] = None
    st.session_state["messages"] = []
    st.session_state["report_loaded"] = False
    st.session_state["report_meta"] = None


def is_clearly_clinical(question: str) -> bool:
    """
    Cheap heuristic to make sure obviously clinical child/ASD questions
    are never rejected even if the LLM classifier is uncertain.
    """
    q = (question or "").lower()
    clinical_terms = [
        "child", "kid", "toddler", "preschool", "school-age", "adolescent",
        "development", "developmental", "milestone", "speech", "language",
        "social", "communication", "interaction", "behaviour", "behavior",
        "eye contact", "play", "school performance", "sensory", "stimming",
        "repetitive", "autism", "asd",
        "patient", "report", "observation", "screening", "diagnosis",
        "symptom", "concern", "assessment", "evaluate", "finding",
    ]
    return any(term in q for term in clinical_terms)


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <span class="name">ASD <em>Clinical</em> Assistant</span>
            <span class="tag">DSM-5 · ADOS-2 · ASD Literature</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="sb-label">Patient Report</span>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload PDF report",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded and st.button("Load report", type="primary", use_container_width=True):
            pipeline = get_pipeline()
            with st.spinner("Processing report…"):
                meta = pipeline.load_report_from_uploaded_file(uploaded)
                st.session_state["report_loaded"] = True
                st.session_state["report_meta"]   = meta
                st.session_state["messages"] = []
                st.rerun()

        meta = st.session_state.get("report_meta")
        if meta:
            st.markdown(f"""
            <div class="report-pill">
                <div class="live-dot"></div>
                <div>
                    <div class="rname">{meta.filename}</div>
                    <div class="rstats">{meta.num_pages} pages · {meta.num_chunks} chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state["messages"]:
            st.markdown('<span class="sb-label">Conversation</span>', unsafe_allow_html=True)
            if st.button("Clear chat", type="secondary", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

        # End session is only meaningful once a report is loaded.
        if st.session_state["report_loaded"]:
            st.markdown('<span class="sb-label">Session</span>', unsafe_allow_html=True)
            if st.button("End session", type="secondary", use_container_width=True):
                end_session()
                st.rerun()

        st.markdown("""
        <div class="sb-info">
            Reports are encrypted, fragmented, and held only in a system tempdir
            for the duration of this session. Loading a new report, ending the
            session, or closing the tab wipes all patient data.
        </div>
        <div class="disclaimer">
            For educational &amp; decision-support use only.<br>
            Not a substitute for formal diagnosis.
        </div>
        """, unsafe_allow_html=True)


def md_to_html(text: str) -> str:
    """Convert a small subset of markdown to HTML for safe injection into bubbles."""
    import re
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic (not inside bold)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Numbered headings like "1. Title" → bold line
    text = re.sub(r'(?m)^(\d+)\.\s+\*\*(.+?)\*\*', r'<strong>\1. \2</strong>', text)
    # Paragraph breaks (double newline → </p><p>)
    paragraphs = re.split(r'\n{2,}', text.strip())
    html = "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())
    return html


# ══════════════════════════════════════════════════════════════════════════
#  RENDER A SINGLE MESSAGE
# ══════════════════════════════════════════════════════════════════════════

def render_message(msg: dict, is_streaming: bool = False):
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources")

    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user">You</div>
            <div class="bubble user">{content}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        cursor_html   = '<span class="cursor"></span>' if is_streaming else ""
        content_html  = md_to_html(content)
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar ai">AI</div>
            <div class="bubble ai">{content_html}{cursor_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if sources and not is_streaming:
            report_chunks = [c for c in sources if c.source_type == "report"]
            kb_chunks     = [c for c in sources if c.source_type != "report"]

            # RAGAS-style inline metrics
            sims = [c.similarity for c in sources if hasattr(c, "similarity")]
            avg_sim  = sum(sims) / len(sims) if sims else 0
            top_sim  = max(sims) if sims else 0
            n_chunks = len(sources)
            n_report = len(report_chunks)
            # Quality label: based on whether we retrieved the expected mix.
            # Cross-encoder logits (normalized via sigmoid) are not comparable
            # to cosine similarity, so we judge by chunk coverage instead.
            if n_chunks >= 7 and n_report >= 3:
                grounding = "High"
            elif n_chunks >= 4 and n_report >= 1:
                grounding = "Medium"
            else:
                grounding = "Low"
            g_color = "#4caf50" if grounding == "High" else ("#ff9800" if grounding == "Medium" else "#f44336")

            # Sources expander first; retrieval-quality strip below it.
            label = f"Sources \u2014 {len(report_chunks)} from report \u00b7 {len(kb_chunks)} from knowledge base"
            with st.expander(label, expanded=False):
                for chunk in sources:
                    is_rep   = chunk.source_type == "report"
                    badge_cl = "badge-report" if is_rep else "badge-kb"
                    badge_tx = "REPORT" if is_rep else "KB"
                    src_meta = chunk.metadata or {}
                    sim      = f"{chunk.similarity:.3f}" if hasattr(chunk, "similarity") else "—"
                    section  = src_meta.get("section", "") or "—"
                    filename = src_meta.get("filename", "unknown")
                    st.markdown(f"""
                    <div class="src-card">
                        <div class="src-card-header">
                            <span class="badge {badge_cl}">{badge_tx}</span>
                            <span class="src-meta">{filename} · {section} · sim {sim}</span>
                        </div>
                        {chunk.text[:320]}{"…" if len(chunk.text) > 320 else ""}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(
                f'<div class="retrieval-meta">'
                f'Retrieval quality: <span style="color:{g_color};font-weight:600;">{grounding}</span>'
                f' &middot; {n_chunks} chunks'
                f' &middot; avg sim {avg_sim:.3f}'
                f' &middot; top sim {top_sim:.3f}'
                f' &middot; {len(report_chunks)} report + {len(kb_chunks)} KB'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    init_state()
    render_sidebar()

    # Capture the chat input EARLY so we know whether a question is in flight
    # before deciding to render the welcome screen. The widget is only
    # mounted when a report is loaded — no disabled-but-visible input box.
    question = None
    if st.session_state["report_loaded"]:
        question = st.chat_input("Ask a clinical question about this patient…")

    # ── Welcome screen ─────────────────────────────────────────────────────
    # Hidden the moment messages exist OR a question has just been submitted,
    # so the instructions don't flash above the streaming answer on the first turn.
    if not st.session_state["messages"] and not question:
        st.markdown("""
        <div class="welcome">
            <div class="welcome-icon">🧠</div>
            <h1>ASD <em>Clinical</em> Assistant</h1>
            <p>
                Upload a patient report, then ask clinical questions.<br>
                Answers are grounded in the uploaded report and your knowledge base.
            </p>
            <div class="welcome-steps">
                <div class="welcome-step">
                    <span class="num">01</span>
                    Upload a patient PDF in the sidebar
                </div>
                <div class="welcome-step">
                    <span class="num">02</span>
                    Click "Load report" to process and embed it
                </div>
                <div class="welcome-step">
                    <span class="num">03</span>
                    Ask clinical questions about the patient
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Handle new question ────────────────────────────────────────────────
    if question:
        pipeline = get_pipeline()

        # Append user turn and render the full conversation. No outer
        # chat-wrap div — the main block container CSS does the centering,
        # so the static loop and the streaming placeholder share the exact
        # same horizontal alignment.
        st.session_state["messages"].append({"role": "user", "content": question})
        for msg in st.session_state["messages"]:
            render_message(msg)

        # ── Stream the AI response ─────────────────────────────────────────
        # Render the typing indicator FIRST so it appears the moment the
        # user submits — before retrieval (which is synchronous and can take
        # a couple of seconds with cross-encoder + compression enabled).
        answer_text = ""
        placeholder = st.empty()
        placeholder.markdown("""
        <div class="typing-row">
            <div class="avatar ai">AI</div>
            <div class="typing-bubble">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        stream, ctx = pipeline.ask_stream(question)

        for token in stream:
            answer_text += token
            placeholder.markdown(f"""
            <div class="msg-row">
                <div class="avatar ai">AI</div>
                <div class="bubble ai">{md_to_html(answer_text)}<span class="cursor"></span></div>
            </div>
            """, unsafe_allow_html=True)

        placeholder.empty()
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": answer_text,
            "sources": ctx.chunks,
        })
        st.rerun()

    # ── Render existing conversation (no new input this run) ───────────────
    elif st.session_state["messages"]:
        for msg in st.session_state["messages"]:
            render_message(msg)


if __name__ == "__main__":
    main()