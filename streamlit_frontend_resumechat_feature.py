import streamlit as st
from langchain_backend import chatbot, retrive_threads, ingest_pdf
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


st.set_page_config(
    page_title="Resume Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "📄"

SUGGESTED_PROMPTS = [
    "Review my resume for a GenAI Engineer role",
    "Summarize the skills section",
    "What's missing from this resume?",
    "Rewrite my project bullets to be more impactful",
]


def generate_thread_id():
    return str(uuid.uuid4())


def build_config(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }


def add_to_chat_thread(thread_id):
    if thread_id not in st.session_state.chat_thread:
        st.session_state.chat_thread.append(thread_id)


def load_converstaion(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def format_messages(messages):
    formatted_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_messages.append({"role": "assistant", "content": message.content})
    return formatted_messages


def get_thread_title(thread_id):
    for message in load_converstaion(thread_id):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            content = message.content.strip()
            if content:
                return content[:40]
    return "New conversation"


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_to_chat_thread(thread_id)
    st.session_state.message_history = []


def select_thread(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = format_messages(load_converstaion(thread_id))


def ingest_uploaded_file(uploaded_file, thread_id, thread_docs):
    if uploaded_file is None:
        return None

    if uploaded_file.name in thread_docs:
        return thread_docs[uploaded_file.name]

    with st.status(f"Reading {uploaded_file.name}", expanded=True) as status_box:
        summary = ingest_pdf(
            uploaded_file.getvalue(),
            thread_id=thread_id,
            filename=uploaded_file.name,
        )
        thread_docs[uploaded_file.name] = summary
        status_box.update(label=f"Indexed {uploaded_file.name}", state="complete", expanded=False)
        return summary


if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = retrive_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

add_to_chat_thread(st.session_state["thread_id"])

current_thread_id = str(st.session_state["thread_id"])
message_history = st.session_state.message_history
thread_docs = st.session_state["ingested_docs"].setdefault(current_thread_id, {})
config = build_config(current_thread_id)
latest_doc = list(thread_docs.values())[-1] if thread_docs else None

# ---------------------------------------------------------------------------
# Styling — quiet, neutral, production UI. One accent, no card-kit chrome.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #fafafa;
        --surface: #ffffff;
        --border: #e4e4e7;
        --text: #18181b;
        --text-muted: #71717a;
        --accent: #4f46e5;
        --accent-soft: #eef2ff;
        --sidebar-bg: #111113;
        --sidebar-border: #232326;
        --sidebar-active: #1c1c1f;
        --sidebar-text: #e4e4e7;
        --sidebar-text-muted: #8b8b93;
        --radius: 10px;
    }

    html, body, .stApp {
        background: var(--bg);
        color: var(--text);
        font-feature-settings: "tnum";
    }

    .block-container {
        max-width: 860px;
        padding-top: 2.25rem;
        padding-bottom: 7rem;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.01em;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.9rem;
        margin-bottom: 1.1rem;
    }
    .app-header h1 {
        font-size: 1.35rem;
        font-weight: 650;
        margin: 0;
    }
    .app-header .meta {
        font-size: 0.82rem;
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }

    /* Document status line */
    .doc-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: var(--text-muted);
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: var(--radius);
        padding: 0.55rem 0.85rem;
        margin-bottom: 1.25rem;
    }
    .doc-status .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .doc-status.active .dot { background: var(--accent); }
    .doc-status.idle .dot { background: #d4d4d8; }
    .doc-status strong { color: var(--text); font-weight: 600; }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem 2rem;
        color: var(--text-muted);
    }
    .empty-state .icon {
        font-size: 2rem;
        margin-bottom: 0.6rem;
    }
    .empty-state h3 {
        font-size: 1.05rem;
        color: var(--text);
        margin-bottom: 0.25rem;
    }
    .empty-state p {
        font-size: 0.88rem;
        margin: 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--sidebar-border);
    }
    [data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
        color: var(--sidebar-text-muted) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--sidebar-text-muted);
        margin: 0.4rem 0 0.5rem 0;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--sidebar-border);
        margin: 0.9rem 0;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-weight: 500;
        font-size: 0.88rem;
        min-height: 2.35rem;
        box-shadow: none;
        transition: border-color 120ms ease, background 120ms ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--accent);
    }

    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid transparent;
        color: var(--sidebar-text);
        text-align: left;
        justify-content: flex-start;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--sidebar-border);
        background: var(--sidebar-active);
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--sidebar-active);
        border-color: #34343a;
        color: #ffffff;
        font-weight: 600;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.15rem 0.35rem;
        margin-bottom: 0.6rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stText"] {
        font-size: 0.94rem;
        line-height: 1.55;
        color: var(--text);
    }

    /* Chat input */
    [data-testid="stChatInputTextArea"] textarea {
        background: var(--surface);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        font-size: 0.92rem;
    }
    [data-testid="stChatInputTextArea"] textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }
    [data-testid="stChatInputFileUploader"] {
        background: var(--surface);
        border-radius: 8px;
        border: 1px dashed var(--border);
    }

    div[data-testid="stBottomBlockContainer"] {
        max-width: 860px;
        margin: 0 auto;
        padding-left: 1rem;
        padding-right: 1rem;
        background: transparent;
        border-top: 1px solid var(--border);
    }

    /* Status / expander boxes */
    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--surface);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 6rem;
        }
        .app-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>Resume Chat Assistant</h1>
        <span class="meta">{len(message_history)} messages · thread {current_thread_id[:8]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if latest_doc:
    st.markdown(
        f"""
        <div class="doc-status active">
            <span class="dot"></span>
            Reading <strong>{latest_doc.get('filename')}</strong>
            — {latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="doc-status idle">
            <span class="dot"></span>
            No document attached — attach a PDF from the message box to ask about it
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Resume Chat")

if st.sidebar.button("＋ New conversation", use_container_width=True):
    reset_chat()
    st.rerun()

if st.sidebar.button("Clear current chat", use_container_width=True):
    st.session_state.message_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### History")

for thread_id in st.session_state["chat_thread"][::-1]:
    title = get_thread_title(thread_id)
    is_active = thread_id == current_thread_id
    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        if not is_active:
            select_thread(thread_id)
            st.rerun()

# ---------------------------------------------------------------------------
# Chat history / empty state
# ---------------------------------------------------------------------------
if not message_history:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📄</div>
            <h3>Start by attaching your resume</h3>
            <p>Upload a PDF from the message box below, or just ask a question to get going.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    prompt_cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with prompt_cols[i % 2]:
            if st.button(prompt, key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()
else:
    for message in message_history:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.text(message["content"])

chat_submission = st.chat_input(
    "Ask about your resume, skills, projects, or request improvements...",
    accept_file=True,
    file_type=["pdf"],
    key=f"chat_input_{current_thread_id}",
)

# A clicked suggested prompt behaves exactly like typed input.
if chat_submission is None and st.session_state.pending_prompt:
    chat_submission = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if chat_submission:
    if isinstance(chat_submission, str):
        user_input = chat_submission
        uploaded_files = []
    else:
        user_input = chat_submission.text.strip()
        uploaded_files = chat_submission.files

    if uploaded_files:
        summary = ingest_uploaded_file(uploaded_files[0], current_thread_id, thread_docs)
        latest_doc = summary or latest_doc
        if not user_input:
            user_input = "Please review the uploaded PDF and answer questions based on it."

    if not user_input:
        st.info("Add a message or attach a PDF to continue.")
        st.stop()

    message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.text(user_input)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using `{tool_name}`", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"Using `{tool_name}`",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and isinstance(message_chunk.content, str):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="Tool finished", state="complete", expanded=False
            )

    message_history.append({"role": "assistant", "content": ai_message})