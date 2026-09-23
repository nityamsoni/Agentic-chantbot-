import streamlit as st
from langchain_backend import chatbot, retrive_threads, ingest_pdf
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import sqlite3
import hashlib
import hmac
import os


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

AUTH_DB_PATH = "auth.db"


def init_auth_db():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"pbkdf2_sha256$100000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, iterations_str, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            # Backward compatibility for old unsalted SHA-256 hashes.
            return hmac.compare_digest(hashlib.sha256(password.encode("utf-8")).hexdigest(), stored_hash)

        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def register_user(name: str, email: str, phone: str, password: str) -> tuple[bool, str]:
    name = name.strip()
    email = normalize_email(email)
    phone = phone.strip()

    if not name:
        return False, "Name is required."
    if "@" not in email or "." not in email:
        return False, "Enter a valid email."
    if len(phone) < 7:
        return False, "Enter a valid phone number."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    try:
        conn = sqlite3.connect(AUTH_DB_PATH)
        conn.execute(
            "INSERT INTO users (name, email, phone, password_hash) VALUES (?, ?, ?, ?)",
            (name, email, phone, hash_password(password)),
        )
        conn.commit()
        conn.close()
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "Email already registered."


def authenticate_user(email: str, password: str):
    email = normalize_email(email)
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.execute(
        "SELECT id, name, email, phone, password_hash FROM users WHERE email = ?",
        (email,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    user_id, name, user_email, phone, password_hash = row
    if not verify_password(password, password_hash):
        return None

    return {
        "id": user_id,
        "name": name,
        "email": user_email,
        "phone": phone,
    }


def user_thread_prefix(user_id: int) -> str:
    return f"u{user_id}:"


def generate_thread_id(user_id: int):
    return f"{user_thread_prefix(user_id)}{uuid.uuid4()}"


def is_user_thread(thread_id: str, user_id: int) -> bool:
    return str(thread_id).startswith(user_thread_prefix(user_id))


def get_user_threads(user_id: int):
    all_threads = retrive_threads()
    return [thread_id for thread_id in all_threads if is_user_thread(thread_id, user_id)]


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


def reset_chat(user_id: int):
    thread_id = generate_thread_id(user_id)
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

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

init_auth_db()

if st.session_state.auth_user is None:
    st.markdown(
        """
        <style>
        .auth-title {
            font-size: clamp(1.8rem, 2.8vw, 2.4rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            color: #141417;
            text-align: center;
        }
        .auth-subtitle {
            text-align: center;
            color: #6b7280;
            margin: 0.4rem 0 1.5rem 0;
            font-size: 0.98rem;
        }
        .auth-surface {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 8px 30px rgba(16, 24, 40, 0.06);
            margin-bottom: 1rem;
        }
        .auth-note {
            text-align: center;
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 0.55rem;
        }
        [data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem 1rem 0.25rem 1rem;
            box-shadow: 0 8px 28px rgba(16, 24, 40, 0.05);
        }
        [data-testid="stTextInput"] input {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    outer_l, center, outer_r = st.columns([1, 1.2, 1])
    with center:
        st.markdown('<h1 class="auth-title">Resume Chat Assistant</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="auth-subtitle">Sign in to access your private conversations and uploaded resume history.</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="auth-surface">', unsafe_allow_html=True)
        page_cols = st.columns(2)
        with page_cols[0]:
            if st.button(
                "Login",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "login" else "secondary",
            ):
                st.session_state.auth_mode = "login"
                st.rerun()
        with page_cols[1]:
            if st.button(
                "Register",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "register" else "secondary",
            ):
                st.session_state.auth_mode = "register"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.subheader("Login")
            with st.form("login_form", clear_on_submit=False):
                login_email = st.text_input("Email", key="login_email", placeholder="you@example.com")
                login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
                login_submit = st.form_submit_button("Login", use_container_width=True)

            if login_submit:
                user = authenticate_user(login_email, login_password)
                if user is None:
                    st.error("Invalid email or password.")
                else:
                    st.session_state.auth_user = user
                    st.session_state.chat_thread = get_user_threads(user["id"])
                    if st.session_state.chat_thread:
                        st.session_state.thread_id = st.session_state.chat_thread[-1]
                    else:
                        st.session_state.thread_id = generate_thread_id(user["id"])
                        st.session_state.chat_thread = [st.session_state.thread_id]
                    st.session_state.message_history = []
                    st.rerun()

            st.markdown('<p class="auth-note">Don\'t have an account? Switch to Register.</p>', unsafe_allow_html=True)
        else:
            st.subheader("Register")
            with st.form("register_form", clear_on_submit=True):
                reg_name = st.text_input("Name", placeholder="Your full name")
                reg_email = st.text_input("Email", placeholder="you@example.com")
                reg_phone = st.text_input("Phone number", placeholder="e.g. +91 9876543210")
                reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
                reg_submit = st.form_submit_button("Create account", use_container_width=True)

            if reg_submit:
                ok, msg = register_user(reg_name, reg_email, reg_phone, reg_password)
                if ok:
                    st.success(msg)
                    st.session_state.auth_mode = "login"
                else:
                    st.error(msg)

            st.markdown('<p class="auth-note">Already registered? Switch to Login.</p>', unsafe_allow_html=True)

    st.stop()

current_user = st.session_state.auth_user
current_user_id = current_user["id"]

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = get_user_threads(current_user_id)

if "thread_id" not in st.session_state or not is_user_thread(st.session_state.thread_id, current_user_id):
    if st.session_state.chat_thread:
        st.session_state.thread_id = st.session_state.chat_thread[-1]
    else:
        st.session_state.thread_id = generate_thread_id(current_user_id)

# Sync sidebar list with persisted checkpoints for this user on each run.
persisted_threads = get_user_threads(current_user_id)
for persisted_thread in persisted_threads:
    if persisted_thread not in st.session_state.chat_thread:
        st.session_state.chat_thread.append(persisted_thread)

st.session_state.chat_thread = [
    thread_id for thread_id in st.session_state.chat_thread if is_user_thread(thread_id, current_user_id)
]

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
st.sidebar.caption(f"Signed in as {current_user['name']}")

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.auth_user = None
    st.session_state.message_history = []
    st.session_state.pending_prompt = None
    st.session_state.pop("thread_id", None)
    st.session_state.pop("chat_thread", None)
    st.rerun()

if st.sidebar.button("＋ New conversation", use_container_width=True):
    reset_chat(current_user_id)
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