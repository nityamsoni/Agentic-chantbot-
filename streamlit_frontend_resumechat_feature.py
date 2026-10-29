import hashlib
import hmac
import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from time import perf_counter
from typing import Any, Iterator

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

st.set_page_config(
    page_title="Resume Chat", page_icon=":material/description:",
    layout="wide", initial_sidebar_state="expanded",
)

AUTH_DB_PATH = "auth.db"
HISTORY_REFRESH_SECONDS = 30
SUGGESTED_PROMPTS = [
    "Review my resume for a GenAI Engineer role",
    "Summarize the skills section",
    "What's missing from this resume?",
    "Rewrite my project bullets to be more impactful",
]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_backend() -> Any:
    """Import the shared graph once, after authentication."""
    import langchain_backend
    return langchain_backend


def init_auth_db() -> None:
    """Create the existing local account table if needed."""
    with closing(sqlite3.connect(AUTH_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL, password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"pbkdf2_sha256$100000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify salted hashes and supported legacy SHA-256 hashes."""
    try:
        if "$" not in stored_hash:
            return hmac.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored_hash)
        algo, iterations, salt, digest = stored_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


def register_user(name: str, email: str, phone: str, password: str) -> tuple[bool, str]:
    name, email, phone = name.strip(), normalize_email(email), phone.strip()
    if not name:
        return False, "Name is required."
    if "@" not in email or "." not in email:
        return False, "Enter a valid email."
    if len(phone) < 7:
        return False, "Enter a valid phone number."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with closing(sqlite3.connect(AUTH_DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO users (name, email, phone, password_hash) VALUES (?, ?, ?, ?)",
                (name, email, phone, hash_password(password)),
            )
            conn.commit()
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "Email already registered."


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    with closing(sqlite3.connect(AUTH_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT id, name, email, phone, password_hash FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()
    if row is None or not verify_password(password, row[4]):
        return None
    return dict(zip(("id", "name", "email", "phone"), row[:4]))


def render_auth() -> None:
    """Render account forms without importing embedding or model resources."""
    with st.container(horizontal_alignment="center"):
        with st.container(width=480):
            st.title("Resume Chat")
            st.caption("Your resume. Your next opportunity.")
            mode = st.segmented_control("Account", ["Login", "Register"], default="Login")
            if mode == "Register":
                with st.form("register_form"):
                    name = st.text_input("Name")
                    email = st.text_input("Email")
                    phone = st.text_input("Phone number")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Create account", use_container_width=True)
                if submitted:
                    ok, message = register_user(name, email, phone, password)
                    (st.success if ok else st.error)(message)
            else:
                with st.form("login_form"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Sign in", use_container_width=True)
                if submitted:
                    user = authenticate_user(email, password)
                    if user:
                        st.session_state.auth_user = user
                        st.rerun()
                    st.error("Invalid email or password.")


def user_thread_prefix(user_id: int) -> str:
    return f"u{user_id}:"


def generate_thread_id(user_id: int) -> str:
    return f"{user_thread_prefix(user_id)}{uuid.uuid4()}"


def is_user_thread(thread_id: str, user_id: int) -> bool:
    return str(thread_id).startswith(user_thread_prefix(user_id))


def build_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id}, "run_name": "chat_turn",
    }


def load_converstaion(thread_id: str) -> list[BaseMessage]:
    return backend.chatbot.get_state(build_config(thread_id)).values.get("messages", [])


def extract_sources(message: ToolMessage) -> list[dict[str, Any]]:
    """Read source previews from the existing RAG tool output."""
    if message.name != "rag_tool" or not isinstance(message.content, str):
        return []
    try:
        payload = json.loads(message.content)
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    sources = []
    metadata = payload.get("metadata", [])
    for index, text in enumerate(payload.get("context", [])):
        meta = metadata[index] if index < len(metadata) else {}
        page = meta.get("page_label")
        if page is None and isinstance(meta.get("page"), int):
            page = meta["page"] + 1
        sources.append({"filename": payload.get("source_file") or "Uploaded PDF",
                        "page": page, "snippet": str(text)[:700]})
    return sources


def format_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Restore visible answers and their retrieved source previews."""
    formatted = []
    sources: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            sources = []
            formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage):
            sources.extend(extract_sources(message))
        elif isinstance(message, AIMessage) and message.content:
            formatted.append({"role": "assistant", "content": message.content,
                              "sources": list(sources)})
    return formatted


def get_thread_title(thread_id: str) -> str:
    titles = st.session_state.setdefault("thread_titles", {})
    if thread_id in titles:
        return titles[thread_id]
    for message in load_converstaion(thread_id):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            content = message.content.strip()
            if content:
                titles[thread_id] = content[:40]
                return titles[thread_id]
    return "New conversation"


def reset_chat(user_id: int) -> None:
    """Start an empty graph thread without deleting saved conversations."""
    thread_id = generate_thread_id(user_id)
    st.session_state.thread_id = thread_id
    st.session_state.chat_thread.append(thread_id)
    st.session_state.message_history = []


def select_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.message_history = format_messages(load_converstaion(thread_id))


def sync_threads(user_id: int, force: bool = False) -> None:
    """Throttle checkpoint enumeration within this user's session."""
    now = perf_counter()
    if not force and now - st.session_state.get("threads_refreshed", -float("inf")) < HISTORY_REFRESH_SECONDS:
        return
    for thread_id in sorted(backend.retrive_threads()):
        if is_user_thread(thread_id, user_id) and thread_id not in st.session_state.chat_thread:
            st.session_state.chat_thread.append(thread_id)
    st.session_state.threads_refreshed = now


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Retrieved sources ({len(sources)})"):
        for source in sources:
            page = f" | Page {source['page']}" if source.get("page") is not None else ""
            st.text(f"{source['filename']}{page}")
            st.caption(source["snippet"])


def render_sidebar(user: dict[str, Any]) -> None:
    with st.sidebar:
        st.header("Resume Chat")
        st.caption(user["name"])
        if st.button("New conversation", icon=":material/add:", use_container_width=True, type="primary"):
            reset_chat(user["id"])
            st.rerun()
        if st.button("Clear current chat", icon=":material/refresh:", use_container_width=True,
                     help="Starts a fresh chat. The previous conversation stays in History."):
            reset_chat(user["id"])
            st.rerun()
        with st.expander("Settings"):
            st.text_input("Model", "openai/gpt-oss-120b", disabled=True)
            st.number_input("Temperature", value=0.7, disabled=True)
            st.number_input("Retrieved chunks", value=4, disabled=True)
            st.toggle("Timing details", key="show_timings")
        with st.expander("About"):
            st.caption("Groq | LangGraph | MiniLM | FAISS")
            st.caption("Chat history is stored locally. PDF indexes stay in server memory and must be uploaded again after a restart.")
        st.divider()
        heading, refresh = st.columns([4, 1])
        heading.subheader("History")
        if refresh.button("", icon=":material/sync:", help="Refresh history"):
            sync_threads(user["id"], force=True)
        for thread_id in reversed(st.session_state.chat_thread):
            if st.button(get_thread_title(thread_id), key=f"thread_{thread_id}",
                         use_container_width=True,
                         type="primary" if thread_id == st.session_state.thread_id else "secondary"):
                select_thread(thread_id)
                st.rerun()
        st.divider()
        if st.button("Sign out", icon=":material/logout:", use_container_width=True):
            for key in list(st.session_state):
                del st.session_state[key]
            st.rerun()


def ingest_uploaded_file(uploaded_file: Any, thread_id: str) -> None:
    """Index changed content once per thread, not merely once per filename."""
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    existing = st.session_state.ingested_docs.get(thread_id)
    if existing and existing["digest"] == digest:
        return
    started = perf_counter()
    with st.status("Indexing PDF...", expanded=False) as status:
        try:
            summary = backend.ingest_pdf(data, thread_id, uploaded_file.name)
        except Exception:
            status.update(label="PDF could not be indexed", state="error")
            raise
        st.session_state.ingested_docs[thread_id] = {"digest": digest, **summary}
        status.update(label="PDF ready", state="complete")
    logger.info("ingestion_seconds=%.3f", perf_counter() - started)


def stream_answer(user_input: str, thread_id: str) -> dict[str, Any]:
    """Stream the unchanged graph and record client-observed latency."""
    started = perf_counter()
    sources: list[dict[str, Any]] = []
    first_token: float | None = None
    answer_parts: list[str] = []
    status = st.status("Thinking...", expanded=False)

    def tokens() -> Iterator[str]:
        nonlocal first_token
        for mode, event in backend.chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=build_config(thread_id), stream_mode=["messages", "updates"],
        ):
            if mode == "updates":
                for update in event.values():
                    if not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage) and message.tool_calls:
                            names = ", ".join(call["name"] for call in message.tool_calls)
                            status.update(label=f"Using {names}...")
                        elif isinstance(message, ToolMessage):
                            retrieved = extract_sources(message)
                            sources.extend(retrieved)
                            if message.name == "rag_tool" and not retrieved:
                                st.info("No PDF sources were returned for this question.")
                            status.update(label="Preparing answer...")
            else:
                message, metadata = event
                if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
                    if first_token is None:
                        first_token = perf_counter() - started
                        logger.info("time_to_first_token_seconds=%.3f", first_token)
                    answer_parts.append(message.content)
                    yield message.content

    error = False
    try:
        st.write_stream(tokens())
        if answer_parts:
            status.update(label="Complete", state="complete")
        else:
            error = True
            status.update(label="No answer received", state="error")
            st.warning("No answer text was returned. Please try a new conversation.")
    except Exception:
        error = True
        logger.exception("chat_turn_failed")
        status.update(label="Response interrupted", state="error")
    total = perf_counter() - started
    logger.info("total_response_seconds=%.3f success=%s", total, not error)
    return {"role": "assistant", "content": "".join(answer_parts), "sources": sources,
            "timing": {"first_token": first_token, "total": total}, "error": error}


def render_answer_details(message: dict[str, Any]) -> None:
    render_sources(message.get("sources", []))
    if message.get("error"):
        st.error("The response could not finish. Check your connection and API configuration. Start a new conversation if this chat cannot continue.")
    if message.get("content"):
        with st.expander("Copy answer"):
            st.code(message["content"], language=None, wrap_lines=True)
    if st.session_state.get("show_timings") and message.get("timing"):
        timing = message["timing"]
        first = timing["first_token"]
        st.caption(f"First text: {first:.2f}s | Total: {timing['total']:.2f}s" if first is not None
                   else f"Total: {timing['total']:.2f}s | No text received")


def render_composer(thread_id: str) -> Any:
    """Render one PDF-enabled composer in the caller's layout context."""
    return st.chat_input(
        "Ask a question or attach a PDF", accept_file=True, file_type=["pdf"],
        key=f"input_{thread_id}",
    )


def render_chat() -> None:
    """Center the empty composer and pin it below an active conversation."""
    thread_id = st.session_state.thread_id
    history = st.session_state.message_history
    had_history = bool(history)
    submission = None
    with st.container(horizontal_alignment="center"):
        with st.container(width=900):
            st.title("Resume Chat")
            doc = st.session_state.ingested_docs.get(thread_id)
            doc_status = st.empty()
            doc_status.caption(f"{doc['filename']} | {doc['documents']} pages" if doc else "No PDF attached")
            welcome = st.empty()
            if not had_history:
                with welcome.container():
                    with st.container(height=420, border=False, vertical_alignment="center"):
                        st.subheader("What would you like to work on?")
                        submission = render_composer(thread_id)
                        for index, column in enumerate(st.columns(2)):
                            with column:
                                for prompt in SUGGESTED_PROMPTS[index::2]:
                                    if st.button(prompt, use_container_width=True):
                                        st.session_state.pending_prompt = prompt
            for message in history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        render_answer_details(message)
            turn = st.container()
    if had_history:
        submission = render_composer(thread_id)
    pending = st.session_state.pop("pending_prompt", None)
    if submission is None and not pending:
        return
    text = (submission if isinstance(submission, str) else submission.text) if submission is not None else pending
    text = (text or "").strip()
    files = getattr(submission, "files", [])
    if files:
        try:
            with turn:
                ingest_uploaded_file(files[0], thread_id)
        except Exception:
            logger.exception("pdf_ingestion_failed")
            turn.error("Could not read this PDF. Use an unencrypted PDF with selectable text and try again.")
            return
        doc = st.session_state.ingested_docs[thread_id]
        doc_status.caption(f"{doc['filename']} | {doc['documents']} pages")
        text = text or "Please review the uploaded PDF and answer questions based on it."
    if not text:
        turn.info("Enter a message or attach a PDF.")
        return
    welcome.empty()
    history.append({"role": "user", "content": text})
    with turn:
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant"):
            answer = stream_answer(text, thread_id)
            render_answer_details(answer)
    history.append(answer)
    st.session_state.thread_titles.setdefault(thread_id, text[:40])
    if not had_history:
        st.rerun()


def main() -> None:
    """Authenticate, restore user state, then render the native chat interface."""
    global backend
    init_auth_db()
    if not st.session_state.get("auth_user"):
        render_auth()
        return
    try:
        with st.spinner("Connecting..."):
            backend = load_backend()
    except Exception:
        logger.exception("backend_initialization_failed")
        st.error("Could not start the assistant. Check GROQ_API_KEY and the installed dependencies.")
        return
    user = st.session_state.auth_user
    for key, default in {"chat_thread": [], "thread_titles": {}, "ingested_docs": {},
                         "message_history": [], "show_timings": False}.items():
        st.session_state.setdefault(key, default)
    try:
        sync_threads(user["id"])
        if not is_user_thread(st.session_state.get("thread_id", ""), user["id"]):
            if st.session_state.chat_thread:
                select_thread(st.session_state.chat_thread[-1])
            else:
                reset_chat(user["id"])
        render_sidebar(user)
        render_chat()
    except sqlite3.DatabaseError as error:
        logger.exception(
            "checkpoint_database_error sqlite_errorname=%s sqlite_errorcode=%s",
            getattr(error, "sqlite_errorname", "unknown"),
            getattr(error, "sqlite_errorcode", "unknown"),
        )
        st.error("Chat storage is unavailable. Saved conversations could not be loaded safely.")
        st.caption("The database has not been reset. The app owner should check Manage app > Logs for checkpoint_database_error before attempting a repair.")
        if st.button("Retry storage", icon=":material/refresh:"):
            st.session_state.pop("threads_refreshed", None)
            st.rerun()


if __name__ == "__main__":
    main()