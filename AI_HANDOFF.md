# Resume Chat: AI Handoff

Updated: 2026-10-29.

This document summarizes the preceding coding conversation and current local
implementation. Give it to another AI along with the relevant source files.
It is a context handoff, not an automatically loaded agent skill. Source code
and newer user instructions take precedence if they differ from this snapshot.
No credentials or private resume contents are included here.

## Start Here

The user wanted a cleaner, wider, smoother Streamlit resume chatbot without
changing answer quality. That UI/performance work is implemented locally.
The current deployment blocker is a confirmed corrupted SQLite checkpoint
database on Streamlit Cloud, not a model or retrieval error.

The backend now supports an explicit alternative database path. This provides
a fresh-start option; it does not repair or migrate the corrupt database.
All eight local regression tests passed after the composer layout follow-up.
Cloud repair, deployment of the latest local changes, and acceptance of empty
history have NOT been verified.

## User Requirements And Decisions

- Diagnose slowness before changing code; do not claim unmeasured speed gains.
- Preserve retrieval, prompts, model choice, and answer quality unless the user
  approves a change. Do not silently shorten LLM history.
- The user selected a clean light theme with a teal accent and selected
  "Preserve answer behavior" when asked about quality-affecting changes.
- Use native Streamlit components and a TOML theme, not large CSS injections.
- Keep authentication, per-user history, PDF uploads, and streaming working.
- Provide useful loading/error states, source previews, copy-friendly answers,
  example questions, and latency measurements.
- Do not add paid services or heavy dependencies. Only the existing Streamlit
  minimum was raised for the native layout APIs used.
- Never hardcode API keys or ask the user to paste secrets into chat.
- Never delete, reset, replace, or migrate existing databases automatically.
- No commits, pushes, or new Git branches were requested or performed by the
  assistant. Do not assume local changes are deployed.

## Project And Environment

- Workspace: `d:\LangGarph-tutorials` on Windows.
- Local environment: `myenv`, with Python 3.13 observed during earlier checks.
- Local Streamlit version observed: 1.63.0.
- Deployment: Streamlit Community Cloud, `https://chat69.streamlit.app/`.
- Cloud repository reported in logs: `nityamsoni/agentic-chantbot-`, branch `main`.
- Cloud working directory: `/mount/src/agentic-chantbot-/`.
- Cloud entry point: `streamlit_frontend_resumechat_feature.py`.
- Provided cloud logs report Python 3.14.10 and Streamlit 1.66.1. These version
  differences are not evidence of the cause of SQLite corruption.
- A local server was started at `http://127.0.0.1:8502` using the VS Code task
  "Resume Chat". Check whether it is still running; do not assume it is.
- Other tutorial notebooks and `streamlit_frontend.py` exist but were not the
  target frontend. Do not edit them without a task-specific reason.

## Relevant Files

- [langchain_backend.py](langchain_backend.py): tools, PDF ingestion, embeddings,
  graph construction, checkpoint configuration, and thread enumeration.
- [streamlit_frontend_resumechat_feature.py](streamlit_frontend_resumechat_feature.py):
  active app, account forms, session state, sidebar, sources, and streaming UI.
- [.streamlit/config.toml](.streamlit/config.toml): native light/teal theme.
- [requirements.txt](requirements.txt): Streamlit minimum is now 1.63.0.
- [test_frontend.py](test_frontend.py): eight isolated regression tests.
- [README.md](README.md): setup, diagnosis, measurement, and cloud recovery notes.
- Local database and WAL files exist. Their presence does not establish their
  integrity or whether they match the deployed database. Do not publish them.

## Backend Behavior To Preserve

- Groq model: `openai/gpt-oss-120b`; temperature: `0.7`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- PDF parsing: `PyPDFLoader`.
- Splitter: chunk size 1000 characters, overlap 200, existing separators retained.
- Vector store: FAISS; similarity search with `k=4`.
- Graph: START -> chat_node -> conditional tools/end; tools -> chat_node.
- Message reducer: LangGraph `add_messages`; full checkpointed history remains.
- Tools: `web_search`, `get_stock_price`, `calculator`, `rag_tool`.
- `rag_tool` uses the thread ID from `RunnableConfig` and returns query, context,
  document metadata, and source filename. Tool choice is still model-driven.
- `_THREAD_RETRIEVERS` and `_THREAD_METADATA` are process-local. Only the most
  recently indexed PDF is active per thread. Indexes do not survive restart.
- Public helper spelling `retrive_threads` is retained to preserve callers.
- Authentication uses a separate local SQLite database. Thread IDs are
  `u<user_id>:<uuid>`, and the frontend filters history by this prefix.

## Implemented Changes

1. Replaced the active frontend's large injected CSS with native components:
   centered chat, sidebar, account selector, examples, Markdown, expanders,
   status indicators, and file-enabled `st.chat_input`.
2. Added `load_backend()` with `st.cache_resource`, called after authentication.
   Python backend imports were already module-cached on normal reruns; do not
   incorrectly claim the original model reloaded on every Streamlit interaction.
3. Added lazy `get_embeddings()` with `st.cache_resource`. MiniLM construction
   happens on the first PDF upload, not before the login screen.
4. Cached nonempty sidebar titles in session state. Empty titles can still
   refresh after a first question. Checkpoint enumeration is throttled to
   30 seconds per session, with manual refresh available.
5. Kept `st.write_stream`; consume both graph `messages` and `updates` for
   streamed text, earlier tool status, and RAG source previews.
6. Sources include filename, human-readable PDF page, and a display-only snippet.
   These are retrieved evidence, NOT verified per-claim citations. Preview
   truncation does not truncate the context given to the model.
7. Added copy expanders, error and empty-response states, retained partial
   responses on failure, and hid welcome prompts after a valid submission.
8. PDF ingestion uses a content digest per thread, not filename alone, to avoid
   redundant indexing while recognizing changed uploads. Empty/extraction-free
   PDFs receive a clear error.
9. Clear chat creates a new graph thread. It does not merely erase the visible
   transcript and does not delete old checkpoints. Old history remains available.
10. Model, temperature, and top-k settings are read-only to preserve quality.
    Timing details are user-toggleable. No history trimming was introduced.
11. Groq requests use a 60-second timeout and up to two retries. This is not a
    whole-turn deadline: tool loops and retries can take longer. Stock requests
    retain the existing 10-second timeout.
12. Removed the embedded stock API credential in favor of `ALPHAVANTAGE_API_KEY`.
    The previously exposed key should be rotated. Groq uses `GROQ_API_KEY`.
13. Account database access uses `contextlib.closing` and explicit commits.
    SQLite connection context managers do not close connections themselves.
14. Checkpoint read failures are caught at the frontend history/render boundary,
    logged as `checkpoint_database_error` with SQLite codes when available, and
    shown as a storage error with "Retry storage". No empty-history or memory
    fallback is used to conceal a damaged SQLite file.
15. Backend `create_checkpointer()` reads `CHATBOT_DB_PATH`, defaulting to
    `chatbot.db`. SQLite connection still uses `check_same_thread=False`.
    The pre-existing MemorySaver fallback remains only when the SQLite saver
    import is unavailable, not when SQLite data is corrupt.

## Composer Layout Follow-Up

The user subsequently reported that the input stayed above the conversation.
They requested a centered input on an empty chat and bottom-pinned input after
an answer. The active composer had been inside nested containers, where
Streamlit renders it inline rather than pinning it.

`render_chat()` is now called at page level. It creates its own centered
900px content area; `render_composer()` is nested only for empty chats. For
active chats, the composer is called outside all layout containers so Streamlit
uses its native bottom region. New streamed messages target a pre-created
container above the input. After the first answer, one UI rerun changes layouts
without another graph call. Clear/new chat returns to the empty centered layout.
Errors are rendered from saved answer state so that rerun does not hide them.

A regression checks exactly one input, main-to-bottom layout paths, no duplicate
first graph turn, a second message, and reset after clearing. A temporary fake
backend browser preview verified 1440x900 desktop and 390x844 mobile layouts,
including centered empty inputs, pinned active inputs, and no horizontal
overflow. The preview file and task were removed after use. This does not
verify the deployed app or actual Groq latency; deploy the updated frontend.

## Confirmed Cloud Incident

The user initially supplied a redacted traceback and then complete cloud logs.
The decisive exception was repeated:

```text
DatabaseError: database disk image is malformed
```

The failing path was:

```text
main()
  -> sync_threads(user["id"])
  -> backend.retrive_threads()
  -> checkpointer.list(None)
  -> SqliteSaver.list(): cur.execute(query, param_values)
```

This confirms corruption of the checkpoint database being queried. It does
not establish what caused the corruption. It is not evidence that Groq,
MiniLM, Python 3.14, chunk size, or the new theme caused it.

The supplied trace showed the older unguarded frontend, rather than the latest
local error-handling patch. Check the deployed revision before assuming the
latest code is live. No cloud filesystem access or repair was performed.

The cloud logs also show large CUDA/PyTorch downloads and a platform workaround
for PyArrow 26. Those are separate deployment concerns, not a demonstrated
explanation of the SQLite corruption. Do not expand into dependency/model
changes without approval and focused evidence.

## Recovery Decision Still Required

Ask whether old conversations must be recovered or whether the user accepts
starting with empty history. Do not assume permission to lose context.

If old conversations matter:

- Obtain a consistent backup of the DEPLOYED checkpoint database before restart,
  redeployment, or repair. Stop writers and account for associated WAL files.
- A local database with the same name is not a backup of the deployed file.
- Diagnose/recover on a copy. Do not run destructive fixes on the only original.
- No recovery tool or migration has been implemented or verified in this session.

If the user explicitly accepts empty history:

1. Deploy the updated code containing `create_checkpointer()`.
2. Add this root-level value in Streamlit Cloud app settings > Secrets,
   outside any TOML table, choosing a filename that does not already exist:

   ```toml
   CHATBOT_DB_PATH = "chatbot_fresh.db"
   ```

3. Reboot the app so the cached backend opens the configured file.
4. Verify login, new-chat streaming, and reopening newly saved history.

Streamlit exposes root-level secrets as environment variables. The path change
does not overwrite the old file, but does NOT migrate or recover its chats.
The auth database path is unchanged. Cloud local storage is ephemeral, so a
reboot/redeploy is not guaranteed to preserve either database or the only backup.
No automatic path switch was applied locally or remotely.

## Validation Already Performed

- All eight tests passed after the composer follow-up. No external API calls
  or embedding downloads were needed for those tests.
- The suite covers registration/login; streaming and restored sources; title
  caching; clear-chat isolation; stream errors; per-user filtering/logout;
  malformed source payloads; SQLite errors and retry; and explicit selection
  of a fresh SQLite file while preserving a damaged fixture byte-for-byte.
- Most frontend tests use AppTest with a fake graph and temporary auth files.
  The configurable database test uses the real installed SQLite saver.
- Editor diagnostics reported no errors in the changed Python files.
- Live login-page checks at 1440px and 390px found no horizontal overflow.
  Authenticated chat was checked via AppTest, not a real-account browser session.
- A focused backend probe confirmed unchanged embedding model selection and
  RAG result shape/missing-document behavior without loading real embeddings.
- An extra Groq client-configuration probe timed out locally. Do not report
  it as a pass. Real Groq streaming and real PDF ingestion remain unverified.
- No real before/after model latency benchmark or multi-user load test was run.
- No deployed database repair, migration, or successful cloud reboot was verified.

## Commands

Run from the project root using the existing Windows environment:

```powershell
.\myenv\Scripts\python.exe -m unittest test_frontend -v
.\myenv\Scripts\python.exe -m streamlit run streamlit_frontend_resumechat_feature.py --server.port=8502
```

If dependencies need installation:

```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

For performance measurement, enable Settings > Timing details. Server logs use
`ingestion_seconds`, `retrieval_seconds`, `time_to_first_token_seconds`, and
`total_response_seconds`. Compare the same PDF, questions, history, and machine;
separate cold startup/first upload from warm runs. Do not present fake-graph
timings as Groq performance numbers.

## Testing And Tooling Pitfalls

- AppTest uses a 30-second harness timeout because first-use imports can exceed
  the default three seconds. This does not change application API timeouts.
- Do not restore the entire `sys.modules` dictionary around AppTests. Removing
  newly imported NumPy modules caused native-extension reload errors. The test
  harness now restores only the mocked `langchain_backend` entry.
- A PowerShell terminal encountered an interactive publisher-trust prompt for
  VS Code shell integration. Do not silently change execution/trust policy.
  Python checks were subsequently run with the selected-interpreter snippet tool.
- Preserve unrelated user edits. Inspect current files before applying patches.
- Do not read, print, commit, or include local secret values in another handoff.

## Suggested Next Message To The User

"The cloud logs confirm a corrupted checkpoint database. Do you need to
recover the old chats, or do you approve starting with a fresh chat-history
file? I can guide that path without changing the model or retrieval settings."