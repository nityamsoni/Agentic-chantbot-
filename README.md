# Agentic RAG Chatbot with Long-Term Memory

## Run

Use the existing environment with Streamlit 1.63 or newer:

```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
.\myenv\Scripts\python.exe -m streamlit run streamlit_frontend_resumechat_feature.py
```

Set `GROQ_API_KEY` in your environment or local `.env` file. Stock lookups also
require `ALPHAVANTAGE_API_KEY`. The embedded stock key was removed; rotate it
because it was exposed in source. Never commit credentials or local databases.

## Diagnosis

1. Full conversation and tool history grows without a bound and is sent to Groq
   on every turn. This can increase inference latency for long chats.
2. Previously, every rerun scanned all saved checkpoints and loaded each
   conversation to calculate sidebar titles. Work grows with saved history.
3. PDF parsing and embedding are synchronous. MiniLM construction previously
   happened during backend import, before the login screen appeared.
4. Tool calls add network latency and another model pass. Groq previously had
   no explicitly configured application timeout or retry count.
5. Streaming already existed. Missing early tool status and timing logs made
   delays hard to distinguish. Four 1,000-character chunks are not, on their
   own, evidence of excessive retrieval.

These are code-based findings, not measured production bottlenecks.

## Changes

- Native centered chat, light teal theme, account selector, sidebar settings,
  example questions, Markdown responses, copy expanders, and source previews.
- Empty chats show a centered inline composer. After the first answer, one
  UI-only rerun moves it to Streamlit's native bottom region without repeating
  the model request. Active chat messages stay above it; clearing starts the
  centered layout again. Stream failures remain visible after layout reruns.
- Backend import deferred until authentication. MiniLM is lazy and cached with
  `st.cache_resource`, loading on the first PDF upload or enabled semantic memory operation. The graph is shared as
  before; user conversations retain distinct thread IDs.
- Conversation titles cached per session. Checkpoint enumeration runs at most
  once per 30 seconds during normal reruns, or immediately on manual refresh.
- Existing message streaming retained; graph updates expose earlier tool status.
- PDF content hashes prevent redundant indexing while detecting changed files
  with identical filenames. Empty/extraction-free PDFs receive an error.
- Clear chat starts a fresh graph thread, not just an empty on-screen transcript.
  Saved conversations remain available in History.
- Groq timeout is 60 seconds per request, with up to two retries. Retries and
  tool loops can make total turn time longer than 60 seconds. The stock request
  retains its existing 10-second timeout. Stream errors retain partial text.
- SQLite account connections close explicitly, including on errors.

The model, temperature, existing tools, chunking, retrieval top-k, and full
LLM history are unchanged. With memory off, model inputs and available tools
remain unchanged. Opting into memory adds a recall system message and the
`save_memory` tool. Model/temperature/top-k remain read-only.

## Measure

Enable **Settings > Timing details** for first-visible-text and total response
time. Server logs include `retrieval_seconds`, `ingestion_seconds`,
`time_to_first_token_seconds`, and `total_response_seconds`.
Enabled memory also logs `memory_recall_seconds` per model call, including
model calls after tool execution. No recalled fact text is logged.

Compare before/after using the same PDF, question, chat history, machine, and
network. Separate process startup, first upload/model download, and warm runs.
Use at least five runs per scenario and compare medians, plus slowest runs.
For the original version, use browser timings or equivalent instrumentation;
it did not have these timing logs. No measured model speed-up is claimed.

Caching reduces local work, not Groq inference latency. Full history can still
slow long conversations. The first PDF upload still loads/downloads MiniLM.
Retrieved-source previews show evidence returned by RAG, not verified per-claim
citations; answers without a RAG call have no PDF sources. Preview snippets are
truncated for display only, not for model input.

PDF indexes remain process-local and must be uploaded again after restart.
Only the most recently indexed PDF is active per thread, as before. Sidebar
updates from another browser session may wait up to 30 seconds plus a rerun.
Clear chat preserves data rather than deleting checkpoints. Local SQLite auth
is retained, not replaced with a production identity provider.

## Test

```powershell
.\myenv\Scripts\python.exe -m unittest test_frontend test_memory -v
```

The 26 tests use fake models, deterministic embeddings, and temporary account databases,
covering registration/login, streaming, restored sources, title caching,
clear-chat isolation, API errors, user filtering, logout, malformed sources,
safe checkpoint-database failure handling and retry, and real SQLite saver
selection of a fresh database without modifying a damaged file.
Composer coverage checks inline-to-bottom placement, exactly one input, no
duplicate first turn, continued pinning, and reset to center after clear.
Memory tests additionally cover trusted identity, per-user CRUD isolation,
cross-thread recall, opt-out, sensitive data, explicit user-only provenance,
conflict replacement, recall limits, injected tool arguments, non-checkpointed
recall prompts, store selection/startup failures, account UUID migration, and
the sidebar controls and saved-memory notice. They make no external API calls
and do not download embeddings. Postgres selection/setup is mocked; no real
Postgres server was available. Live Groq, actual MiniLM memory embeddings,
real PDF embedding, Postgres connectivity, and multi-user load require separate validation.

## Groq Rate Limits And Retry

`groq.RateLimitError` / HTTP 429 means the provider's shared usage limit was
reached, not necessarily a bad API key, network failure, or broken memory store.
The reported Cloud failure was an 8,000 tokens-per-minute limit. The supplied
log included many model calls and growing request sizes; it does not identify
every tool call that contributed. Memory recall itself was completing.

The UI now shows a specific rate-limit message and an error reference matching
`chat_turn_failed error_reference=...` in server logs. After a rate-limit error,
the composer is temporarily blocked to avoid re-appending the same prompt.
Wait about one minute, then click **Retry response** on the latest failed answer.
An early click sends no request. No background wait loop or automatic graph
retry was added; the existing Groq SDK retry settings remain unchanged.

Retry checks for a pending `chat_node` and resumes the saved graph with null
input. It does not duplicate the user's message or replay completed tool work.
New/clear chat remains an explicit alternative. Retries need a still-available
checkpoint, and another user's shared provider usage can still produce a 429.
A single request larger than the provider's limit cannot be fixed by waiting.

To prevent repeated memory-save loops, the model is offered `save_memory` at
most until its first result in each human turn (saved, unchanged, or rejected).
After that, the same model can answer or call the original tools, but is no
longer offered another memory save for that turn. A new human message enables
the memory tool again. No model, temperature, chunking, retrieval, full-history,
or provider-plan changes were made. Lowering history/token budgets or changing
models still requires user approval. Existing checkpoint corruption is separate.

## Opt-In Long-Term Memory

The sidebar **Memory > Enable memory** toggle defaults to off for each login
session. It controls automatic saving and recall, not access to management:
you can still inspect, edit, delete, or clear your stored memories while off.
Disabling does not erase existing memories. Signing out clears session state;
creating or clearing a conversation does not clear cross-thread memories.

The compiled graph receives a LangGraph `BaseStore` alongside the existing
checkpointer. Memories live in `("memories", user_id)`. Here `user_id` is a
server-generated UUID stored in the authenticated account's `memory_user_id`
column, not a model argument or text extracted from a prompt. Existing accounts
are backfilled once without changing their IDs, passwords, or chat threads.
Sessions signed in before this update must sign out and back in to enable memory.
UUIDs prevent a freshly recreated SQLite account database from inheriting
Postgres memories belonging to a previous account with the same numeric ID.
Preserve the auth database/identity mapping if you need continued access to
persisted memories; it remains a local database, not a production identity provider.

### Storage And Dependencies

If `DATABASE_URL` is unset, the app uses an indexed `InMemoryStore`: shared across
threads in the same process but lost on restart. A configured `DATABASE_URL`
selects `PostgresStore`, runs its setup migrations, and requires a PostgreSQL
server with `pgvector` available and suitable extension/table permissions.
Use an existing or self-hosted server; this code provisions no paid service.
Set the connection string privately in your environment or root-level Cloud
Secrets. Do not paste it into chat or commit it. Reboot after changing it.
Configured Postgres failures do not silently downgrade to RAM. Store choice is
logged as `memory_store=postgres` or `memory_store=in_memory` without the URL.

New direct requirement specifications are:

```text
langgraph-checkpoint-postgres>=3.0.0
psycopg[binary,pool]>=3.2.0
```

The latter installs `psycopg`, its `psycopg-binary` wheel, and `psycopg-pool`.
Existing MiniLM embeddings are reused; the semantic index is 384-dimensional
and indexes only the `fact` field. No additional embedding model or API is used.
This Postgres store does not replace or repair the SQLite chat checkpointer.

### Saving And Conflicts

To make provenance strict, saving requires a latest user message of the form
`Remember that <one first-person fact>`. The tool must quote that fact exactly
(case and trailing full stop aside). It cannot supply identity or source text:
LangGraph injects trusted config, state messages, and the store. A tool-only,
PDF, web, quoted, paraphrased, or inferred claim is not a valid source.

Supported single-value categories and example facts:

| Category | Example |
| --- | --- |
| `target_role` | My target role is GenAI Engineer |
| `preferred_location` | My preferred location is Pune |
| `response_style` | I prefer concise answers |
| `preferred_language` | My preferred language is English |
| `career_goal` | My career goal is building useful products |
| `skills` | My skills are Python and SQL |

Each category has one stable key per user. Repeating an identical fact is a
no-op; a changed fact replaces the value in that category rather than adding
a conflicting duplicate. The skills category represents the whole declared
skill set; include all skills you want retained when replacing it. This is a
deliberately bounded profile, not an unrestricted autobiography or automatic
fact extractor. Arbitrary categories and paraphrasing are not supported.

Before each enabled model call, the latest human message is used to retrieve
up to four semantic matches in that user's namespace. They are injected under
**Known about the user** as profile data, not instructions. The recall message
is not appended to checkpointed state. Existing model/tool outputs are not
automatically saved. The sidebar lists the stored facts, supports edits and
deletes, and requires confirmation for Clear all; a toast confirms saved memory.

### Privacy Boundaries

Writes and sidebar edits use the same conservative validation before indexing:
short first-person category-matched statements only; no numbers, identifiers,
credential markers, common health/payment terms, URLs, email addresses, control
characters, quoted document markers, or instruction-like text. This rejects
some harmless facts too, such as numeric software versions or healthcare roles.
Filtering is deterministic defense in depth, not a guarantee of understanding
every secret or medical phrase in every language. Do not submit sensitive data.

These filters govern the new memory store. Existing chat checkpoints still
contain chat messages; deleting a memory does not scrub previous conversations,
old model answers, database backups, or previously saved chat tool outputs.
Disabling memory prevents new recall/saves but does not make earlier messages
in the current thread disappear. Verify forgetting behavior in a fresh thread.

### Manual Two-Thread Verification

1. Restart the updated app, sign in, and enable Memory. In thread A send:
  `Remember that My target role is GenAI Engineer`.
  Confirm the Memory saved toast and the target-role entry in the sidebar.
2. Click New conversation to create thread B, leaving memory enabled. Ask:
  `What is my target role?` It should recall GenAI Engineer. Server logs should
  include `memory_recall_seconds`. A missing toast is not proof of a save;
  check the list because the LLM must actually invoke the tool.
3. In B send `Remember that My target role is Data Engineer`. The sidebar
  should show one target-role entry, not two. A fresh third thread should use
  Data Engineer rather than the old value.
4. Edit or delete the entry in the sidebar and verify in another fresh thread.
  Clear all requires the confirmation checkbox. Disable memory and verify no
  memory recall/save occurs in a new thread.
5. Sign out and sign in as a different account. Enable Memory: the first
  account's list and target role must not appear. Edits/deletes must affect
  only this second account.
6. With dummy, never-real values, request saving a password, identifier, or
  health fact. Confirm no corresponding item appears. Uploading a PDF or
  receiving web/tool output alone must not create memories.
7. Local RAM mode: restart and confirm memory is gone. Postgres mode: restart,
  sign into the same preserved account, enable memory, and confirm retained
  items. This persistence check needs your own configured Postgres server.

## Cloud SQLite Errors

If saved history cannot load, the app stops chat initialization, keeps the
database unchanged, and offers **Retry storage**. It does not silently replace
SQLite with an in-memory store or treat a failed query as an empty history.

In Streamlit Cloud, open **Manage app > Logs** and find
`checkpoint_database_error`. The log includes the original exception and the
SQLite result name/code when available. The public redacted traceback alone
cannot distinguish a malformed database, incompatible schema, file permission
problem, or lock. Retry is for transient failures; it does not repair corruption.

Back up deployed database files before any repair or replacement. Do not delete
them or edit their schema based only on the redacted traceback. Diagnose the
specific SQLite error first. Repository edits do not repair an existing deployed
database, and Streamlit Cloud local storage should not be treated as durable
long-term storage.

### Confirmed Malformed Database

`DatabaseError: database disk image is malformed` confirms corruption in the
checkpoint database. Retrying, adjusting the model, or reinstalling packages
does not repair that file. The logs do not establish how it became corrupted.

If old chats matter, back up the deployed database before restarting or
redeploying. Obtain a consistent backup with writers stopped, including any
associated WAL files. A separate local database is not a backup of the cloud
database. Recovery should operate on a copy, not the only original.

If you explicitly accept starting with empty chat history, deploy the updated
backend and set this root-level value in Streamlit Cloud app secrets:

```toml
CHATBOT_DB_PATH = "chatbot_fresh.db"
```

Place it outside any TOML table. Streamlit exposes root-level secrets as
environment variables. Choose a filename that does not already exist. Reboot
the app to discard the cached backend and open the new file. Accounts still
use the existing auth database; previous conversations are not migrated to
the fresh checkpoint file. This changes neither the model nor retrieval.

The default remains `chatbot.db` when the variable is unset. No automatic
reset, replacement, or recovery is performed. Changing the path does not
overwrite the damaged file, but Cloud runtime files are still ephemeral:
do not rely on a reboot or redeploy to preserve your only backup.