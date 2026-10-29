# Resume Chat

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
- Backend import deferred until authentication. MiniLM is lazy and cached with
  `st.cache_resource`, loading on the first PDF upload. The graph is shared as
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

The model, temperature, prompts, tool selection, chunking, retrieval top-k,
and full LLM history are unchanged. Model/temperature/top-k are displayed
read-only because changing them or trimming history requires your approval.

## Measure

Enable **Settings > Timing details** for first-visible-text and total response
time. Server logs include `retrieval_seconds`, `ingestion_seconds`,
`time_to_first_token_seconds`, and `total_response_seconds`.

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
.\myenv\Scripts\python.exe -m unittest test_frontend -v
```

The five isolated AppTests use a fake graph and temporary account databases,
covering registration/login, streaming, restored sources, title caching,
clear-chat isolation, API errors, user filtering, logout, and malformed sources.
They make no external API calls and do not download embeddings. Live Groq,
real PDF embedding, and multi-user load performance need separate validation.