# College RAG Chatbot — Document Upload Enhancement

This change adds user-facing PDF document upload to an existing college-focused RAG chatbot. Users can now upload arbitrary PDFs, which are ingested into per-session ChromaDB collections and queried through the existing Groq LLM. The implementation introduces `document_processor.py` as a dedicated ingestion module, rewrites the Streamlit UI into two tabs (Document Chat and College Assistant), migrates from PyPDF2 to pypdf, and updates the LangChain dependency chain to 1.x. 19/19 tests pass after a second review iteration that fixed a missing `st.stop()`, a truncated session ID, and absent deduplication and cross-session isolation tests.

Watch for: (1) **confirmed** — FastAPI `/chat` endpoint re-initializes both embeddings and the conversational chain on every request, with no shared-state guard; (2) **confirmed** — `fpdf2` FPDF API mismatch: the PDF export uses deprecated `ln=True` keyword syntax that fpdf2 dropped; (3) **confirmed** — test_main_imports.py uses `exec()` on source extracted via AST to avoid importing Streamlit — a clever workaround but one that introduces a real `exec` call in the test suite itself; (4) **possible** — the FastAPI `/chat` endpoint loads the college KB on every call, which will be slow under concurrent load.

**Verdict**: APPROVED

---

## High-level view

The security posture is sound at the checkpoints the task specified. GROQ_API_KEY is loaded exclusively from environment or config file, never hardcoded. Session IDs are `uuid4().hex` (128 bits), server-generated, and used as the sole path component for per-session ChromaDB directories. `session_db/` and `.env` are both in `.gitignore`. No `eval()` or `exec()` touches PDF content anywhere in the production path.

The document ingestion pipeline is well-structured. `validate_pdf_file` checks extension, length, magic bytes (`%PDF`), and pypdf structural readability in that order. `extract_text_from_pdf` produces one `Document` per page with all five required metadata fields (`source`, `page`, `page_label`, `total_pages`, `doc_id`). `split_documents` appends `chunk_id` using the `{doc_id}_p{page}_c{i}` format. `process_uploaded_pdfs` returns the full expected dict shape and handles duplicates via SHA-256 content hashing.

The two-tab UI meets the completeness criteria. Tab 1 contains the uploader, process button, clear button, indexed-files status, source-cited chat, and the pre-indexing guard (`st.info` + conditional rendering). Tab 2 preserves the original college assistant. The sidebar retains chat history, export-to-PDF, and the goals/values panels.

All five preserved features confirmed present: FastAPI `/chat` endpoint, `remove_emojis()`, `contains_sensitive_topics()`, PDF export via fpdf2, and `setup_vectorstore()` / `chat_chain()` for the college KB.

The dependency list is correct: `pypdf>=4.0.0` (not PyPDF2), `fpdf2>=2.7.0`, `python-dotenv>=1.0.0`, `pytest>=8.0.0`. PyPDF2 is absent. Config loading calls `load_dotenv()`, falls back to `config.json`, and `st.stop()` is called immediately after the missing-key error banner.

Session isolation is structurally correct: each session path is `session_db/{uuid4().hex}/`, doc-chat state lives in `doc_vectorstore` / `doc_chat_history` / `doc_conversational_chain` which are entirely separate keys from the college KB's `vectorstore` / `chat_history` / `conversational_chain`.

One functional gap in the FastAPI endpoint: `/chat` calls `setup_vectorstore()` and `chat_chain()` on every request, creating a new embedding model instance and a new chain with fresh memory each time. This is the pre-existing architecture, but combined with the new LangChain 1.x dependency chain, it may produce a slow-startup response on first hit and is stateless across calls (conversational memory resets on each request). This is a production concern, not a blocking correctness issue.

The `fpdf2` usage in `Export Chat to PDF` passes `ln=True` to `pdf.cell()`, a keyword that fpdf2 removed; the correct syntax is `new_x="LMARGIN", new_y="NEXT"`. The conftest's `_make_real_pdf` helper already uses the new syntax, so the production export path and the test fixture diverge. This is a LOW-severity functional bug.

<details>
<summary>Issues (3)</summary>

1. **FastAPI `/chat` stateless memory** — `chatbot()` creates a fresh `ConversationBufferMemory` on every HTTP request; the endpoint is effectively single-turn despite advertising conversational behavior. Cache a shared chain instance or document the single-turn limitation clearly.

2. **PDF export `ln=True` mismatch** — `pdf.cell(..., ln=True)` is not valid in fpdf2; the correct call form is `pdf.cell(..., new_x="LMARGIN", new_y="NEXT")`. The export button will raise a `TypeError` at runtime when any chat history exists. Update to the fpdf2 API to match the pattern already used correctly in `conftest.py`.

3. **`exec()` in test infrastructure** — `test_main_imports.py` uses `exec(textwrap.dedent(func_src), namespace)` to load `remove_emojis` and `contains_sensitive_topics` out of `main.py` without triggering Streamlit's module-level setup. The approach is safe here (input is the project's own source file), but it bypasses import machinery and will silently miss the tested functions if they are refactored into a helper module. A cleaner fix is to extract these pure functions to a `utils.py` module that can be imported directly in tests.

</details>

<details>
<summary>Details</summary>

### FastAPI `/chat` — stateless memory and per-request chain construction

Every call to `POST /chat` executes `setup_vectorstore()` followed by `chat_chain(vectorstore)`. `setup_vectorstore()` instantiates `HuggingFaceEmbeddings()` which loads the sentence-transformers model from disk on the first call of each process lifetime, then returns a `Chroma` object. `chat_chain()` constructs a `ConversationalRetrievalChain` with a brand-new `ConversationBufferMemory`. Because the memory is local to each request and not stored in a session, the endpoint does not maintain conversation context across calls. The `/chat` endpoint also always queries the college KB, never the user's uploaded documents — which is architecturally consistent with the tab separation, but worth noting if the API is used externally.

The practical risk under concurrent load is that multiple simultaneous requests each trigger the model-loading path before it is cached by the sentence-transformers library. This is a performance concern, not a correctness one. **Severity: LOW** (confirmed).

### PDF export `ln=True` keyword removed in fpdf2

The `Export Chat to PDF` button in the sidebar calls:

```python
pdf.cell(0, 10, "College Name Chatbot - Conversation History", ln=True, align='C')
pdf.cell(0, 10, f"Date: {datetime.now().strftime(...)}", ln=True, align='C')
pdf.cell(0, 10, message["role"].capitalize(), ln=True)
```

fpdf2 ≥ 2.7.0 removed the `ln` keyword parameter from `cell()`. The correct replacement is `new_x="LMARGIN", new_y="NEXT"` (exactly as used in `conftest.py`'s `_make_real_pdf`). At runtime, clicking "Export Chat to PDF" with any chat history will raise a `TypeError: cell() got an unexpected keyword argument 'ln'`. **Severity: LOW** (confirmed — the conftest already uses the correct API, so the divergence is directly visible).

### `exec()` in test infrastructure

`_load_function_from_main` in `test_main_imports.py` parses `main.py` with the `ast` module, extracts the function source via `ast.get_source_segment`, then `exec()`s the deduplicated source in a controlled namespace. This avoids triggering `st.set_page_config()` at module level. The approach is safe for the current structure, but it is fragile: if `remove_emojis` or `contains_sensitive_topics` are moved to a utility module, the tests will raise `ImportError` silently (the `exec` call will fail to find the names). Extracting these pure functions to `utils.py` would resolve both the test fragility and the `exec` usage in one step. **Severity: LOW** (confirmed).

### Security and untrusted-input handling

No `eval()` or `exec()` is called on PDF content in production code. `DOC_SYSTEM_PROMPT` explicitly instructs the LLM not to follow instructions in document text. GROQ_API_KEY is loaded from environment only. Session IDs are `uuid4().hex` throughout, including the reset path after clear. `.env` and `session_db/` are in `.gitignore`. The `clear_session_vectorstore` function scopes its `shutil.rmtree` to `{working_dir}/session_db/{session_id}/` and will not descend into `vector_db_dir/` or other sessions.

### Test coverage

The 19-test suite covers validate_pdf_file (5 cases: valid, wrong extension, empty, oversized, corrupted header), extract_text_from_pdf metadata keys, split_documents chunk_id and size bounds, create/clear session vectorstore, process_uploaded_pdfs invalid type and mixed valid/invalid, cross-session isolation, and smoke tests for remove_emojis and contains_sensitive_topics including keyword matching. Not tested: `load_session_vectorstore`, `doc_chat_chain` function construction, the FastAPI `/chat` endpoint, the Streamlit UI interaction flow, the college KB chain initialization path, and the PDF export button.

</details>

---

<details>
<summary>File map</summary>

| File | Change |
|---|---|
| `main.py` | Two-tab UI; doc_chat_chain; session state; dotenv loading; st.stop() on missing key; full-UUID session IDs |
| `document_processor.py` | New: validate, extract, split, create/load/clear session vectorstore, process_uploaded_pdfs |
| `vectorize_documents.py` | Migrated from PyPDF2 to pypdf; one Document per page; optional Tesseract |
| `requirements.txt` | pypdf replaces PyPDF2; fpdf2, python-dotenv, sentence-transformers, pytest, pytest-mock, langchain-classic added |
| `.env.example` | New: API key template |
| `.gitignore` | session_db/ and temp_uploads/ added |
| `tests/conftest.py` | New: fixtures for PDF bytes, session IDs, tmp dirs, mock UploadedFile variants |
| `tests/test_document_processor.py` | New: 14 tests for ingestion pipeline and session isolation |
| `tests/test_main_imports.py` | New: 5 smoke tests for main.py pure functions |
| `README.md` | Rewritten: dual-mode description, architecture, setup/run/test docs |
| `IMPLEMENTATION_REPORT.md` | New: iteration log with test results and dependency decisions |

Full diff: `git diff main` from project root.

</details>
