# Document Intelligence Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** with two modes:

1. **Document Chat** — Upload your own PDF documents and ask questions grounded in their content.
2. **College Assistant** — Ask questions about the college knowledge base (pre-indexed documents).

---

## Features

- Upload one or more PDF files directly in the browser (no manual file copying)
- Per-session document isolation — each browser session gets its own vector store
- Source citations with filename and page number on every answer
- Conversational memory for multi-turn Q&A
- College knowledge-base chat mode preserved from the original application
- Export chat history to PDF
- FastAPI REST endpoint for programmatic access
- Graceful handling of empty, corrupted, and password-protected PDFs

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | HuggingFace — sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (persistent, per-session) |
| Frontend | Streamlit |
| Backend | FastAPI |
| Orchestration | LangChain 1.x |
| PDF Parsing | pypdf 4.x |
| PDF Export | fpdf2 |

---

## Prerequisites

- Python 3.10 or higher (tested on 3.11.9)
- A Groq API key — get one free at [console.groq.com](https://console.groq.com)

---

## Installation

```bash
# 1. Clone or download the repository
cd College_RAG_Chatbot-main

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

### Option A — `.env` file (recommended)

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_actual_key
```

### Option B — `config.json`

Create `config.json` in the project root:

```json
{
  "GROQ_API_KEY": "gsk_your_actual_key_here"
}
```

`config.json` is already in `.gitignore` and will not be committed.

---

## Adding College Documents (College Assistant Mode)

Place PDF files in the `data/` directory, then run:

```bash
python vectorize_documents.py
```

This indexes the documents into `vector_db_dir/` for the College Assistant tab.

---

## Running the Application

```bash
streamlit run main.py
```

Opens at `http://localhost:8501`.

**FastAPI endpoint** (run in a separate terminal):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

POST `/chat` with `{"message": "your question"}`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Document Upload Workflow

1. Open the **Document Chat** tab.
2. Click **Browse files** and select one or more PDFs (max 50 MB each).
3. Click **Process & Index Documents**.
4. Wait for the success message showing how many chunks were indexed.
5. Type your question in the chat input.
6. Answers include source citations (filename + page number).
7. Click **Clear Session** to remove all uploaded documents and start fresh.

---

## Architecture

```
User Browser
    |
    v
Streamlit Frontend (main.py)
    |                          |
Document Chat Tab         College Assistant Tab
    |                          |
document_processor.py     setup_vectorstore()
    |                          |
Session ChromaDB           College ChromaDB
(session_db/{id}/)         (vector_db_dir/)
    |                          |
    +-----------+--------------+
                |
           ChatGroq LLM
      (llama-3.3-70b-versatile)
                |
         Answer + Sources
```

---

## Security Notes and Limitations

- **PDF content is treated as untrusted data.** Instructions embedded in uploaded PDFs are not executed or followed.
- **Session isolation is per-browser-session**, not per-authenticated-user. This application does not implement authentication. In a shared or multi-user deployment, session state is scoped only to the `st.session_state` object, not to separate OS users or network identities.
- **No authentication means no authorization.** Do not use this application to process confidential or sensitive documents in a shared-access environment without adding proper authentication.
- Session vector stores are stored in `session_db/<session_id>/` and are cleared when the user clicks **Clear Session**.
- The GROQ_API_KEY must be kept secret. Never commit it to version control.

---

## Known Limitations

- Scanned PDFs (image-only) cannot be processed — only text-extractable PDFs are supported.
- Password-protected PDFs are not supported.
- The College Assistant tab requires pre-indexed documents in `vector_db_dir/`. If none exist, it will return no relevant results.
- Session vector stores are not automatically cleaned up when a browser tab is closed. Run the following to clear all sessions manually:
  ```bash
  python -c "import shutil; shutil.rmtree('session_db', ignore_errors=True)"
  ```

---

## License

MIT License — see [LICENSE](LICENSE) file.  
Original project by [Deep Shah](https://github.com/DeepShah1406).
