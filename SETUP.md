# Document Intelligence Chatbot — Quick Setup

## Requirements
- Python 3.11
- Internet connection (for Groq API + model downloads on first run)

---

## Setup Steps

### 1. Extract the ZIP
Extract the folder anywhere on your PC.

### 2. Open a terminal in the project folder
```
cd College_RAG_Chatbot-main
```

### 3. (Optional but recommended) Create a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Set up the API key
Copy `.env.example` to `.env`:
```
copy .env.example .env
```
The `.env` file already contains the Groq API key — no changes needed.

### 6. Run the app
```
python -m streamlit run main.py
```

Then open your browser at: **http://localhost:8501**

---

## How to use

- **Document Chat tab** — Upload one or more PDFs, click "Process Documents", then ask questions about them.
- **College Assistant tab** — Ask general questions answered from the preloaded college knowledge base.

---

## Notes
- First run takes ~1–2 minutes to download the embedding model (~90 MB).
- The Groq API key is included in `.env` — no registration needed.
- If port 8501 is busy, use: `python -m streamlit run main.py --server.port 8502`
