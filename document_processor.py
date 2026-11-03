"""
document_processor.py
---------------------
PDF ingestion pipeline for the Document Intelligence Chatbot.

Responsibilities:
  - Validate uploaded PDF files (type, size, magic bytes, readability)
  - Extract text from PDFs using pypdf (one Document per page)
  - Split documents into overlapping chunks
  - Generate embeddings via HuggingFace sentence-transformers
  - Create / load / delete per-session ChromaDB collections

Security note:
  All document text is treated as untrusted data. Instructions embedded in PDFs
  are never executed or followed.
"""

import io
import os
import hashlib
import logging
import shutil
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (all configurable by editing this file)
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = 50
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
SESSION_DB_BASE_DIR: str = "session_db"   # relative to working_dir; absolutised in functions


# ---------------------------------------------------------------------------
# 1. validate_pdf_file
# ---------------------------------------------------------------------------
def validate_pdf_file(file_bytes: bytes, filename: str) -> tuple:
    """
    Validate that file_bytes represent a valid, non-empty, readable PDF.

    Returns (True, "") on success.
    Returns (False, reason_string) on any failure.

    Security: this function never executes or evals any content from the file.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    # 1. Extension check
    if not filename.lower().endswith(".pdf"):
        return False, f"'{filename}' is not a PDF file."

    # 2. Empty file check
    if len(file_bytes) == 0:
        return False, f"'{filename}' is empty."

    # 3. Size limit check
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"'{filename}' exceeds the {MAX_FILE_SIZE_MB} MB size limit."

    # 4. Magic bytes check
    if file_bytes[:4] != b"%PDF":
        return False, f"'{filename}' does not appear to be a valid PDF."

    # 5. Structural readability check
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            return False, f"'{filename}' is password-protected and cannot be processed."
    except (PdfReadError, Exception) as e:
        return False, f"'{filename}' could not be read: {e}"

    return True, ""


# ---------------------------------------------------------------------------
# 2. extract_text_from_pdf
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list:
    """
    Extract text from a PDF, producing one LangChain Document per page.

    Each Document's metadata contains:
        source       (str)  — filename
        page         (int)  — 0-based page index
        page_label   (str)  — 1-based human-readable page number as string
        total_pages  (int)  — total number of pages in the document
        doc_id       (str)  — sha256 hex digest of file_bytes (first 16 chars)

    Pages with no extractable text are skipped (logged at WARNING level).
    Returns list of Document objects (may be empty if no page had text).

    Security: page_content is treated as untrusted text; it is never exec'd or eval'd.
    """
    from pypdf import PdfReader

    doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)

    documents = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Could not extract text from page %d of '%s': %s",
                           page_num, filename, e)
            text = ""

        if not text.strip():
            logger.warning("Page %d of '%s' yielded no text — skipping.", page_num, filename)
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": page_num,
                "page_label": str(page_num + 1),
                "total_pages": total_pages,
                "doc_id": doc_id,
            }
        ))

    return documents


# ---------------------------------------------------------------------------
# 3. split_documents
# ---------------------------------------------------------------------------
def split_documents(documents: list) -> list:
    """
    Split a list of Documents into overlapping chunks.

    Adds chunk_id metadata to each chunk:
        chunk_id  (str)  — "{doc_id}_p{page}_c{chunk_index}"

    Returns list of chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        doc_id = chunk.metadata.get("doc_id", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{doc_id}_p{page}_c{i}"

    return chunks


# ---------------------------------------------------------------------------
# 4. _get_embeddings (internal helper)
# ---------------------------------------------------------------------------
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance using EMBEDDING_MODEL."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# 5. create_session_vectorstore
# ---------------------------------------------------------------------------
def create_session_vectorstore(
    chunks: list,
    session_id: str,
    working_dir: str,
) -> "Chroma":
    """
    Create (or overwrite) a Chroma collection for the given session.

    Collection name : f"session_{session_id}"
    Persist directory: {working_dir}/session_db/{session_id}/

    Isolation: each session gets its own subdirectory and collection name.
    No two sessions share document content.

    Returns the Chroma vectorstore.
    Raises ValueError if chunks is empty.
    Raises RuntimeError on ChromaDB write failure.
    """
    if not chunks:
        raise ValueError("No document chunks to index.")

    persist_dir = os.path.join(working_dir, SESSION_DB_BASE_DIR, session_id)
    os.makedirs(persist_dir, exist_ok=True)

    collection_name = f"session_{session_id}"
    # ChromaDB collection names must be 3-63 chars, no special chars except hyphens
    # Truncate/sanitize to be safe
    collection_name = collection_name[:63].replace(" ", "_")

    embeddings = _get_embeddings()

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        return vectorstore
    except Exception as e:
        shutil.rmtree(persist_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to create vector store: {e}") from e


# ---------------------------------------------------------------------------
# 6. load_session_vectorstore
# ---------------------------------------------------------------------------
def load_session_vectorstore(
    session_id: str,
    working_dir: str,
) -> Optional["Chroma"]:
    """
    Load an existing session vectorstore from disk.

    Returns None if the persist directory does not exist.
    """
    persist_dir = os.path.join(working_dir, SESSION_DB_BASE_DIR, session_id)
    if not os.path.isdir(persist_dir):
        return None

    collection_name = f"session_{session_id}"
    collection_name = collection_name[:63].replace(" ", "_")

    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


# ---------------------------------------------------------------------------
# 7. clear_session_vectorstore
# ---------------------------------------------------------------------------
def clear_session_vectorstore(session_id: str, working_dir: str) -> bool:
    """
    Delete the session's vectorstore directory from disk.

    Returns True if deletion succeeded or directory did not exist.
    Returns False on error (logs the exception).

    Safety: only deletes the path {working_dir}/session_db/{session_id}/.
    Never touches vector_db_dir/ (college KB) or other sessions.
    """
    persist_dir = os.path.join(working_dir, SESSION_DB_BASE_DIR, session_id)
    if not os.path.exists(persist_dir):
        return True

    try:
        shutil.rmtree(persist_dir)
        return True
    except Exception as e:
        logger.error("Failed to clear session '%s': %s", session_id, e)
        return False


# ---------------------------------------------------------------------------
# 8. process_uploaded_pdfs  (main pipeline entry point)
# ---------------------------------------------------------------------------
def process_uploaded_pdfs(
    uploaded_files: list,
    session_id: str,
    working_dir: str,
) -> dict:
    """
    Full pipeline: validate -> extract -> split -> embed -> index.

    Parameters
    ----------
    uploaded_files : list of Streamlit UploadedFile objects
                     (each has .name and .getvalue() / .read())
    session_id     : str  — unique identifier for the current user session
    working_dir    : str  — absolute path to the project root

    Returns
    -------
    dict with keys:
        success        (bool)
        indexed_files  (list[str])   — filenames successfully indexed
        failed_files   (list[dict])  — [{"filename": str, "reason": str}, ...]
        total_chunks   (int)
        vectorstore    (Chroma | None)
        error          (str | None)  — top-level error if success is False
    """
    indexed_files: list = []
    failed_files: list = []
    all_chunks: list = []
    seen_doc_ids: set = set()  # deduplicate by content hash so re-uploading is a no-op

    for uf in uploaded_files:
        filename = getattr(uf, "name", "unknown.pdf")

        # Get bytes — Streamlit UploadedFile supports .getvalue()
        try:
            file_bytes = uf.getvalue()
        except AttributeError:
            try:
                file_bytes = uf.read()
            except Exception as e:
                failed_files.append({"filename": filename, "reason": f"Could not read file: {e}"})
                continue

        # Validate
        valid, reason = validate_pdf_file(file_bytes, filename)
        if not valid:
            failed_files.append({"filename": filename, "reason": reason})
            continue

        # Deduplicate by content hash — same bytes = same doc_id
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]
        if doc_id in seen_doc_ids:
            # File was already included in this batch (duplicate upload)
            failed_files.append({
                "filename": filename,
                "reason": "Duplicate file: identical content was already included in this batch.",
            })
            continue
        seen_doc_ids.add(doc_id)

        # Extract text
        try:
            docs = extract_text_from_pdf(file_bytes, filename)
        except Exception as e:
            failed_files.append({"filename": filename, "reason": f"Extraction error: {e}"})
            continue

        if not docs:
            failed_files.append({
                "filename": filename,
                "reason": (
                    "No extractable text found. The PDF may be scanned or image-only. "
                    "OCR is not available in this application."
                ),
            })
            continue

        # Split
        try:
            chunks = split_documents(docs)
        except Exception as e:
            failed_files.append({"filename": filename, "reason": f"Chunking error: {e}"})
            continue

        all_chunks.extend(chunks)
        indexed_files.append(filename)

    # Nothing to index
    if not all_chunks:
        return {
            "success": False,
            "indexed_files": [],
            "failed_files": failed_files,
            "total_chunks": 0,
            "vectorstore": None,
            "error": "No text could be extracted from the uploaded documents.",
        }

    # Create vector store
    try:
        vectorstore = create_session_vectorstore(all_chunks, session_id, working_dir)
    except Exception as e:
        return {
            "success": False,
            "indexed_files": indexed_files,
            "failed_files": failed_files,
            "total_chunks": 0,
            "vectorstore": None,
            "error": str(e),
        }

    return {
        "success": True,
        "indexed_files": indexed_files,
        "failed_files": failed_files,
        "total_chunks": len(all_chunks),
        "vectorstore": vectorstore,
        "error": None,
    }
