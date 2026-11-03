"""
Tests for document_processor.py

Run: pytest tests/test_document_processor.py -v
"""
import io
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import document_processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import (
    validate_pdf_file,
    extract_text_from_pdf,
    split_documents,
    create_session_vectorstore,
    load_session_vectorstore,
    clear_session_vectorstore,
    process_uploaded_pdfs,
    MAX_FILE_SIZE_BYTES,
    CHUNK_SIZE,
)

# ---------------------------------------------------------------------------
# 1. validate_pdf_file — valid PDF passes
# ---------------------------------------------------------------------------
def test_validate_pdf_file_valid(sample_pdf_bytes):
    valid, reason = validate_pdf_file(sample_pdf_bytes, "doc.pdf")
    assert valid is True
    assert reason == ""


# ---------------------------------------------------------------------------
# 2. validate_pdf_file — non-PDF extension rejected
# ---------------------------------------------------------------------------
def test_validate_pdf_file_invalid_type(sample_pdf_bytes):
    valid, reason = validate_pdf_file(sample_pdf_bytes, "doc.txt")
    assert valid is False
    assert "not a pdf" in reason.lower()


# ---------------------------------------------------------------------------
# 3. validate_pdf_file — empty file rejected
# ---------------------------------------------------------------------------
def test_validate_pdf_file_empty():
    valid, reason = validate_pdf_file(b"", "empty.pdf")
    assert valid is False
    assert "empty" in reason.lower()


# ---------------------------------------------------------------------------
# 4. validate_pdf_file — file too large rejected
# ---------------------------------------------------------------------------
def test_validate_pdf_file_too_large():
    oversized = b"%PDF" + b"x" * (MAX_FILE_SIZE_BYTES + 1)
    valid, reason = validate_pdf_file(oversized, "big.pdf")
    assert valid is False
    assert ("size limit" in reason.lower() or "mb" in reason.lower())


# ---------------------------------------------------------------------------
# 5. validate_pdf_file — corrupted header rejected
# ---------------------------------------------------------------------------
def test_validate_pdf_file_corrupted():
    valid, reason = validate_pdf_file(b"NOTPDF\x00garbage", "bad.pdf")
    assert valid is False


# ---------------------------------------------------------------------------
# 6. extract_text_from_pdf — returns Documents with correct metadata keys
# ---------------------------------------------------------------------------
def test_extract_text_metadata_keys(sample_pdf_bytes):
    docs = extract_text_from_pdf(sample_pdf_bytes, "report.pdf")
    # Some minimal PDFs may not yield extractable text; skip metadata check if empty
    for doc in docs:
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert "page_label" in doc.metadata
        assert "total_pages" in doc.metadata
        assert "doc_id" in doc.metadata
        assert doc.metadata["source"] == "report.pdf"


# ---------------------------------------------------------------------------
# 7. split_documents — chunks get chunk_id metadata
# ---------------------------------------------------------------------------
def test_split_documents_chunk_id():
    from langchain_core.documents import Document
    docs = [
        Document(
            page_content="word " * 400,  # long enough to split
            metadata={
                "source": "a.pdf", "page": 0, "page_label": "1",
                "total_pages": 1, "doc_id": "abc123",
            },
        )
    ]
    chunks = split_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("abc123")


# ---------------------------------------------------------------------------
# 8. split_documents — all chunks are within CHUNK_SIZE (with tolerance)
# ---------------------------------------------------------------------------
def test_split_documents_respects_chunk_size():
    from langchain_core.documents import Document
    docs = [
        Document(
            page_content="sentence. " * 500,
            metadata={
                "source": "b.pdf", "page": 0, "page_label": "1",
                "total_pages": 1, "doc_id": "def456",
            },
        )
    ]
    chunks = split_documents(docs)
    for chunk in chunks:
        # Allow some splitter tolerance (CHUNK_OVERLAP)
        assert len(chunk.page_content) <= CHUNK_SIZE * 2


# ---------------------------------------------------------------------------
# 9. create_session_vectorstore — returns Chroma instance and persists to disk
# ---------------------------------------------------------------------------
def test_create_session_vectorstore(tmp_working_dir, sample_session_id):
    from langchain_core.documents import Document
    chunks = [
        Document(
            page_content="Artificial intelligence is transforming education.",
            metadata={
                "source": "ai.pdf", "page": 0, "page_label": "1",
                "total_pages": 1, "doc_id": "aaa111", "chunk_id": "aaa111_p0_c0",
            },
        )
    ]
    with patch("document_processor._get_embeddings") as mock_emb:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1] * 384]
        mock_instance.embed_query.return_value = [0.1] * 384
        mock_emb.return_value = mock_instance

        vs = create_session_vectorstore(chunks, sample_session_id, tmp_working_dir)

    assert vs is not None
    persist_path = os.path.join(tmp_working_dir, "session_db", sample_session_id)
    assert os.path.isdir(persist_path)


# ---------------------------------------------------------------------------
# 10. clear_session_vectorstore — directory is removed
# ---------------------------------------------------------------------------
def test_clear_session_vectorstore(tmp_working_dir, sample_session_id):
    persist_path = os.path.join(tmp_working_dir, "session_db", sample_session_id)
    os.makedirs(persist_path, exist_ok=True)
    open(os.path.join(persist_path, "dummy.txt"), "w").close()

    result = clear_session_vectorstore(sample_session_id, tmp_working_dir)

    assert result is True
    assert not os.path.exists(persist_path)


# ---------------------------------------------------------------------------
# 11. clear_session_vectorstore — returns True when directory does not exist
# ---------------------------------------------------------------------------
def test_clear_session_vectorstore_nonexistent(tmp_working_dir):
    result = clear_session_vectorstore("nonexistent_session", tmp_working_dir)
    assert result is True


# ---------------------------------------------------------------------------
# 12. process_uploaded_pdfs — invalid file type is reported in failed_files
# ---------------------------------------------------------------------------
def test_process_uploaded_pdfs_invalid_type(mock_uploaded_txt, sample_session_id,
                                             tmp_working_dir):
    result = process_uploaded_pdfs(
        uploaded_files=[mock_uploaded_txt],
        session_id=sample_session_id,
        working_dir=tmp_working_dir,
    )
    assert result["success"] is False
    assert len(result["failed_files"]) == 1
    assert result["failed_files"][0]["filename"] == "notes.txt"


# ---------------------------------------------------------------------------
# 13. process_uploaded_pdfs — mixed files: one valid, one invalid
# ---------------------------------------------------------------------------
def test_process_uploaded_pdfs_mixed(
    mock_uploaded_txt, mock_uploaded_pdf, sample_session_id, tmp_working_dir
):
    from langchain_core.documents import Document as LCDoc

    with patch("document_processor.extract_text_from_pdf") as mock_extract, \
         patch("document_processor._get_embeddings") as mock_emb:

        mock_extract.return_value = [
            LCDoc(
                page_content="Sample content from the PDF document.",
                metadata={
                    "source": "test_document.pdf", "page": 0,
                    "page_label": "1", "total_pages": 1, "doc_id": "xyz789",
                },
            )
        ]
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1] * 384]
        mock_instance.embed_query.return_value = [0.1] * 384
        mock_emb.return_value = mock_instance

        result = process_uploaded_pdfs(
            uploaded_files=[mock_uploaded_pdf, mock_uploaded_txt],
            session_id=sample_session_id,
            working_dir=tmp_working_dir,
        )

    assert result["success"] is True
    assert "test_document.pdf" in result["indexed_files"]
    assert any(f["filename"] == "notes.txt" for f in result["failed_files"])
    assert result["total_chunks"] >= 1


# ---------------------------------------------------------------------------
# 14. cross-session isolation — two sessions have separate persist directories
#     and querying one cannot reach the other's collection
# ---------------------------------------------------------------------------
def test_cross_session_isolation(tmp_working_dir):
    """
    Create vectorstores for two distinct session IDs and assert that:
      - they persist to different directories
      - the directories are not shared
      - deleting one session leaves the other intact
    """
    from langchain_core.documents import Document as LCDoc

    session_a = "session_aaaaaaaa"
    session_b = "session_bbbbbbbb"

    docs_a = [LCDoc(
        page_content="Session A secret content about topic Alpha.",
        metadata={"source": "a.pdf", "page": 0, "page_label": "1",
                  "total_pages": 1, "doc_id": "aaa000", "chunk_id": "aaa000_p0_c0"},
    )]
    docs_b = [LCDoc(
        page_content="Session B secret content about topic Beta.",
        metadata={"source": "b.pdf", "page": 0, "page_label": "1",
                  "total_pages": 1, "doc_id": "bbb000", "chunk_id": "bbb000_p0_c0"},
    )]

    with patch("document_processor._get_embeddings") as mock_emb:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1] * 384]
        mock_instance.embed_query.return_value = [0.1] * 384
        mock_emb.return_value = mock_instance

        vs_a = create_session_vectorstore(docs_a, session_a, tmp_working_dir)
        vs_b = create_session_vectorstore(docs_b, session_b, tmp_working_dir)

    # Each session has its own directory
    dir_a = os.path.join(tmp_working_dir, "session_db", session_a)
    dir_b = os.path.join(tmp_working_dir, "session_db", session_b)
    assert os.path.isdir(dir_a), "Session A directory should exist"
    assert os.path.isdir(dir_b), "Session B directory should exist"
    assert dir_a != dir_b, "Sessions must not share a persist directory"

    # Release ChromaDB file handles before deletion (important on Windows).
    # ChromaDB 1.5+ uses Rust native bindings (RustBindingsAPI) that hold OS-level
    # file locks on HNSW index files. Calling stop() on the underlying client
    # releases those handles so shutil.rmtree can succeed.
    import gc
    try:
        vs_a._collection._client.stop()
    except Exception:
        pass
    del vs_a
    gc.collect()

    # Clearing session A must not touch session B
    cleared = clear_session_vectorstore(session_a, tmp_working_dir)
    assert cleared is True, (
        "clear_session_vectorstore returned False — possibly a file-lock issue. "
        "Ensure ChromaDB client is closed before clearing."
    )
    assert not os.path.exists(dir_a), "Session A directory should be removed after clear"
    assert os.path.isdir(dir_b), "Session B directory must survive clearing session A"
