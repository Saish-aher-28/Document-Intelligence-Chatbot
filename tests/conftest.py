"""
Shared pytest fixtures for the College RAG Chatbot test suite.
"""
import io
import os
import pytest


# ---------------------------------------------------------------------------
# Minimal valid PDF bytes fixture
# A 1-page PDF with ASCII text "Hello Test Page 1" embedded as a Type1 font
# string. The xref offsets are approximate — pypdf is lenient enough to parse
# this for basic structure tests, but text extraction may or may not succeed
# depending on the pypdf version.
# ---------------------------------------------------------------------------
MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello Test Page 1) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n"
)


def _make_real_pdf(text: str = "Hello Test Page 1") -> bytes:
    """
    Generate a real PDF using fpdf2 so pypdf can extract text from it.
    Falls back to MINIMAL_PDF_BYTES if fpdf2 is unavailable.
    """
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text=text, new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())
    except Exception:
        return MINIMAL_PDF_BYTES


@pytest.fixture
def sample_pdf_bytes():
    """Return bytes of a real single-page PDF with extractable text."""
    return _make_real_pdf("Hello Test Page 1")


@pytest.fixture
def sample_session_id():
    """Return a reproducible test session ID."""
    return "test_session_abc123"


@pytest.fixture
def tmp_working_dir(tmp_path):
    """Return a temporary directory that acts as the project working_dir."""
    return str(tmp_path)


class MockUploadedFile:
    """Simulates a Streamlit UploadedFile object."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content


@pytest.fixture
def mock_uploaded_pdf(sample_pdf_bytes):
    """Return a mock Streamlit UploadedFile with valid PDF content."""
    return MockUploadedFile("test_document.pdf", sample_pdf_bytes)


@pytest.fixture
def mock_uploaded_txt():
    """Return a mock Streamlit UploadedFile with a .txt file (invalid type)."""
    return MockUploadedFile("notes.txt", b"This is not a PDF")


@pytest.fixture
def mock_uploaded_empty_pdf():
    """Return a mock Streamlit UploadedFile with 0 bytes."""
    return MockUploadedFile("empty.pdf", b"")


@pytest.fixture
def mock_uploaded_corrupted_pdf():
    """Return a mock Streamlit UploadedFile with invalid PDF header."""
    return MockUploadedFile("corrupted.pdf", b"%PDF-BROKEN\x00\x01\x02garbage")
