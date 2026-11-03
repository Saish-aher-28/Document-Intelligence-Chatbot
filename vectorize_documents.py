"""
vectorize_documents.py
----------------------
Indexes PDF files from the data/ directory into the ChromaDB college knowledge base (vector_db_dir/).

Usage:
    python vectorize_documents.py

Place your college PDF documents in the data/ directory, then run this script once.
The resulting vector_db_dir/ is used by the College Assistant tab in main.py.
"""

import os
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Optional Tesseract OCR (not required for text-based PDFs)
try:
    import pytesseract
    _TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.isfile(_TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def load_pdf_documents(directory):
    """Load and extract text from PDF files, one Document per page."""
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file}...")
            file_path = os.path.join(directory, pdf_file)

            reader = PdfReader(file_path)
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file,
                        "page": page_num,
                        "page_label": str(page_num + 1),
                        "total_pages": total_pages,
                    },
                )
                documents.append(doc)

            print(f"Successfully processed {pdf_file} ({total_pages} pages)")

        except PdfReadError as e:
            print(f"Could not read {pdf_file}: {e}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    return documents


def main():
    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF files here.")
        return

    # Ensure the vector database directory exists
    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")
        print("Created 'vector_db_dir' directory for storing vectorized documents.")

    try:
        # Load the embedding model
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings()

        # Load and process PDF documents
        print("Loading and processing PDF documents...")
        documents = load_pdf_documents("data")

        if not documents:
            print("No documents were successfully processed. Please check your PDF files.")
            return

        print(f"Successfully loaded {len(documents)} page(s) from PDFs")

        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        text_chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(text_chunks)} chunks")

        # Create and persist the vector database
        print("Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )

        print("Successfully vectorized and stored documents in 'vector_db_dir'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure all required dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
