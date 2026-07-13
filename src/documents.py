from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdfs(pdf_folder: str) -> list[Document]:
    documents = []
    pdf_paths = list(Path(pdf_folder).glob("*.pdf"))

    print(f"Found {len(pdf_paths)} PDF files")

    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    print(f"Loaded {len(documents)} PDF pages")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks")
    return chunks
