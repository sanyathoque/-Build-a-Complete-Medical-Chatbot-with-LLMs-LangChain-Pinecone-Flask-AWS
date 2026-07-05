from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings


def load_pdf_documents(data_dir: str) -> List[Document]:
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"PDF folder not found: {data_dir}")

    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()

    return simplify_metadata(documents)


def simplify_metadata(documents: List[Document]) -> List[Document]:
    simple_documents = []

    for document in documents:
        simple_documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    "source": document.metadata.get("source", "unknown"),
                    "page": document.metadata.get("page"),
                },
            )
        )

    return simple_documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return splitter.split_documents(documents)
