import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str = require_env("PINECONE_API_KEY")
    openai_api_key: str = require_env("OPENAI_API_KEY")

    index_name: str = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    embedding_dimension: int = 384

    pdf_data_dir: str = os.getenv("PDF_DATA_DIR", "data")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "3"))


settings = Settings()
