from pinecone import Pinecone, ServerlessSpec

from src.config import (
    EMBEDDING_DIMENSION,
    INDEX_NAME,
    PDF_FOLDER,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    check_keys,
)
from src.documents import load_pdfs, split_documents
from src.rag import get_embeddings, upload_documents


def create_pinecone_index_if_needed() -> None:
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    if pinecone.has_index(INDEX_NAME):
        print(f"Pinecone index already exists: {INDEX_NAME}")
        return

    pinecone.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )
    print(f"Created Pinecone index: {INDEX_NAME}")


def main() -> None:
    check_keys()

    documents = load_pdfs(PDF_FOLDER)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    create_pinecone_index_if_needed()
    upload_documents(chunks, embeddings)

    print("Done. You can now run: python app.py")


if __name__ == "__main__":
    main()
