from pinecone import Pinecone, ServerlessSpec

from src.config import settings
from src.documents import load_pdf_documents, split_documents
from src.rag import build_embeddings, build_vector_store_from_documents


def ensure_index_exists() -> None:
    pinecone = Pinecone(api_key=settings.pinecone_api_key)

    if pinecone.has_index(settings.index_name):
        return

    pinecone.create_index(
        name=settings.index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )


def main() -> None:
    print(f"Loading PDFs from: {settings.pdf_data_dir}")
    documents = load_pdf_documents(settings.pdf_data_dir)

    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks. Preparing Pinecone index...")
    ensure_index_exists()

    print(f"Uploading chunks to index: {settings.index_name}")
    embeddings = build_embeddings()
    build_vector_store_from_documents(chunks, embeddings)

    print("Done. You can now run: python app.py")


if __name__ == "__main__":
    main()
