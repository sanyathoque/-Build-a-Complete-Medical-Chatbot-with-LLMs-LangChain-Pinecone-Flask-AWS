import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PDF_FOLDER = os.getenv("PDF_DATA_DIR", "data")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def check_keys():
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY in .env")

    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")


def load_pdfs(pdf_folder):
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


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks")
    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_pinecone_index_if_needed():
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    if pinecone.has_index(INDEX_NAME):
        print(f"Pinecone index already exists: {INDEX_NAME}")
        return

    pinecone.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index: {INDEX_NAME}")


def ingest_pdfs():
    check_keys()

    documents = load_pdfs(PDF_FOLDER)
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    create_pinecone_index_if_needed()

    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    print("Finished uploading PDF chunks to Pinecone")


def build_rag_chain():
    check_keys()

    embeddings = get_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a careful medical question-answering assistant.
Use only the context to answer.
If the answer is not in the context, say you do not know.
Keep the answer under three sentences.

Context:
{context}
""",
            ),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    answer_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, answer_chain)


def ask(question):
    rag_chain = build_rag_chain()
    response = rag_chain.invoke({"input": question})

    print(response["answer"])


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_rag_one_file.py ingest")
        print('  python simple_rag_one_file.py ask "What is diabetes?"')
        return

    command = sys.argv[1]

    if command == "ingest":
        ingest_pdfs()
        return

    if command == "ask":
        question = " ".join(sys.argv[2:]).strip()

        if not question:
            raise RuntimeError("Please provide a question after 'ask'")

        ask(question)
        return

    raise RuntimeError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
