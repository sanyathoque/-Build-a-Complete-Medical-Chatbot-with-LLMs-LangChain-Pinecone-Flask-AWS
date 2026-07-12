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

    prompt = ChatPromptTemplate.from_template(
    """Answer the medical question using only the context below.
    If the answer is not in the context, say you do not know.
    Use no more than three sentences.
    
    Context:
    {context}
    
    Question:
    {input}
    """
    )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    answer_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, answer_chain)


def ask(question):
    rag_chain = build_rag_chain().invoke({"input": question})

    print(response["answer"])


def main():
    # sys.argv stores everything typed in the terminal as a Python list.
    #
    # Example:
    #
    # Terminal:
    #   python simple_rag_one_file.py
    #
    # sys.argv becomes:
    #   ["simple_rag_one_file.py"]
    #
    # The list contains only 1 item, so len(sys.argv) is 1.
    #
    # We need at least 2 items:
    #   1. The Python filename
    #   2. A command such as "ingest" or "ask"
    #
    # Therefore, len(sys.argv) < 2 means:
    # "The user did not provide a command."
    if len(sys.argv) < 2:

        # Show the user the correct ways to run the program.
        print("Usage:")
        print("  python simple_rag_one_file.py ingest")
        print('  python simple_rag_one_file.py ask "What is diabetes?"')

        # Stop the main() function here.
        # Without this return, the next line would try to access
        # sys.argv[1], which does not exist when no command was provided.
        return

    # sys.argv[0] is always the filename:
    #   "simple_rag_one_file.py"
    #
    # sys.argv[1] is the first value after the filename.
    #
    # Example:
    #
    # Terminal:
    #   python simple_rag_one_file.py ingest
    #
    # sys.argv becomes:
    #   ["simple_rag_one_file.py", "ingest"]
    #
    # Therefore:
    #   sys.argv[1] == "ingest"
    #
    # Another example:
    #
    # Terminal:
    #   python simple_rag_one_file.py ask "What is diabetes?"
    #
    # sys.argv becomes:
    #   [
    #       "simple_rag_one_file.py",
    #       "ask",
    #       "What is diabetes?"
    #   ]
    #
    # Therefore:
    #   sys.argv[1] == "ask"
    command = sys.argv[1]

    if command == "ingest":
        # Run the PDF ingestion process.
        ingest_pdfs()

    elif command == "ask":
        # sys.argv[2:] gets everything after the "ask" command.
        #
        # Example:
        #   ["What", "is", "diabetes?"]
        #
        # " ".join(...) combines those items into one string:
        #   "What is diabetes?"
        #
        # strip() removes unnecessary spaces from the beginning and end.
        question = " ".join(sys.argv[2:]).strip()

        # Check that the user actually entered a question.
        if not question:
            print("Please enter a question.")
            return

        # Send the question to the RAG system.
        ask(question)

    else:
        # This runs when the command is neither "ingest" nor "ask".
        #
        # Example:
        #   python simple_rag_one_file.py delete
        print("Unknown command. Use 'ingest' or 'ask'.")


# Python sets __name__ to "__main__" when this file is run directly.
#
# Example:
#   python simple_rag_one_file.py ingest
#
# This condition then calls main() and starts the program.
#
# If this file is imported into another Python file, main() will not
# automatically run.
if __name__ == "__main__":
    main()
