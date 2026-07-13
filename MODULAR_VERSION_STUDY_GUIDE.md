# Modular Version Study Guide

This project has two versions of the same idea:

- `simple_rag_one_file.py`: one-file version for memorizing the whole flow.
- Modular version: the same flow split across small files.

The modular version is easier to maintain because each file has one job.

## Structure

```text
app.py
  Starts the Flask API.
  Receives a user question.
  Sends the question to the RAG chain.
  Returns the answer.

ingest.py
  Loads PDFs.
  Splits them into chunks.
  Creates the Pinecone index if needed.
  Uploads chunks to Pinecone.

src/config.py
  Reads environment variables from .env.
  Stores API keys, model names, index name, and chunk settings.
  Provides check_keys().

src/documents.py
  Loads PDF files from the data folder.
  Splits PDF pages into text chunks.

src/rag.py
  Creates embeddings.
  Uploads chunks to Pinecone.
  Builds the retriever.
  Builds the prompt.
  Builds the OpenAI answer chain.
```

## Memorize The Flow

There are two commands:

```text
python ingest.py
python app.py
```

`ingest.py` prepares the knowledge base:

```text
check keys
load PDFs
split PDFs into chunks
create Pinecone index
upload chunks to Pinecone
```

`app.py` answers questions:

```text
build RAG chain
receive question from /get
retrieve matching chunks from Pinecone
send context + question to OpenAI
return answer
```

The shortest memory phrase is:

```text
PDFs -> chunks -> embeddings -> Pinecone -> retriever -> prompt -> OpenAI -> answer
```

## One-File Mental Version

This is not a replacement for the code. It is the modular version flattened into one study file so the pieces are easier to memorize.

```python
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# src/config.py

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DIMENSION = 384

PDF_FOLDER = os.getenv("PDF_DATA_DIR", "data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))


def check_keys():
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY in .env")

    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")


# src/documents.py

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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks")
    return chunks


# src/rag.py

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def upload_documents(documents, embeddings):
    PineconeVectorStore.from_documents(
        documents=documents,
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


def build_rag_chain():
    check_keys()

    embeddings = get_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

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


# ingest.py

def create_pinecone_index_if_needed():
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


def ingest_main():
    check_keys()

    documents = load_pdfs(PDF_FOLDER)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    create_pinecone_index_if_needed()
    upload_documents(chunks, embeddings)

    print("Done. You can now run: python app.py")


# app.py

app = Flask(__name__)
rag_chain = build_rag_chain()


@app.post("/get")
def chat():
    question = request.form.get("msg", "").strip()

    if not question:
        return "Please ask a question.", 400

    result = rag_chain.invoke({"input": question})
    return result["answer"]
```

## File-By-File Explanation

### `src/config.py`

This file reads `.env` once and stores the values in simple constants.

Important values:

```text
PINECONE_API_KEY     lets the app connect to Pinecone
OPENAI_API_KEY       lets the app call OpenAI
INDEX_NAME           tells Pinecone which index to use
EMBEDDING_MODEL      turns text into vectors
EMBEDDING_DIMENSION  must match the embedding model
PDF_FOLDER           tells ingestion where PDFs live
CHUNK_SIZE           controls chunk length
CHUNK_OVERLAP        repeats a little text between chunks
RETRIEVER_K          number of chunks retrieved per question
```

Memorize:

```text
config.py = environment settings + check_keys()
```

### `src/documents.py`

This file handles PDFs only.

`load_pdfs()`:

```text
find PDFs
load each PDF
tag each page with source filename
return all pages
```

`split_documents()`:

```text
take PDF pages
split them into smaller chunks
return chunks
```

Memorize:

```text
documents.py = PDFs -> chunks
```

### `src/rag.py`

This file handles the RAG logic.

`get_embeddings()` creates the embedding model.

```text
text -> vector
```

`upload_documents()` stores chunks in Pinecone.

```text
chunks + embeddings -> Pinecone index
```

`build_rag_chain()` builds the question-answering pipeline.

```text
embedding model
existing Pinecone index
retriever
prompt
OpenAI model
retrieval chain
```

Memorize:

```text
rag.py = embeddings + retriever + prompt + OpenAI chain
```

### `ingest.py`

This file is run before the chatbot.

It does not answer questions. It prepares Pinecone.

```text
check_keys()
load_pdfs()
split_documents()
get_embeddings()
create_pinecone_index_if_needed()
upload_documents()
```

Memorize:

```text
ingest.py = prepare the database
```

### `app.py`

This file starts the Flask app.

At startup:

```text
rag_chain = build_rag_chain()
```

When the user sends a question:

```text
question = request.form.get("msg", "").strip()
result = rag_chain.invoke({"input": question})
return result["answer"]
```

Memorize:

```text
app.py = receive question -> invoke chain -> return answer
```

## Final Memory Map

```text
config.py
  read .env

documents.py
  load PDFs
  split chunks

rag.py
  embed chunks
  connect Pinecone
  build retriever
  build prompt
  build OpenAI chain

ingest.py
  PDFs -> chunks -> Pinecone

app.py
  question -> Pinecone context -> OpenAI answer
```

The two most important ideas:

```text
ingest.py stores knowledge.
app.py uses stored knowledge to answer.
```
