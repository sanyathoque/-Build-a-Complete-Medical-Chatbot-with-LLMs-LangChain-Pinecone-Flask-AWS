# Clean Medical Chatbot RAG

This is a study-friendly rewrite of the original Flask + LangChain + Pinecone medical chatbot.

The app has two steps:

1. `python ingest.py` reads PDFs from `data/`, splits them into chunks, embeds them, and stores them in Pinecone.
2. `python app.py` starts a Flask API that retrieves the best chunks and asks OpenAI to answer from that context.

## Project Map

```text
medical_chatbot_clean/
  app.py                 # Flask routes only
  ingest.py              # Build/update the Pinecone index
  setup.py               # Package metadata
  requirements.txt       # Python dependencies
  .env.example           # Required environment variables
  src/
    config.py            # Settings from environment variables
    documents.py         # Load PDFs and split text
    prompts.py           # System prompt
    rag.py               # Embeddings, vector store, and RAG chain
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then edit `.env` and add your API keys.

## Run

Put your PDFs inside `data/`, then build the vector index:

```bash
python ingest.py
```

Start the API:

```bash
python app.py
```

Ask a question:

```bash
curl -X POST -F "msg=What is diabetes?" http://localhost:8080/get
```

## Memory Model

Remember the flow as:

```text
PDFs -> chunks -> embeddings -> Pinecone -> retriever -> prompt -> OpenAI -> answer
```

## Notes

This project is for learning. Medical answers should include a disclaimer in real products, and users should be told to consult qualified medical professionals.
