# Code Assessment

## What the Original Code Does

The original project is a basic RAG chatbot:

1. `store_index.py` loads PDFs from `data/`.
2. `helper.py` splits PDF text into small chunks.
3. Hugging Face embeddings convert each chunk into a 384-dimensional vector.
4. Pinecone stores those vectors.
5. `app.py` retrieves the most similar chunks for each question.
6. OpenAI generates a short answer from the retrieved context.

## What Was Hard to Memorize

- Several files put many imports and statements on one line.
- `app.py` does too much at import time: loads keys, creates embeddings, connects to Pinecone, builds the retriever, builds the prompt, and defines routes.
- Environment variables are read but not validated clearly.
- The index name, model name, chunk size, region, and retrieval count are scattered as hard-coded values.
- The ingestion script and the web app share concepts but not a clear structure.
- Medical safety is only lightly handled by the prompt.

## Cleaner Structure Used Here

- `src/config.py`: one place for settings and required environment variables.
- `src/documents.py`: one place for loading PDFs and splitting chunks.
- `src/rag.py`: one place for the prompt, embeddings, Pinecone, retriever, and chain creation.
- `ingest.py`: only builds the Pinecone index.
- `app.py`: only exposes API routes and calls the RAG chain.

## Simple Version to Memorize

```text
ingest.py
  load PDFs
  split text
  create Pinecone index
  upload embedded chunks

app.py
  receive question
  retrieve relevant chunks
  ask OpenAI with context
  return answer
```

## Important Production Improvements Later

- Add authentication if deployed publicly.
- Add rate limiting to protect API costs.
- Add source citations in answers.
- Add logging without printing sensitive data.
- Add a stronger medical disclaimer and safety policy.
- Add tests for document loading and chain setup.
