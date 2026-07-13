from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import (
    EMBEDDING_MODEL,
    INDEX_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RETRIEVER_K,
    check_keys,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def upload_documents(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> None:
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
