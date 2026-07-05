from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import settings
from src.prompts import SYSTEM_PROMPT


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vector_store_from_documents(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> PineconeVectorStore:
    return PineconeVectorStore.from_documents(
        documents=documents,
        index_name=settings.index_name,
        embedding=embeddings,
    )


def build_vector_store_from_existing_index(
    embeddings: HuggingFaceEmbeddings,
) -> PineconeVectorStore:
    return PineconeVectorStore.from_existing_index(
        index_name=settings.index_name,
        embedding=embeddings,
    )


def build_rag_chain():
    embeddings = build_embeddings()
    vector_store = build_vector_store_from_existing_index(embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    answer_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, answer_chain)
