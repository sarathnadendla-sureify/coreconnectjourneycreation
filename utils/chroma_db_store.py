# ChromaDB integration for vector store
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import tempfile

class ChromaDBVectorStore:
    def __init__(self, embedding, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.embedding = embedding
        self.vector_store = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents, ids=None):
        # Chroma does not use explicit IDs in the same way as Pinecone
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def as_retriever(self, search_type="mmr", lambda_mult=0.5, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {}
        # Set lambda_mult for MMR if not already set
        if search_type == "mmr" and "lambda_mult" not in search_kwargs:
            search_kwargs["lambda_mult"] = lambda_mult
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
