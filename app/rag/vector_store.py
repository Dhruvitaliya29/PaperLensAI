"""
Vector Store Module

Creates, saves, loads and manages the FAISS vector database.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.rag.embedding_model import EmbeddingModel


class VectorStore:
    """
    Handles all operations related to the FAISS vector database.
    """

    def __init__(self):

        self.embedding_model = EmbeddingModel().get_model()

        self.vector_store = None

    def create_vector_store(self, documents):
        """
        Create a FAISS vector store from LangChain Documents.
        """

        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding_model
        )

        return self.vector_store

    def save_vector_store(self, folder_path):
        """
        Save the vector database to disk.
        """

        if self.vector_store is None:
            raise ValueError("Vector store has not been created.")

        folder = Path(folder_path)

        folder.mkdir(parents=True, exist_ok=True)

        self.vector_store.save_local(str(folder))

    def load_vector_store(self, folder_path):
        """
        Load an existing FAISS vector database.
        """

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"Vector store not found: {folder}"
            )

        self.vector_store = FAISS.load_local(
            str(folder),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        return self.vector_store

    def similarity_search(self, query, k=5):
        """
        Search for the top-k most similar chunks.
        """

        if self.vector_store is None:
            raise ValueError(
                "Vector store is not loaded."
            )

        return self.vector_store.similarity_search(
            query,
            k=k
        )

    def get_vector_store(self):
        """
        Return the FAISS vector store.
        """

        return self.vector_store