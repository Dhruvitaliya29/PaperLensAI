"""
RAG Pipeline

Coordinates the complete Retrieval-Augmented Generation workflow.
"""

from pathlib import Path

from app.document_loaders.document_loader import DocumentLoader
from app.rag.text_splitter import TextSplitter
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.rag.llm import LLM


class RAGPipeline:
    """
    Complete RAG Pipeline
    """

    def __init__(self):

        self.loader = DocumentLoader()

        self.splitter = TextSplitter()

        self.vector_store = VectorStore()

        self.retriever = None

        self.llm = LLM()

    def index_documents(self, upload_folder):
        """
        Load documents, split them,
        create embeddings and build FAISS.
        """

        upload_folder = Path(upload_folder)

        documents = self.loader.load_documents(upload_folder)

        print(f"\nLoaded {len(documents)} documents.")

        chunks = self.splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks.")

        vector_db = self.vector_store.create_vector_store(chunks)

        self.retriever = Retriever(vector_db)

        return {
            "documents": len(documents),
            "chunks": len(chunks)
        }

    def save_vector_store(self, save_path):

        self.vector_store.save_vector_store(save_path)

    def load_vector_store(self, save_path):

        vector_db = self.vector_store.load_vector_store(save_path)

        self.retriever = Retriever(vector_db)

    def ask(self, question, k=5):
        """
        Ask a question using indexed documents.
        """

        if self.retriever is None:
            raise ValueError(
                "Vector store has not been loaded."
            )

        retrieved_docs = self.retriever.retrieve(
            question,
            k
        )

        print("\n" + "=" * 80)
        print("RETRIEVED CHUNKS")
        print("=" * 80)

        for i, doc in enumerate(retrieved_docs, start=1):

            print(f"\nChunk {i}")

            print(f"Source : {doc.metadata.get('source')}")

            print(f"Page   : {doc.metadata.get('page')}")

            print("-" * 80)

            print(doc.page_content[:700])

            print("-" * 80)

        answer = self.llm.generate_answer(
            question,
            retrieved_docs
        )

        sources = []

        for doc in retrieved_docs:

            sources.append(
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
            )

        return {
            "answer": answer,
            "sources": sources
        }