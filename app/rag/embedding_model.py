"""
Embedding Model Module

Loads and manages the HuggingFace embedding model.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Singleton wrapper around HuggingFaceEmbeddings.

    The embedding model is loaded only once and reused
    throughout the application.
    """

    _embedding_model = None

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        if EmbeddingModel._embedding_model is None:

            print(f"Loading embedding model: {model_name}")

            EmbeddingModel._embedding_model = HuggingFaceEmbeddings(
                model_name=model_name
            )

            print("Embedding model loaded successfully.")

    def get_model(self):
        """
        Return the embedding model.
        """

        return EmbeddingModel._embedding_model