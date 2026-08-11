"""
Embedding Model Module

Loads and manages the HuggingFace embedding model.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Wrapper around HuggingFaceEmbeddings.

    Uses a stronger embedding model for semantic
    retrieval of research papers.
    """

    _embedding_model = None

    def __init__(
        self,
        model_name="BAAI/bge-base-en-v1.5"
    ):

        if EmbeddingModel._embedding_model is None:

            print(
                f"Loading embedding model: {model_name}"
            )

            EmbeddingModel._embedding_model = (
                HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={
                        "normalize_embeddings": True
                    }
                )
            )

            print(
                "Embedding model loaded successfully."
            )

    def get_model(self):
        """
        Return the embedding model.
        """

        return EmbeddingModel._embedding_model