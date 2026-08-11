"""
Retriever Module

Retrieves the most relevant document chunks
from the FAISS vector database.
"""


class Retriever:
    """
    Handles semantic retrieval using a FAISS vector store.
    """

    def __init__(self, vector_store):

        if vector_store is None:
            raise ValueError(
                "Vector store cannot be None."
            )

        self.vector_store = vector_store

    def retrieve(self, query, k=5):
    """
    Retrieve the most relevant document chunks.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    results = self.vector_store.similarity_search_with_score(
        query=query,
        k=k
    )

    # Keep only reasonably relevant results
    filtered_results = [
        doc
        for doc, score in results
        if score < 1.2
    ]

    return filtered_results

    def retrieve_with_scores(self, query, k=5):
        """
        Retrieve chunks along with similarity scores.

        Returns:
            list[(Document, score)]
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )