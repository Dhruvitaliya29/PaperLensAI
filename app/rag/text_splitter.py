"""
Text Splitter Module

Splits LangChain Documents into smaller chunks
while preserving metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """
    Splits documents into chunks for RAG.
    """

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
    ):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(self, documents):
        """
        Split LangChain documents into chunks.
        """

        return self.text_splitter.split_documents(documents)

    def split_text(self, text):
        """
        Split raw text.
        """

        return self.text_splitter.split_text(text)