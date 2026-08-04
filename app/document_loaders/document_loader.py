"""
Document Loader Module

Loads supported documents using LangChain document loaders.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)


class DocumentLoader:
    """
    Loads PDF, DOCX, TXT and Markdown documents.
    """

    def __init__(self):

        self.supported_loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }

    def find_documents(self, folder_path):
        """
        Returns all supported files.
        """

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder '{folder}' does not exist."
            )

        files = []

        for file in folder.iterdir():

            if (
                file.is_file()
                and file.suffix.lower() in self.supported_loaders
            ):
                files.append(file)

        return sorted(files)

    def load_document(self, file_path):
        """
        Load one document.
        """

        extension = Path(file_path).suffix.lower()

        if extension not in self.supported_loaders:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        loader = self.supported_loaders[extension](str(file_path))

        return loader.load()

    def load_documents(self, folder_path):
        """
        Load every document inside a folder.
        """

        documents = []

        files = self.find_documents(folder_path)

        for file in files:
            docs = self.load_document(file)
            documents.extend(docs)

        return documents