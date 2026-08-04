"""
Literature Review Service
"""


class LiteratureReview:

    def generate(self, retrieved_documents):

        review = ""

        for index, document in enumerate(retrieved_documents, start=1):

            review += f"\nPaper {index}\n"
            review += "-" * 40 + "\n"
            review += document.page_content
            review += "\n\n"

        return review