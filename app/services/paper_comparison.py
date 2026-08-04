"""
Paper Comparison Service
"""


class PaperComparison:

    def compare(self, retrieved_documents):

        comparison = []

        for index, document in enumerate(retrieved_documents, start=1):

            comparison.append({

                "Paper": index,

                "Characters": len(document.page_content),

                "Preview": document.page_content[:300]

            })

        return comparison