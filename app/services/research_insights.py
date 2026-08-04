"""
Research Insight Service
"""


class ResearchInsights:

    def generate(self, retrieved_documents):

        insights = []

        for document in retrieved_documents:

            text = document.page_content

            insights.append({

                "Length": len(text),

                "Words": len(text.split()),

                "Preview": text[:250]

            })

        return insights