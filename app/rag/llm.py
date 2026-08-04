"""
LLM Module

Connects PaperLens AI with Groq using LangChain.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class LLM:

    def __init__(
        self,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    ):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found inside .env file."
            )

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )

    def generate_answer(self, question, documents):

        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI Research Assistant.

Use ONLY the provided context to answer.

If the answer is partially available, explain it using the available information.

If the answer is spread across multiple chunks, combine them into one complete answer.

Do NOT say "I couldn't find the information" unless the context is completely unrelated.

------------------------
Context:
{context}
------------------------

Question:
{question}

Answer:
"""
        )

        chain = prompt | self.llm

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return response.content