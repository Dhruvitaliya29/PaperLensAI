"""
LLM Module

Connects PaperLens AI with Groq using LangChain.
Supports:
- Local development using .env
- Streamlit Cloud using Streamlit Secrets
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load variables from .env for local development
load_dotenv()


class LLM:

    def __init__(
        self,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    ):

        # --------------------------------------------------
        # Get Groq API Key
        # --------------------------------------------------

        # First try Streamlit Cloud Secrets
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

        # If not found, try local .env
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        # --------------------------------------------------
        # Validate API Key
        # --------------------------------------------------

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add GROQ_API_KEY to Streamlit Secrets "
                "or your local .env file."
            )

        # --------------------------------------------------
        # Initialize Groq LLM
        # --------------------------------------------------

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )

    # ------------------------------------------------------
    # Generate Answer
    # ------------------------------------------------------

    def generate_answer(self, question, documents):

        # Combine retrieved documents into context
        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI Research Assistant.

Use ONLY the provided context to answer the question.

If the answer is partially available, explain it using
the available information.

If the answer is spread across multiple chunks, combine
them into one complete answer.

Do NOT say "I couldn't find the information" unless the
context is completely unrelated.

---

Context:
{context}

---

Question:
{question}

Answer:
"""
        )

        # --------------------------------------------------
        # Create LangChain Chain
        # --------------------------------------------------

        chain = prompt | self.llm

        # --------------------------------------------------
        # Generate Response
        # --------------------------------------------------

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return response.content