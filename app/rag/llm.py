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


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


class LLM:
    """
    Handles communication with the Groq LLM.
    """

    def __init__(
        self,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    ):

        # -------------------------------------------------
        # Get API key from Streamlit Secrets
        # -------------------------------------------------

        try:
            api_key = st.secrets.get(
                "GROQ_API_KEY"
            )
        except Exception:
            api_key = None


        # -------------------------------------------------
        # Fallback to local .env
        # -------------------------------------------------

        if not api_key:

            api_key = os.getenv(
                "GROQ_API_KEY"
            )


        # -------------------------------------------------
        # Validate API key
        # -------------------------------------------------

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add it to Streamlit Secrets "
                "or your local .env file."
            )


        # -------------------------------------------------
        # Initialize Groq
        # -------------------------------------------------

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )


    # -----------------------------------------------------
    # Generate Answer
    # -----------------------------------------------------

    def generate_answer(
        self,
        question,
        documents
    ):
        """
        Generate a grounded answer using
        the retrieved document chunks.
        """

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context = "\n\n".join(
            [
                doc.page_content
                for doc in documents
            ]
        )


        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI research assistant analyzing uploaded research papers.

Answer the user's question using ONLY the provided context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. If the context contains information from multiple papers,
   keep each paper's claims clearly separated.
4. Only compare papers when the user explicitly asks for a comparison.
5. Do not attribute information from one paper to another.
6. If the context does not contain enough information, say:
   "The uploaded papers do not provide enough information to answer this."
7. For technical questions, preserve the exact meaning of the papers.
8. Never infer that two architectures are identical just because
   one paper is based on another.
9. When discussing an architecture, distinguish between:
   - encoder
   - decoder
   - self-attention
   - masked/causal self-attention
   - bidirectional attention

--------------------------------------------------

Context:

{context}

--------------------------------------------------

Question:

{question}

--------------------------------------------------

Answer:
"""
        )


        # -------------------------------------------------
        # Create LangChain chain
        # -------------------------------------------------

        chain = prompt | self.llm


        # -------------------------------------------------
        # Generate response
        # -------------------------------------------------

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )


        return response.content