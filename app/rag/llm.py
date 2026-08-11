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
                "Add GROQ_API_KEY to Streamlit Secrets "
                "or your local .env file."
            )

        # -------------------------------------------------
        # Initialize Groq LLM
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
        # Build context with source information
        # -------------------------------------------------

        context_parts = []

        for doc in documents:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            context_parts.append(
                f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI research assistant analyzing
uploaded research papers.

Your job is to answer the user's question using
ONLY the provided context.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not invent facts that are unsupported
   by the provided context.

3. You MAY synthesize information from multiple
   retrieved chunks when those chunks belong to
   the same research paper.

4. If the question asks for a paper's
   "main contribution", infer the contribution
   from the paper's described:
   - motivation
   - objective
   - proposed method
   - methodology
   - results

   You may combine these pieces when they
   collectively support the conclusion.

5. Clearly distinguish between:
   - information explicitly stated in the paper
   - conclusions reasonably synthesized from
     the provided context

6. If the user asks to compare multiple papers,
   keep each claim associated with the correct paper.

7. Do not attribute information from one paper
   to another paper.

8. If the retrieved context genuinely does not
   contain enough information, say:

   "The provided context does not contain enough
   information to answer this reliably."

9. Do not refuse to answer merely because the
   exact phrase "main contribution" is not present.
   Use the relevant information from the retrieved
   chunks to produce a grounded synthesis.

10. Prefer precise technical explanations over
    generic summaries.

11. When possible, mention the paper title,
    authors, method, and contribution only when
    those details are supported by the context.

--------------------------------------------------

PROVIDED CONTEXT:

{context}

--------------------------------------------------

QUESTION:

{question}

--------------------------------------------------

ANSWER:
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