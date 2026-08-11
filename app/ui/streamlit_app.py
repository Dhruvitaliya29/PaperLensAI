import sys
from pathlib import Path

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


import os
import shutil
import streamlit as st

from app.rag.rag_pipeline import RAGPipeline


# ---------------------------------------------------------
# Folders
# ---------------------------------------------------------

UPLOAD_FOLDER = "data/uploads"
VECTOR_FOLDER = "data/vectorstore"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="PaperLens AI",
    page_icon="📚",
    layout="wide"
)


st.title("📚 PaperLens AI")
st.write("AI Powered Research Assistant using RAG")


# ---------------------------------------------------------
# Initialize Pipeline
# ---------------------------------------------------------

if "pipeline" not in st.session_state:

    st.session_state.pipeline = RAGPipeline()

    # Load previously saved FAISS index
    index_path = os.path.join(
        VECTOR_FOLDER,
        "index.faiss"
    )

    if os.path.exists(index_path):

        try:

            st.session_state.pipeline.load_vector_store(
                VECTOR_FOLDER
            )

        except Exception as e:

            st.warning(
                f"Could not load existing vector store: {e}"
            )


pipeline = st.session_state.pipeline


# ---------------------------------------------------------
# Upload Research Papers
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------------------------------
# Index Documents
# ---------------------------------------------------------

if st.button("Index Documents"):

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF."
        )

    else:

        # -------------------------------------------------
        # Remove OLD uploaded PDFs
        # -------------------------------------------------

        for filename in os.listdir(UPLOAD_FOLDER):

            file_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            if os.path.isfile(file_path):

                os.remove(file_path)

            elif os.path.isdir(file_path):

                shutil.rmtree(file_path)


        # -------------------------------------------------
        # Remove OLD FAISS vector store
        # -------------------------------------------------

        for filename in os.listdir(VECTOR_FOLDER):

            file_path = os.path.join(
                VECTOR_FOLDER,
                filename
            )

            if os.path.isfile(file_path):

                os.remove(file_path)

            elif os.path.isdir(file_path):

                shutil.rmtree(file_path)


        # -------------------------------------------------
        # Save NEW uploaded PDFs
        # -------------------------------------------------

        for file in uploaded_files:

            save_path = os.path.join(
                UPLOAD_FOLDER,
                file.name
            )

            with open(save_path, "wb") as f:

                shutil.copyfileobj(
                    file,
                    f
                )


        # -------------------------------------------------
        # Create NEW FAISS index
        # -------------------------------------------------

        with st.spinner(
            "Indexing documents..."
        ):

            result = pipeline.index_documents(
                UPLOAD_FOLDER
            )

            pipeline.save_vector_store(
                VECTOR_FOLDER
            )


        st.success(
            f"Documents Indexed Successfully! "
            f"Loaded {result['documents']} documents "
            f"and created {result['chunks']} chunks."
        )


# ---------------------------------------------------------
# Ask Question
# ---------------------------------------------------------

question = st.text_input(
    "Ask a Question"
)


if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Generating Answer..."
            ):

                # Retrieve top 5 relevant chunks
                response = pipeline.ask(
                    question,
                    k=5
                )


            # -------------------------------------------------
            # Answer
            # -------------------------------------------------

            st.subheader("Answer")

            st.write(
                response["answer"]
            )


            # -------------------------------------------------
            # Sources
            # -------------------------------------------------

            st.subheader("Sources")

            shown = set()

            for source in response["sources"]:

                key = (
                    source["source"],
                    source["page"]
                )

                if key not in shown:

                    shown.add(key)

                    st.write(
                        f"📄 {source['source']} "
                        f"(Page {source['page']})"
                    )


        except Exception as e:

            st.error(
                f"Error while generating answer: {e}"
            )