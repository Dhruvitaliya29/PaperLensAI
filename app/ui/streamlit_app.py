import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import os
import shutil
import streamlit as st

from app.rag.rag_pipeline import RAGPipeline

UPLOAD_FOLDER = "data/uploads"
VECTOR_FOLDER = "data/vectorstore"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="PaperLens AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PaperLens AI")
st.write("AI Powered Research Assistant using RAG")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

pipeline = st.session_state.pipeline

uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Index Documents"):

    if uploaded_files:

        for file in uploaded_files:

            save_path = os.path.join(
                UPLOAD_FOLDER,
                file.name
            )

            with open(save_path, "wb") as f:
                shutil.copyfileobj(file, f)

        with st.spinner("Indexing documents..."):

            pipeline.index_documents(UPLOAD_FOLDER)

            pipeline.save_vector_store(VECTOR_FOLDER)

        st.success("Documents Indexed Successfully!")

question = st.text_input(
    "Ask a Question"
)

if st.button("Ask"):

    if question.strip():

        pipeline.load_vector_store(VECTOR_FOLDER)

        with st.spinner("Generating Answer..."):

            response = pipeline.ask(question, k=10)

        st.subheader("Answer")

        st.write(response["answer"])

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
                    f"📄 {source['source']} (Page {source['page']})"
                )