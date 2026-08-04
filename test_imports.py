packages = [
    ("langchain", "langchain"),
    ("langchain_community", "langchain_community"),
    ("langchain_text_splitters", "langchain_text_splitters"),
    ("langchain_huggingface", "langchain_huggingface"),
    ("sentence_transformers", "sentence_transformers"),
    ("faiss", "faiss"),
    ("fitz", "fitz"),
    ("docx", "docx"),
    ("streamlit", "streamlit"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("dotenv", "dotenv"),
]

for name, module in packages:
    print(f"Checking {name}...")
    try:
        __import__(module)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}")
        print(e)