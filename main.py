from app.rag.rag_pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline()

    pipeline.index_documents("data/uploads")

    pipeline.save_vector_store("data/vectorstore")

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        response = pipeline.ask(question , k=10)

        print("\nAnswer:")
        print(response["answer"])

        print("\nSources:")
        for source in response["sources"]:
            print(f"- {source['source']} (Page {source['page']})")


if __name__ == "__main__":
    main()