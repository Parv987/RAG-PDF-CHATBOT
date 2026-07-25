"""RAG PDF Chatbot - Main entry point."""
from dotenv import load_dotenv
from src.pdf_loader import load_and_split_pdf
from src.embedding import create_embeddings
from src.vector_database import save_vector_store
from src.retrieval import search
from src.llm import generate_answer

load_dotenv()


def main():
    print("1. Loading and splitting PDF...")
    docs = load_and_split_pdf()

    print("2. Creating embeddings...")
    embeddings, documents_text, sources, model = create_embeddings(docs)

    print("3. Saving vector store...")
    save_vector_store(embeddings, documents_text, sources)

    print("4. Ready! Enter your query below.")
    query = input("Enter your query: ")

    print("5. Searching for relevant documents...")
    combined_similar_documents_content, similar_documents_sources = search(query, model)

    if combined_similar_documents_content is None:
        return

    print("6. Generating answer...")
    generate_answer(query, combined_similar_documents_content, similar_documents_sources)


if __name__ == "__main__":
    main()
