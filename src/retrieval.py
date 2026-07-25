import faiss
import pandas as pd


def search(query, model, store_dir="Vector_Store", k=5, threshold=1.7):
    """Search the vector store for similar documents given a query."""
    index = faiss.read_index(f"{store_dir}/vector_db.index")
    df = pd.read_csv(f"{store_dir}/docs.csv")

    query_embedding = model.encode(query).reshape(1, -1)

    distances, indices = index.search(query_embedding, k)

    if distances[0][0] > threshold:
        print("Please ask a relevant question.")
        return None, None

    combined_similar_documents_content_list = []
    similar_documents_sources = []

    for i in indices[0]:
        similar_document_content = df.loc[i, "documents"]
        combined_similar_documents_content_list.append(similar_document_content)

        similar_document_source = df.loc[i, "source"]
        similar_documents_sources.append(similar_document_source)

    combined_similar_documents_content = " ".join(combined_similar_documents_content_list)

    return combined_similar_documents_content, similar_documents_sources
