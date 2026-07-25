import faiss
import numpy as np
import pandas as pd
import os


def save_vector_store(embeddings, documents_text, sources, store_dir="Vector_Store"):
    """Save embeddings and metadata to a FAISS index and CSV file."""
    embedding_dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(np.array(embeddings, dtype="float32"))

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    df = pd.DataFrame({
        "documents": documents_text,
        "source": sources
    })

    df.to_csv(f"{store_dir}/docs.csv", index=False)

    faiss.write_index(index, f"{store_dir}/vector_db.index")

    print("FAISS index created successfully")
