import faiss
import numpy as np
import pandas as pd
import os

embedding_dimension = len(embeddings[0])

index = faiss.IndexFlatL2(embedding_dimension)

index.add(np.array(embeddings, dtype="float32"))

if not os.path.exists("Vector_Store"):
    os.makedirs("Vector_Store")

df = pd.DataFrame({
    "documents": documents_text,
    "source": sources
})

df.to_csv("Vector_Store/docs.csv", index=False)

faiss.write_index(
    index,
    "Vector_Store/vector_db.index"
)

print("FAISS index created")