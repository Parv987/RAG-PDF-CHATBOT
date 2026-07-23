index = faiss.read_index(
    "Vector_Store/vector_db.index"
)

df = pd.read_csv(
    "Vector_Store/docs.csv"
)

query = input("Enter your query: ")

query_embedding = sentence_transformer_model.encode(
    query
).reshape(1, -1)

distances, indices = index.search(
    query_embedding,
    k=5
)

threshold = 1.7

if distances[0][0] > threshold:

    print("Please ask a relevant question.")

else:

    combined_similar_documents_content_list = []

    similar_documents_sources = []

    for i in indices[0]:

        similar_document_content = df.loc[
            i,
            "documents"
        ]

        combined_similar_documents_content_list.append(
            similar_document_content
        )

        similar_document_source = df.loc[
            i,
            "source"
        ]

        similar_documents_sources.append(
            similar_document_source
        )

    combined_similar_documents_content = " ".join(
        combined_similar_documents_content_list
    )