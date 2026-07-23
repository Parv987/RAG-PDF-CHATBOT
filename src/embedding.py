from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"

sentence_transformer_model = SentenceTransformer(model_name)

embeddings = []
documents_text = []
sources = []

for document in docs:

    document_embedding = sentence_transformer_model.encode(
        document.page_content
    )

    embeddings.append(document_embedding)

    documents_text.append(document.page_content)

    sources.append("www.rheadata.com")