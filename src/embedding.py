from sentence_transformers import SentenceTransformer


def create_embeddings(docs):
    """Create embeddings for a list of document chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []
    documents_text = []
    sources = []

    for document in docs:
        document_embedding = model.encode(document.page_content)
        embeddings.append(document_embedding)
        documents_text.append(document.page_content)
        # HARDCODED - replace with metadata source when using multi-PDF loader
        sources.append("Parv_Pandey_Resume.pdf")

    return embeddings, documents_text, sources, model


# =====================================================================
# GOAL: Get the PDF filename dynamically instead of hardcoded URL
# When you switch to the multi-PDF loader in pdf_loader.py, replace
# the `sources.append("www.rheadata.com")` line inside the for loop with:
#
#     source = document.metadata.get("source", "unknown.pdf")
#     sources.append(source)
#
# This will read the PDF filename stored in chunk.metadata["source"]
# by the updated pdf_loader for each chunk.
# =====================================================================
