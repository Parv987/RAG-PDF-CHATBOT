from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path="data/Parv_Pandey_Resume.pdf"):
    """Load a PDF and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    all_chunks = []
    for docu in documents:
        chunks = splitter.create_documents([docu.page_content])
        all_chunks.extend(chunks)

    return all_chunks


# =====================================================================
# GOAL: Load ALL PDFs from the data/ directory dynamically (supports 4-5 PDFs)
# Replace the entire file content with this when ready:
#
# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# def load_and_split_pdf(data_dir="data"):
#     """Load all PDFs from the data directory and split them into chunks."""
#     pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
#     if not pdf_files:
#         print(f"No PDF files found in '{data_dir}' directory.")
#         return []
#     print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     all_chunks = []
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(data_dir, pdf_file)
#         print(f"Loading: {pdf_file}")
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#         for docu in documents:
#             chunks = splitter.create_documents([docu.page_content])
#             for chunk in chunks:
#                 chunk.metadata["source"] = pdf_file
#                 all_chunks.append(chunk)
#     print(f"Loaded {len(pdf_files)} PDF(s), created {len(all_chunks)} chunks.")
#     return all_chunks
# =====================================================================
