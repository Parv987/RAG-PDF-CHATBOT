from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "data/Rhea_resume.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

latex_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

for docu in documents:
    docs = latex_splitter.create_documents([docu.page_content])