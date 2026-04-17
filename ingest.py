import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = "data/plants"
CHROMA_DIR = "chroma_db"

def ingest():
    all_docs = []
    pdf_files = list(Path(PDF_DIR).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        plant_name = pdf_path.stem.lower()
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["plant"] = plant_name
            doc.metadata["source_file"] = pdf_path.name
        all_docs.extend(docs)
        print(f"  Loaded {len(docs)} pages from {pdf_path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Done! Saved to {CHROMA_DIR}")

if __name__ == "__main__":
    ingest()