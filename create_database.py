from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
from pathlib import Path

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "Missing GOOGLE_API_KEY in environment."

PDF_FOLDER = "data"
CHROMA_PATH = "chroma"

def main():
    documents = load_all_pdfs(PDF_FOLDER)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_all_pdfs(folder_path: str):
    all_documents = []
    for pdf_file in Path(folder_path).rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        all_documents.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_file.name}")
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
