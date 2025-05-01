from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, pickle

def load_docs(path):
    docs = []
    for file in os.listdir(path):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    print("📚 Loading documents...")
    docs = load_docs("docs")
    chunks = split_docs(docs)

    print(f"🔎 {len(chunks)} chunks generated. Creating index with free HuggingFace model...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print("✅ Done. Index saved to vectorstore.pkl")
