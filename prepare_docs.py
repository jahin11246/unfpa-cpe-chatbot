from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

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
    all_docs = load_docs("docs")
    chunks = split_docs(all_docs)
    print(f"Loaded and split into {len(chunks)} text chunks.")
