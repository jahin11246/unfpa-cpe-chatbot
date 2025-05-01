import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
import pickle
import requests

# Constants
VECTORSTORE_URL = "https://huggingface.co/jahin2025/unfpa-cpe-vectorstore/resolve/main/vectorstore.pkl"
VECTORSTORE_PATH = "vectorstore.pkl"

# Streamlit config
st.set_page_config(page_title="UNFPA Evaluation Chatbot", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 UNFPA Evaluation Chatbot</h1>", unsafe_allow_html=True)
st.write("Ask questions based on uploaded UNFPA CPE evaluation documents.")

# Download vectorstore if missing
def download_vectorstore(url, path):
    try:
        with st.spinner("🔄 Downloading vectorstore from Hugging Face..."):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            st.success("✅ Vectorstore downloaded.")
    except Exception as e:
        st.error(f"❌ Failed to download vectorstore: {e}")
        st.stop()

if not os.path.exists(VECTORSTORE_PATH):
    download_vectorstore(VECTORSTORE_URL, VECTORSTORE_PATH)

# Load vectorstore
try:
    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load vectorstore. Details: {e}")
    st.stop()

# Load LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Query UI
query = st.text_input("🔍 Your question:")

if query:
    try:
        answer = qa.run(query)
        if not answer.strip() or "i don't know" in answer.lower():
            st.warning("🤖 I do not know the answer. Please check the Final Evaluation Handbook.")
        else:
            st.success(answer)
    except Exception as e:
        st.error(f"⚠️ Error while answering: {e}")
