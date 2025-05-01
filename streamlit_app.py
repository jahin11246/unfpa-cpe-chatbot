import streamlit as st
import pickle
import os
import requests
from pathlib import Path
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ---- App Config ----
st.set_page_config(page_title="UNFPA Evaluation Chatbot", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 UNFPA Evaluation Chatbot</h1>", unsafe_allow_html=True)
st.write("Ask questions based on uploaded UNFPA CPE evaluation documents.")

# ---- Vectorstore: Auto-download if missing ----
VSTORE_URL = "https://huggingface.co/jahin2025/unfpa-cpe-vectorstore/resolve/main/vectorstore.pkl"
LOCAL_FILE = "vectorstore.pkl"

if not Path(LOCAL_FILE).exists():
    st.info("📥 Downloading vectorstore from Hugging Face...")
    try:
        r = requests.get(VSTORE_URL)
        r.raise_for_status()
        with open(LOCAL_FILE, "wb") as f:
            f.write(r.content)
        st.success("✅ Vectorstore downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download vectorstore: {e}")
        st.stop()

# ---- Load vectorstore ----
try:
    with open(LOCAL_FILE, "rb") as f:
        vectorstore = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load vectorstore. Please ensure the file is valid.")
    st.stop()

# ---- Load Hugging Face LLM ----
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# ---- QA Chain ----
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# ---- Input ----
query = st.text_input("🔍 Your question:")

if query:
    try:
        answer = qa.run(query)
        if not answer.strip() or "i don't know" in answer.lower():
            st.warning("🤖 I do not know the answer to that. Please check the Final Evaluation Handbook.")
        else:
            st.success(answer)
    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
