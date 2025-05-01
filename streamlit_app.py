import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import os
import requests

VECTORSTORE_URL = "https://huggingface.co/jahin2025/unfpa-cpe-vectorstore/resolve/main/vectorstore.pkl"
VECTORSTORE_PATH = "vectorstore.pkl"

# 1. Download vectorstore.pkl if not present
if not os.path.exists(VECTORSTORE_PATH):
    with st.spinner("🔄 Downloading vectorstore..."):
        r = requests.get(VECTORSTORE_URL)
        if r.status_code == 200:
            with open(VECTORSTORE_PATH, "wb") as f:
                f.write(r.content)
        else:
            st.error("❌ Failed to download vectorstore from Hugging Face.")
            st.stop()

# 2. Load vectorstore
try:
    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)
except Exception:
    st.error("❌ Failed to load vectorstore. Please ensure it is valid.")
    st.stop()

# 3. Setup HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# 4. Streamlit UI
st.set_page_config(page_title="UNFPA Evaluation Chatbot", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 UNFPA Evaluation Chatbot</h1>", unsafe_allow_html=True)
st.write("Ask questions based on uploaded UNFPA CPE evaluation documents.")

query = st.text_input("🔍 Your question:")

if query:
    try:
        answer = qa.run(query)
        if not answer.strip() or "i don't know" in answer.lower():
            st.warning("🤖 I do not know the answer to that. Please check the Final Evaluation Handbook.")
        else:
            st.success(answer)
    except Exception as e:
        st.error("⚠️ Error occurred. Try rephrasing your question.")
