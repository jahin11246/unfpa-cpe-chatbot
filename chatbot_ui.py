import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import os

# Set Streamlit UI configuration
st.set_page_config(page_title="UNFPA Evaluation Chatbot", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 UNFPA Evaluation Chatbot</h1>", unsafe_allow_html=True)
st.write("Ask questions based on uploaded UNFPA CPE evaluation documents.")

# Load vectorstore (pre-built FAISS index)
try:
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load vectorstore. Please ensure 'vectorstore.pkl' is uploaded.")
    st.stop()

# Load Hugging Face token securely from Streamlit Cloud Secrets
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    st.error("❌ Hugging Face API token missing. Please set it in Streamlit Cloud secrets.")
    st.stop()

# Initialize Hugging Face LLM
try:
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=token
    )
except Exception as e:
    st.error(f"❌ Failed to load Hugging Face model. Details: {e}")
    st.stop()

# Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Query input box
query = st.text_input("🔍 Your question:")

# Handle user query
if query:
    try:
        answer = qa.run(query)
        if not answer.strip() or "i don't know" in answer.lower():
            st.warning("🤖 I do not know the answer to that. Please check the Final Evaluation Handbook.")
        else:
            st.success(answer)
    except Exception as e:
        st.error("⚠️ Error occurred. Try rephrasing your question or refer to the Final Evaluation Handbook.")
