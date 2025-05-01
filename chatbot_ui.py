import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import os
from dotenv import load_dotenv

# Load environment variables (make sure .env exists with your HF token)
load_dotenv()

st.set_page_config(page_title="UNFPA Evaluation Chatbot", page_icon="🤖", layout="centered")

st.markdown("<h1 style='text-align: center;'>🤖 UNFPA Evaluation Chatbot</h1>", unsafe_allow_html=True)
st.write("Ask questions based on uploaded UNFPA CPE evaluation documents.")

# Load vector store (FAISS index built from your PDFs)
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# Load LLM (Hugging Face model - free tier)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Input UI
query = st.text_input("🔍 Your question:")

if query:
    try:
        answer = qa.run(query)
        if not answer.strip() or "i don't know" in answer.lower():
            st.warning("🤖 I do not know the answer to that. Please check the Final Evaluation Handbook.")
        else:
            st.success(answer)
    except Exception as e:
        st.error("⚠️ Error occurred. Try rephrasing your question or refer to the Final Evaluation Handbook.")
