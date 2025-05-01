import pickle
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# Load the vector index
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# Basic offline-friendly LLM (or switch to Gemini UI for manual Q&A)
from langchain.llms import OpenAI
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")  # or switch later

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

print("\n🤖 Ask me anything about UNFPA evaluation documents.\nType 'exit' to quit.\n")

while True:
    query = input("🟡 You: ")
    if query.lower() in ['exit', 'quit']: break
    answer = qa.run(query)
    print("🔵 Chatbot:", answer, "\n")
