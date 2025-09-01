import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Function to create FAISS vector store (optimized)
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Use a smaller, faster embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)

# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vector_store = FAISS.load_local(path, embeddings,  
                 allow_dangerous_deserialization=True)
    return vector_store

# Build QA Chain with HuggingFace LLM
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Load HuggingFace model pipeline (free, local)
    model_name = "google/flan-t5-large" 
    hf_pipeline = pipeline("text2text-generation", model=model_name, device=-1)

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain

# Streamlit App
st.title("🚀 Advanced RAG Chatbot")
st.write("📄 Upload a PDF and ask questions using a free Hugging Face model.")

uploaded_file = st.file_uploader("📂 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)

    st.info("🔍 Creating FAISS vector store... ⏳😊")
    create_faiss_vector_store(text)

    st.info("🤖 Initializing Hugging Face QA model... ⚡😎")
    st.session_state.qa_chain = build_qa_chain()
    st.success("✅ Chatbot is ready! 🎉🤩")

if 'qa_chain' in st.session_state:
    question = st.text_input("💬 Ask a question about the uploaded PDF:")
    if question:
        st.info("📝 Querying the document... 🤔📚")
        answer = st.session_state.qa_chain.run(question)
        st.success(f"✨ **Answer:** {answer} 😃👍")
