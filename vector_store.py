# vector_store.py
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def build_vector_store(chunks):
    # Use a free Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
