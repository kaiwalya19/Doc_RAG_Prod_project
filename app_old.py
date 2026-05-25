# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest import load_documents
from retriever import get_retriever

# 1. Page Configuration
st.set_page_config(page_title="PDF RAG Q&A Assistant", page_icon="📚", layout="wide")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Helper function to combine documents cleanly into text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Cached Initialization Function (prevents rebuilds on every click)
@st.cache_resource(show_spinner=False)
def initialize_rag_components():
    if not HF_TOKEN:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN missing from your .env file!")
        st.stop()
        
    try:
        # Step A: Load documents
        docs = load_documents("data/")
        
        # Step B: Build vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = get_retriever(vectorstore)
        
        # Step C: Setup HuggingFace Models
        base_llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.1,
            max_new_tokens=512,
            timeout=300
        )
        llm = ChatHuggingFace(llm=base_llm)
        
        # Step D: Construct Prompts
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using only the provided context. If you do not know, say you do not know."),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])
        
        # Step E: Construct Chains
        # We need a standalone retrieval step to show user references/sources
        retrieval_chain = retriever | format_docs
        
        generation_chain = (
            {
                "context": retriever | format_docs, 
                "input": RunnablePassthrough()
            }

            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        return generation_chain, retriever
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()

# 3. Sidebar UI Management
with st.sidebar:
    st.title("📂 Document Repository")
    st.write("Ensure your files are uploaded inside your project's `data/` folder.")
    
    # List files currently present inside data folder
    if os.path.exists("data"):
        files = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if files:
            st.success(f"Found {len(files)} target PDFs:")
            for f in files:
                st.caption(f"📄 {f}")
        else:
            st.warning("⚠️ No PDFs found in `data/` directory.")
    else:
        st.error("📂 Missing a `data/` directory in your root folder path.")

# 4. Main App UI Header
st.title("📚 PDF RAG Q&A Assistant")
st.write("Ask questions directly against your local document library using open-source models.")
st.divider()

# Initialize the pipeline silently behind a loader spinner
with st.spinner("🔄 Ingesting document vector layers and loading models..."):
    rag_chain, doc_retriever = initialize_rag_components()
st.toast("🚀 System successfully initialized!", icon="✅")

# 5. User Execution Workflow Layout
user_query = st.text_input(
    "💬 Ask something about your documents:", 
    placeholder="e.g., Summarize the main points of the operational report."
)

if user_query:
    # Set up columns to show answer next to references side by side
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💡 Answer")
        with st.spinner("🧠 Querying Llama-3.2-1B Endpoint..."):
            try:
                # Generate final answer from LLM
                answer = rag_chain.invoke(user_query)
                st.write(answer)
            except Exception as e:
                st.error(f"Generation failure: {str(e)}")
                
    with col2:
        st.subheader("🔍 Source Context")
        try:
            # Query retriever to fetch relevant document objects to print on screen
            matched_chunks = doc_retriever.invoke(user_query)
            for i, doc in enumerate(matched_chunks):
                with st.expander(f"Reference Chunk {i+1}", expanded=(i==0)):
                    st.write(doc.page_content)
                    if "source" in doc.metadata:
                        st.caption(f"📍 Source: {os.path.basename(doc.metadata['source'])}")
        except Exception as e:
            st.caption(f"Could not load reference chunks: {str(e)}")
