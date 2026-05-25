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
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure the data directory exists safely
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Cached Initialization Function
# We use a custom hash value for tracking cache invalidation manually
@st.cache_resource(show_spinner=False)
def initialize_rag_components(cache_breaker=0):
    if not HF_TOKEN:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN missing from your .env file!")
        st.stop()
        
    try:
        # Step A: Load documents
        docs = load_documents(f"{DATA_DIR}/")
        
        # Fallback check if the folder is completely empty
        if not docs:
            return None, None

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
        
        # Step E: Construct Generation Chain
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
        st.error(f"Initialization failure: {str(e)}")
        return None, None

# Initialize state to track file upload triggers dynamically
if "cache_version" not in st.session_state:
    st.session_state.cache_version = 0

# 3. Sidebar UI Management
with st.sidebar:
    st.title("📂 Document Repository")
    st.write("Current stored repository index:")
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if files:
        st.success(f"Found {len(files)} target PDFs:")
        for f in files:
            st.caption(f"📄 {f}")
    else:
        st.warning("⚠️ No PDFs found. Please upload documents.")

# 4. Main App Header Interface
st.title("📚 PDF RAG Assistant")
st.write("Manage your knowledge layers and run inference using secure open-source execution components.")
st.divider()

# Load RAG modules silently based on state versions
rag_chain, doc_retriever = initialize_rag_components(st.session_state.cache_version)

# 5. UI Layout Interface Tab Splitting
tab1, tab2 = st.tabs(["💬 Chat Q&A", "📤 Upload Documents"])

with tab1:
    if rag_chain is None:
        st.info("💡 Your document workspace is empty. Please navigate to the **Upload Documents** tab to begin.")
    else:
        user_query = st.text_input(
            "💬 Ask something about your documents:", 
            placeholder="e.g., Summarize the main points of the operational report."
        )

        if user_query:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💡 Answer")
                with st.spinner("🧠 Querying Llama-3.2-1B Endpoint..."):
                    try:
                        answer = rag_chain.invoke(user_query)
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Generation failure: {str(e)}")
                        
            with col2:
                st.subheader("🔍 Source Context")
                try:
                    matched_chunks = doc_retriever.invoke(user_query)
                    for i, doc in enumerate(matched_chunks):
                        with st.expander(f"Reference Chunk {i+1}", expanded=(i==0)):
                            st.write(doc.page_content)
                            if "source" in doc.metadata:
                                st.caption(f"📍 Source: {os.path.basename(doc.metadata['source'])}")
                except Exception as e:
                    st.caption(f"Could not load reference chunks: {str(e)}")

with tab2:
    st.subheader("📤 Document Upload Center")
    st.write("Upload your PDF files to automatically parse, chunk, and embed them into the FAISS database layers.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files:", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if st.button("🚀 Process & Ingest Files", type="primary"):
        if uploaded_files:
            saved_count = 0
            with st.spinner("💾 Writing files and updating vector embeddings..."):
                for uploaded_file in uploaded_files:
                    # Target path generation string block
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    
                    # Write bytes out onto local system storage disks
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
            
            # Increment core sequence layout string flag to break the previous cache instance
            st.session_state.cache_version += 1
            
            st.success(f"Successfully processed {saved_count} new file(s) into database layers!")
            st.rerun()
        else:
            st.warning("Please upload at least one PDF file before clicking process.")
