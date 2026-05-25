# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest import load_documents
from retriever import get_retriever

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Helper function to combine documents cleanly into text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_pipeline(query):
    # Step 1: Load documents from data/ directory
    docs = load_documents("data/")
    print(f"✅ Loaded {len(docs)} document chunks")

    # Step 2: Build vector store and initialize your custom retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = get_retriever(vectorstore)

    # Step 3: Initialize base model and wrap it in the Chat interface
    # This forces LangChain to use conversational/chat payloads (solving the ValueError)
    base_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,
        max_new_tokens=256,
        timeout=300
    )
    llm = ChatHuggingFace(llm=base_llm)

    # Step 4: Define structural message layout for chat tasks
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "Answer the question using only the provided context. If you do not know, say you do not know."
        ),
        (
            "human", 
            "Context:\n{context}\n\nQuestion: {input}"
        )
    ])

    # Step 5: Pure LCEL Pipeline Construction
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "input": RunnablePassthrough()
        }

        | prompt 
        | llm 
        | StrOutputParser()
    )

    # Step 6: Execute the pipeline and get the text response
    return rag_chain.invoke(query)

if __name__ == "__main__":
    user_query = str(input("Enter your query related to document : "))
    result = rag_pipeline(user_query)
    print("\n💡 Answer:\n", result)
