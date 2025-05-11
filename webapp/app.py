import streamlit as st
import os
import sys
import tempfile

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from config import (
    EMBEDDING_MODEL, 
    LLM_MODEL, 
    OPENAI_API_KEY, 
    VECTOR_DB_PATH, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    RETRIEVAL_TOP_K
)

# Page configuration
st.set_page_config(
    page_title="KnowledgeNexus", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("KnowledgeNexus")
st.markdown("*Your personal knowledge base powered by RAG technology*")

# Initialize the RAG pipeline
@st.cache_resource
def initialize_rag_pipeline():
    try:
        return RAGPipeline(
            embedding_model_name=EMBEDDING_MODEL,
            llm_model_name=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            vector_store_dir=VECTOR_DB_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            retrieval_top_k=RETRIEVAL_TOP_K
        )
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None

rag_pipeline = initialize_rag_pipeline()

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Document upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or text files", 
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    process_btn = st.button("Process Documents", type="primary")
    
    if process_btn and uploaded_files:
        with st.spinner("Processing documents..."):
            # Create temp directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temp directory
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                
                # Process the documents
                try:
                    rag_pipeline.ingest_documents(temp_dir)
                    st.success(f"Successfully processed {len(uploaded_files)} documents")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # System information section
    st.subheader("System Information")
    if st.button("Show System Info"):
        if rag_pipeline:
            system_info = rag_pipeline.get_system_info()
            st.json(system_info)
        else:
            st.warning("RAG pipeline not initialized")

# Main content area
st.header("Ask Questions About Your Documents")

# Query input
query = st.text_input("Enter your question:")
num_results = st.slider("Number of sources to return", min_value=1, max_value=10, value=4)

# Process query
if st.button("Submit Question", type="primary") and query:
    if rag_pipeline:
        with st.spinner("Generating answer..."):
            try:
                # Override the default number of results if changed
                result = rag_pipeline.query(query)
                
                # Display the answer
                st.subheader("Answer")
                st.markdown(result["answer"])
                
                # Display the sources
                st.subheader("Sources")
                for i, source in enumerate(result["sources"]):
                    with st.expander(f"Source {i+1}: {source['metadata'].get('source', 'Unknown')}"):
                        st.markdown(f"**Content:**\n{source['content']}")
                        st.markdown("**Metadata:**")
                        st.json(source['metadata'])
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please initialize the RAG pipeline first")

# Footer
st.markdown("---")
st.markdown("KnowledgeNexus - A Retrieval-Augmented Generation System")