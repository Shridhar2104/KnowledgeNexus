from typing import List, Dict, Any, Optional, Tuple
from langchain.docstore.document import Document

from src.document_loader import DocumentProcessor
from src.embeddings import EmbeddingEngine
from src.retriever import VectorStore
from src.llm import LLMProcessor

class RAGPipeline:
    """
    The main RAG pipeline class that coordinates document processing, embedding,
    retrieval, and response generation.
    """
    
    def __init__(
        self,
        embedding_model_name: str,
        llm_model_name: str,
        openai_api_key: str,
        vector_store_dir: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_top_k: int = 4
    ):
        """
        Initialize the RAG pipeline with specified components.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM model
            openai_api_key: OpenAI API key
            vector_store_dir: Directory for vector store persistence
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            retrieval_top_k: Number of documents to retrieve
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.openai_api_key = openai_api_key
        self.vector_store_dir = vector_store_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_top_k = retrieval_top_k
        
        # Initialize components
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedding_engine = EmbeddingEngine(embedding_model_name, openai_api_key)
        self.vector_store = VectorStore(
            self.embedding_engine.get_model(),
            vector_store_dir
        )
        self.llm_processor = LLMProcessor(llm_model_name, openai_api_key)
        
        # Load vector store if it exists
        self.vector_store.load()
    
    def ingest_documents(self, directory_path: str) -> None:
        """
        Process and index documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
        """
        # Process documents
        documents = self.doc_processor.process_documents(directory_path)
        
        # Create or update vector store
        if self.vector_store.vector_db is None:
            print("Creating a new vector store...")
            self.vector_store.create_from_documents(documents)
        else:
            try:
                self.vector_store.add_documents(documents)
            except Exception as e:
                print(f"Error adding to existing vector store: {e}")
                print("Creating a new vector store instead...")
                self.vector_store.create_from_documents(documents)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: The query to process
            
        Returns:
            Dictionary containing the query, answer, and source documents
        """
        if self.vector_store.vector_db is None:
            return {
                "query": question,
                "answer": "No documents have been indexed yet. Please ingest documents first.",
                "sources": []
            }
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(
            question, 
            k=self.retrieval_top_k
        )
        
        # Generate response
        answer = self.llm_processor.generate_response(question, retrieved_docs)
        
        # Format source information
        sources = []
        for i, doc in enumerate(retrieved_docs):
            sources.append({
                "id": f"doc{i+1}",
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "query": question,
            "answer": answer,
            "sources": sources
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG pipeline components.
        
        Returns:
            Dictionary containing system information
        """
        # Get vector store stats if available
        vector_stats = {}
        if self.vector_store.vector_db is not None:
            try:
                vector_stats = self.vector_store.get_collection_stats()
            except Exception as e:
                vector_stats = {"error": str(e)}
        
        return {
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store": {
                "directory": self.vector_store_dir,
                "status": "initialized" if self.vector_store.vector_db is not None else "not initialized",
                "stats": vector_stats
            }
        }