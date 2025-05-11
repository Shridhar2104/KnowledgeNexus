import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

class VectorStore:
    """
    A class to handle vector database operations, including indexing, 
    persisting, and searching document embeddings.
    """
    
    def __init__(self, embedding_function: Embeddings, persist_dir: str):
        """
        Initialize the vector store with an embedding function and persistence directory.
        
        Args:
            embedding_function: The embedding function to use for vectorization
            persist_dir: Directory where the vector store will be persisted
        """
        self.embedding_function = embedding_function
        self.persist_dir = persist_dir
        self.vector_db = None
        
    def create_from_documents(self, documents: List[Document], persist: bool = True) -> None:
        """
        Create a vector store from a list of documents.
        
        Args:
            documents: List of documents to index
            persist: Whether to persist the vector store to disk
        """
        print(f"Creating vector store from {len(documents)} documents...")
        try:
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persist_dir if persist else None
            )
            
            if persist:
                print(f"Persisting vector store to {self.persist_dir}...")
                self.vector_db.persist()
                print("Vector store persisted successfully.")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise
        
    def load(self) -> bool:
        """
        Load a vector store from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(self.persist_dir):
            print(f"No vector store found at {self.persist_dir}")
            return False
        
        print(f"Loading vector store from {self.persist_dir}...")
        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_function
        )
        print("Vector store loaded successfully.")
        return True
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search based on a query.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_db:
            raise ValueError("Vector store not initialized")
        
        return self.vector_db.similarity_search(query, k=k)
    
    def similarity_search_with_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search based on a query, returning scores.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of tuples containing documents and their similarity scores
        """
        if not self.vector_db:
            raise ValueError("Vector store not initialized")
        
        return self.vector_db.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: List[Document], persist: bool = True) -> None:
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of documents to add
            persist: Whether to persist the vector store to disk
        """
        if not self.vector_db:
            raise ValueError("Vector store not initialized")
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_db.add_documents(documents)
        
        if persist:
            print("Persisting updated vector store...")
            self.vector_db.persist()
            print("Vector store persisted successfully.")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary containing statistics about the collection
        """
        if not self.vector_db:
            raise ValueError("Vector store not initialized")
        
        collection = self.vector_db._collection
        return {
            "count": collection.count(),
            "dimension": collection._embedding_function.get_model().dimension
        }