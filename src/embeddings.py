from typing import List, Dict, Any, Union
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

class EmbeddingEngine:
    """
    A class to handle different embedding models for text vectorization.
    Supports both local models (Hugging Face) and OpenAI models.
    """
    
    def __init__(self, model_name: str, openai_api_key: str = None):
        """
        Initialize the embedding engine with the specified model.
        
        Args:
            model_name: Name of the embedding model to use
            openai_api_key: API key for OpenAI (only needed for OpenAI models)
        """
        self.model_name = model_name
        self.embedding_model = self._initialize_model(model_name, openai_api_key)
        
    def _initialize_model(self, model_name: str, openai_api_key: str = None):
        """
        Initialize the appropriate embedding model based on the model name.
        
        Args:
            model_name: Name of the embedding model
            openai_api_key: OpenAI API key (if needed)
            
        Returns:
            An initialized embedding model
        """
        # Check if it's an OpenAI model
        if model_name.startswith("text-embedding") or "openai" in model_name.lower():
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=openai_api_key
                )
            except ImportError:
                raise ImportError("langchain_openai package is required for OpenAI models")
        else:
            # Use HuggingFace model
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except ImportError:
                raise ImportError("langchain_huggingface package is required for Hugging Face models. Please install with: pip install langchain-huggingface")
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text string into an embedding vector.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        return self.embedding_model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple text strings into embedding vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embedding_model.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            The dimension of the embedding vectors
        """
        # Embed a sample text and check its dimension
        sample_embedding = self.embed_text("Sample text to determine embedding dimension")
        return len(sample_embedding)

    def get_model(self):
        """
        Get the underlying embedding model.
        
        Returns:
            The embedding model object
        """
        return self.embedding_model