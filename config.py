import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Document processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector database settings
VECTOR_DB_PATH = "./db"

# Retrieval settings
RETRIEVAL_TOP_K = 4

# LLM settings
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")