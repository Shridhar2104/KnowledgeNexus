import os
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

class DocumentProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on file extension."""
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext.lower() == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all documents from a directory."""
        documents = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    docs = self.load_document(file_path)
                    documents.extend(docs)
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self, directory_path: str) -> List[Document]:
        """Process all documents in a directory: load and split."""
        documents = self.load_directory(directory_path)
        print(f"Loaded {len(documents)} documents.")
        
        split_docs = self.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")
        
        return split_docs