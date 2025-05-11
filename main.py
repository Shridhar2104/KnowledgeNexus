import os
import argparse
import json
from typing import Dict, Any

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

def initialize_rag_pipeline() -> RAGPipeline:
    """
    Initialize the RAG pipeline with configuration from config.py.
    
    Returns:
        Initialized RAGPipeline object
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set it in an .env file or as an environment variable.")
    
    return RAGPipeline(
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        vector_store_dir=VECTOR_DB_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        retrieval_top_k=RETRIEVAL_TOP_K
    )

def ingest_command(rag_pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    """
    Handle the ingest command.
    
    Args:
        rag_pipeline: The RAG pipeline
        args: Command line arguments
    """
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.")
        return
    
    print(f"Ingesting documents from '{args.data_dir}'...")
    rag_pipeline.ingest_documents(args.data_dir)
    print("Ingestion complete.")

def query_command(rag_pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    """
    Handle the query command.
    
    Args:
        rag_pipeline: The RAG pipeline
        args: Command line arguments
    """
    if args.query:
        print(f"Processing query: '{args.query}'")
        result = rag_pipeline.query(args.query)
        
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['id']}: {source['metadata'].get('source', 'Unknown source')}")
    else:
        # Interactive mode
        print("Enter 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
            
            result = rag_pipeline.query(query)
            
            print("\nAnswer:")
            print(result["answer"])
            
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['id']}: {source['metadata'].get('source', 'Unknown source')}")

def info_command(rag_pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    """
    Handle the info command.
    
    Args:
        rag_pipeline: The RAG pipeline
        args: Command line arguments
    """
    info = rag_pipeline.get_system_info()
    print(json.dumps(info, indent=2))

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="KnowledgeRAG: A Retrieval-Augmented Generation System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--data-dir", required=True, help="Directory containing documents to ingest")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--query", help="Query to process (omit for interactive mode)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get system information")
    
    args = parser.parse_args()
    
    try:
        rag_pipeline = initialize_rag_pipeline()
        
        if args.command == "ingest":
            ingest_command(rag_pipeline, args)
        elif args.command == "query":
            query_command(rag_pipeline, args)
        elif args.command == "info":
            info_command(rag_pipeline, args)
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())