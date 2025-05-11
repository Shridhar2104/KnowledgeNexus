# KnowledgeRAG

> A powerful Retrieval-Augmented Generation system for augmenting LLMs with your own knowledge base

KnowledgeRAG enables you to build an AI-powered question-answering system that leverages your own documents, databases, and knowledge sources. By combining the power of Large Language Models with efficient vector search, it delivers accurate, contextual responses grounded in your data.

## 🌟 Features

- **Document Processing Pipeline**: Ingest and process documents from multiple formats (PDF, TXT, HTML, DOCX)
- **Semantic Chunking**: Intelligently split documents to preserve context and meaning
- **Vector Embedding**: Transform text into high-dimensional vector representations for semantic search
- **Hybrid Search**: Combine vector similarity and keyword-based search for optimal retrieval
- **Enhanced Prompting**: Leverage advanced prompt engineering for better LLM responses
- **Source Attribution**: Provide citations to original documents for factual verification
- **Evaluation Framework**: Built-in metrics to measure retrieval and generation quality

## 🏗️ Architecture

KnowledgeRAG implements the Retrieval-Augmented Generation pattern with the following components:

1. **Ingestion Pipeline**: Processes and indexes your knowledge base
2. **Vector Database**: Stores document embeddings for semantic search
3. **Retrieval Engine**: Finds the most relevant information based on queries
4. **Generation Module**: Creates coherent, accurate responses using an LLM

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Document   │     │   Vector    │     │  Retrieval  │     │    LLM      │
│  Processing ├────►│  Database   ├────►│   Engine    ├────►│  Generation  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                               ▲                   │
                                               │                   │
                                               ▼                   ▼
                                         ┌─────────────────────────────┐
                                         │         User Query          │
                                         └─────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/knowledgerag.git
cd knowledgerag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Quick Usage

```python
from knowledgerag import RAGSystem

# Initialize the system
rag = RAGSystem()

# Index your documents
rag.ingest_documents("path/to/your/documents")

# Ask questions
response = rag.query("What is the refund policy for international orders?")
print(response)
```

## 📋 Usage Examples

### Basic Query

```python
response = rag.query("What are the main features of our premium plan?")
print(response.answer)  # Get the generated answer
print(response.sources)  # Get the source documents used
```

### Custom Configuration

```python
from knowledgerag import RAGSystem, RAGConfig

config = RAGConfig(
    chunk_size=500,
    chunk_overlap=50,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
    retrieval_top_k=4
)

rag = RAGSystem(config)
rag.ingest_documents("path/to/your/documents")
```

### Web UI

```bash
# Start the web interface
python -m knowledgerag.webapp --port 8000
```

Then open your browser to http://localhost:8000

## 📁 Project Structure

```
knowledgerag/
├── data/                 # Storage for raw documents
├── db/                   # Vector database files
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document processing
│   ├── embeddings.py       # Vector embedding functions
│   ├── retriever.py        # Document retrieval
│   ├── llm.py              # LLM integration
│   ├── rag_pipeline.py     # Main pipeline
│   └── evaluation.py       # System evaluation
├── tests/                # Unit and integration tests
├── webapp/              # Web interface
├── examples/            # Example scripts
├── config.py            # Configuration
├── main.py              # CLI entry point
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Size of document chunks in tokens | 500 |
| `chunk_overlap` | Overlap between chunks in tokens | 50 |
| `embedding_model` | Model for creating embeddings | all-MiniLM-L6-v2 |
| `llm_model` | LLM for generation | gpt-3.5-turbo |
| `retrieval_top_k` | Number of documents to retrieve | 4 |
| `vector_db_path` | Path to vector database | ./db |

## 📊 Performance Considerations

- **Memory Usage**: Vector databases can consume significant RAM for large document collections. Consider:
  - Using disk-based storage for production
  - Sharding large collections
  - Implementing filtering to reduce search space

- **Latency**: Several factors impact response time:
  - Embedding computation
  - Vector search efficiency
  - LLM generation speed
  - Network latency for API calls

- **Scaling**: For larger deployments, consider:
  - Distributed vector databases
  - Caching frequent queries
  - Batch processing documents
  - Using quantized models for local deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [RAG Architecture Patterns](https://docs.llamaindex.ai/en/stable/optimizing/rag_architecture_patterns/)

## 🙏 Acknowledgments

- This project builds on research from [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Inspired by LangChain, LlamaIndex, and other open-source RAG implementations
