# Hybrid RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines semantic vector search with analytical querying using SurrealDB, LangChain, and LangGraph.

## Features

- **Hybrid Retrieval**: Combines semantic similarity search with structured analytical queries
- **SurrealDB Integration**: Unified database for both vector embeddings and structured data
- **LangGraph Orchestration**: Intelligent workflow management for query routing and response synthesis
- **OpenAI Compatibility**: Supports OpenAI-compatible APIs for LLM and embeddings
- **Flexible Document Processing**: Supports multiple document formats with intelligent chunking
- **Prompt Management**: Template-based prompt system with Jinja2 support
- **CLI Interface**: Comprehensive command-line interface for system interaction

## Architecture

Based on the research paper "Bridging Analytics and Semantics: A Hybrid Database Approach to Retrieval-Augmented Generation", this system implements:

1. **Query Analysis**: Determines whether to use semantic search, analytical queries, or both
2. **Vector Search Tool**: Performs semantic similarity search using embeddings
3. **Analytical Query Tool**: Executes structured SurrealQL queries for precise data analysis
4. **Response Synthesis**: Combines results from both tools into coherent responses

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hybridrag
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your OpenAI API key and other settings
   ```

5. **Install and start SurrealDB**:
   ```bash
   # Install SurrealDB (see https://surrealdb.com/install)
   curl -sSf https://install.surrealdb.com | sh
   
   # Start SurrealDB
   surreal start --bind 0.0.0.0:8000 memory
   ```

## Configuration

Edit `config/config.yaml` to customize:

- **Database settings**: SurrealDB connection parameters
- **OpenAI settings**: API key, models, and parameters
- **Vector search**: Similarity thresholds and search parameters
- **Document processing**: Chunk sizes and supported formats
- **Prompts**: Language and template settings

## Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Ask a question
python cli.py query "What are the main topics in my documents?"

# Start interactive chat
python cli.py chat

# Ingest documents
python cli.py ingest /path/to/documents --recursive

# List processed files
python cli.py list-files

# Remove a file
python cli.py remove-file "document.pdf"

# Check system status
python cli.py status
```

### Python API

You can also use the system programmatically:

```python
import asyncio
from main import HybridRAG

async def example():
    rag = HybridRAG()
    
    # Ingest a document
    chunk_count = await rag.ingest_file("document.pdf")
    print(f"Created {chunk_count} chunks")
    
    # Ask questions
    result = await rag.query("What is the main theme of the document?")
    print(result["response"])
    
    # Analytical query
    result = await rag.query("How many documents were processed last week?")
    print(result["response"])

asyncio.run(example())
```

### Example Queries

**Semantic Queries** (uses vector search):
- "What does the document say about machine learning?"
- "Find information about climate change impacts"
- "What are the main themes discussed?"

**Analytical Queries** (uses structured queries):
- "How many documents were uploaded this month?"
- "Which files have the most content?"
- "Show me documents from the last 7 days"

**Hybrid Queries** (uses both approaches):
- "Find recent documents about artificial intelligence and count them"
- "What are the main AI topics discussed in documents from 2024?"

## Project Structure

```
hybridrag/
├── src/
│   ├── config/           # Configuration management
│   ├── database/         # SurrealDB client
│   ├── tools/           # Vector search and analytical query tools
│   ├── prompts/         # Prompt management
│   ├── graph/           # LangGraph workflow
│   └── utils/           # Document processing utilities
├── prompts/             # Prompt templates (Markdown files)
├── config/              # Configuration files
├── cli.py              # Command-line interface
├── main.py             # Main entry point
└── requirements.txt    # Python dependencies
```

## Key Components

### 1. SurrealDB Client (`src/database/surrealdb_client.py`)
- Handles connections to SurrealDB
- Manages document storage with vector embeddings
- Supports both vector similarity search and analytical queries

### 2. Vector Search Tool (`src/tools/vector_search.py`)
- Generates embeddings using OpenAI's embedding models
- Performs semantic similarity search
- Returns contextually relevant documents

### 3. Analytical Query Tool (`src/tools/analytical_query.py`)
- Executes structured SurrealQL queries
- Supports aggregations, filtering, and analytics
- Converts natural language to SurrealQL when possible

### 4. LangGraph Workflow (`src/graph/hybrid_rag_graph.py`)
- Orchestrates the entire RAG pipeline
- Routes queries to appropriate tools
- Synthesizes responses from multiple sources

### 5. Document Processor (`src/utils/document_processor.py`)
- Handles document ingestion and chunking
- Generates embeddings for chunks
- Stores processed documents in SurrealDB

### 6. Prompt Manager (`src/prompts/prompt_manager.py`)
- Manages prompt templates stored as Markdown files
- Supports Jinja2 templating and multi-language prompts
- Provides dynamic prompt loading and rendering

## Advanced Features

### Custom Prompts

Create custom prompt templates in the `prompts/` directory:

```markdown
# Custom Analysis Prompt

Analyze the following information and provide insights:

## Query: {{ user_query }}

## Context:
{{ context }}

Please provide a detailed analysis focusing on:
1. Key findings
2. Relationships between concepts
3. Actionable insights
```

### Metadata-based Filtering

Add metadata when ingesting documents:

```python
metadata = {
    "category": "research",
    "author": "John Doe",
    "publication_date": "2024-01-15"
}

await rag.ingest_file("research_paper.pdf", metadata)
```

### Health Monitoring

Monitor system health and performance:

```python
health = await rag.health_check()
statistics = await rag.get_statistics()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research: "Bridging Analytics and Semantics: A Hybrid Database Approach to Retrieval-Augmented Generation"
- Built with LangChain, LangGraph, and SurrealDB
- Inspired by the need for both semantic understanding and analytical precision in RAG systems
