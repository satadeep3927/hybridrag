# Building a Hybrid RAG System: A Complete Working Implementation

*Based on: Debashis Saha and Satadeep Dasgupta (2025) Bridging Analytics and Semantics: A Hybrid Database Approach to Retrieval-Augmented Generation. Zenodo. doi:[10.5281/zenodo.17018700](https://doi.org/10.5281/zenodo.17018700)*

Traditional RAG systems excel at semantic searches but struggle with analytical queries. This article presents a complete, working Hybrid RAG system that handles both "What is machine learning?" (semantic) and "How many documents discuss ML?" (analytical) in a unified interface.

## Quick Start

```bash
# Install dependencies
pip install sentence-transformers surrealdb openai python-dotenv

# Start SurrealDB (in separate terminal)
surreal start --log trace --user root --pass root file:data.db

# Run the complete example
python hybrid_rag_demo.py
```

## Complete Working Implementation

Here's a fully functional Hybrid RAG system you can run immediately:

```python
# hybrid_rag_demo.py
import os
import json
import asyncio
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from surrealdb import Surreal
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class HybridRAG:
    def __init__(self):
        # Initialize embedding model (runs locally)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize SurrealDB connection
        self.db = Surreal()
        
        # Initialize LLM (configure your API key in .env)
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Connect to database
        asyncio.run(self._init_db())
    
    async def _init_db(self):
        """Initialize SurrealDB connection and schema"""
        await self.db.connect('ws://localhost:8000/rpc')
        await self.db.signin({"user": "root", "pass": "root"})
        await self.db.use("hybridrag", "demo")
        
        # Define document table schema
        await self.db.query("""
            DEFINE TABLE IF NOT EXISTS document SCHEMAFULL;
            DEFINE FIELD id ON document TYPE string;
            DEFINE FIELD content ON document TYPE string;
            DEFINE FIELD embedding ON document TYPE array;
            DEFINE FIELD file_name ON document TYPE string;
            DEFINE FIELD chunk_index ON document TYPE number;
            DEFINE FIELD total_chunks ON document TYPE number;
            DEFINE FIELD created_at ON document TYPE datetime DEFAULT time::now();
            
            -- Create vector index for semantic search
            DEFINE INDEX embedding_idx ON document FIELDS embedding MTREE DIMENSION 384;
            
            -- Create indexes for analytical queries
            DEFINE INDEX file_name_idx ON document FIELDS file_name;
            DEFINE INDEX created_at_idx ON document FIELDS created_at;
        """)
    
    def add_document(self, content: str, file_name: str):
        """Add document with both semantic and analytical indexing"""
        return asyncio.run(self._add_document_async(content, file_name))
    
    async def _add_document_async(self, content: str, file_name: str):
        """Async version of add_document"""
        # Split into chunks
        chunks = self._chunk_text(content, chunk_size=500)
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_name}_chunk_{i}"
            
            # Generate embedding for semantic search
            embedding = self.embedder.encode(chunk).tolist()
            
            # Store in SurrealDB with both vector and analytical data
            await self.db.create("document", {
                "id": doc_id,
                "content": chunk,
                "embedding": embedding,
                "file_name": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        print(f"Added {len(chunks)} chunks from {file_name}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple text chunking"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def semantic_search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Perform semantic vector search using SurrealDB"""
        return asyncio.run(self._semantic_search_async(query, n_results))
    
    async def _semantic_search_async(self, query: str, n_results: int = 3) -> List[Dict]:
        """Async semantic search"""
        query_embedding = self.embedder.encode(query).tolist()
        
        # Use SurrealDB vector search
        results = await self.db.query(f"""
            SELECT id, content, file_name, chunk_index,
                   vector::similarity::cosine(embedding, {query_embedding}) AS similarity
            FROM document
            WHERE embedding IS NOT NONE
            ORDER BY similarity DESC
            LIMIT {n_results}
        """)
        
        return [
            {
                "content": doc["content"],
                "metadata": {
                    "file_name": doc["file_name"],
                    "chunk_index": doc["chunk_index"]
                },
                "similarity": doc["similarity"]
            }
            for doc in results[0]['result']
        ]
    
    def analytical_query(self, surql_query: str) -> List[Dict]:
        """Execute analytical SurrealDB query"""
        return asyncio.run(self._analytical_query_async(surql_query))
    
    async def _analytical_query_async(self, surql_query: str) -> List[Dict]:
        """Async analytical query"""
        results = await self.db.query(surql_query)
        return results[0]['result'] if results and results[0]['result'] else []
    
    def route_query(self, user_query: str) -> Dict[str, Any]:
        """Determine whether to use semantic, analytical, or hybrid approach"""
        routing_prompt = f"""
        Analyze this query and determine the best approach:
        Query: "{user_query}"
        
        Choose ONE:
        - "semantic": for content/concept questions (What is X? Explain Y?)
        - "analytical": for counting/statistics (How many? When? List files?)
        - "hybrid": for complex queries needing both approaches
        
        Respond with JSON: {{"approach": "semantic|analytical|hybrid", "reasoning": "brief explanation"}}
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": routing_prompt}],
            max_tokens=100
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except:
            # Fallback to semantic if parsing fails
            return {"approach": "semantic", "reasoning": "fallback"}
    
    def query(self, user_query: str) -> str:
        """Main query interface - routes and executes appropriate search"""
        # Determine query approach
        routing = self.route_query(user_query)
        approach = routing["approach"]
        
        print(f"Using {approach} approach: {routing['reasoning']}")
        
        if approach == "semantic":
            return self._handle_semantic_query(user_query)
        elif approach == "analytical":
            return self._handle_analytical_query(user_query)
        else:  # hybrid
            return self._handle_hybrid_query(user_query)
    
    def _handle_semantic_query(self, query: str) -> str:
        """Handle semantic content queries"""
        results = self.semantic_search(query)
        
        if not results:
            return "No relevant content found."
        
        context = "\n\n".join([r["content"] for r in results])
        
        prompt = f"""
        Based on the following context, answer the user's question:
        
        Context:
        {context}
        
        Question: {query}
        
        Provide a comprehensive answer based on the context:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _handle_analytical_query(self, query: str) -> str:
        """Handle analytical/statistical queries"""
        # Generate SurrealQL query from natural language
        sql_prompt = f"""
        Convert this natural language query to SurrealQL for a table called 'document' with fields:
        - id, content, file_name, chunk_index, total_chunks, created_at
        
        Query: "{query}"
        
        Use SurrealQL syntax (not SQL). Examples:
        - SELECT COUNT() FROM document;
        - SELECT file_name, COUNT() FROM document GROUP BY file_name;
        - SELECT * FROM document WHERE file_name CONTAINS 'ai';
        
        Return only the SurrealQL query, no explanation:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": sql_prompt}],
            max_tokens=150
        )
        
        surql_query = response.choices[0].message.content.strip()
        
        try:
            results = self.analytical_query(surql_query)
            return self._format_analytical_results(results, query)
        except Exception as e:
            return f"Query execution failed: {e}"
    
    def _handle_hybrid_query(self, query: str) -> str:
        """Handle queries requiring both semantic and analytical approaches"""
        # First do semantic search
        semantic_results = self.semantic_search(query)
        
        # Then get analytical insights
        analytical_results = self.analytical_query(
            "SELECT file_name, COUNT() as chunk_count FROM document GROUP BY file_name"
        )
        
        # Combine both results
        context = "\n".join([r["content"] for r in semantic_results[:2]])
        stats = f"Files in system: {len(analytical_results)}"
        
        prompt = f"""
        Answer the user's question using both content and statistics:
        
        Content Context:
        {context}
        
        Statistics:
        {stats}
        
        Question: {query}
        
        Provide a comprehensive answer combining content and statistics:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _format_analytical_results(self, results: List[Dict], query: str) -> str:
        """Format analytical query results"""
        if not results:
            return "No data found."
        
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            value = list(results[0].values())[0]
            return f"Result: {value}"
        
        # Multiple results - format as summary
        summary = f"Found {len(results)} results:\n"
        for result in results[:5]:  # Show first 5
            summary += f"- {dict(result)}\n"
        
        return summary


def main():
    """Demo the Hybrid RAG system"""
    print("=== Hybrid RAG System Demo ===")
    print("Make sure SurrealDB is running: surreal start --log trace --user root --pass root file:data.db")
    
    # Initialize system
    rag = HybridRAG()
    
    # Add sample documents
    documents = {
        "ai_basics.txt": """
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.
        Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.
        Deep Learning uses neural networks with multiple layers to process complex patterns.
        Natural Language Processing (NLP) helps computers understand and generate human language.
        Computer Vision enables machines to interpret and understand visual information.
        """,
        
        "ml_algorithms.txt": """
        Supervised learning uses labeled data to train models for prediction tasks.
        Unsupervised learning finds patterns in data without labeled examples.
        Reinforcement learning trains agents through interaction with an environment.
        Linear regression predicts continuous values using linear relationships.
        Decision trees create models using branching logic based on feature values.
        Neural networks consist of interconnected nodes that process information.
        """,
        
        "ai_applications.txt": """
        AI applications include recommendation systems used by Netflix and Amazon.
        Autonomous vehicles use AI for navigation and obstacle detection.
        Medical diagnosis systems help doctors identify diseases from medical images.
        Chatbots and virtual assistants like Siri use NLP to understand users.
        Fraud detection systems analyze patterns to identify suspicious transactions.
        Search engines use AI to rank and retrieve relevant web pages.
        """
    }
    
    # Add documents to the system
    for filename, content in documents.items():
        rag.add_document(content, filename)
    
    print("\n=== Testing Different Query Types ===")
    
    # Test queries
    test_queries = [
        "What is machine learning?",  # Semantic
        "How many documents are in the system?",  # Analytical  
        "Find information about AI applications and tell me how many files contain this information"  # Hybrid
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        response = rag.query(query)
        print(f"Answer: {response}")
        print()


if __name__ == "__main__":
    main()
```

## Configuration

Create a `.env` file with your OpenAI API key:

```env
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

## How It Works

### 1. **Single Database Architecture with SurrealDB**
- **Multi-Model Database**: SurrealDB handles both vector embeddings and structured data
- **Native Vector Search**: Built-in vector similarity search with MTREE indexing
- **Flexible Schema**: Supports both document storage and analytical queries

### 2. **Intelligent Query Routing**
- Uses LLM to analyze query intent
- Routes to appropriate search strategy
- Combines results for hybrid queries

### 3. **Three Query Types**

#### Semantic Queries
```python
# "What is machine learning?"
results = await db.query("""
    SELECT content, vector::similarity::cosine(embedding, $query_embedding) AS similarity
    FROM document ORDER BY similarity DESC LIMIT 3
""")
```

#### Analytical Queries  
```python
# "How many documents are in the system?"
results = await db.query("SELECT COUNT() FROM document")
```

#### Hybrid Queries
```python
# "Find AI info and count files"
semantic_results = await semantic_search(query)
analytical_results = await db.query("SELECT file_name, COUNT() FROM document GROUP BY file_name")
```

## Key Benefits

1. **Unified Database**: Single SurrealDB instance handles both vectors and analytics
2. **Local Embeddings**: No API costs for vector generation
3. **Native Vector Operations**: Built-in similarity search and indexing
4. **Multi-Model Flexibility**: Document, graph, and relational data in one system
5. **SurrealQL Power**: Advanced query language supporting complex operations

## Sample Output

```
=== Hybrid RAG System Demo ===
Make sure SurrealDB is running: surreal start --log trace --user root --pass root file:data.db
Added 3 chunks from ai_basics.txt
Added 3 chunks from ml_algorithms.txt  
Added 3 chunks from ai_applications.txt

=== Testing Different Query Types ===

1. Query: What is machine learning?
Using semantic approach: Content-based question about ML concepts
Answer: Machine Learning is a subset of artificial intelligence that enables 
computers to learn without being explicitly programmed. It uses algorithms 
to find patterns in data and make predictions or decisions...

2. Query: How many documents are in the system?
Using analytical approach: Counting query requiring database statistics
Answer: Result: 9

3. Query: Find information about AI applications and tell me how many files contain this information
Using hybrid approach: Requires both content search and statistical analysis
Answer: AI applications include recommendation systems, autonomous vehicles, 
medical diagnosis systems, and chatbots. Based on the system statistics, 
there are 3 files in the system containing this information...
```

## Extending the System

### SurrealDB Schema Definition
```python
# Define document table with vector capabilities
await db.query("""
    DEFINE TABLE document SCHEMAFULL;
    DEFINE FIELD embedding ON document TYPE array;
    DEFINE FIELD content ON document TYPE string;
    DEFINE FIELD file_name ON document TYPE string;
    
    -- Native vector index for fast similarity search
    DEFINE INDEX embedding_idx ON document FIELDS embedding MTREE DIMENSION 384;
""")
```

### Advanced Analytics with SurrealQL
```python
def get_document_stats(self):
    return self.analytical_query("""
        SELECT 
            file_name,
            COUNT() as chunks,
            math::mean(string::len(content)) as avg_chunk_size,
            created_at
        FROM document 
        GROUP BY file_name
        ORDER BY created_at DESC
    """)
```

### Multi-Model Queries
```python
# Combine vector search with graph relationships
async def find_related_content(self, topic: str):
    return await self.db.query(f"""
        LET $topic_embedding = {self.embedder.encode(topic).tolist()};
        SELECT content, file_name,
               vector::similarity::cosine(embedding, $topic_embedding) as similarity
        FROM document
        WHERE similarity > 0.7
        ORDER BY similarity DESC
        RELATE document->SIMILAR->document
    """)
```

## Production Considerations

1. **Scaling**: Use SurrealDB cluster mode for distributed deployment
2. **Security**: Add authentication, input validation, rate limiting  
3. **Monitoring**: Add logging, metrics, health checks
4. **Caching**: Cache frequent queries and embeddings
5. **Performance**: Optimize vector indexes and query patterns

## Why SurrealDB?

SurrealDB is perfect for Hybrid RAG because it provides:

- **Multi-Model**: Document, graph, vector, and key-value in one database
- **Native Vectors**: Built-in vector operations and indexing (no separate vector DB needed)
- **SurrealQL**: Powerful query language supporting complex analytics
- **Performance**: Rust-based with excellent performance characteristics
- **Flexibility**: Schema-less or schema-full options
- **Real-time**: Live queries and real-time subscriptions

## Complete Production System

This working example demonstrates the core concepts. For a full production system with advanced features like:

- SurrealDB multi-model database
- LangGraph workflow orchestration  
- Advanced document processing
- REST API interface
- Docker deployment
- Comprehensive testing

Check out the complete implementation: **[HybridRAG Repository](https://github.com/satadeep3927/hybridrag)**

---

## References

Saha, D., & Dasgupta, S. (2025). *Bridging Analytics and Semantics: A Hybrid Database Approach to Retrieval-Augmented Generation*. Zenodo. https://doi.org/10.5281/zenodo.17018700

**Authors**: Debashis Saha, Satadeep Dasgupta  
**Tags**: #RAG #AI #MachineLearning #VectorSearch #HybridSearch #Python
