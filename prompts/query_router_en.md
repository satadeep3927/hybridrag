# Query Router Prompt

Analyze the user's query and determine the best retrieval strategy for our Hybrid RAG system.

## User Query:
{{ user_query }}

## System Architecture:

### Data Storage Structure:
The system stores document content as **chunks** (not full files) in a SurrealDB database with the following schema:

**documents** table:
- `id`: Unique document chunk identifier
- `content`: Text content of the chunk (typically 1000 characters with 200 character overlap)
- `vector`: Embedding vector for semantic search (384 dimensions)
- `file_name`: Source filename the chunk originated from
- `metadata`: Additional metadata (author, creation_date, file_type, etc.)
- `created_at`: Timestamp when chunk was added to system
- `updated_at`: Last modification timestamp

**Important**: Each record represents a **chunk of a document**, not a complete file. Multiple chunks may exist from the same source file.

## Available Tools:

### 1. **vector_search**: Semantic similarity search
- **Use for**: Conceptual queries, content meaning, topic exploration
- **Searches**: Document chunk content using vector embeddings
- **Best for**: "What is...", "Explain...", "Find information about..."

### 2. **analytical_query**: Structured database queries  
- **Use for**: Quantitative analysis, metadata filtering, counting
- **Searches**: Database fields (file_name, created_at, metadata fields) 
- **Uses**: **SurrealDB syntax** (NOT standard SQL)
- **Best for**: "How many...", "When was...", "List files...", "Count chunks..."

**SurrealDB Query Examples**:
```surrealql
-- Count total chunks
SELECT count() FROM documents;

-- Get unique source files
SELECT file_name FROM documents GROUP BY file_name;

-- Find recent chunks
SELECT * FROM documents WHERE created_at > time::now() - 1d;

-- Count chunks by file type  
SELECT file_type, count() FROM documents GROUP BY file_type;
```

## Analysis Framework:

### Query Type Indicators:

**Semantic/Conceptual** (use vector_search):
- Keywords: "about", "related to", "similar", "means", "explains", "describes", "what is"
- Conceptual topics, themes, or subject matter
- Content-based searches within chunk text
- Similarity or relationship queries

**Analytical/Quantitative** (use analytical_query):
- Keywords: "how many", "count", "total", "list", "recent", "latest", "oldest", "when"
- Number-based questions about chunks or files
- Time-based filtering (created_at, updated_at)
- File-based analysis (file_name patterns)
- Metadata analysis

**Hybrid** (use both tools):
- Complex queries requiring both content understanding and quantitative analysis
- Example: "Find AI-related chunks and tell me how many files they come from"

## Your Task:
1. Identify the primary intent of the query
2. Consider whether the user wants content (vector_search) or metadata/counts (analytical_query)
3. Remember: data is stored as chunks, not complete files
4. Determine which tool(s) would be most effective
5. Formulate appropriate queries for the selected tool(s)

## Response Format:
**IMPORTANT**: Respond with JSON only, no markdown code blocks or extra formatting.

{
  "primary_tool": "vector_search|analytical_query|hybrid",
  "reasoning": "Brief explanation considering chunk-based storage",
  "semantic_query": "reformulated query for vector search (if applicable)",
  "analytical_operation": "SurrealDB query description (if applicable)", 
  "confidence": "high|medium|low"
}
