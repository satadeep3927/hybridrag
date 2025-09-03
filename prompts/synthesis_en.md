# Response Synthesis Prompt

You are tasked with synthesizing information from multiple sources to provide a comprehensive answer to the user's query.

## User Query:
{{ user_query }}

## System Context:
**Important**: The retrieved data consists of **document chunks**, not complete files. Each chunk contains:
- Text content (typically 1000 characters from a larger document)
- Source file name it originated from
- Metadata (creation date, file type, etc.)
- Multiple chunks may come from the same source file

## Retrieved Information:

{% if vector_results %}
### Semantic Search Results (Document Chunks):
{{ vector_results }}
{% endif %}

{% if analytical_results %}
### Analytical Query Results (Database Analysis):
{{ analytical_results }}
{% endif %}

## Synthesis Guidelines:

1. **Chunk Awareness**: Remember that results are chunks from larger documents, not complete files
2. **Integration**: Combine semantic chunk content and analytical insights to provide a complete picture  
3. **Accuracy**: Base your response strictly on the retrieved chunk information
4. **Context**: When multiple chunks relate to the same topic, synthesize them coherently
5. **Source Attribution**: Reference file names when available, noting these are source documents for the chunks
6. **Limitations**: Acknowledge when chunk information may be incomplete or represents partial content

## Response Structure:

### Direct Answer:
Provide a direct, concise answer to the user's question based on the chunk content.

---

### Supporting Details:
Include relevant details and context from the retrieved chunks. If multiple chunks cover the same topic, integrate them smoothly.

---

### Sources:
List the source files mentioned in the chunks. Note: "Based on chunks from: [filename1], [filename2]..."

---

### Additional Insights:
If applicable, provide additional insights that emerge from combining chunk content with analytical results.

## Important Notes:
- **Chunk Limitation**: The information comes from document chunks, so you may not have complete context from entire files
- **No Assumptions**: Do not make assumptions beyond what the chunk data shows
- **Conflicting Info**: If chunks contain conflicting information, acknowledge and explain the discrepancy
- **Missing Context**: If chunks seem incomplete or reference content not in the results, mention this limitation
- **File vs Chunks**: Distinguish between "files" (source documents) and "chunks" (retrieved content pieces) in your response
- Maintain a helpful and informative tone throughout your response
