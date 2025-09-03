"""
Example usage of the Hybrid RAG System

This file demonstrates the three main query types.
"""

import asyncio
from main import HybridRAG


async def main():
    """Simple example with three query types."""
    print("Hybrid RAG System - Usage Examples")
    print("=" * 50)

    rag = HybridRAG()

    # Three example queries
    queries = [
        "What is machine learning and how does it relate to AI?",
        "How many document chunks are stored in the system?", 
        "Find chunks about machine learning and tell me which files they come from"
    ]

    for i, question in enumerate(queries, 1):
        print(f"\\n{i}. Query: {question}")
        result = await rag.query(question)
        print(f"Answer: {result['response']}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
