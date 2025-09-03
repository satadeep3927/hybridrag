"""
Main entry point for the Hybrid RAG System

This module provides the main interface for the Hybrid RAG system.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database.surrealdb_client import SurrealDBClient
from src.graph.hybrid_rag_graph import HybridRAGGraph
from src.utils.document_processor import DocumentProcessor


class HybridRAG:
    """
    Main interface for the Hybrid RAG System.

    This class provides a simple interface for using the hybrid RAG system
    with both semantic and analytical retrieval capabilities.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Hybrid RAG system.

        Args:
            config_path: Optional path to configuration file
        """
        self.rag_graph = HybridRAGGraph()
        self.document_processor = DocumentProcessor()
        self.db_client = SurrealDBClient()

    async def query(
        self, question: str, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question to the system.

        Args:
            question: The question to ask
            thread_id: Optional thread ID for conversation continuity

        Returns:
            Dictionary containing the response and metadata
        """
        return await self.rag_graph.process_query(question, thread_id)

    async def ingest_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Ingest a file into the system.

        Args:
            file_path: Path to the file to ingest
            metadata: Optional metadata to attach

        Returns:
            Number of document chunks created
        """
        doc_ids = await self.document_processor.process_file(file_path, metadata)
        return len(doc_ids)

    async def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Ingest all supported files in a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            metadata: Optional metadata to attach

        Returns:
            Dictionary mapping file paths to chunk counts
        """
        results = await self.document_processor.process_directory(
            directory_path, recursive, metadata
        )
        return {path: len(doc_ids) for path, doc_ids in results.items()}

    async def list_files(self) -> list:
        """List all processed files."""
        return await self.document_processor.list_processed_files()

    async def remove_file(self, file_name: str) -> int:
        """
        Remove a file and all its chunks.

        Args:
            file_name: Name of the file to remove

        Returns:
            Number of chunks removed
        """
        return await self.document_processor.remove_file(file_name)

    async def health_check(self) -> Dict[str, bool]:
        """Check system health."""
        return await self.rag_graph.health_check()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return await self.rag_graph.analytical_query_tool.get_document_statistics()


# Convenience functions for direct usage
async def ask_question(question: str, thread_id: Optional[str] = None) -> str:
    """
    Simple function to ask a question and get a response.

    Args:
        question: The question to ask
        thread_id: Optional thread ID

    Returns:
        The response text
    """
    rag = HybridRAG()
    result = await rag.query(question, thread_id)
    return result.get("response", "No response generated")


async def ingest_document(file_path: str) -> int:
    """
    Simple function to ingest a document.

    Args:
        file_path: Path to the document

    Returns:
        Number of chunks created
    """
    rag = HybridRAG()
    return await rag.ingest_file(file_path)


if __name__ == "__main__":
    # Simple demo if run directly
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid RAG System")
    parser.add_argument("--query", "-q", help="Ask a question")
    parser.add_argument("--ingest", "-i", help="Ingest a file or directory")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List processed files"
    )
    parser.add_argument("--health", action="store_true", help="Check system health")

    args = parser.parse_args()

    async def main():
        rag = HybridRAG()

        if args.query:
            print(f"Question: {args.query}")
            result = await rag.query(args.query)
            print(f"Answer: {result['response']}")

        elif args.ingest:
            print(f"Ingesting: {args.ingest}")
            if Path(args.ingest).is_file():
                count = await rag.ingest_file(args.ingest)
                print(f"Created {count} chunks")
            elif Path(args.ingest).is_dir():
                results = await rag.ingest_directory(args.ingest)
                total = sum(results.values())
                print(f"Processed {len(results)} files, created {total} chunks")
            else:
                print("Invalid path")

        elif args.list:
            files = await rag.list_files()
            print(f"Found {len(files)} processed files:")
            for file_info in files:
                print(
                    f"  - {file_info['file_name']}: {file_info['chunk_count']} chunks"
                )

        elif args.health:
            health = await rag.health_check()
            print("System Health:")
            for component, status in health.items():
                status_text = "✅" if status else "❌"
                print(f"  {component}: {status_text}")

        else:
            print("Use --help for available options")

    asyncio.run(main())
