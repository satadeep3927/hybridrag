"""
Document Processing Utilities for Hybrid RAG System

This module provides utilities for processing various document formats
and preparing them for vector storage.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from loguru import logger

from ..config.config_manager import get_config
from ..database.surrealdb_client import SurrealDBClient
from ..tools.vector_search import VectorSearchTool


@dataclass
class DocumentChunk:
    """Represents a chunk of a processed document."""

    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    file_name: str
    file_hash: str


class DocumentProcessor:
    """
    Document processor for the Hybrid RAG system.

    Handles document ingestion, chunking, embedding generation,
    and storage in SurrealDB.
    """

    def __init__(self):
        """Initialize the document processor."""
        self.config = get_config()
        self.doc_config = self.config.document_processing
        self.vector_tool = VectorSearchTool()
        self.db_client = SurrealDBClient()

        logger.info("Document processor initialized")

    async def process_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Process a single file and store it in the database.

        Args:
            file_path: Path to the file to process
            metadata: Optional additional metadata

        Returns:
            List of document IDs created
        """
        try:
            file_path = Path(file_path)

            # Validate file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not self._is_supported_format(file_path):
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.doc_config.max_file_size_mb:
                raise ValueError(
                    f"File too large: {file_size_mb:.1f}MB > {self.doc_config.max_file_size_mb}MB"
                )

            logger.info(f"Processing file: {file_path}")

            # Read and process file content
            content = await self._read_file(file_path)
            file_hash = self._calculate_file_hash(content)

            # Check if file already exists
            if await self._file_already_processed(file_path.name, file_hash):
                logger.info(f"File already processed: {file_path.name}")
                return []

            # Create chunks
            chunks = self._create_chunks(content, file_path.name, file_hash, metadata)

            # Generate unique document ID for this file
            document_id = f"doc_{file_hash}_{file_path.name}"
            
            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.vector_tool.batch_embed(texts)

            # Store in database
            document_ids = []
            await self.db_client.connect()

            try:
                for chunk, embedding in zip(chunks, embeddings):
                    doc_id = await self.db_client.insert_document(
                        content=chunk.content,
                        vector=embedding,
                        file_name=chunk.file_name,
                        file_path=str(file_path.absolute()),
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        chunk_size=len(chunk.content),
                        total_chunks=len(chunks),
                        file_type=file_path.suffix[1:] if file_path.suffix else "unknown",
                        file_size=int(file_path.stat().st_size),
                        content_type="text",
                        metadata={
                            **chunk.metadata,
                            "file_hash": chunk.file_hash,
                        },
                    )
                    document_ids.append(doc_id)

                logger.info(
                    f"Successfully processed {file_path.name}: {len(document_ids)} chunks created"
                )

            finally:
                await self.db_client.disconnect()

            return document_ids

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise

    async def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            metadata: Optional metadata for all files

        Returns:
            Dictionary mapping file paths to document IDs
        """
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        results = {}
        pattern = "**/*" if recursive else "*"

        # Find all supported files
        supported_files = []
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and self._is_supported_format(file_path):
                supported_files.append(file_path)

        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")

        # Process files
        for file_path in supported_files:
            try:
                doc_ids = await self.process_file(str(file_path), metadata)
                results[str(file_path)] = doc_ids
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[str(file_path)] = []

        return results

    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower().lstrip(".") in self.doc_config.supported_formats

    async def _read_file(self, file_path: Path) -> str:
        """Read file content based on format."""
        suffix = file_path.suffix.lower()

        if suffix in [".txt", ".md"]:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                return await f.read()

        elif suffix == ".pdf":
            # Would require PyPDF2 or similar
            # For now, raise not implemented
            raise NotImplementedError("PDF processing not yet implemented")

        elif suffix == ".docx":
            # Would require python-docx
            # For now, raise not implemented
            raise NotImplementedError("DOCX processing not yet implemented")

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _calculate_file_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of file content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _file_already_processed(self, file_name: str, file_hash: str) -> bool:
        """Check if file with same hash already exists."""
        try:
            await self.db_client.connect()

            query = "SELECT * FROM documents WHERE file_name = $file_name AND metadata.file_hash = $file_hash LIMIT 1"
            result = await self.db_client.analytical_query(
                query, {"file_name": file_name, "file_hash": file_hash}
            )

            await self.db_client.disconnect()
            return len(result) > 0

        except Exception as e:
            logger.warning(f"Could not check if file already processed: {e}")
            return False

    def _create_chunks(
        self,
        content: str,
        file_name: str,
        file_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Create chunks from document content."""
        chunks = []

        # Simple text splitting (could be enhanced with more sophisticated methods)
        chunk_size = self.doc_config.chunk_size
        chunk_overlap = self.doc_config.chunk_overlap

        # Split into chunks
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to break at sentence boundaries
            if end < len(content):
                # Look for sentence endings
                for i in range(end, max(start, end - 200), -1):
                    if content[i] in ".!?":
                        end = i + 1
                        break

            chunk_content = content[start:end].strip()

            if chunk_content:  # Only create non-empty chunks
                chunk_metadata = {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "chunk_size": len(chunk_content),
                    "start_pos": start,
                    "end_pos": end,
                    **(metadata or {}),
                }

                chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                        file_name=file_name,
                        file_hash=file_hash,
                    )
                )

                chunk_index += 1

            # Move start position with overlap
            start = max(start + 1, end - chunk_overlap)

            # Prevent infinite loop
            if start >= end:
                break

        logger.debug(f"Created {len(chunks)} chunks from {file_name}")
        return chunks

    async def remove_file(self, file_name: str) -> int:
        """
        Remove all documents for a specific file.

        Args:
            file_name: Name of the file to remove

        Returns:
            Number of documents removed
        """
        try:
            await self.db_client.connect()
            deleted_count = await self.db_client.delete_documents_by_file(file_name)
            await self.db_client.disconnect()

            logger.info(f"Removed {deleted_count} documents for file: {file_name}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to remove file {file_name}: {e}")
            raise

    async def get_file_info(self, file_name: str) -> Dict[str, Any]:
        """
        Get information about a processed file.

        Args:
            file_name: Name of the file

        Returns:
            Dictionary with file information
        """
        try:
            await self.db_client.connect()

            documents = await self.db_client.get_documents_by_file(file_name)

            if not documents:
                return {"file_name": file_name, "exists": False}

            # Calculate statistics
            total_chunks = len(documents)
            total_content_length = sum(len(doc.get("content", "")) for doc in documents)

            # Get unique metadata
            file_hash = documents[0].get("metadata", {}).get("file_hash")
            created_at = min(
                doc.get("created_at") for doc in documents if doc.get("created_at")
            )

            await self.db_client.disconnect()

            return {
                "file_name": file_name,
                "exists": True,
                "total_chunks": total_chunks,
                "total_content_length": total_content_length,
                "file_hash": file_hash,
                "created_at": created_at,
            }

        except Exception as e:
            logger.error(f"Failed to get file info for {file_name}: {e}")
            return {"file_name": file_name, "exists": False, "error": str(e)}

    async def list_processed_files(self) -> List[Dict[str, Any]]:
        """
        List all processed files with basic information.

        Returns:
            List of file information dictionaries
        """
        try:
            await self.db_client.connect()

            query = """
            SELECT file_name, 
                   count() AS chunk_count,
                   math::min(created_at) AS first_processed,
                   math::max(created_at) AS last_processed
            FROM documents 
            GROUP BY file_name 
            ORDER BY first_processed DESC
            """

            results = await self.db_client.analytical_query(query)
            await self.db_client.disconnect()

            return results

        except Exception as e:
            logger.error(f"Failed to list processed files: {e}")
            return []
