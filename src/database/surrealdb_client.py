"""
SurrealDB Client for Hybrid RAG System

This module provides a client interface for SurrealDB operations including
vector search and analytical queries.
"""

import traceback
from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb import AsyncWsSurrealConnection as Surreal
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.config_manager import get_config


class SurrealDBClient:
    """
    SurrealDB client for hybrid RAG operations.

    Supports both vector similarity search and analytical SurrealQL queries.
    """

    def __init__(self):
        """Initialize the SurrealDB client."""
        self.config = get_config()
        self.db_config = self.config.database
        self.db: Optional[Surreal] = None
        self._initialized = False

    async def connect(self) -> None:
        """Connect to SurrealDB."""
        try:
            url = f"ws://{self.db_config.host}:{self.db_config.port}"
            self.db = Surreal(url)

            await self.db.signin(
                {"username": self.db_config.username,
                    "password": self.db_config.password}
            )
            await self.db.use(self.db_config.namespace, self.db_config.database)

            # Initialize schema if needed
            await self._initialize_schema()

            self._initialized = True
            logger.info(f"Connected to SurrealDB at {url}")

        except Exception as e:
            logger.error(f"Failed to connect to SurrealDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from SurrealDB."""
        if self.db:
            await self.db.close()
            self._initialized = False
            logger.info("Disconnected from SurrealDB")

    async def _initialize_schema(self) -> None:
        """Initialize the database schema for hybrid RAG."""
        try:
            # Create documents table with vector field
            schema_queries = [
                """
                DEFINE TABLE documents SCHEMAFULL;
                """,
                """
                DEFINE FIELD content ON documents TYPE string;
                """,
                """
                DEFINE FIELD vector ON documents TYPE array<float>;
                """,
                """
                DEFINE FIELD file_name ON documents TYPE string;
                """,
                """
                DEFINE FIELD file_path ON documents TYPE string;
                """,
                """
                DEFINE FIELD file_type ON documents TYPE string;
                """,
                """
                DEFINE FIELD file_size ON documents TYPE int;
                """,
                """
                DEFINE FIELD document_id ON documents TYPE string;
                """,
                """
                DEFINE FIELD chunk_index ON documents TYPE int;
                """,
                """
                DEFINE FIELD chunk_size ON documents TYPE int;
                """,
                """
                DEFINE FIELD total_chunks ON documents TYPE int;
                """,
                """
                DEFINE FIELD content_type ON documents TYPE string DEFAULT 'text';
                """,
                """
                DEFINE FIELD metadata ON documents TYPE object;
                """,
                """
                DEFINE FIELD created_at ON documents TYPE datetime DEFAULT time::now();
                """,
                """
                DEFINE FIELD updated_at ON documents TYPE datetime DEFAULT time::now();
                """,
                f"""
                DEFINE INDEX idx_documents_vector ON documents FIELDS vector MTREE DIMENSION {self._get_embedding_dimensions()};
                """,
                """
                DEFINE INDEX idx_documents_file_name ON documents FIELDS file_name;
                """,
                """
                DEFINE INDEX idx_documents_document_id ON documents FIELDS document_id;
                """,
                """
                DEFINE INDEX idx_documents_file_type ON documents FIELDS file_type;
                """,
                """
                DEFINE INDEX idx_documents_chunk_index ON documents FIELDS chunk_index;
                """,
            ]

            for query in schema_queries:
                try:
                    await self.db.query(query)
                except Exception as e:
                    # Ignore errors for already existing schema elements
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema initialization warning: {e}")

            logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def _ensure_connected(self) -> None:
        """Ensure the client is connected."""
        if not self._initialized or not self.db:
            raise RuntimeError(
                "SurrealDB client not initialized. Call connect() first."
            )

    def _get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions from config."""
        # Try different config paths for backward compatibility
        if hasattr(self.config, 'embeddings') and hasattr(self.config.embeddings, 'huggingface'):
            return self.config.embeddings.huggingface.dimensions
        elif hasattr(self.config, 'openai') and hasattr(self.config.openai, 'embedding_dimensions'):
            return self.config.openai.embedding_dimensions
        else:
            return 384  # Default fallback

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def insert_document(
        self,
        content: str,
        vector: List[float],
        file_name: str,
        file_path: str,
        document_id: str,
        chunk_index: int,
        chunk_size: int,
        total_chunks: int,
        file_type: str = "txt",
        file_size: Optional[int] = None,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a document chunk with its vector embedding.

        Args:
            content: The text content of the chunk
            vector: The vector embedding of the content
            file_name: The source file name
            file_path: The full file path
            document_id: Unique identifier for the source document
            chunk_index: Index of this chunk (0-based)
            chunk_size: Size of this chunk in characters
            total_chunks: Total number of chunks for this document
            file_type: Type of the source file (txt, pdf, docx, etc.)
            file_size: Size of the source file in bytes
            content_type: Type of content (text, code, table, etc.)
            metadata: Additional metadata

        Returns:
            The ID of the inserted document chunk
        """
        self._ensure_connected()

        try:
            document_data = {
                "content": content,
                "vector": vector,
                "file_name": file_name,
                "file_path": file_path,
                "file_type": file_type,
                "file_size": file_size,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "chunk_size": chunk_size,
                "total_chunks": total_chunks,
                "content_type": content_type,
                "metadata": metadata or {},
            }

            result = await self.db.create("documents", document_data)

            if result and len(result) > 0:
                doc_id = result["id"]
                logger.debug(f"Inserted chunk {chunk_index}/{total_chunks} for document {document_id} with ID: {doc_id}")
                return doc_id
            else:
                raise RuntimeError("Failed to insert document")

        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def vector_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        file_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_vector: The query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            file_filter: Optional file name filter

        Returns:
            List of similar documents with similarity scores
        """
        self._ensure_connected()

        try:
            # Build the vector search query
            base_query = """
            SELECT *, vector::similarity::cosine(vector, $query_vector) AS similarity 
            FROM documents 
            WHERE vector::similarity::cosine(vector, $query_vector) > $threshold
            """

            if file_filter:
                base_query += " AND file_name CONTAINS $file_filter"

            base_query += " ORDER BY similarity DESC LIMIT $limit"

            params = {
                "query_vector": query_vector,
                "threshold": similarity_threshold,
                "limit": limit,
            }

            if file_filter:
                params["file_filter"] = file_filter

            result = await self.db.query(base_query, params)

            if result and len(result) > 0:
                documents = result[0]["result"] if "result" in result[0] else result[0]
                logger.debug(
                    f"Vector search returned {len(documents)} results")
                return documents
            else:
                return []

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def analytical_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an analytical SurrealQL query.

        Args:
            query: The SurrealQL query string
            params: Optional query parameters

        Returns:
            Query results
        """
        self._ensure_connected()

        try:
            result = await self.db.query(query, params or {})

            if result and len(result) > 0:
                # Extract the actual results
                if isinstance(result[0], dict) and "result" in result[0]:
                    return result[0]["result"]
                else:
                    return result[0] if isinstance(result[0], list) else [result[0]]
            else:
                return []

        except Exception as e:
            logger.error(f"Analytical query failed: {e}")
            raise

    async def get_document_count(self) -> int:
        """Get the total number of documents."""
        try:
            result = await self.analytical_query(
                "SELECT count() FROM documents GROUP ALL"
            )
            if result and len(result) > 0:
                return result[0].get("count", 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def get_documents_by_file(self, file_name: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific file."""
        try:
            query = "SELECT * FROM documents WHERE file_name = $file_name ORDER BY created_at"
            return await self.analytical_query(query, {"file_name": file_name})
        except Exception as e:
            logger.error(f"Failed to get documents by file: {e}")
            return []

    async def delete_documents_by_file(self, file_name: str) -> int:
        """Delete all documents from a specific file."""
        try:
            query = "DELETE FROM documents WHERE file_name = $file_name"
            result = await self.analytical_query(query, {"file_name": file_name})
            deleted_count = len(result) if isinstance(result, list) else 1
            logger.info(
                f"Deleted {deleted_count} documents from file: {file_name}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete documents by file: {e}")
            return 0

    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document statistics."""
        try:
            stats = {}
            
            # Total chunks
            chunks_result = await self.analytical_query(
                "SELECT count() AS total_chunks FROM documents"
            )
            stats["total_chunks"] = chunks_result[0].get("total_chunks", 0) if chunks_result else 0
            
            # Total unique documents (using a different approach)
            # First get all document IDs, then count unique ones in Python
            all_docs_result = await self.analytical_query(
                "SELECT document_id FROM documents"
            )
            unique_doc_ids = set()
            if all_docs_result:
                for doc in all_docs_result:
                    if 'document_id' in doc:
                        unique_doc_ids.add(doc['document_id'])
            stats["total_documents"] = len(unique_doc_ids)
            
            # File type distribution
            file_types_result = await self.analytical_query(
                "SELECT file_type, count() AS count FROM documents GROUP BY file_type"
            )
            stats["file_types"] = {item["file_type"]: item["count"] for item in file_types_result} if file_types_result else {}
            
            # Average chunks per document
            if stats["total_documents"] > 0:
                stats["avg_chunks_per_document"] = stats["total_chunks"] / stats["total_documents"]
            else:
                stats["avg_chunks_per_document"] = 0
                
            return stats
        except Exception as e:
            logger.error(f"Failed to get document statistics: {e}")
            return {}

    async def get_documents_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document, ordered by chunk index."""
        try:
            query = """
            SELECT * FROM documents 
            WHERE document_id = $document_id 
            ORDER BY chunk_index ASC
            """
            return await self.analytical_query(query, {"document_id": document_id})
        except Exception as e:
            logger.error(f"Failed to get documents by document ID: {e}")
            return []

    async def search_documents_by_content(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by content using text search."""
        try:
            query = """
            SELECT *, chunk_index, total_chunks, document_id, file_name 
            FROM documents 
            WHERE content CONTAINS $search_term 
            ORDER BY created_at DESC 
            LIMIT $limit
            """
            return await self.analytical_query(query, {"search_term": search_term, "limit": limit})
        except Exception as e:
            logger.error(f"Failed to search documents by content: {e}")
            return []

    async def get_file_list(self) -> List[Dict[str, Any]]:
        """Get a list of all files with their statistics."""
        try:
            query = """
            SELECT 
                file_name,
                file_type,
                file_size,
                count() AS chunk_count,
                count(DISTINCT document_id) AS document_count,
                min(created_at) AS first_added,
                max(updated_at) AS last_updated
            FROM documents 
            GROUP BY file_name, file_type, file_size
            ORDER BY first_added DESC
            """
            return await self.analytical_query(query)
        except Exception as e:
            logger.error(f"Failed to get file list: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            await self.analytical_query("SELECT 1 AS health FROM documents LIMIT 1")
            return True
        except Exception:
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
