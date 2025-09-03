"""
Vector Search Tool for Hybrid RAG System

This tool performs semantic vector search using configurable embedding providers.
"""

import asyncio
from typing import Any, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain.tools import BaseTool
from loguru import logger
from pydantic import Field

from ..config.config_manager import get_config
from ..database.surrealdb_client import SurrealDBClient
from ..embeddings.huggingface_embeddings import HuggingFaceEmbeddings


class VectorSearchTool(BaseTool):
    """
    Tool for performing semantic vector search.

    This tool converts natural language queries into vector embeddings
    and searches for semantically similar documents in SurrealDB.
    """

    name: str = "vector_search"
    description: str = """
    Performs semantic vector search to find relevant documents based on meaning and context.
    Use this tool when you need to find documents that are semantically related to a query,
    even if they don't contain exact keyword matches.
    
    Input should be a natural language query describing what you're looking for.
    """

    db_client: SurrealDBClient = Field(default_factory=SurrealDBClient)
    embeddings: Optional[Any] = Field(default=None)

    def __init__(self, **kwargs):
        """Initialize the vector search tool."""
        super().__init__(**kwargs)
        config = get_config()

        # Initialize embeddings based on provider configuration
        if config.embeddings.provider == "huggingface":
            logger.info("Using Hugging Face embeddings")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embeddings.huggingface.model_name,
                device=config.embeddings.huggingface.device
            )
        else:
            logger.info("Using OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                model=config.openai.embedding_model,
                api_key=config.openai.api_key,
                base_url=config.openai.base_url,
            )

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = None,
        file_filter: Optional[str] = None,
    ) -> str:
        """
        Async implementation of vector search.

        Args:
            query: The search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            file_filter: Optional file name filter

        Returns:
            Formatted search results as string
        """
        try:
            config = get_config()

            # Use config default if threshold not specified
            if similarity_threshold is None:
                similarity_threshold = config.vector_search.similarity_threshold

            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query}")
            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, query
            )

            # Ensure database connection
            if not self.db_client._initialized:
                await self.db_client.connect()

            # Perform vector search
            logger.debug(
                f"Performing vector search with threshold: {similarity_threshold}"
            )
            results = await self.db_client.vector_search(
                query_vector=query_embedding,
                limit=limit,
                similarity_threshold=similarity_threshold,
                file_filter=file_filter,
            )

            if not results:
                return "No relevant documents found for the given query."

            # Format results
            formatted_results = self._format_results(results)
            logger.info(f"Vector search returned {len(results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Vector search tool error: {e}")
            return f"Error performing vector search: {str(e)}"

    def _run(self, query: str, **kwargs) -> str:
        """
        Sync wrapper for async vector search.

        Args:
            query: The search query
            **kwargs: Additional arguments

        Returns:
            Formatted search results as string
        """
        try:
            # Run async function in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._arun(query, **kwargs))
                    return future.result()
            else:
                return asyncio.run(self._arun(query, **kwargs))
        except Exception as e:
            logger.error(f"Vector search sync wrapper error: {e}")
            return f"Error performing vector search: {str(e)}"

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for display.

        Args:
            results: List of search results

        Returns:
            Formatted results string
        """
        if not results:
            return "No results found."

        formatted = []
        formatted.append("# Vector Search Results\\n")

        for i, result in enumerate(results, 1):
            similarity = result.get("similarity", 0)
            content = result.get("content", "")
            file_name = result.get("file_name", "Unknown")

            # Truncate content if too long
            content_preview = content[:300] + \
                "..." if len(content) > 300 else content

            formatted.append(
                f"## Result {i} (Similarity: {similarity:.3f})\\n")
            formatted.append(f"**Source:** {file_name}\\n")
            formatted.append(f"**Content:** {content_preview}\\n")
            formatted.append("---\\n")

        return "\\n".join(formatted)

    async def search_with_context(
        self, query: str, context_window: int = 3, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search with additional context from surrounding chunks.

        Args:
            query: The search query
            context_window: Number of surrounding chunks to include
            **kwargs: Additional search parameters

        Returns:
            Results with expanded context
        """
        try:
            # Get initial results
            config = get_config()
            similarity_threshold = kwargs.get(
                "similarity_threshold", config.vector_search.similarity_threshold
            )

            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, query
            )

            if not self.db_client._initialized:
                await self.db_client.connect()

            results = await self.db_client.vector_search(
                query_vector=query_embedding,
                limit=kwargs.get("limit", 10),
                similarity_threshold=similarity_threshold,
                file_filter=kwargs.get("file_filter"),
            )

            # TODO: Implement context expansion logic
            # This would require storing chunk positions and relationships

            return results

        except Exception as e:
            logger.error(f"Context search error: {e}")
            return []

    async def search_by_file(self, query: str, file_name: str, **kwargs) -> str:
        """
        Search within a specific file.

        Args:
            query: The search query
            file_name: Name of the file to search within
            **kwargs: Additional search parameters

        Returns:
            Formatted search results
        """
        return await self._arun(query, file_filter=file_name, **kwargs)

    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of the embedding model."""
        config = get_config()

        if config.embeddings.provider == "huggingface":
            return config.embeddings.huggingface.dimensions
        else:
            return config.openai.embedding_dimensions

    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Enhanced with individual request fallback for better compatibility.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                logger.warning("Empty texts list provided for embedding")
                return []

            logger.debug(
                f"Embedding {len(texts)} texts using {type(self.embeddings).__name__}")

            # For Hugging Face embeddings, use async methods directly
            if hasattr(self.embeddings, 'aembed_documents'):
                if len(texts) == 1:
                    embedding = await self.embeddings.aembed_query(texts[0])
                    return [embedding]
                else:
                    embeddings = await self.embeddings.aembed_documents(texts)
                    return embeddings

            # For OpenAI embeddings, use the existing fallback logic
            # Handle the case where we only have one text
            if len(texts) == 1:
                embedding = await asyncio.to_thread(self.embeddings.embed_query, texts[0])
                return [embedding]

            # Try batch embedding first
            try:
                embeddings = await asyncio.to_thread(self.embeddings.embed_documents, texts)

                if embeddings and len(embeddings) == len(texts):
                    logger.debug(
                        f"Successfully generated {len(embeddings)} embeddings via batch")
                    return embeddings
                else:
                    logger.warning(
                        f"Batch embedding returned {len(embeddings) if embeddings else 0} embeddings for {len(texts)} texts, using individual requests")

            except Exception as batch_error:
                logger.warning(
                    f"Batch embedding failed: {batch_error}, falling back to individual requests")

            # Fallback: process texts individually
            logger.debug("Processing embeddings individually")
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    embedding = await asyncio.to_thread(self.embeddings.embed_query, text)
                    embeddings.append(embedding)
                    logger.debug(f"Generated embedding {i+1}/{len(texts)}")
                except Exception as e:
                    logger.error(f"Failed to embed text {i+1}: {e}")
                    raise

            logger.debug(
                f"Successfully generated {len(embeddings)} embeddings individually")
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            logger.error(f"Input texts count: {len(texts) if texts else 0}")
            if texts:
                logger.error(f"First text sample: {texts[0][:100]}...")
            raise
