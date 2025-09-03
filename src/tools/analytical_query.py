"""
Analytical Query Tool for Hybrid RAG System

This tool executes structured SurrealQL queries for analytical operations.
"""

import asyncio
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from loguru import logger
from pydantic import Field

from ..database.surrealdb_client import SurrealDBClient


class AnalyticalQueryTool(BaseTool):
    """
    Tool for executing analytical SurrealQL queries.

    This tool allows the system to perform structured queries, aggregations,
    filtering, and analytical operations on the data stored in SurrealDB.
    """

    name: str = "analytical_query"
    description: str = """
    Executes structured analytical queries using SurrealQL for precise data retrieval,
    filtering, aggregation, and statistical analysis.
    
    Use this tool when you need to:
    - Count documents or get statistics
    - Filter data by specific criteria
    - Perform aggregations (sum, average, etc.)
    - Get data from specific time ranges
    - Analyze patterns in the data
    
    Input should be a SurrealQL query string or a natural language description
    that will be converted to SurrealQL.
    """

    db_client: SurrealDBClient = Field(default_factory=SurrealDBClient)

    def __init__(self, **kwargs):
        """Initialize the analytical query tool."""
        super().__init__(**kwargs)

    async def _arun(
        self,
        query_input: str,
        params: Optional[Dict[str, Any]] = None,
        auto_convert: bool = True,
    ) -> str:
        """
        Async implementation of analytical query execution.

        Args:
            query_input: SurrealQL query or natural language description
            params: Optional query parameters
            auto_convert: Whether to attempt natural language to SurrealQL conversion

        Returns:
            Formatted query results as string
        """
        try:
            # Ensure database connection
            if not self.db_client._initialized:
                await self.db_client.connect()

            # Convert natural language to SurrealQL if needed
            if auto_convert and not self._is_surrealql(query_input):
                query = self._convert_to_surrealql(query_input)
            else:
                query = query_input

            logger.debug(f"Executing analytical query: {query}")

            # Execute the query
            results = await self.db_client.analytical_query(query, params)

            # Format and return results
            formatted_results = self._format_results(results, query)
            logger.info(
                f"Analytical query returned {len(results) if isinstance(results, list) else 1} results"
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Analytical query tool error: {e}")
            return f"Error executing analytical query: {str(e)}"

    def _run(self, query_input: str, **kwargs) -> str:
        """
        Sync wrapper for async analytical query execution.

        Args:
            query_input: SurrealQL query or natural language description
            **kwargs: Additional arguments

        Returns:
            Formatted query results as string
        """
        try:
            # Run async function in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._arun(query_input, **kwargs)
                    )
                    return future.result()
            else:
                return asyncio.run(self._arun(query_input, **kwargs))
        except Exception as e:
            logger.error(f"Analytical query sync wrapper error: {e}")
            return f"Error executing analytical query: {str(e)}"

    def _is_surrealql(self, query: str) -> bool:
        """
        Check if the input appears to be a SurrealQL query.

        Args:
            query: Input query string

        Returns:
            True if it looks like SurrealQL, False otherwise
        """
        surreal_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DEFINE",
            "FROM",
            "WHERE",
            "ORDER BY",
            "GROUP BY",
            "LIMIT",
        ]

        query_upper = query.upper().strip()
        return any(keyword in query_upper for keyword in surreal_keywords)

    def _convert_to_surrealql(self, natural_query: str) -> str:
        """
        Convert natural language to SurrealQL query.

        Args:
            natural_query: Natural language query description

        Returns:
            SurrealQL query string
        """
        query_lower = natural_query.lower().strip()

        # Simple pattern matching for common queries
        if "count" in query_lower and "document" in query_lower:
            if "file" in query_lower:
                return "SELECT file_name, count() FROM documents GROUP BY file_name ORDER BY count DESC"
            else:
                return "SELECT count() FROM documents GROUP ALL"

        elif "recent" in query_lower or "latest" in query_lower:
            if "document" in query_lower:
                return "SELECT * FROM documents ORDER BY created_at DESC LIMIT 10"

        elif "file" in query_lower and ("list" in query_lower or "show" in query_lower):
            return "SELECT DISTINCT file_name FROM documents ORDER BY file_name"

        elif "average" in query_lower or "mean" in query_lower:
            if "similarity" in query_lower:
                return "SELECT math::mean(similarity) FROM (SELECT vector::similarity::cosine(vector, vector) AS similarity FROM documents)"

        elif "oldest" in query_lower:
            return "SELECT * FROM documents ORDER BY created_at ASC LIMIT 10"

        elif "largest" in query_lower and "content" in query_lower:
            return "SELECT *, string::len(content) AS content_length FROM documents ORDER BY content_length DESC LIMIT 10"

        elif "empty" in query_lower or "null" in query_lower:
            return "SELECT * FROM documents WHERE content IS NULL OR content = '' OR vector IS NULL"

        else:
            # Fallback to a general document query
            logger.warning(
                f"Could not convert natural language query to SurrealQL: {natural_query}"
            )
            return "SELECT * FROM documents ORDER BY created_at DESC LIMIT 5"

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Format query results for display.

        Args:
            results: Query results
            query: Original query

        Returns:
            Formatted results string
        """
        if not results:
            return "No results found for the analytical query."

        formatted = []
        formatted.append("# Analytical Query Results\\n")
        formatted.append(f"**Query:** `{query}`\\n")
        formatted.append(f"**Results Count:** {len(results)}\\n")
        formatted.append("---\\n")

        # Handle different result types
        if len(results) == 1 and isinstance(results[0], dict):
            result = results[0]

            # Handle count queries
            if "count" in result:
                formatted.append(f"**Total Count:** {result['count']}\\n")

            # Handle grouped results
            elif len(result) > 1:
                formatted.append("**Results:**\\n")
                for key, value in result.items():
                    formatted.append(f"- **{key}:** {value}\\n")

        else:
            # Handle multiple results
            for i, result in enumerate(results[:10], 1):  # Limit to first 10
                formatted.append(f"## Result {i}\\n")

                if isinstance(result, dict):
                    for key, value in result.items():
                        if key == "content":
                            # Truncate long content
                            content_preview = (
                                str(value)[:200] + "..."
                                if len(str(value)) > 200
                                else str(value)
                            )
                            formatted.append(f"**{key}:** {content_preview}\\n")
                        elif key == "vector":
                            # Show vector info without full array
                            vector_info = (
                                f"[{len(value)} dimensions]"
                                if isinstance(value, list)
                                else str(value)
                            )
                            formatted.append(f"**{key}:** {vector_info}\\n")
                        else:
                            formatted.append(f"**{key}:** {value}\\n")
                else:
                    formatted.append(f"**Result:** {result}\\n")

                formatted.append("---\\n")

            if len(results) > 10:
                formatted.append(f"*... and {len(results) - 10} more results*\\n")

        return "\\n".join(formatted)

    async def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the document collection.

        Returns:
            Dictionary containing various statistics
        """
        try:
            if not self.db_client._initialized:
                await self.db_client.connect()

            stats = {}

            # Total document count
            count_result = await self.db_client.analytical_query(
                "SELECT count() FROM documents GROUP ALL"
            )
            stats["total_documents"] = count_result[0]["count"] if count_result else 0

            # Documents by file
            file_result = await self.db_client.analytical_query(
                "SELECT file_name, count() FROM documents GROUP BY file_name ORDER BY count DESC"
            )
            stats["documents_by_file"] = file_result

            # Date range
            date_result = await self.db_client.analytical_query(
                "SELECT math::min(created_at) AS earliest, math::max(created_at) AS latest FROM documents GROUP ALL"
            )
            if date_result:
                stats["date_range"] = date_result[0]

            return stats

        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {}

    async def search_by_metadata(
        self, metadata_key: str, metadata_value: Any, operator: str = "="
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria.

        Args:
            metadata_key: The metadata field to search
            metadata_value: The value to search for
            operator: Comparison operator (=, !=, >, <, etc.)

        Returns:
            List of matching documents
        """
        try:
            if not self.db_client._initialized:
                await self.db_client.connect()

            query = f"SELECT * FROM documents WHERE metadata.{metadata_key} {operator} $value"
            params = {"value": metadata_value}

            return await self.db_client.analytical_query(query, params)

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    async def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recently added documents.

        Args:
            limit: Number of documents to return

        Returns:
            List of recent documents
        """
        try:
            if not self.db_client._initialized:
                await self.db_client.connect()

            query = "SELECT * FROM documents ORDER BY created_at DESC LIMIT $limit"
            return await self.db_client.analytical_query(query, {"limit": limit})

        except Exception as e:
            logger.error(f"Error getting recent documents: {e}")
            return []

    async def delete_old_documents(self, days_old: int) -> int:
        """
        Delete documents older than specified days.

        Args:
            days_old: Number of days threshold

        Returns:
            Number of documents deleted
        """
        try:
            if not self.db_client._initialized:
                await self.db_client.connect()

            query = "DELETE FROM documents WHERE created_at < (time::now() - duration::from::days($days))"
            result = await self.db_client.analytical_query(query, {"days": days_old})

            deleted_count = len(result) if isinstance(result, list) else 0
            logger.info(f"Deleted {deleted_count} documents older than {days_old} days")

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old documents: {e}")
            return 0
