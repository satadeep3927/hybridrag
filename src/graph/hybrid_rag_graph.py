"""
Hybrid RAG Graph Implementation using LangGraph

This module implements the core workflow for the Hybrid RAG system using LangGraph,
orchestrating between semantic retrieval and analytical queries.
"""

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from ..config.config_manager import get_config
from ..prompts.prompt_manager import PromptManager
from ..tools.analytical_query import AnalyticalQueryTool
from ..tools.vector_search import VectorSearchTool


class GraphState(TypedDict):
    """State object for the Hybrid RAG graph."""

    user_query: str
    messages: List[BaseMessage]
    query_analysis: Optional[Dict[str, Any]]
    vector_results: Optional[str]
    analytical_results: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
    iteration_count: int


class HybridRAGGraph:
    """
    Hybrid RAG Graph implementation using LangGraph.

    This class orchestrates the workflow between query analysis, tool selection,
    retrieval execution, and response synthesis.
    """

    def __init__(self):
        """Initialize the Hybrid RAG Graph."""
        self.config = get_config()
        self.prompt_manager = PromptManager()

        # Initialize LLM with OpenAI
        self.llm = ChatOpenAI(
            model=self.config.openai.model,
            temperature=self.config.openai.temperature,
            max_tokens=self.config.openai.max_tokens,
            api_key=self.config.openai.api_key,
            base_url=self.config.openai.base_url,
        )

        # Initialize tools
        self.vector_search_tool = VectorSearchTool()
        self.analytical_query_tool = AnalyticalQueryTool()

        # Store tools for reference
        self.tools = [self.vector_search_tool, self.analytical_query_tool]

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Hybrid RAG Graph initialized successfully")

    def _clean_json_response(self, response_content: str) -> str:
        """
        Clean JSON response by removing markdown code blocks and other formatting.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code block markers
        cleaned = re.sub(r'```json\s*', '', response_content)
        cleaned = re.sub(r'\s*```', '', cleaned)
        
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Try to find JSON content if it's embedded in other text
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        
        return cleaned

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("execute_vector_search", self._execute_vector_search)
        workflow.add_node("execute_analytical_query", self._execute_analytical_query)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("handle_error", self._handle_error)

        # Add edges
        workflow.set_entry_point("analyze_query")

        # Conditional routing from query analysis
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_after_analysis,
            {
                "vector_search": "execute_vector_search",
                "analytical_query": "execute_analytical_query",
                "hybrid": "execute_vector_search",  # Start with vector search for hybrid
                "error": "handle_error",
            },
        )

        # From vector search
        workflow.add_conditional_edges(
            "execute_vector_search",
            self._route_after_vector_search,
            {
                "analytical_query": "execute_analytical_query",
                "synthesize": "synthesize_response",
                "error": "handle_error",
            },
        )

        # From analytical query
        workflow.add_conditional_edges(
            "execute_analytical_query",
            self._route_after_analytical_query,
            {
                "vector_search": "execute_vector_search",
                "synthesize": "synthesize_response",
                "error": "handle_error",
            },
        )

        # End points
        workflow.add_edge("synthesize_response", END)
        workflow.add_edge("handle_error", END)

        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _analyze_query(self, state: GraphState) -> GraphState:
        """Analyze the user query to determine the best retrieval strategy."""
        try:
            logger.debug("Analyzing user query")

            # Load the query router prompt
            router_prompt = self.prompt_manager.render_prompt(
                "query_router", {"user_query": state["user_query"]}
            )

            # Get query analysis from LLM  
            messages = [SystemMessage(content=router_prompt)]
            response = await self.llm.ainvoke(messages)
            print(response.content)

            # Parse the JSON response with cleanup
            try:
                # Clean the response content first
                cleaned_content = self._clean_json_response(response.content)
                query_analysis = json.loads(cleaned_content)
                logger.debug(f"Successfully parsed query analysis: {query_analysis}")
            except json.JSONDecodeError as e:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse query analysis JSON: {e}, using fallback")
                logger.debug(f"Raw response content: {response.content}")
                logger.debug(f"Cleaned content: {cleaned_content if 'cleaned_content' in locals() else 'N/A'}")
                query_analysis = {
                    "primary_tool": "vector_search",
                    "reasoning": "Fallback to vector search due to parsing error",
                    "confidence": "low",
                }

            state["query_analysis"] = query_analysis
            state["messages"].append(response)

            logger.info(
                f"Query analyzed: {query_analysis.get('primary_tool', 'unknown')} approach selected"
            )

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["error"] = f"Query analysis error: {str(e)}"

        return state

    async def _execute_vector_search(self, state: GraphState) -> GraphState:
        """Execute vector search retrieval."""
        try:
            logger.debug("Executing vector search")

            # Determine the query to use
            analysis = state.get("query_analysis", {})
            search_query = analysis.get("semantic_query", state["user_query"])

            # Execute vector search
            vector_results = await self.vector_search_tool._arun(search_query)
            state["vector_results"] = vector_results

            logger.info("Vector search completed successfully")

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            state["error"] = f"Vector search error: {str(e)}"

        return state

    async def _execute_analytical_query(self, state: GraphState) -> GraphState:
        """Execute analytical query."""
        try:
            logger.debug("Executing analytical query")

            # Determine the query to use
            analysis = state.get("query_analysis", {})

            if "analytical_operation" in analysis:
                query_input = analysis["analytical_operation"]
            else:
                query_input = state["user_query"]

            # Execute analytical query
            analytical_results = await self.analytical_query_tool._arun(query_input)
            state["analytical_results"] = analytical_results

            logger.info("Analytical query completed successfully")

        except Exception as e:
            logger.error(f"Analytical query failed: {e}")
            state["error"] = f"Analytical query error: {str(e)}"

        return state

    async def _synthesize_response(self, state: GraphState) -> GraphState:
        """Synthesize the final response from all retrieved information."""
        try:
            logger.debug("Synthesizing final response")

            # Load the synthesis prompt
            synthesis_prompt = self.prompt_manager.render_prompt(
                "synthesis",
                {
                    "user_query": state["user_query"],
                    "vector_results": state.get("vector_results"),
                    "analytical_results": state.get("analytical_results"),
                },
            )

            # Generate the final response
            messages = [SystemMessage(content=synthesis_prompt)]
            response = await self.llm.ainvoke(messages)

            state["final_response"] = response.content
            state["messages"].append(response)

            logger.info("Response synthesis completed")

        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            state["error"] = f"Synthesis error: {str(e)}"

        return state

    async def _handle_error(self, state: GraphState) -> GraphState:
        """Handle errors in the workflow."""
        error_message = state.get("error", "Unknown error occurred")
        logger.error(f"Workflow error: {error_message}")

        # Provide a helpful error response
        state["final_response"] = (
            f"I encountered an error while processing your request: {error_message}. Please try rephrasing your question or contact support if the issue persists."
        )

        return state

    def _route_after_analysis(self, state: GraphState) -> str:
        """Route after query analysis based on the determined strategy."""
        if state.get("error"):
            return "error"

        analysis = state.get("query_analysis", {})
        primary_tool = analysis.get("primary_tool", "vector_search")

        if primary_tool == "hybrid":
            return "vector_search"  # Start with vector search for hybrid queries
        else:
            return primary_tool

    def _route_after_vector_search(self, state: GraphState) -> str:
        """Route after vector search execution."""
        if state.get("error"):
            return "error"

        analysis = state.get("query_analysis", {})
        primary_tool = analysis.get("primary_tool", "vector_search")

        if primary_tool == "hybrid" and not state.get("analytical_results"):
            return "analytical_query"
        else:
            return "synthesize"

    def _route_after_analytical_query(self, state: GraphState) -> str:
        """Route after analytical query execution."""
        if state.get("error"):
            return "error"

        analysis = state.get("query_analysis", {})
        primary_tool = analysis.get("primary_tool", "analytical_query")

        if primary_tool == "hybrid" and not state.get("vector_results"):
            return "vector_search"
        else:
            return "synthesize"

    async def process_query(
        self, user_query: str, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the hybrid RAG workflow.

        Args:
            user_query: The user's natural language query
            thread_id: Optional thread ID for conversation continuity

        Returns:
            Dictionary containing the response and workflow information
        """
        try:
            # Initialize state
            initial_state = GraphState(
                user_query=user_query,
                messages=[HumanMessage(content=user_query)],
                query_analysis=None,
                vector_results=None,
                analytical_results=None,
                final_response=None,
                error=None,
                iteration_count=0,
            )

            # Create config for thread
            config = {"configurable": {"thread_id": thread_id or "default"}}

            # Execute the workflow
            logger.info(f"Processing query: {user_query}")

            final_state = await self.graph.ainvoke(initial_state, config)

            # Extract results
            result = {
                "response": final_state.get("final_response", "No response generated"),
                "query_analysis": final_state.get("query_analysis"),
                "vector_results": final_state.get("vector_results"),
                "analytical_results": final_state.get("analytical_results"),
                "error": final_state.get("error"),
                "success": final_state.get("error") is None,
            }

            logger.info("Query processing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "error": str(e),
                "success": False,
            }

    def get_workflow_visualization(self) -> str:
        """Get a visual representation of the workflow graph."""
        try:
            # This would require graphviz or similar for actual visualization
            # For now, return a text description
            return """
            Hybrid RAG Workflow:
            
            1. analyze_query → Route based on query type
               ├── vector_search (semantic queries)
               ├── analytical_query (quantitative queries)  
               └── hybrid (both)
            
            2. execute_vector_search → Get semantically similar documents
               └── Route to analytical_query (if hybrid) or synthesize
            
            3. execute_analytical_query → Perform structured data analysis
               └── Route to vector_search (if hybrid) or synthesize
            
            4. synthesize_response → Combine all results into final answer
               └── END
            
            Error handling at each step routes to handle_error → END
            """
        except Exception as e:
            logger.error(f"Failed to generate workflow visualization: {e}")
            return "Workflow visualization unavailable"

    async def health_check(self) -> Dict[str, bool]:
        """Check the health of all system components."""
        health_status = {}

        try:
            # Check LLM
            test_response = await self.llm.ainvoke([HumanMessage(content="test")])
            health_status["llm"] = bool(test_response.content)
        except Exception:
            health_status["llm"] = False

        try:
            # Check vector search tool
            health_status["vector_search"] = (
                await self.vector_search_tool.db_client.health_check()
            )
        except Exception:
            health_status["vector_search"] = False

        try:
            # Check analytical query tool
            health_status["analytical_query"] = (
                await self.analytical_query_tool.db_client.health_check()
            )
        except Exception:
            health_status["analytical_query"] = False

        try:
            # Check prompt manager
            prompts = self.prompt_manager.list_prompts()
            health_status["prompt_manager"] = len(prompts) > 0
        except Exception:
            health_status["prompt_manager"] = False

        health_status["overall"] = all(health_status.values())

        return health_status
