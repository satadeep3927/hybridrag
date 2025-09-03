"""
Command Line Interface for Hybrid RAG System

This module provides a CLI for interacting with the Hybrid RAG system.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.config.config_manager import get_config, get_config_manager
from src.graph.hybrid_rag_graph import HybridRAGGraph
from src.utils.document_processor import DocumentProcessor

app = typer.Typer(help="Hybrid RAG System - Semantic and Analytical Retrieval")
console = Console()


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the system"),
    thread_id: Optional[str] = typer.Option(
        None, "--thread", "-t", help="Thread ID for conversation continuity"
    ),
):
    """Ask a question to the Hybrid RAG system."""

    async def _query():
        try:
            console.print(
                Panel(
                    f"[bold blue]Question:[/bold blue] {question}", title="User Query"
                )
            )

            # Initialize the system
            rag_graph = HybridRAGGraph()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing query...", total=None)

                # Process the query
                result = await rag_graph.process_query(question, thread_id)

                progress.update(task, description="Complete!")

            # Display results
            if result["success"]:
                console.print(
                    Panel(
                        result["response"],
                        title="[bold green]Response[/bold green]",
                        border_style="green",
                    )
                )

                # Show analysis if available
                if result.get("query_analysis"):
                    analysis = result["query_analysis"]
                    console.print(
                        f"\\n[dim]Query Type: {analysis.get('primary_tool', 'unknown')}[/dim]"
                    )
                    console.print(
                        f"[dim]Reasoning: {analysis.get('reasoning', 'N/A')}[/dim]"
                    )
            else:
                console.print(
                    Panel(
                        f"[bold red]Error:[/bold red] {result['response']}",
                        border_style="red",
                    )
                )

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    asyncio.run(_query())


@app.command()
def chat():
    """Start an interactive chat session."""

    async def _chat():
        console.print(
            Panel(
                "[bold green]Hybrid RAG Interactive Chat[/bold green]\\n"
                "Ask questions using both semantic search and analytical queries.\\n"
                "Type 'exit' to quit, 'help' for commands.",
                title="Chat Mode",
            )
        )

        rag_graph = HybridRAGGraph()
        thread_id = "interactive_session"

        while True:
            try:
                question = Prompt.ask("\\n[bold blue]You[/bold blue]")

                if question.lower() in ["exit", "quit", "bye"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if question.lower() == "help":
                    console.print(
                        Panel(
                            "[bold]Available commands:[/bold]\\n"
                            "• Ask any question for hybrid retrieval\\n"
                            "• 'exit', 'quit', 'bye' - End the session\\n"
                            "• 'help' - Show this help\\n"
                            "• 'status' - Show system status",
                            title="Help",
                        )
                    )
                    continue

                if question.lower() == "status":
                    health = await rag_graph.health_check()
                    status_table = Table(title="System Status")
                    status_table.add_column("Component", style="cyan")
                    status_table.add_column("Status", style="magenta")

                    for component, status in health.items():
                        status_text = "✅ Healthy" if status else "❌ Unhealthy"
                        status_table.add_row(component.title(), status_text)

                    console.print(status_table)
                    continue

                # Process the question
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Thinking...", total=None)
                    result = await rag_graph.process_query(question, thread_id)

                # Display response
                if result["success"]:
                    console.print(
                        f"\\n[bold green]Assistant:[/bold green] {result['response']}"
                    )
                else:
                    console.print(
                        f"\\n[bold red]Error:[/bold red] {result['response']}"
                    )

            except KeyboardInterrupt:
                console.print("\\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\\n[bold red]Error:[/bold red] {e}")

    asyncio.run(_chat())


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    recursive: bool = typer.Option(
        True, "--recursive", "-r", help="Process directories recursively"
    ),
    metadata: Optional[str] = typer.Option(
        None, "--metadata", "-m", help="JSON metadata to attach to documents"
    ),
):
    """Ingest documents into the system."""

    async def _ingest():
        try:
            path_obj = Path(path)

            if not path_obj.exists():
                console.print(
                    f"[bold red]Error:[/bold red] Path does not exist: {path}"
                )
                return

            # Parse metadata if provided
            doc_metadata = {}
            if metadata:
                try:
                    doc_metadata = json.loads(metadata)
                except json.JSONDecodeError as e:
                    console.print(
                        f"[bold red]Error:[/bold red] Invalid JSON metadata: {e}"
                    )
                    return

            processor = DocumentProcessor()

            if path_obj.is_file():
                console.print(f"Processing file: [cyan]{path}[/cyan]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Processing file...", total=None)
                    doc_ids = await processor.process_file(str(path_obj), doc_metadata)

                console.print(
                    f"[green]✅ Successfully processed {len(doc_ids)} chunks[/green]"
                )

            elif path_obj.is_dir():
                console.print(f"Processing directory: [cyan]{path}[/cyan]")
                console.print(f"Recursive: {recursive}")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Processing directory...", total=None)
                    results = await processor.process_directory(
                        str(path_obj), recursive, doc_metadata
                    )

                # Display results
                total_chunks = sum(len(doc_ids) for doc_ids in results.values())
                successful_files = sum(1 for doc_ids in results.values() if doc_ids)

                console.print(
                    f"[green]✅ Processed {successful_files} files with {total_chunks} total chunks[/green]"
                )

                # Show failed files
                failed_files = [
                    file_path for file_path, doc_ids in results.items() if not doc_ids
                ]
                if failed_files:
                    console.print(
                        f"[yellow]⚠️  Failed to process {len(failed_files)} files[/yellow]"
                    )
                    for failed_file in failed_files[:5]:  # Show first 5
                        console.print(f"  - {failed_file}")
                    if len(failed_files) > 5:
                        console.print(f"  ... and {len(failed_files) - 5} more")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    asyncio.run(_ingest())


@app.command()
def list_files():
    """List all processed files in the system."""

    async def _list_files():
        try:
            processor = DocumentProcessor()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Loading files...", total=None)
                files = await processor.list_processed_files()

            if not files:
                console.print("[yellow]No files found in the system.[/yellow]")
                return

            # Create table
            table = Table(title="Processed Files")
            table.add_column("File Name", style="cyan")
            table.add_column("Chunks", justify="right", style="magenta")
            table.add_column("First Processed", style="green")
            table.add_column("Last Processed", style="green")

            for file_info in files:
                table.add_row(
                    file_info.get("file_name", "Unknown"),
                    str(file_info.get("chunk_count", 0)),
                    str(file_info.get("first_processed", "Unknown"))[:19],
                    str(file_info.get("last_processed", "Unknown"))[:19],
                )

            console.print(table)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    asyncio.run(_list_files())


@app.command()
def remove_file(
    file_name: str = typer.Argument(..., help="Name of the file to remove")
):
    """Remove a file and all its chunks from the system."""

    async def _remove_file():
        try:
            # Confirm deletion
            if not Confirm.ask(
                f"Are you sure you want to remove '[cyan]{file_name}[/cyan]' and all its chunks?"
            ):
                console.print("[yellow]Operation cancelled.[/yellow]")
                return

            processor = DocumentProcessor()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Removing file...", total=None)
                deleted_count = await processor.remove_file(file_name)

            if deleted_count > 0:
                console.print(
                    f"[green]✅ Removed {deleted_count} chunks for file '{file_name}'[/green]"
                )
            else:
                console.print(
                    f"[yellow]No chunks found for file '{file_name}'[/yellow]"
                )

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    asyncio.run(_remove_file())


@app.command()
def status():
    """Show system status and health check."""

    async def _status():
        try:
            rag_graph = HybridRAGGraph()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Checking system status...", total=None)
                health = await rag_graph.health_check()

            # Create status table
            table = Table(title="System Health Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Description", style="dim")

            descriptions = {
                "llm": "OpenAI Language Model",
                "vector_search": "Vector Search Tool & SurrealDB",
                "analytical_query": "Analytical Query Tool & SurrealDB",
                "prompt_manager": "Prompt Template Manager",
                "overall": "Overall System Health",
            }

            for component, status in health.items():
                status_text = "✅ Healthy" if status else "❌ Unhealthy"
                description = descriptions.get(component, "")
                table.add_row(
                    component.replace("_", " ").title(), status_text, description
                )

            console.print(table)

            # Show configuration info
            config = get_config()
            config_table = Table(title="Configuration")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="magenta")

            config_table.add_row("OpenAI Model", config.openai.model)
            config_table.add_row("Embedding Model", config.openai.embedding_model)
            config_table.add_row(
                "Database Host", f"{config.database.host}:{config.database.port}"
            )
            config_table.add_row("Database Name", config.database.database)
            config_table.add_row(
                "Chunk Size", str(config.document_processing.chunk_size)
            )
            config_table.add_row(
                "Vector Threshold", str(config.vector_search.similarity_threshold)
            )

            console.print("\\n")
            console.print(config_table)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    asyncio.run(_status())


@app.command()
def config():
    """Show current configuration."""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()

        console.print(
            Panel(
                f"[bold]Configuration File:[/bold] {config_manager.config_path}",
                title="Configuration",
            )
        )

        # Display config sections
        sections = {
            "Database": {
                "Host": f"{config.database.host}:{config.database.port}",
                "Namespace": config.database.namespace,
                "Database": config.database.database,
                "Username": config.database.username,
            },
            "OpenAI": {
                "Model": config.openai.model,
                "Temperature": str(config.openai.temperature),
                "Max Tokens": str(config.openai.max_tokens),
                "Embedding Model": config.openai.embedding_model,
            },
            "Vector Search": {
                "Similarity Threshold": str(config.vector_search.similarity_threshold),
                "Max Results": str(config.vector_search.max_results),
                "Distance Metric": config.vector_search.distance_metric,
            },
            "Document Processing": {
                "Chunk Size": str(config.document_processing.chunk_size),
                "Chunk Overlap": str(config.document_processing.chunk_overlap),
                "Max File Size (MB)": str(config.document_processing.max_file_size_mb),
            },
        }

        for section_name, section_config in sections.items():
            table = Table(title=section_name)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in section_config.items():
                table.add_row(key, value)

            console.print(table)
            console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    app()
