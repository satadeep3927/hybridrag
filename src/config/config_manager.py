"""
Hybrid RAG System Configuration Manager

This module handles configuration loading and management for the Hybrid RAG system.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """SurrealDB configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    namespace: str = "hybridrag"
    database: str = "knowledge_base"
    username: str = "root"
    password: str = "root"


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    max_tokens: int = 2048
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""

    similarity_threshold: float = 0.7
    max_results: int = 10
    index_type: str = "hnsw"
    distance_metric: str = "cosine"


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    supported_formats: list[str] = ["txt", "md", "pdf", "docx"]


class LangGraphConfig(BaseModel):
    """LangGraph configuration."""

    max_iterations: int = 10
    timeout_seconds: int = 30
    enable_debug: bool = True


class PromptsConfig(BaseModel):
    """Prompts configuration."""

    directory: str = "prompts"
    default_language: str = "en"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/hybridrag.log"
    max_file_size: str = "10MB"
    backup_count: int = 5


class HuggingFaceEmbeddingConfig(BaseModel):
    """Hugging Face embedding configuration."""
    
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    dimensions: int = 384


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    
    provider: str = "huggingface"  # Options: "openai", "huggingface"
    huggingface: HuggingFaceEmbeddingConfig = HuggingFaceEmbeddingConfig()


class Config(BaseModel):
    """Main configuration class."""

    database: DatabaseConfig
    openai: OpenAIConfig
    vector_search: VectorSearchConfig
    document_processing: DocumentProcessingConfig
    langgraph: LangGraphConfig
    prompts: PromptsConfig
    logging: LoggingConfig
    embeddings: EmbeddingConfig = EmbeddingConfig()


class ConfigManager:
    """Configuration manager for the Hybrid RAG system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager."""
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[Config] = None
        self._load_environment()
        self._load_config()

    def _find_config_file(self) -> str:
        """Find the configuration file."""
        possible_paths = ["config/config.yaml", "config.yaml", "../config/config.yaml"]

        for path in possible_paths:
            if Path(path).exists():
                return path

        raise FileNotFoundError("Configuration file not found")

    def _load_environment(self):
        """Load environment variables."""
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)

    def _substitute_env_vars(self, value: Any) -> Any:
        """Substitute environment variables in configuration values."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        return value

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            # Create nested configuration objects
            database_config = DatabaseConfig(
                **config_data.get("database", {}).get("surrealdb", {})
            )
            openai_config = OpenAIConfig(**config_data.get("openai", {}))
            vector_search_config = VectorSearchConfig(
                **config_data.get("vector_search", {})
            )
            document_processing_config = DocumentProcessingConfig(
                **config_data.get("document_processing", {})
            )
            langgraph_config = LangGraphConfig(**config_data.get("langgraph", {}))
            prompts_config = PromptsConfig(**config_data.get("prompts", {}))
            logging_config = LoggingConfig(**config_data.get("logging", {}))

            self.config = Config(
                database=database_config,
                openai=openai_config,
                vector_search=vector_search_config,
                document_processing=document_processing_config,
                langgraph=langgraph_config,
                prompts=prompts_config,
                logging=logging_config,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def get_config(self) -> Config:
        """Get the loaded configuration."""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        return self.config

    def get_database_url(self) -> str:
        """Get the SurrealDB connection URL."""
        db_config = self.config.database
        return f"ws://{db_config.host}:{db_config.port}"

    def get_openai_client_config(self) -> Dict[str, Any]:
        """Get OpenAI client configuration."""
        return {
            "api_key": self.config.openai.api_key,
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """Get the configuration."""
    return get_config_manager().get_config()
