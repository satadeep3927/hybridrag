"""
Prompt Manager for Hybrid RAG System

This module manages prompts stored as markdown files, providing
template rendering and prompt organization capabilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger

from ..config.config_manager import get_config


class PromptManager:
    """
    Manager for prompt templates stored as markdown files.

    Supports Jinja2 templating, multi-language prompts, and
    dynamic prompt loading and rendering.
    """

    def __init__(self, prompts_directory: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            prompts_directory: Path to prompts directory
        """
        config = get_config()
        self.prompts_dir = Path(prompts_directory or config.prompts.directory)
        self.default_language = config.prompts.default_language

        # Ensure prompts directory exists
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache for loaded prompts
        self._prompt_cache: Dict[str, Dict[str, str]] = {}

        logger.info(f"Prompt manager initialized with directory: {self.prompts_dir}")

    def load_prompt(
        self,
        prompt_name: str,
        language: Optional[str] = None,
        force_reload: bool = False,
    ) -> str:
        """
        Load a prompt template from markdown file.

        Args:
            prompt_name: Name of the prompt (without extension)
            language: Language code (defaults to configured default)
            force_reload: Whether to bypass cache and reload from file

        Returns:
            Prompt template string
        """
        lang = language or self.default_language
        cache_key = f"{prompt_name}_{lang}"

        # Check cache first
        if not force_reload and cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        # Try different file patterns
        possible_files = [
            f"{prompt_name}_{lang}.md",
            f"{prompt_name}.{lang}.md",
            f"{prompt_name}.md",
            f"{lang}/{prompt_name}.md",
        ]

        prompt_content = None
        found_file = None

        for filename in possible_files:
            file_path = self.prompts_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        prompt_content = f.read()
                    found_file = filename
                    break
                except Exception as e:
                    logger.warning(f"Failed to read prompt file {filename}: {e}")
                    continue

        if prompt_content is None:
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found for language '{lang}'"
            )

        # Cache the loaded prompt
        self._prompt_cache[cache_key] = prompt_content
        logger.debug(f"Loaded prompt '{prompt_name}' from file: {found_file}")

        return prompt_content

    def render_prompt(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Render a prompt template with variables.

        Args:
            prompt_name: Name of the prompt template
            variables: Dictionary of variables for template rendering
            language: Language code

        Returns:
            Rendered prompt string
        """
        try:
            # Load the prompt template
            prompt_template = self.load_prompt(prompt_name, language)

            # Create Jinja2 template
            template = Template(prompt_template)

            # Render with variables
            rendered = template.render(variables or {})

            logger.debug(
                f"Rendered prompt '{prompt_name}' with {len(variables or {})} variables"
            )
            return rendered

        except Exception as e:
            logger.error(f"Failed to render prompt '{prompt_name}': {e}")
            raise

    def list_prompts(self, language: Optional[str] = None) -> List[str]:
        """
        List available prompt names.

        Args:
            language: Filter by language (optional)

        Returns:
            List of available prompt names
        """
        prompts = set()

        for file_path in self.prompts_dir.rglob("*.md"):
            # Extract prompt name from various patterns
            relative_path = file_path.relative_to(self.prompts_dir)

            if "/" in str(relative_path):
                # Language subdirectory pattern
                parts = str(relative_path).split("/")
                if len(parts) == 2:
                    lang_dir, filename = parts
                    if language is None or lang_dir == language:
                        prompt_name = filename.replace(".md", "")
                        prompts.add(prompt_name)
            else:
                # Flat file pattern
                filename = relative_path.name
                if "_" in filename:
                    # prompt_name_lang.md pattern
                    name_parts = filename.replace(".md", "").split("_")
                    if len(name_parts) >= 2:
                        lang_part = name_parts[-1]
                        prompt_name = "_".join(name_parts[:-1])
                        if language is None or lang_part == language:
                            prompts.add(prompt_name)
                elif "." in filename.replace(".md", ""):
                    # prompt_name.lang.md pattern
                    name_parts = filename.replace(".md", "").split(".")
                    if len(name_parts) >= 2:
                        lang_part = name_parts[-1]
                        prompt_name = ".".join(name_parts[:-1])
                        if language is None or lang_part == language:
                            prompts.add(prompt_name)
                else:
                    # Simple prompt_name.md pattern
                    prompt_name = filename.replace(".md", "")
                    prompts.add(prompt_name)

        return sorted(list(prompts))

    def create_prompt(
        self,
        prompt_name: str,
        content: str,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a new prompt file.

        Args:
            prompt_name: Name of the prompt
            content: Prompt content
            language: Language code
            metadata: Optional metadata to include in prompt
        """
        lang = language or self.default_language

        # Choose file pattern
        filename = f"{prompt_name}_{lang}.md"
        file_path = self.prompts_dir / filename

        # Prepare content with metadata if provided
        full_content = ""
        if metadata:
            # Add metadata as YAML front matter
            full_content += "---\\n"
            for key, value in metadata.items():
                full_content += f"{key}: {value}\\n"
            full_content += "---\\n\\n"

        full_content += content

        # Write the file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            logger.info(f"Created prompt '{prompt_name}' at: {file_path}")

            # Update cache
            cache_key = f"{prompt_name}_{lang}"
            self._prompt_cache[cache_key] = content

        except Exception as e:
            logger.error(f"Failed to create prompt '{prompt_name}': {e}")
            raise

    def update_prompt(
        self, prompt_name: str, content: str, language: Optional[str] = None
    ) -> None:
        """
        Update an existing prompt.

        Args:
            prompt_name: Name of the prompt
            content: New prompt content
            language: Language code
        """
        # Clear cache first
        lang = language or self.default_language
        cache_key = f"{prompt_name}_{lang}"
        self._prompt_cache.pop(cache_key, None)

        # Create/update the prompt
        self.create_prompt(prompt_name, content, language)

    def delete_prompt(self, prompt_name: str, language: Optional[str] = None) -> None:
        """
        Delete a prompt file.

        Args:
            prompt_name: Name of the prompt
            language: Language code
        """
        lang = language or self.default_language

        # Try different file patterns
        possible_files = [
            f"{prompt_name}_{lang}.md",
            f"{prompt_name}.{lang}.md",
            f"{lang}/{prompt_name}.md",
        ]

        deleted = False
        for filename in possible_files:
            file_path = self.prompts_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted prompt file: {filename}")
                    deleted = True

                    # Clear cache
                    cache_key = f"{prompt_name}_{lang}"
                    self._prompt_cache.pop(cache_key, None)

                except Exception as e:
                    logger.error(f"Failed to delete prompt file {filename}: {e}")

        if not deleted:
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found for language '{lang}'"
            )

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._prompt_cache.clear()
        logger.debug("Prompt cache cleared")

    def get_prompt_metadata(
        self, prompt_name: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from a prompt file.

        Args:
            prompt_name: Name of the prompt
            language: Language code

        Returns:
            Dictionary of metadata
        """
        try:
            prompt_content = self.load_prompt(prompt_name, language, force_reload=True)

            # Check for YAML front matter
            if prompt_content.startswith("---\\n"):
                parts = prompt_content.split("---\\n", 2)
                if len(parts) >= 3:
                    import yaml

                    try:
                        metadata = yaml.safe_load(parts[1])
                        return metadata or {}
                    except yaml.YAMLError:
                        pass

            return {}

        except Exception as e:
            logger.error(f"Failed to get metadata for prompt '{prompt_name}': {e}")
            return {}

    def validate_template(
        self, prompt_name: str, language: Optional[str] = None
    ) -> bool:
        """
        Validate that a prompt template syntax is correct.

        Args:
            prompt_name: Name of the prompt
            language: Language code

        Returns:
            True if template is valid, False otherwise
        """
        try:
            prompt_template = self.load_prompt(prompt_name, language)
            template = Template(prompt_template)

            # Try rendering with empty variables to check syntax
            template.render({})
            return True

        except Exception as e:
            logger.error(f"Template validation failed for '{prompt_name}': {e}")
            return False
