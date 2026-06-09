"""Configuration management for GPT Researcher.

This module provides the Config class that manages all configuration
settings for GPT Researcher including LLM providers, embeddings,
retrievers, and various operational parameters.
"""

import json
import os
import urllib.error
import urllib.request
import warnings
from typing import Any, Dict, List, Type, Union, get_args, get_origin

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts

from .runtime import apply_runtime_overrides
from .variables.base import BaseConfig
from .variables.default import DEFAULT_CONFIG


class Config:
    """Configuration manager for GPT Researcher.

    Handles loading, parsing, and managing all configuration settings
    from files, environment variables, and defaults.

    Attributes:
        CONFIG_DIR: Directory containing configuration files.
        config_path: Path to the configuration file.
        llm_kwargs: Additional keyword arguments for LLM.
        embedding_kwargs: Additional keyword arguments for embeddings.
    """

    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "variables")

    def __init__(self, config_path: str | None = None):
        """Initialize the config class.

        Args:
            config_path: Optional path to a JSON configuration file.
        """
        self.config_path = config_path
        self.llm_kwargs: Dict[str, Any] = {}
        self.embedding_kwargs: Dict[str, Any] = {}

        apply_runtime_overrides()
        config_to_use = self.load_config(config_path)
        self._apply_llama_swap_defaults(config_to_use)
        self._set_attributes(config_to_use)
        self._set_embedding_attributes()
        self._set_llm_attributes()
        self._handle_deprecated_attributes()
        if config_to_use['REPORT_SOURCE'] != 'web':
          self._set_doc_path(config_to_use)

        # MCP support configuration
        self.mcp_servers = []  # List of MCP server configurations
        self.mcp_allowed_root_paths = []  # Allowed root paths for MCP servers

        # Read from config
        if hasattr(self, 'mcp_servers'):
            self.mcp_servers = self.mcp_servers
        if hasattr(self, 'mcp_allowed_root_paths'):
            self.mcp_allowed_root_paths = self.mcp_allowed_root_paths

    def _set_attributes(self, config: Dict[str, Any]) -> None:
        """Set configuration attributes from config dictionary.

        Merges environment variables with config file values, with
        environment variables taking precedence.

        Args:
            config: Dictionary of configuration key-value pairs.
        """
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value not in (None, ""):
                value = self.convert_env_value(key, env_value, BaseConfig.__annotations__[key])
            setattr(self, key.lower(), value)

        # Handle RETRIEVER with default value
        retriever_env = os.environ.get("RETRIEVER", config.get("RETRIEVER", "tavily"))
        try:
            self.retrievers = self.parse_retrievers(retriever_env)
        except ValueError as e:
            print(f"Warning: {str(e)}. Defaulting to 'tavily' retriever.")
            self.retrievers = ["tavily"]

    def _apply_llama_swap_defaults(self, config: Dict[str, Any]) -> None:
        """Use llama-swap's currently running model as the default local LLM.

        Explicit environment variables and custom config file values continue to
        win. This only replaces the built-in OpenAI defaults.
        """
        enabled = self.convert_env_value(
            "LLAMA_SWAP_ENABLED",
            os.getenv("LLAMA_SWAP_ENABLED") or str(config.get("LLAMA_SWAP_ENABLED", True)),
            bool,
        )
        if not enabled:
            return

        llm_keys = ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM")
        needs_default = [
            key for key in llm_keys
            if not os.getenv(key) and config.get(key) == DEFAULT_CONFIG[key]
        ]
        if not needs_default:
            return

        model, base_url = self._get_llama_swap_running_model(config)
        if not model:
            return

        for key in needs_default:
            config[key] = f"openai:{model}"

        if not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = self._llama_swap_openai_base_url(base_url)
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "llama-swap"

    @classmethod
    def _get_llama_swap_running_model(cls, config: Dict[str, Any]) -> tuple[str | None, str]:
        base_url, running_url = cls._llama_swap_urls(
            os.getenv("LLAMA_SWAP_URL") or str(config.get("LLAMA_SWAP_URL", "http://localhost:8080"))
        )
        timeout = float(os.getenv("LLAMA_SWAP_TIMEOUT") or config.get("LLAMA_SWAP_TIMEOUT", 1.0))

        try:
            with urllib.request.urlopen(running_url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None, base_url

        return cls._extract_llama_swap_model(payload), base_url

    @staticmethod
    def _llama_swap_urls(url: str) -> tuple[str, str]:
        normalized = url.rstrip("/")
        if normalized.endswith("/running"):
            return normalized[: -len("/running")], normalized
        if normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")]
        return normalized, f"{normalized}/running"

    @staticmethod
    def _llama_swap_openai_base_url(base_url: str) -> str:
        return f"{base_url.rstrip('/')}/v1"

    @classmethod
    def _extract_llama_swap_model(cls, payload: Any) -> str | None:
        running = payload.get("running") if isinstance(payload, dict) else payload
        if isinstance(running, dict):
            running = [
                {"model": model, **(state if isinstance(state, dict) else {"state": state})}
                for model, state in running.items()
            ]
        if not isinstance(running, list):
            return None

        preferred = ("running", "ready", "started", "healthy")
        candidates: list[tuple[int, str]] = []
        for index, entry in enumerate(running):
            model = cls._model_from_running_entry(entry)
            if not model:
                continue
            state = ""
            if isinstance(entry, dict):
                state = str(entry.get("state", "")).lower()
            priority = 0 if state in preferred else 1
            candidates.append((priority * 1000 + index, model))

        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    @staticmethod
    def _model_from_running_entry(entry: Any) -> str | None:
        if isinstance(entry, str):
            return entry or None
        if not isinstance(entry, dict):
            return None
        for key in ("model", "id", "name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _set_embedding_attributes(self) -> None:
        """Parse and set embedding provider and model attributes."""
        self.embedding_provider, self.embedding_model = self.parse_embedding(
            self.embedding
        )

    def _set_llm_attributes(self) -> None:
        """Parse and set LLM provider and model attributes for all LLM types."""
        self.fast_llm_provider, self.fast_llm_model = self.parse_llm(self.fast_llm)
        self.smart_llm_provider, self.smart_llm_model = self.parse_llm(self.smart_llm)
        self.strategic_llm_provider, self.strategic_llm_model = self.parse_llm(self.strategic_llm)
        self.reasoning_effort = self.parse_reasoning_effort(os.getenv("REASONING_EFFORT"))

    def _handle_deprecated_attributes(self) -> None:
        """Handle deprecated configuration attributes with warnings."""
        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated and will be removed soon. Use EMBEDDING instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.embedding_provider = (
                os.environ["EMBEDDING_PROVIDER"] or self.embedding_provider
            )

            embedding_provider = os.environ["EMBEDDING_PROVIDER"]
            if embedding_provider == "ollama":
                self.embedding_model = os.environ["OLLAMA_EMBEDDING_MODEL"]
            elif embedding_provider == "custom":
                self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "custom")
            elif embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-large"
            elif embedding_provider == "azure_openai":
                self.embedding_model = "text-embedding-3-large"
            elif embedding_provider == "huggingface":
                self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            elif embedding_provider == "gigachat":
                self.embedding_model = "Embeddings"
            elif embedding_provider == "google_genai":
                self.embedding_model = "text-embedding-004"
            else:
                raise Exception("Embedding provider not found.")

        _deprecation_warning = (
            "LLM_PROVIDER, FAST_LLM_MODEL and SMART_LLM_MODEL are deprecated and "
            "will be removed soon. Use FAST_LLM and SMART_LLM instead."
        )
        if os.getenv("LLM_PROVIDER") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.fast_llm_provider
            )
            self.smart_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.smart_llm_provider
            )
        if os.getenv("FAST_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_model = os.environ["FAST_LLM_MODEL"] or self.fast_llm_model
        if os.getenv("SMART_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.smart_llm_model = os.environ["SMART_LLM_MODEL"] or self.smart_llm_model

    def _set_doc_path(self, config: Dict[str, Any]) -> None:
        self.doc_path = config['DOC_PATH']
        if self.doc_path:
            try:
                self.validate_doc_path()
            except Exception as e:
                print(f"Warning: Error validating doc_path: {str(e)}. Using default doc_path.")
                self.doc_path = DEFAULT_CONFIG['DOC_PATH']

    @classmethod
    def load_config(cls, config_path: str | None) -> Dict[str, Any]:
        """Load a configuration by name."""
        config_path = config_path or os.environ.get("CONFIG_PATH")
        if not config_path:
            return DEFAULT_CONFIG.copy()

        # config_path = os.path.join(cls.CONFIG_DIR, config_path)
        if not os.path.exists(config_path):
            if config_path and config_path != "default":
                print(f"Warning: Configuration not found at '{config_path}'. Using default configuration.")
                if not config_path.endswith(".json"):
                    print(f"Do you mean '{config_path}.json'?")
            return DEFAULT_CONFIG.copy()

        with open(config_path, "r") as f:
            custom_config = json.load(f)

        # Merge with default config to ensure all keys are present
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(custom_config)
        return merged_config

    @classmethod
    def list_available_configs(cls) -> List[str]:
        """List all available configuration names."""
        configs = ["default"]
        for file in os.listdir(cls.CONFIG_DIR):
            if file.endswith(".json"):
                configs.append(file[:-5])  # Remove .json extension
        return configs

    def parse_retrievers(self, retriever_str: str) -> List[str]:
        """Parse the retriever string into a list of retrievers and validate them."""
        from ..retrievers.utils import get_all_retriever_names
        
        retrievers = [retriever.strip()
                      for retriever in retriever_str.split(",")]
        valid_retrievers = get_all_retriever_names() or []
        invalid_retrievers = [r for r in retrievers if r not in valid_retrievers]
        if invalid_retrievers:
            raise ValueError(
                f"Invalid retriever(s) found: {', '.join(invalid_retrievers)}. "
                f"Valid options are: {', '.join(valid_retrievers)}."
            )
        return retrievers

    @staticmethod
    def parse_llm(llm_str: str | None) -> tuple[str | None, str | None]:
        """Parse llm string into (llm_provider, llm_model)."""
        from gpt_researcher.llm_provider.generic.base import _SUPPORTED_PROVIDERS

        if llm_str is None:
            return None, None
        try:
            llm_provider, llm_model = llm_str.split(":", 1)
            assert llm_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {llm_provider}.\nSupported llm providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return llm_provider, llm_model
        except ValueError:
            raise ValueError(
                "Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>' "
                "Eg 'openai:gpt-4o-mini'"
            )

    @staticmethod
    def parse_reasoning_effort(reasoning_effort_str: str | None) -> str | None:
        """Parse reasoning effort string into (reasoning_effort)."""
        if reasoning_effort_str is None:
            return ReasoningEfforts.Medium.value
        if reasoning_effort_str not in [effort.value for effort in ReasoningEfforts]:
            raise ValueError(f"Invalid reasoning effort: {reasoning_effort_str}. Valid options are: {', '.join([effort.value for effort in ReasoningEfforts])}")
        return reasoning_effort_str

    @staticmethod
    def parse_embedding(embedding_str: str | None) -> tuple[str | None, str | None]:
        """Parse embedding string into (embedding_provider, embedding_model)."""
        from gpt_researcher.memory.embeddings import _SUPPORTED_PROVIDERS

        if embedding_str is None:
            return None, None
        try:
            embedding_provider, embedding_model = embedding_str.split(":", 1)
            assert embedding_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {embedding_provider}.\nSupported embedding providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return embedding_provider, embedding_model
        except ValueError:
            raise ValueError(
                "Set EMBEDDING = '<embedding_provider>:<embedding_model>' "
                "Eg 'openai:text-embedding-3-large'"
            )

    def validate_doc_path(self):
        """Ensure that the folder exists at the doc path"""
        os.makedirs(self.doc_path, exist_ok=True)

    @staticmethod
    def convert_env_value(key: str, env_value: str, type_hint: Type) -> Any:
        """Convert environment variable to the appropriate type based on the type hint."""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            # Handle Union types (e.g., Union[str, None])
            for arg in args:
                if arg is type(None):
                    if env_value.lower() in ("none", "null", ""):
                        return None
                else:
                    try:
                        return Config.convert_env_value(key, env_value, arg)
                    except ValueError:
                        continue
            raise ValueError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif origin is list or origin is List:
            return json.loads(env_value)
        elif type_hint is dict:
            return json.loads(env_value)
        else:
            raise ValueError(f"Unsupported type {type_hint} for key {key}")


    def set_verbose(self, verbose: bool) -> None:
        """Set the verbosity level."""
        self.llm_kwargs["verbose"] = verbose

    def get_mcp_server_config(self, name: str) -> dict:
        """
        Get the configuration for an MCP server.
        
        Args:
            name (str): The name of the MCP server to get the config for.
                
        Returns:
            dict: The server configuration, or an empty dict if the server is not found.
        """
        if not name or not self.mcp_servers:
            return {}
        
        for server in self.mcp_servers:
            if isinstance(server, dict) and server.get("name") == name:
                return server
            
        return {}
