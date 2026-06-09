"""Runtime configuration overrides shared by backend and MCP services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .variables.base import BaseConfig
from .variables.default import DEFAULT_CONFIG


RUNTIME_CONFIG_PATH_ENV = "CONFIG_OVERRIDES_PATH"
DEFAULT_RUNTIME_CONFIG_PATH = Path("data/runtime-config.json")

EXTRA_ENV_DEFAULTS: dict[str, str] = {
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": "",
    "TAVILY_API_KEY": "",
    "XQUIK_API_KEY": "",
    "LANGCHAIN_API_KEY": "",
    "LANGCHAIN_TRACING_V2": "",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_PROJECT": "gpt-researcher",
    "GOOGLE_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "MISTRAL_BASE_URL": "",
    "NEXT_PUBLIC_GPTR_API_URL": "",
    "NEXT_PUBLIC_GA_MEASUREMENT_ID": "",
    "MCP_TRANSPORT": "sse",
    "MCP_PORT": "8001",
}


def runtime_config_path() -> Path:
    return Path(os.getenv(RUNTIME_CONFIG_PATH_ENV, str(DEFAULT_RUNTIME_CONFIG_PATH)))


def read_runtime_overrides(path: Path | None = None) -> dict[str, str]:
    config_path = path or runtime_config_path()
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    overrides = data.get("overrides") if isinstance(data, dict) else None
    if not isinstance(overrides, dict):
        return {}
    return {str(key): str(value) for key, value in overrides.items() if value is not None}


def write_runtime_overrides(overrides: dict[str, Any], path: Path | None = None) -> None:
    config_path = path or runtime_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {
        str(key): str(value)
        for key, value in overrides.items()
        if value is not None and str(value) != ""
    }
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"overrides": cleaned}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(config_path)


def apply_runtime_overrides() -> dict[str, str]:
    overrides = read_runtime_overrides()
    for key, value in overrides.items():
        os.environ[key] = value
    return overrides


def available_runtime_options() -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}
    annotations = BaseConfig.__annotations__
    for key, default_value in DEFAULT_CONFIG.items():
        options[key] = {
            "key": key,
            "category": "GPT Researcher",
            "default": default_value,
            "type": _type_name(annotations.get(key, type(default_value))),
        }

    for key, type_hint in annotations.items():
        options.setdefault(
            key,
            {
                "key": key,
                "category": "GPT Researcher",
                "default": "",
                "type": _type_name(type_hint),
            },
        )

    for key, default_value in EXTRA_ENV_DEFAULTS.items():
        options.setdefault(
            key,
            {
                "key": key,
                "category": "Environment",
                "default": default_value,
                "type": "str",
            },
        )

    for key in os.environ:
        if key.startswith(("GPTR_", "GPT_", "OPENAI_", "TAVILY_", "LANGCHAIN_", "MCP_", "LLAMA_SWAP_")):
            options.setdefault(
                key,
                {
                    "key": key,
                    "category": "Environment",
                    "default": EXTRA_ENV_DEFAULTS.get(key, ""),
                    "type": "str",
                },
            )

    return dict(sorted(options.items()))


def runtime_config_snapshot(apply_saved: bool = True) -> dict[str, Any]:
    if apply_saved:
        apply_runtime_overrides()
    overrides = read_runtime_overrides()
    options = []

    for key, metadata in available_runtime_options().items():
        default_value = metadata["default"]
        value = os.getenv(key)
        source = "environment"

        if key in overrides:
            value = overrides[key]
            source = "saved override"
        elif value in (None, ""):
            value = default_value
            source = "default"

        options.append(
            {
                **metadata,
                "value": value,
                "override": overrides.get(key, ""),
                "source": source,
                "sensitive": _is_sensitive(key),
            }
        )

    return {
        "path": str(runtime_config_path()),
        "options": options,
        "overrides": overrides,
    }


def update_runtime_overrides(updates: dict[str, Any], persist: bool = True) -> dict[str, Any]:
    overrides = read_runtime_overrides()

    for key, value in updates.items():
        normalized_key = str(key)
        if value is None or str(value) == "":
            overrides.pop(normalized_key, None)
            os.environ.pop(normalized_key, None)
        else:
            string_value = str(value)
            overrides[normalized_key] = string_value
            os.environ[normalized_key] = string_value

    if persist:
        write_runtime_overrides(overrides)

    return runtime_config_snapshot(apply_saved=persist)


def _is_sensitive(key: str) -> bool:
    return any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))


def _type_name(type_hint: Any) -> str:
    name = getattr(type_hint, "__name__", None)
    if name:
        return name
    return str(type_hint).replace("typing.", "")
