import json
import os

from gpt_researcher.config.config import Config


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


def clear_llm_env(monkeypatch):
    for key in (
        "FAST_LLM",
        "SMART_LLM",
        "STRATEGIC_LLM",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLAMA_SWAP_URL",
        "LLAMA_SWAP_ENABLED",
        "LLAMA_SWAP_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_llama_swap_running_model_becomes_default_llm(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLAMA_SWAP_URL", "http://llama-swap.local:8080")

    def fake_urlopen(url, timeout):
        assert url == "http://llama-swap.local:8080/running"
        assert timeout == 1.0
        return FakeResponse({"running": [{"model": "qwen3-coder", "state": "running"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = Config()

    assert cfg.fast_llm == "openai:qwen3-coder"
    assert cfg.smart_llm == "openai:qwen3-coder"
    assert cfg.strategic_llm == "openai:qwen3-coder"
    assert cfg.fast_llm_provider == "openai"
    assert cfg.fast_llm_model == "qwen3-coder"
    assert cfg.smart_llm_model == "qwen3-coder"
    assert cfg.strategic_llm_model == "qwen3-coder"
    assert cfg.llama_swap_url == "http://llama-swap.local:8080"
    assert cfg.llama_swap_enabled is True
    assert cfg.llama_swap_timeout == 1.0
    assert "http://llama-swap.local:8080/v1" == os.environ["OPENAI_BASE_URL"]
    assert "llama-swap" == os.environ["OPENAI_API_KEY"]


def test_llama_swap_does_not_override_explicit_llm_env(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("FAST_LLM", "anthropic:claude-haiku-4-5")
    monkeypatch.setenv("SMART_LLM", "anthropic:claude-sonnet-4-6")
    monkeypatch.setenv("STRATEGIC_LLM", "anthropic:claude-opus-4-7")

    def fake_urlopen(url, timeout):
        raise AssertionError("llama-swap should not be queried when all LLM env vars are set")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = Config()

    assert cfg.fast_llm == "anthropic:claude-haiku-4-5"
    assert cfg.smart_llm == "anthropic:claude-sonnet-4-6"
    assert cfg.strategic_llm == "anthropic:claude-opus-4-7"


def test_llama_swap_does_not_override_custom_config_file(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "FAST_LLM": "anthropic:claude-haiku-4-5",
                "SMART_LLM": "anthropic:claude-sonnet-4-6",
                "STRATEGIC_LLM": "anthropic:claude-opus-4-7",
            }
        )
    )

    def fake_urlopen(url, timeout):
        raise AssertionError("llama-swap should not be queried for custom LLM config values")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = Config(str(config_path))

    assert cfg.fast_llm == "anthropic:claude-haiku-4-5"
    assert cfg.smart_llm == "anthropic:claude-sonnet-4-6"
    assert cfg.strategic_llm == "anthropic:claude-opus-4-7"


def test_llama_swap_falls_back_to_default_when_no_running_model(monkeypatch):
    clear_llm_env(monkeypatch)

    def fake_urlopen(url, timeout):
        return FakeResponse({"running": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = Config()

    assert cfg.fast_llm == "openai:gpt-4o-mini"
    assert cfg.smart_llm == "openai:gpt-4.1"
    assert cfg.strategic_llm == "openai:o4-mini"
    assert "OPENAI_BASE_URL" not in os.environ


def test_llama_swap_accepts_running_endpoint_url(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLAMA_SWAP_URL", "http://localhost:8080/running")

    def fake_urlopen(url, timeout):
        assert url == "http://localhost:8080/running"
        return FakeResponse({"running": [{"model": "local-model"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = Config()

    assert cfg.smart_llm == "openai:local-model"
    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:8080/v1"


def test_blank_llm_env_values_still_allow_llama_swap_defaults(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("FAST_LLM", "")
    monkeypatch.setenv("SMART_LLM", "")
    monkeypatch.setenv("STRATEGIC_LLM", "")

    def fake_urlopen(url, timeout):
        return FakeResponse({"running": [{"model": "blank-env-model", "state": "running"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = Config()

    assert cfg.fast_llm == "openai:blank-env-model"
    assert cfg.smart_llm == "openai:blank-env-model"
    assert cfg.strategic_llm == "openai:blank-env-model"
