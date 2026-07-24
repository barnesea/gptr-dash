from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_researcher.utils.llm import create_chat_completion


@pytest.mark.asyncio
async def test_create_chat_completion_accepts_max_tokens_above_old_32k_cap():
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(side_effect=["ok-64000", "ok-128000"])

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as mock_get_llm:
        accepted_values = (
            (64_000, "ok-64000"),
            (128_000, "ok-128000"),
        )

        for max_tokens, expected in accepted_values:
            result = await create_chat_completion(
                messages=[{"role": "user", "content": "Generate a report"}],
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                llm_provider="anthropic",
            )

            assert result == expected

    assert [call.kwargs["max_tokens"] for call in mock_get_llm.call_args_list] == [64_000, 128_000]


@pytest.mark.asyncio
async def test_create_chat_completion_rejects_absurd_max_tokens():
    with patch("gpt_researcher.utils.llm.get_llm") as mock_get_llm:
        with pytest.raises(ValueError, match="env vars|typos"):
            await create_chat_completion(
                messages=[{"role": "user", "content": "Generate a report"}],
                model="claude-sonnet-4-6",
                max_tokens=1_000_000,
                llm_provider="anthropic",
            )

    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_llama_swap_missing_alias_route_retries_base_model_once(monkeypatch):
    monkeypatch.setenv("GPTR_LLAMA_SWAP_ALIAS_SUFFIX", "gptr")
    alias_provider = MagicMock()
    alias_provider.get_chat_response = AsyncMock(
        side_effect=RuntimeError("404 no router for requested model")
    )
    base_provider = MagicMock()
    base_provider.get_chat_response = AsyncMock(return_value="base-ok")

    with patch(
        "gpt_researcher.utils.llm.get_llm",
        side_effect=[alias_provider, base_provider],
    ) as mock_get_llm:
        result = await create_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            model="local-model:gptr",
            llm_provider="openai",
        )

    assert result == "base-ok"
    assert [call.kwargs["model"] for call in mock_get_llm.call_args_list] == [
        "local-model:gptr",
        "local-model",
    ]
    alias_provider.get_chat_response.assert_awaited_once()
    base_provider.get_chat_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_upstream_launch_failure_is_not_retried():
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(
        side_effect=RuntimeError("upstream command exited prematurely")
    )
    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider):
        with pytest.raises(RuntimeError, match="Failed to get response"):
            await create_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                model="local-model",
                llm_provider="openai",
            )

    provider.get_chat_response.assert_awaited_once()
