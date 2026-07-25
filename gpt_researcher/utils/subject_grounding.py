"""Request-scoped subject grounding for research LLM calls.

The grounding statement is created once near the start of deep research and
then inherited by asynchronous branch tasks through ``ContextVar``.  This
keeps concurrent jobs isolated without threading another prompt argument
through every research helper.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator


_SUBJECT_GROUNDING: ContextVar[dict[str, Any] | None] = ContextVar(
    "gptr_subject_grounding",
    default=None,
)
_GROUNDING_MARKER = "SUBJECT GROUNDING FOR THIS RESEARCH"


def get_subject_grounding() -> dict[str, Any] | None:
    """Return the grounding payload active in the current async context."""
    return _SUBJECT_GROUNDING.get()


def set_subject_grounding(
    grounding: dict[str, Any] | None,
) -> Token[dict[str, Any] | None]:
    """Activate a grounding payload and return the token needed to reset it."""
    return _SUBJECT_GROUNDING.set(grounding or None)


def reset_subject_grounding(
    token: Token[dict[str, Any] | None],
) -> None:
    """Restore the grounding context that preceded ``set_subject_grounding``."""
    _SUBJECT_GROUNDING.reset(token)


@contextmanager
def subject_grounding_context(
    grounding: dict[str, Any] | None,
) -> Iterator[None]:
    """Temporarily activate grounding around one async research operation."""
    token = set_subject_grounding(grounding)
    try:
        yield
    finally:
        reset_subject_grounding(token)


def subject_grounding_instruction(
    grounding: dict[str, Any] | None = None,
) -> str:
    """Render the plain-English instruction appended to an LLM system prompt."""
    payload = grounding if grounding is not None else get_subject_grounding()
    statement = str((payload or {}).get("statement") or "").strip()
    if not statement:
        return ""
    facts = []
    for item in (payload or {}).get("defining_facts", [])[:12]:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        fact = str(item.get("fact") or "").strip()
        if not subject or not fact:
            continue
        fact_type = str(item.get("fact_type") or "identity").strip()
        confidence = str(
            item.get("confidence_label") or "medium"
        ).strip()
        evidence_urls = [
            str(url).strip()
            for url in item.get("evidence_urls", [])[:2]
            if str(url).strip()
        ]
        source_suffix = (
            " (orientation source: "
            + ", ".join(evidence_urls)
            + ")"
            if evidence_urls
            else ""
        )
        facts.append(
            f"- [{confidence}; {fact_type}] {subject}: {fact}"
            f"{source_suffix}"
        )
    fact_block = (
        "\n\nPOSITIVE DEFINING FACTS FROM THE GROUNDING SEARCH:\n"
        + "\n".join(facts)
        if facts
        else ""
    )
    return (
        f"{_GROUNDING_MARKER}:\n{statement}{fact_block}\n\n"
        "Use this material to understand what the subject is, its positively "
        "identified variants or components, and how its parts relate. Keep "
        "every later claim attached to the specific subject, variant, version, "
        "operation, region, population, or other scope it actually describes. "
        "These quick-search facts guide relevance and applicability; they are "
        "not final report evidence unless the cited page is fetched and "
        "verified during research."
    )


def inject_subject_grounding(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy messages and append active grounding to the first system prompt."""
    instruction = subject_grounding_instruction()
    copied = [dict(message) for message in messages]
    if not instruction:
        return copied
    if any(
        _GROUNDING_MARKER in str(message.get("content") or "")
        for message in copied
    ):
        return copied

    for message in copied:
        if message.get("role") == "system":
            content = str(message.get("content") or "").rstrip()
            message["content"] = f"{content}\n\n{instruction}".strip()
            return copied

    copied.insert(0, {"role": "system", "content": instruction})
    return copied
