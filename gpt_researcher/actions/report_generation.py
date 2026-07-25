import asyncio
import json
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
from ..config.config import Config
from ..utils.llm import create_chat_completion
from ..utils.logger import get_formatted_logger
from ..prompts import PromptFamily, get_prompt_by_report_type
from ..utils.enum import Tone

logger = get_formatted_logger()

MARKDOWN_LINK_PATTERN = re.compile(
    r"(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)\)",
    re.IGNORECASE,
)
REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^#{1,6}\s+references\s*$"
)
INLINE_LINK_PATTERN = re.compile(
    r"(?<!!)\[[^\]]+\]\(https?://[^)\s]+\)",
    re.IGNORECASE,
)
BARE_URL_PATTERN = re.compile(
    r"(?<!\()https?://[^\s<>\])]+",
    re.IGNORECASE,
)
SUPPLEMENTAL_QUALIFICATION_PATTERN = re.compile(
    r"\b(?:according to|reports?|reported|practitioner|community|"
    r"anecdotal|preliminary|provisional|tentative|unconfirmed|"
    r"background|context only|low confidence|medium confidence|"
    r"adjacent|partially applicable)\b",
    re.IGNORECASE,
)
APPLICABILITY_QUALIFICATION_PATTERN = re.compile(
    r"\b(?:partial(?:ly)? applicable|partial applicability|adjacent|"
    r"context only|supported scope|applies only|not specific to|"
    r"does not specifically|cannot be transferred|corroborated "
    r"practitioner)\b",
    re.IGNORECASE,
)
EVIDENCE_LIMITATION_PATTERN = re.compile(
    r"\b(?:gap|insufficient|missing|not researched|not established|"
    r"unresolved|unknown|could not verify|no verified evidence|"
    r"no verified urls?|not matched|unverified|precludes|cannot support|"
    r"required scope anchors?|incomplete|scope_missing|scrape_failure|"
    r"compression_empty|evidence limitation|uncited synthesis|"
    r"not independently verified|lacks?|lacking|no direct evidence|"
    r"provisional|tentative|partial(?:ly)? applicable|"
    r"not synthesis-ready|not wholly synthesis-ready|supported scope)\b",
    re.IGNORECASE,
)


def _normalized_citation_url(url: str) -> str:
    return str(url or "").strip().rstrip("/").lower()


def report_citation_urls(report: str) -> set[str]:
    """Return ordinary Markdown-link targets, excluding embedded images."""
    return {
        match.group(2).strip()
        for match in MARKDOWN_LINK_PATTERN.finditer(report or "")
    }


def enforce_verified_citation_urls(
    report: str,
    verified_source_urls: list[str],
) -> str:
    """Ensure the final report cannot cite URLs outside verified evidence."""
    allowed = {
        _normalized_citation_url(url): str(url).strip()
        for url in verified_source_urls
        if str(url).strip()
    }
    by_domain: dict[str, list[str]] = {}
    for original in allowed.values():
        domain = urlparse(original).netloc.lower().removeprefix("www.")
        if domain:
            by_domain.setdefault(domain, []).append(original)

    def replace_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        normalized = _normalized_citation_url(url)
        if normalized in allowed:
            return f"[{label}]({allowed[normalized]})"
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        same_domain = list(dict.fromkeys(by_domain.get(domain, [])))
        if len(same_domain) == 1:
            # The claim came from a verified page on this domain, but the model
            # followed an unverified link embedded inside that page.
            return f"[{label}]({same_domain[0]})"
        return label

    # Rebuild the references section from the allowlist so wrapped or
    # hallucinated reference entries cannot survive a regex-only cleanup.
    body = REFERENCE_HEADING_PATTERN.split(report or "", maxsplit=1)[0].rstrip()
    body = MARKDOWN_LINK_PATTERN.sub(replace_link, body)
    stored_links: list[str] = []

    def store_link(match: re.Match[str]) -> str:
        stored_links.append(match.group(0))
        return f"__GPTR_VERIFIED_LINK_{len(stored_links) - 1}__"

    body = MARKDOWN_LINK_PATTERN.sub(store_link, body)

    def replace_bare_url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,:;!?")
        normalized = _normalized_citation_url(url)
        if normalized in allowed:
            return f"[source]({allowed[normalized]})"
        return ""

    body = BARE_URL_PATTERN.sub(replace_bare_url, body)
    for index, link in enumerate(stored_links):
        body = body.replace(f"__GPTR_VERIFIED_LINK_{index}__", link)
    if not allowed:
        return body
    references = "\n".join(
        f"- [{url}]({url})" for url in sorted(allowed.values())
    )
    return f"{body}\n\n## References\n\n{references}\n"


def qualify_supplemental_evidence_paragraphs(
    report: str,
    coverage_ledger: list[dict[str, Any]] | None = None,
) -> str:
    """Label cited provisional/background evidence at its point of use.

    The report writer receives the same evidence labels, but a deterministic
    last pass is still necessary because a model can preserve a supplemental
    citation while dropping its qualifier during drafting or correction.
    """
    evidence_by_url: dict[str, dict[str, Any]] = {}
    for aspect in coverage_ledger or []:
        for source in aspect.get("evidence_pool_sources", []) or []:
            status = str(source.get("claim_status") or "").strip().lower()
            applicability = str(
                source.get("applicability") or "exact"
            ).strip().lower()
            role = str(
                source.get("evidence_role") or ""
            ).strip().lower()
            normalized = _normalized_citation_url(source.get("url"))
            if normalized and (
                status in {"provisional", "background"}
                or applicability in {"partial", "adjacent", "context"}
                or role == "practitioner"
            ):
                evidence_by_url[normalized] = source
    if not evidence_by_url:
        return report

    reference_match = REFERENCE_HEADING_PATTERN.search(report or "")
    if reference_match:
        body = (report or "")[: reference_match.start()].rstrip()
        references = (report or "")[reference_match.start() :].lstrip()
    else:
        body = report or ""
        references = ""

    def conservative_label(
        sources: list[dict[str, Any]],
    ) -> str:
        statuses = {
            str(source.get("claim_status") or "").strip().lower()
            for source in sources
        }
        roles = {
            str(source.get("evidence_role") or "").strip().lower()
            for source in sources
        }
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        confidence = min(
            (
                str(source.get("confidence_label") or "low")
                .strip()
                .lower()
                for source in sources
            ),
            key=lambda value: confidence_order.get(value, 0),
            default="low",
        )
        applicability_order = {
            "unrelated": 0,
            "context": 1,
            "adjacent": 2,
            "partial": 3,
            "exact": 4,
        }
        applicability = min(
            (
                str(source.get("applicability") or "context")
                .strip()
                .lower()
                for source in sources
            ),
            key=lambda value: applicability_order.get(value, 0),
            default="context",
        )
        if statuses == {"provisional"}:
            label = (
                "provisional practitioner evidence"
                if roles == {"practitioner"}
                else "provisional supplemental evidence"
            )
            qualifiers = ["tentative"]
        elif statuses == {"background"}:
            label = (
                "background practitioner context"
                if roles == {"practitioner"}
                else "background or contextual evidence"
            )
            qualifiers = ["not synthesis-ready"]
        elif statuses == {"synthesis_ready"}:
            if roles == {"practitioner"}:
                label = "corroborated practitioner evidence"
            elif applicability == "partial":
                label = "partially applicable evidence"
            elif applicability in {"adjacent", "context"}:
                label = "adjacent or contextual evidence"
            else:
                label = "qualified evidence"
            qualifiers = ["synthesis-ready only within its supported scope"]
        elif "synthesis_ready" in statuses:
            label = "mixed verified and supplemental evidence"
            qualifiers = ["not wholly synthesis-ready"]
        else:
            label = "mixed provisional and background evidence"
            qualifiers = ["not synthesis-ready"]
        scopes = list(
            dict.fromkeys(
                str(value).strip()
                for source in sources
                for value in (
                    source.get("supported_scope")
                    or source.get("supported_entities")
                    or []
                )
                if str(value).strip()
            )
        )
        qualifiers.extend(
            [f"{confidence} confidence", f"{applicability} applicability"]
        )
        if scopes:
            qualifiers.append("scope: " + ", ".join(scopes[:3]))
        return f"*Evidence label: {label}; {'; '.join(qualifiers)}*"

    blocks = re.split(r"(\n\s*\n)", body)
    for index in range(0, len(blocks), 2):
        paragraph = blocks[index]
        if (
            not paragraph.strip()
            or paragraph.lstrip().startswith(("#", "```", ">", "|"))
        ):
            continue
        cited_sources = [
            evidence_by_url[normalized]
            for normalized in {
                _normalized_citation_url(url)
                for url in report_citation_urls(paragraph)
            }
            if normalized in evidence_by_url
        ]
        if not cited_sources:
            continue
        qualification_text = MARKDOWN_LINK_PATTERN.sub(
            lambda match: match.group(1),
            paragraph,
        )
        statuses = {
            str(source.get("claim_status") or "").strip().lower()
            for source in cited_sources
        }
        roles = {
            str(source.get("evidence_role") or "").strip().lower()
            for source in cited_sources
        }
        nonexact = any(
            str(source.get("applicability") or "exact").strip().lower()
            in {"partial", "adjacent", "context"}
            for source in cited_sources
        )
        if statuses <= {"provisional", "background"} and (
            SUPPLEMENTAL_QUALIFICATION_PATTERN.search(qualification_text)
        ):
            continue
        if statuses == {"synthesis_ready"} and nonexact and (
            APPLICABILITY_QUALIFICATION_PATTERN.search(qualification_text)
        ):
            continue
        if (
            statuses == {"synthesis_ready"}
            and not nonexact
            and roles == {"practitioner"}
            and SUPPLEMENTAL_QUALIFICATION_PATTERN.search(qualification_text)
        ):
            continue
        blocks[index] = (
            f"{paragraph.rstrip()} {conservative_label(cited_sources)}"
        )

    qualified = "".join(blocks).rstrip()
    if references:
        return f"{qualified}\n\n{references}"
    return qualified


def qualify_uncited_synthesis_paragraphs(report: str) -> str:
    """Mark substantive uncited synthesis locally instead of relying on a banner."""
    reference_match = REFERENCE_HEADING_PATTERN.search(report or "")
    if reference_match:
        body = (report or "")[: reference_match.start()].rstrip()
        references = (report or "")[reference_match.start() :].lstrip()
    else:
        body = report or ""
        references = ""
    blocks = re.split(r"(\n\s*\n)", body)
    for index in range(0, len(blocks), 2):
        paragraph = blocks[index]
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if (
            len(normalized) < 80
            or paragraph.lstrip().startswith(("#", "```", ">", "|"))
            or INLINE_LINK_PATTERN.search(paragraph)
            or EVIDENCE_LIMITATION_PATTERN.search(paragraph)
            or paragraph.lstrip().startswith("*Evidence label:")
        ):
            continue
        blocks[index] = (
            f"{paragraph.rstrip()} *Evidence label: uncited synthesis; "
            "not independently verified*"
        )
    qualified = "".join(blocks).rstrip()
    if references:
        return f"{qualified}\n\n{references}"
    return qualified


def report_quality_diagnostics(
    report: str,
    coverage_ledger: list[dict[str, Any]] | None = None,
    *,
    query: str = "",
    supplemental_source_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Check report claims against coverage state and inline-citation rules."""
    coverage_ledger = coverage_ledger or []
    supplemental_urls = {
        _normalized_citation_url(url)
        for url in (supplemental_source_urls or [])
        if _normalized_citation_url(url)
    }
    parts = REFERENCE_HEADING_PATTERN.split(report or "", maxsplit=1)
    body = parts[0].strip()
    references = parts[1] if len(parts) > 1 else ""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", body)
        if paragraph.strip()
        and not paragraph.lstrip().startswith(("#", "```", ">"))
        and not paragraph.lstrip().startswith("|")
        and len(re.sub(r"\s+", " ", paragraph)) >= 80
    ]
    has_verified_sources = any(
        item.get("verified_urls") for item in coverage_ledger
    )
    unresolved = [
        str(item.get("aspect_id") or item.get("question") or "unknown")
        for item in coverage_ledger
        if item.get("state") != "evidence_ready"
    ]
    fallback_only = any(
        int((item.get("source_tiers") or {}).get("fallback") or 0) > 0
        and int((item.get("source_tiers") or {}).get("primary") or 0) == 0
        and int((item.get("source_tiers") or {}).get("reputable") or 0) == 0
        for item in coverage_ledger
    )
    body_lower = body.lower()
    references_only = (
        not INLINE_LINK_PATTERN.search(body)
        and bool(INLINE_LINK_PATTERN.search(references))
    )
    unsupported_comprehensive = bool(
        unresolved
        and re.search(r"\b(?:comprehensive|complete|fully researched)\b", body_lower)
    )
    unsupported_primary = bool(
        fallback_only
        and re.search(
            r"\b(?:all|exclusively|entirely)\s+(?:verified\s+)?primary\b"
            r"|\bprimary[- ]source(?: coverage| evidence)?\b",
            body_lower,
        )
    )
    evidence_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not EVIDENCE_LIMITATION_PATTERN.search(paragraph)
    ]
    cited_paragraphs = [
        paragraph
        for paragraph in evidence_paragraphs
        if INLINE_LINK_PATTERN.search(paragraph)
    ]
    citation_rate = (
        len(cited_paragraphs) / len(evidence_paragraphs)
        if evidence_paragraphs and has_verified_sources
        else 1.0
    )
    unsupported_scope_claims: list[str] = []
    for item in coverage_ledger:
        if item.get("state") == "evidence_ready":
            continue
        anchors = item.get("missing_scope_anchors") or []
        for anchor in anchors:
            normalized = " ".join(str(anchor).lower().split())
            if not normalized:
                continue
            for paragraph in paragraphs:
                paragraph_lower = paragraph.lower()
                if (
                    normalized in paragraph_lower
                    and not EVIDENCE_LIMITATION_PATTERN.search(paragraph)
                ):
                    unsupported_scope_claims.append(str(anchor))
                    break
    unlabeled_supplemental_paragraphs = []
    qualified_evidence_by_url: dict[str, dict[str, Any]] = {}
    for item in coverage_ledger:
        for source in item.get("evidence_pool_sources", []) or []:
            normalized = _normalized_citation_url(source.get("url"))
            status = str(source.get("claim_status") or "").strip().lower()
            applicability = str(
                source.get("applicability") or "exact"
            ).strip().lower()
            role = str(
                source.get("evidence_role") or ""
            ).strip().lower()
            if normalized and status == "synthesis_ready" and (
                applicability in {"partial", "adjacent", "context"}
                or role == "practitioner"
            ):
                qualified_evidence_by_url[normalized] = source
    unlabeled_applicability_paragraphs = []
    for paragraph in paragraphs:
        cited_urls = {
            _normalized_citation_url(url)
            for url in report_citation_urls(paragraph)
        }
        qualification_text = MARKDOWN_LINK_PATTERN.sub(
            lambda match: match.group(1),
            paragraph,
        )
        if (
            cited_urls & supplemental_urls
            and not SUPPLEMENTAL_QUALIFICATION_PATTERN.search(
                qualification_text
            )
        ):
            unlabeled_supplemental_paragraphs.append(
                re.sub(r"\s+", " ", paragraph)[:240]
            )
        cited_qualified_sources = [
            qualified_evidence_by_url[url]
            for url in cited_urls
            if url in qualified_evidence_by_url
        ]
        if not cited_qualified_sources:
            continue
        needs_applicability_label = any(
            str(source.get("applicability") or "exact").strip().lower()
            in {"partial", "adjacent", "context"}
            for source in cited_qualified_sources
        )
        needs_practitioner_label = any(
            str(source.get("evidence_role") or "").strip().lower()
            == "practitioner"
            for source in cited_qualified_sources
        )
        applicability_labeled = (
            not needs_applicability_label
            or APPLICABILITY_QUALIFICATION_PATTERN.search(
                qualification_text
            )
        )
        practitioner_labeled = (
            not needs_practitioner_label
            or SUPPLEMENTAL_QUALIFICATION_PATTERN.search(
                qualification_text
            )
        )
        if not applicability_labeled or not practitioner_labeled:
            unlabeled_applicability_paragraphs.append(
                re.sub(r"\s+", " ", paragraph)[:240]
            )
    # Preserve the existing high-confidence taxonomy guard while the v2
    # evidence judge handles arbitrary requested relationships. This is a
    # compatibility safety net, not part of retrieval ranking.
    category_error = False
    predator_immunity_overclaim = False
    if re.search(r"\bpredators?\b", query, re.IGNORECASE):
        category_error = any(
            re.search(
                r"\b(?:disease|parasite|pathogen|microb|pollution|bycatch|plastic|"
                r"climate|temperature|storm)\w*\b|\bhabitat loss\b",
                sentence,
            )
            and re.search(r"\b(?:predator|predation|prey)\w*\b", sentence)
            and not re.search(r"\b(?:not|separate|distinct|unlike)\b", sentence)
            for sentence in re.split(r"(?<=[.!?])\s+", body_lower)
        )
        predator_immunity_overclaim = bool(
            re.search(
                r"\b(?:immune|immunity)\s+(?:to|from|against)\s+predat",
                body_lower,
            )
        )
    issues = []
    if references_only:
        issues.append("citations appear only in the references section")
    if has_verified_sources and citation_rate < 0.9:
        issues.append(
            f"only {len(cited_paragraphs)}/{len(evidence_paragraphs)} substantive evidence paragraphs have inline citations"
        )
    if unsupported_comprehensive:
        issues.append("report claims comprehensive coverage despite unresolved aspects")
    if unsupported_primary:
        issues.append("report overstates primary-source coverage")
    if unsupported_scope_claims:
        issues.append(
            "report presents unresolved scope as established: "
            + ", ".join(sorted(set(unsupported_scope_claims)))
        )
    if unlabeled_supplemental_paragraphs:
        issues.append(
            "supplemental evidence is presented without an explicit "
            "provisional, practitioner, background, or applicability label"
        )
    if unlabeled_applicability_paragraphs:
        issues.append(
            "partial, adjacent, or practitioner evidence is presented "
            "without its applicability or provenance label"
        )
    if category_error:
        issues.append(
            "report may classify disease, parasites, microbes, or "
            "environmental hazards as predators"
        )
    if predator_immunity_overclaim:
        issues.append(
            "report overstates a defense as immunity to predators"
        )
    return {
        "passes": not issues,
        "issues": issues,
        "body_citation_count": len(INLINE_LINK_PATTERN.findall(body)),
        "substantive_paragraph_count": len(evidence_paragraphs),
        "cited_substantive_paragraph_count": len(cited_paragraphs),
        "inline_citation_rate": round(citation_rate, 3),
        "references_only": references_only,
        "unresolved_aspects": unresolved,
        "unsupported_comprehensive_claim": unsupported_comprehensive,
        "unsupported_primary_claim": unsupported_primary,
        "unsupported_scope_claims": sorted(set(unsupported_scope_claims)),
        "unlabeled_supplemental_paragraphs": (
            unlabeled_supplemental_paragraphs
        ),
        "unlabeled_applicability_paragraphs": (
            unlabeled_applicability_paragraphs
        ),
        "category_error": category_error,
        "predator_immunity_overclaim": predator_immunity_overclaim,
    }


async def repair_report_evidence_safety(
    *,
    report: str,
    query: str,
    coverage_ledger: list[dict[str, Any]],
    verified_source_urls: list[str],
    supplemental_source_urls: list[str] | None = None,
    diagnostics: dict[str, Any],
    cfg: Config,
    cost_callback: callable = None,
    **kwargs,
) -> str:
    """Run one bounded corrective pass without introducing new evidence."""
    prompt = f"""Correct only the evidence-safety problems in this report.

User query: {query}
Problems: {diagnostics.get("issues", [])}
Coverage ledger: {json.dumps(coverage_ledger, ensure_ascii=False)}
Verified URL allowlist: {json.dumps(verified_source_urls)}
Supplemental labeled URL allowlist: {json.dumps(supplemental_source_urls or [])}

Rules:
- Preserve supported content, but remove or narrow unsupported claims.
- State unqualified facts only from synthesis-ready evidence in aspects whose
  state is evidence_ready.
- Supplemental provisional evidence may be retained only as an explicitly
  attributed practitioner report, tentative lead, or partially applicable
  observation. Background evidence may orient the reader but may not answer
  an unresolved aspect.
- Synthesis-ready evidence whose applicability is partial, adjacent, or
  context-only must retain that applicability label. Do not transfer a
  numeric setting or workflow from its supported scope to another model,
  version, population, region, or operating mode.
- State unresolved scope as an explicit evidence limitation.
- Do not call fallback sources primary.
- Do not generalize beyond matched scope anchors.
- Preserve the relationship and category asked by the user. Keep adjacent
  causes, effects, risks, mitigations, and correlations separate unless
  judge-approved evidence explicitly supports the classification.
- Keep each supported claim attached to its supported_entities. Attribute a
  feature, workflow, or setting only to the tool, interface, product, version,
  or method named by that claim. Transfer between subjects only when a direct
  evidence claim names that relationship.
- Put an inline Markdown citation at the end of every substantive evidence
  paragraph, using only exact URLs from the allowlist.
- Return only the corrected Markdown report.

REPORT:
{report}"""
    return await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evidence and citation editor.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        llm_provider=cfg.smart_llm_provider,
        stream=False,
        websocket=None,
        max_tokens=cfg.smart_token_limit,
        llm_kwargs=cfg.llm_kwargs,
        cost_callback=cost_callback,
        **kwargs,
    )


def add_visible_evidence_limitation(
    report: str,
    diagnostics: dict[str, Any],
) -> str:
    """Narrow unsafe confidence deterministically when correction still fails."""
    narrowed = re.sub(
        r"\bcomprehensive\b",
        "bounded",
        report or "",
        flags=re.IGNORECASE,
    )
    unresolved = diagnostics.get("unresolved_aspects") or []
    issue_text = "; ".join(diagnostics.get("issues") or [])
    issue_text = re.sub(
        r"\bcomprehensive\b",
        "complete",
        issue_text,
        flags=re.IGNORECASE,
    )
    limitation = (
        "> **Evidence limitation:** "
        + (
            f"Coverage remains unresolved for {', '.join(unresolved)}. "
            if unresolved
            else ""
        )
        + (issue_text or "Some claims could not be fully citation-validated.")
    )
    if narrowed.lstrip().startswith("> **Evidence limitation:**"):
        return narrowed
    return f"{limitation}\n\n{narrowed}"


async def write_report_introduction(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> str:
    """
    Generate an introduction for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        prompt_family: Family of prompts

    Returns:
        str: The generated introduction.
    """
    try:
        introduction = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": prompt_family.generate_report_introduction(
                    question=query,
                    research_summary=context,
                    language=config.language
                )},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return introduction
    except Exception as e:
        logger.error(f"Error in generating report introduction: {e}")
    return ""


async def write_conclusion(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> str:
    """
    Write a conclusion for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        prompt_family: Family of prompts

    Returns:
        str: The generated conclusion.
    """
    try:
        conclusion = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {
                    "role": "user",
                    "content": prompt_family.generate_report_conclusion(query=query,
                                                                        report_content=context,
                                                                        language=config.language),
                },
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return conclusion
    except Exception as e:
        logger.error(f"Error in writing conclusion: {e}")
    return ""


async def summarize_url(
    url: str,
    content: str,
    role: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    **kwargs
) -> str:
    """
    Summarize the content of a URL.

    Args:
        url (str): The URL to summarize.
        content (str): The content of the URL.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.

    Returns:
        str: The summarized content.
    """
    try:
        summary = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"Summarize the following content from {url}:\n\n{content}"},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return summary
    except Exception as e:
        logger.error(f"Error in summarizing URL: {e}")
    return ""


async def generate_draft_section_titles(
    query: str,
    current_subtopic: str,
    context: str,
    role: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> List[str]:
    """
    Generate draft section titles for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        prompt_family: Family of prompts

    Returns:
        List[str]: A list of generated section titles.
    """
    try:
        section_titles = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": prompt_family.generate_draft_titles_prompt(
                    current_subtopic, query, context)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=None,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return section_titles.split("\n")
    except Exception as e:
        logger.error(f"Error in generating draft section titles: {e}")
    return []


async def generate_report(
    query: str,
    context,
    agent_role_prompt: str,
    report_type: str,
    tone: Tone,
    report_source: str,
    websocket,
    cfg,
    main_topic: str = "",
    existing_headers: list = [],
    relevant_written_contents: list = [],
    cost_callback: callable = None,
    custom_prompt: str = "", # This can be any prompt the user chooses with the context
    headers=None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    available_images: list = None,
    verified_source_urls: list[str] | None = None,
    supplemental_source_urls: list[str] | None = None,
    coverage_ledger: list[dict[str, Any]] | None = None,
    **kwargs
):
    """
    generates the final report
    Args:
        query:
        context:
        agent_role_prompt:
        report_type:
        websocket:
        tone:
        cfg:
        main_topic:
        existing_headers:
        relevant_written_contents:
        cost_callback:
        prompt_family: Family of prompts
        available_images: Pre-generated images to embed in the report

    Returns:
        report:

    """
    available_images = available_images or []
    verified_source_urls = [
        str(url).strip()
        for url in (verified_source_urls or [])
        if str(url).strip()
    ]
    supplemental_source_urls = [
        str(url).strip()
        for url in (supplemental_source_urls or [])
        if str(url).strip() and str(url).strip() not in verified_source_urls
    ]
    coverage_ledger = coverage_ledger or []
    generate_prompt = get_prompt_by_report_type(report_type, prompt_family)
    report = ""

    if report_type == "subtopic_report":
        content = f"{generate_prompt(query, existing_headers, relevant_written_contents, main_topic, context, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words, language=cfg.language)}"
    elif custom_prompt:
        content = f"{custom_prompt}\n\nContext: {context}"
    else:
        content = f"{generate_prompt(query, context, report_source, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words, language=cfg.language)}"

    if verified_source_urls:
        content += f"""

VERIFIED CITATION URL ALLOWLIST:
{verified_source_urls}

Citation safety requirements:
- For synthesis-ready factual claims, cite only the exact URLs in this
  verified allowlist.
- URLs appearing inside source-page content are not verified sources and must
  not be cited unless they also appear in one of the explicit allowlists.
- Do not invent, expand, shorten, or substitute citation URLs.
"""
    if supplemental_source_urls:
        content += f"""

LABELED SUPPLEMENTAL EVIDENCE URL ALLOWLIST:
{supplemental_source_urls}

Supplemental-evidence requirements:
- These pages passed content integrity but are not synthesis-ready facts.
- Use only the exact supported_claims recorded for them in the coverage
  ledger. Do not infer additional facts from their titles or URLs.
- Every use must be explicitly attributed and labeled as practitioner,
  provisional, tentative, background, adjacent, or partially applicable,
  matching the ledger's claim_status, evidence_strength, confidence_label,
  and applicability.
- Supplemental evidence cannot resolve an aspect, erase a missing scope
  anchor, support a broad conclusion, or be presented as established fact.
- Cite the exact supplemental URL in the same paragraph.
"""
    if coverage_ledger:
        content += f"""

AUTHORITATIVE COVERAGE LEDGER:
{json.dumps(coverage_ledger, ensure_ascii=False)}

Evidence-safety requirements:
- State unqualified factual content only from synthesis-ready evidence in
  aspects whose state is evidence_ready.
- Provisional evidence may appear only as a clearly attributed and labeled
  observation. Background evidence may explain context but may not answer an
  unresolved aspect.
- Label synthesis-ready evidence as partially applicable or adjacent whenever
  the ledger does. State its supported scope and do not transfer numeric
  settings or workflows to an unmentioned model, version, population, region,
  or operating mode.
- State unresolved aspects and missing scope anchors as evidence limitations.
- Never call fallback evidence primary evidence.
- Do not generalize beyond matched_scope_anchors.
- Do not claim the report is comprehensive when any aspect is unresolved.
- Preserve the relationship and category asked by the user. Keep adjacent
  causes, effects, risks, mitigations, and correlations separate unless
  judge-approved evidence explicitly supports the classification.
- Keep each supported claim attached to its supported_entities. Attribute a
  feature, workflow, or setting only to the tool, interface, product, version,
  or method named by that claim. Transfer between subjects only when a direct
  evidence claim names that relationship.
- Every substantive evidence paragraph must end with an inline citation.
"""
    
    # Add available images instruction if images were pre-generated
    if available_images:
        images_info = "\n".join([
            f"- Image {i+1}: ![{img.get('title', img.get('alt_text', 'Illustration'))}]({img['url']}) - {img.get('section_hint', 'General')}"
            for i, img in enumerate(available_images)
        ])
        content += f"""

AVAILABLE IMAGES:
You have the following pre-generated images available. Embed them in relevant sections of your report using the exact markdown syntax provided:

{images_info}

Place each image on its own line after the relevant section header or paragraph. Use all available images where they add value to the content."""
    try:
        report = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": content},
            ],
            temperature=0.35,
            llm_provider=cfg.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=cfg.smart_token_limit,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
    except Exception:
        try:
            report = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[
                    {"role": "user", "content": f"{agent_role_prompt}\n\n{content}"},
                ],
                temperature=0.35,
                llm_provider=cfg.smart_llm_provider,
                stream=True,
                websocket=websocket,
                max_tokens=cfg.smart_token_limit,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )
        except Exception as e:
            print(f"Error in generate_report: {e}")

    return report
