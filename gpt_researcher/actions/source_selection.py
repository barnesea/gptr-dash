"""Small, bounded source selection before expensive web scraping."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any
import unicodedata
from urllib.parse import unquote, urlparse


LOW_VALUE_HOST_MARKERS = (
    "dictionary.", "merriam-webster.", "vocabulary.com", "thesaurus.",
    "diffchecker.", "textcompare.", "text-compare.", "draftable.",
    # Deep research may use a broader-web fallback, never generic social or
    # self-published aggregation pages as evidence.
    "medium.com", "linkedin.com", "facebook.com", "instagram.com",
    "youtube.com", "youtu.be", "twitter.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com",
    "articsledge.com", "dev.to", "substack.com", "hashnode.",
)
LOW_VALUE_EXACT_HOSTS = ("x.com",)
NAMED_SOCIAL_HOSTS = {
    "x.com": (" x ", "twitter"),
    "twitter.com": ("twitter", " x "),
    "facebook.com": ("facebook",),
    "instagram.com": ("instagram",),
    "youtube.com": ("youtube",),
    "reddit.com": ("reddit",),
    "tiktok.com": ("tiktok",),
    "linkedin.com": ("linkedin",),
}
GENERIC_QUERY_ANCHORS = {
    "about", "after", "affect", "application", "applications", "changes",
    "developers", "does", "from", "have", "https", "http", "linux", "model",
    "models", "project", "research", "support", "what", "when", "which", "with",
    # Organization and hosting-platform names are useful quality priors, but
    # are not enough to prove that a page discusses the requested product.
    "anthropic", "github", "google", "huggingface", "microsoft", "nvidia",
    "openai",
}
PRIMARY_HOST_MARKERS = (
    ".gov", "docs.", "developer.", "github.com", "arxiv.org",
    "aclanthology.org", "doi.org", "openreview.net", "standards.", "ietf.org",
    "w3.org", "kernel.org", "pmc.ncbi.nlm.nih.gov",
    "royalsocietypublishing.org", "journals.plos.org",
    "frontiersin.org", "nature.com", "springer.com", "wiley.com",
    "sciencedirect.com", "tandfonline.com",
)
PRIMARY_HOSTS = (
    "tensorflow.org", "pytorch.org", "huggingface.co", "openai.com",
    "anthropic.com", "cohere.com", "jina.ai", "nvidia.com", "elastic.co",
    "deepset.ai", "qdrant.tech", "weaviate.io", "microsoft.com",
    "google.com",
    "azure.com", "sbert.net",
)
REPUTABLE_TECH_HOST_MARKERS = (
    "lwn.net", "spectrum.ieee.org", "ieeexplore.ieee.org", "dl.acm.org",
    "acm.org", "oreilly.com", "infoq.com", "arstechnica.com",
)
REPUTABLE_GENERAL_HOST_MARKERS = (
    "si.edu", "smithsonian", "museum", "aquarium", "university",
    "nationalgeographic.com", "bbc.com", "bbc.co.uk", "reuters.com",
    "apnews.com", "seaturtlestatus.org", "conserveturtles.org",
    "turtle-foundation.org", "iucn.org", "wwf.org",
)
SUSPICIOUS_SCRAPED_TITLES = (
    "set your api key", "access denied", "just a moment", "sign in", "signin",
    "log in", "login", "page not found", "not found", "error", "captcha",
    "enable javascript", "checking your browser",
)
AUTH_PATH_SEGMENTS = {
    "account", "accounts", "auth", "login", "oauth", "session", "signin",
    "signup",
}
AUTH_HOST_PREFIXES = ("account.", "accounts.", "auth.", "login.")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
URL_PATTERN = re.compile(r"https?://[^\s\]\)>\",;]+", re.IGNORECASE)
PREDATION_QUERY_PATTERN = re.compile(
    r"\b(?:predators?|predation|prey|what eats|natural enemies)\b",
    re.IGNORECASE,
)
PREDATION_EVIDENCE_PATTERN = re.compile(
    r"\b(?:predators?|predation|prey|eats?|eaten|feeds? on|natural enemies)\b",
    re.IGNORECASE,
)
TURTLE_TARGET_PATTERN = re.compile(
    r"\b(?:sea turtles?|freshwater turtles?|terrestrial turtles?|"
    r"land turtles?|box turtles?|tortoises?|turtle (?:eggs?|nests?|"
    r"hatchlings?|juveniles?|adults?))\b",
    re.IGNORECASE,
)
CONCRETE_PREDATION_PATTERN = re.compile(
    r"(?:"
    r"\bpredators?\b.{0,100}\b(?:include|including|such as|are|is|:)\b"
    r"|\b(?:include|including|such as)\b.{0,100}\bpredators?\b"
    r"|\b(?:preyed (?:on|upon) by|eaten by)\b"
    r"|\b(?:preys? (?:on|upon)|eats?|feeds? on)\b"
    r"|\b(?:is|are)\b.{0,50}\b(?:turtle )?predators?\b"
    r")",
    re.IGNORECASE,
)


def source_url(candidate: dict[str, Any]) -> str:
    return str(candidate.get("href") or candidate.get("url") or "").strip()


def source_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def extract_query_urls(query: str) -> list[str]:
    """Return distinct explicit HTTP(S) URLs in first-seen order."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in URL_PATTERN.findall(query or ""):
        url = match.rstrip(".,:;!?")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def canonical_source_alternatives(url: str) -> list[str]:
    """Return bounded canonical/raw/PDF variants after a selected URL fails."""
    parsed = urlparse(url)
    domain = source_domain(url)
    path = parsed.path
    alternatives: list[str] = []
    if domain == "github.com" and "/blob/" in path:
        before, after = path.split("/blob/", 1)
        alternatives.append(
            f"https://raw.githubusercontent.com{before}/{after}"
        )
    elif domain == "huggingface.co" and "/blob/" in path:
        alternatives.append(url.replace("/blob/", "/resolve/", 1))
    elif domain == "arxiv.org" and path.startswith("/abs/"):
        identifier = path.removeprefix("/abs/").strip("/")
        if identifier:
            alternatives.extend(
                [
                    f"https://arxiv.org/html/{identifier}",
                    f"https://arxiv.org/pdf/{identifier}",
                ]
            )
    seen: set[str] = set()
    return [
        candidate
        for candidate in alternatives
        if candidate != url and not (candidate in seen or seen.add(candidate))
    ][:2]


def _visible_url_text(url: str) -> str:
    """Return relevance-bearing URL text without attacker-controlled query data."""
    parsed = urlparse(url)
    return unquote(f"{parsed.netloc} {parsed.path}")


def _looks_like_auth_or_account_page(candidate: dict[str, Any]) -> bool:
    url = source_url(candidate)
    parsed = urlparse(url)
    domain = source_domain(url)
    path_segments = [segment.lower() for segment in parsed.path.split("/") if segment]
    title = str(candidate.get("title") or "").strip().lower()
    return (
        any(domain.startswith(prefix) for prefix in AUTH_HOST_PREFIXES)
        or (len(path_segments) == 1 and path_segments[0] in AUTH_PATH_SEGMENTS)
        or any(marker in title for marker in ("sign in", "log in", "login to your account"))
    )


def source_quality_tier(
    candidate: dict[str, Any],
    query: str = "",
) -> str:
    """Classify a result card without trusting its prose or model judgment."""
    domain = source_domain(source_url(candidate))
    internal_tier = str(candidate.get("_gptr_source_tier") or "")
    if internal_tier in {"primary", "reputable", "fallback", "reject"}:
        return internal_tier
    padded_query = f" {str(query).lower()} "
    for social_host, names in NAMED_SOCIAL_HOSTS.items():
        if (
            domain == social_host or domain.endswith(f".{social_host}")
        ) and any(name in padded_query for name in names):
            return "primary"
    if (
        not domain
        or any(marker in domain for marker in LOW_VALUE_HOST_MARKERS)
        or any(
            domain == host or domain.endswith(f".{host}")
            for host in LOW_VALUE_EXACT_HOSTS
        )
        or _looks_like_auth_or_account_page(candidate)
    ):
        return "reject"
    if any(marker in domain for marker in PRIMARY_HOST_MARKERS) or any(
        domain == host or domain.endswith(f".{host}") for host in PRIMARY_HOSTS
    ):
        return "primary"
    if any(
        marker in domain
        for marker in (
            *REPUTABLE_TECH_HOST_MARKERS,
            *REPUTABLE_GENERAL_HOST_MARKERS,
        )
    ) or domain.endswith(".edu"):
        return "reputable"
    return "fallback"


def post_scrape_integrity_reason(
    query: str, candidate: dict[str, Any] | None, scraped: dict[str, Any]
) -> str | None:
    """Return why a fetched page must not enter research context, if applicable.

    Search cards are only a promise about a page.  A crawler can instead receive
    an auth wall, an error page, or unrelated content.  Do this inexpensive
    verification before the page is registered as a research source.
    """
    raw_content = str(scraped.get("raw_content") or "").strip()
    title = str(scraped.get("title") or "").strip().lower()
    if not raw_content:
        return "post-scrape reject: empty content"
    if any(marker in title for marker in SUSPICIOUS_SCRAPED_TITLES):
        return "post-scrape reject: fetched title indicates an auth wall, error page, or placeholder"

    # The fetched content must still visibly relate to the research question.
    # This catches redirects and generic landing pages without requiring an LLM.
    content_card = {
        "url": scraped.get("url", ""),
        "title": scraped.get("title", ""),
        # Scholarly pages often put the abstract after a long metadata/header
        # block. This remains a bounded lexical check, but samples enough of the
        # fetched page to avoid discarding a valid paper before compression.
        "body": raw_content[:50000],
    }
    if not has_meaningful_query_anchor(query, content_card):
        return "post-scrape reject: fetched page has no meaningful query-anchor coverage"
    if (
        PREDATION_QUERY_PATTERN.search(query or "")
        and not PREDATION_EVIDENCE_PATTERN.search(
            f"{scraped.get('title', '')} {raw_content}"
        )
    ):
        return (
            "post-scrape reject: fetched page covers the subject but not "
            "the requested predation relationship"
        )

    # A SERP title and a fetched title normally share at least one distinctive
    # term.  Do not reject pages with no usable title (PDFs often lack one), but
    # reject a wholly unrelated fetched title when the original card was clear.
    if candidate:
        expected_title = str(candidate.get("title") or "").strip()
        if expected_title and title:
            expected_tokens = _anchor_tokens(expected_title) - GENERIC_QUERY_ANCHORS
            fetched_tokens = _anchor_tokens(title) - GENERIC_QUERY_ANCHORS
            if len(expected_tokens) >= 2 and not (expected_tokens & fetched_tokens):
                return "post-scrape reject: fetched title does not match the selected search result"
    return None


def source_card(candidate: dict[str, Any], source_id: int) -> dict[str, Any]:
    """Return the compact, model-safe fields used for source judgment."""
    url = source_url(candidate)
    return {
        "id": source_id,
        "url": url,
        "domain": source_domain(url),
        "title": str(candidate.get("title") or "").strip()[:240],
        "snippet": str(
            candidate.get("body")
            or candidate.get("snippet")
            or candidate.get("raw_content")
            or candidate.get("content")
            or ""
        ).strip()[:700],
        "engine": str(candidate.get("engine") or "").strip()[:80],
        "date": str(candidate.get("date") or "").strip()[:80],
        "source_tier": source_quality_tier(candidate),
    }


def _anchor_tokens(query: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", query or "")
    normalized = CAMEL_BOUNDARY.sub(" ", normalized)
    return {
        token
        for token in TOKEN_PATTERN.findall(normalized.lower())
        if len(token) >= 4
    }


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(f"{token[:-3]}y")
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 4:
        variants.add(token[:-1])
    return variants


def has_meaningful_query_anchor(query: str, candidate: dict[str, Any]) -> bool:
    """Reject selector choices with no visible relationship to the query."""
    card = source_card(candidate, 0)
    haystack = (
        f"{card['title']} {card['snippet']} {_visible_url_text(card['url'])}"
    ).lower()
    matches = {
        token
        for token in _anchor_tokens(query)
        if any(variant in haystack for variant in _token_variants(token))
    }
    specific_matches = matches - GENERIC_QUERY_ANCHORS
    # A version, underscored identifier, or genuinely distinctive long term is
    # a strong single anchor; otherwise require two independent terms.  Generic
    # company/platform names are deliberately excluded above.
    return (
        len(specific_matches) >= 2
        or any(
            any(char.isdigit() for char in token)
            or "_" in token
            or len(token) >= 11
            for token in specific_matches
        )
    )


def has_requested_evidence_relation(
    query: str,
    text: str,
    *,
    required_scope_anchors: list[str] | None = None,
) -> bool:
    """Verify supported query relationships in retained evidence.

    This deliberately handles only relationships with a safe deterministic
    rule. Unsupported query types return True and continue to rely on semantic
    retrieval plus scope checks.
    """
    if not PREDATION_QUERY_PATTERN.search(query or ""):
        return True

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n|(?<=[.!?])\s+", text or "")
        if paragraph.strip()
    ]
    claims = [
        paragraph
        for paragraph in paragraphs
        if TURTLE_TARGET_PATTERN.search(paragraph)
        and PREDATION_EVIDENCE_PATTERN.search(paragraph)
        and CONCRETE_PREDATION_PATTERN.search(paragraph)
    ]
    if not claims:
        return False
    if not required_scope_anchors:
        return True

    def scope_patterns(anchor: str) -> tuple[str, ...]:
        normalized = " ".join(anchor.lower().split())
        if "terrestrial turtle" in normalized:
            return (
                r"\bterrestrial turtles?\b",
                r"\bland turtles?\b",
                r"\btortoises?\b",
                r"\bbox turtles?\b",
            )
        if "freshwater turtle" in normalized:
            return (
                r"\bfreshwater turtles?\b",
                r"\bpond turtles?\b",
                r"\briver turtles?\b",
            )
        if "sea turtle" in normalized:
            return (r"\bsea turtles?\b", r"\bmarine turtles?\b")
        return (rf"\b{re.escape(normalized)}\b",)

    return all(
        any(
            any(re.search(pattern, claim, re.IGNORECASE) for pattern in scope_patterns(anchor))
            for claim in claims
        )
        for anchor in required_scope_anchors
    )


def requires_supported_evidence_relation(query: str) -> bool:
    """Return whether deterministic relationship validation applies."""
    return bool(PREDATION_QUERY_PATTERN.search(query or ""))


def _candidate_score(
    query: str,
    candidate: dict[str, Any],
    index: int,
    *,
    anchors: set[str] | None = None,
) -> tuple[int, str]:
    """Return the deterministic score and tier used by both selector paths."""
    anchors = anchors if anchors is not None else _anchor_tokens(query)
    card = source_card(candidate, index + 1)
    haystack = " ".join(
        (card["title"], card["snippet"], _visible_url_text(card["url"]))
    ).lower()
    coverage = sum(token in haystack for token in anchors)
    tier = source_quality_tier(candidate, query)
    score = coverage * 3 + min(len(card["snippet"]) // 180, 3)
    score += {
        "primary": 8,
        "reputable": 4,
        "fallback": 0,
        "reject": -10,
    }[tier]
    if (
        PREDATION_QUERY_PATTERN.search(query or "")
        and not PREDATION_EVIDENCE_PATTERN.search(haystack)
    ):
        score -= 12
    domain = source_domain(source_url(candidate))
    if (
        domain == "doi.org"
        or domain == "arxiv.org"
        or domain.endswith(".arxiv.org")
        or domain == "pmc.ncbi.nlm.nih.gov"
    ):
        score += 4
    if urlparse(source_url(candidate)).path.lower().endswith(".pdf"):
        score += 2
    return score, tier


def deterministic_select_sources(
    query: str, candidates: list[dict[str, Any]], max_sources: int, *, strict: bool = False
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rank source cards when the selection model cannot return valid JSON.

    The fallback deliberately favors anchor coverage and variety over scraping
    every result. It is deterministic so it is suitable for failures and tests.
    """
    anchors = _anchor_tokens(query)
    scored: list[tuple[int, int, dict[str, Any], str, str]] = []
    seen_urls: set[str] = set()
    for index, candidate in enumerate(candidates):
        url = source_url(candidate)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        score, tier = _candidate_score(query, candidate, index, anchors=anchors)
        reason = f"deterministic {tier}: query-anchor coverage, source quality, and domain diversity"
        scored.append((score, -index, candidate, reason, tier))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    domains: Counter[str] = Counter()
    selected_urls: set[str] = set()
    has_higher_tier = any(tier in {"primary", "reputable"} and has_meaningful_query_anchor(query, candidate)
                          for _, _, candidate, _, tier in scored)
    for score, _, candidate, reason, tier in scored:
        url = source_url(candidate)
        domain = source_domain(url)
        if tier == "reject":
            reasons[url] = "deterministic reject: low-value utility, dictionary, or comparison result"
            continue
        if strict and not has_meaningful_query_anchor(query, candidate):
            reasons[url] = "deterministic reject: no meaningful query-anchor coverage"
            continue
        if strict and tier == "fallback" and has_higher_tier:
            reasons[url] = "deterministic reject: broader-web fallback is unnecessary"
            continue
        if score < 0 and selected:
            reasons[url] = "deterministic reject: low source-quality score"
            continue
        if not has_meaningful_query_anchor(query, candidate) and selected:
            reasons[url] = "deterministic reject: no meaningful query-anchor coverage"
            continue
        if len(selected) >= max(1, max_sources):
            reasons[url] = "deterministic reject: lower-ranked after the per-query cap"
            continue
        if domains[domain] >= 1:
            reasons[url] = "deterministic reject: duplicate domain; preserving source diversity"
            continue
        selected.append(candidate)
        selected_urls.add(url)
        domains[domain] += 1
        reasons[url] = reason

    for candidate in candidates:
        url = source_url(candidate)
        if url and url not in reasons:
            reasons[url] = "fallback: duplicate URL"
    return selected, reasons


def should_use_model_selector(query: str, candidates: list[dict[str, Any]], mode: str) -> bool:
    """Return true only for close, non-primary candidates needing judgment."""
    if mode == "llm":
        return True
    if mode != "auto":
        return False
    eligible = [
        candidate for candidate in candidates
        if source_quality_tier(candidate, query) != "reject" and has_meaningful_query_anchor(query, candidate)
    ]
    # Qualified primary or reputable sources make this an objective, fast
    # selection. The model is reserved for genuinely close same-tier fallback
    # candidates.
    if len(eligible) < 2 or any(
        source_quality_tier(candidate, query) in {"primary", "reputable"}
        for candidate in eligible
    ):
        return False
    scored = [
        (*_candidate_score(query, candidate, index), index)
        for index, candidate in enumerate(eligible)
    ]
    scored.sort(key=lambda item: (item[0], -item[2]), reverse=True)
    top_score, top_tier, _ = scored[0]
    second_score, second_tier, _ = scored[1]
    return top_tier == second_tier and top_score - second_score < 4


def parse_model_selection(
    payload: Any, candidates: list[dict[str, Any]], max_sources: int
) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    """Validate selected source IDs from a source-selector response."""
    if not isinstance(payload, dict) or not isinstance(payload.get("selected"), list):
        return None
    by_id = {index + 1: candidate for index, candidate in enumerate(candidates)}
    selected: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    domains: set[str] = set()
    for item in payload["selected"]:
        if not isinstance(item, dict):
            return None
        try:
            source_id = int(item.get("id"))
        except (TypeError, ValueError):
            return None
        candidate = by_id.get(source_id)
        if not candidate:
            return None
        url = source_url(candidate)
        domain = source_domain(url)
        if not url or url in {source_url(value) for value in selected} or domain in domains:
            continue
        selected.append(candidate)
        domains.add(domain)
        reasons[url] = str(item.get("reason") or "selected by relevance, quality, and diversity")[:300]
        if len(selected) >= max(1, max_sources):
            break
    if not selected:
        return None
    for item in payload.get("rejected", []):
        if not isinstance(item, dict):
            continue
        try:
            candidate = by_id.get(int(item.get("id")))
        except (TypeError, ValueError):
            candidate = None
        if candidate:
            reasons[source_url(candidate)] = str(item.get("reason") or "not selected")[:300]
    for candidate in candidates:
        url = source_url(candidate)
        if url and url not in reasons:
            reasons[url] = "not selected by the bounded source selector"
    return selected, reasons
