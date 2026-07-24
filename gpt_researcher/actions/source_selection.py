"""Small, bounded source selection before expensive web scraping."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any
import unicodedata
from urllib.parse import quote, unquote, urlparse

from .retrieval_pipeline import (
    canonicalize_url,
    evidence_role,
    meaningful_tokens,
)


LOW_VALUE_HOST_MARKERS = (
    "dictionary.", "merriam-webster.", "vocabulary.com", "thesaurus.",
    "diffchecker.", "textcompare.", "text-compare.", "draftable.",
    "pinterest.com", "articsledge.com",
)
LEGACY_LOW_VALUE_HOST_MARKERS = (
    *LOW_VALUE_HOST_MARKERS,
    "medium.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "tiktok.com",
    "reddit.com",
    "quora.com",
    "dev.to",
    "substack.com",
    "hashnode.",
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
    "apnews.com", "iucn.org", "wwf.org",
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
CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z][a-z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
URL_PATTERN = re.compile(r"https?://[^\s\]\)>\",;]+", re.IGNORECASE)
def source_url(candidate: dict[str, Any]) -> str:
    return str(candidate.get("href") or candidate.get("url") or "").strip()


def source_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def source_site(url: str) -> str:
    """Return a conservative registrable-site key for corroboration."""
    domain = source_domain(url)
    labels = [label for label in domain.split(".") if label]
    if len(labels) <= 2:
        return domain
    multipart_suffixes = {
        "co.uk",
        "com.au",
        "com.br",
        "com.cn",
        "com.sg",
        "co.jp",
        "co.nz",
        "org.uk",
    }
    suffix = ".".join(labels[-2:])
    return ".".join(labels[-3:]) if suffix in multipart_suffixes else suffix


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
    """Return bounded keyless raw/API/PDF variants for supported source types."""
    parsed = urlparse(url)
    domain = source_domain(url)
    path = parsed.path
    alternatives: list[str] = []
    segments = [segment for segment in path.split("/") if segment]
    if domain == "github.com":
        if "/blob/" in path:
            before, after = path.split("/blob/", 1)
            alternatives.append(
                f"https://raw.githubusercontent.com{before}/{after}"
            )
        elif len(segments) == 2:
            owner, repository = segments
            alternatives.extend(
                [
                    f"https://raw.githubusercontent.com/{owner}/{repository}/main/README.md",
                    f"https://raw.githubusercontent.com/{owner}/{repository}/master/README.md",
                ]
            )
    elif domain == "gitlab.com" and len(segments) >= 2:
        owner, repository = segments[:2]
        if "/-/blob/" in path:
            alternatives.append(url.replace("/-/blob/", "/-/raw/", 1))
        elif len(segments) == 2:
            alternatives.append(
                f"https://gitlab.com/{owner}/{repository}/-/raw/main/README.md"
            )
    elif domain == "huggingface.co":
        if "/blob/" in path:
            alternatives.append(url.replace("/blob/", "/resolve/", 1))
        elif len(segments) >= 2 and segments[0] not in {"docs", "blog", "t"}:
            if segments[0] in {"datasets", "spaces"} and len(segments) >= 3:
                prefix, owner, repository = segments[:3]
                alternatives.append(
                    f"https://huggingface.co/{prefix}/{owner}/{repository}/resolve/main/README.md"
                )
            else:
                owner, repository = segments[:2]
                alternatives.append(
                    f"https://huggingface.co/{owner}/{repository}/resolve/main/README.md"
                )
    elif domain == "arxiv.org" and path.startswith("/abs/"):
        identifier = path.removeprefix("/abs/").strip("/")
        if identifier:
            alternatives.extend(
                [
                    f"https://arxiv.org/html/{identifier}",
                    f"https://arxiv.org/pdf/{identifier}",
                ]
            )
    elif domain == "doi.org":
        identifier = path.strip("/")
        if identifier:
            encoded = quote(identifier, safe="")
            alternatives.extend(
                [
                    f"https://api.crossref.org/works/{encoded}",
                    "https://api.openalex.org/works/"
                    f"https://doi.org/{identifier}",
                ]
            )
    elif (
        domain == "pmc.ncbi.nlm.nih.gov"
        and len(segments) >= 2
        and segments[0] == "articles"
    ):
        identifier = segments[1]
        alternatives.extend(
            [
                f"https://pmc.ncbi.nlm.nih.gov/articles/{identifier}/?report=reader",
                f"https://pmc.ncbi.nlm.nih.gov/articles/{identifier}/pdf/",
            ]
        )
    seen: set[str] = set()
    canonical_original = canonicalize_url(url)
    return [
        candidate
        for candidate in alternatives
        if canonicalize_url(candidate) != canonical_original
        and not (candidate in seen or seen.add(candidate))
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
    """Map aspect-relative evidence roles onto the legacy tier vocabulary."""
    internal_tier = str(candidate.get("_gptr_source_tier") or "")
    if internal_tier in {"primary", "reputable", "fallback", "reject"}:
        return internal_tier
    internal_role = str(candidate.get("_gptr_evidence_role") or "")
    if internal_role:
        return {
            "first_party": "primary",
            "original": "primary",
            "reputable_secondary": "reputable",
            "practitioner": "fallback",
            "reject": "reject",
        }.get(internal_role, "fallback")
    # Unstamped candidates use the established legacy classifier. The v2
    # pipeline stamps an aspect-relative role before ranking, so rollback mode
    # retains its exact behavior while neutral hosts such as GitHub and
    # Hugging Face are no longer inherently first-party in v2.
    domain = source_domain(source_url(candidate))
    padded_query = f" {str(query).lower()} "
    for social_host, names in NAMED_SOCIAL_HOSTS.items():
        if (
            domain == social_host or domain.endswith(f".{social_host}")
        ) and any(name in padded_query for name in names):
            return "primary"
    if (
        not domain
        or any(marker in domain for marker in LEGACY_LOW_VALUE_HOST_MARKERS)
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
    # A SERP title and a fetched title normally share at least one distinctive
    # term.  Do not reject pages with no usable title (PDFs often lack one), but
    # reject a wholly unrelated fetched title when the original card was clear.
    if candidate:
        expected_title = str(candidate.get("title") or "").strip()
        if expected_title and title:
            expected_tokens = _anchor_tokens(expected_title) - GENERIC_QUERY_ANCHORS
            fetched_tokens = _anchor_tokens(title) - GENERIC_QUERY_ANCHORS
            if len(expected_tokens) >= 2 and not (expected_tokens & fetched_tokens):
                # Dynamic repository/model-card pages routinely expose a
                # generic fetched title while their canonical path and body are
                # correct. Treat title divergence as a soft signal after the
                # content itself passed the query-anchor guard.
                candidate["_gptr_title_mismatch"] = True
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
        "evidence_role": str(
            candidate.get("_gptr_evidence_role")
            or {
                "primary": "original",
                "reputable": "reputable_secondary",
                "fallback": "practitioner",
                "reject": "reject",
            }[source_quality_tier(candidate)]
        ),
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
    relation: str = "",
    entity_anchors: list[dict[str, Any] | str] | None = None,
) -> bool:
    """Verify entity/relation/scope co-occurrence without topic rules.

    This is deliberately conservative and backs up the model judge only when
    that judge fails. A normal v2 result is accepted from structured judgment,
    not from this lexical approximation.
    """
    normalized_entities: list[list[str]] = []
    entity_alias_groups: list[tuple[str, list[str]]] = []
    entity_tokens: set[str] = set()
    for value in entity_anchors or []:
        if isinstance(value, dict):
            name = str(value.get("name") or "").strip()
            aliases = [
                str(alias).strip()
                for alias in value.get("aliases") or []
                if str(alias).strip()
            ]
            exact = bool(value.get("exact", True))
        else:
            name = str(value or "").strip()
            aliases = []
            exact = True
        if name:
            if exact:
                normalized_entities.append([name, *aliases])
            entity_alias_groups.append((name, aliases))
            entity_tokens.update(meaningful_tokens(name))

    query_tokens = set(meaningful_tokens(query)) - GENERIC_QUERY_ANCHORS
    relation_tokens = [
        token
        for token in meaningful_tokens(relation or query)
        if token not in GENERIC_QUERY_ANCHORS and token not in entity_tokens
    ]

    def token_matches(token: str, paragraph_tokens: set[str]) -> bool:
        variants = _token_variants(token)
        return any(
            candidate in variants
            or any(
                len(candidate) >= 6
                and len(variant) >= 6
                and candidate[:5] == variant[:5]
                for variant in variants
            )
            for candidate in paragraph_tokens
        )

    def relation_matches(paragraph: str) -> bool:
        paragraph_tokens = set(meaningful_tokens(paragraph))
        if relation:
            if not relation_tokens:
                return True
            matched = sum(
                token_matches(token, paragraph_tokens)
                for token in relation_tokens
            )
            minimum = min(
                3,
                max(1, math.ceil(len(set(relation_tokens)) * 0.4)),
            )
            core = list(dict.fromkeys(relation_tokens))[:2]
            return matched >= minimum and any(
                token_matches(token, paragraph_tokens) for token in core
            )
        minimum_overlap = min(3, max(2, len(query_tokens)))
        return (
            sum(
                token_matches(token, paragraph_tokens)
                for token in query_tokens
            )
            >= minimum_overlap
        )

    padded_text = f" {str(text or '').lower()} "
    compact_text = re.sub(r"[^a-z0-9]+", "", padded_text)
    for entity_group in normalized_entities:
        if not any(
            (
                " ".join(entity.lower().split()) in padded_text
                or re.sub(r"[^a-z0-9]+", "", entity.lower())
                in compact_text
            )
            for entity in entity_group
            if entity
        ):
            # Exact named entities are a safety guard against generic verb
            # collisions such as a training query returning transportation.
            return False

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n|(?<=[.!?])\s+", text or "")
        if paragraph.strip()
    ]
    if not required_scope_anchors:
        return any(relation_matches(paragraph) for paragraph in paragraphs)

    def aliases_for_scope(anchor: str) -> list[str]:
        anchor_tokens = set(meaningful_tokens(anchor))
        values = [anchor]
        for name, aliases in entity_alias_groups:
            if anchor_tokens & set(meaningful_tokens(name)):
                values.extend(aliases)
        return list(dict.fromkeys(values))

    return all(
        any(
            any(
                scope_anchor_matches_text(alias, paragraph)
                for alias in aliases_for_scope(anchor)
            )
            and relation_matches(paragraph)
            for paragraph in paragraphs
        )
        for anchor in required_scope_anchors
    )


def scope_anchor_patterns(anchor: str) -> tuple[str, ...]:
    """Return conservative, topic-neutral lexical variants for one scope."""
    normalized = " ".join(str(anchor).lower().split())
    if not normalized:
        return ()
    values = {normalized, normalized.replace("-", " "), normalized.replace(" ", "-")}
    if normalized.endswith("ies"):
        values.add(f"{normalized[:-3]}y")
    elif normalized.endswith("s") and len(normalized) > 4:
        values.add(normalized[:-1])
    else:
        values.add(f"{normalized}s")
    return tuple(rf"\b{re.escape(value)}\b" for value in sorted(values))


def scope_anchor_matches_text(anchor: str, text: str) -> bool:
    """Match a scope anchor using the same aliases as relation validation."""
    return any(
        re.search(pattern, text or "", re.IGNORECASE)
        for pattern in scope_anchor_patterns(anchor)
    )


def requires_supported_evidence_relation(query: str) -> bool:
    """Return whether a query has enough content terms for generic promotion."""
    return len(set(meaningful_tokens(query)) - GENERIC_QUERY_ANCHORS) >= 2


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
    score += round(float(candidate.get("_gptr_semantic_score") or 0.0) * 6)
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
    v2_roles_present = any(
        bool(candidate.get("_gptr_evidence_role")) for candidate in candidates
    )
    qualified_higher_tier_exists = any(
        tier in {"primary", "reputable"}
        and has_meaningful_query_anchor(query, candidate)
        for _, _, candidate, _, tier in scored
    )
    selected: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    domains: Counter[str] = Counter()
    selected_urls: set[str] = set()
    for score, _, candidate, reason, tier in scored:
        url = source_url(candidate)
        domain = source_domain(url)
        if tier == "reject":
            reasons[url] = "deterministic reject: low-value utility, dictionary, or comparison result"
            continue
        if (
            strict
            and not v2_roles_present
            and tier == "fallback"
            and qualified_higher_tier_exists
        ):
            reasons[url] = (
                "deterministic reject: fallback is unnecessary when a "
                "qualified primary or reputable source is available"
            )
            continue
        if strict and not has_meaningful_query_anchor(query, candidate):
            reasons[url] = "deterministic reject: no meaningful query-anchor coverage"
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
        if not candidate.get("_gptr_evidence_role"):
            candidate["_gptr_evidence_role"] = {
                "primary": "original",
                "reputable": "reputable_secondary",
                "fallback": "practitioner",
                "reject": "reject",
            }[tier]
        selected_urls.add(url)
        domains[domain] += 1
        reasons[url] = (
            f"{reason}; evidence role={candidate['_gptr_evidence_role']}"
        )

    for candidate in candidates:
        url = source_url(candidate)
        if url and url not in reasons:
            reasons[url] = "fallback: duplicate URL"
    return selected, reasons


def rebalance_evidence_roles(
    query: str,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    required_roles: list[str],
    max_sources: int,
) -> list[dict[str, Any]]:
    """Preserve the best available functional evidence mix within the cap."""
    required = [
        role
        for role in required_roles
        if role
        in {
            "first_party",
            "original",
            "reputable_secondary",
            "practitioner",
        }
    ]
    if not required or max_sources <= 0:
        return selected[: max(0, max_sources)]

    def role_of(candidate: dict[str, Any]) -> str:
        return str(
            candidate.get("_gptr_evidence_role")
            or evidence_role(candidate, query=query)
        )

    def satisfies(candidate: dict[str, Any], role: str) -> bool:
        actual = role_of(candidate)
        return (
            actual == role
            or (role == "first_party" and actual == "original")
            or (
                role == "practitioner"
                and actual == "reputable_secondary"
            )
        )

    def covers(values: list[dict[str, Any]], role: str) -> bool:
        return any(satisfies(value, role) for value in values)

    ranked = sorted(
        [
            candidate
            for candidate in candidates
            if source_quality_tier(candidate, query) != "reject"
            and has_meaningful_query_anchor(query, candidate)
        ],
        key=lambda candidate: (
            -_candidate_score(
                query,
                candidate,
                candidates.index(candidate),
            )[0],
            candidates.index(candidate),
        ),
    )
    balanced = list(selected[:max_sources])
    for role in required:
        if covers(balanced, role):
            continue
        selected_sites = {
            source_site(source_url(candidate))
            for candidate in balanced
            if source_url(candidate)
        }
        options = [
            candidate
            for candidate in ranked
            if candidate not in balanced and satisfies(candidate, role)
        ]
        options.sort(
            key=lambda candidate: (
                source_site(source_url(candidate)) in selected_sites,
                ranked.index(candidate),
            )
        )
        option = next(iter(options), None)
        if option is None:
            continue
        if len(balanced) < max_sources:
            balanced.append(option)
            continue
        protected_roles = [
            value
            for value in required
            if value != role and covers(balanced, value)
        ]
        for index in range(len(balanced) - 1, -1, -1):
            trial = [
                value
                for position, value in enumerate(balanced)
                if position != index
            ]
            if all(covers(trial, value) for value in protected_roles):
                balanced[index] = option
                break
    return balanced


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
