"""Generalized, job-scoped retrieval planning and evidence bookkeeping.

The helpers in this module deliberately know about retrieval mechanics rather
than research topics.  They preserve evidence discovered at any stage, render
small entity-first queries, and classify authority relative to the subject
being researched instead of treating a hosting domain as inherently primary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import re
from threading import RLock
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


RETRIEVAL_PIPELINE_VERSION = "v2"
EVIDENCE_ROLES = {
    "first_party",
    "original",
    "reputable_secondary",
    "practitioner",
    "reject",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.IGNORECASE)
CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z][a-z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msockid",
    "ref",
    "ref_src",
}
TRACKING_QUERY_PREFIXES = ("utm_",)
QUERY_STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "could",
    "deep",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "into",
    "make",
    "please",
    "research",
    "should",
    "that",
    "the",
    "their",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
EVIDENCE_QUALIFIERS = {
    "first_party": "official source",
    "original": "primary study",
    "reputable_secondary": "independent expert evidence",
    "practitioner": "practical guide",
}
COMMUNITY_HOSTS = {
    "dev.to",
    "hashnode.com",
    "medium.com",
    "note.com",
    "quora.com",
    "reddit.com",
    "substack.com",
    "youtube.com",
}
NEUTRAL_HOSTING_DOMAINS = {
    "github.com",
    "gitlab.com",
    "huggingface.co",
}
SCHOLARLY_ORIGINAL_DOMAINS = {
    "aclanthology.org",
    "arxiv.org",
    "doi.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "journals.plos.org",
    "openreview.net",
    "pmc.ncbi.nlm.nih.gov",
}
SCHOLARLY_PUBLISHER_MARKERS = (
    "frontiersin.org",
    "nature.com",
    "royalsocietypublishing.org",
    "sciencedirect.com",
    "springer.com",
    "tandfonline.com",
    "wiley.com",
)
REPUTABLE_DOMAIN_MARKERS = (
    ".edu",
    "acm.org",
    "apnews.com",
    "arstechnica.com",
    "bbc.co.",
    "bbc.com",
    "ieee.org",
    "infoq.com",
    "iucn.org",
    "lwn.net",
    "museum",
    "nationalgeographic.com",
    "reuters.com",
    "si.edu",
    "smithsonian",
    "university",
)
HARD_REJECT_DOMAIN_MARKERS = (
    "dictionary.",
    "diffchecker.",
    "draftable.",
    "merriam-webster.",
    "pinterest.",
    "text-compare.",
    "textcompare.",
    "thesaurus.",
    "vocabulary.com",
)


def normalized_tokens(value: str) -> list[str]:
    """Return stable searchable tokens while preserving identifiers."""
    value = CAMEL_BOUNDARY.sub(" ", str(value or ""))
    return [token.lower() for token in TOKEN_PATTERN.findall(value)]


def meaningful_tokens(value: str) -> list[str]:
    return [
        token
        for token in normalized_tokens(value)
        if len(token) >= 3 and token not in QUERY_STOP_WORDS
    ]


def canonicalize_url(url: str) -> str:
    """Normalize a URL for job-scoped deduplication without changing identity."""
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return str(url or "").strip()
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in TRACKING_QUERY_KEYS
            and not key.lower().startswith(TRACKING_QUERY_PREFIXES)
        ],
        doseq=True,
    )
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse(
        (parsed.scheme.lower(), host, path, parsed.params, query, "")
    )


def normalize_entities(values: Any) -> list[dict[str, Any]]:
    """Normalize planner entity strings/objects into a conservative schema."""
    if not isinstance(values, list):
        values = [values] if values else []
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            name = str(
                value.get("name")
                or value.get("surface")
                or value.get("value")
                or ""
            ).strip()
            aliases = value.get("aliases") or []
            exact = bool(value.get("exact", True))
        else:
            name = str(value or "").strip()
            aliases = []
            exact = True
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        if not isinstance(aliases, list):
            aliases = [aliases]
        entities.append(
            {
                "name": name,
                "aliases": [
                    str(alias).strip()
                    for alias in aliases
                    if str(alias).strip()
                ][:5],
                "exact": exact,
            }
        )
    return entities[:8]


def infer_intent(query: str) -> str:
    value = str(query or "").lower()
    comparison = any(
        term in value
        for term in ("compare", " versus ", " vs.", "difference")
    )
    recent = any(
        term in value
        for term in ("latest", "recent", "current", "last ")
    )
    procedural = any(
        term in value
        for term in ("how do", "how to", "setup", "configure", "make ")
    )
    if sum((comparison, recent, procedural)) > 1:
        return "mixed"
    if comparison:
        return "comparison"
    if recent:
        return "recent"
    if procedural:
        return "procedural"
    subject = re.sub(
        r"^\s*(?:please\s+)?(?:explain|describe|research|"
        r"tell\s+me\s+about|what\s+(?:is|are)|why|who|where)\s+",
        "",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    if re.search(r"\b[A-Z][A-Za-z0-9_.-]+\b", subject):
        return "entity"
    return "broad"


def infer_evidence_roles(intent: str, expected: str = "") -> list[str]:
    haystack = f"{intent} {expected}".lower()
    roles: list[str] = []
    if any(
        term in haystack
        for term in ("official", "documentation", "first-party", "release")
    ):
        roles.append("first_party")
    if any(
        term in haystack
        for term in ("paper", "study", "scholarly", "original", "dataset")
    ):
        roles.append("original")
    if intent == "procedural":
        roles.extend(["first_party", "practitioner"])
    elif intent == "mixed":
        roles.extend(
            [
                "first_party",
                "reputable_secondary",
                "practitioner",
            ]
        )
    elif intent == "comparison":
        roles.extend(["first_party", "reputable_secondary"])
    elif intent == "broad":
        roles.extend(["original", "reputable_secondary"])
    else:
        roles.extend(["first_party", "reputable_secondary"])
    return list(dict.fromkeys(role for role in roles if role in EVIDENCE_ROLES))[:3]


def normalize_aspect_contract(
    aspect: dict[str, Any], original_query: str
) -> dict[str, Any]:
    """Fill the v2 aspect contract while retaining legacy planner fields."""
    normalized = dict(aspect)
    intent = str(aspect.get("intent") or infer_intent(original_query)).lower()
    if intent not in {
        "broad",
        "entity",
        "procedural",
        "comparison",
        "recent",
        "mixed",
    }:
        intent = infer_intent(original_query)
    raw_entities = (
        aspect.get("entities")
        if "entities" in aspect
        else aspect.get("entities_versions_dates")
    )
    entities = normalize_entities(raw_entities or [])
    relation = str(
        aspect.get("relation")
        or aspect.get("task")
        or aspect.get("question")
        or original_query
    ).strip()
    inferred_roles = infer_evidence_roles(
        intent, str(aspect.get("expected_evidence_type") or "")
    )
    evidence_roles = aspect.get("evidence_roles") or []
    if not isinstance(evidence_roles, list):
        evidence_roles = [evidence_roles]
    evidence_roles = [
        str(role).strip().lower()
        for role in evidence_roles
        if str(role).strip().lower() in EVIDENCE_ROLES - {"reject"}
    ]
    evidence_roles = list(
        dict.fromkeys([*evidence_roles, *inferred_roles])
    )[:3]
    normalized.update(
        {
            "intent": intent,
            "entities": entities,
            "relation": relation,
            "evidence_roles": evidence_roles or inferred_roles,
        }
    )
    return normalized


def _quote_entity(name: str, exact: bool) -> str:
    name = " ".join(str(name).split())
    if exact and " " in name and '"' not in name:
        return f'"{name}"'
    return name


def render_compact_query(
    aspect: dict[str, Any],
    *,
    evidence_role: str | None = None,
    max_terms: int = 12,
) -> str:
    """Render an entity-first query instead of forwarding planner prose."""
    entities = normalize_entities(aspect.get("entities") or [])
    parts: list[str] = []
    term_count = 0
    for entity in entities[:3]:
        rendered = _quote_entity(entity["name"], bool(entity.get("exact", True)))
        entity_terms = max(1, len(normalized_tokens(entity["name"])))
        if parts and term_count + entity_terms > max_terms:
            break
        parts.append(rendered)
        term_count += entity_terms

    entity_tokens = {
        token
        for entity in entities
        for value in [entity["name"], *(entity.get("aliases") or [])]
        for token in normalized_tokens(value)
    }
    relation_tokens = [
        token
        for token in meaningful_tokens(
            str(aspect.get("relation") or aspect.get("question") or "")
        )
        if token not in entity_tokens
    ]
    for token in relation_tokens:
        if term_count >= max_terms:
            break
        if token not in normalized_tokens(" ".join(parts)):
            parts.append(token)
            term_count += 1

    role = str(evidence_role or "").strip().lower()
    qualifier = EVIDENCE_QUALIFIERS.get(role, "")
    if qualifier:
        for token in qualifier.split():
            if term_count >= max_terms:
                break
            if token not in normalized_tokens(" ".join(parts)):
                parts.append(token)
                term_count += 1

    if not parts:
        parts = meaningful_tokens(
            str(aspect.get("search_query") or aspect.get("question") or "")
        )[:max_terms]
    return " ".join(parts).strip()


def candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in ("title", "body", "snippet", "url", "href")
    )


def entity_coverage(
    aspect: dict[str, Any], candidate: dict[str, Any]
) -> tuple[int, int]:
    """Return matched and required exact-entity counts."""
    haystack = candidate_text(candidate).lower()
    entities = normalize_entities(aspect.get("entities") or [])
    matched = 0
    required = 0
    for entity in entities:
        names = [entity["name"], *(entity.get("aliases") or [])]
        if entity.get("exact", True):
            required += 1
        if any(str(name).lower() in haystack for name in names if str(name).strip()):
            matched += 1
    return matched, required


def candidate_aspect_score(
    aspect: dict[str, Any], candidate: dict[str, Any], *, rank: int = 1
) -> float:
    """Score a card for aspect assignment and cross-query fusion."""
    text_tokens = set(meaningful_tokens(candidate_text(candidate)))
    relation_tokens = set(
        meaningful_tokens(
            str(aspect.get("relation") or aspect.get("question") or "")
        )
    )
    matched_entities, required_entities = entity_coverage(aspect, candidate)
    relation_overlap = len(text_tokens & relation_tokens)
    reciprocal_rank = 1.0 / max(1, rank)
    semantic = float(candidate.get("_gptr_semantic_score") or 0.0)
    exact_bonus = matched_entities * 5.0
    if required_entities and matched_entities == 0:
        exact_bonus -= 8.0
    return (
        exact_bonus
        + min(relation_overlap, 5) * 1.5
        + reciprocal_rank * 2.0
        + semantic * 6.0
    )


def lexical_collision(
    aspect: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> bool:
    """Detect pools matching generic verbs while missing named entities."""
    entities = normalize_entities(aspect.get("entities") or [])
    if not entities:
        return False
    values = list(candidates)
    if not values:
        return False
    entity_hits = sum(entity_coverage(aspect, candidate)[0] > 0 for candidate in values)
    relation_tokens = set(
        meaningful_tokens(
            str(aspect.get("relation") or aspect.get("question") or "")
        )
    )

    def overlaps_relation(candidate: dict[str, Any]) -> bool:
        candidate_tokens = set(
            meaningful_tokens(candidate_text(candidate))
        )
        return bool(candidate_tokens & relation_tokens) or any(
            len(left) >= 5
            and len(right) >= 5
            and left[:5] == right[:5]
            for left in candidate_tokens
            for right in relation_tokens
        )

    relation_hits = sum(
        overlaps_relation(candidate)
        for candidate in values
    )
    return relation_hits > 0 and entity_hits == 0


def _namespace_owner(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if domain in {"github.com", "gitlab.com", "huggingface.co"} and segments:
        if domain == "huggingface.co" and segments[0] in {
            "blog",
            "datasets",
            "docs",
            "papers",
            "spaces",
            "t",
        }:
            if segments[0] in {"datasets", "spaces"} and len(segments) > 1:
                return segments[1].lower()
            return ""
        return segments[0].lower()
    return ""


def _namespace_repository(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if domain not in NEUTRAL_HOSTING_DOMAINS:
        return ""
    if domain == "huggingface.co" and segments:
        if segments[0] in {"datasets", "spaces"}:
            return segments[2].lower() if len(segments) > 2 else ""
        if segments[0] in {"blog", "docs", "papers", "t"}:
            return ""
    return segments[1].lower() if len(segments) > 1 else ""


def evidence_role(
    candidate: dict[str, Any],
    aspect: dict[str, Any] | None = None,
    query: str = "",
) -> str:
    """Classify authority relative to the named subject and requested claim."""
    url = str(candidate.get("url") or candidate.get("href") or "")
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    if not domain or any(marker in domain for marker in HARD_REJECT_DOMAIN_MARKERS):
        return "reject"
    if any(
        marker in str(candidate.get("title") or "").lower()
        for marker in ("sign in", "log in", "access denied", "page not found")
    ):
        return "reject"

    padded_query = f" {str(query).lower()} "
    if domain in COMMUNITY_HOSTS or any(
        domain.endswith(f".{host}") for host in COMMUNITY_HOSTS
    ):
        platform_name = domain.split(".")[0]
        return "first_party" if f" {platform_name} " in padded_query else "practitioner"

    normalized_aspect = normalize_aspect_contract(aspect or {}, query)
    entity_values = [
        value
        for entity in normalized_aspect.get("entities", [])
        for value in [entity["name"], *(entity.get("aliases") or [])]
    ]
    entity_tokens = {
        token
        for value in entity_values
        for token in meaningful_tokens(value)
        if len(token) >= 4
    }
    if not entity_tokens:
        entity_tokens = {
            token for token in meaningful_tokens(query) if len(token) >= 4
        }

    owner = _namespace_owner(url)
    repository = _namespace_repository(url)
    if domain in NEUTRAL_HOSTING_DOMAINS:
        path_segments = {
            segment.lower()
            for segment in parsed.path.split("/")
            if segment
        }
        if domain == "huggingface.co" and parsed.path.startswith("/papers/"):
            return "original"
        if path_segments & {
            "discussions",
            "issues",
            "pull",
            "pulls",
            "t",
        }:
            return "practitioner"
        normalized_entity_values = [
            re.sub(r"[\s_-]+", "", str(value).lower())
            for value in entity_values
        ]
        normalized_repository = re.sub(
            r"[\s_-]+", "", repository.lower()
        )
        if normalized_repository and normalized_repository in (
            normalized_entity_values
        ):
            return "first_party"
        if owner and (
            any(owner == token or token in owner for token in entity_tokens)
            or any(
                owner.replace("-", "").replace("_", "") in value
                for value in normalized_entity_values
            )
        ):
            return "first_party"
        return "practitioner"

    if (
        domain in SCHOLARLY_ORIGINAL_DOMAINS
        or any(domain.endswith(f".{value}") for value in SCHOLARLY_ORIGINAL_DOMAINS)
        or domain.endswith(".gov")
    ):
        return "original"
    if any(marker in domain for marker in SCHOLARLY_PUBLISHER_MARKERS):
        scholarly_card_text = candidate_text(candidate).lower()
        if any(
            marker in scholarly_card_text
            for marker in (
                "/article",
                "/doi/",
                "abstract",
                "journal",
                "research article",
                "study",
            )
        ):
            return "original"
        return "reputable_secondary"
    if any(marker in domain for marker in REPUTABLE_DOMAIN_MARKERS):
        return "reputable_secondary"

    domain_tokens = {
        token
        for token in re.split(r"[-.]", domain)
        if len(token) >= 4 and token not in {"docs", "developer", "www"}
    }
    if entity_tokens & domain_tokens:
        return "first_party"
    if domain.startswith(("docs.", "developer.", "support.")) and any(
        token in domain for token in entity_tokens
    ):
        return "first_party"
    return "practitioner"


def roles_cover_aspect(
    aspect: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> bool:
    return not missing_evidence_roles(aspect, candidates)


def missing_evidence_roles(
    aspect: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> list[str]:
    """Return required evidence functions not yet represented by cards."""
    required = set(aspect.get("evidence_roles") or [])
    if not required:
        return [] if list(candidates) else ["relevant"]
    present = {
        evidence_role(candidate, aspect, str(aspect.get("question") or ""))
        for candidate in candidates
        if candidate_aspect_score(aspect, candidate) > 0
    }
    # Reputable secondary evidence can stand in for practitioner experience,
    # while original evidence can satisfy an otherwise first-party factual role.
    if "reputable_secondary" in present:
        present.add("practitioner")
    if "original" in present:
        present.add("first_party")
    return [
        role
        for role in aspect.get("evidence_roles") or []
        if role not in present
    ]


@dataclass
class EvidenceCandidate:
    candidate_id: str
    sequence: int
    canonical_url: str
    card: dict[str, Any]
    discoveries: list[dict[str, Any]] = field(default_factory=list)
    assigned_aspects: set[str] = field(default_factory=set)
    fetch_attempts: list[dict[str, Any]] = field(default_factory=list)
    judgments: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        card = deepcopy(self.card)
        prefetched_chars = len(
            str(card.pop("raw_content", "") or "")
        )
        card.pop("content", None)
        for key in ("body", "snippet"):
            if card.get(key):
                card[key] = str(card[key])[:1200]
        if prefetched_chars:
            card["_gptr_prefetched_content_chars"] = prefetched_chars
        return {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "canonical_url": self.canonical_url,
            "card": card,
            "discoveries": deepcopy(self.discoveries),
            "assigned_aspects": sorted(self.assigned_aspects),
            "fetch_attempts": deepcopy(self.fetch_attempts),
            "judgments": deepcopy(self.judgments),
        }


class EvidenceCandidateLedger:
    """Thread-safe job ledger shared by concurrent research branches."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_url: dict[str, EvidenceCandidate] = {}
        self._sequence = 0
        self._attempted_queries: set[str] = set()

    def register(
        self,
        candidates: Iterable[dict[str, Any]],
        *,
        stage: str,
        query: str,
    ) -> list[str]:
        ids: list[str] = []
        with self._lock:
            for rank, candidate in enumerate(candidates, start=1):
                url = str(candidate.get("url") or candidate.get("href") or "").strip()
                canonical = canonicalize_url(url)
                if not canonical:
                    continue
                entry = self._by_url.get(canonical)
                if entry is None:
                    self._sequence += 1
                    card = deepcopy(candidate)
                    if card.get("href"):
                        card["href"] = canonical
                    else:
                        card["url"] = canonical
                    if canonical != url:
                        card["_gptr_discovered_url"] = url
                    entry = EvidenceCandidate(
                        candidate_id=f"candidate-{self._sequence}",
                        sequence=self._sequence,
                        canonical_url=canonical,
                        card=card,
                    )
                    self._by_url[canonical] = entry
                else:
                    for key in (
                        "title",
                        "body",
                        "snippet",
                        "date",
                        "engine",
                        "raw_content",
                    ):
                        if not entry.card.get(key) and candidate.get(key):
                            entry.card[key] = candidate[key]
                discovery = {
                    "stage": stage,
                    "query": query,
                    "engine": str(candidate.get("engine") or ""),
                    "rank": rank,
                }
                if discovery not in entry.discoveries:
                    entry.discoveries.append(discovery)
                ids.append(entry.candidate_id)
        return ids

    def register_query_attempt(self, query: str) -> bool:
        normalized = " ".join(str(query or "").lower().split())
        if not normalized:
            return False
        with self._lock:
            if normalized in self._attempted_queries:
                return False
            self._attempted_queries.add(normalized)
            return True

    def candidates_for_aspect(
        self,
        aspect: dict[str, Any],
        *,
        limit: int,
        exclude_failed: bool = True,
        exclude_urls: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, int, EvidenceCandidate]] = []
        aspect_id = str(aspect.get("id") or "")
        excluded = {
            canonicalize_url(url)
            for url in exclude_urls
            if str(url or "").strip()
        }
        with self._lock:
            for entry in self._by_url.values():
                if entry.canonical_url in excluded:
                    continue
                aspect_judgments = [
                    judgment
                    for judgment in entry.judgments
                    if str(judgment.get("aspect_id") or "") == aspect_id
                ]
                if exclude_failed and (
                    (
                        aspect_judgments
                        and not aspect_judgments[-1].get(
                            "accepted_for_synthesis",
                            aspect_judgments[-1].get(
                                "supports_aspect", False
                            ),
                        )
                    )
                    or (
                        entry.fetch_attempts
                        and entry.fetch_attempts[-1].get("outcome")
                        == "integrity_rejected"
                    )
                ):
                    continue
                unique_discoveries = {
                    (
                        str(discovery.get("stage") or ""),
                        str(discovery.get("query") or "").lower(),
                        str(discovery.get("engine") or ""),
                    ): int(discovery.get("rank") or 999)
                    for discovery in entry.discoveries
                }
                rank = min(unique_discoveries.values(), default=999)
                # Reciprocal-rank fusion rewards a card independently
                # rediscovered by preliminary and aspect searches without
                # allowing one noisy engine to dominate lexical relevance.
                fusion_score = sum(
                    60.0 / (60.0 + max(1, value))
                    for value in unique_discoveries.values()
                )
                score = candidate_aspect_score(aspect, entry.card, rank=rank)
                score += fusion_score
                if score <= 0:
                    continue
                ranked.append((score, entry.sequence, entry))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            selected: list[dict[str, Any]] = []
            for score, _, entry in ranked[: max(1, limit)]:
                if aspect_id:
                    entry.assigned_aspects.add(aspect_id)
                card = deepcopy(entry.card)
                card["_gptr_candidate_id"] = entry.candidate_id
                card["_gptr_canonical_url"] = entry.canonical_url
                card["_gptr_assignment_score"] = round(score, 4)
                card["_gptr_fusion_score"] = round(
                    sum(
                        60.0
                        / (
                            60.0
                            + max(1, int(discovery.get("rank") or 999))
                        )
                        for discovery in entry.discoveries
                    ),
                    4,
                )
                card["_gptr_discovery_count"] = len(entry.discoveries)
                selected.append(card)
            return selected

    def update_candidate(self, url: str, **metadata: Any) -> None:
        """Attach derived card metadata without recording a fake discovery."""
        canonical = canonicalize_url(url)
        with self._lock:
            entry = self._by_url.get(canonical)
            if entry:
                entry.card.update(
                    {
                        key: deepcopy(value)
                        for key, value in metadata.items()
                        if value is not None
                    }
                )

    def record_fetch(
        self, url: str, *, fetch_url: str, outcome: str, reason: str = ""
    ) -> None:
        canonical = canonicalize_url(url)
        with self._lock:
            entry = self._by_url.get(canonical)
            if entry:
                entry.fetch_attempts.append(
                    {
                        "fetch_url": fetch_url,
                        "outcome": outcome,
                        "reason": reason,
                    }
                )

    def record_judgment(self, url: str, judgment: dict[str, Any]) -> None:
        canonical = canonicalize_url(url)
        with self._lock:
            entry = self._by_url.get(canonical)
            if entry:
                entry.judgments.append(deepcopy(judgment))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pipeline_version": RETRIEVAL_PIPELINE_VERSION,
                "candidate_count": len(self._by_url),
                "attempted_queries": sorted(self._attempted_queries),
                "candidates": [
                    entry.snapshot()
                    for entry in sorted(
                        self._by_url.values(),
                        key=lambda value: value.sequence,
                    )
                ],
            }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
