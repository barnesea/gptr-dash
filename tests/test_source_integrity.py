from gpt_researcher.actions.source_selection import (
    deterministic_select_sources,
    extract_query_urls,
    has_meaningful_query_anchor,
    has_requested_evidence_relation,
    post_scrape_integrity_reason,
    source_quality_tier,
    source_url,
)


def _card(url, title, body):
    return {"href": url, "title": title, "body": body}


def test_known_official_ml_documentation_beats_generic_embedding_blog():
    query = "Word2Vec and transformer embedding model training architectures"
    tensorflow = _card(
        "https://www.tensorflow.org/text/tutorials/word2vec",
        "Word2Vec tutorial",
        "Official TensorFlow guide to Word2Vec embedding model training.",
    )
    generic = _card(
        "https://blog.example.test/vector-embedding",
        "Word2Vec and transformer embeddings",
        "Embedding model architecture and Word2Vec comparison.",
    )

    selected, reasons = deterministic_select_sources(query, [generic, tensorflow], 3, strict=True)

    assert source_quality_tier(tensorflow) == "primary"
    assert [source_url(item) for item in selected] == [source_url(tensorflow)]
    assert "fallback is unnecessary" in reasons[source_url(generic)]


def test_post_scrape_integrity_rejects_auth_placeholder_despite_relevant_body():
    candidate = _card(
        "https://example.test/vector-embedding",
        "Vector embeddings explained",
        "Embedding models and retrieval systems overview.",
    )
    scraped = {
        "url": candidate["href"],
        "title": "Set your API key",
        "raw_content": "Word2Vec and BERT embedding models are used for retrieval.",
    }

    reason = post_scrape_integrity_reason("embedding models and retrieval models", candidate, scraped)

    assert reason == "post-scrape reject: fetched title indicates an auth wall, error page, or placeholder"


def test_post_scrape_integrity_rejects_redirected_off_topic_page():
    candidate = _card(
        "https://example.test/io-uring",
        "Linux 6.12 io_uring changes",
        "Kernel io_uring changes for application developers.",
    )
    scraped = {
        "url": candidate["href"],
        "title": "GNOME fractional scaling",
        "raw_content": "A guide to desktop display scaling and Wayland settings.",
    }

    reason = post_scrape_integrity_reason("Linux 6.12 io_uring application changes", candidate, scraped)

    assert reason == "post-scrape reject: fetched page has no meaningful query-anchor coverage"


def test_post_scrape_integrity_requires_requested_predation_evidence():
    candidate = _card(
        "https://www.fisheries.noaa.gov/species/hawksbill-turtle",
        "Hawksbill Turtle",
        "Sea turtle life stages and habitat.",
    )
    scraped = {
        "url": candidate["href"],
        "title": "Hawksbill Turtle",
        "raw_content": (
            "Hawksbill sea turtles use coral reef habitats throughout "
            "different life stages. NOAA Fisheries manages conservation."
        ),
    }
    reason = post_scrape_integrity_reason(
        "natural predators of sea turtles by life stage",
        candidate,
        scraped,
    )
    assert "requested predation relationship" in reason


def test_post_scrape_integrity_checks_scholarly_content_after_long_header():
    candidate = _card(
        "https://pmc.ncbi.nlm.nih.gov/articles/example/",
        "Survival of freshwater turtle nests",
        "Freshwater turtle nest survival study.",
    )
    scraped = {
        "url": candidate["href"],
        "title": candidate["title"],
        "raw_content": (
            ("journal metadata navigation " * 300)
            + "The main predators of freshwater turtle nests include red "
            "foxes and raccoons."
        ),
    }
    assert (
        post_scrape_integrity_reason(
            "natural predators of freshwater turtles primary research paper",
            candidate,
            scraped,
        )
        is None
    )


def test_predation_relation_must_match_each_required_scope():
    context = (
        "Predators of freshwater turtle nests include foxes and raccoons. "
        "Sea turtles have flippers and migrate long distances."
    )
    assert has_requested_evidence_relation(
        "natural predators of freshwater turtles",
        context,
        required_scope_anchors=["freshwater turtles"],
    )
    assert not has_requested_evidence_relation(
        "natural predators of sea turtles",
        context,
        required_scope_anchors=["sea turtles"],
    )


def test_hugging_face_title_normalization_accepts_concatenated_extracted_title():
    query = "Do deep research on https://huggingface.co/microsoft/Mage-Flow"
    candidate = _card(
        "https://huggingface.co/microsoft/Mage-Flow",
        "microsoft/Mage-Flow · Hugging Face",
        "Native-resolution image generation and editing model.",
    )
    scraped = {
        "url": candidate["href"],
        "title": "Mage-FlowAn Efficient Native-Resolution Foundation Model for Image Generation and Editing",
        "raw_content": (
            "Mage-Flow is a 4B native-resolution image generation and editing "
            "foundation model from Microsoft."
        ),
    }

    assert has_meaningful_query_anchor(query, candidate)
    assert post_scrape_integrity_reason(query, candidate, scraped) is None


def test_url_query_string_cannot_forge_relevance_and_account_pages_are_rejected():
    query = "Do deep research on https://huggingface.co/microsoft/Mage-Flow"
    account = _card(
        "https://account.microsoft.com/account?search=Mage-Flow+microsoft+huggingface",
        "Microsoft account | Sign In or Create Your Account",
        "Manage your Microsoft account.",
    )

    assert source_quality_tier(account) == "reject"
    assert not has_meaningful_query_anchor(query, account)


def test_extract_query_urls_preserves_order_and_removes_duplicates():
    query = (
        "Compare https://huggingface.co/microsoft/Mage-Flow with "
        "https://arxiv.org/abs/2607.19064, then revisit "
        "https://huggingface.co/microsoft/Mage-Flow."
    )

    assert extract_query_urls(query) == [
        "https://huggingface.co/microsoft/Mage-Flow",
        "https://arxiv.org/abs/2607.19064",
    ]
