from gpt_researcher.actions.report_generation import (
    enforce_verified_citation_urls,
    report_citation_urls,
)


def test_report_citation_guard_rewrites_or_removes_unverified_urls():
    verified = "https://huggingface.co/microsoft/Mage-Flow"
    report = f"""# Mage-Flow

The verified model card describes the model ([model card]({verified})).
It links to an unverified sibling ([edit model](https://huggingface.co/microsoft/Mage-Flow-Edit)).
An unrelated URL must not survive ([blog](https://example.com/speculation)).

## References

- [Mage-Flow]({verified})
- [Edit](https://huggingface.co/microsoft/Mage-Flow-Edit)
- [Speculation](https://example.com/speculation)
"""

    guarded = enforce_verified_citation_urls(report, [verified])

    assert report_citation_urls(guarded) == {verified}
    assert "https://huggingface.co/microsoft/Mage-Flow-Edit" not in guarded
    assert "https://example.com/speculation" not in guarded
    assert guarded.count("## References") == 1


def test_report_citation_guard_strips_all_links_without_verified_sources():
    guarded = enforce_verified_citation_urls(
        "Claim ([source](https://unverified.example/a)).\n\n## References\n",
        [],
    )

    assert report_citation_urls(guarded) == set()
    assert "https://" not in guarded
    assert "## References" not in guarded
