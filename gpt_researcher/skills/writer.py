"""Report generator skill for GPT Researcher.

This module provides the ReportGenerator class that handles report
writing, including introductions, conclusions, and subtopic management.
"""

import json
from typing import Dict, Optional

from ..actions import (
    generate_draft_section_titles,
    generate_report,
    stream_output,
    write_conclusion,
    write_report_introduction,
)
from ..actions.report_generation import (
    add_visible_evidence_limitation,
    enforce_verified_citation_urls,
    qualify_supplemental_evidence_paragraphs,
    qualify_uncited_synthesis_paragraphs,
    repair_report_evidence_safety,
    report_citation_urls,
    report_quality_diagnostics,
)
from ..utils.llm import construct_subtopics


class ReportGenerator:
    """Generates reports based on research data.

    This class handles all aspects of report generation including
    writing introductions, conclusions, and managing report structure.

    Attributes:
        researcher: The parent GPTResearcher instance.
        research_params: Dictionary of parameters for report generation.
    """

    def __init__(self, researcher):
        """Initialize the ReportGenerator.

        Args:
            researcher: The GPTResearcher instance that owns this generator.
        """
        self.researcher = researcher
        self.research_params = {
            "query": self.researcher.query,
            "agent_role_prompt": self.researcher.cfg.agent_role or self.researcher.role,
            "report_type": self.researcher.report_type,
            "report_source": self.researcher.report_source,
            "tone": self.researcher.tone,
            "websocket": self.researcher.websocket,
            "cfg": self.researcher.cfg,
            "headers": self.researcher.headers,
        }

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None, custom_prompt="", available_images: list = None) -> str:
        """
        Write a report based on existing headers and relevant contents.

        Args:
            existing_headers (list): List of existing headers.
            relevant_written_contents (list): List of relevant written contents.
            ext_context (Optional): External context, if any.
            custom_prompt (str): Custom prompt for the report.
            available_images (list): Pre-generated images available for embedding.

        Returns:
            str: The generated report.
        """
        available_images = available_images or []
        
        # send the selected images prior to writing report
        research_images = self.researcher.get_research_images()
        if research_images:
            await stream_output(
                "images",
                "selected_images",
                json.dumps(research_images),
                self.researcher.websocket,
                True,
                research_images
            )

        context = ext_context or self.researcher.context

        # Guard against fabricating a report from nothing: if no research content was
        # gathered (every retriever returned empty / was blocked / rate-limited), don't
        # silently write a confident, sourced-looking report - abstain so it is visible.
        _ctx = "\n".join(context) if isinstance(context, list) else str(context or "")
        if not _ctx.strip():
            return (
                f'I could not gather any source material for "{self.researcher.query}". '
                "No sources were retrieved (searches may have returned nothing or been "
                "blocked), so I am not able to produce a reliable, sourced report."
            )
        
        # Log image availability
        if available_images and self.researcher.verbose:
            await stream_output(
                "logs",
                "images_available",
                f"🖼️ {len(available_images)} pre-generated images available for embedding",
                self.researcher.websocket,
            )
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_report",
                f"✍️ Writing report for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        report_params = self.research_params.copy()
        if not report_params["agent_role_prompt"]:
            report_params["agent_role_prompt"] = self.researcher.cfg.agent_role or self.researcher.role
        report_params["context"] = context
        report_params["custom_prompt"] = custom_prompt
        report_params["available_images"] = available_images  # Pass pre-generated images
        verified_source_urls = list(
            dict.fromkeys(
                str(source.get("url") or source.get("href") or "").strip()
                for source in self.researcher.get_research_sources()
                if str(source.get("url") or source.get("href") or "").strip()
            )
        )
        report_params["verified_source_urls"] = verified_source_urls
        coverage_ledger = list(
            getattr(self.researcher, "coverage_ledger", []) or []
        )
        report_params["coverage_ledger"] = coverage_ledger
        supplemental_source_urls = list(
            dict.fromkeys(
                str(source.get("url") or "").strip()
                for item in coverage_ledger
                for source in item.get("evidence_pool_sources", [])
                if str(source.get("url") or "").strip()
                and source.get("claim_status")
                in {"provisional", "background"}
                and str(source.get("url") or "").strip()
                not in verified_source_urls
            )
        )
        report_params["supplemental_source_urls"] = (
            supplemental_source_urls
        )
        citation_source_urls = list(
            dict.fromkeys(
                [*verified_source_urls, *supplemental_source_urls]
            )
        )
        if (
            coverage_ledger
            and not verified_source_urls
            and not supplemental_source_urls
        ):
            unresolved = [
                str(item.get("aspect_id") or "unknown")
                for item in coverage_ledger
                if item.get("state") != "evidence_ready"
            ]
            unresolved_text = ", ".join(unresolved) or "all planned aspects"
            return (
                "# Evidence limitation\n\n"
                "No research aspect produced synthesis-eligible verified "
                "evidence. Retrieved pages either failed the requested scope "
                "or evidence-integrity requirements, so no factual report was "
                "generated from them.\n\n"
                f"Unresolved coverage: {unresolved_text}."
            )

        if self.researcher.report_type == "subtopic_report":
            report_params.update({
                "main_topic": self.researcher.parent_query,
                "existing_headers": existing_headers,
                "relevant_written_contents": relevant_written_contents,
                "cost_callback": self.researcher.add_costs,
            })
        else:
            report_params["cost_callback"] = self.researcher.add_costs

        report = await generate_report(**report_params, **self.researcher.kwargs)
        citations_before = report_citation_urls(report)
        report = enforce_verified_citation_urls(
            report,
            citation_source_urls,
        )
        quality_before = (
            report_quality_diagnostics(
                report,
                coverage_ledger,
                query=self.researcher.query,
                supplemental_source_urls=supplemental_source_urls,
            )
            if coverage_ledger
            else {"passes": True, "issues": []}
        )
        correction_used = False
        if not quality_before["passes"]:
            try:
                corrected = await repair_report_evidence_safety(
                    report=report,
                    query=self.researcher.query,
                    coverage_ledger=coverage_ledger,
                    verified_source_urls=verified_source_urls,
                    supplemental_source_urls=supplemental_source_urls,
                    diagnostics=quality_before,
                    cfg=self.researcher.cfg,
                    cost_callback=self.researcher.add_costs,
                    **self.researcher.kwargs,
                )
                report = enforce_verified_citation_urls(
                    corrected,
                    citation_source_urls,
                )
                correction_used = True
            except Exception:
                correction_used = False
        quality_after_correction = (
            report_quality_diagnostics(
                report,
                coverage_ledger,
                query=self.researcher.query,
                supplemental_source_urls=supplemental_source_urls,
            )
            if coverage_ledger
            else quality_before
        )
        unlabeled_before_deterministic_pass = len(
            quality_after_correction.get(
                "unlabeled_supplemental_paragraphs", []
            )
        )
        report = qualify_supplemental_evidence_paragraphs(
            report,
            coverage_ledger,
        )
        report = qualify_uncited_synthesis_paragraphs(report)
        quality_after = (
            report_quality_diagnostics(
                report,
                coverage_ledger,
                query=self.researcher.query,
                supplemental_source_urls=supplemental_source_urls,
            )
            if coverage_ledger
            else quality_after_correction
        )
        if not quality_after["passes"]:
            report = add_visible_evidence_limitation(report, quality_after)
        citations_after = report_citation_urls(report)
        trace = getattr(self.researcher, "trace_event", None)
        if trace:
            trace(
                "report_citation_guard",
                {
                    "verified_source_count": len(verified_source_urls),
                    "supplemental_source_count": len(
                        supplemental_source_urls
                    ),
                    "citation_count_before": len(citations_before),
                    "citation_count_after": len(citations_after),
                    "removed_or_rewritten_count": len(
                        citations_before - citations_after
                    ),
                    "all_citations_allowlisted": all(
                        any(
                            citation.rstrip("/").lower()
                            == source.rstrip("/").lower()
                            for source in citation_source_urls
                        )
                        for citation in citations_after
                    ),
                },
            )
            trace(
                "report_quality_guard",
                {
                    "before": quality_before,
                    "after": quality_after,
                    "correction_used": correction_used,
                    "deterministic_supplemental_labels_added": max(
                        0,
                        unlabeled_before_deterministic_pass
                        - len(
                            quality_after.get(
                                "unlabeled_supplemental_paragraphs", []
                            )
                        ),
                    ),
                    "visible_limitation_added": not quality_after["passes"],
                },
            )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "report_written",
                f"📝 Report written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return report

    async def write_report_conclusion(self, report_content: str) -> str:
        """
        Write the conclusion for the report.

        Args:
            report_content (str): The content of the report.

        Returns:
            str: The generated conclusion.
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_conclusion",
                f"✍️ Writing conclusion for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        conclusion = await write_conclusion(
            query=self.researcher.query,
            context=report_content,
            config=self.researcher.cfg,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            cost_callback=self.researcher.add_costs,
            websocket=self.researcher.websocket,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "conclusion_written",
                f"📝 Conclusion written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return conclusion

    async def write_introduction(self):
        """Write the introduction section of the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_introduction",
                f"✍️ Writing introduction for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        introduction = await write_report_introduction(
            query=self.researcher.query,
            context=self.researcher.context,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            config=self.researcher.cfg,
            websocket=self.researcher.websocket,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "introduction_written",
                f"📝 Introduction written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return introduction

    async def get_subtopics(self):
        """Retrieve subtopics for the research."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_subtopics",
                f"🌳 Generating subtopics for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        subtopics = await construct_subtopics(
            task=self.researcher.query,
            data=self.researcher.context,
            config=self.researcher.cfg,
            subtopics=self.researcher.subtopics,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subtopics_generated",
                f"📊 Subtopics generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return subtopics

    async def get_draft_section_titles(self, current_subtopic: str):
        """Generate draft section titles for the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_draft_sections",
                f"📑 Generating draft section titles for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        draft_section_titles = await generate_draft_section_titles(
            query=self.researcher.query,
            current_subtopic=current_subtopic,
            context=self.researcher.context,
            role=self.researcher.cfg.agent_role or self.researcher.role,
            websocket=self.researcher.websocket,
            config=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "draft_sections_generated",
                f"🗂️ Draft section titles generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return draft_section_titles
