"""
Input schemas for experiment agent.

This module defines standardized input formats for both research papers and ideas.
The input_type field is critical for routing to appropriate analyzers.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ResearchInput(BaseModel):
    """
    Standardized input schema for experiment agent.

    This schema supports both research papers (.tex) and research ideas (JSON).
    The input_type field determines which analyzer pipeline to use.
    """

    input_type: Literal["paper", "idea"] = Field(
        ...,
        description="Type of research input: 'paper' for LaTeX papers, 'idea' for research ideas",
    )

    content: str = Field(
        ...,
        description="Main content: LaTeX source for papers, or idea description for ideas",
    )

    title: Optional[str] = Field(default=None, description="Title of the paper or idea")

    metadata: Optional[Dict] = Field(
        default_factory=dict,
        description="Additional metadata (authors, date, references, etc.)",
    )

    def to_text(self) -> str:
        """
        Convert to plain text format for agent processing.

        Returns:
            Formatted string with input type marker and content
        """
        sections = [f"INPUT_TYPE: {self.input_type.upper()}", ""]

        if self.title:
            sections.append(f"TITLE: {self.title}")
            sections.append("")

        if self.metadata:
            sections.append("METADATA:")
            for key, value in self.metadata.items():
                sections.append(f"  {key}: {value}")
            sections.append("")

        sections.append("CONTENT:")
        sections.append(self.content)

        return "\n".join(sections)

    @classmethod
    def from_paper(
        cls,
        latex_content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> "ResearchInput":
        """
        Create ResearchInput from a LaTeX paper.

        Args:
            latex_content: LaTeX source code
            title: Optional paper title
            metadata: Optional metadata dict

        Returns:
            ResearchInput with input_type="paper"
        """
        return cls(
            input_type="paper",
            content=latex_content,
            title=title,
            metadata=metadata or {},
        )

    @classmethod
    def from_idea(
        cls,
        idea_content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> "ResearchInput":
        """
        Create ResearchInput from a research idea.

        Args:
            idea_content: Idea description/proposal
            title: Optional idea title
            metadata: Optional metadata dict

        Returns:
            ResearchInput with input_type="idea"
        """
        return cls(
            input_type="idea",
            content=idea_content,
            title=title,
            metadata=metadata or {},
        )


class IdeaMetadata(BaseModel):
    """Metadata specific to research ideas."""

    working_dir: Optional[str] = None
    date_limit: Optional[str] = None
    reference_codebases: List[str] = Field(default_factory=list)
    reference_papers: List[str] = Field(default_factory=list)
    idea_scores: Optional[Dict] = Field(
        default=None, description="Innovation, effectiveness, feasibility scores"
    )
    evaluation_summary: Optional[str] = None


class PaperMetadata(BaseModel):
    """Metadata specific to research papers."""

    authors: List[str] = Field(default_factory=list)
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
