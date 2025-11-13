"""
Input converters for transforming raw files into standardized ResearchInput format.

This module provides utilities to convert:
- LaTeX (.tex) files to ResearchInput with input_type="paper"
- Idea JSON files to ResearchInput with input_type="idea"
"""

import json
import re
from pathlib import Path
from typing import Optional, Union

from src.agents.experiment_agent.input_processing.schemas import (
    ResearchInput,
    IdeaMetadata,
    PaperMetadata,
)


def extract_latex_title(latex_content: str) -> Optional[str]:
    """
    Extract title from LaTeX content.

    Args:
        latex_content: LaTeX source code

    Returns:
        Extracted title or None
    """
    # Look for \title{...}
    title_match = re.search(r"\\title\{([^}]+)\}", latex_content)
    if title_match:
        return title_match.group(1).strip()
    return None


def extract_latex_abstract(latex_content: str) -> Optional[str]:
    """
    Extract abstract from LaTeX content.

    Args:
        latex_content: LaTeX source code

    Returns:
        Extracted abstract or None
    """
    # Look for \begin{abstract}...\end{abstract}
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex_content, re.DOTALL
    )
    if abstract_match:
        return abstract_match.group(1).strip()
    return None


def extract_latex_authors(latex_content: str) -> list:
    """
    Extract authors from LaTeX content.

    Args:
        latex_content: LaTeX source code

    Returns:
        List of author names
    """
    authors = []
    # Look for \author{...}
    author_match = re.search(r"\\author\{([^}]+)\}", latex_content)
    if author_match:
        author_text = author_match.group(1)
        # Split by common separators
        authors = [a.strip() for a in re.split(r"[,;]|\s+and\s+", author_text)]
    return [a for a in authors if a]  # Filter empty strings


def convert_tex_to_research_input(
    tex_path: Union[str, Path], encoding: str = "utf-8"
) -> ResearchInput:
    """
    Convert a LaTeX (.tex) file to standardized ResearchInput.

    Args:
        tex_path: Path to .tex file
        encoding: File encoding (default: utf-8)

    Returns:
        ResearchInput with input_type="paper"
    """
    tex_path = Path(tex_path)

    with open(tex_path, "r", encoding=encoding) as f:
        latex_content = f.read()

    # Extract metadata
    title = extract_latex_title(latex_content)
    abstract = extract_latex_abstract(latex_content)
    authors = extract_latex_authors(latex_content)

    # Build metadata
    paper_metadata = PaperMetadata(authors=authors, abstract=abstract)

    metadata_dict = {
        "source_file": str(tex_path),
        "authors": authors,
        "abstract": abstract,
    }

    return ResearchInput.from_paper(
        latex_content=latex_content, title=title, metadata=metadata_dict
    )


def convert_idea_json_to_research_input(
    json_path: Union[str, Path], encoding: str = "utf-8"
) -> ResearchInput:
    """
    Convert an idea JSON file to standardized ResearchInput.

    Args:
        json_path: Path to idea.json file
        encoding: File encoding (default: utf-8)

    Returns:
        ResearchInput with input_type="idea"
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding=encoding) as f:
        data = json.load(f)

    # Extract main content from messages
    content_parts = []
    title = None

    if "messages" in data:
        for msg in data["messages"]:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                if content:
                    content_parts.append(content)

                    # Try to extract title from first message
                    if not title and "SELECTED IDEA" in content:
                        title_match = re.search(r"###\s+(.+?)(?:\n|$)", content)
                        if title_match:
                            title = title_match.group(1).strip()

    idea_content = "\n\n".join(content_parts)

    # Extract metadata from context_variables
    metadata_dict = {"source_file": str(json_path)}

    if "context_variables" in data:
        ctx_vars = data["context_variables"]

        idea_metadata = IdeaMetadata(
            working_dir=ctx_vars.get("working_dir"),
            date_limit=ctx_vars.get("date_limit"),
            reference_codebases=ctx_vars.get("prepare_result", {}).get(
                "reference_codebases", []
            ),
            reference_papers=ctx_vars.get("prepare_result", {}).get(
                "reference_papers", []
            ),
        )

        # Add idea evaluation scores if present
        if "idea_evaluation" in ctx_vars and ctx_vars["idea_evaluation"]:
            # Get the top-scored idea (last one is usually the selected one)
            top_idea = ctx_vars["idea_evaluation"][-1]
            idea_metadata.idea_scores = {
                "innovation": top_idea.get("innovation_score"),
                "effectiveness": top_idea.get("effectiveness_score"),
                "feasibility": top_idea.get("feasibility_score"),
                "total": top_idea.get("total_score"),
            }
            idea_metadata.evaluation_summary = top_idea.get("recommendation")

        metadata_dict.update(
            {
                "working_dir": idea_metadata.working_dir,
                "date_limit": idea_metadata.date_limit,
                "reference_codebases": idea_metadata.reference_codebases,
                "reference_papers": idea_metadata.reference_papers,
                "idea_scores": idea_metadata.idea_scores,
                "evaluation_summary": idea_metadata.evaluation_summary,
            }
        )

    return ResearchInput.from_idea(
        idea_content=idea_content, title=title, metadata=metadata_dict
    )


def load_research_input(
    file_path: Union[str, Path], encoding: str = "utf-8"
) -> ResearchInput:
    """
    Auto-detect file type and convert to ResearchInput.

    Args:
        file_path: Path to input file (.tex or .json)
        encoding: File encoding (default: utf-8)

    Returns:
        ResearchInput with appropriate input_type

    Raises:
        ValueError: If file type is not supported
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".tex":
        return convert_tex_to_research_input(file_path, encoding=encoding)
    elif file_path.suffix.lower() == ".json":
        return convert_idea_json_to_research_input(file_path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


# Convenience functions for command-line usage
def save_research_input_as_json(
    research_input: ResearchInput,
    output_path: Union[str, Path],
    encoding: str = "utf-8",
) -> None:
    """
    Save ResearchInput to JSON file.

    Args:
        research_input: ResearchInput instance
        output_path: Path to output JSON file
        encoding: File encoding (default: utf-8)
    """
    output_path = Path(output_path)

    with open(output_path, "w", encoding=encoding) as f:
        json.dump(research_input.model_dump(), f, indent=2, ensure_ascii=False)


def save_research_input_as_text(
    research_input: ResearchInput,
    output_path: Union[str, Path],
    encoding: str = "utf-8",
) -> None:
    """
    Save ResearchInput to text file.

    Args:
        research_input: ResearchInput instance
        output_path: Path to output text file
        encoding: File encoding (default: utf-8)
    """
    output_path = Path(output_path)

    with open(output_path, "w", encoding=encoding) as f:
        f.write(research_input.to_text())
