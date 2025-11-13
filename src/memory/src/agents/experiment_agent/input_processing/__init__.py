"""
Input processing module for experiment agent.

This module provides standardized input schemas and converters for:
- Research papers (LaTeX .tex files)
- Research ideas (JSON files)
"""

from src.agents.experiment_agent.input_processing.schemas import (
    ResearchInput,
    IdeaMetadata,
    PaperMetadata,
)
from src.agents.experiment_agent.input_processing.converters import (
    load_research_input,
    convert_tex_to_research_input,
    convert_idea_json_to_research_input,
    save_research_input_as_json,
    save_research_input_as_text,
)

__all__ = [
    "ResearchInput",
    "IdeaMetadata",
    "PaperMetadata",
    "load_research_input",
    "convert_tex_to_research_input",
    "convert_idea_json_to_research_input",
    "save_research_input_as_json",
    "save_research_input_as_text",
]
