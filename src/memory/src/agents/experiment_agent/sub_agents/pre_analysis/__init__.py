"""Pre-analysis agent module."""

from src.agents.experiment_agent.sub_agents.pre_analysis.pre_analysis_agent import (
    create_pre_analysis_agent,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
)


def get_recommended_tools():
    """
    Get recommended tools for pre-analysis agent.

    Returns:
        Dictionary mapping agent types to tool lists
    """
    from src.agents.experiment_agent.tools import DOCUMENT_TOOLS, FILE_TOOLS

    return {
        "paper": DOCUMENT_TOOLS + FILE_TOOLS[:3],  # read, write, list
        "idea": DOCUMENT_TOOLS + FILE_TOOLS[:3],
    }


__all__ = ["create_pre_analysis_agent", "PreAnalysisOutput", "get_recommended_tools"]
