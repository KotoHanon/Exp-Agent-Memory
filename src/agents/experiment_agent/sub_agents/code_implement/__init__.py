"""Code implementation agent module."""

from src.agents.experiment_agent.sub_agents.code_implement.code_implement_agent import (
    create_code_implement_agent,
)
from src.agents.experiment_agent.sub_agents.code_implement.output_schemas import (
    CodeImplementOutput,
)


def get_recommended_tools():
    """
    Get recommended tools for code implement agent.

    Returns:
        Dictionary mapping scenario types to tool lists
    """
    from src.agents.experiment_agent.tools import (
        FILE_TOOLS,
        EXECUTION_TOOLS,
        CODE_ANALYSIS_TOOLS,
    )

    # For implementation, we need file ops, execution (for syntax check, install), and analysis
    tools = FILE_TOOLS + EXECUTION_TOOLS[:6] + CODE_ANALYSIS_TOOLS

    return {
        "initial": tools,
        "fix": tools,
    }


__all__ = [
    "create_code_implement_agent",
    "CodeImplementOutput",
    "get_recommended_tools",
]
