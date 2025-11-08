"""Code planning agent module."""

from src.agents.experiment_agent.sub_agents.code_plan.code_plan_agent import (
    create_code_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def get_recommended_tools():
    """
    Get recommended tools for code plan agent.

    Returns:
        Dictionary mapping scenario types to tool lists
    """
    from src.agents.experiment_agent.tools import FILE_TOOLS, CODE_ANALYSIS_TOOLS

    tools = FILE_TOOLS + CODE_ANALYSIS_TOOLS

    return {
        "initial": tools,
        "judge_feedback": tools,
        "error_feedback": tools,
        "analysis_feedback": tools,
    }


__all__ = ["create_code_plan_agent", "CodePlanOutput", "get_recommended_tools"]
