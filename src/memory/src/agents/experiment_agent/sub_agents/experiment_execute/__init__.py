"""Experiment execute agent module."""

from src.agents.experiment_agent.sub_agents.experiment_execute.experiment_execute_agent import (
    create_experiment_execute_agent,
)
from src.agents.experiment_agent.sub_agents.experiment_execute.output_schemas import (
    ExperimentExecuteOutput,
)


def get_recommended_tools():
    """
    Get recommended tools for experiment execute agent.

    Returns:
        List of tools for code execution
    """
    from src.agents.experiment_agent.tools import FILE_TOOLS, EXECUTION_TOOLS

    return FILE_TOOLS[:3] + EXECUTION_TOOLS  # read, write, list + all execution


__all__ = [
    "create_experiment_execute_agent",
    "ExperimentExecuteOutput",
    "get_recommended_tools",
]
