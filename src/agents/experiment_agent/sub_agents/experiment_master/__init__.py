"""Experiment master agent module with rule-based state machine and caching."""

from src.agents.experiment_agent.sub_agents.experiment_master.experiment_master_agent import (
    create_experiment_master_agent,
)
from src.agents.experiment_agent.sub_agents.experiment_master.output_schemas import (
    ExperimentMasterOutput,
    WorkflowStep,
)
from src.agents.experiment_agent.sub_agents.experiment_master.workflow_state_machine import (
    WorkflowStateMachine,
    WorkflowState,
    WorkflowContext,
    StateTransition,
)
from src.agents.experiment_agent.sub_agents.experiment_master.cache_manager import (
    CacheManager,
)


def get_all_agent_tools():
    """
    Get tools for all sub-agents in experiment master workflow.

    Returns:
        Dictionary mapping agent types to their tools
    """
    from src.agents.experiment_agent.tools import get_tools_for_agent

    return {
        "pre_analysis": get_tools_for_agent("pre_analysis"),
        "code_plan": get_tools_for_agent("code_plan"),
        "code_implement": get_tools_for_agent("code_implement"),
        "code_judge": get_tools_for_agent("code_judge"),
        "experiment_execute": get_tools_for_agent("experiment_execute"),
        "experiment_analysis": get_tools_for_agent("experiment_analysis"),
    }


__all__ = [
    "create_experiment_master_agent",
    "ExperimentMasterOutput",
    "WorkflowStep",
    "WorkflowStateMachine",
    "WorkflowState",
    "WorkflowContext",
    "StateTransition",
    "CacheManager",
    "get_all_agent_tools",
]
