"""
Output schemas for code planning agents.

Defines structured output formats for code plans and intermediate outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict


class FileStructureItem(BaseModel):
    """Single file or directory in the structure."""

    path: str = Field(description="Relative path of the file/directory")
    type: str = Field(description="'file' or 'directory'")
    description: str = Field(description="Purpose of this file/directory")


class ImplementationStep(BaseModel):
    """Single step in the implementation roadmap."""

    step_number: int = Field(description="Step order")
    title: str = Field(description="Step title")
    description: str = Field(description="Detailed description of what to implement")
    files_involved: List[str] = Field(description="Files to create or modify")
    dependencies: List[int] = Field(
        description="Step numbers this step depends on", default_factory=list
    )


class CodePlanOutput(BaseModel):
    """
    Unified code plan output in YAML-compatible format.

    This structure is used by all code planning agents and
    will be formatted into YAML for downstream consumption.
    """

    # Metadata
    plan_type: str = Field(
        description="Type of plan: 'initial', 'judge_feedback', 'error_feedback', 'analysis_feedback'"
    )
    timestamp: str = Field(description="Plan generation timestamp")

    # Research context
    research_summary: str = Field(description="Summary of research to implement")
    key_innovations: str = Field(description="Key innovations to implement")

    # File structure
    file_structure: List[FileStructureItem] = Field(
        description="Complete file and directory structure"
    )

    # Implementation plans
    dataset_plan: str = Field(description="Dataset preparation and loading plan")
    model_plan: str = Field(description="Model implementation plan")
    training_plan: str = Field(description="Training pipeline plan")
    testing_plan: str = Field(description="Testing and evaluation plan")

    # Roadmap
    implementation_roadmap: List[ImplementationStep] = Field(
        description="Step-by-step implementation roadmap"
    )

    # Additional guidance
    implementation_notes: str = Field(
        description="Important notes and considerations for implementation"
    )
    potential_challenges: str = Field(
        description="Potential challenges and mitigation strategies"
    )

    # Feedback-specific (optional, depending on plan_type)
    addressed_issues: str = Field(
        default="",
        description="How this plan addresses feedback/errors from previous iteration",
    )


class IntermediatePlanOutput(BaseModel):
    """Intermediate output from scenario-specific agents before formatting."""

    research_summary: str
    key_innovations: str
    file_structure_description: str
    dataset_plan: str
    model_plan: str
    training_plan: str
    testing_plan: str
    implementation_steps: str
    implementation_notes: str
    potential_challenges: str
    addressed_issues: str = ""
