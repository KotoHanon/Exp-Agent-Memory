"""
Output schemas for experiment analysis agent.

Defines structured output formats for experiment results analysis,
including improvement suggestions for ideas and code plans.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class MetricAnalysis(BaseModel):
    """Analysis of a specific metric."""

    metric_name: str = Field(
        description="Name of the metric (e.g., 'accuracy', 'loss')"
    )
    actual_value: Optional[float] = Field(
        default=None, description="Actual value achieved in experiment"
    )
    expected_value: Optional[float] = Field(
        default=None, description="Expected value from analysis/plan"
    )
    meets_expectation: bool = Field(description="Whether the metric meets expectations")
    analysis: str = Field(description="Detailed analysis of this metric's performance")


class ExperimentAnalysisOutput(BaseModel):
    """
    Output structure for experiment analysis.

    This structure contains the analysis of experiment results,
    comparing them with pre-analysis and plan expectations,
    and providing improvement suggestions.
    """

    # Overall assessment
    meets_requirements: bool = Field(
        description="Whether the experiment meets the requirements from pre-analysis and plan"
    )

    overall_analysis: str = Field(
        description="High-level analysis of experiment results and their alignment with expectations"
    )

    # Metric analysis
    metrics_analysis: List[MetricAnalysis] = Field(
        default_factory=list,
        description="Detailed analysis of individual metrics",
    )

    # Pre-analysis alignment
    pre_analysis_alignment: str = Field(
        description="Analysis of how well results align with pre-analysis expectations"
    )

    key_innovations_validated: bool = Field(
        description="Whether the key innovations from pre-analysis are validated"
    )

    innovations_analysis: str = Field(
        description="Analysis of how well key innovations performed"
    )

    # Plan alignment
    plan_alignment: str = Field(
        description="Analysis of how well implementation follows the code plan"
    )

    plan_completeness: float = Field(
        description="Score (0-1) indicating how completely the plan was implemented"
    )

    # Improvement suggestions
    idea_needs_improvement: bool = Field(
        description="Whether the research idea needs improvement"
    )

    idea_improvements: str = Field(
        default="",
        description="Specific improvements for the research idea (empty if not needed)",
    )

    plan_needs_improvement: bool = Field(
        description="Whether the code plan needs improvement"
    )

    plan_improvements: str = Field(
        default="",
        description="Specific improvements for the code plan (empty if not needed)",
    )

    # Additional findings
    unexpected_findings: List[str] = Field(
        default_factory=list,
        description="Unexpected findings or observations from the experiment",
    )

    potential_issues: List[str] = Field(
        default_factory=list,
        description="Potential issues identified in the experiment",
    )

    strengths: List[str] = Field(
        default_factory=list,
        description="Strengths and positive aspects of the implementation",
    )

    # Recommendations
    next_steps: str = Field(description="Recommended next steps based on analysis")

    priority_actions: List[str] = Field(
        default_factory=list,
        description="Prioritized list of actions to take",
    )
