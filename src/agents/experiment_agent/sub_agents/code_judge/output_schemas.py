"""
Output schemas for code judge agent.

Defines structured output formats for code consistency evaluation
and implementation feedback.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class CodeIssue(BaseModel):
    """Individual code issue identified during review."""

    file_path: str = Field(description="Path to the file with the issue")
    issue_type: str = Field(
        description="Type of issue: 'logic_error', 'missing_implementation', 'inconsistency', 'quality'"
    )
    severity: str = Field(description="Severity level: 'critical', 'major', 'minor'")
    description: str = Field(description="Detailed description of the issue")
    expected: str = Field(description="Expected implementation based on plan/analysis")
    actual: str = Field(description="Actual implementation found in code")
    suggestion: str = Field(description="Suggestion for fixing the issue")
    line_numbers: Optional[str] = Field(
        default=None, description="Line numbers where the issue occurs (if applicable)"
    )


class CodeJudgeOutput(BaseModel):
    """
    Output structure for code consistency evaluation.

    This structure contains the evaluation result and detailed feedback
    for code implementation review.
    """

    is_consistent: bool = Field(
        description="Whether the code implementation is consistent with plan and analysis"
    )

    overall_assessment: str = Field(
        description="High-level assessment of the code implementation quality and consistency"
    )

    # Consistency evaluation
    plan_consistency_score: float = Field(
        description="Score (0-1) indicating consistency with the code plan"
    )
    analysis_consistency_score: float = Field(
        description="Score (0-1) indicating consistency with the pre-analysis"
    )

    # Detailed feedback
    issues: List[CodeIssue] = Field(
        default_factory=list,
        description="List of issues found in the implementation",
    )

    strengths: List[str] = Field(
        default_factory=list,
        description="Aspects of the implementation that are well done",
    )

    missing_components: List[str] = Field(
        default_factory=list,
        description="Components specified in plan but missing in implementation",
    )

    extra_components: List[str] = Field(
        default_factory=list,
        description="Components implemented but not specified in plan",
    )

    # Recommendations
    priority_fixes: List[str] = Field(
        default_factory=list,
        description="High-priority fixes that should be addressed first",
    )

    implementation_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving the implementation",
    )

    next_steps: str = Field(
        description="Recommended next steps based on the evaluation"
    )
