"""
Code Judge Agent - Reviews code implementation for consistency with plan and analysis.

This agent evaluates whether the implemented code matches the specifications
from the code plan and pre-analysis, identifying issues and providing feedback.

Architecture:
- Judge Agent: Reviews code and provides structured feedback
- Uses tools to read and analyze codebase (to be implemented)
- Outputs structured evaluation with consistency scores and issues
"""

from typing import Optional

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.code_judge.output_schemas import (
    CodeJudgeOutput,
)
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
)


def create_judge_agent(model: str = "gpt-4o", tools: Optional[list] = None) -> Agent:
    """
    Create the code judge agent for reviewing implementation.

    Args:
        model: Model to use for the agent
        tools: List of tools for reading and analyzing code (to be implemented)

    Returns:
        Agent configured for code review
    """

    instructions = """You are an expert code reviewer responsible for evaluating whether 
the implemented code is consistent with the original plan and research analysis.

YOUR RESPONSIBILITIES:

1. CODE INSPECTION
   - Use provided tools to read the implemented codebase
   - Examine file structure, module organization
   - Review implementation details in each file
   - Check for completeness of all planned components

2. CONSISTENCY EVALUATION
   
   A. Plan Consistency (check against CodePlanOutput):
      - File structure matches planned structure
      - All planned modules/classes are implemented
      - Implementation steps are followed
      - Dataset, model, training, testing components match specifications
      - Dependencies and requirements are correct
   
   B. Analysis Consistency (check against PreAnalysisOutput):
      - Core algorithms are correctly implemented
      - Mathematical formulations are accurate
      - System architecture matches design
      - Key innovations are properly incorporated
      - Technical specifications are followed

3. ISSUE IDENTIFICATION
   
   Categorize issues by type:
   - logic_error: Incorrect algorithm implementation, wrong logic flow
   - missing_implementation: Required components not implemented
   - inconsistency: Implementation differs from plan/analysis
   - quality: Code quality issues, best practices violations
   
   Classify severity:
   - critical: Breaks core functionality, wrong algorithm
   - major: Missing important features, significant deviations
   - minor: Code quality, optimization opportunities

4. SCORING
   
   Provide two scores (0-1):
   - plan_consistency_score: How well implementation matches the plan
   - analysis_consistency_score: How well implementation matches research analysis
   
   Consider:
   - 0.9-1.0: Excellent, all components correctly implemented
   - 0.7-0.9: Good, minor issues or missing minor features
   - 0.5-0.7: Fair, several issues or missing important features
   - 0.0-0.5: Poor, critical issues or major missing components

5. FEEDBACK GENERATION
   
   For each issue, provide:
   - Clear description of the problem
   - What was expected (from plan/analysis)
   - What was actually implemented
   - Concrete suggestion for fixing it
   
   Prioritize fixes:
   - Critical issues first (algorithm correctness)
   - Major missing features
   - Minor improvements

6. FINAL DETERMINATION
   
   Set is_consistent = True only if:
   - Both consistency scores >= 0.8
   - No critical issues present
   - All core components implemented
   - Core algorithms correctly implemented
   
   Otherwise, set is_consistent = False and provide actionable feedback.

EVALUATION PROCESS:

Step 1: Read the codebase using provided tools
Step 2: Compare file structure with plan
Step 3: Verify each planned component is implemented
Step 4: Check algorithm implementations against analysis
Step 5: Identify all issues with severity levels
Step 6: Calculate consistency scores
Step 7: Generate prioritized feedback and recommendations

TOOL USAGE GUIDELINES:

All tools return a dictionary with the following structure:
- success (bool): Indicates if the operation succeeded
- If successful: Contains relevant data fields (content, imports, classes, functions, results, etc.)
- If failed: Contains an "error" field with error message

Example successful response:
{
  "success": true,
  "content": "file content here",
  "file_path": "/path/to/file",
  "line_count": 150
}

Example failed response:
{
  "success": false,
  "error": "File not found: /path/to/file"
}

Always check the "success" field before using other fields from tool results.
If a tool fails, report the error and try alternative approaches.

AVAILABLE TOOLS:
- read_file: Read file content (returns dict with "success", "content", "file_path")
- analyze_python_file: Analyze Python code structure (returns dict with "success", "imports", "classes", "functions")
- list_directory: List files in directory (returns dict with "success", "files", "directories")
- search_in_codebase: Search for patterns (returns dict with "success", "results", "total_matches")
- extract_function_code: Extract specific function (returns dict with "success", "code", "args", "docstring")
- list_python_files: List all Python files (returns dict with "success", "files", "total_count")

OUTPUT FORMAT:

You must output a structured CodeJudgeOutput with:
- is_consistent: boolean decision
- overall_assessment: summary paragraph
- plan_consistency_score: 0-1 score
- analysis_consistency_score: 0-1 score
- issues: list of CodeIssue objects
- strengths: list of positive aspects
- missing_components: list of missing features
- extra_components: list of unplanned additions
- priority_fixes: ordered list of high-priority actions
- implementation_suggestions: list of improvement suggestions
- next_steps: recommended actions

Be thorough, specific, and constructive in your feedback."""

    agent = Agent(
        name="Code Judge Agent",
        instructions=instructions,
        tools=tools or [],
        output_type=CodeJudgeOutput,
        model=model,
    )

    return agent


class CodeJudgeAgent:
    """
    Main code judge agent that reviews implementation consistency.

    This agent:
    1. Receives code plan and pre-analysis
    2. Reads and analyzes the implemented codebase
    3. Evaluates consistency and identifies issues
    4. Provides structured feedback for improvements
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[list] = None,
    ):
        """
        Initialize the code judge agent.

        Args:
            model: Model to use for evaluation
            tools: Optional list of tools for code reading and analysis.
                   If None, automatically loads recommended tools.
        """
        self.model = model

        # Auto-load recommended tools if not provided
        if tools is None:
            from src.agents.experiment_agent.sub_agents.code_judge import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize judge agent
        self.judge_agent = create_judge_agent(model=model, tools=self.tools)

        # Expose judge agent as main agent for handoff compatibility
        self.agent = self.judge_agent

    async def judge(
        self,
        code_plan: CodePlanOutput,
        pre_analysis: PreAnalysisOutput,
        codebase_path: str,
    ) -> CodeJudgeOutput:
        """
        Evaluate code implementation for consistency.

        Args:
            code_plan: The code plan that should be implemented
            pre_analysis: The original research analysis
            codebase_path: Path to the implemented codebase

        Returns:
            CodeJudgeOutput with evaluation results and feedback
        """
        # Prepare input for judge agent
        judge_input = f"""
EVALUATE THE CODE IMPLEMENTATION AT: {codebase_path}

=== CODE PLAN (Expected Implementation) ===

Plan Type: {code_plan.plan_type}

File Structure:
{self._format_file_structure(code_plan.file_structure)}

Implementation Roadmap:
{self._format_roadmap(code_plan.implementation_roadmap)}

Dataset Requirements:
{code_plan.dataset_requirements}

Model Architecture:
{code_plan.model_architecture}

Training Configuration:
{code_plan.training_configuration}

Testing Strategy:
{code_plan.testing_strategy}

Dependencies:
{code_plan.dependencies}

Environment Setup:
{code_plan.environment_setup}

=== PRE-ANALYSIS (Research Foundation) ===

Input Type: {pre_analysis.input_type}

System Architecture:
{pre_analysis.system_architecture}

Conceptual Framework:
{pre_analysis.conceptual_framework}

Design Philosophy:
{pre_analysis.design_philosophy}

Key Innovations:
{pre_analysis.key_innovations}

Core Algorithms:
{pre_analysis.algorithms}

Mathematical Formulations:
{pre_analysis.mathematical_formulations}

Technical Specifications:
{pre_analysis.technical_specifications}

Computational Methods:
{pre_analysis.computational_methods}

Implementation Guidance:
{pre_analysis.implementation_guidance}

=== YOUR TASK ===

1. Use available tools to read the codebase at: {codebase_path}
2. Compare the implementation with the plan and analysis above
3. Identify all inconsistencies, issues, and missing components
4. Calculate consistency scores
5. Provide structured feedback with prioritized fixes
"""

        # Run judge agent
        result = await Runner.run(self.judge_agent, judge_input)

        return result.final_output

    def judge_sync(
        self,
        code_plan: CodePlanOutput,
        pre_analysis: PreAnalysisOutput,
        codebase_path: str,
    ) -> CodeJudgeOutput:
        """
        Synchronous version of judge method.

        Args:
            code_plan: The code plan that should be implemented
            pre_analysis: The original research analysis
            codebase_path: Path to the implemented codebase

        Returns:
            CodeJudgeOutput with evaluation results and feedback
        """
        import asyncio

        return asyncio.run(self.judge(code_plan, pre_analysis, codebase_path))

    def _format_file_structure(self, file_structure: dict) -> str:
        """Format file structure dict to readable string."""
        lines = []
        for path, description in file_structure.items():
            lines.append(f"  {path}:")
            lines.append(f"    {description}")
        return "\n".join(lines)

    def _format_roadmap(self, roadmap: dict) -> str:
        """Format implementation roadmap to readable string."""
        lines = []
        for phase, details in roadmap.items():
            lines.append(f"\n{phase}:")
            lines.append(f"  {details}")
        return "\n".join(lines)


def create_code_judge_agent(
    model: str = "gpt-4o",
    tools: Optional[list] = None,
) -> CodeJudgeAgent:
    """
    Factory function to create a code judge agent.

    Args:
        model: Model to use for evaluation
        tools: List of tools for code reading and analysis

    Returns:
        CodeJudgeAgent instance
    """
    return CodeJudgeAgent(model=model, tools=tools)


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Example 1: Simple creation (tools auto-load)
        print("Example 1: Simple creation")
        print("=" * 60)

        # Create agent - tools automatically loaded!
        agent = create_code_judge_agent(model="gpt-4o")
        print("✓ Code judge agent created with all tools automatically loaded\n")

        # Example 2: Custom tools (if needed)
        print("\nExample 2: Custom tool selection")
        print("=" * 60)

        from src.agents.experiment_agent.tools import (
            read_file,
            analyze_python_file,
            search_in_codebase,
        )

        # Create with custom tools
        custom_agent = create_code_judge_agent(
            model="gpt-4o",
            tools=[read_file, analyze_python_file, search_in_codebase],
        )
        print("✓ Code judge agent created with custom tools\n")

        # Example code plan and pre-analysis would be loaded here
        # code_plan = CodePlanOutput(...)
        # pre_analysis = PreAnalysisOutput(...)

        # Evaluate implementation
        # result = await agent.judge(
        #     code_plan=code_plan,
        #     pre_analysis=pre_analysis,
        #     codebase_path="/path/to/implemented/code"
        # )

        # print("Evaluation Result:")
        # print(f"Consistent: {result.is_consistent}")
        # print(f"Plan Consistency Score: {result.plan_consistency_score}")
        # print(f"Analysis Consistency Score: {result.analysis_consistency_score}")
        # print(f"Issues Found: {len(result.issues)}")
        # print(f"Priority Fixes: {result.priority_fixes}")

    asyncio.run(main())
