"""
Code Implementation Agent - Main orchestrator using handoff mechanism.

This agent uses OpenAI agents library's handoff mechanism to route inputs
to appropriate implementation agents and produces structured code implementation output.

Architecture:
- Triage Agent: Determines scenario and hands off to appropriate implementer
- Initial Implementation Agent: Implements code from scratch based on plan
- Fix Implementation Agent: Fixes code based on feedback
- Output Unifier: Formats results into structured output
"""

from typing import Dict, Optional
from datetime import datetime

from agents import Agent, Runner, handoff

from src.agents.experiment_agent.sub_agents.code_implement.output_schemas import (
    CodeImplementOutput,
    IntermediateImplementOutput,
)
from src.agents.experiment_agent.sub_agents.code_implement.initial_implement_agent import (
    create_initial_implement_agent,
)
from src.agents.experiment_agent.sub_agents.code_implement.fix_implement_agent import (
    create_fix_implement_agent,
)
from src.agents.experiment_agent.sub_agents.code_implement.output_unifier import (
    create_output_unifier,
)


def create_triage_agent(
    initial_implementer: Agent,
    fix_implementer: Agent,
) -> Agent:
    """
    Create a triage agent that uses handoffs to route to appropriate implementers.

    Args:
        initial_implementer: Agent for initial implementation
        fix_implementer: Agent for fixing implementations

    Returns:
        Triage agent with handoffs configured
    """

    instructions = """You are an implementation triage agent responsible for determining which 
implementation scenario applies and handing off to the appropriate specialized implementer.

ANALYZE the input to determine which scenario it represents:

1. INITIAL IMPLEMENTATION
   - Input: CodePlanOutput from code_plan_agent
   - Contains implementation plan with file structure, roadmap
   - No existing code or error feedback
   - First-time implementation
   → Handoff to Initial Implementation Agent

2. FIX IMPLEMENTATION
   - Input: CodePlanOutput + Error feedback/code review issues
   - Contains information about existing code problems
   - May reference code_judge_agent output or runtime errors
   - Need to fix existing code
   → Handoff to Fix Implementation Agent

IMPORTANT: After analyzing the input, you MUST handoff to the appropriate 
implementation agent. Do not implement code yourself."""

    triage = Agent(
        name="Implementation Triage Agent",
        instructions=instructions,
        handoffs=[
            handoff(
                initial_implementer,
                tool_description_override="Handoff to Initial Implementation Agent for implementing code from scratch based on plan.",
            ),
            handoff(
                fix_implementer,
                tool_description_override="Handoff to Fix Implementation Agent for fixing existing code based on feedback.",
            ),
        ],
    )

    return triage


class CodeImplementAgent:
    """
    Main code implementation agent that orchestrates the entire implementation workflow using handoffs.

    This agent:
    1. Uses triage agent to determine scenario
    2. Hands off to appropriate implementation agent via handoff mechanism
    3. Formats output into structured format
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[Dict[str, list]] = None,
    ):
        """
        Initialize the code implementation agent system.

        Args:
            model: Model to use for all agents
            working_dir: Working directory for code generation
            tools: Optional dictionary mapping scenario types to their tools.
                   If None, automatically loads recommended tools.
                   e.g., {"initial": [...], "fix": [...]}
        """
        self.model = model
        self.working_dir = working_dir

        # Auto-load recommended tools if not provided
        if tools is None:
            from src.agents.experiment_agent.sub_agents.code_implement import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize implementation agents
        self.initial_implementer = create_initial_implement_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools.get("initial", []),
        )

        self.fix_implementer = create_fix_implement_agent(
            model=model, working_dir=working_dir, tools=self.tools.get("fix", [])
        )

        # Initialize triage agent with handoffs
        self.triage_agent = create_triage_agent(
            self.initial_implementer,
            self.fix_implementer,
        )

        # Initialize output unifier
        self.output_unifier = create_output_unifier(model=model)

        # Expose triage agent as main agent for handoff compatibility
        self.agent = self.triage_agent

    async def implement(self, input_data: str) -> CodeImplementOutput:
        """
        Generate code implementation based on input.

        Args:
            input_data: Input string containing plan and optional feedback

        Returns:
            CodeImplementOutput with complete implementation
        """
        # Step 1: Run triage agent (will automatically handoff to appropriate implementer)
        implementation_result = await Runner.run(self.triage_agent, input_data)

        # The final_output will be from the implementer that was handed off to
        intermediate_output: IntermediateImplementOutput = (
            implementation_result.final_output
        )

        # Determine scenario based on which agent produced the output
        active_agent_name = implementation_result.agent.name
        if "Initial" in active_agent_name:
            scenario = "initial"
        elif "Fix" in active_agent_name:
            scenario = "fix"
        else:
            scenario = "initial"  # default

        print(f"Implemented using: {active_agent_name} (scenario: {scenario})")

        # Step 2: Format output
        unifier_input = f"""
Implementation Type: {scenario}
Timestamp: {datetime.now().isoformat()}

=== INTERMEDIATE IMPLEMENTATION OUTPUT ===

Files Description:
{intermediate_output.files_description}

Implementation Summary:
{intermediate_output.implementation_summary_text}

Setup Instructions:
{intermediate_output.setup_instructions}

Usage Examples:
{intermediate_output.usage_examples}

Known Limitations:
{intermediate_output.known_limitations}

Next Steps:
{intermediate_output.next_steps}

Issues Addressed:
{intermediate_output.issues_addressed}
"""

        unifier_result = await Runner.run(self.output_unifier, unifier_input)

        return unifier_result.final_output

    def implement_sync(self, input_data: str) -> CodeImplementOutput:
        """
        Synchronous version of implement method.

        Args:
            input_data: Input string containing plan and optional feedback

        Returns:
            CodeImplementOutput with complete implementation
        """
        import asyncio

        return asyncio.run(self.implement(input_data))


def create_code_implement_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[Dict[str, list]] = None,
) -> CodeImplementAgent:
    """
    Factory function to create a code implementation agent system.

    Args:
        model: Model to use for all agents
        working_dir: Working directory for code generation
        tools: Dictionary mapping scenario types to their tools

    Returns:
        CodeImplementAgent instance
    """
    return CodeImplementAgent(model=model, working_dir=working_dir, tools=tools)


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create the code implementation agent
        agent = create_code_implement_agent(model="gpt-4o", working_dir="/workspace")

        # Example with initial implementation
        input_data = """
        CodePlanOutput:
        - File Structure: ...
        - Dataset Plan: ...
        - Model Plan: ...
        """

        result = await agent.implement(input_data)

        print("Code Implementation Complete:")
        print(f"Type: {result.implementation_type}")
        print(f"Files Generated: {len(result.generated_files)}")
        print(f"Summary: {result.implementation_summary}")

    asyncio.run(main())
