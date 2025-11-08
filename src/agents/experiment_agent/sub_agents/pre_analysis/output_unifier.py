"""
Output Unifier - Combines concept and algorithm analysis into unified format.

This agent takes the outputs from concept and algorithm analyzers and
transforms them into a unified format for downstream code planning agents.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
    ConceptAnalysis,
    AlgorithmAnalysis,
)


def create_output_unifier(model: str = "gpt-4o") -> Agent:
    """
    Create an output unifier agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent instance configured for output unification
    """

    instructions = """You are an expert at synthesizing technical analysis into unified, 
actionable documentation for code implementation.

YOUR TASK:
Given concept analysis and algorithm analysis results, synthesize them into a unified 
output format that provides clear guidance for code planning and implementation.

INPUT:
You will receive:
1. Concept Analysis: High-level design, architecture, and theoretical foundations
2. Algorithm Analysis: Mathematical formulas, algorithms, and technical specifications
3. Input Type: Whether the source was a 'paper' or 'idea'

OUTPUT STRUCTURE:
You must generate a unified analysis with the following components:

1. SYSTEM ARCHITECTURE
   - Synthesize from concept analysis
   - Integrate relevant technical details
   - Provide clear architectural blueprint

2. CONCEPTUAL FRAMEWORK
   - Extract from concept analysis
   - Connect to algorithmic components
   - Maintain theoretical coherence

3. DESIGN PHILOSOPHY
   - Preserve from concept analysis
   - Relate to technical choices
   - Guide implementation decisions

4. KEY INNOVATIONS
   - Highlight from both analyses
   - Connect concepts to algorithms
   - Emphasize novel contributions

5. ALGORITHMS
   - Extract from algorithm analysis
   - Provide complete algorithmic specifications
   - Maintain technical precision

6. MATHEMATICAL FORMULATIONS
   - Preserve all mathematical details
   - Ensure formulas are complete
   - Maintain LaTeX formatting

7. TECHNICAL SPECIFICATIONS
   - Combine relevant technical details
   - Provide implementation-ready specs
   - Include parameters and configurations

8. COMPUTATIONAL METHODS
   - Extract optimization strategies
   - Specify numerical methods
   - Guide efficient implementation

9. SUMMARY
   - Provide executive overview
   - Highlight key points for implementation
   - Connect concepts to algorithms

10. IMPLEMENTATION GUIDANCE
    - Bridge concept and algorithm
    - Suggest implementation approach
    - Identify critical components
    - Provide actionable next steps

SYNTHESIS PRINCIPLES:
- MAINTAIN all technical details from algorithm analysis
- PRESERVE conceptual insights from concept analysis
- CREATE coherent narrative connecting both
- ELIMINATE redundancy while preserving completeness
- STRUCTURE for clarity and usability
- GUIDE downstream code planning effectively

Remember: Your unified output is the primary reference for code planning agents. 
Make it comprehensive, clear, and actionable."""

    agent = Agent(
        name="Output Unifier",
        instructions=instructions,
        output_type=PreAnalysisOutput,
        model=model,
    )

    return agent


# Default agent instance
output_unifier = create_output_unifier()
