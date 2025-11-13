# Pre-Analysis Agent System

A comprehensive multi-agent system for analyzing research papers and ideas to produce unified technical specifications for code implementation.

## Architecture

```
Pre-Analysis Agent System
├── Router Agent          → Detects input type (paper/idea)
├── Paper Analyzers
│   ├── Concept Analyzer  → Extracts architectural concepts
│   └── Algorithm Analyzer → Extracts algorithms & formulas
├── Idea Analyzers
│   ├── Concept Analyzer  → Elaborates conceptual framework
│   └── Algorithm Analyzer → Generates algorithmic specs
└── Output Unifier        → Synthesizes unified output
```

## Components

### 1. Router Agent (`pre_analysis_agent.py`)
Determines whether the input is a research paper (.tex) or idea (JSON).

**Tools Required:**
- `check_file_extension`: Check file type
- `peek_file_content`: Read file headers
- `parse_file_structure`: Analyze structure

### 2. Paper Concept Analyzer (`paper_concept_analyzer.py`)
Extracts high-level concepts from research papers.

**Output:**
- System architecture and design patterns
- Conceptual framework and theory
- Design philosophy and principles
- Key innovations
- Theoretical foundations

**Tools Required:**
- `read_tex_file`: Read LaTeX sections
- `parse_latex_sections`: Extract structured sections
- `extract_figures`: Get figure descriptions

### 3. Paper Algorithm Analyzer (`paper_algorithm_analyzer.py`)
Extracts technical details from research papers.

**Output:**
- Core algorithms and procedures
- Mathematical formulations (LaTeX format)
- Technical specifications
- Computational methods
- Algorithm flow

**Tools Required:**
- `read_tex_file`: Read LaTeX content
- `extract_equations`: Extract math equations
- `parse_algorithm_blocks`: Extract pseudocode

### 4. Idea Concept Analyzer (`idea_concept_analyzer.py`)
Elaborates conceptual framework from research ideas.

**Output:**
- System architecture (elaborated)
- Conceptual framework (expanded)
- Design philosophy (articulated)
- Key innovations (highlighted)
- Theoretical basis (developed)

**Tools Required:**
- `read_json_file`: Read idea JSON
- `parse_idea_sections`: Extract sections
- `extract_evaluation_feedback`: Get evaluator insights

### 5. Idea Algorithm Analyzer (`idea_algorithm_analyzer.py`)
Generates detailed algorithmic specifications from ideas.

**Output:**
- Complete algorithm designs
- Mathematical formulations (generated)
- Technical specifications (detailed)
- Computational methods (proposed)
- Algorithm flow (designed)

**Tools Required:**
- `read_json_file`: Read idea JSON
- `parse_idea_sections`: Extract methodology
- `extract_math_formulas`: Get math descriptions

### 6. Output Unifier (`output_unifier.py`)
Synthesizes concept and algorithm analysis into unified format.

**Output:** `UnifiedAnalysisOutput`
- Consistent structure for both paper and idea inputs
- Complete conceptual and technical specifications
- Implementation guidance

## Usage

### Basic Usage

```python
import asyncio
from src.agents.experiment_agent.sub_agents.pre_analysis import (
    create_pre_analysis_agent
)

async def main():
    # Create the agent system
    agent = create_pre_analysis_agent(
        model="gpt-4o",
        tools={
            "router": [check_file_extension, peek_file_content],
            "paper": [read_tex_file, extract_equations],
            "idea": [read_json_file, parse_idea_sections]
        }
    )
    
    # Analyze a paper
    with open("paper.tex", "r", encoding="utf-8") as f:
        paper_content = f.read()
    
    result = await agent.analyze(paper_content)
    
    # Access unified output
    print(f"Input Type: {result.input_type}")
    print(f"System Architecture: {result.system_architecture}")
    print(f"Algorithms: {result.algorithms}")
    print(f"Mathematical Formulations: {result.mathematical_formulations}")
    print(f"Implementation Guidance: {result.implementation_guidance}")

asyncio.run(main())
```

### Synchronous Usage

```python
from src.agents.experiment_agent.sub_agents.pre_analysis import (
    create_pre_analysis_agent
)

# Create the agent system
agent = create_pre_analysis_agent(model="gpt-4o")

# Analyze synchronously
result = agent.analyze_sync(input_data)
```

### Using Individual Analyzers

```python
from src.agents.experiment_agent.sub_agents.pre_analysis.paper_concept_analyzer import (
    create_paper_concept_analyzer
)
from agents import Runner

# Create a specific analyzer
analyzer = create_paper_concept_analyzer(
    model="gpt-4o",
    tools=[read_tex_file, parse_latex_sections]
)

# Use it directly
result = await Runner.run(analyzer, paper_content)
concept_analysis = result.final_output
```

## Output Schema

### UnifiedAnalysisOutput

```python
{
    "input_type": "paper" | "idea",
    
    # Conceptual Analysis
    "system_architecture": str,
    "conceptual_framework": str,
    "design_philosophy": str,
    "key_innovations": str,
    
    # Technical Analysis
    "algorithms": str,
    "mathematical_formulations": str,
    "technical_specifications": str,
    "computational_methods": str,
    
    # Guidance
    "summary": str,
    "implementation_guidance": str
}
```

## Tool Requirements

Tools should be implemented in `src/agents/experiment_agent/tools/`.

### For Router Agent:
- `check_file_extension(file_path: str) -> str`
- `peek_file_content(file_path: str, lines: int) -> str`
- `parse_file_structure(content: str) -> dict`

### For Paper Analyzers:
- `read_tex_file(file_path: str, section: str = None) -> str`
- `parse_latex_sections(content: str) -> dict`
- `extract_figures(content: str) -> list`
- `extract_equations(content: str) -> list`
- `parse_algorithm_blocks(content: str) -> list`

### For Idea Analyzers:
- `read_json_file(file_path: str) -> dict`
- `parse_idea_sections(data: dict, section: str) -> str`
- `extract_evaluation_feedback(data: dict) -> dict`
- `extract_math_formulas(data: dict) -> list`

## Design Principles

### 1. Separation of Concerns
- **Concept Analysis**: High-level design and theory
- **Algorithm Analysis**: Technical specifications and implementation
- **Output Unification**: Consistent interface for downstream agents

### 2. Type-Specific Processing
- **Papers**: Extract existing content
- **Ideas**: Generate and elaborate content

### 3. Unified Output
Both input types produce the same output schema for seamless integration with code planning agents.

### 4. Modular Architecture
Each analyzer can be used independently or as part of the orchestrated system.

## Integration with Code Planning

The unified output is designed to be consumed by code planning agents:

```python
from src.agents.experiment_agent.sub_agents.pre_analysis import (
    create_pre_analysis_agent
)
from src.agents.code_plan_agent import create_code_plan_agent

# Analyze research input
pre_analysis_agent = create_pre_analysis_agent()
analysis_result = await pre_analysis_agent.analyze(input_data)

# Generate code plan
code_plan_agent = create_code_plan_agent(
    working_dir="/workspace",
    tools=[gen_code_tree_structure, read_file, plan_dataset, ...]
)

plan_input = f"""
# Research Analysis
{analysis_result.summary}

## System Architecture
{analysis_result.system_architecture}

## Algorithms
{analysis_result.algorithms}

## Mathematical Formulations
{analysis_result.mathematical_formulations}

## Implementation Guidance
{analysis_result.implementation_guidance}
"""

code_plan = await Runner.run(code_plan_agent, plan_input)
```

## Environment Setup

To use this system, ensure PYTHONPATH includes the ResearchAgent root:

```bash
export PYTHONPATH="/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent:$PYTHONPATH"
```

Or in your Python script:

```python
import sys
sys.path.insert(0, "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent")
```

## Example Workflow

1. **Input Detection**: Router identifies input type
2. **Concept Analysis**: Extract/generate high-level concepts
3. **Algorithm Analysis**: Extract/generate technical specs
4. **Output Unification**: Synthesize into unified format
5. **Code Planning**: Use unified output for implementation planning

## Notes

- All tools are assumed to be in `src/agents/experiment_agent/tools/`
- Mathematical formulas should be in LaTeX format
- The system uses OpenAI Agents SDK (agents library)
- Outputs are structured using Pydantic models
- Async/await is the primary execution model
