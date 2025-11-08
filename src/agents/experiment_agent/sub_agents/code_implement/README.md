# Code Implementation Agent System

A comprehensive multi-agent system for implementing research code, supporting both initial implementation from plans and fixing existing code based on feedback.

## Architecture

```
Code Implementation Agent System
├── Triage Agent              → Determines scenario
├── Implementation Agents
│   ├── Initial Implementer   → Implements from scratch
│   └── Fix Implementer       → Fixes existing code
└── Output Unifier            → Structures results
```

## Scenarios

### 1. Initial Implementation (`initial`)
**Trigger**: First-time code implementation  
**Input**: `CodePlanOutput` from code_plan_agent  
**Purpose**: Generate complete working code from implementation plan

**Process:**
1. Parse implementation plan
2. Create file structure
3. Implement all components step-by-step
4. Generate tests and documentation

### 2. Fix Implementation (`fix`)
**Trigger**: Code has issues that need fixing  
**Input**: 
- `CodePlanOutput` (original plan)
- Error feedback from code_judge_agent or runtime errors

**Purpose**: Fix identified issues in existing code

**Process:**
1. Analyze error feedback
2. Examine existing code
3. Identify and fix root causes
4. Validate fixes
5. Update tests

## Components

### 1. Triage Agent (`code_implement_agent.py`)
Determines whether to do initial implementation or fix existing code.

**Routing Logic:**
- Checks for existence of error feedback
- Routes to appropriate implementer via handoff

### 2. Initial Implementation Agent (`initial_implement_agent.py`)
Implements complete code from scratch.

**Focus:**
- Follow implementation plan exactly
- Generate complete, runnable code
- Include all necessary files
- Add tests and documentation
- No placeholders or TODOs

**Output:** `IntermediateImplementOutput`

### 3. Fix Implementation Agent (`fix_implement_agent.py`)
Fixes issues in existing code.

**Focus:**
- Analyze error feedback
- Fix root causes
- Maintain functionality
- Improve code quality
- Add defensive programming

**Output:** `IntermediateImplementOutput`

### 4. Output Unifier (`output_unifier.py`)
Formats implementation results into structured output.

**Tasks:**
- Parse file descriptions
- Extract file contents
- Organize implementation summary
- Structure all outputs

**Output:** `CodeImplementOutput`

## Output Schema

### CodeImplementOutput

```python
{
    # Metadata
    "implementation_type": "initial" | "fix",
    "timestamp": "2025-11-05T...",
    
    # Generated code
    "generated_files": [
        {
            "file_path": "project/data/dataset.py",
            "content": "import torch\n...",
            "description": "Dataset loader",
            "dependencies": ["torch", "numpy"]
        },
        # ... more files
    ],
    
    # Summary
    "implementation_summary": {
        "files_created": 10,
        "files_modified": 0,
        "total_lines": 1500,
        "key_components": ["Dataset", "Model", "Trainer"],
        "implementation_notes": "Complete implementation..."
    },
    
    # Tests
    "test_files": [...],
    
    # Documentation
    "setup_instructions": "1. Install requirements...",
    "usage_examples": "python train.py --config...",
    
    # Quality
    "known_limitations": "...",
    "next_steps": "...",
    "issues_addressed": "..." # for fixes
}
```

## Usage

### Basic Usage

```python
import asyncio
from src.agents.experiment_agent.sub_agents.code_implement_agent import (
    create_code_implement_agent
)

async def main():
    # Create the agent system
    agent = create_code_implement_agent(
        model="gpt-4o",
        working_dir="/workspace/project",
        tools={
            "initial": [create_directory, write_file, read_file],
            "fix": [read_file, write_file, list_directory, read_logs]
        }
    )
    
    # Scenario 1: Initial implementation
    plan_data = """
    CodePlanOutput:
    file_structure: [...]
    dataset_plan: ...
    model_plan: ...
    """
    
    result = await agent.implement(plan_data)
    
    # Access results
    print(f"Type: {result.implementation_type}")
    print(f"Files: {len(result.generated_files)}")
    for file in result.generated_files:
        print(f"- {file.file_path}: {file.description}")

asyncio.run(main())
```

### Scenario-Specific Usage

#### Scenario 1: Initial Implementation
```python
input_data = """
CodePlanOutput:
  plan_type: initial
  file_structure:
    - path: project/
      type: directory
    - path: project/data/dataset.py
      type: file
  dataset_plan: ...
  model_plan: ...
  training_plan: ...
  implementation_roadmap: [...]
"""

result = await agent.implement(input_data)
```

#### Scenario 2: Fix Implementation
```python
input_data = """
CodePlanOutput:
  (original plan)

Error Feedback:
  issues:
    - file: project/models/model.py
      issue: Dimension mismatch in forward pass
      line: 45
    - file: project/training/trainer.py
      issue: Missing error handling
  
  suggested_fixes:
    - Add shape validation
    - Implement try-except blocks
"""

result = await agent.implement(input_data)
```

### Synchronous Usage

```python
from src.agents.experiment_agent.sub_agents.code_implement_agent import (
    create_code_implement_agent
)

# Create the agent system
agent = create_code_implement_agent(
    model="gpt-4o",
    working_dir="/workspace"
)

# Implement synchronously
result = agent.implement_sync(input_data)
```

### Accessing Generated Files

```python
# Get implementation result
result = await agent.implement(input_data)

# Write files to disk
for file in result.generated_files:
    file_path = f"/workspace/{file.file_path}"
    
    # Create directories if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file.content)
    
    print(f"Created: {file_path}")
```

## Tool Requirements

Tools should be implemented in `src/agents/experiment_agent/tools/`.

### For Initial Implementation:
- `create_directory(path: str) -> bool`: Create directory
- `write_file(path: str, content: str) -> bool`: Write file content
- `read_file(path: str) -> str`: Read file (for reference)
- `list_directory(path: str) -> List[str]`: List directory contents

### For Fix Implementation:
- `read_file(path: str) -> str`: Read existing code
- `write_file(path: str, content: str) -> bool`: Write updated code
- `list_directory(path: str) -> List[str]`: Check structure
- `read_logs(log_file: str) -> str`: Read execution logs
- `backup_file(path: str) -> str`: Backup before fixing

## Integration Example

### Complete Workflow

```python
from src.agents.experiment_agent.sub_agents.pre_analysis import (
    create_pre_analysis_agent
)
from src.agents.experiment_agent.sub_agents.code_plan_agent import (
    create_code_plan_agent
)
from src.agents.experiment_agent.sub_agents.code_implement_agent import (
    create_code_implement_agent
)

# Step 1: Analyze research
pre_analysis = create_pre_analysis_agent()
analysis = await pre_analysis.analyze(paper_content)

# Step 2: Generate plan
code_planner = create_code_plan_agent(working_dir="/workspace")
plan = await code_planner.plan(str(analysis.dict()))

# Step 3: Implement code
code_implementer = create_code_implement_agent(working_dir="/workspace")
implementation = await code_implementer.implement(str(plan.dict()))

# Step 4: Write files
for file in implementation.generated_files:
    write_to_disk(file.file_path, file.content)
```

### Fix Workflow

```python
# After code judge finds issues
judge_feedback = "..."  # from code_judge_agent

# Fix the code
fixed_implementation = await code_implementer.implement(
    f"{plan.dict()}\n\nError Feedback:\n{judge_feedback}"
)

# Update files
for file in fixed_implementation.generated_files:
    write_to_disk(file.file_path, file.content)
```

## Design Principles

### 1. Complete Implementation
- No placeholders or TODOs
- All functions fully implemented
- Production-ready code

### 2. Code Quality
- Follow best practices
- Add comprehensive documentation
- Include error handling
- Use type hints

### 3. Modularity
- Clear separation of concerns
- Reusable components
- Well-defined interfaces

### 4. Testability
- Include unit tests
- Add integration tests
- Provide test examples

## Implementation Guidelines

### File Organization
```
project/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocessing.py
├── models/
│   ├── __init__.py
│   └── model.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── loss.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── utils/
│   └── __init__.py
├── configs/
│   └── config.yaml
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
├── requirements.txt
├── README.md
├── train.py
└── test.py
```

### Code Quality Standards

1. **Type Hints:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
```

2. **Docstrings:**
```python
def train_epoch(self, dataloader: DataLoader) -> float:
    """
    Train for one epoch.
    
    Args:
        dataloader: Training data loader
        
    Returns:
        Average loss for the epoch
    """
```

3. **Error Handling:**
```python
try:
    data = load_data(path)
except FileNotFoundError:
    logger.error(f"Data file not found: {path}")
    raise
```

## Environment Setup

```bash
export PYTHONPATH="/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent:$PYTHONPATH"
```

## File Structure

```
code_implement_agent/
├── __init__.py                      # Module exports
├── output_schemas.py                # Pydantic models
├── code_implement_agent.py         # Main orchestrator
├── initial_implement_agent.py      # Initial implementation
├── fix_implement_agent.py          # Fix implementation
├── output_unifier.py               # Output formatting
└── README.md                       # Documentation
```

## Notes

- All agents use OpenAI Agents SDK with handoff mechanism
- Async/await is the primary execution model
- Complete, runnable code is the goal
- No placeholders or partial implementations
- Comprehensive testing and documentation included

