# Input Processing Module

This module provides standardized input schemas and converters for the Experiment Agent System.

## Overview

The input processing module standardizes inputs from different sources (research papers in LaTeX format and research ideas in JSON format) into a unified `ResearchInput` format. This ensures consistent processing throughout the experiment workflow.

## Key Components

### 1. Schemas (`schemas.py`)

Defines the standardized input data structures:

- **`ResearchInput`**: Main input schema with `input_type`, `content`, `title`, and `metadata` fields
- **`IdeaMetadata`**: Metadata specific to research ideas
- **`PaperMetadata`**: Metadata specific to research papers

### 2. Converters (`converters.py`)

Provides conversion utilities:

- **`convert_tex_to_research_input()`**: Convert LaTeX (.tex) files to ResearchInput
- **`convert_idea_json_to_research_input()`**: Convert idea JSON files to ResearchInput
- **`load_research_input()`**: Auto-detect file type and convert
- **`save_research_input_as_json()`**: Export ResearchInput to JSON
- **`save_research_input_as_text()`**: Export ResearchInput to text format

### 3. Conversion Script (`convert_script.py`)

Command-line tool for standalone conversion:

```bash
# Convert a LaTeX paper to standardized JSON
python convert_script.py paper.tex --output paper_standardized.json

# Convert an idea JSON to text format
python convert_script.py idea.json --output idea_standardized.txt --format text

# Auto-detect and convert with verbose output
python convert_script.py input_file.tex -v
```

## Usage in Main Workflow

The main experiment workflow (`main.py`) automatically uses this module:

```python
from src.agents.experiment_agent.input_processing import (
    load_research_input,
    ResearchInput,
)

# Load and convert input
research_input_obj = load_research_input(input_path, encoding="utf-8")

# Get standardized text format for agent processing
research_input_text = research_input_obj.to_text()

# The text format includes:
# INPUT_TYPE: PAPER or IDEA
# TITLE: ...
# METADATA: ...
# CONTENT: ...
```

## Input Type Detection

The `PreAnalysisAgent` automatically detects input type using:

1. **Primary**: Standardized format markers (`INPUT_TYPE: PAPER` or `INPUT_TYPE: IDEA`)
2. **Fallback**: Heuristic detection (LaTeX markers, JSON structure)

## Input Formats

### LaTeX Paper (.tex)

Expected structure:
```latex
\documentclass{article}
\title{Paper Title}
\author{Author Names}

\begin{abstract}
Abstract content...
\end{abstract}

\begin{document}
...
\end{document}
```

Extracted metadata:
- Title from `\title{...}`
- Authors from `\author{...}`
- Abstract from `\begin{abstract}...\end{abstract}`

### Idea JSON

Expected structure:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Idea description..."
    },
    {
      "role": "assistant",
      "content": "Detailed proposal..."
    }
  ],
  "context_variables": {
    "working_dir": "...",
    "date_limit": "...",
    "prepare_result": {
      "reference_codebases": [...],
      "reference_papers": [...]
    },
    "idea_evaluation": [...]
  }
}
```

Extracted metadata:
- Working directory, date limit
- Reference codebases and papers
- Idea evaluation scores (innovation, effectiveness, feasibility)

## Example: Standalone Conversion

```python
from src.agents.experiment_agent.input_processing import (
    load_research_input,
    save_research_input_as_json,
)

# Load input file (auto-detects type)
research_input = load_research_input("path/to/input.tex")

# Access standardized data
print(f"Type: {research_input.input_type}")
print(f"Title: {research_input.title}")
print(f"Content length: {len(research_input.content)}")

# Export to different formats
save_research_input_as_json(research_input, "output.json")

# Or get text format for processing
text_format = research_input.to_text()
```

## Benefits

1. **Consistency**: Uniform input format across different source types
2. **Metadata Extraction**: Automatic extraction of titles, authors, references
3. **Type Safety**: Pydantic models ensure data validation
4. **Flexibility**: Support for both LaTeX papers and JSON ideas
5. **Integration**: Seamlessly integrated into the main workflow
6. **Extensibility**: Easy to add support for new input formats

## File Organization

```
input_processing/
├── __init__.py         # Package exports
├── schemas.py          # Pydantic models for standardized inputs
├── converters.py       # Conversion utilities
├── convert_script.py   # Command-line conversion tool
└── README.md           # This file
```

## Error Handling

The converters handle common errors:
- Missing files
- Invalid file formats
- Encoding issues (default: UTF-8)
- Malformed LaTeX or JSON

All functions use UTF-8 encoding by default, which can be overridden if needed.

