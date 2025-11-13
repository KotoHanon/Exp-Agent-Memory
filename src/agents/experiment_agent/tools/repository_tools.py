"""
Repository analysis tools for experiment agents.

Provides tools for analyzing code repositories, generating code trees,
and extracting important code structures.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents import function_tool

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@function_tool
def list_papers_in_directory(
    directory: str,
    file_extensions: List[str] = None,
) -> Dict[str, Any]:
    """
    List all paper files in a directory.

    Args:
        directory: Directory path to search for papers
        file_extensions: List of file extensions to include (default: [".tex", ".pdf", ".txt"])

    Returns:
        Dictionary with list of papers and their metadata
    """
    try:
        directory = os.path.expanduser(directory)

        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        if file_extensions is None:
            file_extensions = [".tex", ".pdf", ".txt", ".md"]

        papers = []
        path = Path(directory)

        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix in file_extensions:
                papers.append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "extension": file_path.suffix,
                        "size": file_path.stat().st_size,
                        "size_kb": round(file_path.stat().st_size / 1024, 2),
                    }
                )

        # Sort by name
        papers.sort(key=lambda x: x["name"])

        return {
            "success": True,
            "directory": directory,
            "papers": papers,
            "total_count": len(papers),
            "extensions_found": list(set(p["extension"] for p in papers)),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing papers: {str(e)}",
        }


@function_tool
def generate_code_tree(
    directory: str,
    max_depth: int = 4,
    include_files: bool = True,
    ignore_patterns: List[str] = None,
) -> Dict[str, Any]:
    """
    Generate a tree structure of a code repository.

    Args:
        directory: Root directory of the repository
        max_depth: Maximum depth to traverse (default: 4)
        include_files: Whether to include files in the tree (default: True)
        ignore_patterns: List of patterns to ignore (default: common patterns)

    Returns:
        Dictionary with tree structure and metadata
    """
    try:
        directory = os.path.expanduser(directory)

        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        if ignore_patterns is None:
            ignore_patterns = [
                "__pycache__",
                ".git",
                ".venv",
                "venv",
                "env",
                "node_modules",
                ".pytest_cache",
                ".mypy_cache",
                "*.pyc",
                "*.pyo",
                "*.egg-info",
                ".DS_Store",
            ]

        def should_ignore(path: Path) -> bool:
            """Check if path should be ignored."""
            name = path.name
            for pattern in ignore_patterns:
                if pattern.startswith("*"):
                    if name.endswith(pattern[1:]):
                        return True
                elif pattern in str(path):
                    return True
            return False

        def build_tree(path: Path, current_depth: int = 0) -> Dict:
            """Recursively build tree structure."""
            if current_depth > max_depth:
                return None

            if should_ignore(path):
                return None

            node = {
                "name": path.name,
                "type": "directory" if path.is_dir() else "file",
                "path": str(path),
            }

            if path.is_file():
                node["size"] = path.stat().st_size
                node["extension"] = path.suffix
                return node

            # Directory
            children = []
            try:
                for child in sorted(path.iterdir()):
                    child_node = build_tree(child, current_depth + 1)
                    if child_node is not None:
                        if include_files or child_node["type"] == "directory":
                            children.append(child_node)
            except PermissionError:
                node["error"] = "Permission denied"

            if children:
                node["children"] = children
                node["child_count"] = len(children)

            return node

        root_path = Path(directory)
        tree = build_tree(root_path)

        # Generate text representation
        def tree_to_text(
            node: Dict, prefix: str = "", is_last: bool = True
        ) -> List[str]:
            """Convert tree to text representation."""
            lines = []

            if node is None:
                return lines

            # Current node
            connector = "└── " if is_last else "├── "
            if prefix == "":
                lines.append(node["name"] + "/")
            else:
                type_indicator = "/" if node["type"] == "directory" else ""
                lines.append(f"{prefix}{connector}{node['name']}{type_indicator}")

            # Children
            if "children" in node:
                children = node["children"]
                for i, child in enumerate(children):
                    is_last_child = i == len(children) - 1
                    extension = "    " if is_last else "│   "
                    child_lines = tree_to_text(
                        child, prefix + extension if prefix != "" else "", is_last_child
                    )
                    lines.extend(child_lines)

            return lines

        tree_text = "\n".join(tree_to_text(tree))

        # Count statistics
        def count_nodes(node: Dict) -> Dict[str, int]:
            """Count files and directories."""
            if node is None:
                return {"files": 0, "directories": 0}

            counts = {"files": 0, "directories": 0}

            if node["type"] == "file":
                counts["files"] = 1
            else:
                counts["directories"] = 1

            if "children" in node:
                for child in node["children"]:
                    child_counts = count_nodes(child)
                    counts["files"] += child_counts["files"]
                    counts["directories"] += child_counts["directories"]

            return counts

        statistics = count_nodes(tree)

        return {
            "success": True,
            "directory": directory,
            "tree": tree,
            "tree_text": tree_text,
            "statistics": statistics,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating code tree: {str(e)}",
        }


@function_tool
def analyze_repository_structure(
    directory: str,
    identify_main_files: bool = True,
) -> Dict[str, Any]:
    """
    Analyze the structure of a code repository and identify key files.

    Args:
        directory: Root directory of the repository
        identify_main_files: Whether to identify main/important files (default: True)

    Returns:
        Dictionary with repository structure analysis
    """
    try:
        directory = os.path.expanduser(directory)

        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        path = Path(directory)

        # Collect all files
        python_files = []
        config_files = []
        doc_files = []
        other_files = []

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip common ignore patterns
            if any(
                ignore in str(file_path)
                for ignore in ["__pycache__", ".git", "node_modules", ".venv", "venv"]
            ):
                continue

            relative_path = file_path.relative_to(path)
            file_info = {
                "name": file_path.name,
                "path": str(file_path),
                "relative_path": str(relative_path),
                "size": file_path.stat().st_size,
            }

            if file_path.suffix == ".py":
                python_files.append(file_info)
            elif file_path.name in [
                "requirements.txt",
                "setup.py",
                "pyproject.toml",
                "setup.cfg",
                "environment.yml",
                "Dockerfile",
            ]:
                config_files.append(file_info)
            elif file_path.suffix in [".md", ".rst", ".txt"] or file_path.name in [
                "README",
                "LICENSE",
            ]:
                doc_files.append(file_info)
            else:
                other_files.append(file_info)

        # Identify important files
        important_files = []

        if identify_main_files:
            # Main entry points
            main_patterns = [
                "main.py",
                "run.py",
                "app.py",
                "__main__.py",
                "train.py",
                "test.py",
            ]
            for pattern in main_patterns:
                for py_file in python_files:
                    if py_file["name"] == pattern:
                        important_files.append(
                            {
                                **py_file,
                                "importance": "entry_point",
                                "reason": f"Main entry point: {pattern}",
                            }
                        )

            # README and docs
            for doc_file in doc_files:
                if "README" in doc_file["name"].upper():
                    important_files.append(
                        {
                            **doc_file,
                            "importance": "documentation",
                            "reason": "Main documentation file",
                        }
                    )
                    break

            # Config files
            for config_file in config_files:
                if config_file["name"] in [
                    "requirements.txt",
                    "setup.py",
                    "pyproject.toml",
                ]:
                    important_files.append(
                        {
                            **config_file,
                            "importance": "configuration",
                            "reason": "Dependency configuration",
                        }
                    )

            # Largest Python files (likely contain main logic)
            sorted_py_files = sorted(
                python_files, key=lambda x: x["size"], reverse=True
            )
            for py_file in sorted_py_files[:5]:
                if py_file["size"] > 1024:  # > 1KB
                    # Check if not already in important files
                    if not any(f["path"] == py_file["path"] for f in important_files):
                        important_files.append(
                            {
                                **py_file,
                                "importance": "core_logic",
                                "reason": f"Large file ({py_file['size']} bytes), likely contains core logic",
                            }
                        )

        # Directory structure summary
        directories = set()
        for py_file in python_files:
            dir_path = Path(py_file["relative_path"]).parent
            if str(dir_path) != ".":
                directories.add(str(dir_path))

        return {
            "success": True,
            "directory": directory,
            "summary": {
                "total_python_files": len(python_files),
                "total_config_files": len(config_files),
                "total_doc_files": len(doc_files),
                "total_other_files": len(other_files),
                "total_directories": len(directories),
            },
            "python_files": python_files[:50],  # Limit to first 50
            "config_files": config_files,
            "doc_files": doc_files,
            "important_files": important_files,
            "directories": sorted(list(directories)),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing repository: {str(e)}",
        }


@function_tool
def get_repository_overview(
    directory: str,
) -> Dict[str, Any]:
    """
    Get a comprehensive overview of a code repository including tree structure and analysis.

    Args:
        directory: Root directory of the repository

    Returns:
        Dictionary with complete repository overview
    """
    try:
        # Generate tree structure
        tree_result = generate_code_tree(directory, max_depth=3, include_files=True)

        if not tree_result["success"]:
            return tree_result

        # Analyze structure
        analysis_result = analyze_repository_structure(
            directory, identify_main_files=True
        )

        if not analysis_result["success"]:
            return analysis_result

        return {
            "success": True,
            "directory": directory,
            "tree_structure": tree_result["tree_text"],
            "statistics": tree_result["statistics"],
            "analysis": analysis_result["summary"],
            "important_files": analysis_result["important_files"],
            "python_files": analysis_result["python_files"][:20],  # Top 20
            "config_files": analysis_result["config_files"],
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting repository overview: {str(e)}",
        }


def _extract_pdf_with_pypdf2(pdf_path: str, max_chars: int = 50000) -> str:
    """
    Extract text from PDF using PyPDF2 (fallback method).

    Args:
        pdf_path: Path to PDF file
        max_chars: Maximum characters to extract

    Returns:
        Extracted text
    """
    if not PYPDF2_AVAILABLE:
        return ""

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if len(text) >= max_chars:
                    break
                text += page.extract_text() + "\n"
            return text[:max_chars]
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
        return ""


def _extract_pdf_with_docling(pdf_path: str, timeout: int = 60) -> str:
    """
    Extract text from PDF using Docling (preferred method).

    Args:
        pdf_path: Path to PDF file
        timeout: Timeout in seconds

    Returns:
        Extracted text in markdown format
    """
    if not DOCLING_AVAILABLE:
        return ""

    try:
        # Check if corresponding .md file exists
        md_path = pdf_path.rsplit(".", 1)[0] + ".md"
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    print(f"Using cached markdown: {os.path.basename(pdf_path)}")
                    return content

        # Convert using Docling
        print(f"Converting PDF with Docling: {os.path.basename(pdf_path)}")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()

        # Save to cache
        if text and text.strip():
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Successfully extracted: {os.path.basename(pdf_path)}")
            return text

        return ""

    except Exception as e:
        print(f"Docling extraction failed: {e}")
        return ""


def _truncate_text(text: str, max_tokens: int = 15000, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenization

    Returns:
        Truncated text
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: roughly 4 characters per token
        max_chars = max_tokens * 4
        return text[:max_chars]

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text


@function_tool
def read_pdf_paper(
    pdf_path: str,
    max_tokens: int = 0,  # 0 means no truncation
    use_docling: bool = True,
) -> Dict[str, Any]:
    """
    Read and extract text from a PDF paper file.

    Args:
        pdf_path: Path to the PDF file
        max_tokens: Maximum tokens to extract (default: 15000)
        use_docling: Whether to use Docling for extraction (default: True)

    Returns:
        Dictionary with extracted text and metadata
    """
    try:
        pdf_path = os.path.expanduser(pdf_path)

        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
            }

        # Try Docling first if available
        text = ""
        if use_docling and DOCLING_AVAILABLE:
            text = _extract_pdf_with_docling(pdf_path)

        # Fallback to PyPDF2
        if not text and PYPDF2_AVAILABLE:
            print(f"Falling back to PyPDF2 for: {os.path.basename(pdf_path)}")
            text = _extract_pdf_with_pypdf2(pdf_path)

        if not text:
            return {
                "success": False,
                "error": "Failed to extract text from PDF. PyPDF2 or Docling required.",
            }

        # Remove content before introduction
        intro_patterns = [
            r"(?i)^1\.?\s*introduction",
            r"(?i)^I\.?\s*introduction",
            r"(?i)^introduction",
        ]
        lines = text.split("\n")
        intro_idx = 0
        for i, line in enumerate(lines):
            if any(re.match(pattern, line.strip()) for pattern in intro_patterns):
                intro_idx = i
                break

        if intro_idx > 0:
            text = "\n".join(lines[intro_idx:])

        # Truncate to token limit (if max_tokens > 0)
        if max_tokens > 0:
            text = _truncate_text(text, max_tokens)

        # Get file stats
        file_stats = os.stat(pdf_path)

        return {
            "success": True,
            "pdf_path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "text": text,
            "text_length": len(text),
            "estimated_tokens": len(text) // 4,  # Rough estimate
            "file_size": file_stats.st_size,
            "extraction_method": (
                "docling" if use_docling and DOCLING_AVAILABLE else "pypdf2"
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading PDF: {str(e)}",
        }
