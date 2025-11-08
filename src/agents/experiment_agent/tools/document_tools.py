"""
Document analysis tools for experiment agents.

Provides tools for parsing and analyzing research documents (LaTeX, JSON, Markdown).
Compatible with openai-agents SDK.
"""

import json
import re
from typing import Dict, Any, Optional, List

from agents import function_tool


@function_tool
def parse_latex_sections(
    latex_content: str,
) -> Dict[str, Any]:
    """
    Parse LaTeX document and extract major sections.

    Args:
        latex_content: LaTeX document content

    Returns:
        Dictionary with extracted sections
    """
    try:
        sections = {}

        # Extract title
        title_match = re.search(r"\\title\{([^}]+)\}", latex_content)
        if title_match:
            sections["title"] = title_match.group(1)

        # Extract abstract
        abstract_match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex_content, re.DOTALL
        )
        if abstract_match:
            sections["abstract"] = abstract_match.group(1).strip()

        # Extract sections
        section_pattern = r"\\section\{([^}]+)\}(.*?)(?=\\section|\\end\{document\}|$)"
        section_matches = re.findall(section_pattern, latex_content, re.DOTALL)

        section_list = []
        for title, content in section_matches:
            section_list.append(
                {
                    "title": title,
                    "content": content.strip()[:1000],  # Limit content preview
                }
            )

        sections["sections"] = section_list
        sections["section_count"] = len(section_list)

        return {
            "success": True,
            "sections": sections,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error parsing LaTeX: {str(e)}",
        }


@function_tool
def extract_latex_equations(
    latex_content: str,
) -> Dict[str, Any]:
    """
    Extract mathematical equations from LaTeX document.

    Args:
        latex_content: LaTeX document content

    Returns:
        Dictionary with extracted equations
    """
    try:
        equations = []

        # Extract inline math: $...$
        inline_matches = re.findall(r"\$([^\$]+)\$", latex_content)
        equations.extend(
            [{"type": "inline", "content": eq} for eq in inline_matches[:50]]
        )

        # Extract display math: $$...$$
        display_matches = re.findall(r"\$\$([^\$]+)\$\$", latex_content)
        equations.extend([{"type": "display", "content": eq} for eq in display_matches])

        # Extract equation environment
        eq_pattern = r"\\begin\{equation\}(.*?)\\end\{equation\}"
        eq_matches = re.findall(eq_pattern, latex_content, re.DOTALL)
        equations.extend(
            [{"type": "equation", "content": eq.strip()} for eq in eq_matches]
        )

        # Extract align environment
        align_pattern = r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}"
        align_matches = re.findall(align_pattern, latex_content, re.DOTALL)
        equations.extend(
            [{"type": "align", "content": eq.strip()} for eq in align_matches]
        )

        return {
            "success": True,
            "equations": equations,
            "total_count": len(equations),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error extracting equations: {str(e)}",
        }


@function_tool
def parse_json_file(
    file_path: str,
) -> Dict[str, Any]:
    """
    Parse a JSON file and return its contents.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with parsed JSON content
    """
    try:
        import os

        file_path = os.path.expanduser(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "success": True,
            "data": data,
            "type": type(data).__name__,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading JSON: {str(e)}",
        }


@function_tool
def extract_code_blocks(
    markdown_content: str,
) -> Dict[str, Any]:
    """
    Extract code blocks from Markdown content.

    Args:
        markdown_content: Markdown document content

    Returns:
        Dictionary with extracted code blocks
    """
    try:
        # Pattern for fenced code blocks with language
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, markdown_content, re.DOTALL)

        code_blocks = []
        for lang, code in matches:
            code_blocks.append(
                {
                    "language": lang or "plaintext",
                    "code": code.strip(),
                }
            )

        return {
            "success": True,
            "code_blocks": code_blocks,
            "total_count": len(code_blocks),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error extracting code blocks: {str(e)}",
        }


@function_tool
def summarize_document(
    content: str,
    max_length: int = 500,
) -> Dict[str, Any]:
    """
    Create a simple summary of document content.

    Args:
        content: Document content
        max_length: Maximum length of summary

    Returns:
        Dictionary with summary information
    """
    try:
        # Simple extractive summary: first paragraph and some statistics
        lines = content.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        # Get first paragraph as preview
        preview = ""
        for line in non_empty_lines:
            if len(preview) + len(line) < max_length:
                preview += line + " "
            else:
                break

        # Statistics
        word_count = len(content.split())
        line_count = len(lines)
        char_count = len(content)

        return {
            "success": True,
            "preview": preview.strip(),
            "statistics": {
                "word_count": word_count,
                "line_count": line_count,
                "character_count": char_count,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error summarizing document: {str(e)}",
        }


@function_tool
def extract_urls(
    content: str,
) -> Dict[str, Any]:
    """
    Extract URLs from text content.

    Args:
        content: Text content

    Returns:
        Dictionary with extracted URLs
    """
    try:
        # Simple URL regex
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))

        return {
            "success": True,
            "urls": unique_urls,
            "total_count": len(unique_urls),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error extracting URLs: {str(e)}",
        }


@function_tool
def parse_requirements_txt(
    file_path: str,
) -> Dict[str, Any]:
    """
    Parse a requirements.txt file and extract package specifications.

    Args:
        file_path: Path to requirements.txt file

    Returns:
        Dictionary with parsed requirements
    """
    try:
        import os

        file_path = os.path.expanduser(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        requirements = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse package specification
            # Simple parsing: package[==|>=|<=|>|<]version
            match = re.match(r"([a-zA-Z0-9_-]+)([<>=!]+.*)?", line)
            if match:
                package = match.group(1)
                version = match.group(2) if match.group(2) else ""
                requirements.append(
                    {
                        "package": package,
                        "version_spec": version,
                        "raw": line,
                    }
                )

        return {
            "success": True,
            "requirements": requirements,
            "total_count": len(requirements),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error parsing requirements: {str(e)}",
        }


@function_tool
def extract_key_terms(
    content: str,
    min_length: int = 3,
) -> Dict[str, Any]:
    """
    Extract potential key terms from content (simple word frequency).

    Args:
        content: Text content
        min_length: Minimum word length to consider

    Returns:
        Dictionary with key terms
    """
    try:
        # Simple word extraction and frequency
        words = re.findall(r"\b[a-zA-Z]{" + str(min_length) + r",}\b", content.lower())

        # Common stopwords to filter
        stopwords = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "who",
            "boy",
            "did",
            "man",
            "she",
            "too",
            "use",
            "way",
            "with",
            "this",
            "that",
            "from",
        }

        words = [w for w in words if w not in stopwords]

        # Count frequencies
        from collections import Counter

        word_freq = Counter(words)

        # Get top terms
        top_terms = word_freq.most_common(20)

        return {
            "success": True,
            "top_terms": [
                {"term": term, "frequency": freq} for term, freq in top_terms
            ],
            "total_unique": len(word_freq),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error extracting key terms: {str(e)}",
        }
