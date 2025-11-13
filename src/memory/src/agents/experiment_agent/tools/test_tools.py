"""
Test script for experiment agent tools.

Verifies that all tools can be imported and executed correctly.
"""

import os
import tempfile
from pathlib import Path


def test_imports():
    """Test that all tools can be imported."""
    print("Testing tool imports...")

    try:
        from src.agents.experiment_agent.tools import (
            # File tools
            read_file,
            write_file,
            list_directory,
            create_directory,
            delete_file,
            copy_file,
            file_exists,
            get_file_info,
            # Execution tools
            run_python_script,
            run_shell_command,
            run_python_code,
            install_package,
            check_python_syntax,
            get_environment_info,
            list_installed_packages,
            create_log_file,
            append_to_log,
            # Document tools
            parse_latex_sections,
            extract_latex_equations,
            parse_json_file,
            extract_code_blocks,
            summarize_document,
            extract_urls,
            parse_requirements_txt,
            extract_key_terms,
            # Code analysis tools
            analyze_python_file,
            search_in_codebase,
            count_lines_of_code,
            extract_function_code,
            list_python_files,
            check_imports_available,
            get_file_dependencies,
        )

        print("✓ All tool imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_file_tools():
    """Test file operation tools."""
    print("\nTesting file tools...")

    from src.agents.experiment_agent.tools import (
        write_file,
        read_file,
        file_exists,
        delete_file,
    )

    # Create temp file
    temp_file = tempfile.mktemp(suffix=".txt")

    try:
        # Test write
        result = write_file(temp_file, "Hello, World!")
        assert result["success"], f"Write failed: {result.get('error')}"
        print("✓ write_file works")

        # Test exists
        result = file_exists(temp_file)
        assert result["exists"], "File should exist"
        print("✓ file_exists works")

        # Test read
        result = read_file(temp_file)
        assert result["success"], f"Read failed: {result.get('error')}"
        assert result["content"] == "Hello, World!", "Content mismatch"
        print("✓ read_file works")

        # Test delete
        result = delete_file(temp_file)
        assert result["success"], f"Delete failed: {result.get('error')}"
        print("✓ delete_file works")

        return True
    except AssertionError as e:
        print(f"✗ File tools test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_execution_tools():
    """
    Test code execution tools.

    ⚠️ NOTE: These tests require a running Docker container with TCP server.
    Set environment variables:
        - DOCKER_HOST (default: localhost)
        - DOCKER_PORT (default: 8000)
    """
    print("\nTesting execution tools (Docker required)...")

    from src.agents.experiment_agent.tools import (
        test_docker_connection,
        run_python_code,
        get_environment_info,
    )

    try:
        # First test Docker connection
        print("  Checking Docker connection...")
        result = test_docker_connection()
        if not result["success"]:
            print(f"  ⚠ Docker not available: {result['message']}")
            print("  Skipping execution tool tests (Docker required)")
            return True  # Don't fail if Docker is not available
        print(f"  ✓ Docker connection OK ({result['host']}:{result['port']})")

        # Test run python code
        result = run_python_code("print('test')")
        assert result["success"], f"Code execution failed: {result.get('error')}"
        assert "test" in result["stdout"], "Output mismatch"
        print("  ✓ run_python_code works")

        # Test environment info
        result = get_environment_info()
        assert result["success"], f"Env info failed: {result.get('error')}"
        assert "python_version" in result, "Missing python_version"
        assert (
            result["execution_mode"] == "Docker Container"
        ), "Should report Docker mode"
        print("  ✓ get_environment_info works")

        print("✓ Execution tools work correctly")
        return True
    except AssertionError as e:
        print(f"✗ Execution tools test failed: {e}")
        return False


def test_document_tools():
    """Test document analysis tools."""
    print("\nTesting document tools...")

    from src.agents.experiment_agent.tools import (
        parse_latex_sections,
        extract_code_blocks,
        summarize_document,
    )

    try:
        # Test LaTeX parsing
        latex = "\\title{Test}\\begin{abstract}This is a test.\\end{abstract}"
        result = parse_latex_sections(latex)
        assert result["success"], f"LaTeX parsing failed: {result.get('error')}"
        assert "title" in result["sections"], "Missing title"
        print("✓ parse_latex_sections works")

        # Test code block extraction
        markdown = "```python\nprint('hello')\n```"
        result = extract_code_blocks(markdown)
        assert result["success"], f"Code extraction failed: {result.get('error')}"
        assert len(result["code_blocks"]) > 0, "No code blocks found"
        print("✓ extract_code_blocks works")

        # Test document summary
        content = "This is a test document with multiple sentences. " * 10
        result = summarize_document(content)
        assert result["success"], f"Summarization failed: {result.get('error')}"
        assert "statistics" in result, "Missing statistics"
        print("✓ summarize_document works")

        return True
    except AssertionError as e:
        print(f"✗ Document tools test failed: {e}")
        return False


def test_code_analysis_tools():
    """Test code analysis tools."""
    print("\nTesting code analysis tools...")

    from src.agents.experiment_agent.tools import analyze_python_file, list_python_files

    try:
        # Create a test Python file
        temp_file = tempfile.mktemp(suffix=".py")
        with open(temp_file, "w") as f:
            f.write(
                """
import os

def test_function(x):
    '''Test function'''
    return x * 2

class TestClass:
    def method(self):
        pass
"""
            )

        # Test analyze
        result = analyze_python_file(temp_file)
        assert result["success"], f"Analysis failed: {result.get('error')}"
        assert result["class_count"] == 1, "Wrong class count"
        print("✓ analyze_python_file works")

        # Test list Python files
        temp_dir = os.path.dirname(temp_file)
        result = list_python_files(temp_dir)
        assert result["success"], f"List files failed: {result.get('error')}"
        print("✓ list_python_files works")

        os.remove(temp_file)
        return True
    except AssertionError as e:
        print(f"✗ Code analysis tools test failed: {e}")
        return False


def test_tool_collections():
    """Test tool collection exports."""
    print("\nTesting tool collections...")

    from src.agents.experiment_agent.tools import (
        FILE_TOOLS,
        EXECUTION_TOOLS,
        DOCUMENT_TOOLS,
        CODE_ANALYSIS_TOOLS,
        ALL_TOOLS,
        get_tools_for_agent,
    )

    try:
        assert len(FILE_TOOLS) == 8, f"Wrong FILE_TOOLS count: {len(FILE_TOOLS)}"
        print(f"✓ FILE_TOOLS: {len(FILE_TOOLS)} tools")

        assert (
            len(EXECUTION_TOOLS) == 9
        ), f"Wrong EXECUTION_TOOLS count: {len(EXECUTION_TOOLS)}"
        print(f"✓ EXECUTION_TOOLS: {len(EXECUTION_TOOLS)} tools")

        assert (
            len(DOCUMENT_TOOLS) == 8
        ), f"Wrong DOCUMENT_TOOLS count: {len(DOCUMENT_TOOLS)}"
        print(f"✓ DOCUMENT_TOOLS: {len(DOCUMENT_TOOLS)} tools")

        assert (
            len(CODE_ANALYSIS_TOOLS) == 7
        ), f"Wrong CODE_ANALYSIS_TOOLS count: {len(CODE_ANALYSIS_TOOLS)}"
        print(f"✓ CODE_ANALYSIS_TOOLS: {len(CODE_ANALYSIS_TOOLS)} tools")

        assert len(ALL_TOOLS) == 32, f"Wrong ALL_TOOLS count: {len(ALL_TOOLS)}"
        print(f"✓ ALL_TOOLS: {len(ALL_TOOLS)} tools")

        # Test get_tools_for_agent
        tools = get_tools_for_agent("code_judge")
        assert len(tools) > 0, "Should return tools"
        print(f"✓ get_tools_for_agent works")

        return True
    except AssertionError as e:
        print(f"✗ Tool collections test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Experiment Agent Tools Test Suite")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("File Tools", test_file_tools()))
    results.append(("Execution Tools", test_execution_tools()))
    results.append(("Document Tools", test_document_tools()))
    results.append(("Code Analysis Tools", test_code_analysis_tools()))
    results.append(("Tool Collections", test_tool_collections()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25} {status}")

    all_passed = all(result[1] for result in results)

    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
