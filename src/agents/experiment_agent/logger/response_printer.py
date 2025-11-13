"""
Model response printer for agent outputs.

Provides utilities to display agent responses in a formatted and readable way.
"""

import textwrap
from typing import Any


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_model_response(
    response_obj: Any,
    agent_name: str = "Agent",
    max_field_length: int = 2000,
    show_full: bool = True,
):
    """
    Print agent model response in a formatted way.

    Args:
        response_obj: The response object from Runner.run()
        agent_name: Name of the agent for display
        max_field_length: Maximum length for each field (0 = no limit)
        show_full: If True, show all fields; if False, show summary only
    """
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'═' * 80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}🤖 {agent_name.upper()} RESPONSE{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'═' * 80}{Colors.ENDC}")

    try:
        if not hasattr(response_obj, "final_output"):
            print(f"\n{Colors.WARNING}No final_output in response{Colors.ENDC}")
            return

        output = response_obj.final_output

        # If it's a structured output (Pydantic model or similar)
        if hasattr(output, "__dict__"):
            fields = {k: v for k, v in output.__dict__.items() if not k.startswith("_")}

            if show_full:
                for key, value in fields.items():
                    _print_field(key, value, max_field_length)
            else:
                # Summary mode - just show field names and lengths
                print(f"\n{Colors.BOLD}Fields:{Colors.ENDC}")
                for key, value in fields.items():
                    value_str = str(value)
                    print(
                        f"  • {Colors.OKCYAN}{key}{Colors.ENDC}: {len(value_str)} chars"
                    )
        else:
            # Plain text output
            output_str = str(output)
            _print_field("output", output_str, max_field_length)

    except Exception as e:
        print(f"\n{Colors.FAIL}Error displaying response: {e}{Colors.ENDC}")

    print(f"{Colors.OKBLUE}{Colors.BOLD}{'═' * 80}{Colors.ENDC}\n")


def _print_field(key: str, value: Any, max_length: int):
    """
    Print a single field with proper formatting.

    Args:
        key: Field name
        value: Field value
        max_length: Maximum length to display (0 = no limit)
    """
    value_str = str(value)
    original_length = len(value_str)

    # Truncate if needed
    if max_length > 0 and len(value_str) > max_length:
        value_str = value_str[:max_length]
        truncated = True
    else:
        truncated = False

    # Print field header
    print(f"\n{Colors.BOLD}┌─ {Colors.OKCYAN}{key}{Colors.ENDC}")
    if truncated:
        print(
            f"{Colors.BOLD}│{Colors.ENDC}  {Colors.WARNING}(Showing first {max_length} of {original_length} chars){Colors.ENDC}"
        )

    # Word wrap the content
    wrapper = textwrap.TextWrapper(
        width=76, initial_indent="│  ", subsequent_indent="│  ", break_long_words=False
    )

    lines = value_str.split("\n")
    for line in lines:
        if line.strip():
            wrapped_lines = wrapper.wrap(line)
            for wrapped in wrapped_lines:
                print(f"{Colors.BOLD}{wrapped}{Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}│{Colors.ENDC}")

    if truncated:
        print(
            f"{Colors.BOLD}│{Colors.ENDC}  {Colors.WARNING}... (truncated){Colors.ENDC}"
        )

    print(f"{Colors.BOLD}└{'─' * 78}{Colors.ENDC}")


def print_response_summary(response_obj: Any, agent_name: str = "Agent"):
    """
    Print a brief summary of the response without full content.

    Args:
        response_obj: The response object from Runner.run()
        agent_name: Name of the agent for display
    """
    print_model_response(response_obj, agent_name, show_full=False)


class ResponsePrinter:
    """
    Context manager for automatically printing agent responses.

    Usage:
        with ResponsePrinter("My Agent", enabled=True):
            result = await Runner.run(agent, prompt)
        # Response is automatically printed
    """

    def __init__(
        self,
        agent_name: str = "Agent",
        enabled: bool = True,
        max_field_length: int = 2000,
        show_full: bool = True,
    ):
        """
        Initialize the response printer.

        Args:
            agent_name: Name of the agent
            enabled: Whether to print responses
            max_field_length: Maximum length for each field
            show_full: Whether to show full content or summary
        """
        self.agent_name = agent_name
        self.enabled = enabled
        self.max_field_length = max_field_length
        self.show_full = show_full
        self.response = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.response is not None:
            print_model_response(
                self.response,
                self.agent_name,
                self.max_field_length,
                self.show_full,
            )
        return False

    def set_response(self, response):
        """Set the response to be printed on exit."""
        self.response = response
        return response
