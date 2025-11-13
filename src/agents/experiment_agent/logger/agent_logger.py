"""
Custom logging configuration for OpenAI agents library.

This module provides:
- AgentLogFilter: Filters agent logs to show only relevant information
- ColoredFormatter: Formats logs with colors and icons
- setup_agent_logging: Configures the agents logger with custom settings
"""

import logging
import re
import sys


class AgentLogFilter(logging.Filter):
    """Custom filter to show only tool calls and model responses."""

    def filter(self, record):
        """
        Filter log records to only show:
        - Tool invocations
        - Model responses
        - HTTP requests
        - Agent running status
        
        Returns:
            bool: True if the record should be logged, False otherwise
        """
        msg = record.getMessage()

        # Keep these types of messages
        keep_patterns = [
            "Invoking tool",
            "Tool .* completed",
            "Received model response",
            "HTTP Request:",
            "Running agent",
        ]

        # Filter out these
        ignore_patterns = [
            "Tracing is disabled",
            "Setting current trace",
            "Not creating span",
            "Not creating trace",
            "Calling LLM",
            "Resetting current trace",
        ]

        # Check if we should ignore
        for pattern in ignore_patterns:
            if pattern in msg:
                return False

        # Check if we should keep
        for pattern in keep_patterns:
            if re.search(pattern, msg):
                return True

        # By default, don't show
        return False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different message types."""

    COLORS = {
        "tool": "\033[92m",  # Green
        "response": "\033[94m",  # Blue
        "http": "\033[93m",  # Yellow
        "agent": "\033[96m",  # Cyan
        "reset": "\033[0m",
    }

    def format(self, record):
        """
        Format log record with colors and icons based on message type.
        
        Args:
            record: LogRecord to format
            
        Returns:
            str: Formatted log message
        """
        msg = record.getMessage()

        # Color code based on message type
        if "Invoking tool" in msg:
            color = self.COLORS["tool"]
            prefix = "🔧 TOOL CALL"
        elif "completed" in msg:
            color = self.COLORS["tool"]
            prefix = "✓ TOOL DONE"
        elif "Received model response" in msg:
            color = self.COLORS["response"]
            prefix = "🤖 MODEL"
        elif "HTTP Request" in msg:
            color = self.COLORS["http"]
            prefix = "📡 API"
        elif "Running agent" in msg:
            color = self.COLORS["agent"]
            prefix = "🏃 AGENT"
        else:
            color = ""
            prefix = ""

        reset = self.COLORS["reset"]

        if prefix:
            return f"{color}[{prefix}]{reset} {msg}"
        else:
            return msg


def setup_agent_logging(verbose: bool = False):
    """
    Configure agents library logging.

    Args:
        verbose: If True, enable DEBUG level logging with custom filtering
                 showing only tool calls and model responses.
                 If False, disable all agent logging.
    """
    # Get the agents logger
    agents_logger = logging.getLogger("openai.agents")

    # Remove existing handlers to avoid duplicates
    agents_logger.handlers.clear()

    if verbose:
        # Set to DEBUG level to capture all messages
        agents_logger.setLevel(logging.DEBUG)

        # Create colored formatter
        formatter = ColoredFormatter()

        # Add console handler with custom filter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(AgentLogFilter())
        agents_logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate logs
        agents_logger.propagate = False

        print(f"\033[94mℹ\033[0m Agent custom logging enabled (tools + responses only)")
    else:
        # Completely disable agents logging in non-verbose mode
        agents_logger.setLevel(logging.CRITICAL)
        agents_logger.propagate = False

        print(f"\033[94mℹ\033[0m Agent logging disabled (use --verbose to enable)")

