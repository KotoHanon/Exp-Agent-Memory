"""
Custom hooks for OpenAI agents library.

Provides hooks to intercept and display:
- Agent execution flow
- LLM requests and responses
- Tool calls and results
"""

import json
from typing import Any, Dict

from agents import RunHooks


class Colors:
    """ANSI color codes for terminal output."""

    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def _truncate_to_tokens(text: str, max_tokens: int = 500, model: str = "gpt-4") -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenization

    Returns:
        Truncated text with token count info
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, ~{len(text)} chars total)"


class VerboseRunHooks(RunHooks):
    """
    Custom hooks to print detailed agent execution information.

    Intercepts:
    - Agent start/end
    - LLM requests/responses
    - Tool calls/results
    """

    def __init__(self, show_llm_responses: bool = True, show_tools: bool = True):
        """
        Initialize verbose hooks.

        Args:
            show_llm_responses: Whether to show LLM response content
            show_tools: Whether to show tool calls and results
        """
        super().__init__()
        self.show_llm_responses = show_llm_responses
        self.show_tools = show_tools

    async def on_agent_start(self, *args, **kwargs):
        """Called when an agent starts running."""
        # Extract agent name from context wrapper or args
        agent = kwargs.get("agent_name", args[0] if args else None)

        # Try to get actual agent name
        if hasattr(agent, "name"):
            agent_name = agent.name
        elif isinstance(agent, str):
            agent_name = agent
        else:
            agent_name = "Agent"

        print(f"\n{Colors.OKCYAN}🏃 START: {agent_name}{Colors.ENDC}")

    async def on_agent_end(self, *args, **kwargs):
        """Called when an agent finishes running."""
        # Extract agent name
        agent = kwargs.get("agent_name", args[0] if args else None)

        if hasattr(agent, "name"):
            agent_name = agent.name
        elif isinstance(agent, str):
            agent_name = agent
        else:
            agent_name = "Agent"

        print(f"{Colors.OKGREEN}✅ END: {agent_name}{Colors.ENDC}\n")

    async def on_llm_start(self, *args, **kwargs):
        """Called when LLM request starts."""
        if not self.show_llm_responses:
            return

        print(f"{Colors.WARNING}📤 LLM Request{Colors.ENDC}")

        # Debug: print what we received
        # print(f"DEBUG: args={len(args)}, kwargs keys={list(kwargs.keys())}")

        # Try to extract and display input messages (first 500 tokens)
        try:
            # Try to get messages from various sources
            messages = None
            if "messages" in kwargs:
                messages = kwargs["messages"]
            elif len(args) > 1:
                messages = args[1]
            elif len(args) > 0:
                # Sometimes the whole context is in args[0]
                ctx = args[0]
                if hasattr(ctx, "messages"):
                    messages = ctx.messages

            # print(f"DEBUG: messages type={type(messages)}, is None={messages is None}")

            if messages:
                # Try to extract text from messages
                input_text = ""

                # Handle different message formats
                if isinstance(messages, list):
                    # Extract content from message list
                    for msg in messages:
                        if isinstance(msg, dict):
                            if "content" in msg:
                                content = msg["content"]
                                if isinstance(content, str):
                                    input_text += content + "\n"
                                elif isinstance(content, list):
                                    # Handle structured content
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            input_text += item["text"] + "\n"
                        elif hasattr(msg, "content"):
                            input_text += str(msg.content) + "\n"

                elif hasattr(messages, "__iter__") and not isinstance(messages, str):
                    # Try to iterate if it's some other iterable (but not string)
                    try:
                        for msg in messages:
                            if hasattr(msg, "content"):
                                content = getattr(msg, "content", None)
                                if content:
                                    input_text += str(content) + "\n"
                            elif isinstance(msg, dict) and "content" in msg:
                                input_text += str(msg["content"]) + "\n"
                    except:
                        pass

                # Display truncated input
                if input_text.strip():
                    truncated_input = _truncate_to_tokens(
                        input_text.strip(), max_tokens=500
                    )
                    print(f"{Colors.OKCYAN}📥 Input (first 500 tokens):{Colors.ENDC}")
                    print(f"{Colors.OKBLUE}{truncated_input}{Colors.ENDC}\n")
                # else:
                #     print(f"  {Colors.WARNING}(Empty input){Colors.ENDC}\n")

        except Exception as e:
            # Show error for debugging
            print(
                f"  {Colors.WARNING}(Failed to extract input: {type(e).__name__}){Colors.ENDC}\n"
            )

    async def on_llm_end(self, *args, **kwargs):
        """Called when LLM response is received."""
        if not self.show_llm_responses:
            return

        response = kwargs.get("response", args[0] if args else None)
        print(f"{Colors.OKGREEN}✓ LLM Response{Colors.ENDC}")

        if response is None:
            return

        try:
            # Try to extract content from response
            content = None
            tool_calls = []

            # Check different response formats
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message"):
                    message = choice.message

                    # Check for text content
                    if hasattr(message, "content") and message.content:
                        content = message.content

                    # Check for tool calls
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = (
                                tool_call.function.name
                                if hasattr(tool_call, "function")
                                else "unknown"
                            )
                            tool_calls.append(tool_name)

            # Display content if available (first 500 tokens)
            if content:
                truncated_output = _truncate_to_tokens(content, max_tokens=500)
                print(f"{Colors.OKCYAN}📤 Output (first 500 tokens):{Colors.ENDC}")
                print(f"{Colors.OKGREEN}{truncated_output}{Colors.ENDC}\n")

            # Display tool calls if available
            if tool_calls:
                print(
                    f"{Colors.OKCYAN}🔧 Will call tools: {', '.join(tool_calls)}{Colors.ENDC}\n"
                )

        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}\n")

    async def on_tool_start(self, *args, **kwargs):
        """Called when a tool execution starts."""
        if not self.show_tools:
            return

        # Extract tool info from ToolContext
        tool_ctx = args[0] if args else None

        # Try to extract tool name
        if hasattr(tool_ctx, "tool_name"):
            tool_name = tool_ctx.tool_name
        else:
            tool_name = kwargs.get("tool_name", "Unknown")

        # Try to extract arguments
        arguments = None
        if hasattr(tool_ctx, "tool_arguments"):
            try:
                arguments = (
                    json.loads(tool_ctx.tool_arguments)
                    if isinstance(tool_ctx.tool_arguments, str)
                    else tool_ctx.tool_arguments
                )
            except:
                arguments = tool_ctx.tool_arguments

        print(f"{Colors.OKGREEN}🔧 {tool_name}{Colors.ENDC}", end="")

        # Show key arguments only
        if arguments and isinstance(arguments, dict):
            key_args = []
            for key in ["file_path", "directory", "tool_name", "pattern"]:
                if key in arguments and arguments[key]:
                    val = str(arguments[key])
                    if len(val) > 50:
                        val = val[:47] + "..."
                    key_args.append(f"{key}={val}")

            if key_args:
                print(f" ({', '.join(key_args)})", end="")

        print(f" {Colors.WARNING}...{Colors.ENDC}", end=" ")

    async def on_tool_end(self, *args, **kwargs):
        """Called when a tool execution ends."""
        if not self.show_tools:
            return

        # Extract result
        result = args[1] if len(args) > 1 else kwargs.get("result", None)

        print(f"{Colors.OKGREEN}✓{Colors.ENDC}")

        # Try to show useful result info
        if result is not None and isinstance(result, dict):
            # For dict results, show success status and key fields
            if result.get("success"):
                key_info = []
                if "total_count" in result:
                    key_info.append(f"{result['total_count']} items")
                if "file_path" in result:
                    key_info.append(f"→ {result['file_path']}")

                if key_info:
                    print(f"  {Colors.OKCYAN}{', '.join(key_info)}{Colors.ENDC}")

    async def on_handoff(self, *args, **kwargs):
        """Called when control is handed off between agents."""
        from_agent = kwargs.get("from_agent", args[0] if args else "Unknown")
        to_agent = kwargs.get("to_agent", args[1] if len(args) > 1 else "Unknown")
        print(f"\n{Colors.WARNING}🔄 {from_agent} → {to_agent}{Colors.ENDC}")


def create_verbose_hooks(
    show_llm_responses: bool = True,
    show_tools: bool = True,
) -> VerboseRunHooks:
    """
    Create verbose run hooks for agent execution.

    Args:
        show_llm_responses: Whether to show full LLM responses
        show_tools: Whether to show tool calls and results

    Returns:
        VerboseRunHooks instance
    """
    return VerboseRunHooks(
        show_llm_responses=show_llm_responses,
        show_tools=show_tools,
    )
