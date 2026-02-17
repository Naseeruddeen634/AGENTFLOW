"""
Tool registry for AgentFlow.
Handles registration, discovery, and schema generation for all available tools.
The agent reads tool descriptions at runtime to decide which tool to use.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """A single tool the agent can invoke."""

    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)

    def execute(self, **kwargs) -> str:
        """Run the tool and return the result as a string."""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            return f"ERROR: {str(e)}"

    def to_schema(self) -> dict:
        """Generate a schema the LLM can read to understand this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Central registry for all tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: dict | None = None,
    ):
        """Register a new tool."""
        tool = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters or {},
        )
        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [tool.to_schema() for tool in self._tools.values()]

    def get_tool_descriptions(self) -> str:
        """Format all tool descriptions for inclusion in the LLM prompt."""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(
                f"{k}: {v}" for k, v in tool.parameters.items()
            ) if tool.parameters else "none"
            lines.append(
                f"- {tool.name}: {tool.description} (parameters: {params})"
            )
        return "\n".join(lines)

    def execute(self, name: str, **kwargs) -> str:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return f"ERROR: Tool '{name}' not found. Available: {list(self._tools.keys())}"
        return tool.execute(**kwargs)
