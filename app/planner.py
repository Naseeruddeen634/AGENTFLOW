"""
Task planner for AgentFlow.
Uses GPT-4 to decompose a high-level goal into an ordered list of subtasks,
each mapped to a specific tool.
"""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


PLANNING_PROMPT = """You are a task planner for an AI agent. Given a user's goal and 
a list of available tools, decompose the goal into a sequence of concrete subtasks.

Available tools:
{tools}

Rules:
1. Each subtask must use exactly one tool from the list above.
2. Subtasks should be ordered logically — later steps can depend on earlier results.
3. Keep the plan minimal. Use the fewest steps needed to achieve the goal.
4. Each subtask needs: a description, the tool to use, and the input parameters.
5. Maximum {max_steps} steps.

Respond ONLY with a JSON array. No explanation, no markdown, no backticks.

Format:
[
  {{"step": 1, "task": "description of what to do", "tool": "tool_name", "input": {{"param": "value"}}}},
  {{"step": 2, "task": "description", "tool": "tool_name", "input": {{"param": "value or {{prev_result}}"}}}},
]

Use {{prev_result}} as a placeholder when a step needs the output of the previous step.
Use {{result_N}} to reference the output of step N specifically.
"""


class Planner:
    """Decomposes a user goal into an executable task plan."""

    def __init__(self, llm: ChatOpenAI, tool_descriptions: str, max_steps: int = 10):
        self.llm = llm
        self.tool_descriptions = tool_descriptions
        self.max_steps = max_steps

    def plan(self, goal: str) -> list[dict]:
        """
        Generate an ordered list of subtasks for a given goal.
        Returns a list of dicts, each with: step, task, tool, input.
        """
        messages = [
            SystemMessage(content=PLANNING_PROMPT.format(
                tools=self.tool_descriptions,
                max_steps=self.max_steps,
            )),
            HumanMessage(content=f"Goal: {goal}"),
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Clean up common LLM formatting issues
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            tasks = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan: {e}\nRaw output: {raw}")
            # Fallback: single-step plan
            tasks = [{
                "step": 1,
                "task": goal,
                "tool": "web_search",
                "input": {"query": goal},
            }]

        # Validate and cap at max steps
        validated = []
        for task in tasks[:self.max_steps]:
            if all(k in task for k in ("step", "task", "tool", "input")):
                validated.append(task)

        if not validated:
            validated = [{
                "step": 1,
                "task": goal,
                "tool": "web_search",
                "input": {"query": goal},
            }]

        logger.info(f"Plan generated: {len(validated)} steps for goal: {goal[:80]}")
        return validated

    def replan(self, goal: str, completed_steps: list[dict], failed_step: dict, error: str) -> list[dict]:
        """
        Generate a revised plan after a step fails.
        Takes into account what has already been completed and what went wrong.
        """
        context = f"""The original goal was: {goal}

Steps completed so far:
{json.dumps(completed_steps, indent=2)}

This step failed:
{json.dumps(failed_step, indent=2)}

Error: {error}

Generate a revised plan to complete the remaining work. 
Do not repeat steps that already succeeded. 
Find an alternative approach for the failed step."""

        messages = [
            SystemMessage(content=PLANNING_PROMPT.format(
                tools=self.tool_descriptions,
                max_steps=self.max_steps,
            )),
            HumanMessage(content=context),
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            tasks = json.loads(raw)
            return [t for t in tasks[:self.max_steps] if all(k in t for k in ("step", "task", "tool", "input"))]
        except json.JSONDecodeError:
            logger.error("Replan failed to parse. Returning empty plan.")
            return []
