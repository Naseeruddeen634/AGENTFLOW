"""
Core agent for AgentFlow.
Implements the ReAct (Reasoning + Acting) loop with:
- Goal decomposition via the Planner
- Tool selection and execution
- Result evaluation
- Self-correction on failure with retry and replanning
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import Settings
from app.planner import Planner
from app.tools.registry import ToolRegistry
from app.tools.builtin import register_all_tools

logger = logging.getLogger(__name__)


EVAL_PROMPT = """You are evaluating whether a tool's output successfully completed the given task.

Task: {task}
Tool used: {tool}
Tool output:
{output}

Respond with ONLY a JSON object:
{{"success": true/false, "reason": "brief explanation"}}
"""

SYNTHESIS_PROMPT = """You are compiling the final answer for a user's goal based on the 
results of multiple completed steps.

Original goal: {goal}

Step results:
{results}

Write a clear, complete answer to the user's goal. Include all relevant information 
gathered from the steps. Be thorough but not redundant. No filler."""


@dataclass
class StepResult:
    step: int
    task: str
    tool: str
    output: str
    status: str  # "success", "failed", "retried"
    attempts: int = 1


@dataclass
class AgentResult:
    goal: str
    result: str
    steps_executed: int
    tools_used: list[str]
    execution_trace: list[dict]
    total_time_seconds: float


class Agent:
    """
    Autonomous agent that executes multi-step goals using the ReAct pattern.

    Flow: Plan -> (Reason -> Act -> Evaluate) per step -> Synthesize
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        self.llm = ChatOpenAI(
            model_name=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

        # Initialize tool registry
        self.registry = ToolRegistry()
        register_all_tools(self.registry, api_key=settings.openai_api_key)

        # Initialize planner
        self.planner = Planner(
            llm=self.llm,
            tool_descriptions=self.registry.get_tool_descriptions(),
            max_steps=settings.max_steps,
        )

        logger.info(
            f"Agent initialized — model: {settings.openai_model}, "
            f"tools: {len(self.registry.list_tools())}, "
            f"max_steps: {settings.max_steps}"
        )

    def run(self, goal: str) -> AgentResult:
        """
        Execute a goal end-to-end.

        1. Plan: decompose into subtasks
        2. Execute: run each step through the ReAct loop
        3. Synthesize: compile results into a final answer
        """
        start_time = time.time()
        logger.info(f"Starting goal: {goal[:100]}")

        # Step 1: Plan
        tasks = self.planner.plan(goal)
        logger.info(f"Plan: {len(tasks)} steps")

        # Step 2: Execute each task
        completed: list[StepResult] = []
        tools_used = set()

        for task in tasks:
            step_result = self._execute_step(task, completed)
            completed.append(step_result)
            tools_used.add(step_result.tool)

            # If a step failed after retries, try replanning
            if step_result.status == "failed":
                logger.warning(f"Step {task['step']} failed. Attempting replan.")
                new_tasks = self.planner.replan(
                    goal=goal,
                    completed_steps=[self._step_to_dict(s) for s in completed if s.status == "success"],
                    failed_step=task,
                    error=step_result.output,
                )
                if new_tasks:
                    for new_task in new_tasks:
                        new_result = self._execute_step(new_task, completed)
                        completed.append(new_result)
                        tools_used.add(new_result.tool)

        # Step 3: Synthesize final answer
        result = self._synthesize(goal, completed)

        elapsed = round(time.time() - start_time, 2)

        trace = []
        for s in completed:
            trace.append({
                "step": s.step,
                "task": s.task,
                "tool": s.tool,
                "status": s.status,
                "attempts": s.attempts,
            })

        return AgentResult(
            goal=goal,
            result=result,
            steps_executed=len(completed),
            tools_used=sorted(tools_used),
            execution_trace=trace,
            total_time_seconds=elapsed,
        )

    def _execute_step(self, task: dict, prior_results: list[StepResult]) -> StepResult:
        """Execute a single step with retry logic."""
        step_num = task.get("step", 0)
        task_desc = task.get("task", "")
        tool_name = task.get("tool", "")
        raw_input = task.get("input", {})

        # Resolve references to previous results
        resolved_input = self._resolve_references(raw_input, prior_results)

        logger.info(f"Step {step_num}: {task_desc} (tool: {tool_name})")

        attempts = 0
        max_attempts = self.settings.max_retries + 1

        while attempts < max_attempts:
            attempts += 1

            # Act: execute the tool
            output = self.registry.execute(tool_name, **resolved_input)

            # Evaluate: did the step succeed?
            success = self._evaluate(task_desc, tool_name, output)

            if success:
                logger.info(f"Step {step_num}: success (attempt {attempts})")
                return StepResult(
                    step=step_num,
                    task=task_desc,
                    tool=tool_name,
                    output=output,
                    status="success" if attempts == 1 else "retried",
                    attempts=attempts,
                )

            logger.warning(f"Step {step_num}: failed (attempt {attempts}/{max_attempts})")

        # All retries exhausted
        return StepResult(
            step=step_num,
            task=task_desc,
            tool=tool_name,
            output=output,
            status="failed",
            attempts=attempts,
        )

    def _evaluate(self, task: str, tool: str, output: str) -> bool:
        """Use the LLM to evaluate whether a step's output is satisfactory."""
        # Quick checks before calling the LLM
        if output.startswith("ERROR:"):
            return False
        if not output or len(output.strip()) < 10:
            return False

        try:
            messages = [
                SystemMessage(content="You evaluate tool outputs. Respond with only a JSON object."),
                HumanMessage(content=EVAL_PROMPT.format(
                    task=task, tool=tool, output=output[:2000],
                )),
            ]
            response = self.llm.invoke(messages)
            raw = response.content.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

            result = json.loads(raw)
            return result.get("success", False)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # If evaluation itself fails, assume success if output looks reasonable
            return len(output.strip()) > 20

    def _synthesize(self, goal: str, steps: list[StepResult]) -> str:
        """Compile all step results into a final answer."""
        successful = [s for s in steps if s.status in ("success", "retried")]

        if not successful:
            return "I was unable to complete this goal. All steps failed."

        results_text = ""
        for s in successful:
            # Truncate long outputs
            output = s.output[:1500] if len(s.output) > 1500 else s.output
            results_text += f"\nStep {s.step} ({s.tool}): {s.task}\nResult: {output}\n"

        messages = [
            SystemMessage(content="You compile step results into a final answer. Be thorough and clear."),
            HumanMessage(content=SYNTHESIS_PROMPT.format(
                goal=goal, results=results_text,
            )),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _resolve_references(self, raw_input: dict, prior_results: list[StepResult]) -> dict:
        """Replace {{prev_result}} and {{result_N}} placeholders with actual values."""
        resolved = {}
        for key, value in raw_input.items():
            if isinstance(value, str):
                # Replace {{prev_result}} with the last successful result
                if "{{prev_result}}" in value and prior_results:
                    last_success = None
                    for s in reversed(prior_results):
                        if s.status in ("success", "retried"):
                            last_success = s
                            break
                    if last_success:
                        value = value.replace("{{prev_result}}", last_success.output[:2000])

                # Replace {{result_N}} with specific step results
                for match in re.findall(r"\{\{result_(\d+)\}\}", value):
                    step_num = int(match)
                    for s in prior_results:
                        if s.step == step_num and s.status in ("success", "retried"):
                            value = value.replace(f"{{{{result_{match}}}}}", s.output[:2000])
                            break

            resolved[key] = value
        return resolved

    def _step_to_dict(self, step: StepResult) -> dict:
        return {
            "step": step.step,
            "task": step.task,
            "tool": step.tool,
            "status": step.status,
        }
