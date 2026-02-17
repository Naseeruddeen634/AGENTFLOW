"""FastAPI application for AgentFlow."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.agent import Agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    settings = get_settings()
    agent = Agent(settings)
    logger.info("AgentFlow ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AgentFlow",
    description="Agentic Workflow Orchestrator — submit a goal, get autonomous execution",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GoalRequest(BaseModel):
    goal: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="A high-level goal for the agent to accomplish",
        json_schema_extra={"example": "Find the top 3 AI startups in Dublin and summarize what each one does"},
    )


class StepTrace(BaseModel):
    step: int
    task: str
    tool: str
    status: str
    attempts: int


class GoalResponse(BaseModel):
    goal: str
    result: str
    steps_executed: int
    tools_used: list[str]
    execution_trace: list[StepTrace]
    total_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    tools_available: int
    max_steps: int


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    tools = agent.registry.list_tools()
    return HealthResponse(
        status="healthy",
        tools_available=len(tools),
        max_steps=agent.settings.max_steps,
    )


@app.get("/tools", tags=["System"])
async def list_tools():
    """List all tools available to the agent."""
    return {"tools": agent.registry.list_tools()}


@app.post("/run", response_model=GoalResponse, tags=["Agent"])
async def run_goal(request: GoalRequest):
    """
    Submit a goal for autonomous execution.

    The agent will plan, execute, evaluate, and self-correct
    until the goal is completed or max steps are reached.
    """
    try:
        result = agent.run(request.goal)
        return GoalResponse(
            goal=result.goal,
            result=result.result,
            steps_executed=result.steps_executed,
            tools_used=result.tools_used,
            execution_trace=[
                StepTrace(**step) for step in result.execution_trace
            ],
            total_time_seconds=result.total_time_seconds,
        )
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
