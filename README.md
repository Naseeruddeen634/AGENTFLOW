# AgentFlow — Agentic Workflow Orchestrator

An autonomous AI agent that takes a high-level goal in plain English, decomposes it into subtasks, selects and executes tools, evaluates results, and self-corrects on failure. Implements the ReAct (Reasoning + Acting) pattern with dynamic replanning.

Built with LangChain, OpenAI GPT-4, and FastAPI. Fully containerized with Docker.

---

## How It Works

Traditional chatbots handle one question at a time. AgentFlow handles multi-step goals autonomously.

You give it a goal like "Research the top 3 AI startups in Dublin and summarize what they do" and it will:

1. Break the goal into subtasks (plan)
2. Pick the right tool for each step (reason)
3. Execute the tool and capture results (act)
4. Evaluate whether the step succeeded (verify)
5. Retry or replan if something fails (self-correct)
6. Compile a final output once all steps are complete

This is the ReAct loop — Reasoning and Acting in alternation until the goal is met.

---

## Architecture

```
User Goal
    |
    v
Planner (GPT-4 decomposes goal into ordered subtasks)
    |
    v
Task Queue
    |
    v
+---------------------------+
| For each task:            |
|   Reasoner: pick a tool   |
|   Executor: run the tool  |
|   Evaluator: check output |
|       |                   |
|    Success? ---No---> Retry (max 2) or Replan
|       |                   |
|      Yes                  |
|       |                   |
|   Next task               |
+---------------------------+
    |
    v
Synthesizer (GPT-4 compiles all results into final answer)
    |
    v
Structured Response with execution trace
```

---

## Available Tools

| Tool           | Description                                      |
|----------------|--------------------------------------------------|
| `web_search`   | Search the web using SerpAPI / DuckDuckGo        |
| `web_scrape`   | Extract text content from a URL                  |
| `calculator`   | Evaluate mathematical expressions safely         |
| `file_writer`  | Save content to a local file                     |
| `summarizer`   | Condense long text into key points using an LLM  |

Tools are registered in a central registry. Adding a new tool requires a function, a name, a description, and an input schema — the agent discovers and selects tools at runtime.

---

## Tech Stack

| Component       | Technology                    |
|-----------------|-------------------------------|
| LLM             | OpenAI GPT-4 / GPT-3.5-turbo |
| Agent Framework | LangChain + custom ReAct loop |
| API             | FastAPI + Uvicorn             |
| Web Search      | DuckDuckGo Search             |
| Web Scraping    | BeautifulSoup4 + httpx        |
| Containerization| Docker + Docker Compose       |

---

## Quick Start

### Local

```bash
git clone https://github.com/Naseeruddeen634/agentflow.git
cd agentflow

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export OPENAI_API_KEY="sk-your-key-here"

uvicorn app.main:app --reload --port 8001
```

### Docker

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
docker-compose up --build
```

API at `http://localhost:8001`. Docs at `http://localhost:8001/docs`.

---

## API Endpoints

| Method | Endpoint        | Description                              |
|--------|-----------------|------------------------------------------|
| POST   | `/run`          | Submit a goal for the agent to execute   |
| GET    | `/tools`        | List all registered tools                |
| GET    | `/health`       | System health check                      |

### Example

```bash
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Find the top 3 AI companies in Dublin and summarize what each one does"}'
```

Response:
```json
{
  "goal": "Find the top 3 AI companies in Dublin...",
  "result": "Here are the top 3 AI companies in Dublin: ...",
  "steps_executed": 5,
  "tools_used": ["web_search", "web_scrape", "summarizer"],
  "execution_trace": [
    {"step": 1, "task": "Search for AI companies in Dublin", "tool": "web_search", "status": "success"},
    {"step": 2, "task": "Scrape company details", "tool": "web_scrape", "status": "success"},
    {"step": 3, "task": "Summarize findings", "tool": "summarizer", "status": "success"}
  ],
  "total_time_seconds": 12.4
}
```

---

## Project Structure

```
agentflow/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── agent.py             # Core agent with ReAct loop
│   ├── planner.py           # Goal decomposition into subtasks
│   ├── config.py            # Configuration management
│   └── tools/
│       ├── __init__.py
│       ├── registry.py      # Tool registration and discovery
│       └── builtin.py       # Built-in tool implementations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Configuration

| Variable           | Default              | Description                  |
|--------------------|----------------------|------------------------------|
| `OPENAI_API_KEY`   | (required)           | OpenAI API key               |
| `OPENAI_MODEL`     | `gpt-4`             | Model for planning/reasoning |
| `MAX_RETRIES`      | `2`                  | Retries per failed step      |
| `MAX_STEPS`        | `10`                 | Maximum steps per goal       |
| `TEMPERATURE`      | `0.1`                | LLM temperature              |

---

## Author

Naseeruddeen — MSc AI / Machine Learning, Dublin City University
- GitHub: [github.com/Naseeruddeen634](https://github.com/Naseeruddeen634)
- LinkedIn: [linkedin.com/in/naseeruddeen](https://linkedin.com/in/naseeruddeen)

## License

MIT
