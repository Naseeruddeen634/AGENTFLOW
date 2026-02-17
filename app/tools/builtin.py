"""
Built-in tool implementations for AgentFlow.
Each tool is a standalone function that gets registered in the tool registry.
"""

import logging
import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# -- Web Search --

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', 'N/A')}\n"
                f"    {r.get('body', 'No snippet')}"
            )
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search failed: {str(e)}"


# -- Web Scrape --

def web_scrape(url: str) -> str:
    """Extract readable text content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AgentFlow/1.0)"
        }
        response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, nav elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        # Truncate to avoid blowing up the context window
        if len(cleaned) > 4000:
            cleaned = cleaned[:4000] + "\n\n[Content truncated at 4000 characters]"

        return cleaned if cleaned else "No readable content found at this URL."

    except Exception as e:
        return f"Scraping failed: {str(e)}"


# -- Calculator --

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Supports basic math operations."""
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "ERROR: Expression contains invalid characters. Use only numbers and +-*/()."

        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except ZeroDivisionError:
        return "ERROR: Division by zero."
    except Exception as e:
        return f"Calculation failed: {str(e)}"


# -- File Writer --

def file_writer(filename: str, content: str) -> str:
    """Save content to a file in the outputs directory."""
    try:
        output_dir = Path("./agent_outputs")
        output_dir.mkdir(exist_ok=True)

        # Sanitize filename
        safe_name = re.sub(r'[^\w\-.]', '_', filename)
        file_path = output_dir / safe_name

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File saved: {file_path}"

    except Exception as e:
        return f"File write failed: {str(e)}"


# -- Summarizer --

def summarizer(text: str, api_key: str = "", model: str = "gpt-3.5-turbo") -> str:
    """Condense long text into a concise summary using an LLM."""
    try:
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY", "")

        llm = ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            temperature=0.1,
            max_tokens=512,
        )

        # Truncate input if too long
        if len(text) > 6000:
            text = text[:6000] + "\n[Truncated]"

        messages = [
            SystemMessage(content="Summarize the following text concisely. Focus on key facts, findings, and takeaways. No filler."),
            HumanMessage(content=text),
        ]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"Summarization failed: {str(e)}"


# -- Registration Helper --

def register_all_tools(registry, api_key: str = ""):
    """Register all built-in tools with the given registry."""

    registry.register(
        name="web_search",
        description="Search the web for current information. Returns top results with titles, URLs, and snippets.",
        func=web_search,
        parameters={"query": "string (search query)", "max_results": "int (default 5)"},
    )

    registry.register(
        name="web_scrape",
        description="Extract readable text content from a web page URL.",
        func=web_scrape,
        parameters={"url": "string (full URL to scrape)"},
    )

    registry.register(
        name="calculator",
        description="Evaluate a mathematical expression. Supports +, -, *, /, parentheses.",
        func=calculator,
        parameters={"expression": "string (math expression like '(15 * 3) + 42')"},
    )

    registry.register(
        name="file_writer",
        description="Save text content to a file. Use for generating reports or saving results.",
        func=file_writer,
        parameters={"filename": "string", "content": "string"},
    )

    # Wrap summarizer to inject api_key
    def _summarizer(text: str) -> str:
        return summarizer(text, api_key=api_key)

    registry.register(
        name="summarizer",
        description="Condense long text into a concise summary. Use when you have large amounts of text that need to be shortened.",
        func=_summarizer,
        parameters={"text": "string (the text to summarize)"},
    )
