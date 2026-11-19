"""Research agent built with the OpenAI Agents SDK, using MCP servers as tools.

The agent connects to two MCP servers over stdio:
  - docs-rag  : this repo's own RAG server (ingest + semantic search)
  - playwright: Microsoft's Playwright MCP server (browser automation)

It runs on Groq's OpenAI-compatible API (free tier), so no OpenAI key is needed.
Conversation memory is kept in a SQLite session.

    uv run agents_sdk_app.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    SQLiteSession,
    set_tracing_disabled,
)
from agents.items import ToolCallItem
from agents.mcp import MCPServerStdio

load_dotenv()
# The Agents SDK's built-in tracing uploads to OpenAI; we use LangSmith for the
# LangChain app instead, so turn it off here.
set_tracing_disabled(True)

GROQ_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")

INSTRUCTIONS = """You are a research assistant with a knowledge base of the user's documents.
- For EVERY question about policies, people, company rules or any fact that could be
  in the documents, ALWAYS call the search_docs tool first, even if an earlier answer
  in this conversation said nothing was found. Never answer from memory alone.
- Answer only from the returned chunks and cite the source file of each fact,
  like [handbook.pdf].
- If search_docs returns nothing relevant, say the documents do not contain it.
- To add documents, call ingest_folder with the folder path the user gives.
- Use the browser tools only when the answer needs live web content."""


async def main() -> None:
    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )
    model = OpenAIChatCompletionsModel(model=GROQ_MODEL, openai_client=client)

    async with (
        MCPServerStdio(
            name="docs-rag",
            params={"command": "uv", "args": ["run", "rag_server.py"]},
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        ) as rag_server,
        MCPServerStdio(
            name="playwright",
            params={"command": "npx", "args": ["-y", "@playwright/mcp@latest"]},
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        ) as browser_server,
    ):
        agent = Agent(
            name="Research Assistant",
            instructions=INSTRUCTIONS,
            model=model,
            mcp_servers=[rag_server, browser_server],
        )
        session = SQLiteSession("cli-session", "agent_memory.db")

        print("Agents SDK research assistant. Type 'exit' to quit, 'clear' to reset memory.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input.lower() == "clear":
                await session.clear_session()
                print("Memory cleared.")
                continue
            if not user_input:
                continue
            try:
                result = await Runner.run(agent, user_input, session=session, max_turns=15)
                for item in result.new_items:  # show which MCP tools were used
                    if isinstance(item, ToolCallItem):
                        name = getattr(item.raw_item, "name", "tool")
                        args = getattr(item.raw_item, "arguments", "")
                        print(f"  [tool] {name}({args})")
                print(f"\nAssistant: {result.final_output}")
            except Exception as exc:  # keep the chat alive on tool/model errors
                print(f"\nError: {exc}")


if __name__ == "__main__":
    asyncio.run(main())