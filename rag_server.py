"""RAG MCP server: exposes document ingestion and semantic search as MCP tools.

Run over stdio (default, used by agents and Claude Desktop):
    uv run rag_server.py
Run over SSE (for remote / HTTP clients):
    uv run rag_server.py --sse
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

from rag.store import DocStore

mcp = FastMCP(
    name="docs-rag",
    instructions=(
        "Search the user's ingested documents. Call search_docs before answering "
        "questions about them, and cite the source file of each fact."
    ),
)

_store: DocStore | None = None


def store() -> DocStore:
    """Create the store lazily so the server starts fast."""
    global _store
    if _store is None:
        _store = DocStore(db_path=os.getenv("RAG_DB_PATH", "./chroma_db"))
    return _store


@mcp.tool()
def ingest_folder(folder: str) -> dict[str, int]:
    """Ingest all .txt, .md and .pdf files in a folder into the knowledge base.

    Returns each file name with the number of chunks stored.
    """
    return store().ingest_folder(folder)


@mcp.tool()
def search_docs(query: str, k: int = 4) -> list[dict]:
    """Semantic search over ingested documents.

    Returns up to k chunks with their source file, chunk number and
    cosine-similarity score (higher is closer).
    """
    return store().search(query, k)


@mcp.tool()
def list_sources() -> dict[str, int]:
    """List ingested files and how many chunks each one has."""
    return store().sources()


if __name__ == "__main__":
    mcp.run(transport="sse" if "--sse" in sys.argv else "stdio")
