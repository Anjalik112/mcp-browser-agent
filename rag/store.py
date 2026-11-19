"""ChromaDB-backed document store used by the RAG MCP server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb

from rag.chunking import SUPPORTED_SUFFIXES, chunk_text, load_text


class DocStore:
    """Ingests files into a persistent ChromaDB collection and searches them."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection: str = "docs",
        embedding_function: Any | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=db_path)
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        # Cosine distance suits sentence embeddings (default MiniLM model).
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, **kwargs
        )

    def ingest_file(self, path: Path, chunk_size: int = 800, overlap: int = 120) -> int:
        """Chunk one file and upsert its chunks. Returns the number of chunks."""
        chunks = chunk_text(load_text(path), chunk_size, overlap)
        if not chunks:
            return 0
        source = path.name
        ids = [
            hashlib.sha1(f"{source}:{i}:{c[:50]}".encode()).hexdigest()
            for i, c in enumerate(chunks)
        ]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def ingest_folder(self, folder: str) -> dict[str, int]:
        """Ingest every supported file in a folder (recursively)."""
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Folder not found: {root}")
        counts: dict[str, int] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                counts[path.name] = self.ingest_file(path)
        return counts

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return the k most similar chunks with source and similarity score."""
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_texts=[query], n_results=min(k, self.collection.count())
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk": meta["chunk"],
                    "score": round(1 - dist, 4),  # cosine similarity
                }
            )
        return hits

    def sources(self) -> dict[str, int]:
        """Return each ingested file name with its chunk count."""
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for m in metas:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts
