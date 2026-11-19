"""Unit tests for chunking and the ChromaDB document store.

A tiny deterministic embedding function is used so tests run offline,
without downloading an embedding model.
"""

from __future__ import annotations

import hashlib

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from rag.chunking import chunk_text
from rag.store import DocStore


class HashEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words hashing into 64 dims: similar words give similar vectors."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = [0.0] * 64
            for word in text.lower().split():
                v[int(hashlib.md5(word.strip(".,?!").encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(v)
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-test"


def test_chunk_text_respects_size_and_overlap():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share text because of the overlap
    assert chunks[0][-30:].strip() in chunks[1]


def test_chunk_text_empty_and_invalid():
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)


def test_store_ingest_and_search(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.md").write_text("Employees get 24 days of paid annual leave each year.")
    (docs / "laptop.txt").write_text("Laptops are refreshed every three years by IT support.")

    store = DocStore(db_path=str(tmp_path / "db"), embedding_function=HashEmbedding())
    counts = store.ingest_folder(str(docs))
    assert counts == {"laptop.txt": 1, "leave.md": 1}

    hits = store.search("how many days of paid leave", k=1)
    assert hits[0]["source"] == "leave.md"
    assert 0 <= hits[0]["score"] <= 1
    assert store.sources() == {"laptop.txt": 1, "leave.md": 1}


def test_reingest_does_not_duplicate(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Same content.")
    store = DocStore(db_path=str(tmp_path / "db"), embedding_function=HashEmbedding())
    store.ingest_folder(str(docs))
    store.ingest_folder(str(docs))
    assert store.sources() == {"a.txt": 1}
