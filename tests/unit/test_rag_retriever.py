"""Unit tests for the RAG runbook retriever (Layer 2).

Runs fully offline: an in-memory Qdrant + the deterministic ``HashingEmbedder``
(no model download, no network), yet exercises the real indexing/search path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import RunbookRetriever, load_runbooks

RUNBOOKS = Path(__file__).resolve().parents[2] / "data" / "runbooks"


@pytest.fixture
def retriever() -> RunbookRetriever:
    r = RunbookRetriever(HashingEmbedder(), collection="test_runbooks")
    r.index_directory(RUNBOOKS)
    return r


def test_loads_all_runbooks() -> None:
    docs = load_runbooks(RUNBOOKS)
    sources = {d.source for d in docs}
    assert "checkout-payment-5xx.md" in sources
    assert len(docs) >= 3
    # Title is parsed from the markdown H1.
    checkout = next(d for d in docs if d.source == "checkout-payment-5xx.md")
    assert checkout.title == "Checkout 5xx / Payment Errors"


def test_search_returns_most_relevant_runbook_first(retriever: RunbookRetriever) -> None:
    hits = retriever.search(
        "checkout returning 500 NullPointerException in PaymentClient after deploy",
        k=3,
    )
    assert hits, "expected at least one hit"
    assert hits[0].source == "checkout-payment-5xx.md"
    # Scores are sorted descending.
    assert all(
        hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)
    )


def test_search_discriminates_other_topics(retriever: RunbookRetriever) -> None:
    hits = retriever.search("database connection pool timeout acquire", k=1)
    assert hits[0].source == "database-connection-pool.md"


def test_search_before_index_raises() -> None:
    r = RunbookRetriever(HashingEmbedder(), collection="empty")
    with pytest.raises(RuntimeError):
        r.search("anything")
