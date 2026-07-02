"""Runbook retriever — the RAG read path over runbooks/postmortems.

Indexes a corpus of markdown runbooks into Qdrant and answers semantic queries.
By default it uses an **in-memory** Qdrant instance, so it indexes and searches
with no Docker and no network — ideal for tests and quick local runs. Point it at
a real Qdrant server (the docker-compose service) by passing a configured client.

The embedding model is injected (see ``app.rag.embeddings``), so tests can use a
deterministic offline embedder while runtime uses fastembed.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.rag.embeddings import Embedder


class RunbookDoc(BaseModel):
    """A single runbook document to be indexed."""

    id: str
    title: str
    source: str
    content: str


class RetrievedDoc(BaseModel):
    """A search hit: a runbook plus its relevance score."""

    id: str
    title: str
    source: str
    score: float
    content: str


def _title_of(text: str, fallback: str) -> str:
    """Use the first markdown H1 as the title, else a fallback (filename)."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_runbooks(directory: str | Path) -> list[RunbookDoc]:
    """Load every ``*.md`` file in a directory into ``RunbookDoc`` objects."""
    docs: list[RunbookDoc] = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            RunbookDoc(
                id=path.stem,
                title=_title_of(text, path.stem),
                source=path.name,
                content=text,
            )
        )
    return docs


class RunbookRetriever:
    """Index runbooks into Qdrant and run semantic search over them."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection: str = "runbooks",
        client: QdrantClient | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        # ":memory:" -> a throwaway in-process Qdrant (no server, no network).
        self.client = client or QdrantClient(":memory:")
        self._indexed = 0

    def index(self, docs: list[RunbookDoc]) -> int:
        """(Re)build the collection from the given documents. Returns the count."""
        if not docs:
            return 0
        vectors = self.embedder.embed_documents([d.content for d in docs])
        dim = len(vectors[0])

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload={
                        "doc_id": doc.id,
                        "title": doc.title,
                        "source": doc.source,
                        "content": doc.content,
                    },
                )
                for i, doc in enumerate(docs)
            ],
        )
        self._indexed = len(docs)
        return self._indexed

    def index_directory(self, directory: str | Path) -> int:
        """Convenience: load a directory of ``*.md`` runbooks and index them."""
        return self.index(load_runbooks(directory))

    def search(self, query: str, k: int = 3) -> list[RetrievedDoc]:
        """Return the top-``k`` runbooks most relevant to ``query``."""
        if self._indexed == 0:
            raise RuntimeError("RunbookRetriever.search called before index()")
        query_vector = self.embedder.embed_query(query)
        hits = self.client.query_points(
            self.collection, query=query_vector, limit=k
        ).points
        return [
            RetrievedDoc(
                id=str(hit.payload["doc_id"]),
                title=hit.payload["title"],
                source=hit.payload["source"],
                score=hit.score,
                content=hit.payload["content"],
            )
            for hit in hits
        ]
