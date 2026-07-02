"""Embedding backends for the RAG layer.

We depend on an ``Embedder`` *protocol*, not a concrete model — the same
dependency-inversion trick used for the SRE integrations. That gives us:

* ``FastEmbedEmbedder`` — real local ONNX embeddings (no embedding API needed),
  used at runtime. It downloads the model on first use.
* ``HashingEmbedder`` — a tiny, deterministic, dependency-free embedder used in
  unit tests so retrieval can be exercised fully offline (no model download).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Documents and queries may embed differently."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (for tests / offline use).

    Not semantically smart, but good enough to rank a clearly-relevant document
    above unrelated ones based on shared vocabulary — and it needs no network or
    model download, so tests stay fast and reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    """Local ONNX embeddings via the ``fastembed`` library (runtime default)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # imported lazily (heavy + downloads)

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(next(self._model.query_embed(text)))
