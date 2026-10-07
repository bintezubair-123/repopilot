"""Embeddings, Chroma store, and hybrid (vector + BM25) retrieval with RRF."""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from .ingest import EmptyRepoError, chunk_repo
from .schemas import Chunk, IndexStats, SearchHit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- embeddings

class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local, free embeddings (no API). Model is loaded lazily on first use."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                ) from exc
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._load().encode(
            list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._load().encode(
            [self.QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


# ---------------------------------------------------------------- tokenizing (BM25)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARTS = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase identifiers, also split camelCase / snake_case into sub-words."""
    out: list[str] = []
    for word in _WORD.findall(text):
        out.append(word.lower())
        parts = _PARTS.findall(word)
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


# ---------------------------------------------------------------- Chroma store

class ChromaStore:
    def __init__(self, persist_dir: Path | str, embedder: Embedder,
                 collection_name: str = "repopilot") -> None:
        self._embedder = embedder
        self._name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:  # error type differs across chromadb versions
            logger.debug("No existing collection to delete")
        self._collection = self._get_collection()

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[_to_meta(c) for c in batch],
                embeddings=self._embedder.embed_documents([c.embed_text for c in batch]),
            )

    def all_chunks(self) -> list[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        chunks = [_from_meta(d, m) for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.file_path, c.start_line))

    def vector_search(self, query: str, k: int) -> list[SearchHit]:
        n = self.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchHit(chunk=_from_meta(d, m), score=1.0 - dist)
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


def _to_meta(c: Chunk) -> dict:
    return {
        "file_path": c.file_path, "language": c.language, "start_line": c.start_line,
        "end_line": c.end_line, "symbol_name": c.symbol_name, "chunk_type": c.chunk_type,
    }


def _from_meta(doc: str, m: dict) -> Chunk:
    return Chunk(text=doc, **m)


# ---------------------------------------------------------------- hybrid retrieval

def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Merge several ranked id lists. Higher score = better."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Vector search + BM25, merged by reciprocal rank fusion."""

    def __init__(self, store: ChromaStore, candidates: int = 20, rrf_k: int = 60) -> None:
        self.store = store
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._built_for = -1

    def _ensure_bm25(self) -> None:
        n = self.store.count()
        if n == self._built_for:
            return
        self._chunks = self.store.all_chunks()
        self._bm25 = (
            BM25Okapi([tokenize(c.embed_text) for c in self._chunks]) if self._chunks else None
        )
        self._built_for = n

    def _bm25_ranking(self, query: str) -> list[str]:
        self._ensure_bm25()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i].chunk_id for i in order[: self.candidates] if scores[i] > 0]

    def search(self, query: str, k: int = 6) -> list[SearchHit]:
        if self.store.count() == 0:
            return []
        vector_hits = self.store.vector_search(query, self.candidates)
        bm25_ids = self._bm25_ranking(query)  # also refreshes self._chunks
        by_id = {c.chunk_id: c for c in self._chunks}
        by_id.update({h.chunk.chunk_id: h.chunk for h in vector_hits})
        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in vector_hits], bm25_ids], k=self.rrf_k
        )
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [SearchHit(chunk=by_id[cid], score=score) for cid, score in top]


# ---------------------------------------------------------------- indexing

def index_repo(repo_path: Path | str, store: ChromaStore) -> IndexStats:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repo path not found or not a directory: {root}")
    chunks = list(chunk_repo(root))
    if not chunks:
        raise EmptyRepoError(
            f"No indexable files found in {root} (supported: Python, JS/TS, Markdown, and a few others)."
        )
    store.reset()  # only after we know there is something to index
    store.add_chunks(chunks)
    stats = IndexStats(files=len({c.file_path for c in chunks}), chunks=len(chunks))
    logger.info("Indexed %d chunks from %d files", stats.chunks, stats.files)
    return stats
