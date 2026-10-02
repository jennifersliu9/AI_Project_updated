"""OpenAI cloud embeddings with an in-process cosine index.

The process never downloads sentence-transformer weights. Ingest calls the
OpenAI embeddings API and caches the vectors on disk. Query time embeds one
string and ranks the cached matrix in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from harborline.config import Settings, get_settings
from harborline.ingest import Chunk, load_chunks

_EMBED_BATCH = 64
_indexes: dict[str, EmbeddingIndex] = {}


class VectorStoreError(RuntimeError):
    """Missing credentials or a failed call to the OpenAI embeddings API."""


def _meta_value(value: object) -> str | int | float | bool:
    if isinstance(value, bool | int | float):
        return value
    if value is None:
        return ""
    return str(value)


def chunk_to_metadata(chunk: Chunk) -> dict:
    meta = {
        "chunk_id": chunk.chunk_id,
        "source_path": chunk.source_path,
        "source_name": chunk.source_name,
        "source_format": chunk.source_format,
        "title": chunk.title,
        "section": chunk.section,
        "kind": chunk.kind,
        "snippet": chunk.snippet,
        "order": chunk.order,
        "text": chunk.text,
    }
    for key, value in chunk.extra.items():
        meta[key] = _meta_value(value)
    return meta


def metadata_to_chunk(meta: dict) -> Chunk:
    extra = {
        k: meta.get(k, "")
        for k in ("dataset", "employee_id", "office_id", "ticket_id", "policy_hint")
        if k in meta
    }
    text = str(meta.get("text", ""))
    return Chunk(
        chunk_id=str(meta.get("chunk_id", "")),
        source_path=str(meta.get("source_path", "")),
        source_name=str(meta.get("source_name", "")),
        source_format=str(meta.get("source_format", "")),
        title=str(meta.get("title", "")),
        section=str(meta.get("section", "")),
        kind=str(meta.get("kind", "policy")),
        text=text,
        snippet=str(meta.get("snippet") or text[:240]),
        order=int(meta.get("order") or 0),
        extra=extra,
    )


def vector_index_label(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.retrieve_backend == "openai":
        return f"openai:{settings.embedding_model}"
    if settings.retrieve_backend == "tfidf":
        return "local:tfidf"
    return settings.retrieve_backend


def _is_local_embedding_model(model: str) -> bool:
    """HuggingFace repo ids and sentence-transformer names download weights locally."""
    lowered = model.strip().lower()
    if not lowered:
        return False
    markers = (
        "/",
        "sentence-transformer",
        "sentence_transformer",
        "huggingface",
        "all-minilm",
        "fastembed",
    )
    return any(marker in lowered for marker in markers)


class OpenAIEmbeddings:
    """Cloud embeddings via the OpenAI API.

    This is the only embedding class. It does not construct a HuggingFace
    pipeline and it does not load sentence-transformer weights into RAM.
    """

    provider = "openai"

    def __init__(self, settings: Settings):
        if settings.embedding_provider != self.provider:
            raise VectorStoreError(
                f"embedding_provider={settings.embedding_provider!r} is not supported. "
                "This process only constructs OpenAIEmbeddings."
            )
        model = (settings.embedding_model or "").strip()
        if _is_local_embedding_model(model):
            raise VectorStoreError(
                f"HARBORLINE_EMBEDDING_MODEL={model!r} is a local embedding model. "
                "Embeddings use OpenAIEmbeddings, which calls the OpenAI API and "
                "keeps no model weights in memory. Set "
                "HARBORLINE_EMBEDDING_MODEL=text-embedding-3-small."
            )
        key = (settings.openai_api_key or "").strip()
        if not key:
            raise VectorStoreError(
                "OPENAI_API_KEY is not set. OpenAIEmbeddings calls "
                f"the OpenAI API for {model or 'text-embedding-3-small'} and does not "
                "download sentence-transformer weights. Set the key, or use "
                "HARBORLINE_RETRIEVE_BACKEND=tfidf for offline lexical search."
            )
        from openai import OpenAI

        kwargs: dict = {"api_key": key, "timeout": 60.0}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self.model = model or "text-embedding-3-small"
        self._client = OpenAI(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            response = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise VectorStoreError(
                    f"OpenAI returned {len(ordered)} embeddings for {len(batch)} inputs."
                )
            vectors.extend(list(item.embedding) for item in ordered)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Always return OpenAIEmbeddings. No HuggingFace embeddings class exists."""
    settings = settings or get_settings()
    return OpenAIEmbeddings(settings)


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed with OpenAIEmbeddings. No local model weights are loaded."""
    if not texts:
        return []
    return build_embeddings(settings).embed_documents(texts)


@dataclass
class EmbeddingIndex:
    """L2-normalized vectors for one embedding model, aligned with chunks."""

    model: str
    chunks: list[Chunk]
    matrix: np.ndarray


def clear_index_cache() -> None:
    """Drop the in-process matrix. The on-disk cache is left in place."""
    _indexes.clear()


def _cache_key(settings: Settings) -> str:
    return str(settings.cache_dir.resolve())


def _cache_paths(settings: Settings) -> tuple:
    root = settings.cache_dir
    return root / "openai_vectors.npz", root / "openai_chunks.json"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _read_disk(settings: Settings) -> EmbeddingIndex | None:
    npz_path, json_path = _cache_paths(settings)
    if not npz_path.exists() or not json_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    model = str(payload.get("model") or "")
    if model != settings.embedding_model:
        return None
    raw_chunks = payload.get("chunks") or []
    chunks = [metadata_to_chunk(item) for item in raw_chunks]
    with np.load(npz_path) as data:
        matrix = np.asarray(data["vectors"], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        return None
    return EmbeddingIndex(model=model, chunks=chunks, matrix=_normalize_rows(matrix))


def load_index(settings: Settings | None = None) -> EmbeddingIndex | None:
    """Return the cached OpenAI embedding index, or None when ingest has not run."""
    settings = settings or get_settings()
    key = _cache_key(settings)
    cached = _indexes.get(key)
    if cached is not None and cached.model == settings.embedding_model:
        return cached
    loaded = _read_disk(settings)
    if loaded is not None:
        _indexes[key] = loaded
    return loaded


def persist_chunks(chunks: list[Chunk] | None = None, settings: Settings | None = None) -> int:
    """Embed chunks with the OpenAI API and replace the local cosine index."""
    settings = settings or get_settings()
    chunks = chunks if chunks is not None else load_chunks(settings)
    if not chunks:
        return 0
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), _EMBED_BATCH):
        piece = chunks[start : start + _EMBED_BATCH]
        batch = embed_texts([chunk.text for chunk in piece], settings)
        if len(batch) != len(piece):
            raise VectorStoreError(
                f"Expected {len(piece)} embeddings and received {len(batch)}."
            )
        vectors.extend(batch)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise VectorStoreError(
            f"Expected a ({len(chunks)}, dim) embedding matrix and received shape {matrix.shape}."
        )
    index = EmbeddingIndex(
        model=settings.embedding_model,
        chunks=list(chunks),
        matrix=_normalize_rows(matrix),
    )
    npz_path, json_path = _cache_paths(settings)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, vectors=index.matrix)
    json_path.write_text(
        json.dumps(
            {
                "model": settings.embedding_model,
                "chunks": [chunk_to_metadata(chunk) for chunk in chunks],
            }
        ),
        encoding="utf-8",
    )
    _indexes[_cache_key(settings)] = index
    return len(chunks)


def collection_count(settings: Settings | None = None) -> int:
    index = load_index(settings)
    return 0 if index is None else len(index.chunks)


def query_vectors(
    vector: list[float],
    top_k: int,
    settings: Settings | None = None,
) -> list[tuple[float, dict]]:
    """Return (cosine, metadata) pairs from the OpenAI embedding index."""
    settings = settings or get_settings()
    index = load_index(settings)
    if index is None or not index.chunks or top_k <= 0:
        return []
    query = np.asarray(vector, dtype=np.float32).reshape(-1)
    width = int(index.matrix.shape[1])
    if query.shape != (width,):
        raise VectorStoreError(
            f"Query embedding dimension {query.shape[0]} does not match the stored "
            f"{width} from {settings.embedding_model}."
        )
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = index.matrix @ query
    k = min(top_k, int(scores.shape[0]))
    if k == scores.shape[0]:
        order = np.argsort(-scores)
    else:
        picked = np.argpartition(-scores, k - 1)[:k]
        order = picked[np.argsort(-scores[picked])]
    matches: list[tuple[float, dict]] = []
    for position in order:
        meta = chunk_to_metadata(index.chunks[int(position)])
        matches.append((float(scores[int(position)]), meta))
    return matches
