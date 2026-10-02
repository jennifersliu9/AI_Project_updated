"""OpenAI embeddings stored in a hosted Pinecone index.

The process never downloads sentence-transformer weights and never keeps the
corpus index resident. Ingest uploads vectors to Pinecone. Query time embeds
one string and reads back the top matches.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from harborline.config import Settings, get_settings
from harborline.ingest import Chunk, load_chunks

_EMBED_BATCH = 64
_UPSERT_BATCH = 50
_host_by_index: dict[str, str] = {}


class VectorStoreError(RuntimeError):
    """Missing credentials or a failed call to OpenAI or Pinecone."""


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
    if settings.retrieve_backend == "pinecone":
        return f"hosted:pinecone/{settings.pinecone_namespace}"
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


def _require_pinecone(settings: Settings) -> None:
    if not (settings.pinecone_api_key or "").strip():
        raise VectorStoreError(
            "PINECONE_API_KEY is not set. The vector index is hosted in Pinecone "
            "so this process does not keep it in memory. Set PINECONE_API_KEY and "
            "PINECONE_INDEX_HOST (or PINECONE_INDEX_NAME), or use "
            "HARBORLINE_RETRIEVE_BACKEND=tfidf."
        )
    if not (settings.pinecone_index_host or settings.pinecone_index_name):
        raise VectorStoreError(
            "Set PINECONE_INDEX_HOST or PINECONE_INDEX_NAME. Create a cosine index "
            f"with dimension 1536 for {settings.embedding_model}."
        )


def _pinecone_json(settings: Settings, method: str, url: str, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("Api-Key", settings.pinecone_api_key or "")
    req.add_header("X-Pinecone-API-Version", settings.pinecone_api_version)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VectorStoreError(
            f"Pinecone {method} {url} failed ({exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise VectorStoreError(f"Pinecone {method} {url} failed: {exc.reason}") from exc
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_host(raw: str | None) -> str:
    host = (raw or "").strip()
    if not host:
        return ""
    host = host.removeprefix("https://").removeprefix("http://").strip("/")
    return host.split("/")[0]


def index_host(settings: Settings) -> str:
    """Data-plane host. Resolves PINECONE_INDEX_NAME through the control plane once."""
    explicit = _normalize_host(settings.pinecone_index_host)
    if explicit:
        return explicit
    _require_pinecone(settings)
    name = (settings.pinecone_index_name or "").strip()
    cached = _host_by_index.get(name)
    if cached:
        return cached
    described = _pinecone_json(
        settings,
        "GET",
        f"https://api.pinecone.io/indexes/{urllib.parse.quote(name)}",
    )
    host = _normalize_host(str(described.get("host") or ""))
    if not host:
        raise VectorStoreError(f"Pinecone index {name!r} did not return a host.")
    _host_by_index[name] = host
    return host


def _pinecone_metadata(chunk: Chunk) -> dict:
    """Flat metadata. Empty strings are omitted; Pinecone stores the rest."""
    meta = {}
    for key, value in chunk_to_metadata(chunk).items():
        if value == "":
            continue
        meta[key] = value
    return meta


def persist_chunks(chunks: list[Chunk] | None = None, settings: Settings | None = None) -> int:
    """Replace the Pinecone namespace with embeddings of these chunks."""
    settings = settings or get_settings()
    chunks = chunks if chunks is not None else load_chunks(settings)
    if not chunks:
        return 0
    _require_pinecone(settings)
    host = index_host(settings)
    namespace = settings.pinecone_namespace
    _pinecone_json(
        settings,
        "POST",
        f"https://{host}/vectors/delete",
        {"deleteAll": True, "namespace": namespace},
    )
    for start in range(0, len(chunks), _UPSERT_BATCH):
        piece = chunks[start : start + _UPSERT_BATCH]
        vectors = embed_texts([c.text for c in piece], settings)
        if len(vectors) != len(piece):
            raise VectorStoreError(
                f"Expected {len(piece)} embeddings and received {len(vectors)}."
            )
        _pinecone_json(
            settings,
            "POST",
            f"https://{host}/vectors/upsert",
            {
                "namespace": namespace,
                "vectors": [
                    {"id": chunk.chunk_id, "values": vector, "metadata": _pinecone_metadata(chunk)}
                    for chunk, vector in zip(piece, vectors, strict=True)
                ],
            },
        )
    return len(chunks)


def collection_count(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    _require_pinecone(settings)
    host = index_host(settings)
    payload = _pinecone_json(settings, "POST", f"https://{host}/describe_index_stats", {})
    namespaces = payload.get("namespaces") or {}
    info = namespaces.get(settings.pinecone_namespace) or {}
    return int(info.get("vectorCount") or 0)


def query_vectors(
    vector: list[float],
    top_k: int,
    settings: Settings | None = None,
) -> list[tuple[float, dict]]:
    """Return (score, metadata) pairs. Vector values are not copied back."""
    settings = settings or get_settings()
    _require_pinecone(settings)
    host = index_host(settings)
    payload = _pinecone_json(
        settings,
        "POST",
        f"https://{host}/query",
        {
            "namespace": settings.pinecone_namespace,
            "vector": vector,
            "topK": top_k,
            "includeMetadata": True,
            "includeValues": False,
        },
    )
    matches: list[tuple[float, dict]] = []
    for match in payload.get("matches") or []:
        meta = dict(match.get("metadata") or {})
        if not meta.get("chunk_id"):
            meta["chunk_id"] = str(match.get("id") or "")
        matches.append((float(match.get("score") or 0.0), meta))
    return matches
