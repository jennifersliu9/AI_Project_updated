"""Hosted embeddings: OpenAI for vectors, Pinecone for the index."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from harborline.config import get_settings
from harborline.ingest import Chunk
from harborline.retrieve import VectorRetriever, build_retriever
from harborline.store import (
    OpenAIEmbeddings,
    VectorStoreError,
    build_embeddings,
    chunk_to_metadata,
    collection_count,
    embed_texts,
    index_host,
    metadata_to_chunk,
    persist_chunks,
)


def _settings(**kwargs):
    return replace(get_settings(), **kwargs)


def _chunk(i: int = 0, **kwargs) -> Chunk:
    payload = dict(
        chunk_id=f"corpus:pto.md:{i}",
        source_path="corpus/01-paid-time-off.md",
        source_name="01-paid-time-off.md",
        source_format="md",
        title="PTO",
        section="Eligibility",
        kind="policy",
        text="Employees accrue 15 days of PTO.",
        snippet="Employees accrue 15 days of PTO.",
        order=i,
        extra={},
    )
    payload.update(kwargs)
    return Chunk(**payload)


class _Resp:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _headers(req) -> dict[str, str]:
    return {key.lower(): value for key, value in req.header_items()}


def test_runtime_deps_omit_local_embedding_weights():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    for text in (requirements, pyproject):
        lowered = text.lower()
        assert "faiss" not in lowered
        assert "fastembed" not in lowered
        assert "sentence-transformers" not in lowered
        assert "sentence_transformers" not in lowered


def test_import_store_does_not_load_weight_libraries():
    import harborline.store  # noqa: F401

    assert "faiss" not in sys.modules
    assert "fastembed" not in sys.modules
    assert "sentence_transformers" not in sys.modules
    assert "transformers" not in sys.modules
    assert "langchain_huggingface" not in sys.modules


def test_metadata_roundtrip():
    chunk = _chunk(extra={"employee_id": "EMP-1008", "dataset": "employees"})
    again = metadata_to_chunk(chunk_to_metadata(chunk))
    assert again.chunk_id == chunk.chunk_id
    assert again.text == chunk.text
    assert again.extra["employee_id"] == "EMP-1008"


def test_embed_texts_calls_openai_and_preserves_order(monkeypatch):
    class _Item:
        def __init__(self, index: int, embedding: list[float]):
            self.index = index
            self.embedding = embedding

    class _EmbedResponse:
        def __init__(self, data):
            self.data = data

    class _Embeddings:
        def create(self, model, input):
            assert model == "text-embedding-3-small"
            data = [_Item(i, [float(i), 1.0]) for i in range(len(input))]
            data.reverse()
            return _EmbedResponse(data)

    class _OpenAI:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "sk-test"
            assert kwargs["base_url"] == "https://example.openai.test/v1"
            self.embeddings = _Embeddings()

    monkeypatch.setattr("openai.OpenAI", _OpenAI)
    settings = _settings(
        openai_api_key="sk-test",
        openai_base_url="https://example.openai.test/v1",
        embedding_model="text-embedding-3-small",
    )
    embedder = build_embeddings(settings)
    assert isinstance(embedder, OpenAIEmbeddings)
    assert embedder.provider == "openai"
    assert settings.embedding_provider == "openai"
    assert embedder.embed_documents(["alpha", "beta"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert embed_texts(["alpha", "beta"], settings) == [[0.0, 1.0], [1.0, 1.0]]
    assert embedder.embed_query("alpha") == [0.0, 1.0]


def test_local_embedding_model_is_rejected():
    settings = _settings(
        openai_api_key="sk-test",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_provider="openai",
    )
    with pytest.raises(VectorStoreError, match="OpenAIEmbeddings"):
        build_embeddings(settings)


def test_embed_texts_requires_api_key():
    settings = _settings(openai_api_key=None, embedding_model="text-embedding-3-small")
    with pytest.raises(VectorStoreError, match="OPENAI_API_KEY"):
        embed_texts(["hello"], settings)


def test_pinecone_requires_key_and_index():
    missing_key = _settings(pinecone_api_key=None, pinecone_index_host="idx.svc.pinecone.io")
    with pytest.raises(VectorStoreError, match="PINECONE_API_KEY"):
        collection_count(missing_key)
    missing_index = _settings(
        pinecone_api_key="key",
        pinecone_index_host=None,
        pinecone_index_name=None,
    )
    with pytest.raises(VectorStoreError, match="PINECONE_INDEX_HOST"):
        collection_count(missing_index)


def test_index_host_resolves_from_control_plane(monkeypatch):
    from harborline import store

    store._host_by_index.clear()

    def fake_urlopen(req, timeout=60):
        assert req.full_url == "https://api.pinecone.io/indexes/harborline-test"
        assert req.get_method() == "GET"
        assert _headers(req)["api-key"] == "pc-test"
        assert _headers(req)["x-pinecone-api-version"] == "2025-04"
        return _Resp({"host": "https://harborline-abc.svc.pinecone.io"})

    monkeypatch.setattr("harborline.store.urllib.request.urlopen", fake_urlopen)
    settings = _settings(
        pinecone_api_key="pc-test",
        pinecone_index_host=None,
        pinecone_index_name="harborline-test",
        pinecone_api_version="2025-04",
    )
    assert index_host(settings) == "harborline-abc.svc.pinecone.io"
    assert index_host(settings) == "harborline-abc.svc.pinecone.io"


def test_persist_upserts_without_keeping_an_index(monkeypatch):
    calls: list[dict] = []

    def fake_embed(texts, settings=None):
        assert texts == ["Employees accrue 15 days of PTO."]
        return [[0.1, 0.2, 0.3]]

    def fake_urlopen(req, timeout=60):
        body = json.loads(req.data.decode()) if req.data else None
        calls.append({"url": req.full_url, "body": body, "headers": _headers(req)})
        if req.full_url.endswith("/vectors/delete"):
            assert body == {"deleteAll": True, "namespace": "harborline"}
            return _Resp({})
        if req.full_url.endswith("/vectors/upsert"):
            assert body["namespace"] == "harborline"
            vector = body["vectors"][0]
            assert vector["id"] == "corpus:pto.md:0"
            assert vector["values"] == [0.1, 0.2, 0.3]
            assert vector["metadata"]["text"] == "Employees accrue 15 days of PTO."
            assert "values" not in vector["metadata"]
            return _Resp({"upsertedCount": 1})
        raise AssertionError(req.full_url)

    monkeypatch.setattr("harborline.store.embed_texts", fake_embed)
    monkeypatch.setattr("harborline.store.urllib.request.urlopen", fake_urlopen)
    settings = _settings(
        pinecone_api_key="pc-test",
        pinecone_index_host="harborline.svc.pinecone.io",
        pinecone_namespace="harborline",
    )
    assert persist_chunks([_chunk()], settings) == 1
    assert [call["url"] for call in calls] == [
        "https://harborline.svc.pinecone.io/vectors/delete",
        "https://harborline.svc.pinecone.io/vectors/upsert",
    ]
    assert calls[0]["headers"]["api-key"] == "pc-test"


def test_search_reads_matches_and_does_not_retain_vectors(monkeypatch):
    def fake_embed(texts, settings=None):
        assert texts == ["pto policy employee_id EMP-1008"]
        return [[0.4, 0.5]]

    def fake_urlopen(req, timeout=60):
        body = json.loads(req.data.decode()) if req.data else None
        if req.full_url.endswith("/describe_index_stats"):
            return _Resp({"namespaces": {"harborline": {"vectorCount": 2}}})
        if req.full_url.endswith("/query"):
            assert body["includeValues"] is False
            assert body["vector"] == [0.4, 0.5]
            assert body["topK"] == 8
            return _Resp(
                {
                    "matches": [
                        {
                            "id": "data:employees.json:EMP-1001",
                            "score": 0.4,
                            "metadata": {
                                "chunk_id": "data:employees.json:EMP-1001",
                                "source_path": "data/employees.json",
                                "source_name": "employees.json",
                                "source_format": "json",
                                "title": "Other",
                                "section": "Profile",
                                "kind": "structured",
                                "text": "Someone else",
                                "snippet": "Someone else",
                                "order": 1,
                                "employee_id": "EMP-1001",
                            },
                        },
                        {
                            "id": "corpus:pto.md:0",
                            "score": 0.88,
                            "metadata": {
                                "chunk_id": "corpus:pto.md:0",
                                "source_path": "corpus/01-paid-time-off.md",
                                "source_name": "01-paid-time-off.md",
                                "source_format": "md",
                                "title": "PTO",
                                "section": "Eligibility",
                                "kind": "policy",
                                "text": "Employees accrue 15 days of PTO.",
                                "snippet": "Employees accrue 15 days of PTO.",
                                "order": 0,
                            },
                        },
                    ]
                }
            )
        raise AssertionError(req.full_url)

    monkeypatch.setattr("harborline.store.embed_texts", fake_embed)
    monkeypatch.setattr("harborline.store.urllib.request.urlopen", fake_urlopen)
    settings = _settings(
        retrieve_backend="pinecone",
        pinecone_api_key="pc-test",
        pinecone_index_host="harborline.svc.pinecone.io",
        pinecone_namespace="harborline",
        top_k=2,
    )
    retriever = VectorRetriever(settings)
    hits = retriever.search("pto policy", top_k=2, employee_id="EMP-1008")
    assert [hit.chunk.chunk_id for hit in hits] == ["corpus:pto.md:0"]
    assert hits[0].score == 0.88
    assert set(vars(retriever)) == {"settings"}


def test_stdio_env_forwards_hosted_index(monkeypatch):
    monkeypatch.delenv("HARBORLINE_RETRIEVE_BACKEND", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.setenv("PINECONE_INDEX_HOST", "harborline.svc.pinecone.io")
    monkeypatch.setenv("HARBORLINE_EMBEDDING_MODEL", "text-embedding-3-small")
    from harborline.mcp_client import stdio_server_env

    env = stdio_server_env()
    assert env["HARBORLINE_RETRIEVE_BACKEND"] == "pinecone"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["PINECONE_API_KEY"] == "pc-test"
    assert env["PINECONE_INDEX_HOST"] == "harborline.svc.pinecone.io"
    assert env["HARBORLINE_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert "FASTEMBED_CACHE_PATH" not in env


def test_faiss_backend_is_rejected():
    settings = _settings(retrieve_backend="faiss")
    with pytest.raises(ValueError, match="faiss"):
        build_retriever(settings)


def test_pinecone_http_error_is_raised(monkeypatch):
    def fake_urlopen(req, timeout=60):
        raise urllib.error.HTTPError(
            req.full_url,
            400,
            "bad",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"dimension mismatch"}'),
        )

    monkeypatch.setattr("harborline.store.urllib.request.urlopen", fake_urlopen)
    settings = _settings(
        pinecone_api_key="pc-test",
        pinecone_index_host="harborline.svc.pinecone.io",
    )
    with pytest.raises(VectorStoreError, match="dimension mismatch"):
        collection_count(settings)
