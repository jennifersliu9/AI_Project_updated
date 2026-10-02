"""OpenAI cloud embeddings and the in-process cosine index."""

from __future__ import annotations

import sys
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
    clear_index_cache,
    collection_count,
    embed_texts,
    metadata_to_chunk,
    persist_chunks,
    query_vectors,
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


def _openai_settings(tmp_path, **kwargs):
    payload = dict(
        retrieve_backend="openai",
        embedding_model="text-embedding-3-small",
        openai_api_key="sk-test",
        cache_dir=tmp_path,
    )
    payload.update(kwargs)
    return _settings(**payload)


def test_empty_cache_has_no_vectors(tmp_path):
    clear_index_cache()
    settings = _openai_settings(tmp_path)
    assert collection_count(settings) == 0


def test_persist_caches_normalized_vectors_and_reloads(tmp_path, monkeypatch):
    clear_index_cache()

    def fake_embed(texts, settings=None):
        assert texts == ["Employees accrue 15 days of PTO."]
        return [[0.0, 3.0]]

    monkeypatch.setattr("harborline.store.embed_texts", fake_embed)
    settings = _openai_settings(tmp_path)
    assert persist_chunks([_chunk()], settings) == 1
    assert collection_count(settings) == 1
    clear_index_cache()
    assert collection_count(settings) == 1
    matches = query_vectors([0.0, 1.0], 1, settings)
    assert matches[0][1]["chunk_id"] == "corpus:pto.md:0"
    assert matches[0][0] == pytest.approx(1.0)


def test_search_ranks_by_cosine_and_filters_employee(tmp_path, monkeypatch):
    clear_index_cache()

    def fake_embed(texts, settings=None):
        vectors = []
        for text in texts:
            if "Someone else" in text:
                vectors.append([1.0, 0.0])
            elif "accrue" in text:
                vectors.append([0.0, 1.0])
            else:
                assert "EMP-1008" in text
                vectors.append([1.0, 0.0])
        return vectors

    monkeypatch.setattr("harborline.store.embed_texts", fake_embed)
    settings = _openai_settings(tmp_path, top_k=2)
    other = _chunk(
        1,
        chunk_id="data:employees.json:EMP-1001",
        source_path="data/employees.json",
        source_name="employees.json",
        source_format="json",
        title="Other",
        section="Profile",
        kind="structured",
        text="Someone else",
        snippet="Someone else",
        extra={"employee_id": "EMP-1001"},
    )
    assert persist_chunks([other, _chunk()], settings) == 2
    retriever = VectorRetriever(settings)
    hits = retriever.search("pto policy", top_k=2, employee_id="EMP-1008")
    assert [hit.chunk.chunk_id for hit in hits] == ["corpus:pto.md:0"]
    assert hits[0].score == pytest.approx(0.0)
    assert set(vars(retriever)) == {"settings"}


def test_query_dimension_mismatch(tmp_path, monkeypatch):
    clear_index_cache()
    monkeypatch.setattr("harborline.store.embed_texts", lambda texts, settings=None: [[1.0, 0.0]])
    settings = _openai_settings(tmp_path)
    persist_chunks([_chunk()], settings)
    with pytest.raises(VectorStoreError, match="dimension"):
        query_vectors([1.0, 0.0, 0.0], 1, settings)


def test_stale_embedding_model_cache_is_ignored(tmp_path, monkeypatch):
    clear_index_cache()
    monkeypatch.setattr("harborline.store.embed_texts", lambda texts, settings=None: [[1.0, 0.0]])
    settings = _openai_settings(tmp_path)
    persist_chunks([_chunk()], settings)
    other = _openai_settings(tmp_path, embedding_model="text-embedding-3-large")
    assert collection_count(other) == 0


def test_stdio_env_forwards_openai(monkeypatch):
    monkeypatch.delenv("HARBORLINE_RETRIEVE_BACKEND", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HARBORLINE_EMBEDDING_MODEL", "text-embedding-3-small")
    from harborline.mcp_client import stdio_server_env

    env = stdio_server_env()
    assert env["HARBORLINE_RETRIEVE_BACKEND"] == "openai"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["HARBORLINE_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert "PINECONE_API_KEY" not in env
    assert "FASTEMBED_CACHE_PATH" not in env


def test_removed_backends_are_rejected():
    for name in ("faiss", "pinecone"):
        with pytest.raises(ValueError, match=name):
            build_retriever(_settings(retrieve_backend=name))
