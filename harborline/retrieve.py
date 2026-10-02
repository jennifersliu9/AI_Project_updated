"""Retrieval: hosted Pinecone by default, TF-IDF as an offline fallback."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from harborline.config import Settings, get_settings
from harborline.ingest import Chunk, load_chunks
from harborline.store import collection_count, metadata_to_chunk, persist_chunks


@dataclass(frozen=True)
class Hit:
    score: float
    chunk: Chunk


def _keep_hit(
    chunk: Chunk,
    employee_id: str | None,
    kind: str | None = None,
    source_format: str | None = None,
) -> bool:
    if kind and chunk.kind != kind:
        return False
    if source_format and chunk.source_format != source_format:
        return False
    if not employee_id:
        return True
    extra_id = chunk.extra.get("employee_id") or ""
    if chunk.kind == "structured" and extra_id and extra_id != employee_id:
        return False
    return True


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk], settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        texts = [c.text for c in chunks]
        if not texts:
            raise ValueError("No chunks to index. Check corpus/ and data/.")
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        employee_id: str | None = None,
        kind: str | None = None,
        source_format: str | None = None,
    ) -> list[Hit]:
        k = top_k or self.settings.top_k
        q = query.strip()
        if employee_id:
            q = f"{q} employee_id {employee_id}"
        vec = self.vectorizer.transform([q])
        scores = cosine_similarity(vec, self.matrix).ravel()
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )
        hits: list[Hit] = []
        for i in ranked:
            chunk = self.chunks[i]
            if not _keep_hit(chunk, employee_id, kind, source_format):
                continue
            hits.append(Hit(score=float(scores[i]), chunk=chunk))
            if len(hits) >= k:
                break
        return hits


class VectorRetriever:
    """Search Pinecone. This object does not hold the corpus index."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if collection_count(self.settings) == 0:
            persist_chunks(load_chunks(self.settings), self.settings)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        employee_id: str | None = None,
        kind: str | None = None,
        source_format: str | None = None,
    ) -> list[Hit]:
        from harborline.store import embed_texts, query_vectors

        k = top_k or self.settings.top_k
        q = query.strip()
        if employee_id:
            q = f"{q} employee_id {employee_id}"
        qvec = embed_texts([q], self.settings)
        fetch = max(k * 4, k)
        hits: list[Hit] = []
        for score, meta in query_vectors(qvec[0], fetch, self.settings):
            chunk = metadata_to_chunk(meta)
            if not _keep_hit(chunk, employee_id, kind, source_format):
                continue
            hits.append(Hit(score=float(score), chunk=chunk))
            if len(hits) >= k:
                break
        return hits


def build_retriever(settings: Settings | None = None):
    settings = settings or get_settings()
    backend = settings.retrieve_backend
    if backend == "tfidf":
        return TfidfRetriever(load_chunks(settings), settings)
    if backend == "pinecone":
        return VectorRetriever(settings)
    if backend == "faiss":
        raise ValueError(
            "HARBORLINE_RETRIEVE_BACKEND=faiss has been removed. "
            "This process no longer downloads sentence-transformer weights or "
            "loads a FAISS index into memory. Use pinecone "
            "(OpenAI text-embedding-3-small + a hosted Pinecone index) or tfidf."
        )
    raise ValueError(
        f"Unknown HARBORLINE_RETRIEVE_BACKEND={backend!r}. Use pinecone or tfidf."
    )


# Back-compat alias used in older tests/docs
Retriever = TfidfRetriever
