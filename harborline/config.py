"""Load settings from environment variables. Secrets never live in source."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

# openai: OpenAI embeddings API + in-process cosine index. tfidf: offline, no API keys.
RETRIEVE_BACKENDS = ("openai", "tfidf")
# Fixed. There is no HuggingFace or sentence-transformer embedding provider.
EMBEDDING_PROVIDER = "openai"


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_answer_mode(explicit: str | None = None, api_key: str | None = None) -> str:
    """Pick retrieve vs llm.

    An explicit HARBORLINE_ANSWER_MODE wins. When it is unset, a non-empty
    OPENAI_API_KEY selects llm; otherwise the offline retrieve path is used.
    Cursor Settings → Models does not set this variable.
    """
    if explicit is None:
        explicit = os.getenv("HARBORLINE_ANSWER_MODE")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    mode = (explicit or "").strip().lower()
    if mode:
        return mode
    if (api_key or "").strip():
        return "llm"
    return "retrieve"


@dataclass(frozen=True)
class Settings:
    seed: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    answer_mode: str
    retrieve_backend: str
    embedding_model: str
    embedding_provider: str
    min_score: float
    fetch_k: int
    rewrite_queries: bool
    rerank: bool
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str | None
    root: Path
    corpus_dir: Path
    data_dir: Path
    eval_path: Path
    cache_dir: Path

    def apply_seeds(self) -> None:
        """Fix process-wide RNGs used for evaluation sampling."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def require_llm_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env "
                "or export the variable. Do not commit real keys."
            )
        return self.openai_api_key


def describe_answer_mode(settings: Settings) -> str:
    """Short status for /health. Names why the process is not using the model."""
    explicit = (os.getenv("HARBORLINE_ANSWER_MODE") or "").strip().lower()
    if settings.answer_mode == "llm" and settings.openai_api_key:
        return "llm"
    if settings.answer_mode == "llm":
        return "llm (OPENAI_API_KEY is not set in this process)"
    if explicit == "retrieve":
        return "retrieve (HARBORLINE_ANSWER_MODE=retrieve)"
    return "retrieve (no OPENAI_API_KEY in this process)"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    return Settings(
        seed=_int("HARBORLINE_SEED", 42),
        chunk_size=_int("HARBORLINE_CHUNK_SIZE", 900),
        chunk_overlap=_int("HARBORLINE_CHUNK_OVERLAP", 120),
        top_k=_int("HARBORLINE_TOP_K", 5),
        answer_mode=resolve_answer_mode(),
        retrieve_backend=_str("HARBORLINE_RETRIEVE_BACKEND", "openai").lower(),
        embedding_model=_str("HARBORLINE_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_provider=EMBEDDING_PROVIDER,
        min_score=_float("HARBORLINE_MIN_SCORE", 0.22),
        fetch_k=_int("HARBORLINE_FETCH_K", 20),
        rewrite_queries=_bool("HARBORLINE_REWRITE", True),
        rerank=_bool("HARBORLINE_RERANK", True),
        openai_api_key=key,
        openai_model=_str("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        root=ROOT,
        corpus_dir=ROOT / "corpus",
        data_dir=ROOT / "data",
        eval_path=ROOT / "eval" / "gold_questions.json",
        cache_dir=ROOT / ".cache",
    )
