import json
from pathlib import Path

from harborline.agent import run_agent
from harborline.config import describe_answer_mode, get_settings, resolve_answer_mode
from harborline.mcp_client import stdio_server_env


def test_unset_mode_uses_llm_only_when_key_present(monkeypatch):
    monkeypatch.delenv("HARBORLINE_ANSWER_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_answer_mode() == "retrieve"
    assert resolve_answer_mode(explicit=None, api_key="sk-test") == "llm"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_answer_mode() == "llm"
    monkeypatch.setenv("HARBORLINE_ANSWER_MODE", "retrieve")
    assert resolve_answer_mode() == "retrieve"


def test_describe_answer_mode_names_the_pin(monkeypatch):
    monkeypatch.setenv("HARBORLINE_ANSWER_MODE", "retrieve")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        detail = describe_answer_mode(get_settings())
    finally:
        get_settings.cache_clear()
    assert detail.startswith("retrieve")
    assert "HARBORLINE_ANSWER_MODE=retrieve" in detail


def test_stdio_env_forwards_key_and_does_not_invent_retrieve(monkeypatch):
    monkeypatch.delenv("HARBORLINE_ANSWER_MODE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    env = stdio_server_env()
    assert "HARBORLINE_ANSWER_MODE" not in env
    assert env["OPENAI_API_KEY"] == "sk-test"
    monkeypatch.setenv("HARBORLINE_ANSWER_MODE", "llm")
    assert stdio_server_env()["HARBORLINE_ANSWER_MODE"] == "llm"


def test_cursor_mcp_json_does_not_pin_retrieve_mode():
    raw = json.loads(Path(".cursor/mcp.json").read_text(encoding="utf-8"))
    server = raw["mcpServers"]["harborline"]
    assert "HARBORLINE_ANSWER_MODE" not in server["env"]
    assert server["envFile"].endswith(".env")


def test_openai_key_is_recognized_and_generates_an_answer(monkeypatch):
    monkeypatch.delenv("HARBORLINE_ANSWER_MODE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()

    class _Message:
        content = "After the second anniversary, PTO is 20 days."

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-4o-mini"
            assert kwargs["temperature"] == 0
            assert kwargs["messages"][0]["role"] == "system"
            joined = kwargs["messages"][1]["content"]
            assert "PTO" in joined
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _OpenAI:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "sk-test"
            self.chat = _Chat()

    monkeypatch.setattr("openai.OpenAI", _OpenAI)
    from harborline.answer import ask
    from harborline.ingest import load_chunks
    from harborline.retrieve import TfidfRetriever

    try:
        settings = get_settings()
        assert settings.openai_api_key == "sk-test"
        assert settings.answer_mode == "llm"
        retriever = TfidfRetriever(load_chunks(settings), settings)
        result = ask(
            "How many PTO days do I get after my second anniversary?",
            settings=settings,
            retriever=retriever,
        )
    finally:
        get_settings.cache_clear()
    assert result["mode"] == "llm"
    assert result["guardrail_ok"] is True
    assert result["answer"] == "After the second anniversary, PTO is 20 days."
    assert result["sources"]


def test_llm_mode_rewrites_agent_answer(monkeypatch):
    monkeypatch.setenv("HARBORLINE_ANSWER_MODE", "llm")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()

    def fake(query, draft, sources, tool_notes, settings, employee_id):
        assert query
        assert "Alex Kim" in draft
        assert sources
        assert tool_notes
        assert settings.openai_api_key == "sk-test"
        assert employee_id == "EMP-1008"
        return "MODEL " + draft

    monkeypatch.setattr("harborline.agent.llm_rewrite_answer", fake)
    try:
        result = run_agent(
            "Am I eligible for fully remote work living in Tacoma?",
            employee_id="EMP-1008",
            transport="mcp-inproc",
        )
    finally:
        get_settings.cache_clear()
    assert result.answer_mode == "llm"
    assert result.answer.startswith("MODEL ")
    assert "Alex Kim" in result.answer
