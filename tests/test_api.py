from fastapi.testclient import TestClient

from harborline.api import app

client = TestClient(app)


def test_home_serves_chat_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "Harborline People Desk" in res.text
    assert 'id="demo-list"' in res.text
    assert 'fetch("/demos")' in res.text
    assert "/demos/" in res.text


def test_health_includes_mcp():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["app"] == "ok"
    assert body["answer_mode"] == "retrieve"
    assert body["retrieve_backend"] == "tfidf"
    assert body["vector_index"] == "local:tfidf"
    assert body["embedding_provider"] == "openai"
    assert body["embedding_class"] == "OpenAIEmbeddings"
    assert body["embedding_model"] == "text-embedding-3-small"
    assert "retrieve" in body["answer_mode_detail"]
    assert "mcp" in body
    assert body["mcp"]["available"] is True
    assert "search_policy_documents" in body["mcp"]["discovered_tools"]


def test_demos_list():
    res = client.get("/demos")
    assert res.status_code == 200
    demos = res.json()["demos"]
    ids = [d["id"] for d in demos]
    assert ids == ["remote-emp-1008", "pto-emp-1014"]
    assert len({d["query"] for d in demos}) == 2
    for demo in demos:
        assert demo["query"]
        assert demo["employee_id"]
        assert demo["curl"].endswith(f"/demos/{demo['id']}")
        assert demo["chat"] == {"query": demo["query"], "employee_id": demo["employee_id"]}


def _assert_stable_demo(demo: dict) -> dict:
    first = client.post(f"/demos/{demo['id']}")
    second = client.post(f"/demos/{demo['id']}")
    chat = client.post("/chat", json=demo["chat"])
    assert first.status_code == second.status_code == chat.status_code == 200
    a, b, c = first.json(), second.json(), chat.json()
    assert a["demo_id"] == demo["id"]
    assert a["answer"] == b["answer"] == c["answer"]
    assert a["answer"]
    assert [step["tool"] for step in a["trace"]] == [step["tool"] for step in b["trace"]]
    assert [step["tool"] for step in a["trace"]] == [step["tool"] for step in c["trace"]]
    assert all(step["ok"] for step in a["trace"])
    paths = [item["source_path"] for item in a["citations"]]
    assert paths
    assert paths == [item["source_path"] for item in b["citations"]]
    assert paths == [item["source_path"] for item in c["citations"]]
    assert all(demo["citation_source"] in path for path in paths)
    assert a["snippets"]
    return a


def test_remote_and_pto_demos_match_chat():
    demos = client.get("/demos").json()["demos"]
    remote, pto = demos
    remote_body = _assert_stable_demo(remote)
    pto_body = _assert_stable_demo(pto)
    assert remote_body["intent"] == "remote_eligibility"
    assert "Alex Kim" in remote_body["answer"]
    assert "hub" in remote_body["answer"].lower()
    remote_tools = [step["tool"] for step in remote_body["trace"]]
    assert "lookup_employee_profile" in remote_tools
    assert "search_policy_documents" in remote_tools
    assert "get_policy_section" in remote_tools
    assert "check_policy_compliance" in remote_tools
    assert pto_body["intent"] == "pto_guidance"
    assert "Devon Walsh" in pto_body["answer"]
    assert "5.0" in pto_body["answer"]
    assert "2026-10-08" in pto_body["answer"]
    pto_tools = [step["tool"] for step in pto_body["trace"]]
    assert pto_tools == [
        "lookup_employee_profile",
        "check_pto_balance",
        "get_policy_section",
    ]
    assert remote_body["answer"] != pto_body["answer"]
    assert [step["tool"] for step in remote_body["trace"]] != pto_tools


def test_unknown_demo_is_404():
    res = client.post("/demos/benefits-emp-1008")
    assert res.status_code == 404
