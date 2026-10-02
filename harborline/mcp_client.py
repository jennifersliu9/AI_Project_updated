"""MCP client used by the Harborline agent.

Discovery: `tools/list`. Invocation: `tools/call`. The agent never reaches into
`harborline.tools` during execution; it only speaks the MCP tool protocol.

Transports:
- mcp-stdio: spawn `python -m harborline.mcp_server` (same process Cursor uses)
- mcp-inproc: FastMCP list_tools/call_tool in this process (tests / no extra PID)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any

from harborline.config import ROOT
from harborline.mcp_server import MCP_TOOL_NAMES, create_server


def json_tool_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop Nones so FastMCP optional strings stay omitted or defaulted."""
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        out[key] = value
    return out


def _parse_tool_payload(name: str, text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"tool": name, "ok": False, "error": "Empty MCP tool result."}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"tool": name, "ok": True, "text": text}
    if isinstance(payload, dict):
        return payload
    return {"tool": name, "ok": True, "result": payload}


def _content_text(content: list) -> str:
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        elif isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts)


class McpToolBus:
    """Protocol: discover via MCP, invoke via MCP, expose .call / .close / .available."""

    transport = "mcp"
    available = False
    discovered_tools: list[str] = []

    def call(self, name: str, **kwargs: Any) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        return None


class InProcessMcpBus(McpToolBus):
    """Wrap FastMCP so tests still go through tools/list + tools/call, not raw functions."""

    transport = "mcp-inproc"

    def __init__(self) -> None:
        self._server = create_server()
        listed = asyncio.run(self._server.list_tools())
        self.discovered_tools = [t.name for t in listed]
        self.available = True
        self.tool_schemas = [
            {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in listed
        ]

    def call(self, name: str, **kwargs: Any) -> dict:
        if not self.available:
            return {"tool": name, "ok": False, "error": "MCP tools unavailable."}
        if name not in self.discovered_tools:
            return {
                "tool": name,
                "ok": False,
                "error": f"Unknown MCP tool {name}. Discovered: {self.discovered_tools}",
            }
        try:
            content = asyncio.run(self._server.call_tool(name, json_tool_args(kwargs)))
        except Exception as exc:  # noqa: BLE001 — surface in the operational trace
            return {"tool": name, "ok": False, "error": str(exc)}
        return _parse_tool_payload(name, _content_text(list(content)))


class StdioMcpBus(McpToolBus):
    """Spawn the stdio MCP server and call tools over JSON-RPC."""

    transport = "mcp-stdio"

    def __init__(self, extra_env: dict[str, str] | None = None) -> None:
        self.available = False
        self.discovered_tools = []
        self.tool_schemas: list[dict] = []
        self._extra_env = extra_env or {}
        self._error: str | None = None
        self._session = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="harborline-mcp-stdio", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=60):
            self._error = self._error or "Timed out starting MCP stdio server."
        if self._error:
            self.available = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._session_lifetime())
        self._loop.run_forever()

    async def _session_lifetime(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._closed = asyncio.Event()
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "harborline.mcp_server", "--transport", "stdio"],
            cwd=str(ROOT),
            env=stdio_server_env(self._extra_env),
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    listed = await session.list_tools()
                    self._session = session
                    self.server_name = getattr(getattr(init, "serverInfo", None), "name", "harborline")
                    self.discovered_tools = [t.name for t in listed.tools]
                    self.tool_schemas = [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema,
                        }
                        for t in listed.tools
                    ]
                    self.available = True
                    self._ready.set()
                    await self._closed.wait()
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            self.available = False
            self._ready.set()
        finally:
            self._session = None
            self._loop.stop()

    async def _invoke(self, name: str, arguments: dict[str, Any]) -> dict:
        result = await self._session.call_tool(name, arguments)
        text = _content_text(list(result.content or []))
        if getattr(result, "isError", False):
            return {"tool": name, "ok": False, "error": text or "MCP tools/call error"}
        return _parse_tool_payload(name, text)

    def call(self, name: str, **kwargs: Any) -> dict:
        if not self.available or self._session is None:
            return {
                "tool": name,
                "ok": False,
                "error": self._error or "MCP stdio tools unavailable.",
            }
        if name not in self.discovered_tools:
            return {
                "tool": name,
                "ok": False,
                "error": f"Unknown MCP tool {name}. Discovered: {self.discovered_tools}",
            }
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._invoke(name, json_tool_args(kwargs)),
                self._loop,
            )
            return future.result(timeout=90)
        except Exception as exc:  # noqa: BLE001
            return {"tool": name, "ok": False, "error": str(exc)}

    def close(self) -> None:
        closed = getattr(self, "_closed", None)
        if closed is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(closed.set)
        self._thread.join(timeout=15)


def stdio_server_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the spawned MCP server.

    Do not default HARBORLINE_ANSWER_MODE to retrieve. An explicit parent value
    is forwarded; otherwise the child reads `.env` and uses llm when
    OPENAI_API_KEY is set.
    """
    env = {
        "HARBORLINE_RETRIEVE_BACKEND": os.environ.get("HARBORLINE_RETRIEVE_BACKEND", "openai"),
        "HARBORLINE_REWRITE": os.environ.get("HARBORLINE_REWRITE", "true"),
        "HARBORLINE_RERANK": os.environ.get("HARBORLINE_RERANK", "true"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.environ.get("PYTHONPATH", str(ROOT)),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
    }
    if os.environ.get("HARBORLINE_ANSWER_MODE"):
        env["HARBORLINE_ANSWER_MODE"] = os.environ["HARBORLINE_ANSWER_MODE"]
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "HARBORLINE_EMBEDDING_MODEL",
    ):
        if os.environ.get(name):
            env[name] = os.environ[name]
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v})
    return {k: v for k, v in env.items() if v}


def open_mcp_bus(transport: str = "mcp-stdio", extra_env: dict[str, str] | None = None) -> McpToolBus:
    if extra_env:
        os.environ.update({k: v for k, v in extra_env.items() if v})
        from harborline.config import get_settings

        get_settings.cache_clear()
    if transport in {"mcp-inproc", "inproc", "local"}:
        return InProcessMcpBus()
    if transport in {"mcp-stdio", "stdio"}:
        return StdioMcpBus(extra_env=extra_env)
    raise ValueError(f"Unknown MCP transport {transport!r}. Use mcp-stdio or mcp-inproc.")


REQUIRED_MCP_TOOLS = MCP_TOOL_NAMES
