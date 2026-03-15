# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI dashboard for the cuoptopt-agent.

Endpoints
---------
GET  /api/branches          list GitHub branches + stored metrics
GET  /api/history           chat history for the caller's session
WS   /ws/{session_id}       real-time agent interaction channel
GET  /                      serves the SPA (index.html)

WebSocket message protocol
--------------------------
Client → Server:
  {"type": "query",   "query": "...", "model": "claude"}
  {"type": "approve", "value": true|false}

Server → Client:
  {"type": "log",              "text": "..."}
  {"type": "section",          "title": "..."}
  {"type": "approval_request", "prompt": "..."}
  {"type": "metrics",          "data": {...}}   (RegressionReport JSON)
  {"type": "pr_url",           "url": "..."}
  {"type": "done",             "success": true|false}
  {"type": "error",            "text": "..."}
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------

_AGENT_ROOT = Path(__file__).parent.parent          # cuoptopt-agent/
_REPO_ROOT = _AGENT_ROOT.parent                     # cuopt-opt/
_STATIC = Path(__file__).parent / "static"
_CONFIG_DIR = _AGENT_ROOT / "config"

# ---------------------------------------------------------------------------
# Lazy imports that are only available after the agent package is installed
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _import_orchestrator():  # type: ignore[return]
    from cuoptopt_agent import orchestrator  # noqa: PLC0415
    return orchestrator


def _import_db():  # type: ignore[return]
    from dashboard import db  # noqa: PLC0415
    return db


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    db = _import_db()
    await db.init_db()
    yield


app = FastAPI(title="cuoptopt-agent dashboard", lifespan=_lifespan)

# Serve the SPA static files at /static (JS/CSS/assets); index.html is served
# manually so we can inject the session cookie before the SPA loads.
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _resolve_session(request: Request, sid: str | None) -> str:
    db = _import_db()
    ip = _client_ip(request)
    return await db.get_or_create_session(ip, sid)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request, sid: str | None = Cookie(default=None)):
    session_id = await _resolve_session(request, sid)
    response = FileResponse(str(_STATIC / "index.html"))
    response.set_cookie("sid", session_id, samesite="lax", httponly=True)
    return response


@app.get("/api/history")
async def history(
    request: Request,
    sid: str | None = Cookie(default=None),
    limit: int = 200,
):
    db = _import_db()
    session_id = await _resolve_session(request, sid)
    messages = await db.get_history(session_id, limit=limit)
    return JSONResponse({"session_id": session_id, "messages": messages})


@app.get("/api/branches")
async def branches():
    """Return GitHub branches and their stored benchmark metrics."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return JSONResponse({"error": "GITHUB_TOKEN not set"}, status_code=503)

    try:
        from github import Github, UnknownObjectException  # type: ignore[import]

        gh = Github(token)
        # Infer slug from the repo's origin remote
        from cuoptopt_agent.git_utils import _get_repo_slug  # type: ignore[import]
        owner, repo_name = _get_repo_slug(_REPO_ROOT)
        repo = gh.get_repo(f"{owner}/{repo_name}")

        result: list[dict[str, Any]] = []
        for branch in repo.get_branches():
            entry: dict[str, Any] = {
                "name": branch.name,
                "sha": branch.commit.sha[:7],
                "date": branch.commit.commit.author.date.isoformat(),
                "metrics": None,
                "pr_url": None,
            }

            # Fetch stored metrics JSON if it exists on this branch
            metrics_path = f"results/benchmarks/{branch.name}.json"
            try:
                content = repo.get_contents(metrics_path, ref=branch.name)
                entry["metrics"] = json.loads(content.decoded_content)
            except UnknownObjectException:
                pass
            except Exception as exc:
                logger.debug("Could not read metrics for %s: %s", branch.name, exc)

            # Check for open PR targeting main/master
            try:
                pulls = repo.get_pulls(state="open", head=f"{owner}:{branch.name}")
                for pr in pulls:
                    entry["pr_url"] = pr.html_url
                    break
            except Exception:
                pass

            result.append(entry)

        return JSONResponse(result)

    except Exception as exc:
        logger.exception("Failed to fetch branches")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket — real-time agent interaction
# ---------------------------------------------------------------------------

class _AgentSession:
    """Bridges a WebSocket connection to a blocking orchestrator thread."""

    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        # Messages sent FROM the client to unblock approval gates
        self._approval_queue: queue.Queue[bool] = queue.Queue()
        # Outbound messages buffered by the orchestrator thread
        self._out_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._loop = asyncio.get_event_loop()

    # -- Thread-safe helpers called from the orchestrator thread ------------

    def send_nowait(self, payload: dict[str, Any]) -> None:
        self._out_queue.put_nowait(payload)

    def approval_fn(self, prompt: str) -> bool:
        """Block the orchestrator thread until the user responds."""
        self.send_nowait({"type": "approval_request", "prompt": prompt})
        return self._approval_queue.get()  # blocks until client responds

    def progress_fn(self, title: str) -> None:
        self.send_nowait({"type": "section", "title": title})

    # -- Async drain loop called from the WebSocket coroutine ---------------

    async def drain(self) -> None:
        """Forward buffered messages to the WebSocket client."""
        while True:
            try:
                payload = self._out_queue.get_nowait()
                await self.ws.send_json(payload)
            except queue.Empty:
                await asyncio.sleep(0.05)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    db = _import_db()
    orchestrator = _import_orchestrator()

    agent: _AgentSession | None = None
    drain_task: asyncio.Task | None = None

    async def _start_run(query: str, model: str) -> None:
        nonlocal agent, drain_task
        agent = _AgentSession(websocket, session_id)
        drain_task = asyncio.create_task(agent.drain())

        await db.append_message(session_id, "user", query)

        # Capture orchestrator stdout/stderr as "log" frames
        captured_log: list[str] = []

        class _LogSink(io.TextIOBase):
            def write(self_inner, s: str) -> int:  # type: ignore[override]
                if s.strip():
                    agent.send_nowait({"type": "log", "text": s.rstrip()})  # type: ignore[union-attr]
                    captured_log.append(s)
                return len(s)

        def _run_thread() -> None:
            try:
                import sys  # noqa: PLC0415
                old_stdout = sys.stdout
                sys.stdout = _LogSink()  # type: ignore[assignment]
                try:
                    orchestrator.run(
                        query=query,
                        model_key=model,
                        config_dir=_CONFIG_DIR,
                        repo_root=_REPO_ROOT,
                        approval_fn=agent.approval_fn,  # type: ignore[union-attr]
                        progress_fn=agent.progress_fn,  # type: ignore[union-attr]
                    )
                finally:
                    sys.stdout = old_stdout
                agent.send_nowait({"type": "done", "success": True})  # type: ignore[union-attr]
            except SystemExit:
                agent.send_nowait({"type": "done", "success": False})  # type: ignore[union-attr]
            except Exception:
                tb = traceback.format_exc()
                agent.send_nowait({"type": "error", "text": tb})  # type: ignore[union-attr]
                agent.send_nowait({"type": "done", "success": False})  # type: ignore[union-attr]
            finally:
                # Signal that the drain loop should flush and stop after this batch
                agent.send_nowait({"type": "_eof"})  # type: ignore[union-attr]

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_thread)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "query":
                query = data.get("query", "").strip()
                model = data.get("model", "claude")
                if not query:
                    await websocket.send_json({"type": "error", "text": "Empty query."})
                    continue
                await _start_run(query, model)

            elif msg_type == "approve":
                if agent is not None:
                    agent._approval_queue.put(bool(data.get("value", False)))

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        if drain_task:
            drain_task.cancel()


# ---------------------------------------------------------------------------
# Entry point (used by the ``cuoptopt-dashboard`` script)
# ---------------------------------------------------------------------------

def start() -> None:
    import uvicorn  # noqa: PLC0415
    uvicorn.run(
        "dashboard.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("DASHBOARD_PORT", "8080")),
        proxy_headers=True,
        forwarded_allow_ips="*",
        reload=False,
    )


if __name__ == "__main__":
    start()
