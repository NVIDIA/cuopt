# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite persistence layer for the agent dashboard.

Sessions are identified by a SHA-256 hash of the client IP (from
X-Forwarded-For when behind a reverse proxy such as Brev) combined with a
UUID cookie that survives IP changes (e.g. mobile networks).

The cookie identity takes precedence when an existing session exists; a new
session always receives a fresh UUID which is set as the ``sid`` cookie.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

# Path to the SQLite database file.  Falls back to a local path if the env
# variable is not set.
_DB_PATH = Path(os.environ.get("DASHBOARD_DB", "/tmp/cuoptopt_dashboard.db"))

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    ip_hash     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL,   -- 'user' | 'agent' | 'system'
    content     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
"""


async def init_db() -> None:
    """Create tables if they do not exist."""
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.executescript(_DDL)
        await db.commit()


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


async def get_or_create_session(ip: str, cookie_sid: str | None) -> str:
    """Return a session ID for this visitor, creating one if needed.

    Priority:
    1. If *cookie_sid* is provided and an existing session has that ID, reuse it
       (even if the IP changed — handles mobile users on different networks).
    2. Otherwise look for a recent session matching the hashed IP.
    3. Fall back to creating a new session with a fresh UUID.
    """
    ip_hash = _hash_ip(ip)

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Option 1: cookie match
        if cookie_sid:
            async with db.execute(
                "SELECT id FROM sessions WHERE id = ?", (cookie_sid,)
            ) as cur:
                row = await cur.fetchone()
            if row:
                await db.execute(
                    "UPDATE sessions SET last_seen = datetime('now'), ip_hash = ? WHERE id = ?",
                    (ip_hash, cookie_sid),
                )
                await db.commit()
                return cookie_sid

        # Option 2: IP match — pick the most-recently-seen session
        async with db.execute(
            "SELECT id FROM sessions WHERE ip_hash = ? ORDER BY last_seen DESC LIMIT 1",
            (ip_hash,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            sid = row["id"]
            await db.execute(
                "UPDATE sessions SET last_seen = datetime('now') WHERE id = ?", (sid,)
            )
            await db.commit()
            return sid

        # Option 3: create new
        sid = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO sessions (id, ip_hash) VALUES (?, ?)", (sid, ip_hash)
        )
        await db.commit()
        return sid


async def append_message(session_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        await db.commit()


async def get_history(session_id: str, limit: int = 200) -> list[dict[str, Any]]:
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT role, content, created_at
               FROM messages
               WHERE session_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (session_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    # Return in chronological order
    return [dict(r) for r in reversed(rows)]
