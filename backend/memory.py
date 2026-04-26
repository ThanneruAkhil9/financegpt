"""
memory.py — SQLite-based conversation memory for FinanceGPT
Stores chat history so the bot remembers what was said earlier in a session.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "financegpt.db")


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,       -- 'user' or 'assistant'
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    conn.close()


def get_or_create_session(session_id: str) -> str:
    """Get existing session or create a new one."""
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    if not existing:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)", (session_id, now, now)
        )
        conn.commit()

    conn.close()
    return session_id


def save_message(session_id: str, role: str, content: str):
    """Save a single message to the database."""
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, last_n: int = 10) -> list[dict]:
    """
    Retrieve the last N messages for a session.
    Returns list of {"role": "user"/"assistant", "content": "..."} dicts.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE session_id = ?
           ORDER BY id DESC LIMIT ?""",
        (session_id, last_n),
    ).fetchall()
    conn.close()

    # Reverse so oldest is first
    history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    return history


def clear_session(session_id: str):
    """Delete all messages for a session (used by 'Clear Chat' button)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def get_all_sessions() -> list[dict]:
    """Get all sessions (for admin/debug purposes)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"session_id": r[0], "created_at": r[1], "updated_at": r[2]} for r in rows]
