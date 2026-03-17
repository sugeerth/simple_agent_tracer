"""Simple token-based auth with SQLite user storage. No external dependencies."""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any


class AuthStore:
    """SQLite-backed user auth with token sessions."""

    def __init__(self, db_path: str = "omniscope.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """)
        conn.commit()

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), iterations=100_000
        ).hex()

    def signup(self, email: str, username: str, password: str) -> dict[str, Any]:
        conn = self._get_conn()

        # Validate
        if not email or "@" not in email:
            return {"error": "Invalid email"}
        if not username or len(username) < 2:
            return {"error": "Username must be at least 2 characters"}
        if not password or len(password) < 6:
            return {"error": "Password must be at least 6 characters"}

        # Check uniqueness
        existing = conn.execute(
            "SELECT user_id FROM users WHERE email = ? OR username = ?",
            (email.lower(), username.lower())
        ).fetchone()
        if existing:
            return {"error": "Email or username already exists"}

        user_id = secrets.token_hex(16)
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        now = datetime.utcnow().isoformat() + "Z"

        conn.execute(
            "INSERT INTO users (user_id, email, username, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, email.lower(), username.lower(), password_hash, salt, now)
        )
        conn.commit()

        # Auto-login
        token = self._create_session(user_id)
        return {"token": token, "user_id": user_id, "username": username, "email": email}

    def login(self, username_or_email: str, password: str) -> dict[str, Any]:
        conn = self._get_conn()
        key = username_or_email.lower()

        user = conn.execute(
            "SELECT * FROM users WHERE email = ? OR username = ?",
            (key, key)
        ).fetchone()

        if not user:
            return {"error": "Invalid credentials"}

        password_hash = self._hash_password(password, user["salt"])
        if password_hash != user["password_hash"]:
            return {"error": "Invalid credentials"}

        # Update last_login
        now = datetime.utcnow().isoformat() + "Z"
        conn.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (now, user["user_id"]))
        conn.commit()

        token = self._create_session(user["user_id"])
        return {
            "token": token,
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
        }

    def validate_token(self, token: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        session = conn.execute(
            "SELECT s.*, u.username, u.email FROM sessions s JOIN users u ON s.user_id = u.user_id WHERE s.token = ? AND s.expires_at > ?",
            (token, now)
        ).fetchone()

        if not session:
            return None

        return {
            "user_id": session["user_id"],
            "username": session["username"],
            "email": session["email"],
        }

    def logout(self, token: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()

    def _create_session(self, user_id: str, days: int = 30) -> str:
        conn = self._get_conn()
        token = secrets.token_urlsafe(48)
        now = datetime.utcnow()
        expires = now + timedelta(days=days)

        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat() + "Z", expires.isoformat() + "Z")
        )
        conn.commit()
        return token

    def _cleanup_expired(self):
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        conn.commit()
