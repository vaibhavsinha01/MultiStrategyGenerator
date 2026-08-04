"""PostgreSQL persistence and authorization helpers for saved strategies."""
from __future__ import annotations

import uuid
from typing import Any

import psycopg2.extras
from auth_db import _connect


def init_strategy_store() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_strategies (
            id UUID PRIMARY KEY, owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL, strategy JSONB NOT NULL, is_public BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS strategy_optimizations (
            id UUID PRIMARY KEY, strategy_id UUID REFERENCES saved_strategies(id) ON DELETE CASCADE,
            owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE, status TEXT NOT NULL,
            result JSONB, error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS strategy_generation_jobs (
            id UUID PRIMARY KEY, owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            status TEXT NOT NULL, requested_count INTEGER NOT NULL, selected_signals JSONB NOT NULL,
            result JSONB, error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ
        );
        """)


def _row(row: Any) -> dict | None:
    return dict(row) if row else None


def create_strategy(owner_id: int, name: str, strategy: dict, is_public: bool = False) -> dict:
    sid = str(uuid.uuid4())
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""INSERT INTO saved_strategies (id,owner_id,name,strategy,is_public)
                       VALUES (%s,%s,%s,%s,%s) RETURNING *""",
                    (sid, owner_id, name, psycopg2.extras.Json(strategy), is_public))
        return _row(cur.fetchone())


def list_strategies(user_id: int) -> list[dict]:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM saved_strategies WHERE owner_id=%s OR is_public=TRUE ORDER BY updated_at DESC", (user_id,))
        return [dict(row) for row in cur.fetchall()]


def get_strategy(strategy_id: str, user_id: int) -> dict | None:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM saved_strategies WHERE id=%s AND (owner_id=%s OR is_public=TRUE)", (strategy_id, user_id))
        return _row(cur.fetchone())


def update_strategy(strategy_id: str, owner_id: int, name: str, strategy: dict) -> dict | None:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""UPDATE saved_strategies SET name=%s,strategy=%s,updated_at=NOW()
                       WHERE id=%s AND owner_id=%s AND is_public=FALSE RETURNING *""",
                    (name, psycopg2.extras.Json(strategy), strategy_id, owner_id))
        return _row(cur.fetchone())


def delete_strategy(strategy_id: str, owner_id: int) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM saved_strategies WHERE id=%s AND owner_id=%s AND is_public=FALSE", (strategy_id, owner_id))
        return cur.rowcount == 1


def create_optimization(strategy_id: str, owner_id: int) -> str:
    oid = str(uuid.uuid4())
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO strategy_optimizations (id,strategy_id,owner_id,status) VALUES (%s,%s,%s,'running')", (oid, strategy_id, owner_id))
    return oid


def finish_optimization(optimization_id: str, result: dict | None = None, error: str | None = None) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE strategy_optimizations SET status=%s,result=%s,error=%s,completed_at=NOW() WHERE id=%s",
                    ("failed" if error else "completed", psycopg2.extras.Json(result) if result else None, error, optimization_id))


def get_optimization(optimization_id: str, user_id: int) -> dict | None:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM strategy_optimizations WHERE id=%s AND owner_id=%s", (optimization_id, user_id))
        return _row(cur.fetchone())


def create_generation_job(owner_id: int, requested_count: int, selected_signals: list[str]) -> str:
    job_id = str(uuid.uuid4())
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO strategy_generation_jobs (id,owner_id,status,requested_count,selected_signals)
                       VALUES (%s,%s,'running',%s,%s)""",
                    (job_id, owner_id, requested_count, psycopg2.extras.Json(selected_signals)))
    return job_id


def finish_generation_job(job_id: str, result: dict | None = None, error: str | None = None) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE strategy_generation_jobs SET status=%s,result=%s,error=%s,completed_at=NOW() WHERE id=%s",
                    ("failed" if error else "completed", psycopg2.extras.Json(result) if result else None, error, job_id))


def get_generation_job(job_id: str, owner_id: int) -> dict | None:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM strategy_generation_jobs WHERE id=%s AND owner_id=%s", (job_id, owner_id))
        return _row(cur.fetchone())
