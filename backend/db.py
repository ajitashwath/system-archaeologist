from __future__ import annotations

import json
import os
import logging
import uuid
from datetime import datetime
from typing import Any

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/system_archaeologist",
)

_pool: asyncpg.Pool | None = None




CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    system_name  TEXT        NOT NULL,
    evidence     JSONB       NOT NULL DEFAULT '[]',
    architecture TEXT        NOT NULL DEFAULT '',
    contradictions JSONB     NOT NULL DEFAULT '[]',
    final_design TEXT        NOT NULL DEFAULT '',
    mermaid_code TEXT        NOT NULL DEFAULT '',
    debate       JSONB       NOT NULL DEFAULT '{}',
    confidence   INTEGER     NOT NULL DEFAULT 0,
    alternative_hypotheses JSONB NOT NULL DEFAULT '[]',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_system_name ON analyses (system_name);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at  ON analyses (created_at DESC);
"""


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register JSON/JSONB codecs so asyncpg auto-encodes dicts/lists."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=60,
        init=_init_connection,
    )
    async with _pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
    logger.info("PostgreSQL pool initialized and schema ready.")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed.")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def save_analysis(
    system_name: str,
    evidence: list[str],
    architecture: str,
    contradictions: list[str],
    final_design: str,
    mermaid_code: str,
    debate: dict[str, Any],
    confidence: int,
    alternative_hypotheses: list[str],
) -> str:
    pool = get_pool()
    record_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analyses (
                id, system_name, evidence, architecture, contradictions,
                final_design, mermaid_code, debate, confidence, alternative_hypotheses
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            record_id,
            system_name,
            evidence,               # JSONB codec handles serialisation
            architecture,
            contradictions,
            final_design,
            mermaid_code,
            debate,
            confidence,
            alternative_hypotheses,
        )

    logger.info("Saved analysis '%s' with id=%s", system_name, record_id)
    return record_id


async def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM analyses WHERE id = $1",
            analysis_id,
        )

    if row is None:
        return None

    return row_to_dict(row)


async def list_analyses(limit: int = 20, deduplicate: bool = True) -> list[dict[str, Any]]:
    """Return recent analyses.

    When ``deduplicate=True`` (default) returns the **latest** run per unique
    system_name so the history UI doesn't fill up with repeated entries.
    Pass ``deduplicate=False`` to get every row.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        if deduplicate:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (system_name)
                    id, system_name, confidence, created_at
                FROM analyses
                ORDER BY system_name, created_at DESC
                LIMIT $1
                """,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, system_name, confidence, created_at
                FROM analyses
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

    return [
        {
            "id": str(row["id"]),
            "system_name": row["system_name"],
            "confidence": row["confidence"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


async def find_recent_analysis(
    system_name: str,
    within_hours: int = 1,
) -> dict[str, Any] | None:
    """Return the most recent full analysis for ``system_name`` if it was
    created within ``within_hours`` hours, otherwise ``None``.

    Useful to skip re-analysis when a fresh result already exists.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM analyses
            WHERE system_name ILIKE $1
              AND created_at >= NOW() - ($2 || ' hours')::INTERVAL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            system_name,
            str(within_hours),
        )
    return row_to_dict(row) if row else None


async def search_analyses(query: str, limit: int = 10) -> list[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, system_name, confidence, created_at
            FROM analyses
            WHERE system_name ILIKE $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            f"%{query}%",
            limit,
        )

    return [
        {
            "id": str(row["id"]),
            "system_name": row["system_name"],
            "confidence": row["confidence"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


def row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    d = dict(row)
    # asyncpg returns JSONB columns as native Python objects (list/dict).
    # If for any reason they come back as strings (e.g. stored as TEXT), parse them.
    for field in ("evidence", "contradictions", "debate", "alternative_hypotheses"):
        if isinstance(d.get(field), str):
            d[field] = json.loads(d[field])
    if "id" in d:
        d["id"] = str(d["id"])
    if "created_at" in d and isinstance(d["created_at"], datetime):
        d["created_at"] = d["created_at"].isoformat()
    return d
