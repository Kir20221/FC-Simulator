# connection.py
"""Configuration et accès à la base PostgreSQL."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

DB_CONFIG = {
    "host":     os.environ.get("DB_HOST", "db"),
    "port":     os.environ.get("DB_PORT", "5432"),
    "dbname":   os.environ.get("DB_NAME", "drake"),
    "user":     os.environ.get("DB_USER", "drake"),
    "password": os.environ.get("DB_PASSWORD", "drake"),
}

_CONNINFO = " ".join(f"{k}={v}" for k, v in DB_CONFIG.items())

# Pool ouvert automatiquement à la première utilisation.
pool: ConnectionPool = ConnectionPool(conninfo=_CONNINFO, min_size=1, max_size=5)


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Fournit une connexion du pool, commit en sortie de bloc, rollback si exception."""
    with pool.connection() as conn:
        yield conn