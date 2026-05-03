"""Configuration et accès à la base PostgreSQL.

Pool de connexions partagé par l'application FastAPI.
Les paramètres viennent de variables d'environnement (cf. docker-compose.yml).
"""
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

# Pool ouvert au démarrage de l'application, fermé à l'arrêt.
pool: ConnectionPool = ConnectionPool(conninfo=_CONNINFO, min_size=1, max_size=5, open=False)


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Fournit une connexion du pool, avec commit automatique en sortie de bloc.

    En cas d'exception, la transaction est annulée par le context manager de psycopg.
    """
    with pool.connection() as conn:
        yield conn
