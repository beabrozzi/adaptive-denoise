
# SQLite log of denoising requests, used for monitoring

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

TRAIN_SIGMA_MAX = 0.25 # highest noise level seen during training

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    method      TEXT    NOT NULL,
    param       REAL    NOT NULL,
    sigma_hat   REAL    NOT NULL,
    height      INTEGER NOT NULL,
    width       INTEGER NOT NULL,
    runtime_ms  REAL    NOT NULL
)
"""

def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn

def log_request(path: Path, method: str, param: float, sigma_hat: float,
                shape: tuple, runtime_ms: float) -> None:
    with closing(connect(path)) as conn:
        with conn:  # commits the transaction
            conn.execute(
                "INSERT INTO requests (created_at, method, param, sigma_hat, "
                "height, width, runtime_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), method, param,
                 sigma_hat, shape[0], shape[1], runtime_ms),
            )

def get_stats(path: Path) -> dict:
    with closing(connect(path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        rows = conn.execute(
            "SELECT method, COUNT(*), AVG(runtime_ms), AVG(param) "
            "FROM requests GROUP BY method"
        ).fetchall()
        out_of_range = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE sigma_hat > ?", (TRAIN_SIGMA_MAX,)
        ).fetchone()[0]
    return {
        "total_requests": total,
        "by_method": {
            m: {"count": n, "avg_runtime_ms": round(rt, 2), "avg_param": round(p, 4)}
            for m, n, rt, p in rows
        },
        "out_of_training_range": out_of_range,
    }

def get_recent(path: Path, limit: int = 20) -> list[dict]:
    with closing(connect(path)) as conn:
        conn.row_factory = sqlite3.Row      # rows behave like dicts
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]