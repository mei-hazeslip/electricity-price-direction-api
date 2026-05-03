# src/database.py

import sqlite3
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "prediction_logs.db"


def init_db():
    """
    Create prediction_logs table if it does not already exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            filename TEXT,
            n_predictions INTEGER,
            mean_probability REAL,
            min_probability REAL,
            max_probability REAL
        )
        """
    )

    conn.commit()
    conn.close()


def log_prediction(
    filename,
    n_predictions,
    mean_probability,
    min_probability,
    max_probability,
):
    """
    Insert one prediction summary row into SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_logs (
            created_at,
            filename,
            n_predictions,
            mean_probability,
            min_probability,
            max_probability
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            filename,
            int(n_predictions),
            float(mean_probability),
            float(min_probability),
            float(max_probability),
        ),
    )

    conn.commit()
    conn.close()


def get_recent_logs(limit=10):
    """
    Return recent prediction logs.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            created_at,
            filename,
            n_predictions,
            mean_probability,
            min_probability,
            max_probability
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    logs = []
    for row in rows:
        logs.append(
            {
                "id": row[0],
                "created_at": row[1],
                "filename": row[2],
                "n_predictions": row[3],
                "mean_probability": row[4],
                "min_probability": row[5],
                "max_probability": row[6],
            }
        )

    return logs