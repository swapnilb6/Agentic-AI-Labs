from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import CrewOutput, FintechScenario


class FeedbackStore:
    def __init__(self, db_path: str | Path = "./data/feedback.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    scenario_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    clarity INTEGER NOT NULL,
                    fairness INTEGER NOT NULL,
                    compliance INTEGER NOT NULL,
                    comments TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_feedback(
        self,
        scenario: FintechScenario,
        output: CrewOutput,
        rating: int,
        clarity: int,
        fairness: int,
        compliance: int,
        comments: str,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    created_at, scenario_json, output_json, rating, clarity,
                    fairness, compliance, comments
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    json.dumps(asdict(scenario), ensure_ascii=False),
                    json.dumps(output.as_dict(), ensure_ascii=False),
                    rating,
                    clarity,
                    fairness,
                    compliance,
                    comments,
                ),
            )
            conn.commit()

    def count_feedback(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM feedback").fetchone()
            return int(row["c"]) if row else 0

    def average_rating(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT AVG(rating) AS avg_rating FROM feedback").fetchone()
            return float(row["avg_rating"] or 0.0)

    def recent_rows(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, rating, clarity, fairness, compliance, comments, scenario_json
                FROM feedback
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def build_policy_memory(self, limit: int = 8) -> str:
        rows = self.recent_rows(limit=limit)
        if not rows:
            return (
                "No prior reviewer feedback exists yet. Keep explanations "
                "clear, fair, auditable, and aligned with U.S. adverse-action norms."
            )

        positives: list[str] = []
        issues: list[str] = []
        for row in rows:
            rating = int(row["rating"])
            comments = str(row["comments"]).strip()
            if rating >= 4 and comments:
                positives.append(f"- Strong review: {comments}")
            elif comments:
                issues.append(f"- Improve: {comments}")

        if not positives and not issues:
            return "Prior feedback exists, but no useful text comments were provided."

        parts = ["Use this reviewer memory on the next run:"]
        if positives:
            parts.append("Helpful patterns:")
            parts.extend(positives[:4])
        if issues:
            parts.append("Fix these recurring issues:")
            parts.extend(issues[:4])
        return "\n".join(parts)

    def rating_distribution(self) -> dict[int, int]:
        counts = {i: 0 for i in range(1, 6)}
        with self._connect() as conn:
            rows = conn.execute("SELECT rating, COUNT(*) AS c FROM feedback GROUP BY rating").fetchall()
        for row in rows:
            counts[int(row["rating"])] = int(row["c"])
        return counts
