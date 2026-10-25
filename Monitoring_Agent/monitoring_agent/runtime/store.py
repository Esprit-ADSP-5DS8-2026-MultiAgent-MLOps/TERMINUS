"""
Mémoire de la gateway : chaque prédiction servie, les labels reçus plus tard, les fenêtres
analysées et les alertes reçues d'Alertmanager.

SQLite : un seul fichier, zéro service à lancer, transactions sûres entre threads.
La gateway écrit, le moniteur lit ; un verrou suffit à ce volume (quelques centaines de req/s).

Chaque prédiction RÉUSSIE reçoit un numéro de séquence `seq` (1, 2, 3...). La fenêtre k
regroupe les seq de k*W+1 à (k+1)*W : les fenêtres sont stables même si des labels arrivent après.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id  TEXT PRIMARY KEY,
    seq         INTEGER UNIQUE,          -- NULL si la requête a échoué
    ts          REAL NOT NULL,
    features    TEXT NOT NULL,           -- JSON envoyé par le client
    prediction  REAL,
    proba       REAL,
    latency_ms  REAL,
    status      INTEGER NOT NULL,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    request_id  TEXT PRIMARY KEY,
    label       REAL NOT NULL,
    ts          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS windows (
    idx               INTEGER PRIMARY KEY,
    seq_start         INTEGER NOT NULL,
    seq_end           INTEGER NOT NULL,
    analyzed_at       REAL NOT NULL,
    labels_available  INTEGER NOT NULL,
    severity          TEXT NOT NULL,
    trigger_agent     INTEGER NOT NULL,
    report            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at  REAL NOT NULL,
    status       TEXT NOT NULL,
    alertname    TEXT NOT NULL,
    severity     TEXT,
    payload      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id            TEXT PRIMARY KEY,
    window_idx    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    action        TEXT,
    mode          TEXT,
    status        TEXT NOT NULL,
    signals       TEXT NOT NULL,          -- signaux de la fenêtre (anti-rafale)
    payload       TEXT NOT NULL           -- AgentRun complet (JSON)
);
CREATE INDEX IF NOT EXISTS idx_predictions_seq ON predictions(seq);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._seq = self._db.execute("SELECT COALESCE(MAX(seq), 0) FROM predictions").fetchone()[0]

    def close(self) -> None:
        self._db.close()

    # -- écriture -----------------------------------------------------------------

    def add_prediction(self, request_id: str, features: dict, prediction: float | None, proba: float | None,
                       latency_ms: float, status: int, error: str | None = None, ts: float | None = None) -> int | None:
        ok = 200 <= status < 300 and prediction is not None
        with self._lock:
            seq = self._seq + 1 if ok else None
            self._db.execute(
                "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                (request_id, seq, ts or time.time(), json.dumps(features, default=str),
                 prediction, proba, latency_ms, status, error),
            )
            if ok:
                self._seq = seq
        return seq

    def add_labels(self, items: list[tuple[str, float]]) -> int:
        """Enregistre des labels ; renvoie combien correspondent à une prédiction connue."""
        now = time.time()
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                                 [(rid, float(lbl), now) for rid, lbl in items])
            ids = [rid for rid, _ in items]
            marks = ",".join("?" * len(ids))
            return self._db.execute(f"SELECT COUNT(*) FROM predictions WHERE request_id IN ({marks})", ids).fetchone()[0]

    def save_window(self, idx: int, seq_start: int, seq_end: int, labels_available: bool,
                    severity: str, trigger: bool, report_json: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO windows VALUES (?,?,?,?,?,?,?,?)",
                             (idx, seq_start, seq_end, time.time(), int(labels_available), severity,
                              int(trigger), report_json))

    def add_alert(self, status: str, alertname: str, severity: str | None, payload: dict) -> None:
        with self._lock:
            self._db.execute("INSERT INTO alerts (received_at, status, alertname, severity, payload) VALUES (?,?,?,?,?)",
                             (time.time(), status, alertname, severity, json.dumps(payload)))

    # -- lecture ------------------------------------------------------------------

    @property
    def n_ok(self) -> int:
        return self._seq

    def last_window_idx(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT MAX(idx) FROM windows").fetchone()
        return -1 if row[0] is None else row[0]

    def windows_without_labels(self, limit: int = 20) -> list[tuple[int, int, int]]:
        with self._lock:
            return self._db.execute(
                "SELECT idx, seq_start, seq_end FROM windows WHERE labels_available = 0 ORDER BY idx DESC LIMIT ?",
                (limit,)).fetchall()

    def labelled_share(self, seq_start: int, seq_end: int) -> float:
        with self._lock:
            n, k = self._db.execute(
                "SELECT COUNT(*), COUNT(l.label) FROM predictions p LEFT JOIN labels l USING(request_id) "
                "WHERE p.seq BETWEEN ? AND ?", (seq_start, seq_end)).fetchone()
        return k / n if n else 0.0

    def window_frame(self, seq_start: int, seq_end: int, target: str | None) -> pd.DataFrame:
        """Les prédictions d'une fenêtre, à plat : features + y_pred + y_pred_proba (+ label)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT p.request_id, p.ts, p.features, p.prediction, p.proba, l.label "
                "FROM predictions p LEFT JOIN labels l USING(request_id) "
                "WHERE p.seq BETWEEN ? AND ? ORDER BY p.seq", (seq_start, seq_end)).fetchall()
        records = []
        for rid, ts, feats, pred, proba, label in rows:
            rec = json.loads(feats)
            rec.update(request_id=rid, timestamp=pd.Timestamp(ts, unit="s").isoformat(timespec="seconds"),
                       y_pred=pred, y_pred_proba=proba)
            if target:
                rec[target] = label
            records.append(rec)
        return pd.DataFrame.from_records(records)

    def window_signal_kinds(self, idx: int) -> set[str]:
        with self._lock:
            row = self._db.execute("SELECT report FROM windows WHERE idx = ?", (idx,)).fetchone()
        return {s["kind"] for s in json.loads(row[0])["signals"]} if row else set()

    def window_reports(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT report FROM windows ORDER BY idx DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT received_at, status, alertname, severity, payload FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"received_at": r[0], "status": r[1], "alertname": r[2], "severity": r[3],
                 "payload": json.loads(r[4])} for r in rows]

    def save_decision(self, run, signals: list[str]) -> None:
        action = run.policy.final_action if run.policy else None
        mode = run.policy.mode if run.policy else None
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                             (run.id, run.window_index, run.created_at.isoformat(), action, mode, run.status,
                              json.dumps(sorted(signals)), run.model_dump_json()))

    def decisions(self, limit: int = 50) -> list[dict]:
        """Résumés, du plus ancien au plus récent."""
        with self._lock:
            rows = self._db.execute("SELECT id, window_idx, created_at, action, mode, status, signals FROM decisions "
                                    "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "window_index": r[1], "created_at": r[2], "action": r[3], "mode": r[4], "status": r[5],
                 "signals": json.loads(r[6])} for r in reversed(rows)]

    def decision(self, decision_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def firing_alerts(self) -> list[dict]:
        """Dernier état connu de chaque alerte (alertname + labels) : on garde celles encore actives."""
        with self._lock:
            rows = self._db.execute("SELECT status, alertname, severity, payload FROM alerts ORDER BY id").fetchall()
        last: dict[str, dict] = {}
        for status, name, sev, payload in rows:
            p = json.loads(payload)
            key = name + json.dumps(p.get("labels", {}), sort_keys=True)
            last[key] = {"status": status, "alertname": name, "severity": sev, "payload": p}
        return [a for a in last.values() if a["status"] == "firing"]

    def labelled_frame(self, seq_start: int, seq_end: int, target: str) -> pd.DataFrame:
        df = self.window_frame(seq_start, seq_end, target)
        return df[df[target].notna()] if target in df.columns else df.iloc[0:0]

    def traffic_stats(self) -> dict:
        with self._lock:
            total, ok, labelled = self._db.execute(
                "SELECT COUNT(*), COUNT(seq), (SELECT COUNT(*) FROM labels) FROM predictions").fetchone()
        return {"requests": total, "predictions": ok, "labels": labelled}
