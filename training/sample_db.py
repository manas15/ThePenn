"""
training/sample_db.py

SQLite index of training samples. Source of truth stays in the jsonl files
under training_data/; this DB is a derived index for fast dedup, counts, and
per-word / per-source queries.

Usage (CLI):
    python -m training.sample_db stats
    python -m training.sample_db rebuild
    python -m training.sample_db find <word>
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
DB_PATH = DATA_DIR / "samples.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    word TEXT NOT NULL,
    num_samples INTEGER,
    duration_s REAL,
    sample_rate_hz REAL,
    recorded_at TEXT,
    created_at TEXT,
    audio_file TEXT,
    source_file TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_word ON samples(word);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source_file);
"""


def _relpath(filepath):
    """Return path relative to training_data/, or absolute if outside."""
    try:
        return str(Path(filepath).resolve().relative_to(DATA_DIR.resolve()))
    except ValueError:
        return str(filepath)


def get_conn(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


def record_sample(sample, source_file, conn=None):
    """Insert a sample into the index. Returns True if inserted, False if dup."""
    own = conn is None
    if own:
        conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO samples
               (id, word, num_samples, duration_s, sample_rate_hz,
                recorded_at, created_at, audio_file, source_file, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sample["id"],
                sample.get("word", ""),
                sample.get("num_samples"),
                sample.get("duration_s"),
                sample.get("sample_rate_hz"),
                sample.get("recorded_at"),
                sample.get("created_at"),
                sample.get("audio_file"),
                _relpath(source_file),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        if own:
            conn.close()


def delete_sample_from_db(sample_id, conn=None):
    own = conn is None
    if own:
        conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        if own:
            conn.close()


def rebuild(data_dir=None, verbose=True):
    """Scan all jsonl files under training_data/ and rebuild the index."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    db = data_dir / "samples.db"
    if db.exists():
        db.unlink()
    conn = get_conn(db)

    patterns = [
        data_dir / "samples.jsonl",
        *sorted(data_dir.glob("sessions/*.jsonl")),
        *sorted(data_dir.glob("sessions/auto/*.jsonl")),
    ]

    inserted = 0
    duplicates = 0
    for path in patterns:
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record_sample(sample, str(path), conn=conn):
                    inserted += 1
                else:
                    duplicates += 1
    conn.close()
    if verbose:
        print(f"Rebuilt {db}")
        print(f"  inserted: {inserted}")
        print(f"  duplicates skipped: {duplicates}")
    return inserted, duplicates


def stats(conn=None):
    own = conn is None
    if own:
        conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        duration = conn.execute("SELECT COALESCE(SUM(duration_s), 0) FROM samples").fetchone()[0]
        per_word = conn.execute(
            "SELECT word, COUNT(*) FROM samples GROUP BY word ORDER BY 2 DESC"
        ).fetchall()
        per_source = conn.execute(
            "SELECT source_file, COUNT(*) FROM samples GROUP BY source_file ORDER BY 2 DESC"
        ).fetchall()
        return {
            "total": total,
            "total_duration_s": round(duration, 1),
            "per_word": per_word,
            "per_source": per_source,
        }
    finally:
        if own:
            conn.close()


def find_word(word, conn=None):
    own = conn is None
    if own:
        conn = get_conn()
    try:
        return conn.execute(
            """SELECT id, word, duration_s, source_file, recorded_at
               FROM samples WHERE word = ? ORDER BY recorded_at""",
            (word,),
        ).fetchall()
    finally:
        if own:
            conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sample index DB")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebuild", help="Rebuild the index from jsonl sources")
    sub.add_parser("stats", help="Show index statistics")
    find_p = sub.add_parser("find", help="Find samples by word")
    find_p.add_argument("word")

    args = parser.parse_args(argv)

    if args.cmd == "rebuild":
        rebuild()
    elif args.cmd == "stats":
        s = stats()
        print(f"Total unique samples: {s['total']}")
        print(f"Total duration: {s['total_duration_s']}s")
        print()
        print(f"Per word ({len(s['per_word'])} unique):")
        for word, n in s["per_word"]:
            print(f"  {n:4d}  {word}")
        print()
        print(f"Per source ({len(s['per_source'])} files):")
        for src, n in s["per_source"]:
            print(f"  {n:4d}  {src}")
    elif args.cmd == "find":
        rows = find_word(args.word)
        print(f"{len(rows)} samples for '{args.word}':")
        for sid, word, dur, src, rec in rows:
            print(f"  {sid[:8]}  {dur:.2f}s  {src}  {rec}")


if __name__ == "__main__":
    main()
