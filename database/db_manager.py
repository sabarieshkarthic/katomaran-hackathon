"""
SQLite Database Manager
Two tables: face (clear face detected) and body (face never clear / temp_id)
"""

import sqlite3
import json
import uuid
import numpy as np
from datetime import datetime


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str):
    """Initialize the database with face and body tables."""
    conn = get_connection(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS face (
            face_id         TEXT PRIMARY KEY,
            entry_timestamp TEXT NOT NULL,
            exit_timestamp  TEXT,
            face_embedding  TEXT,
            body_embedding  TEXT,
            entry_image     TEXT,
            exit_image      TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS body (
            body_id         TEXT PRIMARY KEY,
            entry_timestamp TEXT NOT NULL,
            exit_timestamp  TEXT,
            body_embedding  TEXT,
            linked_face_id  TEXT REFERENCES face(face_id),
            entry_image     TEXT,
            exit_image      TEXT
        )
    """)

    conn.commit()
    conn.close()


def _serialize(vec) -> str:
    if vec is None:
        return None
    arr = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
    return json.dumps(arr)


def _deserialize(s) -> np.ndarray:
    if s is None:
        return None
    return np.array(json.loads(s), dtype=np.float32)


# ---------- FACE TABLE ----------

def insert_face(db_path, face_id, entry_ts, face_emb, body_emb, entry_image):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO face (face_id, entry_timestamp, face_embedding, body_embedding, entry_image) VALUES (?,?,?,?,?)",
        (face_id, entry_ts, _serialize(face_emb), _serialize(body_emb), entry_image)
    )
    conn.commit()
    conn.close()


def update_face_exit(db_path, face_id, exit_ts, exit_image):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE face SET exit_timestamp=?, exit_image=? WHERE face_id=?",
        (exit_ts, exit_image, face_id)
    )
    conn.commit()
    conn.close()


def update_face_embeddings(db_path, face_id, face_emb, body_emb):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE face SET face_embedding=?, body_embedding=? WHERE face_id=?",
        (_serialize(face_emb), _serialize(body_emb), face_id)
    )
    conn.commit()
    conn.close()


def get_all_faces(db_path):
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM face").fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            'face_id': r['face_id'],
            'entry_timestamp': r['entry_timestamp'],
            'exit_timestamp': r['exit_timestamp'],
            'face_embedding': _deserialize(r['face_embedding']),
            'body_embedding': _deserialize(r['body_embedding']),
            'entry_image': r['entry_image'],
            'exit_image': r['exit_image'],
        })
    return result


# ---------- BODY TABLE ----------

def insert_body(db_path, body_id, entry_ts, body_emb, entry_image):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO body (body_id, entry_timestamp, body_embedding, entry_image) VALUES (?,?,?,?)",
        (body_id, entry_ts, _serialize(body_emb), entry_image)
    )
    conn.commit()
    conn.close()


def update_body_exit(db_path, body_id, exit_ts, exit_image):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE body SET exit_timestamp=?, exit_image=? WHERE body_id=?",
        (exit_ts, exit_image, body_id)
    )
    conn.commit()
    conn.close()


def update_body_link(db_path, body_id, face_id):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE body SET linked_face_id=? WHERE body_id=?",
        (face_id, body_id)
    )
    conn.commit()
    conn.close()


def get_body_entry_ts(db_path, body_id) -> str:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT entry_timestamp FROM body WHERE body_id=?", (body_id,)
    ).fetchone()
    conn.close()
    return row['entry_timestamp'] if row else None


def get_all_bodies(db_path):
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM body").fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            'body_id': r['body_id'],
            'entry_timestamp': r['entry_timestamp'],
            'exit_timestamp': r['exit_timestamp'],
            'body_embedding': _deserialize(r['body_embedding']),
            'linked_face_id': r['linked_face_id'],
            'entry_image': r['entry_image'],
            'exit_image': r['exit_image'],
        })
    return result


# ---------- UNIQUE COUNT ----------

def get_unique_count(db_path) -> int:
    conn = get_connection(db_path)
    face_count = conn.execute(
        "SELECT COUNT(*) FROM face WHERE exit_timestamp IS NOT NULL"
    ).fetchone()[0]
    body_count = conn.execute(
        "SELECT COUNT(*) FROM body WHERE exit_timestamp IS NOT NULL AND linked_face_id IS NULL"
    ).fetchone()[0]
    conn.close()
    return face_count + body_count


def get_stats(db_path) -> dict:
    conn = get_connection(db_path)
    total_face = conn.execute("SELECT COUNT(*) FROM face").fetchone()[0]
    total_body = conn.execute(
        "SELECT COUNT(*) FROM body WHERE linked_face_id IS NULL"
    ).fetchone()[0]
    exits_face = conn.execute(
        "SELECT COUNT(*) FROM face WHERE exit_timestamp IS NOT NULL"
    ).fetchone()[0]
    exits_body = conn.execute(
        "SELECT COUNT(*) FROM body WHERE exit_timestamp IS NOT NULL AND linked_face_id IS NULL"
    ).fetchone()[0]
    conn.close()
    return {
        'total_entries': total_face + total_body,
        'total_exits': exits_face + exits_body,
        'unique_visitors': exits_face + exits_body,
    }


def face_exists(db_path, face_id) -> bool:
    conn = get_connection(db_path)
    row = conn.execute("SELECT 1 FROM face WHERE face_id=?", (face_id,)).fetchone()
    conn.close()
    return row is not None


def body_exists(db_path, body_id) -> bool:
    conn = get_connection(db_path)
    row = conn.execute("SELECT 1 FROM body WHERE body_id=?", (body_id,)).fetchone()
    conn.close()
    return row is not None


# ---------- FIX 2: entry_image update functions (were missing) ----------

def update_face_entry_image(db_path, face_id, entry_image):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE face SET entry_image=? WHERE face_id=?",
        (entry_image, face_id)
    )
    conn.commit()
    conn.close()


def update_body_entry_image(db_path, body_id, entry_image):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE body SET entry_image=? WHERE body_id=?",
        (entry_image, body_id)
    )
    conn.commit()
    conn.close()


# ---------- FIX 3b: body embedding update (was missing) ----------

def update_body_embedding(db_path, body_id, body_emb):
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE body SET body_embedding=? WHERE body_id=?",
        (_serialize(body_emb), body_id)
    )
    conn.commit()
    conn.close()