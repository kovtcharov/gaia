# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Transactional run admission, event history and restart-safe interaction receipts."""

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = 1
TERMINAL = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}


class StoreError(RuntimeError):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code, self.status = code, status


def encode(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def request_hash(request):
    return hashlib.sha256(encode({"schema": 1, **request}).encode()).hexdigest()


class Store:
    """One local writer; transactions reserve both workspace and session."""

    def __init__(self, directory, deployment, workspaces, slots=1):
        root = Path(directory)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_mode & 0o077:
            raise ValueError("Controller state requires a private directory")
        if (root / "RESTORE_INCOMPLETE").exists():
            raise StoreError("restore_incomplete", 503)
        self.root, self.slots = root, slots
        self.lock = threading.RLock()
        self.owner = (root / "controller.lock").open("a+")
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self._initialize(deployment, workspaces)
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.owner.close()
            raise

    def _initialize(self, deployment, workspaces):
        root = self.root
        self.db = sqlite3.connect(
            root / "service.sqlite3",
            timeout=1,
            check_same_thread=False,
            isolation_level=None,
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise StoreError("unsupported_schema", 503)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS deployment(id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS workspaces(id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS sessions(
          id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id),
          title TEXT NOT NULL, created_at REAL NOT NULL, history_status TEXT NOT NULL DEFAULT 'complete',
          deleted_at REAL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), workspace_id TEXT NOT NULL,
          key TEXT UNIQUE NOT NULL, request_hash TEXT NOT NULL, request_schema_version INTEGER NOT NULL,
          request TEXT NOT NULL, state TEXT NOT NULL, execution_status TEXT NOT NULL,
          generation TEXT NOT NULL, policy TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL,
          ended_at REAL, reason TEXT, outcome_uncertain INTEGER NOT NULL DEFAULT 0,
          seq INTEGER NOT NULL DEFAULT 0, earliest_seq INTEGER NOT NULL DEFAULT 1,
          event_bytes INTEGER NOT NULL DEFAULT 0, terminal_seq INTEGER);
        CREATE UNIQUE INDEX IF NOT EXISTS active_session ON runs(session_id) WHERE ended_at IS NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS active_workspace ON runs(workspace_id) WHERE ended_at IS NULL;
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
          run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, role TEXT NOT NULL, content TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, seq INTEGER NOT NULL,
          type TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL, encoded_bytes INTEGER NOT NULL,
          PRIMARY KEY(run_id,seq));
        CREATE TABLE IF NOT EXISTS interactions(
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
          generation TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
          digest TEXT NOT NULL, expires_at REAL NOT NULL, decision TEXT, submission_id TEXT,
          delivery TEXT NOT NULL DEFAULT 'not_attempted');
        CREATE TABLE IF NOT EXISTS tombstones(
          key TEXT PRIMARY KEY, request_hash TEXT NOT NULL, run_id TEXT NOT NULL, expires_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cleanup_jobs(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, state TEXT NOT NULL, error TEXT);
        """)
        with self.transaction():
            row = self.db.execute("SELECT id FROM deployment").fetchone()
            if row and row[0] != deployment:
                raise StoreError("deployment_mismatch", 503)
            self.db.execute(
                "INSERT OR IGNORE INTO deployment VALUES (?)", (deployment,)
            )
            for workspace in workspaces:
                self.db.execute(
                    "INSERT OR IGNORE INTO workspaces VALUES (?)", (workspace,)
                )
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        os.chmod(root / "service.sqlite3", 0o600)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                raise

    def close(self):
        with self.lock:
            self.db.close()
            self.owner.close()

    def session(self, session_id):
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM sessions WHERE id=? AND deleted_at IS NULL",
                (session_id,),
            ).fetchone()
            if row is None:
                raise StoreError("session_not_found", 404)
            return dict(row)

    def create_session(self, workspace, title):
        if not isinstance(title, str) or len(title) > 200:
            raise StoreError("invalid_title", 422)
        with self.transaction():
            if not self.db.execute(
                "SELECT id FROM workspaces WHERE id=?", (workspace,)
            ).fetchone():
                raise StoreError("workspace_not_found", 404)
            session_id = str(uuid4())
            self.db.execute(
                "INSERT INTO sessions(id,workspace_id,title,created_at) VALUES (?,?,?,?)",
                (session_id, workspace, title, time.time()),
            )
        return self.session(session_id)

    def run(self, run_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise StoreError("run_not_found", 404)
            result = dict(row)
            result["request"] = json.loads(result["request"])
            result["policy"] = json.loads(result["policy"])
            return result

    def admit(
        self, session_id, key, prompt, max_steps, policy, acknowledge_history_loss=False
    ):
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 128
            or any(c.isspace() for c in key)
        ):
            raise StoreError("invalid_idempotency_key", 422)
        if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > 65536:
            raise StoreError("invalid_prompt", 422)
        if type(max_steps) is not int or not 1 <= max_steps <= 20:
            raise StoreError("invalid_step_limit", 422)
        request = {
            "session_id": session_id,
            "prompt": prompt,
            "max_steps": max_steps,
            "acknowledge_history_loss": acknowledge_history_loss,
        }
        digest = request_hash(request)
        with self.transaction():
            previous = self.db.execute(
                "SELECT id,request_hash FROM runs WHERE key=?", (key,)
            ).fetchone()
            if previous:
                if previous["request_hash"] != digest:
                    raise StoreError("idempotency_conflict")
                return self.run(previous["id"]), False
            tombstone = self.db.execute(
                "SELECT * FROM tombstones WHERE key=?", (key,)
            ).fetchone()
            if tombstone:
                if tombstone["request_hash"] != digest:
                    raise StoreError("idempotency_conflict")
                raise StoreError("run_history_gone", 410)
            session = self.session(session_id)
            if session["history_status"] != "complete" and not acknowledge_history_loss:
                raise StoreError("history_expired_acknowledgement_required")
            if self.db.execute(
                "SELECT 1 FROM runs WHERE session_id=? AND ended_at IS NULL",
                (session_id,),
            ).fetchone():
                raise StoreError("session_busy")
            if self.db.execute(
                "SELECT 1 FROM runs WHERE workspace_id=? AND ended_at IS NULL",
                (session["workspace_id"],),
            ).fetchone():
                raise StoreError("workspace_busy")
            if (
                self.db.execute(
                    "SELECT count(*) FROM runs WHERE ended_at IS NULL"
                ).fetchone()[0]
                >= self.slots
            ):
                raise StoreError("capacity_exhausted", 503)
            run_id, generation = str(uuid4()), str(uuid4())
            self.db.execute(
                """INSERT INTO runs(id,session_id,workspace_id,key,request_hash,request_schema_version,
                request,state,execution_status,generation,policy,created_at) VALUES (?,?,?,?,?,1,?,'accepted','not_started',?,?,?)""",
                (
                    run_id,
                    session_id,
                    session["workspace_id"],
                    key,
                    digest,
                    encode(request),
                    generation,
                    encode(policy),
                    time.time(),
                ),
            )
            self.db.execute(
                "INSERT INTO messages(session_id,run_id,role,content) VALUES (?,?,'user',?)",
                (session_id, run_id, prompt),
            )
            self._append(run_id, {"type": "accepted", "run_id": run_id})
            return self.run(run_id), True

    def _append(self, run_id, payload, terminal=False):
        raw = encode(payload)
        size = len(raw.encode())
        row = self.db.execute(
            "SELECT seq,event_bytes FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise StoreError("run_not_found", 404)
        limit = 16 * 1024 * 1024 if terminal else 16 * 1024 * 1024 - 262144
        if size > 262144 or row["event_bytes"] + size > limit:
            raise StoreError("event_storage_limit", 413)
        seq = row["seq"] + 1
        self.db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (run_id, seq, payload["type"], raw, time.time(), size),
        )
        self.db.execute(
            "UPDATE runs SET seq=?,event_bytes=event_bytes+? WHERE id=?",
            (seq, size, run_id),
        )
        return seq

    def append(self, run_id, event):
        with self.transaction():
            if self.run(run_id)["ended_at"] is not None:
                raise StoreError("run_terminal")
            return self._append(run_id, event)

    def dispatching(self, run_id):
        with self.transaction():
            changed = self.db.execute(
                "UPDATE runs SET execution_status='unknown' WHERE id=? AND state='accepted' AND execution_status='not_started'",
                (run_id,),
            )
            return changed.rowcount == 1

    def started(self, run_id):
        with self.transaction():
            self.db.execute(
                "UPDATE runs SET state='running',execution_status='active',started_at=? WHERE id=? AND state='accepted'",
                (time.time(), run_id),
            )

    def stopping(self, run_id, reason="cancelled"):
        with self.transaction():
            row = self.run(run_id)
            if row["ended_at"] is not None:
                return row
            self.db.execute(
                "UPDATE runs SET state='stopping',reason=COALESCE(reason,?) WHERE id=?",
                (reason, run_id),
            )
            return self.run(run_id)

    def finish(
        self,
        run_id,
        state,
        event,
        confirmed_stopped,
        uncertain=False,
        rotate_generation=False,
    ):
        if not confirmed_stopped or state not in TERMINAL:
            raise StoreError("termination_unconfirmed", 503)
        with self.transaction():
            row = self.run(run_id)
            if row["ended_at"] is not None:
                return row
            if row["state"] == "stopping" and state != "interrupted":
                state = "cancelled"
                uncertain = row["execution_status"] != "not_started"
                event = {"type": "error", "detail": "Run cancelled.", "status": 499}
            if len(encode({**event, "state": state}).encode()) > 262144:
                state, uncertain = "failed", True
                event = {
                    "type": "error",
                    "detail": "Terminal output exceeded storage limit.",
                    "status": 413,
                }
            seq = self._append(run_id, {**event, "state": state}, terminal=True)
            self.db.execute(
                "UPDATE runs SET state=?,execution_status='stopped',ended_at=?,terminal_seq=?,outcome_uncertain=? WHERE id=?",
                (state, time.time(), seq, int(uncertain), run_id),
            )
            self.db.execute(
                "UPDATE interactions SET state='invalidated' WHERE run_id=? AND state='pending'",
                (run_id,),
            )
            if rotate_generation:
                self.db.execute(
                    "UPDATE runs SET generation=? WHERE id=?", (str(uuid4()), run_id)
                )
            if state == "succeeded":
                answer = str(event.get("answer", ""))
                self.db.execute(
                    "INSERT INTO messages(session_id,run_id,role,content) VALUES (?,?,'assistant',?)",
                    (row["session_id"], run_id, answer),
                )
            return self.run(run_id)

    def active(self):
        with self.lock:
            return [
                dict(row)
                for row in self.db.execute(
                    "SELECT id,generation,state,execution_status FROM runs WHERE ended_at IS NULL"
                )
            ]

    def events(self, run_id, after=0, limit=100):
        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise StoreError("invalid_cursor", 422)
        with self.lock:
            run = self.run(run_id)
            if after < run["earliest_seq"] - 1:
                raise StoreError("event_history_gone", 410)
            if after > run["seq"]:
                raise StoreError("future_cursor")
            result, size = [], 0
            for row in self.db.execute(
                "SELECT seq,payload FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after, limit),
            ):
                event = {"seq": row["seq"], "event": json.loads(row["payload"])}
                encoded = len(encode(event).encode())
                if size + encoded > 1024 * 1024:
                    break
                result.append(event)
                size += encoded
            return {
                "events": result,
                "earliest_seq": run["earliest_seq"],
                "latest_seq": run["seq"],
                "terminal": run["ended_at"] is not None,
            }

    def messages(self, session_id, after=0, limit=100):
        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise StoreError("invalid_cursor", 422)
        with self.lock:
            session = self.session(session_id)
            result, size = [], 0
            for row in self.db.execute(
                "SELECT id,run_id,role,content FROM messages WHERE session_id=? AND id>? ORDER BY id LIMIT ?",
                (session_id, after, limit),
            ):
                message = dict(row)
                encoded = len(encode(message).encode())
                if size + encoded > 1024 * 1024:
                    break
                size += encoded
                result.append(message)
            return {"session": session, "messages": result}

    def context(self, session_id, current_run):
        with self.lock:
            rows, size = [], 0
            for row in self.db.execute(
                "SELECT role,content FROM messages WHERE session_id=? AND run_id<>? ORDER BY id",
                (session_id, current_run),
            ):
                value = dict(row)
                size += len(encode(value).encode())
                if size > 512 * 1024:
                    raise StoreError("context_limit_start_new_session", 422)
                rows.append(value)
            return rows

    def interaction(self, run_id, interaction_id=None):
        with self.lock:
            if interaction_id:
                row = self.db.execute(
                    "SELECT * FROM interactions WHERE run_id=? AND id=?",
                    (run_id, interaction_id),
                ).fetchone()
            else:
                row = self.db.execute(
                    "SELECT * FROM interactions WHERE run_id=? AND state='pending' ORDER BY expires_at LIMIT 1",
                    (run_id,),
                ).fetchone()
            if row is None:
                raise StoreError("interaction_not_found", 404)
            return dict(row)

    def create_interaction(self, run_id, kind, raw, expires):
        with self.transaction():
            run = self.run(run_id)
            if run["state"] != "running":
                raise StoreError("run_not_running")
            if self.db.execute(
                "SELECT id FROM interactions WHERE run_id=? AND state='pending'",
                (run_id,),
            ).fetchone():
                raise StoreError("interaction_already_pending")
            interaction_id = str(uuid4())
            digest = hashlib.sha256(encode(raw).encode()).hexdigest()
            self.db.execute(
                "INSERT INTO interactions(id,run_id,generation,kind,state,digest,expires_at) VALUES (?,?,?,?,'pending',?,?)",
                (interaction_id, run_id, run["generation"], kind, digest, expires),
            )
            self._append(
                run_id,
                {
                    "type": kind,
                    "interaction_id": interaction_id,
                    "generation": run["generation"],
                    "argument_digest": digest,
                },
            )
            return self.interaction(run_id, interaction_id)

    def receipt(self, run_id, interaction_id, generation, decision, submission_id=None):
        with self.transaction():
            run = self.run(run_id)
            row = self.interaction(run_id, interaction_id)
            if generation != run["generation"] or generation != row["generation"]:
                raise StoreError("stale_generation")
            if row["state"] == "accepted":
                if row["kind"] == "needs_confirmation" and decision == row["decision"]:
                    return row, False
                if (
                    row["kind"] == "needs_input"
                    and submission_id == row["submission_id"]
                ):
                    return row, False
                raise StoreError("interaction_already_decided")
            if (
                run["state"] != "running"
                or row["state"] != "pending"
                or time.time() >= row["expires_at"]
            ):
                raise StoreError("interaction_expired")
            if row["kind"] == "needs_confirmation" and decision not in {
                "approve",
                "deny",
            }:
                raise StoreError("invalid_decision", 422)
            if row["kind"] == "needs_input" and (
                decision != "answered" or not submission_id
            ):
                raise StoreError("submission_id_required", 422)
            self.db.execute(
                "UPDATE interactions SET state='accepted',decision=?,submission_id=?,delivery='unknown' WHERE id=?",
                (decision, submission_id, interaction_id),
            )
            self._append(
                run_id,
                {
                    "type": "interaction_receipt",
                    "interaction_id": interaction_id,
                    "decision": decision,
                },
            )
            return self.interaction(run_id, interaction_id), True

    def delivered(self, interaction_id):
        with self.transaction():
            self.db.execute(
                "UPDATE interactions SET delivery='acknowledged' WHERE id=? AND state='accepted'",
                (interaction_id,),
            )

    def delete_session(self, session_id):
        with self.transaction():
            self.session(session_id)
            if self.db.execute(
                "SELECT id FROM runs WHERE session_id=? AND ended_at IS NULL",
                (session_id,),
            ).fetchone():
                raise StoreError("session_busy")
            for row in self.db.execute(
                "SELECT key,request_hash,id,ended_at FROM runs WHERE session_id=?",
                (session_id,),
            ):
                self.db.execute(
                    "INSERT OR REPLACE INTO tombstones VALUES (?,?,?,?)",
                    (
                        row["key"],
                        row["request_hash"],
                        row["id"],
                        max(time.time(), row["ended_at"]) + 30 * 86400,
                    ),
                )
            self.db.execute("DELETE FROM runs WHERE session_id=?", (session_id,))
            self.db.execute(
                "UPDATE sessions SET title='',deleted_at=? WHERE id=?",
                (time.time(), session_id),
            )
            job = str(uuid4())
            # All files in this preview belong to operator-managed workspaces.
            # Delete logical history only; never remove operator-owned files.
            self.db.execute(
                "INSERT INTO cleanup_jobs VALUES (?,?,'succeeded',NULL)",
                (job, session_id),
            )
            return {
                "id": job,
                "state": "succeeded",
                "operator_workspace_retained": True,
            }

    def cleanup(self, job_id):
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM cleanup_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise StoreError("cleanup_not_found", 404)
            return dict(row)

    def expire_history(self, before):
        with self.transaction():
            old = self.db.execute(
                "SELECT id,session_id,seq FROM runs WHERE ended_at IS NOT NULL AND ended_at<? AND earliest_seq<=seq LIMIT 100",
                (before,),
            ).fetchall()
            for row in old:
                self.db.execute("DELETE FROM events WHERE run_id=?", (row["id"],))
                self.db.execute("DELETE FROM messages WHERE run_id=?", (row["id"],))
                self.db.execute(
                    "UPDATE runs SET earliest_seq=?,request='{}' WHERE id=?",
                    (row["seq"] + 1, row["id"]),
                )
                self.db.execute(
                    "UPDATE sessions SET history_status='expired' WHERE id=?",
                    (row["session_id"],),
                )
            return len(old)

    def maintain(self, now=None):
        now = time.time() if now is None else now
        expired = self.expire_history(now - 7 * 86400)
        with self.transaction():
            self.db.execute("DELETE FROM tombstones WHERE expires_at < ?", (now,))
        return expired

    def backup(self, destination):
        destination = Path(destination)
        with self.lock:
            if self.active():
                raise StoreError("drain_before_backup")
            if destination.exists():
                raise StoreError("backup_exists")
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            os.close(descriptor)
            target = sqlite3.connect(destination)
            try:
                self.db.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise StoreError("backup_integrity_failed", 503)
            finally:
                target.close()
        return {
            "schema_version": SCHEMA_VERSION,
            "scope": "metadata_only",
            "workspaces_require_stopped_backup": True,
        }
