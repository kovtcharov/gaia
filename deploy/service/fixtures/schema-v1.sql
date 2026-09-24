BEGIN TRANSACTION;
CREATE TABLE cleanup_jobs(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, state TEXT NOT NULL, error TEXT);
CREATE TABLE deployment(id TEXT PRIMARY KEY);
INSERT INTO "deployment" VALUES('11111111-1111-4111-8111-111111111111');
CREATE TABLE events(
          run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, seq INTEGER NOT NULL,
          type TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL, encoded_bytes INTEGER NOT NULL,
          PRIMARY KEY(run_id,seq));
INSERT INTO "events" VALUES('73401b17-4db7-4772-9d83-11cee7a101e6',1,'accepted','{"run_id":"73401b17-4db7-4772-9d83-11cee7a101e6","type":"accepted"}',1790153414.364969,67);
INSERT INTO "events" VALUES('73401b17-4db7-4772-9d83-11cee7a101e6',2,'final','{"answer":"preserved beta answer","state":"succeeded","type":"final"}',1790153414.3651898,69);
CREATE TABLE interactions(
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
          generation TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
          digest TEXT NOT NULL, expires_at REAL NOT NULL, decision TEXT, submission_id TEXT,
          delivery TEXT NOT NULL DEFAULT 'not_attempted');
CREATE TABLE messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
          run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, role TEXT NOT NULL, content TEXT NOT NULL);
INSERT INTO "messages" VALUES(1,'ef9f7433-6d82-438b-9b85-8db6ce17ed83','73401b17-4db7-4772-9d83-11cee7a101e6','user','fixture content');
INSERT INTO "messages" VALUES(2,'ef9f7433-6d82-438b-9b85-8db6ce17ed83','73401b17-4db7-4772-9d83-11cee7a101e6','assistant','preserved beta answer');
CREATE TABLE runs(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), workspace_id TEXT NOT NULL,
          key TEXT UNIQUE NOT NULL, request_hash TEXT NOT NULL, request_schema_version INTEGER NOT NULL,
          request TEXT NOT NULL, state TEXT NOT NULL, execution_status TEXT NOT NULL,
          generation TEXT NOT NULL, policy TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL,
          ended_at REAL, reason TEXT, outcome_uncertain INTEGER NOT NULL DEFAULT 0,
          seq INTEGER NOT NULL DEFAULT 0, earliest_seq INTEGER NOT NULL DEFAULT 1,
          event_bytes INTEGER NOT NULL DEFAULT 0, terminal_seq INTEGER);
INSERT INTO "runs" VALUES('73401b17-4db7-4772-9d83-11cee7a101e6','ef9f7433-6d82-438b-9b85-8db6ce17ed83','22222222-2222-4222-8222-222222222222','beta-fixture-key','bedb8d0cba84950d4087c426051c5f4e3817e7d6622b0c3921255b3e8bf2bf42',1,'{"acknowledge_history_loss":false,"max_steps":20,"prompt":"fixture content","session_id":"ef9f7433-6d82-438b-9b85-8db6ce17ed83"}','succeeded','stopped','f3a39330-7980-4f55-b85a-b1e35e241209','{"model":"fixture"}',1790153414.3649311,NULL,1790153414.3651969,NULL,0,2,1,136,2);
CREATE TABLE sessions(
          id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id),
          title TEXT NOT NULL, created_at REAL NOT NULL, history_status TEXT NOT NULL DEFAULT 'complete',
          deleted_at REAL);
INSERT INTO "sessions" VALUES('ef9f7433-6d82-438b-9b85-8db6ce17ed83','22222222-2222-4222-8222-222222222222','beta-fixture',1790153414.3645749,'complete',NULL);
CREATE TABLE tombstones(
          key TEXT PRIMARY KEY, request_hash TEXT NOT NULL, run_id TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE workspaces(id TEXT PRIMARY KEY);
INSERT INTO "workspaces" VALUES('22222222-2222-4222-8222-222222222222');
CREATE UNIQUE INDEX active_session ON runs(session_id) WHERE ended_at IS NULL;
CREATE UNIQUE INDEX active_workspace ON runs(workspace_id) WHERE ended_at IS NULL;
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('messages',2);
COMMIT;
PRAGMA user_version=1;
