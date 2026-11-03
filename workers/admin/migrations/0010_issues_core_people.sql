-- Enough of each issue to list it. Rows from before this migration have NULL
-- updated_at, which makes the next collection read the full history once.
ALTER TABLE items ADD COLUMN title TEXT;
ALTER TABLE items ADD COLUMN comments INTEGER;
ALTER TABLE items ADD COLUMN updated_at TEXT;
ALTER TABLE items ADD COLUMN labels TEXT;   -- JSON array of label names

-- The core team as edited on the Developers page. One row per person; `entry`
-- joins one person's accounts with +, as MAINTAINER_LOGINS does.
CREATE TABLE core_people (
  entry    TEXT PRIMARY KEY,
  added_by TEXT NOT NULL,
  added_at TEXT NOT NULL
);
