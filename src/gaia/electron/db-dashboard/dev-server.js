// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Development Server (Browser Mode)
 *
 * Provides an HTTP API backend that mirrors the Electron IPC handlers,
 * allowing the React + Vite frontend to run in a browser without Electron.
 *
 * The React frontend (served by Vite on port 5173) calls this API server
 * through a browser-mode dbAPI shim injected at startup.
 *
 * Usage:
 *   node dev-server.js [workspace-path]
 *
 * Example:
 *   node dev-server.js ~/.gaia/workspace
 *   node dev-server.js C:/Users/me/.gaia/workspace
 *   node dev-server.js /home/user/.gaia/workspace
 *
 * Then open http://localhost:5173 (Vite) which proxies API calls to this server,
 * or open http://localhost:3847 for the standalone legacy renderer.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PORT = process.env.PORT || 3847;

/**
 * Detect the workspace path, supporting WSL and native environments.
 */
function detectWorkspacePath() {
  // If explicitly provided, use that
  if (process.argv[2]) {
    return process.argv[2];
  }

  const platform = os.platform();

  // Check if running inside WSL
  if (platform === 'linux') {
    try {
      const release = fs.readFileSync('/proc/version', 'utf-8');
      if (release.toLowerCase().includes('microsoft') || release.toLowerCase().includes('wsl')) {
        // Running in WSL
        const wslPath = path.join(os.homedir(), '.gaia', 'workspace');
        if (fs.existsSync(wslPath)) {
          return wslPath;
        }
      }
    } catch (e) {
      // Not WSL
    }
  }

  return path.join(os.homedir(), '.gaia', 'workspace');
}

const WORKSPACE = detectWorkspacePath();

let Database;
try {
  Database = require('better-sqlite3');
} catch (err) {
  console.error('better-sqlite3 not available. Install with: npm install better-sqlite3');
  console.error('The dev server requires better-sqlite3 for database access.');
  process.exit(1);
}

// MIME types
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.tsx': 'application/javascript',
  '.ts': 'application/javascript',
};

// Database connection cache
const connections = new Map();

function getConnection(dbPath, readOnly = false) {
  const key = `${dbPath}:${readOnly ? 'ro' : 'rw'}`;
  if (connections.has(key)) return connections.get(key);
  const db = new Database(dbPath, { readonly: readOnly });
  db.pragma('journal_mode = WAL');
  db.pragma('busy_timeout = 5000');
  connections.set(key, db);
  return db;
}

// Known databases
const KNOWN_DBS = ['memory.db', 'knowledge.db', 'tools.db', 'skills.db', 'agents.db', 'plan.db', 'logs.db'];

// API handlers (mirror the Electron IPC handlers)
const apiHandlers = {
  'listDatabases': (params) => {
    const dir = params.workspacePath || WORKSPACE;
    if (!fs.existsSync(dir)) {
      return { success: false, error: `Workspace not found: ${dir}`, databases: [] };
    }
    const databases = [];

    for (const name of KNOWN_DBS) {
      const fullPath = path.join(dir, name);
      const exists = fs.existsSync(fullPath);
      let sizeBytes = 0, lastModified = null;
      if (exists) {
        const stat = fs.statSync(fullPath);
        sizeBytes = stat.size;
        lastModified = stat.mtime.toISOString();
      }
      databases.push({ name, label: name.replace('.db', ''), path: fullPath, exists, sizeBytes, lastModified });
    }

    // Scan for extra .db files
    try {
      const files = fs.readdirSync(dir);
      for (const file of files) {
        if (file.endsWith('.db') && !KNOWN_DBS.includes(file)) {
          const fullPath = path.join(dir, file);
          const stat = fs.statSync(fullPath);
          databases.push({
            name: file, label: file.replace('.db', ''), description: 'Custom database',
            path: fullPath, exists: true, sizeBytes: stat.size, lastModified: stat.mtime.toISOString(),
          });
        }
      }
    } catch (e) { /* ignore scan errors */ }

    return { success: true, databases, workspacePath: dir };
  },

  'getTables': (params) => {
    const db = getConnection(params.dbPath, true);
    const tables = db.prepare(`SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name ASC`).all();
    const result = tables.map(t => {
      const isFts = t.name.endsWith('_fts');
      let rowCount = 0;
      try { rowCount = db.prepare(`SELECT COUNT(*) as c FROM "${t.name}"`).get().c; } catch (e) {}
      let columns = [];
      try {
        columns = db.prepare(`PRAGMA table_info("${t.name}")`).all().map(c => ({
          name: c.name, type: c.type || 'TEXT', notNull: c.notnull === 1, defaultValue: c.dflt_value, primaryKey: c.pk === 1,
        }));
      } catch (e) {}
      return { name: t.name, type: t.type, isFts, rowCount, columns };
    });
    return { success: true, tables: result };
  },

  'queryTable': (params) => {
    const db = getConnection(params.dbPath, true);
    const { tableName, options = {} } = params;
    const { page = 1, limit = 50, sortColumn, sortDirection = 'ASC', filterColumn, filterValue } = options;
    const offset = (page - 1) * limit;
    let query = `SELECT * FROM "${tableName}"`;
    const qParams = [];
    if (filterColumn && filterValue) { query += ` WHERE "${filterColumn}" LIKE ?`; qParams.push(`%${filterValue}%`); }
    if (sortColumn) { query += ` ORDER BY "${sortColumn}" ${sortDirection === 'DESC' ? 'DESC' : 'ASC'}`; }
    query += ` LIMIT ? OFFSET ?`;
    qParams.push(limit, offset);
    const rows = db.prepare(query).all(...qParams);
    let countQuery = `SELECT COUNT(*) as total FROM "${tableName}"`;
    const cParams = [];
    if (filterColumn && filterValue) { countQuery += ` WHERE "${filterColumn}" LIKE ?`; cParams.push(`%${filterValue}%`); }
    const totalRows = db.prepare(countQuery).get(...cParams).total;
    return { success: true, rows, pagination: { page, limit, totalRows, totalPages: Math.ceil(totalRows / limit) } };
  },

  'executeSQL': (params) => {
    const db = getConnection(params.dbPath, params.readOnly);
    const sql = params.sql.trim();
    const isSelect = /^(SELECT|PRAGMA|EXPLAIN|WITH)\b/i.test(sql);
    const start = Date.now();
    if (isSelect) {
      const rows = db.prepare(sql).all();
      return { success: true, type: 'query', rows, columns: rows.length ? Object.keys(rows[0]) : [], rowCount: rows.length, duration: Date.now() - start };
    } else {
      const info = db.prepare(sql).run();
      return { success: true, type: 'statement', changes: info.changes, lastInsertRowid: info.lastInsertRowid, duration: Date.now() - start };
    }
  },

  'getSchema': (params) => {
    const db = getConnection(params.dbPath, true);
    const schemas = db.prepare(`SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type DESC, name ASC`).all();
    return { success: true, schemas };
  },

  'searchFTS5': (params) => {
    const db = getConnection(params.dbPath, true);
    const { ftsTable, query, options = {} } = params;
    const sanitized = query.replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
    if (!sanitized) return { success: true, rows: [], rowCount: 0 };
    const ftsQuery = sanitized.split(' ').join(' OR ');
    const limit = options.limit || 100;

    const schemaRow = db.prepare(`SELECT sql FROM sqlite_master WHERE name = ?`).get(ftsTable);
    let rows;
    if (schemaRow && schemaRow.sql && schemaRow.sql.includes('content=')) {
      const m = schemaRow.sql.match(/content=(\w+)/);
      if (m) {
        rows = db.prepare(`SELECT c.* FROM "${m[1]}" c JOIN "${ftsTable}" f ON c.rowid = f.rowid WHERE "${ftsTable}" MATCH ? ORDER BY rank LIMIT ?`).all(ftsQuery, limit);
      } else {
        rows = db.prepare(`SELECT *, rank FROM "${ftsTable}" WHERE "${ftsTable}" MATCH ? ORDER BY rank LIMIT ?`).all(ftsQuery, limit);
      }
    } else {
      rows = db.prepare(`SELECT *, rank FROM "${ftsTable}" WHERE "${ftsTable}" MATCH ? ORDER BY rank LIMIT ?`).all(ftsQuery, limit);
    }
    return { success: true, rows, rowCount: rows.length };
  },

  'getWorkspacePath': () => WORKSPACE,
};

// Create server
const server = http.createServer((req, res) => {
  // CORS for dev (allow Vite dev server on port 5173)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // API routes
  if (req.url.startsWith('/api/') && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const action = req.url.replace('/api/', '');
      const handler = apiHandlers[action];
      if (!handler) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: 'Unknown action' }));
        return;
      }
      try {
        const params = body ? JSON.parse(body) : {};
        const result = handler(params);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: err.message }));
      }
    });
    return;
  }

  // Static files - try renderer/ directory for legacy fallback
  let filePath = req.url === '/' ? '/index.html' : req.url;
  filePath = path.join(__dirname, 'renderer', filePath);

  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    // For index.html, inject the browser mock dbAPI before other scripts
    if (filePath.endsWith('index.html')) {
      let html = data.toString();
      const mockScript = `
<script>
// Browser-mode mock for window.dbAPI (calls dev-server HTTP API instead of Electron IPC)
window.dbAPI = {
  async listDatabases(wp) {
    return fetch('/api/listDatabases', { method: 'POST', body: JSON.stringify({ workspacePath: wp }) }).then(r => r.json());
  },
  async getWorkspacePath() {
    return fetch('/api/getWorkspacePath', { method: 'POST' }).then(r => r.json());
  },
  async selectWorkspace() { return { success: false, error: 'Not available in browser mode' }; },
  async selectDatabaseFile() { return { success: false, error: 'Not available in browser mode' }; },
  async getTables(dbPath) {
    return fetch('/api/getTables', { method: 'POST', body: JSON.stringify({ dbPath }) }).then(r => r.json());
  },
  async queryTable(dbPath, tableName, options) {
    return fetch('/api/queryTable', { method: 'POST', body: JSON.stringify({ dbPath, tableName, options }) }).then(r => r.json());
  },
  async executeSQL(dbPath, sql, readOnly) {
    return fetch('/api/executeSQL', { method: 'POST', body: JSON.stringify({ dbPath, sql, readOnly }) }).then(r => r.json());
  },
  async updateRow(dbPath, tableName, pk, updates) {
    return { success: false, error: 'Write operations not available in browser mode' };
  },
  async deleteRow() { return { success: false, error: 'Write operations not available in browser mode' }; },
  async insertRow() { return { success: false, error: 'Write operations not available in browser mode' }; },
  async getSchema(dbPath) {
    return fetch('/api/getSchema', { method: 'POST', body: JSON.stringify({ dbPath }) }).then(r => r.json());
  },
  async searchFTS5(dbPath, ftsTable, query, options) {
    return fetch('/api/searchFTS5', { method: 'POST', body: JSON.stringify({ dbPath, ftsTable, query, options }) }).then(r => r.json());
  },
  async exportTable() { return { success: false, error: 'Export not available in browser mode' }; },
  async backup() { return { success: false, error: 'Backup not available in browser mode' }; },
  async closeConnection() { return { success: true }; },
  onMenuEvent() {},
};
</script>`;
      // Insert mock script before the first app script
      html = html.replace('<script src="js/state.js">', mockScript + '\n  <script src="js/state.js">');
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(html);
      return;
    }

    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`\nGAIA DB Dashboard - Development Server`);
  console.log(`  API URL:   http://localhost:${PORT}`);
  console.log(`  Workspace: ${WORKSPACE}`);
  console.log(`  Mode:      Browser API backend (read-only)\n`);
  console.log(`  For React UI: Run 'npm run dev' in another terminal, then open http://localhost:5173`);
  console.log(`  For Legacy UI: Open http://localhost:${PORT} directly\n`);

  // List available databases
  if (fs.existsSync(WORKSPACE)) {
    const dbs = fs.readdirSync(WORKSPACE).filter(f => f.endsWith('.db'));
    if (dbs.length > 0) {
      console.log(`  Databases found:`);
      dbs.forEach(db => {
        const stat = fs.statSync(path.join(WORKSPACE, db));
        const kb = (stat.size / 1024).toFixed(1);
        console.log(`    - ${db} (${kb} KB)`);
      });
    } else {
      console.log(`  No .db files found in workspace.`);
    }
  } else {
    console.log(`  Workspace directory does not exist yet.`);
    console.log(`  Run a GAIA agent first to create databases.`);
  }
  console.log('');
});
