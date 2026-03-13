// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Electron Main Process
 *
 * Provides IPC handlers for SQLite database operations.
 * All database access happens here in the main process for security.
 */

const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

// Disable GPU acceleration in WSL2 to prevent GPU init errors (falls back to software rendering)
try {
  const proc = fs.readFileSync('/proc/version', 'utf-8');
  if (proc.toLowerCase().includes('microsoft') || proc.toLowerCase().includes('wsl')) {
    app.disableHardwareAcceleration();
  }
} catch { /* not WSL */ }

// ============================================================================
// Configuration & Path Detection
// ============================================================================

/**
 * Detect the environment and return the appropriate workspace path.
 * Supports: native Windows, WSL, Linux, macOS.
 */
function detectWorkspacePath() {
  const platform = os.platform();

  // Check if running inside WSL
  if (platform === 'linux') {
    try {
      const release = fs.readFileSync('/proc/version', 'utf-8');
      if (release.toLowerCase().includes('microsoft') || release.toLowerCase().includes('wsl')) {
        // Running in WSL - use the WSL home path
        const wslHome = os.homedir();
        const wslPath = path.join(wslHome, '.gaia', 'workspace');
        if (fs.existsSync(wslPath)) {
          return wslPath;
        }
        // Also check the Windows-accessible WSL path
        // For Windows apps accessing WSL: \\wsl.localhost\Ubuntu-24.04\home\user\.gaia\workspace
        return wslPath; // Default to WSL native path
      }
    } catch (e) {
      // Not WSL, regular Linux
    }
  }

  // Check if running on Windows but workspace is in WSL
  if (platform === 'win32') {
    // Try WSL path first (common for WSL2 users running Electron from Windows)
    const wslDistros = ['Ubuntu-24.04', 'Ubuntu-22.04', 'Ubuntu', 'Debian'];
    for (const distro of wslDistros) {
      const wslPath = `\\\\wsl.localhost\\${distro}\\home`;
      try {
        if (fs.existsSync(wslPath)) {
          // Find the user home directory
          const homes = fs.readdirSync(wslPath);
          for (const home of homes) {
            const workspacePath = path.join(wslPath, home, '.gaia', 'workspace');
            if (fs.existsSync(workspacePath)) {
              return workspacePath;
            }
          }
        }
      } catch (e) {
        // WSL path not accessible
      }
    }

    // Fall back to Windows native path
    return path.join(os.homedir(), '.gaia', 'workspace');
  }

  // Default for Linux/macOS
  return path.join(os.homedir(), '.gaia', 'workspace');
}

const DEFAULT_WORKSPACE = detectWorkspacePath();

const KNOWN_DATABASES = [
  { name: 'memory.db', label: 'Memory (Session Cache)', description: 'File cache and tool results' },
  { name: 'knowledge.db', label: 'Knowledge (Insights)', description: 'Insights, preferences, learnings' },
  { name: 'tools.db', label: 'Tools (Registry)', description: 'Tool registry with FTS5 search' },
  { name: 'skills.db', label: 'Skills (Workflows)', description: 'Learned workflow patterns' },
  { name: 'agents.db', label: 'Agents (Specialists)', description: 'Specialist agent registry' },
  { name: 'plan.db', label: 'Plan (Task Tree)', description: 'Hierarchical task tree' },
  { name: 'logs.db', label: 'Logs (Runtime)', description: 'Runtime logs with FTS5 search' },
];

// ============================================================================
// Database Manager
// ============================================================================

let Database;
try {
  Database = require('better-sqlite3');
} catch (err) {
  console.error('better-sqlite3 not available. Install with: npm install better-sqlite3');
  Database = null;
}

/**
 * Manages SQLite database connections with connection pooling.
 * Detects file recreation (inode change) and automatically reopens connections,
 * which handles the case where Python deletes and recreates a .db file while
 * the Electron app holds an open connection to the old inode.
 */
class DatabaseManager {
  constructor() {
    this.connections = new Map(); // key -> { db, ino }
  }

  /**
   * Get or create a database connection.
   * If the file has been recreated (different inode), closes the old connection
   * and opens a fresh one so we always read from the current file.
   * @param {string} dbPath - Absolute path to the .db file
   * @param {boolean} readOnly - Open in read-only mode
   * @returns {import('better-sqlite3').Database}
   */
  getConnection(dbPath, readOnly = false) {
    const key = `${dbPath}:${readOnly ? 'ro' : 'rw'}`;
    if (this.connections.has(key)) {
      const { db, ino } = this.connections.get(key);
      try {
        const stat = fs.statSync(dbPath);
        if (stat.ino === ino) return db; // Same file — reuse connection
        // File was recreated (different inode) — close stale connection
        try { db.close(); } catch (e) { /* ignore */ }
        this.connections.delete(key);
      } catch (e) {
        return db; // Can't stat — use cached connection
      }
    }

    if (!Database) {
      throw new Error('better-sqlite3 is not installed');
    }

    if (!fs.existsSync(dbPath)) {
      throw new Error(`Database not found: ${dbPath}`);
    }

    const db = new Database(dbPath, { readonly: readOnly });
    // Only set journal_mode on writable connections — read-only connections
    // cannot execute PRAGMA journal_mode = WAL and will throw.
    if (!readOnly) {
      db.pragma('journal_mode = WAL');
    }
    db.pragma('busy_timeout = 5000');

    let ino = 0;
    try { ino = fs.statSync(dbPath).ino; } catch (e) { /* ignore */ }
    this.connections.set(key, { db, ino });
    return db;
  }

  /**
   * Close a specific connection.
   * @param {string} dbPath
   */
  closeConnection(dbPath) {
    for (const [key, { db }] of this.connections.entries()) {
      if (key.startsWith(dbPath)) {
        try { db.close(); } catch (e) { /* ignore */ }
        this.connections.delete(key);
      }
    }
  }

  /**
   * Close all connections.
   */
  closeAll() {
    for (const [, { db }] of this.connections.entries()) {
      try { db.close(); } catch (e) { /* ignore */ }
    }
    this.connections.clear();
  }
}

const dbManager = new DatabaseManager();

// ============================================================================
// IPC Handlers
// ============================================================================

function setupIpcHandlers() {

  // --- Database Discovery ---

  ipcMain.handle('db:listDatabases', async (event, workspacePath) => {
    const dir = workspacePath || DEFAULT_WORKSPACE;
    if (!fs.existsSync(dir)) {
      return { success: false, error: `Workspace not found: ${dir}`, databases: [] };
    }

    // Helper: get the effective mtime for a SQLite db, accounting for WAL files.
    // In WAL mode, the .db-wal file receives all recent writes and the main .db
    // mtime is only updated on checkpoint, which can be minutes later. We use
    // the most recent mtime across .db, .db-wal, and .db-shm to show accurate
    // "last modified" timestamps.
    function dbLastModified(dbPath) {
      let latest = null;
      for (const suffix of ['', '-wal', '-shm']) {
        const p = dbPath + suffix;
        try {
          if (fs.existsSync(p)) {
            const t = fs.statSync(p).mtime;
            if (!latest || t > latest) latest = t;
          }
        } catch { /* ignore */ }
      }
      return latest ? latest.toISOString() : null;
    }

    const databases = [];
    for (const known of KNOWN_DATABASES) {
      const fullPath = path.join(dir, known.name);
      const exists = fs.existsSync(fullPath);
      let sizeBytes = 0;
      let lastModified = null;
      if (exists) {
        const stat = fs.statSync(fullPath);
        sizeBytes = stat.size;
        lastModified = dbLastModified(fullPath);
      }
      databases.push({
        ...known,
        path: fullPath,
        exists,
        sizeBytes,
        lastModified,
      });
    }

    // Also scan for any other .db files in the workspace
    try {
      const files = fs.readdirSync(dir);
      for (const file of files) {
        if (file.endsWith('.db') && !KNOWN_DATABASES.find(k => k.name === file)) {
          const fullPath = path.join(dir, file);
          const stat = fs.statSync(fullPath);
          databases.push({
            name: file,
            label: file.replace('.db', ''),
            description: 'Custom database',
            path: fullPath,
            exists: true,
            sizeBytes: stat.size,
            lastModified: dbLastModified(fullPath),
          });
        }
      }
    } catch (e) { /* ignore scan errors */ }

    return { success: true, databases, workspacePath: dir };
  });

  // --- Table Operations ---

  ipcMain.handle('db:getTables', async (event, dbPath) => {
    try {
      const db = dbManager.getConnection(dbPath, true);
      const tables = db.prepare(`
        SELECT name, type FROM sqlite_master
        WHERE type IN ('table', 'view')
        AND name NOT LIKE 'sqlite_%'
        ORDER BY type DESC, name ASC
      `).all();

      const result = [];
      for (const table of tables) {
        // Check if it's an FTS5 table
        const isFts = table.name.endsWith('_fts');

        // Get row count
        let rowCount = 0;
        try {
          const countRow = db.prepare(`SELECT COUNT(*) as count FROM "${table.name}"`).get();
          rowCount = countRow.count;
        } catch (e) {
          // FTS tables may error on COUNT, skip
        }

        // Get column info
        let columns = [];
        try {
          const colInfo = db.prepare(`PRAGMA table_info("${table.name}")`).all();
          columns = colInfo.map(c => ({
            name: c.name,
            type: c.type || 'TEXT',
            notNull: c.notnull === 1,
            defaultValue: c.dflt_value,
            primaryKey: c.pk === 1,
          }));
        } catch (e) {
          // FTS5 virtual tables don't support PRAGMA table_info
          // Try to get columns from a sample row
          try {
            const sampleStmt = db.prepare(`SELECT * FROM "${table.name}" LIMIT 1`);
            const sample = sampleStmt.get();
            if (sample) {
              columns = Object.keys(sample).map(name => ({
                name,
                type: 'TEXT',
                notNull: false,
                defaultValue: null,
                primaryKey: false,
              }));
            }
          } catch (e2) { /* ignore */ }
        }

        result.push({
          name: table.name,
          type: table.type,
          isFts,
          rowCount,
          columns,
        });
      }

      return { success: true, tables: result };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // --- Query Table with Pagination ---

  ipcMain.handle('db:queryTable', async (event, dbPath, tableName, options = {}) => {
    try {
      const db = dbManager.getConnection(dbPath, true);
      const {
        page = 1,
        limit = 50,
        sortColumn = null,
        sortDirection = 'ASC',
        filterColumn = null,
        filterValue = null,
      } = options;

      const offset = (page - 1) * limit;

      // Build query
      let query = `SELECT * FROM "${tableName}"`;
      const params = [];

      // Filter
      if (filterColumn && filterValue) {
        query += ` WHERE "${filterColumn}" LIKE ?`;
        params.push(`%${filterValue}%`);
      }

      // Sort
      if (sortColumn) {
        const dir = sortDirection === 'DESC' ? 'DESC' : 'ASC';
        query += ` ORDER BY "${sortColumn}" ${dir}`;
      }

      // Pagination
      query += ` LIMIT ? OFFSET ?`;
      params.push(limit, offset);

      const rows = db.prepare(query).all(...params);

      // Get total count
      let countQuery = `SELECT COUNT(*) as total FROM "${tableName}"`;
      const countParams = [];
      if (filterColumn && filterValue) {
        countQuery += ` WHERE "${filterColumn}" LIKE ?`;
        countParams.push(`%${filterValue}%`);
      }
      const totalRow = db.prepare(countQuery).get(...countParams);
      const totalRows = totalRow.total;
      const totalPages = Math.ceil(totalRows / limit);

      return {
        success: true,
        rows,
        pagination: { page, limit, totalRows, totalPages },
      };
    } catch (err) {
      return { success: false, error: err.message, rows: [], pagination: {} };
    }
  });

  // --- Execute Custom SQL ---

  ipcMain.handle('db:executeSQL', async (event, dbPath, sql, readOnly = false) => {
    try {
      const db = dbManager.getConnection(dbPath, readOnly);
      const trimmed = sql.trim();
      const isSelect = /^(SELECT|PRAGMA|EXPLAIN|WITH)\b/i.test(trimmed);

      if (isSelect) {
        const startTime = Date.now();
        const rows = db.prepare(trimmed).all();
        const duration = Date.now() - startTime;
        const columns = rows.length > 0 ? Object.keys(rows[0]) : [];
        return {
          success: true,
          type: 'query',
          rows,
          columns,
          rowCount: rows.length,
          duration,
        };
      } else {
        if (readOnly) {
          return { success: false, error: 'Database is in read-only mode. Disable read-only to execute write operations.' };
        }
        const startTime = Date.now();
        const info = db.prepare(trimmed).run();
        const duration = Date.now() - startTime;
        return {
          success: true,
          type: 'statement',
          changes: info.changes,
          lastInsertRowid: info.lastInsertRowid,
          duration,
        };
      }
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // --- Row Operations ---

  ipcMain.handle('db:updateRow', async (event, dbPath, tableName, primaryKey, updates) => {
    try {
      const db = dbManager.getConnection(dbPath, false);

      const setClauses = [];
      const values = [];
      for (const [col, val] of Object.entries(updates)) {
        setClauses.push(`"${col}" = ?`);
        values.push(val);
      }

      // Determine primary key column
      const pkCol = primaryKey.column;
      const pkVal = primaryKey.value;
      values.push(pkVal);

      const sql = `UPDATE "${tableName}" SET ${setClauses.join(', ')} WHERE "${pkCol}" = ?`;
      const info = db.prepare(sql).run(...values);

      return { success: true, changes: info.changes };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle('db:deleteRow', async (event, dbPath, tableName, primaryKey) => {
    try {
      const db = dbManager.getConnection(dbPath, false);
      const sql = `DELETE FROM "${tableName}" WHERE "${primaryKey.column}" = ?`;
      const info = db.prepare(sql).run(primaryKey.value);
      return { success: true, changes: info.changes };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle('db:clearTable', async (event, dbPath, tableName) => {
    try {
      const db = dbManager.getConnection(dbPath, false);
      const info = db.prepare(`DELETE FROM "${tableName}"`).run();
      return { success: true, changes: info.changes };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle('db:insertRow', async (event, dbPath, tableName, data) => {
    try {
      const db = dbManager.getConnection(dbPath, false);
      const columns = Object.keys(data);
      const placeholders = columns.map(() => '?').join(', ');
      const values = Object.values(data);

      const sql = `INSERT INTO "${tableName}" (${columns.map(c => `"${c}"`).join(', ')}) VALUES (${placeholders})`;
      const info = db.prepare(sql).run(...values);
      return { success: true, lastInsertRowid: info.lastInsertRowid };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // --- Schema ---

  ipcMain.handle('db:getSchema', async (event, dbPath) => {
    try {
      const db = dbManager.getConnection(dbPath, true);
      const schemas = db.prepare(`
        SELECT name, sql FROM sqlite_master
        WHERE sql IS NOT NULL
        ORDER BY type DESC, name ASC
      `).all();

      return { success: true, schemas };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // --- FTS5 Search ---

  ipcMain.handle('db:searchFTS5', async (event, dbPath, ftsTable, query, options = {}) => {
    try {
      const db = dbManager.getConnection(dbPath, true);
      const { limit = 100 } = options;

      // Sanitize query for FTS5
      const sanitized = query.replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
      if (!sanitized) {
        return { success: true, rows: [], rowCount: 0 };
      }
      const ftsQuery = sanitized.split(' ').join(' OR ');

      // Check if the FTS table has a content= reference
      const schemaRow = db.prepare(
        `SELECT sql FROM sqlite_master WHERE name = ?`
      ).get(ftsTable);

      let rows;
      if (schemaRow && schemaRow.sql && schemaRow.sql.includes('content=')) {
        // Content-synced FTS table - join with the content table
        const contentMatch = schemaRow.sql.match(/content=(\w+)/);
        if (contentMatch) {
          const contentTable = contentMatch[1];
          rows = db.prepare(`
            SELECT c.* FROM "${contentTable}" c
            JOIN "${ftsTable}" f ON c.rowid = f.rowid
            WHERE "${ftsTable}" MATCH ?
            ORDER BY rank
            LIMIT ?
          `).all(ftsQuery, limit);
        } else {
          rows = db.prepare(`
            SELECT *, rank FROM "${ftsTable}" WHERE "${ftsTable}" MATCH ? ORDER BY rank LIMIT ?
          `).all(ftsQuery, limit);
        }
      } else {
        rows = db.prepare(`
          SELECT *, rank FROM "${ftsTable}" WHERE "${ftsTable}" MATCH ? ORDER BY rank LIMIT ?
        `).all(ftsQuery, limit);
      }

      return { success: true, rows, rowCount: rows.length };
    } catch (err) {
      return { success: false, error: err.message, rows: [] };
    }
  });

  // --- Export ---

  ipcMain.handle('db:exportTable', async (event, dbPath, tableName, format) => {
    try {
      const db = dbManager.getConnection(dbPath, true);
      const rows = db.prepare(`SELECT * FROM "${tableName}"`).all();

      let content;
      let ext;
      if (format === 'csv') {
        ext = 'csv';
        if (rows.length === 0) {
          content = '';
        } else {
          const headers = Object.keys(rows[0]);
          const csvRows = rows.map(row =>
            headers.map(h => {
              const val = row[h];
              if (val === null || val === undefined) return '';
              const str = String(val);
              if (str.includes(',') || str.includes('"') || str.includes('\n')) {
                return `"${str.replace(/"/g, '""')}"`;
              }
              return str;
            }).join(',')
          );
          content = [headers.join(','), ...csvRows].join('\n');
        }
      } else {
        ext = 'json';
        content = JSON.stringify(rows, null, 2);
      }

      const result = await dialog.showSaveDialog({
        title: `Export ${tableName}`,
        defaultPath: `${tableName}.${ext}`,
        filters: [
          format === 'csv'
            ? { name: 'CSV Files', extensions: ['csv'] }
            : { name: 'JSON Files', extensions: ['json'] },
        ],
      });

      if (!result.canceled && result.filePath) {
        fs.writeFileSync(result.filePath, content, 'utf-8');
        return { success: true, filePath: result.filePath, rowCount: rows.length };
      }
      return { success: false, error: 'Export cancelled' };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // --- Backup ---

  ipcMain.handle('db:backup', async (event, dbPath) => {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const baseName = path.basename(dbPath, '.db');
      const backupName = `${baseName}_backup_${timestamp}.db`;

      const result = await dialog.showSaveDialog({
        title: `Backup ${path.basename(dbPath)}`,
        defaultPath: backupName,
        filters: [{ name: 'SQLite Database', extensions: ['db'] }],
      });

      if (!result.canceled && result.filePath) {
        fs.copyFileSync(dbPath, result.filePath);
        return { success: true, filePath: result.filePath };
      }
      return { success: false, error: 'Backup cancelled' };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // --- Utility ---

  ipcMain.handle('db:getWorkspacePath', async () => {
    return DEFAULT_WORKSPACE;
  });

  ipcMain.handle('db:selectWorkspace', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select GAIA Workspace Directory',
      properties: ['openDirectory'],
      defaultPath: DEFAULT_WORKSPACE,
    });
    if (!result.canceled && result.filePaths.length > 0) {
      return { success: true, path: result.filePaths[0] };
    }
    return { success: false };
  });

  ipcMain.handle('db:selectDatabaseFile', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Open SQLite Database File',
      properties: ['openFile'],
      filters: [
        { name: 'SQLite Database', extensions: ['db', 'sqlite', 'sqlite3'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (!result.canceled && result.filePaths.length > 0) {
      return { success: true, path: result.filePaths[0] };
    }
    return { success: false };
  });

  ipcMain.handle('db:closeConnection', async (event, dbPath) => {
    dbManager.closeConnection(dbPath);
    return { success: true };
  });
}

// ============================================================================
// Window Management
// ============================================================================

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    title: 'GAIA DB Dashboard',
    backgroundColor: '#0f1117',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // Needed for better-sqlite3
    },
  });

  // In development, load from Vite dev server. In production, load the built dist.
  const isDev = process.env.NODE_ENV === 'development' || process.env.VITE_DEV_SERVER_URL;
  if (isDev) {
    const devUrl = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
    mainWindow.loadURL(devUrl);
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  // Build menu
  const menuTemplate = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Workspace...',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            mainWindow.webContents.send('menu:openWorkspace');
          },
        },
        { type: 'separator' },
        {
          label: 'Refresh',
          accelerator: 'CmdOrCtrl+R',
          click: () => {
            mainWindow.webContents.send('menu:refresh');
          },
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { role: 'resetZoom' },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(menuTemplate);
  Menu.setApplicationMenu(menu);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ============================================================================
// App Lifecycle
// ============================================================================

app.whenReady().then(() => {
  setupIpcHandlers();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  dbManager.closeAll();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  dbManager.closeAll();
});
