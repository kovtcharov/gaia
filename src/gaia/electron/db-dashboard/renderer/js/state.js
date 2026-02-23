// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Application State
 *
 * Central state store for the DB Dashboard.
 * All modules read/write from this shared state object.
 */

const AppState = {
  // Workspace
  workspacePath: '',

  // Databases
  databases: [],       // Array of database info objects from listDatabases
  currentDbPath: null,  // Currently selected database path
  currentDbName: null,  // Currently selected database filename

  // Tables
  tables: [],          // Tables in the current database
  currentTable: null,  // Currently selected table name
  currentTableInfo: null, // Column info for the current table

  // Data grid
  rows: [],            // Current page of data
  pagination: {
    page: 1,
    limit: 50,
    totalRows: 0,
    totalPages: 0,
  },
  sortColumn: null,
  sortDirection: 'ASC',
  filterColumn: null,
  filterValue: null,

  // Read-only mode
  readOnly: true,

  // Active tab
  activeTab: 'dashboard',

  // Active database tab (which database tab is selected, or 'dashboard')
  activeDatabaseTab: 'dashboard',

  // Pending operations (for modals)
  pendingDelete: null,    // { dbPath, table, primaryKey }
  pendingEdit: null,      // { dbPath, table, primaryKey, column, value }

  // Auto-refresh settings
  autoRefresh: {
    enabled: true,
    interval: 2000,       // milliseconds
    timerId: null,        // setInterval ID
    lastUpdate: null,     // Date object of last update
    lastUpdateDisplay: null, // setInterval for display updates
    isPaused: false,      // paused during editing
  },

  // File watcher state
  fileWatcher: {
    lastModTimes: {},     // { dbPath: mtime } for change detection
    pollTimerId: null,    // polling interval ID
  },

  // Dashboard data cache
  dashboardData: {
    totalSize: 0,
    dbStats: [],
    recentErrors: [],
    activeTasks: [],
    recentInsights: [],
    contextWarnings: [],
    topTools: [],
    errorTrend: [],
    contextUsage: [],
    lastActivity: null,
  },

  // Change tracking for row highlighting
  changeTracking: {
    changedRowIds: new Set(),  // Set of row identifiers that recently changed
    highlightTimer: null,       // Timer to clear highlights
  },

  // Notification preferences
  notifications: {
    autoScrollToNew: false,  // Whether to auto-scroll when new rows appear
    showChangeToasts: true,  // Whether to show toast when data changes
  },

  // History log: circular buffer of recent database operations
  // Each entry: { id, timestamp, database, operation, table, rowCount, details }
  historyLog: [],
};
