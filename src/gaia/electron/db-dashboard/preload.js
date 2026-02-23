// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Preload Script
 *
 * Exposes a safe DatabaseAPI to the renderer process via contextBridge.
 * All database operations are forwarded to the main process via IPC.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('dbAPI', {

  // --- Database Discovery ---

  /** List all databases in the workspace directory */
  listDatabases: (workspacePath) =>
    ipcRenderer.invoke('db:listDatabases', workspacePath),

  /** Get the default workspace path */
  getWorkspacePath: () =>
    ipcRenderer.invoke('db:getWorkspacePath'),

  /** Open a directory picker to select workspace */
  selectWorkspace: () =>
    ipcRenderer.invoke('db:selectWorkspace'),

  /** Open a file picker to select a single database file */
  selectDatabaseFile: () =>
    ipcRenderer.invoke('db:selectDatabaseFile'),

  // --- Table Operations ---

  /** Get all tables and their metadata for a database */
  getTables: (dbPath) =>
    ipcRenderer.invoke('db:getTables', dbPath),

  /** Query a table with pagination, sorting, and filtering */
  queryTable: (dbPath, tableName, options) =>
    ipcRenderer.invoke('db:queryTable', dbPath, tableName, options),

  // --- SQL Console ---

  /** Execute arbitrary SQL on a database */
  executeSQL: (dbPath, sql, readOnly) =>
    ipcRenderer.invoke('db:executeSQL', dbPath, sql, readOnly),

  // --- Row CRUD ---

  /** Update a row by primary key */
  updateRow: (dbPath, tableName, primaryKey, updates) =>
    ipcRenderer.invoke('db:updateRow', dbPath, tableName, primaryKey, updates),

  /** Delete a row by primary key */
  deleteRow: (dbPath, tableName, primaryKey) =>
    ipcRenderer.invoke('db:deleteRow', dbPath, tableName, primaryKey),

  /** Insert a new row */
  insertRow: (dbPath, tableName, data) =>
    ipcRenderer.invoke('db:insertRow', dbPath, tableName, data),

  // --- Schema ---

  /** Get all CREATE TABLE statements for a database */
  getSchema: (dbPath) =>
    ipcRenderer.invoke('db:getSchema', dbPath),

  // --- FTS5 ---

  /** Search an FTS5 virtual table */
  searchFTS5: (dbPath, ftsTable, query, options) =>
    ipcRenderer.invoke('db:searchFTS5', dbPath, ftsTable, query, options),

  // --- Export & Backup ---

  /** Export a table as JSON or CSV */
  exportTable: (dbPath, tableName, format) =>
    ipcRenderer.invoke('db:exportTable', dbPath, tableName, format),

  /** Create a backup of a database */
  backup: (dbPath) =>
    ipcRenderer.invoke('db:backup', dbPath),

  /** Close a database connection */
  closeConnection: (dbPath) =>
    ipcRenderer.invoke('db:closeConnection', dbPath),

  // --- Menu Events ---

  /** Listen for menu events from main process */
  onMenuEvent: (channel, callback) => {
    const validChannels = ['menu:openWorkspace', 'menu:refresh'];
    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, (event, ...args) => callback(...args));
    }
  },
});
