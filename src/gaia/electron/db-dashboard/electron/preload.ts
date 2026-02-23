// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Preload Script (TypeScript)
 *
 * TypeScript source for the Electron preload script.
 * The actual runtime preload.js is at the project root (db-dashboard/preload.js).
 *
 * This exposes a type-safe DatabaseAPI to the renderer process via contextBridge.
 */

import { contextBridge, ipcRenderer } from 'electron';

export interface DatabaseAPI {
  // Database Discovery
  listDatabases(workspacePath?: string): Promise<unknown>;
  getWorkspacePath(): Promise<string>;
  selectWorkspace(): Promise<{ success: boolean; path?: string }>;
  selectDatabaseFile(): Promise<{ success: boolean; path?: string }>;

  // Table Operations
  getTables(dbPath: string): Promise<unknown>;
  queryTable(dbPath: string, tableName: string, options?: unknown): Promise<unknown>;

  // SQL Console
  executeSQL(dbPath: string, sql: string, readOnly?: boolean): Promise<unknown>;

  // Row CRUD
  updateRow(dbPath: string, tableName: string, primaryKey: unknown, updates: unknown): Promise<unknown>;
  deleteRow(dbPath: string, tableName: string, primaryKey: unknown): Promise<unknown>;
  insertRow(dbPath: string, tableName: string, data: unknown): Promise<unknown>;

  // Schema
  getSchema(dbPath: string): Promise<unknown>;

  // FTS5
  searchFTS5(dbPath: string, ftsTable: string, query: string, options?: unknown): Promise<unknown>;

  // Export & Backup
  exportTable(dbPath: string, tableName: string, format: string): Promise<unknown>;
  backup(dbPath: string): Promise<unknown>;
  closeConnection(dbPath: string): Promise<{ success: boolean }>;

  // Menu Events
  onMenuEvent(channel: string, callback: (...args: unknown[]) => void): void;
}

const api: DatabaseAPI = {
  // Database Discovery
  listDatabases: (workspacePath) => ipcRenderer.invoke('db:listDatabases', workspacePath),
  getWorkspacePath: () => ipcRenderer.invoke('db:getWorkspacePath'),
  selectWorkspace: () => ipcRenderer.invoke('db:selectWorkspace'),
  selectDatabaseFile: () => ipcRenderer.invoke('db:selectDatabaseFile'),

  // Table Operations
  getTables: (dbPath) => ipcRenderer.invoke('db:getTables', dbPath),
  queryTable: (dbPath, tableName, options) => ipcRenderer.invoke('db:queryTable', dbPath, tableName, options),

  // SQL Console
  executeSQL: (dbPath, sql, readOnly) => ipcRenderer.invoke('db:executeSQL', dbPath, sql, readOnly),

  // Row CRUD
  updateRow: (dbPath, tableName, primaryKey, updates) =>
    ipcRenderer.invoke('db:updateRow', dbPath, tableName, primaryKey, updates),
  deleteRow: (dbPath, tableName, primaryKey) =>
    ipcRenderer.invoke('db:deleteRow', dbPath, tableName, primaryKey),
  insertRow: (dbPath, tableName, data) =>
    ipcRenderer.invoke('db:insertRow', dbPath, tableName, data),

  // Schema
  getSchema: (dbPath) => ipcRenderer.invoke('db:getSchema', dbPath),

  // FTS5
  searchFTS5: (dbPath, ftsTable, query, options) =>
    ipcRenderer.invoke('db:searchFTS5', dbPath, ftsTable, query, options),

  // Export & Backup
  exportTable: (dbPath, tableName, format) =>
    ipcRenderer.invoke('db:exportTable', dbPath, tableName, format),
  backup: (dbPath) => ipcRenderer.invoke('db:backup', dbPath),
  closeConnection: (dbPath) => ipcRenderer.invoke('db:closeConnection', dbPath),

  // Menu Events
  onMenuEvent: (channel, callback) => {
    const validChannels = ['menu:openWorkspace', 'menu:refresh'];
    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, (_event, ...args) => callback(...args));
    }
  },
};

contextBridge.exposeInMainWorld('dbAPI', api);
