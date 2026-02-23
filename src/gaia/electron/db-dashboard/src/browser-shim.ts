// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Browser-mode shim for window.dbAPI.
 *
 * When running outside of Electron (e.g., via `npm run dev` + dev-server.js),
 * this shim provides a fetch-based implementation of the dbAPI interface
 * that proxies requests to the dev-server HTTP API.
 *
 * In Electron mode, window.dbAPI is already set by the preload script,
 * so this shim is a no-op.
 */

import type { DatabaseAPI } from './types/database';

function createBrowserAPI(): DatabaseAPI {
  const post = async (action: string, body: Record<string, unknown> = {}): Promise<unknown> => {
    const res = await fetch(`/api/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return res.json();
  };

  return {
    listDatabases: (wp) => post('listDatabases', { workspacePath: wp }) as Promise<any>,
    getWorkspacePath: () => post('getWorkspacePath') as Promise<string>,
    selectWorkspace: async () => ({ success: false, path: undefined }),
    selectDatabaseFile: async () => ({ success: false, path: undefined }),
    getTables: (dbPath) => post('getTables', { dbPath }) as Promise<any>,
    queryTable: (dbPath, tableName, options) =>
      post('queryTable', { dbPath, tableName, options }) as Promise<any>,
    executeSQL: (dbPath, sql, readOnly) =>
      post('executeSQL', { dbPath, sql, readOnly }) as Promise<any>,
    updateRow: async () => ({ success: false, error: 'Write operations not available in browser mode' }),
    deleteRow: async () => ({ success: false, error: 'Write operations not available in browser mode' }),
    insertRow: async () => ({ success: false, error: 'Write operations not available in browser mode' }),
    getSchema: (dbPath) => post('getSchema', { dbPath }) as Promise<any>,
    searchFTS5: (dbPath, ftsTable, query, options) =>
      post('searchFTS5', { dbPath, ftsTable, query, options }) as Promise<any>,
    exportTable: async () => ({ success: false, error: 'Export not available in browser mode' }),
    backup: async () => ({ success: false, error: 'Backup not available in browser mode' }),
    closeConnection: async () => ({ success: true }),
    onMenuEvent: () => {},
  };
}

// Only inject if not already present (Electron preload sets it)
if (!window.dbAPI) {
  window.dbAPI = createBrowserAPI();
}
