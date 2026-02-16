// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA Code Preload Script
 *
 * Exposes a safe API to the renderer process via contextBridge.
 * All communication with the main process goes through this bridge.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gaiaCode', {
  // ========================================================================
  // Agent Interaction
  // ========================================================================

  /** Send a chat message to the agent */
  sendMessage: (message) => ipcRenderer.invoke('agent:send-message', message),

  /** Get current agent status */
  getStatus: () => ipcRenderer.invoke('agent:get-status'),

  /** Get the current task plan */
  getPlan: () => ipcRenderer.invoke('agent:get-plan'),

  /** Get quality gate results */
  getQualityGates: () => ipcRenderer.invoke('agent:get-quality-gates'),

  /** Get agent call stack */
  getCallStack: () => ipcRenderer.invoke('agent:get-call-stack'),

  /** Get pending messages */
  getMessages: () => ipcRenderer.invoke('agent:get-messages'),

  // ========================================================================
  // Database Inspector
  // ========================================================================

  /** Execute a SQL query against a database */
  dbQuery: (database, query) => ipcRenderer.invoke('db:query', { database, query }),

  /** Get list of tables in a database */
  dbTables: (database) => ipcRenderer.invoke('db:tables', database),

  /** Get schema for a table */
  dbSchema: (database, table) => ipcRenderer.invoke('db:schema', { database, table }),

  /** Browse a table with pagination */
  dbBrowse: (database, table, limit, offset) =>
    ipcRenderer.invoke('db:browse', { database, table, limit, offset }),

  // ========================================================================
  // Audit Log
  // ========================================================================

  /** Get audit log entries */
  getAuditLog: (limit, offset, filter) =>
    ipcRenderer.invoke('audit:get-log', { limit, offset, filter }),

  // ========================================================================
  // Codebase Index
  // ========================================================================

  /** Get codebase index data */
  getCodebaseIndex: () => ipcRenderer.invoke('codebase:get-index'),

  // ========================================================================
  // Specialists
  // ========================================================================

  /** Get specialist agent information */
  getSpecialists: () => ipcRenderer.invoke('agents:get-specialists'),

  // ========================================================================
  // Performance Metrics
  // ========================================================================

  /** Get performance metrics */
  getMetrics: () => ipcRenderer.invoke('metrics:get'),

  // ========================================================================
  // Checkpoint Manager
  // ========================================================================

  /** List all checkpoints */
  listCheckpoints: () => ipcRenderer.invoke('checkpoint:list'),

  /** Create a new checkpoint */
  createCheckpoint: () => ipcRenderer.invoke('checkpoint:create'),

  /** Restore from a checkpoint */
  restoreCheckpoint: (id) => ipcRenderer.invoke('checkpoint:restore', id),

  // ========================================================================
  // File Operations
  // ========================================================================

  /** Open a file in external editor */
  openFile: (filePath) => ipcRenderer.invoke('file:open-external', filePath),

  // ========================================================================
  // Window Controls
  // ========================================================================

  minimizeWindow: () => ipcRenderer.send('window:minimize'),
  maximizeWindow: () => ipcRenderer.send('window:maximize'),
  closeWindow: () => ipcRenderer.send('window:close'),

  // ========================================================================
  // Event Listeners (from Main Process)
  // ========================================================================

  /** Listen for real-time updates from IPC server */
  onServerLog: (callback) => {
    ipcRenderer.on('ipc-server-log', (event, msg) => callback(msg));
  },

  /** Listen for agent state changes */
  onStateChange: (callback) => {
    ipcRenderer.on('agent-state-change', (event, state) => callback(state));
  },

  /** Listen for new audit entries */
  onAuditEntry: (callback) => {
    ipcRenderer.on('audit-entry', (event, entry) => callback(entry));
  },

  /** Listen for quality gate updates */
  onQualityGateUpdate: (callback) => {
    ipcRenderer.on('quality-gate-update', (event, data) => callback(data));
  },

  /** Remove all listeners for a channel */
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  },
});
