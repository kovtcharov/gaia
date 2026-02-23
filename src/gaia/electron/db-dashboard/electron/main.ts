// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Electron Main Process (TypeScript)
 *
 * This is a TypeScript source file. For Electron to load it, you must
 * either compile it to JS (via `tsc`) or load the JS version directly.
 *
 * The actual runtime main.js is at the project root (db-dashboard/main.js).
 * This file serves as the canonical TypeScript source for documentation
 * and type-checking purposes.
 *
 * All IPC handlers and database management remain in main.js since
 * Electron loads CommonJS modules and better-sqlite3 is a native addon.
 */

import { app, BrowserWindow, ipcMain, dialog, Menu, MenuItemConstructorOptions } from 'electron';
import path from 'path';
import fs from 'fs';
import os from 'os';

// ============================================================================
// Types
// ============================================================================

interface DatabaseEntry {
  name: string;
  label: string;
  description: string;
  path?: string;
  exists?: boolean;
  sizeBytes?: number;
  lastModified?: string | null;
}

interface QueryOptions {
  page?: number;
  limit?: number;
  sortColumn?: string | null;
  sortDirection?: 'ASC' | 'DESC';
  filterColumn?: string | null;
  filterValue?: string | null;
}

// ============================================================================
// Configuration & Path Detection
// ============================================================================

function detectWorkspacePath(): string {
  const platform = os.platform();

  if (platform === 'linux') {
    try {
      const release = fs.readFileSync('/proc/version', 'utf-8');
      if (release.toLowerCase().includes('microsoft') || release.toLowerCase().includes('wsl')) {
        const wslHome = os.homedir();
        const wslPath = path.join(wslHome, '.gaia', 'workspace');
        return wslPath;
      }
    } catch {
      // Not WSL
    }
  }

  if (platform === 'win32') {
    const wslDistros = ['Ubuntu-24.04', 'Ubuntu-22.04', 'Ubuntu', 'Debian'];
    for (const distro of wslDistros) {
      const wslPath = `\\\\wsl.localhost\\${distro}\\home`;
      try {
        if (fs.existsSync(wslPath)) {
          const homes = fs.readdirSync(wslPath);
          for (const home of homes) {
            const workspacePath = path.join(wslPath, home, '.gaia', 'workspace');
            if (fs.existsSync(workspacePath)) {
              return workspacePath;
            }
          }
        }
      } catch {
        // WSL path not accessible
      }
    }
    return path.join(os.homedir(), '.gaia', 'workspace');
  }

  return path.join(os.homedir(), '.gaia', 'workspace');
}

const DEFAULT_WORKSPACE = detectWorkspacePath();

const KNOWN_DATABASES: DatabaseEntry[] = [
  { name: 'memory.db', label: 'Memory (Session Cache)', description: 'File cache and tool results' },
  { name: 'knowledge.db', label: 'Knowledge (Insights)', description: 'Insights, preferences, learnings' },
  { name: 'tools.db', label: 'Tools (Registry)', description: 'Tool registry with FTS5 search' },
  { name: 'skills.db', label: 'Skills (Workflows)', description: 'Learned workflow patterns' },
  { name: 'agents.db', label: 'Agents (Specialists)', description: 'Specialist agent registry' },
  { name: 'plan.db', label: 'Plan (Task Tree)', description: 'Hierarchical task tree' },
  { name: 'logs.db', label: 'Logs (Runtime)', description: 'Runtime logs with FTS5 search' },
];

// ============================================================================
// Note: Database Manager and IPC handlers remain in the root main.js
// because Electron loads CommonJS and better-sqlite3 is a native addon.
// This TypeScript file is for type reference only.
// ============================================================================

export {
  detectWorkspacePath,
  DEFAULT_WORKSPACE,
  KNOWN_DATABASES,
};

export type { DatabaseEntry, QueryOptions };
