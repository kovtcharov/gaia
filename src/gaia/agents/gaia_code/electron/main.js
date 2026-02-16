// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA Code Electron Main Process
 *
 * Comprehensive Electron wrapper for the GAIA Code autonomous agent.
 * Features:
 * - IPC bridge to Python backend (WebSocket)
 * - Real-time state synchronization
 * - Database inspector via SQLite queries
 * - Audit log streaming
 * - Checkpoint management
 */

const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');

// Configuration
const IPC_PORT = parseInt(process.env.GAIA_CODE_IPC_PORT || '9720', 10);
const WINDOW_TITLE = 'GAIA Code - Autonomous Coding Agent';
const DEV_MODE = process.env.GAIA_APP_MODE === 'development' || process.argv.includes('--dev');

let mainWindow = null;
let ipcProcess = null;
let wsConnection = null;

// ============================================================================
// Window Management
// ============================================================================

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1920,
    height: 1080,
    minWidth: 1200,
    minHeight: 700,
    title: WINDOW_TITLE,
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    backgroundColor: '#0a0a0f',
    show: false,
    center: true,
    titleBarStyle: 'hiddenInset',
    frame: process.platform !== 'darwin',
  });

  // Load the renderer
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // Show when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.center();
    mainWindow.show();
    mainWindow.focus();
  });

  // Open DevTools in dev mode
  if (DEV_MODE) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopIPCServer();
  });
}

// ============================================================================
// IPC Server Management (Python Backend)
// ============================================================================

function startIPCServer() {
  const pythonScript = path.join(__dirname, 'gaia_code_ipc.py');

  ipcProcess = spawn('python', [pythonScript, '--port', String(IPC_PORT)], {
    cwd: path.join(__dirname, '..'),
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      GAIA_CODE_IPC_PORT: String(IPC_PORT),
    },
  });

  ipcProcess.stdout.on('data', (data) => {
    const msg = data.toString().trim();
    console.log(`[IPC Server] ${msg}`);
    if (mainWindow) {
      mainWindow.webContents.send('ipc-server-log', msg);
    }
  });

  ipcProcess.stderr.on('data', (data) => {
    console.error(`[IPC Server Error] ${data.toString().trim()}`);
  });

  ipcProcess.on('close', (code) => {
    console.log(`[IPC Server] Exited with code ${code}`);
    ipcProcess = null;
  });
}

function stopIPCServer() {
  if (ipcProcess) {
    ipcProcess.kill();
    ipcProcess = null;
  }
}

// ============================================================================
// IPC Handlers (Renderer <-> Main Process)
// ============================================================================

function setupIPCHandlers() {
  // Send chat message to agent
  ipcMain.handle('agent:send-message', async (event, message) => {
    return sendToBackend('chat', { message });
  });

  // Get agent status
  ipcMain.handle('agent:get-status', async () => {
    return sendToBackend('status', {});
  });

  // Get task plan
  ipcMain.handle('agent:get-plan', async () => {
    return sendToBackend('plan', {});
  });

  // Get quality gate results
  ipcMain.handle('agent:get-quality-gates', async () => {
    return sendToBackend('quality_gates', {});
  });

  // Database inspector queries
  ipcMain.handle('db:query', async (event, { database, query }) => {
    return sendToBackend('db_query', { database, query });
  });

  // Get database tables
  ipcMain.handle('db:tables', async (event, database) => {
    return sendToBackend('db_tables', { database });
  });

  // Get database schema
  ipcMain.handle('db:schema', async (event, { database, table }) => {
    return sendToBackend('db_schema', { database, table });
  });

  // Browse database table
  ipcMain.handle('db:browse', async (event, { database, table, limit, offset }) => {
    return sendToBackend('db_browse', { database, table, limit: limit || 50, offset: offset || 0 });
  });

  // Get audit log
  ipcMain.handle('audit:get-log', async (event, { limit, offset, filter }) => {
    return sendToBackend('audit_log', { limit: limit || 100, offset: offset || 0, filter: filter || null });
  });

  // Codebase index
  ipcMain.handle('codebase:get-index', async () => {
    return sendToBackend('codebase_index', {});
  });

  // Get specialists
  ipcMain.handle('agents:get-specialists', async () => {
    return sendToBackend('specialists', {});
  });

  // Get performance metrics
  ipcMain.handle('metrics:get', async () => {
    return sendToBackend('metrics', {});
  });

  // Checkpoint operations
  ipcMain.handle('checkpoint:list', async () => {
    return sendToBackend('checkpoint_list', {});
  });

  ipcMain.handle('checkpoint:create', async () => {
    return sendToBackend('checkpoint_create', {});
  });

  ipcMain.handle('checkpoint:restore', async (event, checkpointId) => {
    return sendToBackend('checkpoint_restore', { checkpoint_id: checkpointId });
  });

  // Call stack
  ipcMain.handle('agent:get-call-stack', async () => {
    return sendToBackend('call_stack', {});
  });

  // Message queue
  ipcMain.handle('agent:get-messages', async () => {
    return sendToBackend('messages', {});
  });

  // Open file in external editor
  ipcMain.handle('file:open-external', async (event, filePath) => {
    shell.openPath(filePath);
    return { success: true };
  });

  // Window controls
  ipcMain.on('window:minimize', () => mainWindow?.minimize());
  ipcMain.on('window:maximize', () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
  });
  ipcMain.on('window:close', () => mainWindow?.close());
}

// ============================================================================
// Backend Communication (HTTP to Python IPC Server)
// ============================================================================

async function sendToBackend(action, payload) {
  try {
    const http = require('http');

    return new Promise((resolve, reject) => {
      const data = JSON.stringify({ action, ...payload });

      const options = {
        hostname: '127.0.0.1',
        port: IPC_PORT,
        path: '/api',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
        timeout: 30000,
      };

      const req = http.request(options, (res) => {
        let body = '';
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            resolve({ error: 'Invalid JSON response', raw: body });
          }
        });
      });

      req.on('error', (err) => {
        resolve({ error: `Backend connection failed: ${err.message}`, offline: true });
      });

      req.on('timeout', () => {
        req.destroy();
        resolve({ error: 'Backend request timed out', offline: true });
      });

      req.write(data);
      req.end();
    });
  } catch (err) {
    return { error: `Backend error: ${err.message}`, offline: true };
  }
}

// ============================================================================
// App Lifecycle
// ============================================================================

try {
  if (require('electron-squirrel-startup')) {
    app.quit();
  }
} catch (error) {
  // electron-squirrel-startup not available
}

app.whenReady().then(() => {
  setupIPCHandlers();
  createWindow();

  // Start IPC server if not already running
  checkBackendConnection().then((connected) => {
    if (!connected) {
      console.log('[Main] Starting IPC server...');
      startIPCServer();
    } else {
      console.log('[Main] IPC server already running');
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopIPCServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

async function checkBackendConnection() {
  return new Promise((resolve) => {
    const client = new net.Socket();
    client.setTimeout(2000);
    client.on('connect', () => {
      client.destroy();
      resolve(true);
    });
    client.on('error', () => resolve(false));
    client.on('timeout', () => {
      client.destroy();
      resolve(false);
    });
    client.connect(IPC_PORT, '127.0.0.1');
  });
}

console.log(`[GAIA Code] Electron app starting...`);
console.log(`[GAIA Code] IPC Port: ${IPC_PORT}`);
console.log(`[GAIA Code] Dev Mode: ${DEV_MODE}`);
