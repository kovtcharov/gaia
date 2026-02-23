// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Main Application
 *
 * Wires together all components: state, dashboard, database tabs, table list,
 * data grid, SQL console, FTS search, modals, and auto-refresh.
 */

(function () {
  'use strict';

  // ========================================================================
  // Initialization
  // ========================================================================

  async function init() {
    // Initialize modals
    Modals.init();

    // Load default workspace path
    try {
      AppState.workspacePath = await window.dbAPI.getWorkspacePath();
    } catch (e) {
      console.warn('Could not get workspace path:', e);
    }

    // Display workspace path in header
    updateWorkspaceDisplay();

    // Load databases and build tabs
    await loadDatabases();

    // Bind all event listeners
    bindEvents();

    // Initialize auto-refresh
    AutoRefresh.init();
    AutoRefresh.start();

    // Load dashboard as default view
    await Dashboard.load();
  }

  // ========================================================================
  // Workspace Display
  // ========================================================================

  function updateWorkspaceDisplay() {
    const el = document.getElementById('workspace-path-display');
    if (el) {
      el.textContent = AppState.workspacePath || 'No workspace set';
      el.title = AppState.workspacePath || '';
    }
  }

  // ========================================================================
  // Database Loading and Tab Building
  // ========================================================================

  async function loadDatabases() {
    const result = await window.dbAPI.listDatabases(AppState.workspacePath);

    if (!result.success) {
      showToast(`Could not load databases: ${result.error}`, 'error');
      return;
    }

    AppState.databases = result.databases;
    AppState.workspacePath = result.workspacePath;
    updateWorkspaceDisplay();

    // Build database tabs
    buildDatabaseTabs(result.databases);
  }

  /**
   * Build the database tab bar from the list of databases.
   * @param {Array} databases
   */
  function buildDatabaseTabs(databases) {
    const tabBar = document.getElementById('db-tab-bar');
    if (!tabBar) return;

    // Keep the Dashboard tab, remove the rest
    const dashTab = tabBar.querySelector('[data-db="dashboard"]');
    tabBar.innerHTML = '';
    if (dashTab) {
      tabBar.appendChild(dashTab);
    }

    for (const db of databases) {
      if (!db.exists) continue;
      const btn = document.createElement('button');
      btn.className = 'db-tab';
      btn.dataset.db = db.path;
      btn.textContent = db.name.replace('.db', '');
      btn.title = `${db.name} (${formatBytes(db.sizeBytes)})`;
      btn.addEventListener('click', () => switchDatabaseTab(db.path));
      tabBar.appendChild(btn);
    }

    // Mark the active tab
    tabBar.querySelectorAll('.db-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.db === AppState.activeDatabaseTab);
    });
  }

  /**
   * Switch to a database tab (or back to dashboard).
   * @param {string} dbPathOrDashboard - 'dashboard' or a database file path
   */
  async function switchDatabaseTab(dbPathOrDashboard) {
    AppState.activeDatabaseTab = dbPathOrDashboard;

    // Update tab bar active state
    const tabBar = document.getElementById('db-tab-bar');
    tabBar.querySelectorAll('.db-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.db === dbPathOrDashboard);
    });

    if (dbPathOrDashboard === 'dashboard') {
      // Show dashboard, hide database content
      document.getElementById('dashboard-tab').classList.remove('hidden');
      document.getElementById('database-content').classList.add('hidden');
      document.getElementById('sidebar').classList.add('hidden');

      // Close any previous DB connection
      if (AppState.currentDbPath) {
        await window.dbAPI.closeConnection(AppState.currentDbPath);
      }
      AppState.currentDbPath = null;
      AppState.currentDbName = null;
      AppState.currentTable = null;

      await Dashboard.load();
    } else {
      // Show database content, hide dashboard
      document.getElementById('dashboard-tab').classList.add('hidden');
      document.getElementById('database-content').classList.remove('hidden');
      document.getElementById('sidebar').classList.remove('hidden');

      // Close previous connection if switching databases
      if (AppState.currentDbPath && AppState.currentDbPath !== dbPathOrDashboard) {
        await window.dbAPI.closeConnection(AppState.currentDbPath);
      }

      AppState.currentDbPath = dbPathOrDashboard;
      AppState.currentDbName = AppState.databases.find(d => d.path === dbPathOrDashboard)?.name || '';

      // Switch to data grid tab
      switchTab('data-grid');

      // Load tables
      await TableList.load();
    }
  }

  // ========================================================================
  // Event Binding
  // ========================================================================

  function bindEvents() {

    // --- Refresh ---
    document.getElementById('btn-refresh').addEventListener('click', async () => {
      if (AppState.activeDatabaseTab === 'dashboard') {
        await Dashboard.load();
        showToast('Dashboard refreshed', 'info', 1500);
      } else if (AppState.currentDbPath) {
        await window.dbAPI.closeConnection(AppState.currentDbPath);
        await TableList.load();
        showToast('Refreshed', 'info', 1500);
      } else {
        await loadDatabases();
      }
      AppState.autoRefresh.lastUpdate = new Date();
    });

    // --- Workspace Change ---
    document.getElementById('btn-workspace').addEventListener('click', async () => {
      const result = await window.dbAPI.selectWorkspace();
      if (result.success) {
        AppState.workspacePath = result.path;
        await loadDatabases();
        updateWorkspaceDisplay();
        showToast(`Workspace: ${result.path}`, 'info');
        // Switch to dashboard to show new workspace
        await switchDatabaseTab('dashboard');
      }
    });

    // --- Open File ---
    document.getElementById('btn-open-file').addEventListener('click', async () => {
      if (window.dbAPI.selectDatabaseFile) {
        const result = await window.dbAPI.selectDatabaseFile();
        if (result.success && result.path) {
          // Add the file as a custom database tab
          const name = result.path.split(/[\\/]/).pop();
          const customDb = {
            name,
            label: name.replace('.db', ''),
            description: 'External database',
            path: result.path,
            exists: true,
            sizeBytes: 0,
            lastModified: new Date().toISOString(),
          };
          // Check if not already in the list
          if (!AppState.databases.find(d => d.path === result.path)) {
            AppState.databases.push(customDb);
            buildDatabaseTabs(AppState.databases);
          }
          // Switch to it
          await switchDatabaseTab(result.path);
          showToast(`Opened ${name}`, 'info');
        }
      } else {
        showToast('File picker not available in browser mode', 'warning');
      }
    });

    // --- Read-Only Toggle ---
    document.getElementById('toggle-readonly').addEventListener('change', (e) => {
      AppState.readOnly = e.target.checked;
      if (AppState.currentTable) {
        DataGrid.load(); // Re-render to show/hide edit controls
      }
      showToast(
        AppState.readOnly ? 'Read-only mode enabled' : 'Read-only mode disabled - be careful!',
        AppState.readOnly ? 'info' : 'warning'
      );
    });

    // --- Dashboard Tab Click ---
    const dashTab = document.querySelector('[data-db="dashboard"]');
    if (dashTab) {
      dashTab.addEventListener('click', () => switchDatabaseTab('dashboard'));
    }

    // --- View Tab Switching (Data / SQL / FTS) ---
    document.querySelectorAll('#tab-bar .tab').forEach(tab => {
      tab.addEventListener('click', () => {
        switchTab(tab.dataset.tab);
      });
    });

    // --- Pagination ---
    document.getElementById('btn-first-page').addEventListener('click', () => DataGrid.goToPage(1));
    document.getElementById('btn-prev-page').addEventListener('click', () =>
      DataGrid.goToPage(AppState.pagination.page - 1));
    document.getElementById('btn-next-page').addEventListener('click', () =>
      DataGrid.goToPage(AppState.pagination.page + 1));
    document.getElementById('btn-last-page').addEventListener('click', () =>
      DataGrid.goToPage(AppState.pagination.totalPages));

    document.getElementById('page-size').addEventListener('change', (e) => {
      AppState.pagination.limit = parseInt(e.target.value, 10);
      AppState.pagination.page = 1;
      DataGrid.load();
    });

    // --- Filter / Search ---
    const filterInput = document.getElementById('filter-value');
    const debouncedFilter = debounce(() => DataGrid.applyFilter(), 300);
    filterInput.addEventListener('input', debouncedFilter);
    document.getElementById('filter-column').addEventListener('change', () => {
      if (filterInput.value.trim()) {
        DataGrid.applyFilter();
      }
    });

    // --- Add Row ---
    document.getElementById('btn-add-row').addEventListener('click', () => {
      AutoRefresh.pause();
      Modals.showAddRow();
    });
    document.getElementById('btn-confirm-add').addEventListener('click', () => {
      Modals.executeAddRow();
      AutoRefresh.resume();
    });
    document.getElementById('btn-cancel-add').addEventListener('click', () => {
      Modals.hide('add-row-modal');
      AutoRefresh.resume();
    });

    // --- Delete Row ---
    document.getElementById('btn-confirm-delete').addEventListener('click', () => {
      DataGrid.executeDelete();
      AutoRefresh.resume();
    });
    document.getElementById('btn-cancel-delete').addEventListener('click', () => {
      Modals.hide('delete-modal');
      AutoRefresh.resume();
    });

    // --- Edit Cell ---
    document.getElementById('btn-confirm-edit').addEventListener('click', () => {
      DataGrid.confirmEdit();
      AutoRefresh.resume();
    });
    document.getElementById('btn-cancel-edit').addEventListener('click', () => {
      Modals.hide('edit-cell-modal');
      AutoRefresh.resume();
    });

    // --- Schema ---
    document.getElementById('btn-schema').addEventListener('click', () => {
      Modals.showSchema();
    });

    // --- Backup ---
    document.getElementById('btn-backup').addEventListener('click', async () => {
      if (!AppState.currentDbPath) {
        showToast('No database selected', 'warning');
        return;
      }
      const result = await window.dbAPI.backup(AppState.currentDbPath);
      if (result.success) {
        showToast(`Backup saved to ${result.filePath}`, 'success');
      } else if (result.error !== 'Backup cancelled') {
        showToast(`Backup failed: ${result.error}`, 'error');
      }
    });

    // --- Export ---
    document.getElementById('btn-export-json').addEventListener('click', async () => {
      if (!AppState.currentDbPath || !AppState.currentTable) return;
      const result = await window.dbAPI.exportTable(AppState.currentDbPath, AppState.currentTable, 'json');
      if (result.success) {
        showToast(`Exported ${result.rowCount} rows to ${result.filePath}`, 'success');
      } else if (result.error !== 'Export cancelled') {
        showToast(`Export failed: ${result.error}`, 'error');
      }
    });

    document.getElementById('btn-export-csv').addEventListener('click', async () => {
      if (!AppState.currentDbPath || !AppState.currentTable) return;
      const result = await window.dbAPI.exportTable(AppState.currentDbPath, AppState.currentTable, 'csv');
      if (result.success) {
        showToast(`Exported ${result.rowCount} rows to ${result.filePath}`, 'success');
      } else if (result.error !== 'Export cancelled') {
        showToast(`Export failed: ${result.error}`, 'error');
      }
    });

    // --- SQL Console ---
    document.getElementById('btn-run-sql').addEventListener('click', () => {
      SqlConsole.execute();
    });
    document.getElementById('btn-clear-sql').addEventListener('click', () => {
      SqlConsole.clear();
    });

    // --- FTS5 Search ---
    document.getElementById('btn-fts-search').addEventListener('click', () => {
      FtsSearch.search();
    });
    document.getElementById('fts-query').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        FtsSearch.search();
      }
    });

    // --- Keyboard Shortcuts ---
    document.addEventListener('keydown', (e) => {
      // Ctrl+F: Focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        document.getElementById('filter-value').focus();
      }

      // Ctrl+Enter: Run SQL (when in SQL tab)
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (AppState.activeTab === 'sql-console') {
          e.preventDefault();
          SqlConsole.execute();
        }
      }

      // Ctrl+E: Switch to SQL console
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        switchTab('sql-console');
      }

      // Ctrl+D: Switch to dashboard
      if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        switchDatabaseTab('dashboard');
      }
    });

    // --- Menu Events (from Electron main process) ---
    if (window.dbAPI.onMenuEvent) {
      window.dbAPI.onMenuEvent('menu:openWorkspace', async () => {
        const result = await window.dbAPI.selectWorkspace();
        if (result.success) {
          AppState.workspacePath = result.path;
          await loadDatabases();
          updateWorkspaceDisplay();
          showToast(`Workspace: ${result.path}`, 'info');
          await switchDatabaseTab('dashboard');
        }
      });

      window.dbAPI.onMenuEvent('menu:refresh', async () => {
        if (AppState.activeDatabaseTab === 'dashboard') {
          await Dashboard.load();
        } else if (AppState.currentDbPath) {
          await window.dbAPI.closeConnection(AppState.currentDbPath);
          await TableList.load();
        }
        showToast('Refreshed', 'info', 1500);
      });
    }

    // Pause auto-refresh when any modal is open
    const observer = new MutationObserver((mutations) => {
      const anyModalOpen = document.querySelectorAll('.modal:not(.hidden)').length > 0;
      if (anyModalOpen) {
        AutoRefresh.pause();
      } else {
        AutoRefresh.resume();
      }
    });
    document.querySelectorAll('.modal').forEach(modal => {
      observer.observe(modal, { attributes: true, attributeFilter: ['class'] });
    });
  }

  // ========================================================================
  // View Tab Switching (Data / SQL / FTS within a database)
  // ========================================================================

  function switchTab(tabName) {
    AppState.activeTab = tabName;

    // Update tab buttons
    document.querySelectorAll('#tab-bar .tab').forEach(t => {
      t.classList.toggle('active', t.dataset.tab === tabName);
    });

    // Show/hide tab panels
    document.getElementById('data-grid-tab').classList.toggle('hidden', tabName !== 'data-grid');
    document.getElementById('sql-console-tab').classList.toggle('hidden', tabName !== 'sql-console');
    document.getElementById('fts-search-tab').classList.toggle('hidden', tabName !== 'fts-search');
  }

  // ========================================================================
  // Start
  // ========================================================================

  document.addEventListener('DOMContentLoaded', init);

})();
