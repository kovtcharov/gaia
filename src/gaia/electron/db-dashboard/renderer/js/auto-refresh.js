// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Auto-Refresh Module
 *
 * Provides configurable auto-refresh, file change detection,
 * row change highlighting, and notification toasts for new data.
 */

const AutoRefresh = {

  /**
   * Initialize auto-refresh UI controls and state.
   */
  init() {
    this._bindControls();
    this._startLastUpdateTimer();
  },

  /**
   * Bind the auto-refresh UI controls.
   */
  _bindControls() {
    const toggle = document.getElementById('auto-refresh-toggle');
    const intervalSelect = document.getElementById('auto-refresh-interval');

    if (toggle) {
      toggle.addEventListener('change', (e) => {
        if (e.target.checked) {
          this.start();
        } else {
          this.stop();
        }
      });
    }

    if (intervalSelect) {
      intervalSelect.addEventListener('change', (e) => {
        const ms = parseInt(e.target.value, 10);
        AppState.autoRefresh.interval = ms;
        if (AppState.autoRefresh.enabled) {
          // Restart with new interval
          this.stop();
          this.start();
        }
      });
    }
  },

  /**
   * Start auto-refresh polling.
   */
  start() {
    this.stop(); // Clear any existing timer
    AppState.autoRefresh.enabled = true;
    AppState.autoRefresh.isPaused = false;

    const toggle = document.getElementById('auto-refresh-toggle');
    if (toggle) toggle.checked = true;

    this._startFilePolling();
    this._scheduleRefresh();

    this._updateStatusIndicator();
  },

  /**
   * Stop auto-refresh polling.
   */
  stop() {
    AppState.autoRefresh.enabled = false;

    if (AppState.autoRefresh.timerId) {
      clearInterval(AppState.autoRefresh.timerId);
      AppState.autoRefresh.timerId = null;
    }

    this._stopFilePolling();
    this._updateStatusIndicator();
  },

  /**
   * Temporarily pause auto-refresh (e.g., during editing).
   */
  pause() {
    if (!AppState.autoRefresh.enabled) return;
    AppState.autoRefresh.isPaused = true;
    if (AppState.autoRefresh.timerId) {
      clearInterval(AppState.autoRefresh.timerId);
      AppState.autoRefresh.timerId = null;
    }
    this._updateStatusIndicator();
  },

  /**
   * Resume auto-refresh after pause.
   */
  resume() {
    if (!AppState.autoRefresh.enabled) return;
    AppState.autoRefresh.isPaused = false;
    this._scheduleRefresh();
    this._updateStatusIndicator();
  },

  /**
   * Schedule the next refresh cycle.
   */
  _scheduleRefresh() {
    if (AppState.autoRefresh.timerId) {
      clearInterval(AppState.autoRefresh.timerId);
    }

    AppState.autoRefresh.timerId = setInterval(() => {
      if (AppState.autoRefresh.isPaused) return;
      this._doRefresh();
    }, AppState.autoRefresh.interval);
  },

  /**
   * Execute a single refresh cycle.
   * Debounced to prevent overlapping refreshes.
   */
  async _doRefresh() {
    if (this._refreshing) return; // Prevent overlap
    this._refreshing = true;

    try {
      // Get previous row fingerprints before refresh for change detection
      const previousRows = this._fingerprint(AppState.rows);

      // Refresh based on the active view
      if (AppState.activeDatabaseTab === 'dashboard') {
        await Dashboard.load();
      } else if (AppState.currentDbPath && AppState.currentTable && AppState.activeTab === 'data-grid') {
        await DataGrid.load();

        // Detect changed rows
        const newRows = this._fingerprint(AppState.rows);
        this._detectChanges(previousRows, newRows);
      }

      AppState.autoRefresh.lastUpdate = new Date();
      this._updateLastUpdateDisplay();
    } catch (e) {
      // Silently handle refresh errors to avoid spamming user
      console.warn('Auto-refresh error:', e);
    } finally {
      this._refreshing = false;
    }
  },

  /**
   * Start polling for file modification changes.
   */
  _startFilePolling() {
    this._stopFilePolling();

    // Poll every 1 second for file changes (separate from data refresh)
    AppState.fileWatcher.pollTimerId = setInterval(async () => {
      if (!AppState.workspacePath) return;

      try {
        const result = await window.dbAPI.listDatabases(AppState.workspacePath);
        if (!result.success) return;

        let anyChanged = false;
        for (const db of result.databases) {
          if (!db.exists || !db.lastModified) continue;
          const prevMtime = AppState.fileWatcher.lastModTimes[db.path];
          const curMtime = db.lastModified;

          if (prevMtime && prevMtime !== curMtime) {
            anyChanged = true;
          }
          AppState.fileWatcher.lastModTimes[db.path] = curMtime;
        }

        if (anyChanged && AppState.notifications.showChangeToasts) {
          showToast('Database files changed on disk', 'info', 2000);
        }
      } catch (e) {
        // Silently ignore file polling errors
      }
    }, 1500);
  },

  /**
   * Stop file polling.
   */
  _stopFilePolling() {
    if (AppState.fileWatcher.pollTimerId) {
      clearInterval(AppState.fileWatcher.pollTimerId);
      AppState.fileWatcher.pollTimerId = null;
    }
  },

  /**
   * Generate a fingerprint array for a set of rows.
   * Uses a fast hash of JSON keys we care about.
   * @param {Array} rows
   * @returns {Map<string, string>} Map of row ID to value hash
   */
  _fingerprint(rows) {
    const map = new Map();
    if (!rows || rows.length === 0) return map;

    const pkCol = getPrimaryKeyColumn(
      AppState.currentTableInfo ? AppState.currentTableInfo.columns : []
    );

    for (const row of rows) {
      const key = pkCol && row[pkCol] != null ? String(row[pkCol]) : JSON.stringify(row);
      const val = JSON.stringify(row);
      map.set(key, val);
    }

    return map;
  },

  /**
   * Detect changes between previous and new row fingerprints.
   * Highlights changed rows and shows notifications.
   * @param {Map} prevFP
   * @param {Map} newFP
   */
  _detectChanges(prevFP, newFP) {
    if (prevFP.size === 0) return; // First load, skip

    const changedIds = new Set();
    let newRowCount = 0;

    for (const [key, val] of newFP) {
      if (!prevFP.has(key)) {
        // New row
        changedIds.add(key);
        newRowCount++;
      } else if (prevFP.get(key) !== val) {
        // Changed row
        changedIds.add(key);
      }
    }

    if (changedIds.size > 0) {
      AppState.changeTracking.changedRowIds = changedIds;
      this._highlightChangedRows();

      if (newRowCount > 0 && AppState.notifications.showChangeToasts) {
        showToast(`${newRowCount} new row${newRowCount > 1 ? 's' : ''} detected`, 'info', 2000);
      }

      // Auto-scroll to new rows if enabled
      if (newRowCount > 0 && AppState.notifications.autoScrollToNew) {
        const tableContainer = document.getElementById('table-container');
        if (tableContainer) {
          tableContainer.scrollTop = tableContainer.scrollHeight;
        }
      }
    }
  },

  /**
   * Apply highlight animation to changed rows in the data grid.
   */
  _highlightChangedRows() {
    const changedIds = AppState.changeTracking.changedRowIds;
    if (changedIds.size === 0) return;

    const pkCol = getPrimaryKeyColumn(
      AppState.currentTableInfo ? AppState.currentTableInfo.columns : []
    );
    if (!pkCol) return;

    const tbody = document.getElementById('data-table-body');
    if (!tbody) return;

    const rows = tbody.querySelectorAll('tr');
    for (const tr of rows) {
      // The first data cell (after row number) should let us find the PK value
      // We stored rows in AppState, correlate by index
      const rowIdx = Array.from(rows).indexOf(tr);
      if (rowIdx < 0 || rowIdx >= AppState.rows.length) continue;

      const rowData = AppState.rows[rowIdx];
      const rowKey = rowData && rowData[pkCol] != null ? String(rowData[pkCol]) : null;

      if (rowKey && changedIds.has(rowKey)) {
        tr.classList.add('row-changed');
      }
    }

    // Clear highlights after 3 seconds
    if (AppState.changeTracking.highlightTimer) {
      clearTimeout(AppState.changeTracking.highlightTimer);
    }
    AppState.changeTracking.highlightTimer = setTimeout(() => {
      AppState.changeTracking.changedRowIds.clear();
      const highlighted = document.querySelectorAll('.row-changed');
      highlighted.forEach(el => el.classList.remove('row-changed'));
    }, 3000);
  },

  /**
   * Update the status indicator text in the toolbar.
   */
  _updateStatusIndicator() {
    const indicator = document.getElementById('auto-refresh-status');
    if (!indicator) return;

    if (!AppState.autoRefresh.enabled) {
      indicator.textContent = 'Auto-refresh: OFF';
      indicator.className = 'auto-refresh-status status-off';
    } else if (AppState.autoRefresh.isPaused) {
      indicator.textContent = 'Auto-refresh: PAUSED';
      indicator.className = 'auto-refresh-status status-paused';
    } else {
      const sec = (AppState.autoRefresh.interval / 1000).toFixed(1).replace(/\.0$/, '');
      indicator.textContent = `Auto-refresh: ${sec}s`;
      indicator.className = 'auto-refresh-status status-on';
    }
  },

  /**
   * Update the "Last updated: X ago" display.
   */
  _updateLastUpdateDisplay() {
    const el = document.getElementById('last-update-display');
    if (!el) return;

    if (AppState.autoRefresh.lastUpdate) {
      el.textContent = `Updated: ${formatRelativeTime(AppState.autoRefresh.lastUpdate)}`;
    } else {
      el.textContent = '';
    }
  },

  /**
   * Start a timer that updates the "last update" display every second.
   */
  _startLastUpdateTimer() {
    if (AppState.autoRefresh.lastUpdateDisplay) {
      clearInterval(AppState.autoRefresh.lastUpdateDisplay);
    }
    AppState.autoRefresh.lastUpdateDisplay = setInterval(() => {
      this._updateLastUpdateDisplay();
    }, 1000);
  },
};
