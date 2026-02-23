// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * History Log
 *
 * Tracks database operations (SELECT, INSERT, UPDATE, DELETE) and renders
 * activity history panels on the dashboard and within each database tab.
 *
 * Entries are stored in a circular buffer (max 100) in AppState.historyLog.
 */

const HistoryLog = {

  /** Auto-incrementing ID for entries. */
  _nextId: 1,

  /** Relative time update interval ID. */
  _relativeTimeTimer: null,

  /**
   * Add a history entry.
   * @param {string} database - Database filename (e.g. "logs.db")
   * @param {string} operation - One of: SELECT, INSERT, UPDATE, DELETE
   * @param {string} table - Table name
   * @param {number} rowCount - Number of rows affected or returned
   * @param {string} [details] - Optional detail string
   */
  addEntry(database, operation, table, rowCount, details) {
    const entry = {
      id: this._nextId++,
      timestamp: new Date(),
      database: database || 'unknown',
      operation: (operation || 'SELECT').toUpperCase(),
      table: table || '',
      rowCount: rowCount || 0,
      details: details || '',
    };

    AppState.historyLog.unshift(entry);

    // Circular buffer: keep at most 100 entries
    if (AppState.historyLog.length > 100) {
      AppState.historyLog.length = 100;
    }
  },

  /**
   * Get recent operations, optionally filtered by database.
   * @param {string} [database] - Filter by database name (null = all)
   * @param {number} [limit=20] - Maximum entries to return
   * @returns {Array} Filtered entries
   */
  getRecentOperations(database, limit) {
    limit = limit || 20;
    if (!database) {
      return AppState.historyLog.slice(0, limit);
    }
    const filtered = [];
    for (const entry of AppState.historyLog) {
      if (entry.database === database) {
        filtered.push(entry);
        if (filtered.length >= limit) break;
      }
    }
    return filtered;
  },

  /**
   * Render the dashboard-level Activity History section.
   * Returns an HTMLElement (a dash-section) to be appended to the dashboard.
   * @returns {HTMLElement}
   */
  renderDashboardHistory() {
    const entries = this.getRecentOperations(null, 20);

    const section = document.createElement('div');
    section.className = 'dash-section';
    section.id = 'dash-activity-history';

    const header = document.createElement('div');
    header.className = 'dash-section-header';
    header.textContent = 'Activity History (Last 20 Operations)';
    section.appendChild(header);

    const body = document.createElement('div');
    body.className = 'dash-section-body';

    if (entries.length === 0) {
      body.innerHTML = '<div class="dash-empty">No operations recorded yet</div>';
    } else {
      const list = document.createElement('div');
      list.className = 'history-log-list';

      for (const entry of entries) {
        list.appendChild(this._renderEntry(entry, true));
      }
      body.appendChild(list);
    }

    section.appendChild(body);
    return section;
  },

  /**
   * Render the database-specific history panel (collapsible).
   * @param {string} database - Database filename to filter by
   * @returns {HTMLElement}
   */
  renderDatabaseHistory(database) {
    const entries = this.getRecentOperations(database, 10);

    const panel = document.createElement('div');
    panel.className = 'history-db-panel';
    panel.id = 'history-db-panel';

    // Collapsible header
    const header = document.createElement('div');
    header.className = 'history-db-header';
    header.innerHTML = `<span class="history-db-toggle">[+]</span> History <span class="history-db-count">${entries.length}</span>`;
    header.addEventListener('click', () => {
      const content = panel.querySelector('.history-db-content');
      const toggle = header.querySelector('.history-db-toggle');
      if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        toggle.textContent = '[-]';
      } else {
        content.classList.add('hidden');
        toggle.textContent = '[+]';
      }
    });
    panel.appendChild(header);

    // Content (starts collapsed)
    const content = document.createElement('div');
    content.className = 'history-db-content hidden';

    if (entries.length === 0) {
      content.innerHTML = '<div class="history-db-empty">No operations for this database yet</div>';
    } else {
      const list = document.createElement('div');
      list.className = 'history-log-list';

      for (const entry of entries) {
        list.appendChild(this._renderEntry(entry, false));
      }
      content.appendChild(list);
    }

    panel.appendChild(content);
    return panel;
  },

  /**
   * Render a single history entry row.
   * @param {Object} entry
   * @param {boolean} showDatabase - Whether to show the database column
   * @returns {HTMLElement}
   */
  _renderEntry(entry, showDatabase) {
    const row = document.createElement('div');
    row.className = 'history-entry';

    // Timestamp (relative with absolute tooltip)
    const timeEl = document.createElement('span');
    timeEl.className = 'history-time';
    timeEl.textContent = this._formatRelativeShort(entry.timestamp);
    timeEl.title = entry.timestamp.toLocaleString();
    timeEl.dataset.ts = entry.timestamp.getTime();

    // Operation badge
    const opEl = document.createElement('span');
    opEl.className = `history-op history-op-${entry.operation.toLowerCase()}`;
    opEl.textContent = entry.operation;

    // Database name (only on dashboard)
    let dbEl = null;
    if (showDatabase) {
      dbEl = document.createElement('span');
      dbEl.className = 'history-db-name';
      dbEl.textContent = entry.database;
    }

    // Table name
    const tableEl = document.createElement('span');
    tableEl.className = 'history-table';
    tableEl.textContent = entry.table;

    // Row count
    const countEl = document.createElement('span');
    countEl.className = 'history-count';
    countEl.textContent = `(${entry.rowCount} row${entry.rowCount !== 1 ? 's' : ''})`;

    // Details
    let detailsEl = null;
    if (entry.details) {
      detailsEl = document.createElement('div');
      detailsEl.className = 'history-details';
      const truncated = entry.details.length > 80
        ? entry.details.substring(0, 80) + '...'
        : entry.details;
      detailsEl.textContent = truncated;
      if (entry.details.length > 80) {
        detailsEl.title = entry.details;
      }
    }

    // Assemble the row
    const mainLine = document.createElement('div');
    mainLine.className = 'history-entry-main';
    mainLine.appendChild(timeEl);
    if (dbEl) mainLine.appendChild(dbEl);
    mainLine.appendChild(opEl);
    mainLine.appendChild(tableEl);
    mainLine.appendChild(countEl);

    row.appendChild(mainLine);
    if (detailsEl) {
      row.appendChild(detailsEl);
    }

    return row;
  },

  /**
   * Format a timestamp as a short relative string (e.g. "2s ago").
   * @param {Date} date
   * @returns {string}
   */
  _formatRelativeShort(date) {
    const diffMs = Date.now() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 5) return 'now';
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;
    return `${Math.floor(diffHour / 24)}d ago`;
  },

  /**
   * Start a timer that updates all relative timestamps every second.
   */
  startRelativeTimeUpdates() {
    if (this._relativeTimeTimer) {
      clearInterval(this._relativeTimeTimer);
    }
    this._relativeTimeTimer = setInterval(() => {
      const timeEls = document.querySelectorAll('.history-time[data-ts]');
      for (const el of timeEls) {
        const ts = parseInt(el.dataset.ts, 10);
        if (!isNaN(ts)) {
          el.textContent = this._formatRelativeShort(new Date(ts));
        }
      }
    }, 1000);
  },

  /**
   * Extract the database filename from a full path.
   * @param {string} dbPath
   * @returns {string}
   */
  dbNameFromPath(dbPath) {
    if (!dbPath) return 'unknown';
    // Handle both forward and backslash separators
    const parts = dbPath.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1] || 'unknown';
  },

  /**
   * Build a details string from a key-value object.
   * @param {Object} data
   * @param {number} [maxPairs=5]
   * @returns {string}
   */
  buildDetails(data, maxPairs) {
    if (!data || typeof data !== 'object') return '';
    maxPairs = maxPairs || 5;
    const pairs = [];
    const keys = Object.keys(data);
    for (let i = 0; i < Math.min(keys.length, maxPairs); i++) {
      const key = keys[i];
      let val = data[key];
      if (typeof val === 'string' && val.length > 40) {
        val = val.substring(0, 40) + '...';
      }
      pairs.push(`${key}=${JSON.stringify(val)}`);
    }
    if (keys.length > maxPairs) {
      pairs.push(`... +${keys.length - maxPairs} more`);
    }
    return pairs.join(', ');
  },
};
