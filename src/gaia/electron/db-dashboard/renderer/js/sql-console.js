// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * SQL Console
 *
 * Allows executing arbitrary SQL queries against the selected database.
 */

const SqlConsole = {

  /**
   * Execute the SQL in the editor.
   */
  async execute() {
    if (!AppState.currentDbPath) {
      showToast('No database selected', 'warning');
      return;
    }

    const sql = document.getElementById('sql-editor').value.trim();
    if (!sql) {
      showToast('Enter a SQL query', 'warning');
      return;
    }

    const resultsEl = document.getElementById('sql-results');
    resultsEl.innerHTML = '<div class="empty-state"><span class="loading-spinner"></span></div>';

    const result = await window.dbAPI.executeSQL(
      AppState.currentDbPath,
      sql,
      AppState.readOnly
    );

    if (!result.success) {
      resultsEl.innerHTML = `
        <div style="padding: 16px; color: var(--accent-danger); font-family: var(--font-mono); font-size: 12px;">
          Error: ${escapeHtml(result.error)}
        </div>
      `;
      return;
    }

    // Log to history
    const dbName = HistoryLog.dbNameFromPath(AppState.currentDbPath);
    if (result.type === 'query') {
      HistoryLog.addEntry(dbName, 'SELECT', this._extractTable(sql), result.rowCount || 0, `SQL: ${truncate(sql, 60)}`);
      this.renderQueryResult(result);
    } else {
      const op = this._detectOperation(sql);
      HistoryLog.addEntry(dbName, op, this._extractTable(sql), result.changes || 0, `SQL: ${truncate(sql, 60)}`);
      this.renderStatementResult(result);
    }
  },

  /**
   * Detect the SQL operation type from a query string.
   * @param {string} sql
   * @returns {string}
   */
  _detectOperation(sql) {
    const first = sql.trim().split(/\s+/)[0].toUpperCase();
    if (first === 'INSERT') return 'INSERT';
    if (first === 'UPDATE') return 'UPDATE';
    if (first === 'DELETE') return 'DELETE';
    if (first === 'SELECT') return 'SELECT';
    return first;
  },

  /**
   * Extract the table name from a SQL query (best-effort).
   * @param {string} sql
   * @returns {string}
   */
  _extractTable(sql) {
    const normalized = sql.trim().replace(/\s+/g, ' ');
    // Match: FROM tableName, INTO tableName, UPDATE tableName
    const match = normalized.match(/(?:FROM|INTO|UPDATE)\s+[`"']?(\w+)[`"']?/i);
    return match ? match[1] : '(sql)';
  },

  /**
   * Render SELECT query results as a table.
   * @param {Object} result
   */
  renderQueryResult(result) {
    const resultsEl = document.getElementById('sql-results');

    const info = document.createElement('div');
    info.className = 'sql-result-info';
    info.innerHTML = `
      <span class="count">${result.rowCount} row${result.rowCount !== 1 ? 's' : ''}</span>
      returned in
      <span class="duration">${result.duration}ms</span>
    `;

    if (result.rows.length === 0) {
      resultsEl.innerHTML = '';
      resultsEl.appendChild(info);
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = 'No results';
      resultsEl.appendChild(empty);
      return;
    }

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    for (const col of result.columns) {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const row of result.rows) {
      const tr = document.createElement('tr');
      for (const col of result.columns) {
        const td = document.createElement('td');
        const value = row[col];
        const cellClass = getCellClass(value, col);
        if (cellClass) td.classList.add(cellClass);
        td.innerHTML = DataGrid.renderCellValue(value, col);
        if (value !== null && value !== undefined && String(value).length > 50) {
          td.title = isJsonLike(String(value)) ? prettyJson(String(value)) : String(value).substring(0, 500);
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    resultsEl.innerHTML = '';
    resultsEl.appendChild(info);
    resultsEl.appendChild(table);
  },

  /**
   * Render INSERT/UPDATE/DELETE statement result.
   * @param {Object} result
   */
  renderStatementResult(result) {
    const resultsEl = document.getElementById('sql-results');
    resultsEl.innerHTML = `
      <div class="sql-result-info" style="padding: 16px;">
        Statement executed successfully.<br>
        <span class="count">${result.changes} row${result.changes !== 1 ? 's' : ''}</span> affected
        in <span class="duration">${result.duration}ms</span>
        ${result.lastInsertRowid ? `<br>Last insert rowid: ${result.lastInsertRowid}` : ''}
      </div>
    `;

    // Refresh the data grid and table list since data may have changed
    if (result.changes > 0) {
      DataGrid.load();
      TableList.load();
    }
  },

  /**
   * Clear the SQL editor.
   */
  clear() {
    document.getElementById('sql-editor').value = '';
    document.getElementById('sql-results').innerHTML =
      '<div class="empty-state">Run a query to see results</div>';
  },
};
