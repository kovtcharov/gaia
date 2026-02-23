// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * FTS5 Search
 *
 * Provides a dedicated UI for searching FTS5 virtual tables.
 */

const FtsSearch = {

  /**
   * Update the FTS table selector dropdown based on available FTS tables.
   */
  updateTableSelector() {
    const selector = document.getElementById('fts-table-selector');
    selector.innerHTML = '<option value="">-- Select FTS5 Table --</option>';

    const ftsTables = AppState.tables.filter(t => t.isFts);
    for (const table of ftsTables) {
      const opt = document.createElement('option');
      opt.value = table.name;
      opt.textContent = table.name;
      selector.appendChild(opt);
    }

    // Also add non-FTS tables that might have associated FTS tables
    // (e.g., runtime_logs has logs_fts)
    if (ftsTables.length === 0) {
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.textContent = 'No FTS5 tables found';
      selector.appendChild(opt);
    }
  },

  /**
   * Execute an FTS5 search.
   */
  async search() {
    const ftsTable = document.getElementById('fts-table-selector').value;
    const query = document.getElementById('fts-query').value.trim();

    if (!ftsTable) {
      showToast('Select an FTS5 table first', 'warning');
      return;
    }
    if (!query) {
      showToast('Enter a search query', 'warning');
      return;
    }
    if (!AppState.currentDbPath) {
      showToast('No database selected', 'warning');
      return;
    }

    const resultsEl = document.getElementById('fts-results');
    resultsEl.innerHTML = '<div class="empty-state"><span class="loading-spinner"></span></div>';

    const result = await window.dbAPI.searchFTS5(
      AppState.currentDbPath,
      ftsTable,
      query,
      { limit: 100 }
    );

    if (!result.success) {
      resultsEl.innerHTML = `
        <div style="padding: 16px; color: var(--accent-danger); font-family: var(--font-mono); font-size: 12px;">
          Error: ${escapeHtml(result.error)}
        </div>
      `;
      return;
    }

    if (result.rows.length === 0) {
      resultsEl.innerHTML = `
        <div class="sql-result-info" style="padding: 16px;">
          <span class="count">0 results</span> for "${escapeHtml(query)}" in ${escapeHtml(ftsTable)}
        </div>
        <div class="empty-state">No matches found</div>
      `;
      return;
    }

    // Render results as a table
    const info = document.createElement('div');
    info.className = 'sql-result-info';
    info.innerHTML = `
      <span class="count">${result.rowCount} result${result.rowCount !== 1 ? 's' : ''}</span>
      for "${escapeHtml(query)}" in ${escapeHtml(ftsTable)}
    `;

    const columns = Object.keys(result.rows[0]);
    const table = document.createElement('table');

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    for (const col of columns) {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const row of result.rows) {
      const tr = document.createElement('tr');
      for (const col of columns) {
        const td = document.createElement('td');
        const value = row[col];
        const cellClass = getCellClass(value, col);
        if (cellClass) td.classList.add(cellClass);
        td.innerHTML = DataGrid.renderCellValue(value, col);
        if (value !== null && value !== undefined && String(value).length > 50) {
          td.title = String(value).substring(0, 500);
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
};
