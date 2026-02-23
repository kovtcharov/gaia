// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Table List (Sidebar)
 *
 * Renders the list of tables in the selected database.
 */

const TableList = {

  /**
   * Load tables for the currently selected database.
   */
  async load() {
    if (!AppState.currentDbPath) return;

    const tableListEl = document.getElementById('table-list');
    tableListEl.innerHTML = '<div class="empty-state"><span class="loading-spinner"></span></div>';

    const result = await window.dbAPI.getTables(AppState.currentDbPath);

    if (!result.success) {
      tableListEl.innerHTML = `<div class="empty-state">Error: ${escapeHtml(result.error)}</div>`;
      showToast(result.error, 'error');
      return;
    }

    AppState.tables = result.tables;
    document.getElementById('table-count').textContent = result.tables.length;

    if (result.tables.length === 0) {
      tableListEl.innerHTML = '<div class="empty-state">No tables found</div>';
      return;
    }

    this.render();

    // Auto-select first non-FTS table
    const firstTable = result.tables.find(t => !t.isFts) || result.tables[0];
    if (firstTable) {
      this.selectTable(firstTable.name);
    }
  },

  /**
   * Render the table list in the sidebar.
   */
  render() {
    const tableListEl = document.getElementById('table-list');
    tableListEl.innerHTML = '';

    for (const table of AppState.tables) {
      const item = document.createElement('div');
      item.className = `table-item${table.isFts ? ' fts' : ''}`;
      item.dataset.table = table.name;

      if (table.name === AppState.currentTable) {
        item.classList.add('active');
      }

      item.innerHTML = `
        <span class="table-item-name">${escapeHtml(table.name)}</span>
        <span class="table-item-count">${table.rowCount.toLocaleString()}</span>
      `;

      item.addEventListener('click', () => this.selectTable(table.name));
      tableListEl.appendChild(item);
    }
  },

  /**
   * Select a table and load its data.
   * @param {string} tableName
   */
  selectTable(tableName) {
    AppState.currentTable = tableName;
    const tableInfo = AppState.tables.find(t => t.name === tableName);
    AppState.currentTableInfo = tableInfo;

    // Reset pagination and filters
    AppState.pagination.page = 1;
    AppState.sortColumn = null;
    AppState.sortDirection = 'ASC';
    AppState.filterColumn = null;
    AppState.filterValue = null;

    // Update sidebar selection
    document.querySelectorAll('.table-item').forEach(el => {
      el.classList.toggle('active', el.dataset.table === tableName);
    });

    // Update filter column dropdown
    const filterColumnEl = document.getElementById('filter-column');
    filterColumnEl.innerHTML = '<option value="">All columns</option>';
    if (tableInfo && tableInfo.columns) {
      for (const col of tableInfo.columns) {
        const opt = document.createElement('option');
        opt.value = col.name;
        opt.textContent = col.name;
        filterColumnEl.appendChild(opt);
      }
    }

    // Clear filter input
    document.getElementById('filter-value').value = '';

    // Update FTS table selector
    FtsSearch.updateTableSelector();

    // Load data
    DataGrid.load();
  },
};
