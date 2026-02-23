// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Data Grid
 *
 * Renders the data table with pagination, sorting, filtering, and inline editing.
 */

const DataGrid = {

  /**
   * Load data for the current table with current pagination/sort/filter settings.
   */
  async load() {
    if (!AppState.currentDbPath || !AppState.currentTable) return;

    const tableInfo = AppState.currentTableInfo;
    document.getElementById('table-name-display').textContent = AppState.currentTable;

    const result = await window.dbAPI.queryTable(
      AppState.currentDbPath,
      AppState.currentTable,
      {
        page: AppState.pagination.page,
        limit: AppState.pagination.limit,
        sortColumn: AppState.sortColumn,
        sortDirection: AppState.sortDirection,
        filterColumn: AppState.filterColumn,
        filterValue: AppState.filterValue,
      }
    );

    if (!result.success) {
      showToast(`Query error: ${result.error}`, 'error');
      return;
    }

    AppState.rows = result.rows;
    AppState.pagination = { ...AppState.pagination, ...result.pagination };

    document.getElementById('row-count-display').textContent =
      `${result.pagination.totalRows.toLocaleString()} rows`;

    this.renderHead();
    this.renderBody();
    this.renderPagination();
  },

  /**
   * Render table header with sortable columns.
   */
  renderHead() {
    const thead = document.getElementById('data-table-head');
    thead.innerHTML = '';

    const tableInfo = AppState.currentTableInfo;
    const columns = tableInfo && tableInfo.columns ? tableInfo.columns : [];

    // If we have rows but no column info, derive from first row
    let colNames = columns.map(c => c.name);
    if (colNames.length === 0 && AppState.rows.length > 0) {
      colNames = Object.keys(AppState.rows[0]);
    }

    if (colNames.length === 0) return;

    const tr = document.createElement('tr');

    // Row number column
    const thNum = document.createElement('th');
    thNum.textContent = '#';
    thNum.style.width = '40px';
    thNum.style.cursor = 'default';
    tr.appendChild(thNum);

    for (const colName of colNames) {
      const th = document.createElement('th');
      th.textContent = colName;
      th.dataset.column = colName;

      // Sort indicator
      if (AppState.sortColumn === colName) {
        th.classList.add(AppState.sortDirection === 'ASC' ? 'sorted-asc' : 'sorted-desc');
      }

      th.addEventListener('click', () => this.toggleSort(colName));
      tr.appendChild(th);
    }

    // Actions column
    if (!AppState.readOnly) {
      const thActions = document.createElement('th');
      thActions.textContent = 'Actions';
      thActions.style.width = '80px';
      thActions.style.cursor = 'default';
      tr.appendChild(thActions);
    }

    thead.appendChild(tr);
  },

  /**
   * Render table body with data rows.
   */
  renderBody() {
    const tbody = document.getElementById('data-table-body');
    tbody.innerHTML = '';

    if (AppState.rows.length === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 100;
      td.className = 'empty-state';
      td.textContent = 'No data';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    const columns = this.getColumnNames();
    const pkColumn = getPrimaryKeyColumn(
      AppState.currentTableInfo ? AppState.currentTableInfo.columns : []
    );
    const startNum = (AppState.pagination.page - 1) * AppState.pagination.limit + 1;

    for (let i = 0; i < AppState.rows.length; i++) {
      const row = AppState.rows[i];
      const tr = document.createElement('tr');

      // Row number
      const tdNum = document.createElement('td');
      tdNum.textContent = startNum + i;
      tdNum.style.color = 'var(--text-muted)';
      tdNum.style.fontSize = '10px';
      tr.appendChild(tdNum);

      for (const colName of columns) {
        const td = document.createElement('td');
        const value = row[colName];
        const cellClass = getCellClass(value, colName);

        if (cellClass) td.classList.add(cellClass);

        // Special rendering for known column types
        const rendered = this.renderCellValue(value, colName);
        td.innerHTML = rendered;

        // Make cells clickable for editing (if not read-only)
        if (!AppState.readOnly && pkColumn) {
          td.classList.add('cell-clickable');
          td.addEventListener('click', () => {
            this.editCell(row, pkColumn, colName, value);
          });
        }

        // Tooltip for truncated/JSON values
        if (value !== null && value !== undefined) {
          const str = String(value);
          if (str.length > 50) {
            td.title = isJsonLike(str) ? prettyJson(str) : str.substring(0, 500);
          }
        }

        tr.appendChild(td);
      }

      // Actions column
      if (!AppState.readOnly && pkColumn) {
        const tdActions = document.createElement('td');
        tdActions.innerHTML = `
          <div class="row-actions">
            <button class="row-action-btn delete" title="Delete row" data-pk="${escapeHtml(String(row[pkColumn]))}">Del</button>
          </div>
        `;
        tdActions.querySelector('.delete').addEventListener('click', (e) => {
          e.stopPropagation();
          this.confirmDelete(row, pkColumn);
        });
        tr.appendChild(tdActions);
      }

      tbody.appendChild(tr);
    }
  },

  /**
   * Render a cell value with special formatting.
   * @param {*} value
   * @param {string} columnName
   * @returns {string} HTML string
   */
  renderCellValue(value, columnName) {
    if (value === null || value === undefined) {
      return '<span style="color: var(--text-muted); font-style: italic;">NULL</span>';
    }

    // Log level badges
    if (columnName === 'level' && typeof value === 'string') {
      const levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
      if (levels.includes(value.toUpperCase())) {
        return `<span class="level-badge ${escapeHtml(value.toUpperCase())}">${escapeHtml(value)}</span>`;
      }
    }

    // Task status badges
    if (columnName === 'status' && typeof value === 'string') {
      const statuses = ['pending', 'in_progress', 'completed', 'failed'];
      if (statuses.includes(value)) {
        return `<span class="status-badge ${escapeHtml(value)}">${escapeHtml(value)}</span>`;
      }
    }

    // Boolean display
    if (typeof value === 'number' && (columnName === 'enabled' || columnName === 'success' || columnName === 'notnull' || columnName === 'read')) {
      return value ? '<span style="color: var(--accent-success);">true</span>'
                    : '<span style="color: var(--accent-danger);">false</span>';
    }

    return escapeHtml(formatCellValue(value, columnName));
  },

  /**
   * Get column names for the current data.
   * @returns {string[]}
   */
  getColumnNames() {
    const tableInfo = AppState.currentTableInfo;
    if (tableInfo && tableInfo.columns && tableInfo.columns.length > 0) {
      return tableInfo.columns.map(c => c.name);
    }
    if (AppState.rows.length > 0) {
      return Object.keys(AppState.rows[0]);
    }
    return [];
  },

  /**
   * Toggle sort on a column.
   * @param {string} colName
   */
  toggleSort(colName) {
    if (AppState.sortColumn === colName) {
      AppState.sortDirection = AppState.sortDirection === 'ASC' ? 'DESC' : 'ASC';
    } else {
      AppState.sortColumn = colName;
      AppState.sortDirection = 'ASC';
    }
    AppState.pagination.page = 1;
    this.load();
  },

  /**
   * Apply a search filter.
   */
  applyFilter() {
    const filterValue = document.getElementById('filter-value').value.trim();
    const filterColumn = document.getElementById('filter-column').value;

    AppState.filterValue = filterValue || null;
    AppState.filterColumn = filterColumn || null;
    AppState.pagination.page = 1;
    this.load();
  },

  /**
   * Render pagination controls.
   */
  renderPagination() {
    const { page, totalPages, totalRows, limit } = AppState.pagination;

    document.getElementById('page-info').textContent = `Page ${page} of ${totalPages || 1}`;
    document.getElementById('btn-first-page').disabled = page <= 1;
    document.getElementById('btn-prev-page').disabled = page <= 1;
    document.getElementById('btn-next-page').disabled = page >= totalPages;
    document.getElementById('btn-last-page').disabled = page >= totalPages;
  },

  /**
   * Navigate to a page.
   * @param {number} page
   */
  goToPage(page) {
    const { totalPages } = AppState.pagination;
    const newPage = Math.max(1, Math.min(page, totalPages || 1));
    if (newPage !== AppState.pagination.page) {
      AppState.pagination.page = newPage;
      this.load();
    }
  },

  /**
   * Open the edit cell modal.
   */
  editCell(row, pkColumn, colName, value) {
    AppState.pendingEdit = {
      dbPath: AppState.currentDbPath,
      table: AppState.currentTable,
      primaryKey: { column: pkColumn, value: row[pkColumn] },
      column: colName,
      currentValue: value,
    };

    document.getElementById('edit-cell-label').textContent =
      `${AppState.currentTable}.${colName} (PK: ${pkColumn} = ${row[pkColumn]})`;

    const textarea = document.getElementById('edit-cell-value');
    if (value !== null && value !== undefined) {
      textarea.value = isJsonLike(String(value)) ? prettyJson(String(value)) : String(value);
    } else {
      textarea.value = '';
    }

    Modals.show('edit-cell-modal');
    textarea.focus();
  },

  /**
   * Confirm and execute cell edit.
   */
  async confirmEdit() {
    const edit = AppState.pendingEdit;
    if (!edit) return;

    const newValue = document.getElementById('edit-cell-value').value;

    // Check if value changed
    const oldStr = edit.currentValue !== null && edit.currentValue !== undefined
      ? String(edit.currentValue) : '';
    if (newValue === oldStr) {
      Modals.hide('edit-cell-modal');
      return;
    }

    const result = await window.dbAPI.updateRow(
      edit.dbPath,
      edit.table,
      edit.primaryKey,
      { [edit.column]: newValue || null }
    );

    if (result.success) {
      showToast(`Updated ${edit.column}`, 'success');
      Modals.hide('edit-cell-modal');
      this.load(); // Refresh
    } else {
      showToast(`Update failed: ${result.error}`, 'error');
    }
  },

  /**
   * Show delete confirmation modal.
   */
  confirmDelete(row, pkColumn) {
    const pkValue = row[pkColumn];
    AppState.pendingDelete = {
      dbPath: AppState.currentDbPath,
      table: AppState.currentTable,
      primaryKey: { column: pkColumn, value: pkValue },
    };

    document.getElementById('delete-message').textContent =
      `Delete row where ${pkColumn} = ${pkValue} from ${AppState.currentTable}?`;

    Modals.show('delete-modal');
  },

  /**
   * Execute the pending delete.
   */
  async executeDelete() {
    const del = AppState.pendingDelete;
    if (!del) return;

    const result = await window.dbAPI.deleteRow(del.dbPath, del.table, del.primaryKey);

    if (result.success) {
      showToast('Row deleted', 'success');
      Modals.hide('delete-modal');
      this.load(); // Refresh
      // Refresh table list to update row counts
      TableList.load();
    } else {
      showToast(`Delete failed: ${result.error}`, 'error');
    }
  },
};
