// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Modal Management
 */

const Modals = {

  /**
   * Show a modal by ID.
   * @param {string} modalId
   */
  show(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
      modal.classList.remove('hidden');
    }
  },

  /**
   * Hide a modal by ID.
   * @param {string} modalId
   */
  hide(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
      modal.classList.add('hidden');
    }
  },

  /**
   * Initialize all modal close buttons and backdrop clicks.
   */
  init() {
    // Close buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
      btn.addEventListener('click', () => {
        const modalId = btn.dataset.close;
        if (modalId) this.hide(modalId);
      });
    });

    // Backdrop click
    document.querySelectorAll('.modal').forEach(modal => {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.add('hidden');
        }
      });
    });

    // Escape key closes top-most modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        const visibleModals = document.querySelectorAll('.modal:not(.hidden)');
        if (visibleModals.length > 0) {
          visibleModals[visibleModals.length - 1].classList.add('hidden');
        }
      }
    });
  },

  /**
   * Show the schema modal.
   */
  async showSchema() {
    if (!AppState.currentDbPath) {
      showToast('No database selected', 'warning');
      return;
    }

    const result = await window.dbAPI.getSchema(AppState.currentDbPath);
    if (!result.success) {
      showToast(`Schema error: ${result.error}`, 'error');
      return;
    }

    const content = result.schemas
      .map(s => `-- ${s.name}\n${s.sql};`)
      .join('\n\n');

    document.getElementById('schema-content').textContent = content || 'No schema found';
    this.show('schema-modal');
  },

  /**
   * Show the add row modal with form fields for the current table.
   */
  showAddRow() {
    if (!AppState.currentTable || !AppState.currentTableInfo) {
      showToast('Select a table first', 'warning');
      return;
    }
    if (AppState.readOnly) {
      showToast('Disable read-only mode to add rows', 'warning');
      return;
    }

    const form = document.getElementById('add-row-form');
    form.innerHTML = '';

    const columns = AppState.currentTableInfo.columns || [];
    for (const col of columns) {
      // Skip auto-increment primary keys
      if (col.primaryKey && col.type && col.type.toUpperCase().includes('INTEGER')) {
        continue;
      }

      const group = document.createElement('div');
      group.className = 'form-group';

      const label = document.createElement('label');
      label.className = 'form-label';
      label.textContent = `${col.name} (${col.type || 'TEXT'})${col.notNull ? ' *' : ''}`;

      const input = document.createElement('input');
      input.type = 'text';
      input.name = col.name;
      input.placeholder = col.defaultValue ? `Default: ${col.defaultValue}` : '';

      group.appendChild(label);
      group.appendChild(input);
      form.appendChild(group);
    }

    this.show('add-row-modal');

    // Focus first input
    const firstInput = form.querySelector('input');
    if (firstInput) firstInput.focus();
  },

  /**
   * Execute the add row form.
   */
  async executeAddRow() {
    if (!AppState.currentDbPath || !AppState.currentTable) return;

    const form = document.getElementById('add-row-form');
    const inputs = form.querySelectorAll('input');
    const data = {};

    for (const input of inputs) {
      const value = input.value.trim();
      if (value !== '') {
        data[input.name] = value;
      }
    }

    if (Object.keys(data).length === 0) {
      showToast('Enter at least one field value', 'warning');
      return;
    }

    const result = await window.dbAPI.insertRow(
      AppState.currentDbPath,
      AppState.currentTable,
      data
    );

    if (result.success) {
      showToast('Row inserted', 'success');
      this.hide('add-row-modal');

      // Log INSERT operation to history
      HistoryLog.addEntry(
        HistoryLog.dbNameFromPath(AppState.currentDbPath),
        'INSERT',
        AppState.currentTable,
        1,
        HistoryLog.buildDetails(data)
      );

      DataGrid.load();
      TableList.load();
    } else {
      showToast(`Insert failed: ${result.error}`, 'error');
    }
  },
};
