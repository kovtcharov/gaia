// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Utility Functions
 */

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'success'|'error'|'info'|'warning'} type
 * @param {number} duration - ms to show
 */
function showToast(message, type = 'info', duration = 3000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

/**
 * Format a byte count to a human-readable size.
 * @param {number} bytes
 * @returns {string}
 */
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
}

/**
 * Determine the CSS class for a cell value.
 * @param {*} value
 * @param {string} columnName
 * @returns {string}
 */
function getCellClass(value, columnName) {
  if (value === null || value === undefined) return 'cell-null';
  if (typeof value === 'number') return 'cell-number';
  const str = String(value);
  if (str.startsWith('{') || str.startsWith('[')) return 'cell-json';
  return '';
}

/**
 * Format a cell value for display.
 * @param {*} value
 * @param {string} columnName
 * @returns {string}
 */
function formatCellValue(value, columnName) {
  if (value === null || value === undefined) return 'NULL';
  if (typeof value === 'boolean') return value ? 'true' : 'false';

  const str = String(value);

  // Truncate very long values
  if (str.length > 200) {
    return str.substring(0, 200) + '...';
  }

  return str;
}

/**
 * Get the primary key column for a table.
 * @param {Array} columns - Column info array
 * @returns {string|null}
 */
function getPrimaryKeyColumn(columns) {
  if (!columns || columns.length === 0) return null;

  // Look for explicit PK
  const pk = columns.find(c => c.primaryKey);
  if (pk) return pk.name;

  // Fall back to 'id' or 'rowid'
  const idCol = columns.find(c => c.name === 'id' || c.name === 'rowid');
  if (idCol) return idCol.name;

  // Fall back to first column
  return columns[0].name;
}

/**
 * Escape HTML to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

/**
 * Debounce a function call.
 * @param {Function} fn
 * @param {number} delay
 * @returns {Function}
 */
function debounce(fn, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

/**
 * Check if a value looks like JSON.
 * @param {*} value
 * @returns {boolean}
 */
function isJsonLike(value) {
  if (typeof value !== 'string') return false;
  const trimmed = value.trim();
  return (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
         (trimmed.startsWith('[') && trimmed.endsWith(']'));
}

/**
 * Pretty-print JSON if possible.
 * @param {string} value
 * @returns {string}
 */
function prettyJson(value) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

/**
 * Format a relative time string (e.g., "2 minutes ago").
 * @param {string|Date} timestamp - ISO string or Date object
 * @returns {string}
 */
function formatRelativeTime(timestamp) {
  if (!timestamp) return 'never';
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHour < 24) return `${diffHour}h ago`;
  return `${diffDay}d ago`;
}

/**
 * Format a timestamp for display with date and time.
 * @param {string} timestamp - ISO string
 * @returns {string}
 */
function formatTimestamp(timestamp) {
  if (!timestamp) return '--';
  try {
    const date = new Date(timestamp);
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return timestamp;
  }
}

/**
 * Truncate a string to a maximum length.
 * @param {string} str
 * @param {number} maxLen
 * @returns {string}
 */
function truncate(str, maxLen = 80) {
  if (!str) return '';
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen) + '...';
}
