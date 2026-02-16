// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA Code Renderer Process
 *
 * Handles all UI logic, state management, and IPC communication
 * for the GAIA Code Electron application.
 */

// ============================================================================
// Application State
// ============================================================================

const state = {
  connected: false,
  activePanel: 'chat',
  activeDatabase: 'memory',
  activeDatabaseTable: null,
  chatMessages: [],
  auditLog: [],
  auditFilter: 'all',
  qualityGates: {},
  tasks: [],
  specialists: [],
  metrics: {},
  checkpoints: [],
  codebaseIndex: null,
  sessionStartTime: Date.now(),
  pollInterval: null,
  timerInterval: null,
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initChatInput();
  initKeyboardShortcuts();
  startStatusPolling();
  startTimer();

  // Listen for real-time updates from main process
  if (window.gaiaCode) {
    window.gaiaCode.onStateChange((newState) => {
      handleStateUpdate(newState);
    });

    window.gaiaCode.onAuditEntry((entry) => {
      addAuditEntry(entry);
    });

    window.gaiaCode.onQualityGateUpdate((data) => {
      updateQualityGates(data);
    });
  }
});

// ============================================================================
// Navigation
// ============================================================================

function initNavigation() {
  document.querySelectorAll('.nav-item[data-panel]').forEach((btn) => {
    btn.addEventListener('click', () => {
      switchPanel(btn.dataset.panel);

      // Update active nav item
      document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

function switchPanel(panelId) {
  state.activePanel = panelId;

  // Hide all panels
  document.querySelectorAll('.panel-view').forEach((p) => p.classList.remove('active'));

  // Show target panel
  const panel = document.getElementById(`panel-${panelId}`);
  if (panel) {
    panel.classList.add('active');
  }

  // Trigger data refresh for certain panels
  if (panelId === 'database') {
    loadDatabaseTables(state.activeDatabase);
  } else if (panelId === 'audit') {
    refreshAuditLog();
  } else if (panelId === 'metrics') {
    refreshMetrics();
  } else if (panelId === 'checkpoints') {
    refreshCheckpoints();
  } else if (panelId === 'specialists') {
    refreshSpecialists();
  }
}

// ============================================================================
// Chat Interface
// ============================================================================

function initChatInput() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');

  // Enable/disable send button
  input.addEventListener('input', () => {
    sendBtn.disabled = !input.value.trim();
    autoResizeTextarea(input);
  });

  // Send on Enter (Shift+Enter for newline)
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.value.trim()) {
        sendChatMessage();
      }
    }
  });

  sendBtn.addEventListener('click', sendChatMessage);
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;

  // Add user message to UI
  addMessageToUI('user', message);
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('chat-send').disabled = true;

  // Send to backend
  try {
    const response = await window.gaiaCode.sendMessage(message);

    if (response.error) {
      addMessageToUI('agent', `Error: ${response.error}`);
    } else if (response.reply) {
      addMessageToUI('agent', response.reply);
    } else {
      addMessageToUI('agent', 'Task received. The agent is processing...');
    }

    // Refresh status after sending
    refreshStatus();
  } catch (err) {
    addMessageToUI('agent', `Connection error: ${err.message}`);
  }
}

function addMessageToUI(role, content) {
  const container = document.getElementById('chat-messages');

  // Remove empty state if present
  const emptyState = container.querySelector('.chat-empty');
  if (emptyState) {
    emptyState.remove();
  }

  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const messageEl = document.createElement('div');
  messageEl.className = `message ${role}`;

  const avatarText = role === 'user' ? 'U' : 'G';

  // Process content for code blocks
  const processedContent = processMessageContent(content);

  messageEl.innerHTML = `
    <div class="message-avatar">${avatarText}</div>
    <div>
      <div class="message-body">${processedContent}</div>
      <div class="message-time">${timeStr}</div>
    </div>
  `;

  container.appendChild(messageEl);

  // Scroll to bottom
  container.scrollTop = container.scrollHeight;

  // Store in state
  state.chatMessages.push({ role, content, time: now });
}

function processMessageContent(content) {
  // Escape HTML
  let safe = content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks (```...```)
  safe = safe.replace(/```(\w*)\n?([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  // Inline code (`...`)
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Line breaks
  safe = safe.replace(/\n/g, '<br>');

  return safe;
}

function autoResizeTextarea(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
}

// ============================================================================
// Status Polling
// ============================================================================

function startStatusPolling() {
  refreshStatus();
  state.pollInterval = setInterval(refreshStatus, 3000);
}

async function refreshStatus() {
  if (!window.gaiaCode) return;

  try {
    const status = await window.gaiaCode.getStatus();

    if (status.error && status.offline) {
      setConnectionStatus('disconnected', 'IPC server offline');
      state.connected = false;
    } else {
      setConnectionStatus('connected', 'Connected');
      state.connected = true;

      // Update metrics
      if (status.progress) {
        updateProgressMetrics(status.progress);
      }

      // Update quality gates
      if (status.quality_gates) {
        updateQualityGates(status.quality_gates);
      }

      // Update task plan
      if (status.tasks) {
        updateTaskPlan(status.tasks);
      }

      // Update specialists
      if (status.specialists) {
        updateSpecialistsList(status.specialists);
      }
    }
  } catch (err) {
    setConnectionStatus('disconnected', 'Connection error');
    state.connected = false;
  }
}

function setConnectionStatus(status, text) {
  const barDot = document.getElementById('status-bar-dot');
  const barText = document.getElementById('status-bar-text');
  const indicatorDot = document.querySelector('#agent-status-indicator .status-dot');
  const indicatorText = document.getElementById('agent-status-text');

  barDot.className = `status-dot ${status}`;
  barText.textContent = text;

  if (indicatorDot) indicatorDot.className = `status-dot ${status}`;
  if (indicatorText) indicatorText.textContent = text;
}

// ============================================================================
// Timer
// ============================================================================

function startTimer() {
  state.timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    document.getElementById('status-bar-timer').textContent =
      `${minutes}:${String(seconds).padStart(2, '0')}`;
  }, 1000);
}

// ============================================================================
// Quality Gates
// ============================================================================

function updateQualityGates(gates) {
  state.qualityGates = gates;

  const container = document.getElementById('quality-gates-list');
  if (!container) return;

  const gateNames = Object.keys(gates);
  if (gateNames.length === 0) return;

  container.innerHTML = gateNames.map((name) => {
    const passed = gates[name];
    let iconClass, iconChar, statusText;

    if (passed === true) {
      iconClass = 'pass';
      iconChar = '\u2713';
      statusText = 'Passed';
    } else if (passed === false) {
      iconClass = 'fail';
      iconChar = '\u2717';
      statusText = 'Failed';
    } else if (passed === 'running') {
      iconClass = 'running';
      iconChar = '\u21BB';
      statusText = 'Running';
    } else {
      iconClass = 'pending';
      iconChar = '-';
      statusText = 'Pending';
    }

    return `
      <div class="gate-item">
        <span class="gate-icon ${iconClass}">${iconChar}</span>
        <span class="gate-name">${capitalize(name)}</span>
        <span class="gate-status">${statusText}</span>
      </div>
    `;
  }).join('');
}

// ============================================================================
// Task Plan
// ============================================================================

function updateTaskPlan(tasks) {
  state.tasks = tasks;

  const container = document.getElementById('task-progress-list');
  if (!container) return;

  if (!tasks || tasks.length === 0) {
    container.innerHTML = '<div style="font-size: var(--font-size-sm); color: var(--text-tertiary);">No active tasks</div>';
    return;
  }

  container.innerHTML = tasks.slice(0, 8).map((task) => {
    let iconChar;
    if (task.status === 'completed') iconChar = '\u2713';
    else if (task.status === 'in_progress') iconChar = '\u25B6';
    else if (task.status === 'failed') iconChar = '\u2717';
    else iconChar = '\u25CB';

    const descClass = task.status === 'completed' ? 'completed' : '';

    return `
      <div class="task-item">
        <span class="task-icon ${task.status}">${iconChar}</span>
        <span class="task-desc ${descClass}">${escapeHtml(truncate(task.description, 60))}</span>
      </div>
    `;
  }).join('');

  if (tasks.length > 8) {
    container.innerHTML += `<div style="font-size: var(--font-size-xs); color: var(--text-tertiary); padding-left: 22px;">... and ${tasks.length - 8} more</div>`;
  }

  // Also update the Plan panel
  updatePlanPanel(tasks);
}

function updatePlanPanel(tasks) {
  const container = document.getElementById('plan-content');
  if (!container || !tasks || tasks.length === 0) return;

  let html = '<table class="data-table"><thead><tr>';
  html += '<th>Status</th><th>Description</th><th>Owner</th><th>Created</th>';
  html += '</tr></thead><tbody>';

  tasks.forEach((task) => {
    let statusBadge;
    if (task.status === 'completed') statusBadge = '<span style="color: var(--color-success);">\u2713 Completed</span>';
    else if (task.status === 'in_progress') statusBadge = '<span style="color: var(--color-warning);">\u25B6 In Progress</span>';
    else if (task.status === 'failed') statusBadge = '<span style="color: var(--color-error);">\u2717 Failed</span>';
    else statusBadge = '<span style="color: var(--color-pending);">\u25CB Pending</span>';

    html += `<tr>
      <td>${statusBadge}</td>
      <td>${escapeHtml(task.description || '')}</td>
      <td>${escapeHtml(task.owner || '--')}</td>
      <td>${task.created_at ? new Date(task.created_at).toLocaleString() : '--'}</td>
    </tr>`;
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

async function refreshPlan() {
  if (!window.gaiaCode) return;
  const result = await window.gaiaCode.getPlan();
  if (result.tasks) {
    updateTaskPlan(result.tasks);
  }
}

// ============================================================================
// Progress Metrics
// ============================================================================

function updateProgressMetrics(progress) {
  setTextContent('metric-elapsed', formatDuration(progress.elapsed_seconds || 0));
  setTextContent('metric-steps', String(progress.current_step || 0));
  setTextContent('metric-tasks', String(progress.total_tasks || 0));
  setTextContent('metric-depth', String(progress.recursion_depth || 0));
}

// ============================================================================
// Database Inspector
// ============================================================================

function selectDatabase(element, dbName) {
  state.activeDatabase = dbName;
  state.activeDatabaseTable = null;

  // Update active state
  document.querySelectorAll('.db-item').forEach((el) => el.classList.remove('active'));
  element.classList.add('active');

  // Load tables
  loadDatabaseTables(dbName);

  // Update SQL placeholder
  document.getElementById('sql-input').placeholder = `SELECT * FROM ... (${dbName}.db) LIMIT 50`;
}

async function loadDatabaseTables(dbName) {
  if (!window.gaiaCode) return;

  const result = await window.gaiaCode.dbTables(dbName);
  const container = document.getElementById(`db-tables-${dbName}`);
  if (!container) return;

  if (result.tables && result.tables.length > 0) {
    container.innerHTML = result.tables.map((table) => `
      <div class="db-table-item" onclick="browseTable('${dbName}', '${table}')">
        ${table}
      </div>
    `).join('');
  } else {
    container.innerHTML = '';
  }
}

async function browseTable(dbName, tableName) {
  state.activeDatabaseTable = tableName;

  // Update active table
  document.querySelectorAll('.db-table-item').forEach((el) => el.classList.remove('active'));
  event.target.classList.add('active');

  // Set SQL query
  const sqlInput = document.getElementById('sql-input');
  sqlInput.value = `SELECT * FROM ${tableName} LIMIT 50`;

  // Execute
  executeSqlQuery();
}

async function executeSqlQuery() {
  if (!window.gaiaCode) return;

  const sqlInput = document.getElementById('sql-input');
  const query = sqlInput.value.trim();
  if (!query) return;

  const container = document.getElementById('db-results');
  container.innerHTML = '<div class="empty-state"><div class="loading-spinner"></div><p>Executing query...</p></div>';

  const result = await window.gaiaCode.dbQuery(state.activeDatabase, query);

  if (result.error) {
    container.innerHTML = `
      <div class="empty-state">
        <p style="color: var(--color-error); font-family: var(--font-mono); font-size: var(--font-size-sm);">
          Error: ${escapeHtml(result.error)}
        </p>
      </div>
    `;
    return;
  }

  if (!result.rows || result.rows.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>No Results</h3>
        <p>Query returned 0 rows.</p>
      </div>
    `;
    return;
  }

  renderDataTable(container, result.columns, result.rows);
}

function renderDataTable(container, columns, rows) {
  let html = '<table class="data-table"><thead><tr>';
  columns.forEach((col) => {
    html += `<th>${escapeHtml(col)}</th>`;
  });
  html += '</tr></thead><tbody>';

  rows.forEach((row) => {
    html += '<tr>';
    row.forEach((cell) => {
      if (cell === null || cell === undefined) {
        html += '<td><span class="null-value">NULL</span></td>';
      } else {
        const cellStr = String(cell);
        const display = cellStr.length > 100 ? cellStr.substring(0, 97) + '...' : cellStr;
        html += `<td title="${escapeHtml(cellStr)}">${escapeHtml(display)}</td>`;
      }
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  html += `<div style="padding: var(--space-2) var(--space-3); font-size: var(--font-size-xs); color: var(--text-tertiary);">${rows.length} row(s)</div>`;

  container.innerHTML = html;
}

// ============================================================================
// Audit Log
// ============================================================================

async function refreshAuditLog() {
  if (!window.gaiaCode) return;

  const filter = state.auditFilter === 'all' ? null : state.auditFilter;
  const result = await window.gaiaCode.getAuditLog(200, 0, filter);

  if (result.entries) {
    renderAuditLog(result.entries);
  }
}

function addAuditEntry(entry) {
  state.auditLog.unshift(entry);
  if (state.activePanel === 'audit') {
    renderAuditLog(state.auditLog);
  }
}

function renderAuditLog(entries) {
  state.auditLog = entries;
  const container = document.getElementById('audit-entries');

  if (!entries || entries.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>No Audit Entries</h3>
        <p>Audit entries appear as the agent processes tasks.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = entries.map((entry) => {
    const time = entry.timestamp
      ? new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : '--:--:--';

    const typeClass = getAuditTypeClass(entry.action_type || entry.type || '');
    const typeName = entry.action_type || entry.type || 'Unknown';
    const details = typeof entry.details === 'object'
      ? JSON.stringify(entry.details, null, 0)
      : String(entry.details || '');

    return `
      <div class="audit-entry">
        <span class="audit-timestamp">${time}</span>
        <span class="audit-type ${typeClass}">${typeName}</span>
        <span class="audit-details">${escapeHtml(truncate(details, 200))}</span>
      </div>
    `;
  }).join('');
}

function getAuditTypeClass(type) {
  const upper = type.toUpperCase();
  if (upper.includes('TASK')) return 'task';
  if (upper.includes('PLAN')) return 'plan';
  if (upper.includes('QUALITY') || upper.includes('GATE')) return 'quality';
  if (upper.includes('ERROR') || upper.includes('FAIL')) return 'error';
  if (upper.includes('ESCALAT')) return 'escalation';
  return 'task';
}

function filterAudit(element, filter) {
  state.auditFilter = filter;

  // Update active filter chip
  document.querySelectorAll('.filter-chip').forEach((chip) => chip.classList.remove('active'));
  element.classList.add('active');

  refreshAuditLog();
}

// ============================================================================
// Codebase Index
// ============================================================================

async function refreshCodebaseIndex() {
  if (!window.gaiaCode) return;

  const result = await window.gaiaCode.getCodebaseIndex();

  if (result.stats) {
    setTextContent('stat-files', String(result.stats.files_indexed || 0));
    setTextContent('stat-symbols', String(result.stats.symbols_found || 0));
    setTextContent('stat-deps', String(result.stats.dependencies || 0));
    setTextContent('stat-issues', String(result.stats.issues_found || 0));
  }

  if (result.tree) {
    renderCodebaseTree(result.tree);
  }

  if (result.issues) {
    renderCodebaseIssues(result.issues);
  }

  if (result.symbols) {
    renderCodebaseSymbols(result.symbols);
  }

  state.codebaseIndex = result;
}

function renderCodebaseTree(tree) {
  const container = document.getElementById('codebase-tree');
  if (!tree || tree.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding: var(--space-4);"><p style="font-size: var(--font-size-xs);">No files indexed</p></div>';
    return;
  }

  container.innerHTML = tree.map((item) => {
    const isDir = item.type === 'directory';
    const icon = isDir ? '\uD83D\uDCC1' : '\uD83D\uDCC4';
    const className = isDir ? 'directory' : 'file';

    return `<div class="tree-node ${className}" title="${escapeHtml(item.path || '')}">${icon} ${escapeHtml(item.name)}</div>`;
  }).join('');
}

function renderCodebaseIssues(issues) {
  const container = document.getElementById('codebase-tab-issues');
  if (!issues || issues.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No issues detected</p></div>';
    return;
  }

  container.innerHTML = issues.map((issue) => `
    <div class="issue-item">
      <span class="issue-severity ${issue.severity || 'info'}">${(issue.severity || 'info').toUpperCase()}</span>
      <div>
        <div class="issue-text">${escapeHtml(issue.message || issue.description || '')}</div>
        ${issue.file ? `<div class="issue-file">${escapeHtml(issue.file)}</div>` : ''}
      </div>
    </div>
  `).join('');
}

function renderCodebaseSymbols(symbols) {
  const container = document.getElementById('codebase-tab-symbols');
  if (!symbols || symbols.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No symbols indexed</p></div>';
    return;
  }

  const columns = ['Name', 'Type', 'File', 'Line'];
  const rows = symbols.slice(0, 100).map((s) => [s.name, s.type, s.file || '', String(s.line || '')]);
  renderDataTable(container, columns, rows);
}

function switchCodebaseTab(element, tab) {
  // Update tab UI
  element.parentElement.querySelectorAll('.tab-item').forEach((t) => t.classList.remove('active'));
  element.classList.add('active');

  // Show/hide tab content
  ['graph', 'issues', 'symbols'].forEach((t) => {
    const el = document.getElementById(`codebase-tab-${t}`);
    if (el) el.style.display = t === tab ? '' : 'none';
  });
}

// ============================================================================
// Specialists
// ============================================================================

async function refreshSpecialists() {
  if (!window.gaiaCode) return;

  const result = await window.gaiaCode.getSpecialists();
  if (result.specialists) {
    renderSpecialistsPanel(result.specialists);
    updateSpecialistsList(result.specialists);
  }
}

function updateSpecialistsList(specialists) {
  state.specialists = specialists;

  const container = document.getElementById('specialists-list');
  if (!container || !specialists) return;

  const specialistColors = {
    debugger: 'var(--specialist-debugger)',
    security: 'var(--specialist-security)',
    refactoring: 'var(--specialist-refactoring)',
    testing: 'var(--specialist-testing)',
    documentation: 'var(--specialist-docs)',
    performance: 'var(--specialist-performance)',
    architecture: 'var(--specialist-architecture)',
  };

  container.innerHTML = specialists.map((s) => {
    const color = specialistColors[s.name?.toLowerCase()] || 'var(--text-tertiary)';
    const activeClass = s.active ? 'active' : '';
    const status = s.active ? 'Active' : 'Idle';

    return `
      <div class="specialist-item">
        <span class="specialist-dot ${activeClass}" style="background: ${color};"></span>
        <span class="specialist-name">${escapeHtml(capitalize(s.name || 'Unknown'))}</span>
        <span class="specialist-status">${status}</span>
      </div>
    `;
  }).join('');
}

function renderSpecialistsPanel(specialists) {
  const container = document.getElementById('specialists-content');
  if (!container || !specialists) return;

  const columns = ['Name', 'Status', 'Description', 'Tasks Completed', 'Last Used'];
  const rows = specialists.map((s) => [
    capitalize(s.name || ''),
    s.active ? 'Active' : 'Idle',
    s.description || '',
    String(s.tasks_completed || 0),
    s.last_used || '--',
  ]);

  renderDataTable(container, columns, rows);
}

// ============================================================================
// Performance Metrics
// ============================================================================

async function refreshMetrics() {
  if (!window.gaiaCode) return;

  const result = await window.gaiaCode.getMetrics();
  if (!result || result.error) return;

  setTextContent('perf-session-time', formatDuration(result.session_time || 0));
  setTextContent('perf-task-time', formatDuration(result.task_time || 0));
  setTextContent('perf-total-steps', String(result.total_steps || 0));
  setTextContent('perf-gate-runs', String(result.gate_runs || 0));
  setTextContent('perf-retries', String(result.retries || 0));
  setTextContent('perf-escalations', String(result.escalations || 0));
  setTextContent('perf-agent-calls', String(result.agent_calls || 0));
  setTextContent('perf-insights', String(result.insights_stored || 0));
}

// ============================================================================
// Checkpoint Manager
// ============================================================================

async function refreshCheckpoints() {
  if (!window.gaiaCode) return;

  const result = await window.gaiaCode.listCheckpoints();
  if (result.checkpoints) {
    renderCheckpoints(result.checkpoints);
  }
}

async function createCheckpoint() {
  if (!window.gaiaCode) return;

  const result = await window.gaiaCode.createCheckpoint();
  if (result.success) {
    refreshCheckpoints();
  }
}

async function restoreCheckpoint(id) {
  if (!window.gaiaCode) return;

  if (!confirm('Restore this checkpoint? Current state will be replaced.')) return;

  const result = await window.gaiaCode.restoreCheckpoint(id);
  if (result.success) {
    refreshStatus();
    refreshCheckpoints();
  }
}

function renderCheckpoints(checkpoints) {
  const container = document.getElementById('checkpoint-list');
  state.checkpoints = checkpoints;

  if (!checkpoints || checkpoints.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>No Checkpoints</h3>
        <p>Checkpoints allow you to save and restore the agent's state.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = checkpoints.map((cp) => {
    const time = cp.timestamp ? new Date(cp.timestamp).toLocaleString() : 'Unknown';
    const id = cp.id || cp.timestamp || 'unknown';

    return `
      <div class="checkpoint-card">
        <div class="checkpoint-header">
          <span class="checkpoint-time">${time}</span>
          <span class="checkpoint-id">${escapeHtml(truncate(String(id), 20))}</span>
        </div>
        <div class="checkpoint-meta">
          <span>Tasks: ${cp.task_count || 0}</span>
          <span>Audit entries: ${cp.audit_count || 0}</span>
          ${cp.task_start ? `<span>Task started: ${new Date(cp.task_start).toLocaleTimeString()}</span>` : ''}
        </div>
        <div class="checkpoint-actions">
          <button class="btn btn-sm" onclick="restoreCheckpoint('${escapeHtml(String(id))}')">Restore</button>
        </div>
      </div>
    `;
  }).join('');
}

// ============================================================================
// State Update Handler
// ============================================================================

function handleStateUpdate(newState) {
  if (newState.quality_gates) {
    updateQualityGates(newState.quality_gates);
  }

  if (newState.tasks) {
    updateTaskPlan(newState.tasks);
  }

  if (newState.progress) {
    updateProgressMetrics(newState.progress);
  }

  if (newState.specialists) {
    updateSpecialistsList(newState.specialists);
  }

  if (newState.audit_entry) {
    addAuditEntry(newState.audit_entry);
  }

  if (newState.chat_reply) {
    addMessageToUI('agent', newState.chat_reply);
  }
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + 1-9 for panel switching
    if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '9') {
      e.preventDefault();
      const panels = ['chat', 'plan', 'database', 'audit', 'codebase', 'specialists', 'metrics', 'checkpoints', 'settings'];
      const index = parseInt(e.key) - 1;
      if (index < panels.length) {
        const navBtn = document.querySelector(`.nav-item[data-panel="${panels[index]}"]`);
        if (navBtn) {
          navBtn.click();
        }
      }
    }

    // Ctrl/Cmd + Enter to send in SQL editor
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && document.activeElement.id === 'sql-input') {
      e.preventDefault();
      executeSqlQuery();
    }

    // Escape to focus chat input
    if (e.key === 'Escape') {
      document.getElementById('chat-input')?.focus();
    }
  });
}

// ============================================================================
// Sidebar Section Toggle
// ============================================================================

function toggleSection(header) {
  const section = header.parentElement;
  section.classList.toggle('collapsed');
}

// ============================================================================
// Utility Functions
// ============================================================================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function truncate(text, maxLen) {
  if (!text) return '';
  return text.length > maxLen ? text.substring(0, maxLen - 3) + '...' : text;
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function setTextContent(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function formatDuration(seconds) {
  if (typeof seconds !== 'number' || seconds < 0) return '--';

  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  } else {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }
}
