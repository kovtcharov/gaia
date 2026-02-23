// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Dashboard Overview
 *
 * Renders the main overview page with statistics, summaries, analysis,
 * and GitHub-style data visualizations across all databases in the workspace.
 */

const Dashboard = {

  /**
   * Load and render the full dashboard.
   */
  async load() {
    const container = document.getElementById('dashboard-content');
    if (!container) return;

    container.innerHTML = '<div class="empty-state"><span class="loading-spinner"></span> Loading dashboard...</div>';

    try {
      // Step 1: Load database list to get file stats
      const dbResult = await window.dbAPI.listDatabases(AppState.workspacePath);
      if (!dbResult.success) {
        container.innerHTML = `<div class="empty-state">Could not load workspace: ${escapeHtml(dbResult.error)}</div>`;
        return;
      }

      AppState.databases = dbResult.databases;
      const existingDbs = dbResult.databases.filter(d => d.exists);

      // Step 2: Compute overview stats
      const totalSize = existingDbs.reduce((sum, db) => sum + db.sizeBytes, 0);
      const lastActivity = existingDbs.reduce((latest, db) => {
        if (!db.lastModified) return latest;
        const t = new Date(db.lastModified);
        return (!latest || t > latest) ? t : latest;
      }, null);

      // Step 3: Query individual databases for summary data
      const dashData = {
        totalSize,
        dbCount: existingDbs.length,
        lastActivity,
        dbStats: [],
        recentErrors: [],
        activeTasks: [],
        recentInsights: [],
        contextWarnings: [],
        topTools: [],
        errorTrend: [],
        contextUsage: [],
        activityHeatmap: [],
        trendStats: null,
      };

      // Gather row counts per database
      for (const db of existingDbs) {
        try {
          const tablesResult = await window.dbAPI.getTables(db.path);
          if (tablesResult.success) {
            const totalRows = tablesResult.tables.reduce((sum, t) => sum + (t.rowCount || 0), 0);
            dashData.dbStats.push({
              name: db.name,
              label: db.label || db.name.replace('.db', ''),
              sizeBytes: db.sizeBytes,
              lastModified: db.lastModified,
              tableCount: tablesResult.tables.length,
              totalRows,
            });
          }
        } catch (e) {
          // Skip databases that fail
        }
      }

      // Query logs.db for recent errors, activity, and trends
      const logsDb = existingDbs.find(d => d.name === 'logs.db');
      if (logsDb) {
        try {
          const errResult = await window.dbAPI.executeSQL(
            logsDb.path,
            `SELECT level, message, timestamp, step_number FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL') ORDER BY timestamp DESC LIMIT 10`,
            true
          );
          if (errResult.success && errResult.rows) {
            dashData.recentErrors = errResult.rows;
          }
        } catch (e) { /* ignore */ }

        // Context warnings
        try {
          const ctxResult = await window.dbAPI.executeSQL(
            logsDb.path,
            `SELECT message, timestamp, step_number FROM runtime_logs WHERE message LIKE '%context%' OR message LIKE '%token%' ORDER BY timestamp DESC LIMIT 10`,
            true
          );
          if (ctxResult.success && ctxResult.rows) {
            dashData.contextWarnings = ctxResult.rows;
          }
        } catch (e) { /* ignore */ }

        // Context usage over time (token tracking)
        try {
          const usageResult = await window.dbAPI.executeSQL(
            logsDb.path,
            `SELECT step_number, context_tokens, max_context_tokens, timestamp FROM runtime_logs WHERE context_tokens IS NOT NULL ORDER BY step_number ASC LIMIT 100`,
            true
          );
          if (usageResult.success && usageResult.rows) {
            dashData.contextUsage = usageResult.rows;
          }
        } catch (e) { /* ignore */ }

        // Error rate trend (errors per step range)
        try {
          const trendResult = await window.dbAPI.executeSQL(
            logsDb.path,
            `SELECT step_number, level, COUNT(*) as cnt FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL', 'WARNING') GROUP BY step_number, level ORDER BY step_number ASC LIMIT 50`,
            true
          );
          if (trendResult.success && trendResult.rows) {
            dashData.errorTrend = trendResult.rows;
          }
        } catch (e) { /* ignore */ }

        // Activity heatmap: count log entries per day for the last 7 days
        try {
          const heatmapResult = await window.dbAPI.executeSQL(
            logsDb.path,
            `SELECT DATE(timestamp) as date, COUNT(*) as count FROM runtime_logs WHERE timestamp >= DATE('now', '-7 days') GROUP BY DATE(timestamp) ORDER BY date ASC`,
            true
          );
          if (heatmapResult.success && heatmapResult.rows) {
            dashData.activityHeatmap = heatmapResult.rows;
          }
        } catch (e) { /* ignore */ }

        // Trend stats: total logs, error rate, context usage -- last 24h vs previous 24h
        try {
          dashData.trendStats = await this._computeTrendStats(logsDb.path);
        } catch (e) { /* ignore */ }
      }

      // Query plan.db for active tasks
      const planDb = existingDbs.find(d => d.name === 'plan.db');
      if (planDb) {
        try {
          const taskResult = await window.dbAPI.executeSQL(
            planDb.path,
            `SELECT id, description, status, priority FROM tasks ORDER BY CASE status WHEN 'in_progress' THEN 0 WHEN 'pending' THEN 1 WHEN 'completed' THEN 2 WHEN 'failed' THEN 3 ELSE 4 END, id DESC LIMIT 10`,
            true
          );
          if (taskResult.success && taskResult.rows) {
            dashData.activeTasks = taskResult.rows;
          }
        } catch (e) { /* ignore */ }
      }

      // Query knowledge.db for recent insights
      const knowledgeDb = existingDbs.find(d => d.name === 'knowledge.db');
      if (knowledgeDb) {
        try {
          const insightResult = await window.dbAPI.executeSQL(
            knowledgeDb.path,
            `SELECT id, content, category, confidence, created_at FROM insights ORDER BY created_at DESC LIMIT 8`,
            true
          );
          if (insightResult.success && insightResult.rows) {
            dashData.recentInsights = insightResult.rows;
          }
        } catch (e) { /* ignore */ }
      }

      // Query tools.db for top tools (get top 10 for the bar chart)
      const toolsDb = existingDbs.find(d => d.name === 'tools.db');
      if (toolsDb) {
        try {
          const toolResult = await window.dbAPI.executeSQL(
            toolsDb.path,
            `SELECT name, usage_count, success_count, avg_duration_ms FROM tools ORDER BY usage_count DESC LIMIT 10`,
            true
          );
          if (toolResult.success && toolResult.rows) {
            dashData.topTools = toolResult.rows;
          }
        } catch (e) { /* ignore */ }

        // Also get total tool call count for trend stats
        if (dashData.trendStats) {
          try {
            const totalToolResult = await window.dbAPI.executeSQL(
              toolsDb.path,
              `SELECT COALESCE(SUM(usage_count), 0) as total FROM tools`,
              true
            );
            if (totalToolResult.success && totalToolResult.rows && totalToolResult.rows.length > 0) {
              dashData.trendStats.totalToolCalls = totalToolResult.rows[0].total || 0;
            }
          } catch (e) { /* ignore */ }
        }
      }

      AppState.dashboardData = dashData;
      this.render(dashData);
    } catch (err) {
      container.innerHTML = `<div class="empty-state">Dashboard error: ${escapeHtml(err.message)}</div>`;
    }
  },

  /**
   * Compute trend stats comparing last 24h vs previous 24h.
   * @param {string} logsDbPath
   * @returns {Object}
   */
  async _computeTrendStats(logsDbPath) {
    const stats = {
      totalLogs24h: 0,
      totalLogsPrev24h: 0,
      errors24h: 0,
      errorsPrev24h: 0,
      avgContext24h: 0,
      avgContextPrev24h: 0,
      totalToolCalls: 0,
    };

    // Total logs last 24h
    try {
      const r = await window.dbAPI.executeSQL(
        logsDbPath,
        `SELECT COUNT(*) as cnt FROM runtime_logs WHERE timestamp >= DATETIME('now', '-1 day')`,
        true
      );
      if (r.success && r.rows && r.rows.length > 0) stats.totalLogs24h = r.rows[0].cnt || 0;
    } catch (e) { /* ignore */ }

    // Total logs previous 24h (24h-48h ago)
    try {
      const r = await window.dbAPI.executeSQL(
        logsDbPath,
        `SELECT COUNT(*) as cnt FROM runtime_logs WHERE timestamp >= DATETIME('now', '-2 days') AND timestamp < DATETIME('now', '-1 day')`,
        true
      );
      if (r.success && r.rows && r.rows.length > 0) stats.totalLogsPrev24h = r.rows[0].cnt || 0;
    } catch (e) { /* ignore */ }

    // Errors last 24h
    try {
      const r = await window.dbAPI.executeSQL(
        logsDbPath,
        `SELECT COUNT(*) as cnt FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL') AND timestamp >= DATETIME('now', '-1 day')`,
        true
      );
      if (r.success && r.rows && r.rows.length > 0) stats.errors24h = r.rows[0].cnt || 0;
    } catch (e) { /* ignore */ }

    // Errors previous 24h
    try {
      const r = await window.dbAPI.executeSQL(
        logsDbPath,
        `SELECT COUNT(*) as cnt FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL') AND timestamp >= DATETIME('now', '-2 days') AND timestamp < DATETIME('now', '-1 day')`,
        true
      );
      if (r.success && r.rows && r.rows.length > 0) stats.errorsPrev24h = r.rows[0].cnt || 0;
    } catch (e) { /* ignore */ }

    // Average context tokens last 24h
    try {
      const r = await window.dbAPI.executeSQL(
        logsDbPath,
        `SELECT AVG(context_tokens) as avg_ctx FROM runtime_logs WHERE context_tokens IS NOT NULL AND timestamp >= DATETIME('now', '-1 day')`,
        true
      );
      if (r.success && r.rows && r.rows.length > 0) stats.avgContext24h = r.rows[0].avg_ctx || 0;
    } catch (e) { /* ignore */ }

    // Average context tokens previous 24h
    try {
      const r = await window.dbAPI.executeSQL(
        logsDbPath,
        `SELECT AVG(context_tokens) as avg_ctx FROM runtime_logs WHERE context_tokens IS NOT NULL AND timestamp >= DATETIME('now', '-2 days') AND timestamp < DATETIME('now', '-1 day')`,
        true
      );
      if (r.success && r.rows && r.rows.length > 0) stats.avgContextPrev24h = r.rows[0].avg_ctx || 0;
    } catch (e) { /* ignore */ }

    return stats;
  },

  /**
   * Render the dashboard with gathered data.
   * @param {Object} data - Dashboard data
   */
  render(data) {
    const container = document.getElementById('dashboard-content');
    if (!container) return;

    container.innerHTML = '';

    // ================================================================
    // Row 1: Stats Cards with Trends + Donut Chart (side by side)
    // ================================================================
    const vizRow1 = document.createElement('div');
    vizRow1.className = 'dash-viz-row';

    // --- Stats Cards ---
    const { section: statsSection, body: statsBody } = this._createSection('Stats (Last 24h)', 'trend-stats');
    statsSection.classList.add('dash-viz-flex-grow');
    statsBody.id = 'chart-stats-cards';
    this._renderTrendStats(data);
    vizRow1.appendChild(statsSection);

    // --- Donut Chart ---
    if (data.dbStats.length > 0) {
      const { section: donutSection, body: donutBody } = this._createSection('Database Size Breakdown', 'size-donut');
      donutBody.innerHTML = `
        <div class="donut-chart-layout">
          <div class="donut-chart-canvas-wrap" style="position:relative">
            <canvas id="chart-donut-canvas"></canvas>
          </div>
          <div class="donut-chart-legend" id="chart-donut-legend"></div>
        </div>
      `;
      vizRow1.appendChild(donutSection);
    }

    container.appendChild(vizRow1);

    // Draw the donut chart after DOM is ready
    if (data.dbStats.length > 0) {
      requestAnimationFrame(() => {
        const donutData = data.dbStats.map(db => ({
          label: db.label,
          value: db.sizeBytes,
        }));
        Charts.drawDonutChart('chart-donut-canvas', donutData, {
          size: 170,
          lineWidth: 26,
          centerLabel: formatBytes(data.totalSize),
          centerSub: 'Total',
        });
        Charts.drawDonutLegend('chart-donut-legend', donutData);
      });
    }

    // ================================================================
    // Row 2: Activity Heatmap + Tool Usage Bar Chart
    // ================================================================
    const vizRow2 = document.createElement('div');
    vizRow2.className = 'dash-viz-row';

    // --- Activity Heatmap ---
    const { section: heatSection, body: heatBody } = this._createSection('Activity (Last 7 Days)', 'activity-heatmap');
    heatSection.classList.add('dash-viz-flex-grow');
    heatBody.innerHTML = '<div id="chart-heatmap" class="chart-heatmap-container"></div>';
    vizRow2.appendChild(heatSection);

    // --- Top Tools Bar Chart ---
    const { section: barSection, body: barBody } = this._createSection('Top Tools', 'tools-bar');
    barBody.innerHTML = '<div id="chart-tools-bar" class="chart-bar-container"></div>';
    vizRow2.appendChild(barSection);

    container.appendChild(vizRow2);

    // Draw heatmap and bar chart after DOM ready
    requestAnimationFrame(() => {
      Charts.drawHeatmap('chart-heatmap', data.activityHeatmap || []);
      const barData = (data.topTools || []).map(t => ({
        label: t.name,
        value: t.usage_count || 0,
      }));
      Charts.drawBarChart('chart-tools-bar', barData);
    });

    // ================================================================
    // Overview Card (simpler now that stats cards exist above)
    // ================================================================
    const { section: overviewSection, body: overviewBody } = this._createSection('Overview', 'overview');
    overviewBody.innerHTML = `
      <div class="dash-overview-grid">
        <div class="dash-stat-card">
          <div class="dash-stat-value">${escapeHtml(formatBytes(data.totalSize))}</div>
          <div class="dash-stat-label">Total Size</div>
        </div>
        <div class="dash-stat-card">
          <div class="dash-stat-value">${data.dbCount}</div>
          <div class="dash-stat-label">Databases</div>
        </div>
        <div class="dash-stat-card">
          <div class="dash-stat-value">${data.dbStats.reduce((s, d) => s + d.totalRows, 0).toLocaleString()}</div>
          <div class="dash-stat-label">Total Rows</div>
        </div>
        <div class="dash-stat-card">
          <div class="dash-stat-value">${data.lastActivity ? formatRelativeTime(data.lastActivity) : 'N/A'}</div>
          <div class="dash-stat-label">Last Activity</div>
        </div>
      </div>
      <div class="dash-workspace-path">Workspace: ${escapeHtml(AppState.workspacePath)}</div>
    `;
    container.appendChild(overviewSection);

    // === Activity History ===
    container.appendChild(HistoryLog.renderDashboardHistory());

    // === Database Stats Table ===
    if (data.dbStats.length > 0) {
      const { section: dbSection, body: dbBody } = this._createSection('Database Details', 'db-details');
      let tableHtml = `<table class="dash-table">
        <thead><tr><th>Database</th><th>Size</th><th>Tables</th><th>Rows</th><th>Last Modified</th></tr></thead>
        <tbody>`;
      for (const db of data.dbStats) {
        tableHtml += `<tr>
          <td class="dash-db-name">${escapeHtml(db.label)}</td>
          <td>${escapeHtml(formatBytes(db.sizeBytes))}</td>
          <td>${db.tableCount}</td>
          <td>${db.totalRows.toLocaleString()}</td>
          <td>${db.lastModified ? formatRelativeTime(db.lastModified) : '--'}</td>
        </tr>`;
      }
      tableHtml += '</tbody></table>';
      dbBody.innerHTML = tableHtml;
      container.appendChild(dbSection);
    }

    // === Two column layout for summaries ===
    const summaryRow = document.createElement('div');
    summaryRow.className = 'dash-two-col';

    // --- Recent Errors ---
    const { section: errSection, body: errBody } = this._createSection('Recent Errors', 'errors');
    if (data.recentErrors.length > 0) {
      let errHtml = '<div class="dash-list">';
      for (const err of data.recentErrors) {
        const levelClass = (err.level || '').toUpperCase();
        errHtml += `<div class="dash-list-item dash-list-item-error">
          <span class="level-badge ${escapeHtml(levelClass)}">${escapeHtml(err.level)}</span>
          <span class="dash-list-msg">${escapeHtml(truncate(err.message, 100))}</span>
          <span class="dash-list-meta">${err.step_number != null ? 'Step ' + err.step_number : ''} ${err.timestamp ? formatRelativeTime(err.timestamp) : ''}</span>
        </div>`;
      }
      errHtml += '</div>';
      errBody.innerHTML = errHtml;
    } else {
      errBody.innerHTML = '<div class="dash-empty">No recent errors</div>';
    }
    summaryRow.appendChild(errSection);

    // --- Active Tasks ---
    const { section: taskSection, body: taskBody } = this._createSection('Active Tasks', 'tasks');
    if (data.activeTasks.length > 0) {
      let taskHtml = '<div class="dash-list">';
      for (const task of data.activeTasks) {
        const statusIcon = this._getStatusIcon(task.status);
        taskHtml += `<div class="dash-list-item">
          <span class="status-badge ${escapeHtml(task.status || '')}">${statusIcon} ${escapeHtml(task.status || 'unknown')}</span>
          <span class="dash-list-msg">${escapeHtml(truncate(task.description, 80))}</span>
          ${task.priority ? `<span class="dash-list-meta">P${escapeHtml(String(task.priority))}</span>` : ''}
        </div>`;
      }
      taskHtml += '</div>';
      taskBody.innerHTML = taskHtml;
    } else {
      taskBody.innerHTML = '<div class="dash-empty">No tasks found</div>';
    }
    summaryRow.appendChild(taskSection);
    container.appendChild(summaryRow);

    // === Second two column row ===
    const summaryRow2 = document.createElement('div');
    summaryRow2.className = 'dash-two-col';

    // --- Recent Insights ---
    const { section: insSection, body: insBody } = this._createSection('Recent Insights', 'insights');
    if (data.recentInsights.length > 0) {
      let insHtml = '<div class="dash-list">';
      for (const ins of data.recentInsights) {
        insHtml += `<div class="dash-list-item">
          ${ins.category ? `<span class="dash-category-badge">${escapeHtml(ins.category)}</span>` : ''}
          <span class="dash-list-msg">${escapeHtml(truncate(ins.content, 100))}</span>
          <span class="dash-list-meta">${ins.confidence != null ? Math.round(ins.confidence * 100) + '%' : ''} ${ins.created_at ? formatRelativeTime(ins.created_at) : ''}</span>
        </div>`;
      }
      insHtml += '</div>';
      insBody.innerHTML = insHtml;
    } else {
      insBody.innerHTML = '<div class="dash-empty">No insights yet</div>';
    }
    summaryRow2.appendChild(insSection);

    // --- Top Tools (original list view, kept as secondary) ---
    const { section: toolSection, body: toolBody } = this._createSection('Tool Success Rates', 'tools-detail');
    if (data.topTools.length > 0) {
      let toolHtml = '<div class="dash-list">';
      const maxUsage = Math.max(...data.topTools.map(t => t.usage_count || 0), 1);
      for (const tool of data.topTools) {
        const pct = Math.round(((tool.usage_count || 0) / maxUsage) * 100);
        const successRate = tool.usage_count > 0 && tool.success_count != null
          ? Math.round((tool.success_count / tool.usage_count) * 100) + '%'
          : '--';
        toolHtml += `<div class="dash-list-item dash-tool-item">
          <span class="dash-tool-name">${escapeHtml(tool.name)}</span>
          <div class="dash-tool-bar-container">
            <div class="dash-tool-bar" style="width: ${pct}%"></div>
          </div>
          <span class="dash-tool-count">${(tool.usage_count || 0).toLocaleString()} uses</span>
          <span class="dash-tool-success">${successRate}</span>
        </div>`;
      }
      toolHtml += '</div>';
      toolBody.innerHTML = toolHtml;
    } else {
      toolBody.innerHTML = '<div class="dash-empty">No tool usage data</div>';
    }
    summaryRow2.appendChild(toolSection);
    container.appendChild(summaryRow2);

    // === Context Usage Sparkline ===
    if (data.contextUsage.length > 0) {
      const { section: ctxSection, body: ctxBody } = this._createSection('Context Token Usage', 'context');
      ctxBody.appendChild(this._renderContextChart(data.contextUsage));
      container.appendChild(ctxSection);
    }

    // === Context Warnings ===
    if (data.contextWarnings.length > 0) {
      const { section: warnSection, body: warnBody } = this._createSection('Context Warnings', 'ctx-warnings');
      let warnHtml = '<div class="dash-list">';
      for (const w of data.contextWarnings.slice(0, 5)) {
        warnHtml += `<div class="dash-list-item dash-list-item-warning">
          <span class="dash-list-msg">${escapeHtml(truncate(w.message, 120))}</span>
          <span class="dash-list-meta">${w.step_number != null ? 'Step ' + w.step_number : ''} ${w.timestamp ? formatRelativeTime(w.timestamp) : ''}</span>
        </div>`;
      }
      warnHtml += '</div>';
      warnBody.innerHTML = warnHtml;
      container.appendChild(warnSection);
    }
  },

  /**
   * Render trend stats cards using the Charts module.
   * @param {Object} data - Dashboard data
   */
  _renderTrendStats(data) {
    const ts = data.trendStats;
    const stats = [];

    // Total Logs
    const logsDiff = ts ? ts.totalLogs24h - ts.totalLogsPrev24h : 0;
    stats.push({
      label: 'Total Logs (24h)',
      value: ts ? ts.totalLogs24h.toLocaleString() : data.dbStats.reduce((s, d) => s + d.totalRows, 0).toLocaleString(),
      trend: logsDiff > 0 ? 'up' : logsDiff < 0 ? 'down' : 'stable',
      trendValue: logsDiff !== 0 ? (logsDiff > 0 ? '+' : '') + logsDiff.toLocaleString() : 'stable',
      trendGood: null, // Neutral - more logs is neither good nor bad
    });

    // Error Rate
    if (ts) {
      const errRate24 = ts.totalLogs24h > 0 ? (ts.errors24h / ts.totalLogs24h) * 100 : 0;
      const errRatePrev = ts.totalLogsPrev24h > 0 ? (ts.errorsPrev24h / ts.totalLogsPrev24h) * 100 : 0;
      const errDiff = errRate24 - errRatePrev;
      stats.push({
        label: 'Error Rate',
        value: errRate24.toFixed(1) + '%',
        trend: errDiff > 0.1 ? 'up' : errDiff < -0.1 ? 'down' : 'stable',
        trendValue: Math.abs(errDiff) > 0.1 ? (errDiff > 0 ? '+' : '') + errDiff.toFixed(1) + '%' : 'stable',
        trendGood: errDiff < -0.1 ? true : errDiff > 0.1 ? false : null,
      });
    } else {
      stats.push({
        label: 'Error Rate',
        value: '--',
        trend: 'stable',
        trendValue: 'no data',
        trendGood: null,
      });
    }

    // Average Context Usage
    if (ts && ts.avgContext24h > 0) {
      const ctxDiff = ts.avgContext24h - ts.avgContextPrev24h;
      const ctxK = Math.round(ts.avgContext24h / 1000);
      stats.push({
        label: 'Avg Context',
        value: ctxK > 0 ? ctxK + 'K tokens' : Math.round(ts.avgContext24h) + ' tokens',
        trend: ctxDiff > 100 ? 'up' : ctxDiff < -100 ? 'down' : 'stable',
        trendValue: Math.abs(ctxDiff) > 100 ? (ctxDiff > 0 ? '+' : '') + Math.round(ctxDiff / 1000) + 'K' : 'stable',
        trendGood: null,
      });
    } else {
      stats.push({
        label: 'Avg Context',
        value: '--',
        trend: 'stable',
        trendValue: 'no data',
        trendGood: null,
      });
    }

    // Tool Calls
    const totalToolCalls = ts ? (ts.totalToolCalls || 0) : 0;
    stats.push({
      label: 'Tool Calls',
      value: totalToolCalls.toLocaleString(),
      trend: totalToolCalls > 0 ? 'up' : 'stable',
      trendValue: totalToolCalls > 0 ? 'active' : 'no data',
      trendGood: totalToolCalls > 0 ? true : null,
    });

    // Use requestAnimationFrame to ensure the container is in DOM
    requestAnimationFrame(() => {
      Charts.drawStatsCards('chart-stats-cards', stats);
    });
  },

  /**
   * Create a section wrapper element with header and body.
   * @param {string} title
   * @param {string} id
   * @returns {{ section: HTMLElement, body: HTMLElement }}
   */
  _createSection(title, id) {
    const section = document.createElement('div');
    section.className = 'dash-section';
    section.id = `dash-${id}`;
    const header = document.createElement('div');
    header.className = 'dash-section-header';
    header.textContent = title;
    section.appendChild(header);
    const body = document.createElement('div');
    body.className = 'dash-section-body';
    section.appendChild(body);
    return { section, body };
  },

  /**
   * Get a status icon character for task status.
   * @param {string} status
   * @returns {string}
   */
  _getStatusIcon(status) {
    switch (status) {
      case 'completed': return '[OK]';
      case 'in_progress': return '[..]';
      case 'failed': return '[!!]';
      case 'pending': return '[--]';
      default: return '[??]';
    }
  },

  /**
   * Render a simple CSS-based sparkline chart for context token usage.
   * @param {Array} usageData - Array of { step_number, context_tokens, max_context_tokens }
   * @returns {HTMLElement}
   */
  _renderContextChart(usageData) {
    const chartContainer = document.createElement('div');
    chartContainer.className = 'dash-chart-container';

    if (usageData.length === 0) {
      chartContainer.innerHTML = '<div class="dash-empty">No context data</div>';
      return chartContainer;
    }

    // Build a simple bar chart using CSS
    const maxTokens = Math.max(
      ...usageData.map(d => d.max_context_tokens || 32768),
      ...usageData.map(d => d.context_tokens || 0)
    );
    const peak = Math.max(...usageData.map(d => d.context_tokens || 0));
    const peakPct = maxTokens > 0 ? Math.round((peak / maxTokens) * 100) : 0;
    const maxLimit = usageData[0]?.max_context_tokens || maxTokens;

    const infoEl = document.createElement('div');
    infoEl.className = 'dash-chart-info';
    infoEl.innerHTML = `Peak: <strong>${peak.toLocaleString()}</strong> tokens (${peakPct}% of ${maxLimit.toLocaleString()} limit)`;
    chartContainer.appendChild(infoEl);

    const chartEl = document.createElement('div');
    chartEl.className = 'dash-sparkline';

    // Take last 50 data points for display
    const displayData = usageData.slice(-50);

    for (const dp of displayData) {
      const tokens = dp.context_tokens || 0;
      const pct = maxTokens > 0 ? Math.round((tokens / maxTokens) * 100) : 0;
      const bar = document.createElement('div');
      bar.className = 'dash-spark-bar';
      bar.style.height = `${Math.max(pct, 2)}%`;

      // Color based on usage percentage
      if (pct >= 80) {
        bar.classList.add('spark-danger');
      } else if (pct >= 60) {
        bar.classList.add('spark-warning');
      } else {
        bar.classList.add('spark-ok');
      }

      bar.title = `Step ${dp.step_number || '?'}: ${tokens.toLocaleString()} tokens (${pct}%)`;
      chartEl.appendChild(bar);
    }

    chartContainer.appendChild(chartEl);
    return chartContainer;
  },
};
