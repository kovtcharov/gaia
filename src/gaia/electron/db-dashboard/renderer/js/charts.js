// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Charts Module
 *
 * GitHub-style data visualization using vanilla Canvas API and inline SVG.
 * No external charting dependencies.
 *
 * Provides:
 *   - drawDonutChart(canvasId, data, options) -- Canvas donut with legend
 *   - drawHeatmap(containerId, data)          -- SVG activity heatmap
 *   - drawBarChart(containerId, data)         -- Horizontal bar chart
 *   - drawStatsCards(containerId, stats)      -- Trend stat cards
 */

const Charts = {

  // ========================================================================
  // Color palettes
  // ========================================================================

  /** Database donut colors - GitHub-inspired palette. */
  DONUT_COLORS: [
    '#5b7cf7', // blue
    '#4caf6a', // green
    '#e6a23c', // amber
    '#e74c5e', // red
    '#58b8d8', // cyan
    '#a78bfa', // violet
    '#f472b6', // pink
    '#34d399', // emerald
    '#fb923c', // orange
    '#94a3b8', // slate
  ],

  /** Heatmap green gradient - 5 levels (GitHub contribution style). */
  HEATMAP_LEVELS: [
    'var(--bg-tertiary)',  // Level 0: no activity
    '#0e4429',            // Level 1: low
    '#006d32',            // Level 2: medium-low
    '#26a641',            // Level 3: medium-high
    '#39d353',            // Level 4: high
  ],

  /** Bar chart colors. */
  BAR_COLORS: [
    '#5b7cf7', '#4caf6a', '#e6a23c', '#e74c5e', '#58b8d8',
    '#a78bfa', '#f472b6', '#34d399', '#fb923c', '#94a3b8',
  ],

  // ========================================================================
  // Donut Chart (Canvas)
  // ========================================================================

  /**
   * Draw a donut chart on a canvas element.
   *
   * @param {string} canvasId - The canvas element ID
   * @param {Array<{label: string, value: number, color?: string}>} data
   * @param {Object} [options]
   * @param {number} [options.size=180]      - Canvas logical size (square)
   * @param {number} [options.lineWidth=28]  - Donut ring thickness
   * @param {string} [options.centerLabel]   - Text in the center
   * @param {string} [options.centerSub]     - Sub-text below center label
   */
  drawDonutChart(canvasId, data, options) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !canvas.getContext) return;

    const opts = Object.assign({
      size: 180,
      lineWidth: 28,
      centerLabel: '',
      centerSub: '',
    }, options || {});

    // High-DPI support
    const dpr = window.devicePixelRatio || 1;
    canvas.width = opts.size * dpr;
    canvas.height = opts.size * dpr;
    canvas.style.width = opts.size + 'px';
    canvas.style.height = opts.size + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const cx = opts.size / 2;
    const cy = opts.size / 2;
    const radius = (opts.size - opts.lineWidth) / 2 - 4;

    // Clear
    ctx.clearRect(0, 0, opts.size, opts.size);

    const total = data.reduce((sum, d) => sum + d.value, 0);
    if (total === 0) {
      // Draw empty ring
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(42, 46, 63, 0.6)';
      ctx.lineWidth = opts.lineWidth;
      ctx.stroke();
      this._drawDonutCenter(ctx, cx, cy, 'No data', '', opts);
      return;
    }

    // Assign colors
    const coloredData = data.map((d, i) => ({
      ...d,
      color: d.color || this.DONUT_COLORS[i % this.DONUT_COLORS.length],
    }));

    // Store segment positions for hover detection
    const segments = [];
    let startAngle = -Math.PI / 2; // Start at 12 o'clock

    for (const item of coloredData) {
      const sliceAngle = (item.value / total) * Math.PI * 2;
      const endAngle = startAngle + sliceAngle;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, endAngle);
      ctx.strokeStyle = item.color;
      ctx.lineWidth = opts.lineWidth;
      ctx.lineCap = 'butt';
      ctx.stroke();

      segments.push({
        startAngle,
        endAngle,
        label: item.label,
        value: item.value,
        color: item.color,
        pct: ((item.value / total) * 100).toFixed(1),
      });

      startAngle = endAngle;
    }

    // Center text
    this._drawDonutCenter(ctx, cx, cy, opts.centerLabel, opts.centerSub, opts);

    // Store segment data for tooltip handling
    canvas._chartSegments = segments;
    canvas._chartOpts = { cx, cy, radius, lineWidth: opts.lineWidth, size: opts.size };

    // Attach hover handler (once)
    if (!canvas._chartHoverBound) {
      canvas._chartHoverBound = true;
      canvas.addEventListener('mousemove', (e) => this._handleDonutHover(e, canvas));
      canvas.addEventListener('mouseleave', () => this._hideDonutTooltip(canvas));
    }
  },

  /**
   * Draw center text in donut.
   * @private
   */
  _drawDonutCenter(ctx, cx, cy, label, sub, opts) {
    if (label) {
      ctx.fillStyle = '#e1e4f0';
      ctx.font = 'bold 20px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cx, sub ? cy - 8 : cy);
    }
    if (sub) {
      ctx.fillStyle = '#8b90a8';
      ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(sub, cx, cy + 12);
    }
  },

  /**
   * Handle mouse hover on donut chart for tooltip display.
   * @private
   */
  _handleDonutHover(event, canvas) {
    if (!canvas._chartSegments) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const { cx, cy, radius, lineWidth } = canvas._chartOpts;

    // Calculate angle and distance from center
    const dx = x - cx;
    const dy = y - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const innerR = radius - lineWidth / 2;
    const outerR = radius + lineWidth / 2;

    if (dist < innerR || dist > outerR) {
      this._hideDonutTooltip(canvas);
      return;
    }

    let angle = Math.atan2(dy, dx);
    // Normalize to match our start at -PI/2
    if (angle < -Math.PI / 2) angle += Math.PI * 2;

    // Find which segment
    for (const seg of canvas._chartSegments) {
      let sStart = seg.startAngle;
      let sEnd = seg.endAngle;
      // Normalize
      if (sStart < -Math.PI / 2) sStart += Math.PI * 2;
      if (sEnd < -Math.PI / 2) sEnd += Math.PI * 2;

      if (angle >= sStart && angle <= sEnd) {
        this._showDonutTooltip(canvas, event, seg);
        return;
      }
    }

    this._hideDonutTooltip(canvas);
  },

  /**
   * Show tooltip near the donut chart.
   * @private
   */
  _showDonutTooltip(canvas, event, segment) {
    let tooltip = canvas.parentElement.querySelector('.chart-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.className = 'chart-tooltip';
      canvas.parentElement.appendChild(tooltip);
    }

    tooltip.innerHTML = `
      <span class="chart-tooltip-dot" style="background:${segment.color}"></span>
      <strong>${escapeHtml(segment.label)}</strong>
      <span class="chart-tooltip-value">${this._formatBytesCompact(segment.value)}</span>
      <span class="chart-tooltip-pct">${segment.pct}%</span>
    `;
    tooltip.style.display = 'flex';

    // Position relative to canvas parent
    const parentRect = canvas.parentElement.getBoundingClientRect();
    const x = event.clientX - parentRect.left + 12;
    const y = event.clientY - parentRect.top - 10;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  },

  /**
   * Hide the donut tooltip.
   * @private
   */
  _hideDonutTooltip(canvas) {
    const tooltip = canvas.parentElement.querySelector('.chart-tooltip');
    if (tooltip) {
      tooltip.style.display = 'none';
    }
  },

  // ========================================================================
  // Activity Heatmap (SVG)
  // ========================================================================

  /**
   * Draw a GitHub-style activity heatmap for the last 7 days.
   *
   * @param {string} containerId - Container element ID
   * @param {Array<{date: string, count: number}>} data
   *    - date is ISO date string (YYYY-MM-DD)
   *    - count is the activity count for that day
   */
  drawHeatmap(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    // Build a map of date -> count
    const countMap = {};
    let maxCount = 0;
    for (const d of data) {
      countMap[d.date] = d.count;
      if (d.count > maxCount) maxCount = d.count;
    }

    // Generate last 7 days
    const days = [];
    const today = new Date();
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      const dateStr = d.toISOString().split('T')[0];
      const dayLabel = d.toLocaleDateString('en-US', { weekday: 'short' });
      const fullLabel = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      days.push({
        date: dateStr,
        dayLabel,
        fullLabel,
        count: countMap[dateStr] || 0,
      });
    }

    // Create the SVG
    const cellSize = 36;
    const cellGap = 4;
    const labelHeight = 20;
    const svgWidth = days.length * (cellSize + cellGap) - cellGap;
    const svgHeight = cellSize + labelHeight + 8;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
    svg.setAttribute('class', 'heatmap-svg');
    svg.style.maxWidth = svgWidth + 'px';

    for (let i = 0; i < days.length; i++) {
      const day = days[i];
      const x = i * (cellSize + cellGap);
      const level = this._getHeatmapLevel(day.count, maxCount);

      // Cell rect
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x);
      rect.setAttribute('y', 0);
      rect.setAttribute('width', cellSize);
      rect.setAttribute('height', cellSize);
      rect.setAttribute('rx', '4');
      rect.setAttribute('ry', '4');
      rect.setAttribute('class', `heatmap-cell heatmap-level-${level}`);
      rect.setAttribute('data-date', day.date);
      rect.setAttribute('data-count', day.count);

      // Tooltip via title element
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${day.fullLabel}: ${day.count} ${day.count === 1 ? 'action' : 'actions'}`;
      rect.appendChild(title);

      svg.appendChild(rect);

      // Count text inside cell (if non-zero)
      if (day.count > 0) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x + cellSize / 2);
        text.setAttribute('y', cellSize / 2 + 1);
        text.setAttribute('class', 'heatmap-count');
        text.textContent = day.count > 999 ? Math.round(day.count / 1000) + 'k' : day.count;
        svg.appendChild(text);
      }

      // Day label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', x + cellSize / 2);
      label.setAttribute('y', cellSize + labelHeight);
      label.setAttribute('class', 'heatmap-label');
      label.textContent = day.dayLabel;
      svg.appendChild(label);
    }

    container.appendChild(svg);

    // Legend
    const legend = document.createElement('div');
    legend.className = 'heatmap-legend';
    legend.innerHTML = '<span class="heatmap-legend-label">Less</span>';
    for (let i = 0; i <= 4; i++) {
      legend.innerHTML += `<span class="heatmap-legend-cell heatmap-level-${i}"></span>`;
    }
    legend.innerHTML += '<span class="heatmap-legend-label">More</span>';
    container.appendChild(legend);
  },

  /**
   * Determine heatmap intensity level (0-4).
   * @private
   */
  _getHeatmapLevel(count, maxCount) {
    if (count === 0) return 0;
    if (maxCount === 0) return 0;
    const ratio = count / maxCount;
    if (ratio <= 0.25) return 1;
    if (ratio <= 0.50) return 2;
    if (ratio <= 0.75) return 3;
    return 4;
  },

  // ========================================================================
  // Horizontal Bar Chart
  // ========================================================================

  /**
   * Draw a horizontal bar chart (like GitHub's language breakdown).
   *
   * @param {string} containerId - Container element ID
   * @param {Array<{label: string, value: number, color?: string}>} data
   *    - Sorted descending by value. Top 10 shown.
   */
  drawBarChart(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    if (!data || data.length === 0) {
      container.innerHTML = '<div class="dash-empty">No tool usage data</div>';
      return;
    }

    const top10 = data.slice(0, 10);
    const total = top10.reduce((sum, d) => sum + d.value, 0);
    if (total === 0) {
      container.innerHTML = '<div class="dash-empty">No tool usage data</div>';
      return;
    }

    // Stacked bar at top (like GitHub's language bar)
    const stackedBar = document.createElement('div');
    stackedBar.className = 'bar-chart-stacked';

    for (let i = 0; i < top10.length; i++) {
      const item = top10[i];
      const pct = (item.value / total) * 100;
      const color = item.color || this.BAR_COLORS[i % this.BAR_COLORS.length];

      const segment = document.createElement('div');
      segment.className = 'bar-chart-segment';
      segment.style.width = Math.max(pct, 0.5) + '%';
      segment.style.backgroundColor = color;
      segment.title = `${item.label}: ${item.value.toLocaleString()} calls (${pct.toFixed(1)}%)`;

      // Rounded corners on first and last
      if (i === 0) segment.style.borderRadius = '4px 0 0 4px';
      if (i === top10.length - 1) segment.style.borderRadius = '0 4px 4px 0';
      if (top10.length === 1) segment.style.borderRadius = '4px';

      stackedBar.appendChild(segment);
    }

    container.appendChild(stackedBar);

    // Legend list below
    const legendList = document.createElement('div');
    legendList.className = 'bar-chart-legend';

    for (let i = 0; i < top10.length; i++) {
      const item = top10[i];
      const pct = ((item.value / total) * 100).toFixed(1);
      const color = item.color || this.BAR_COLORS[i % this.BAR_COLORS.length];

      const entry = document.createElement('div');
      entry.className = 'bar-chart-legend-item';
      entry.innerHTML = `
        <span class="bar-chart-legend-dot" style="background:${color}"></span>
        <span class="bar-chart-legend-name">${escapeHtml(item.label)}</span>
        <span class="bar-chart-legend-pct">${pct}%</span>
        <span class="bar-chart-legend-count">${item.value.toLocaleString()}</span>
      `;
      legendList.appendChild(entry);
    }

    container.appendChild(legendList);
  },

  // ========================================================================
  // Stats Cards with Trends
  // ========================================================================

  /**
   * Draw stats cards with trend indicators.
   *
   * @param {string} containerId - Container element ID
   * @param {Array<{label: string, value: string|number, trend: 'up'|'down'|'stable', trendValue: string, trendGood?: boolean}>} stats
   *    - trend: direction of change
   *    - trendValue: display string (e.g. "+156", "-0.5%")
   *    - trendGood: true if the trend is positive/good (green), false = red, null = neutral
   */
  drawStatsCards(containerId, stats) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    for (const stat of stats) {
      const card = document.createElement('div');
      card.className = 'stats-card';

      const trendIcon = stat.trend === 'up' ? '&#x2191;' : stat.trend === 'down' ? '&#x2193;' : '&#x2192;';
      let trendClass = 'stats-trend-neutral';
      if (stat.trendGood === true) trendClass = 'stats-trend-good';
      else if (stat.trendGood === false) trendClass = 'stats-trend-bad';

      card.innerHTML = `
        <div class="stats-card-label">${escapeHtml(stat.label)}</div>
        <div class="stats-card-value">${escapeHtml(String(stat.value))}</div>
        <div class="stats-card-trend ${trendClass}">
          <span class="stats-trend-icon">${trendIcon}</span>
          <span class="stats-trend-text">${escapeHtml(stat.trendValue)}</span>
        </div>
      `;

      container.appendChild(card);
    }
  },

  // ========================================================================
  // Donut Chart Legend (standalone)
  // ========================================================================

  /**
   * Render a legend for the donut chart.
   *
   * @param {string} containerId - Container element ID
   * @param {Array<{label: string, value: number, color?: string}>} data
   */
  drawDonutLegend(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const total = data.reduce((sum, d) => sum + d.value, 0);

    for (let i = 0; i < data.length; i++) {
      const item = data[i];
      const color = item.color || this.DONUT_COLORS[i % this.DONUT_COLORS.length];
      const pct = total > 0 ? ((item.value / total) * 100).toFixed(1) : '0.0';

      const entry = document.createElement('div');
      entry.className = 'donut-legend-item';
      entry.innerHTML = `
        <span class="donut-legend-dot" style="background:${color}"></span>
        <span class="donut-legend-name">${escapeHtml(item.label)}</span>
        <span class="donut-legend-size">${this._formatBytesCompact(item.value)}</span>
        <span class="donut-legend-pct">${pct}%</span>
      `;
      container.appendChild(entry);
    }
  },

  // ========================================================================
  // Helpers
  // ========================================================================

  /**
   * Compact byte formatting for chart labels.
   * @private
   */
  _formatBytesCompact(bytes) {
    if (bytes === 0) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
  },
};
