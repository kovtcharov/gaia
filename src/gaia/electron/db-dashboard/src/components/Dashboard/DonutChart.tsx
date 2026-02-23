// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import type { DbStats } from '../../types/database';

interface DonutChartProps {
  dbStats: DbStats[];
  totalSize: number;
}

const COLORS = [
  '#58a6ff', // blue
  '#3fb950', // green
  '#d29922', // yellow
  '#f85149', // red
  '#a371f7', // purple
  '#79c0ff', // light blue
  '#56d364', // light green
  '#e3b341', // light yellow
];

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: { name: string; value: number; fill: string } }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const { name, value, fill } = payload[0].payload;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg">
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: fill }} />
        <span className="text-xs font-medium text-gh-fg-default">{name}</span>
      </div>
      <span className="text-xs text-gh-fg-muted">{formatBytes(value)}</span>
    </div>
  );
}

export default function DonutChart({ dbStats, totalSize }: DonutChartProps) {
  const data = dbStats.map((db) => ({
    name: db.label,
    value: db.sizeBytes,
  }));

  return (
    <div className="card card-hover p-4">
      <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider mb-3">
        Database Size Breakdown
      </h3>
      <div className="flex items-center gap-4">
        <div className="w-40 h-40 relative">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={45}
                outerRadius={65}
                paddingAngle={2}
                dataKey="value"
                animationBegin={0}
                animationDuration={600}
                animationEasing="ease-out"
              >
                {data.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
          {/* Center label */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-sm font-bold text-gh-fg-default">{formatBytes(totalSize)}</span>
            <span className="text-2xs text-gh-fg-subtle">Total</span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex-1 space-y-1.5">
          {data.map((entry, i) => (
            <div key={entry.name} className="flex items-center gap-2 text-xs">
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: COLORS[i % COLORS.length] }}
              />
              <span className="text-gh-fg-muted truncate flex-1">{entry.name}</span>
              <span className="text-gh-fg-subtle font-mono">{formatBytes(entry.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
