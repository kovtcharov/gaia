// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
} from 'recharts';
import { HardDrive, Table2, Rows3, Clock, Wrench } from 'lucide-react';
import type { DashboardData, DbStats } from '../../types/database';
import StatsCards from './StatsCards';
import DonutChart from './DonutChart';
import ActivityHeatmap from './ActivityHeatmap';
import RecentActivity from './RecentActivity';

interface OverviewProps {
  data: DashboardData;
  workspacePath: string;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatRelative(date: Date | string | null): string {
  if (!date) return 'N/A';
  const d = typeof date === 'string' ? new Date(date) : date;
  const diff = Date.now() - d.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  return Math.floor(diff / 86400000) + 'd ago';
}

function ToolsBarTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { name: string; value: number } }> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg text-xs">
      <div className="text-gh-fg-default font-medium">{payload[0].payload.name}</div>
      <div className="text-gh-fg-muted">{payload[0].payload.value} calls</div>
    </div>
  );
}

function ContextTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { step_number: number; context_tokens: number } }> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg text-xs">
      <div className="text-gh-fg-default font-medium">Step {payload[0].payload.step_number}</div>
      <div className="text-gh-fg-muted">{payload[0].payload.context_tokens.toLocaleString()} tokens</div>
    </div>
  );
}

export default function Overview({ data, workspacePath }: OverviewProps) {
  const totalRows = data.dbStats.reduce((s, d) => s + d.totalRows, 0);

  const toolBarData = data.topTools.map((t) => ({
    name: t.name.length > 15 ? t.name.slice(0, 15) + '...' : t.name,
    fullName: t.name,
    value: t.usage_count || 0,
    successRate:
      t.usage_count > 0 && t.success_count != null
        ? Math.round((t.success_count / t.usage_count) * 100)
        : null,
  }));

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* Stats Cards Row */}
      <StatsCards
        trendStats={data.trendStats}
        totalRows={totalRows}
        totalSize={data.totalSize}
        dbCount={data.dbCount}
        lastActivity={data.lastActivity}
      />

      {/* Donut + Heatmap Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {data.dbStats.length > 0 && (
          <DonutChart dbStats={data.dbStats} totalSize={data.totalSize} />
        )}
        <ActivityHeatmap data={data.activityHeatmap} />
      </div>

      {/* Top Tools Bar Chart + Context Usage Line */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Tool Usage */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.15 }}
          className="card card-hover p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <Wrench size={14} className="text-gh-fg-muted" />
            <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
              Top Tools
            </h3>
          </div>
          {toolBarData.length === 0 ? (
            <div className="py-8 text-center text-xs text-gh-fg-subtle">No tool usage data</div>
          ) : (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={toolBarData} layout="vertical" margin={{ left: 0, right: 16 }}>
                  <XAxis type="number" tick={{ fontSize: 10, fill: '#8b949e' }} axisLine={false} tickLine={false} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fontSize: 10, fill: '#8b949e' }}
                    width={110}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<ToolsBarTooltip />} cursor={{ fill: 'rgba(88, 166, 255, 0.08)' }} />
                  <Bar
                    dataKey="value"
                    fill="#58a6ff"
                    radius={[0, 4, 4, 0]}
                    animationDuration={600}
                    animationEasing="ease-out"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </motion.div>

        {/* Context Token Usage Line */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="card card-hover p-4"
        >
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider mb-3">
            Context Token Usage
          </h3>
          {data.contextUsage.length === 0 ? (
            <div className="py-8 text-center text-xs text-gh-fg-subtle">No context data</div>
          ) : (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.contextUsage} margin={{ top: 5, right: 16, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                  <XAxis
                    dataKey="step_number"
                    tick={{ fontSize: 10, fill: '#8b949e' }}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'Step', fontSize: 10, fill: '#6e7681', position: 'insideBottomRight', offset: -5 }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: '#8b949e' }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(v: number) => (v >= 1000 ? (v / 1000).toFixed(0) + 'K' : String(v))}
                  />
                  <Tooltip content={<ContextTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="context_tokens"
                    stroke="#a371f7"
                    strokeWidth={2}
                    dot={false}
                    animationDuration={600}
                    animationEasing="ease-out"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </motion.div>
      </div>

      {/* Recent Activity (Errors, Tasks, Insights) */}
      <RecentActivity
        errors={data.recentErrors}
        tasks={data.activeTasks}
        insights={data.recentInsights}
      />

      {/* Database Details Table */}
      {data.dbStats.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.25 }}
          className="card card-hover p-4"
        >
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider mb-3">
            Database Details
          </h3>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Database</th>
                  <th>Size</th>
                  <th>Tables</th>
                  <th>Rows</th>
                  <th>Last Modified</th>
                </tr>
              </thead>
              <tbody>
                {data.dbStats.map((db) => (
                  <tr key={db.name}>
                    <td className="font-medium text-gh-accent-fg">{db.label}</td>
                    <td className="font-mono text-gh-fg-muted">{formatBytes(db.sizeBytes)}</td>
                    <td>{db.tableCount}</td>
                    <td className="font-mono">{db.totalRows.toLocaleString()}</td>
                    <td className="text-gh-fg-muted">
                      {db.lastModified ? formatRelative(db.lastModified) : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Workspace path footer */}
          <div className="mt-3 pt-3 border-t border-gh-border-muted text-2xs text-gh-fg-subtle font-mono">
            Workspace: {workspacePath}
          </div>
        </motion.div>
      )}
    </div>
  );
}
