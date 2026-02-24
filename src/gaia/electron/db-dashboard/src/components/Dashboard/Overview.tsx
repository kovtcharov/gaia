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
import { HardDrive, Table2, Rows3, Clock, Wrench, Users, Zap, BrainCircuit, BookOpen, AlertTriangle, Trash2, Hammer, MemoryStick } from 'lucide-react';
import type { DashboardData, DbStats, AgentEntry, SkillEntry, MemoryToolEntry, KnowledgeInsightEntry, LearnedToolEntry, WorkingMemoryEntry } from '../../types/database';
import Badge from '../shared/Badge';
import StatsCards from './StatsCards';
import DonutChart from './DonutChart';
import ActivityHeatmap from './ActivityHeatmap';
import RecentActivity from './RecentActivity';
import PlanView from './PlanView';

interface OverviewProps {
  data: DashboardData;
  workspacePath: string;
  onClearAll?: () => Promise<void>;
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

export default function Overview({ data, workspacePath, onClearAll }: OverviewProps) {
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
      {/* Live Plan -- front and center */}
      <PlanView plan={data.activePlan} planHistory={data.planHistory} />

      {/* Stats Cards Row */}
      <StatsCards
        trendStats={data.trendStats}
        totalRows={totalRows}
        totalSize={data.totalSize}
        dbCount={data.dbCount}
        lastActivity={data.lastActivity}
        learnedToolsCount={data.learnedToolsCount}
      />

      {/* Donut + Heatmap Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {data.dbStats.length > 0 && (
          <DonutChart dbStats={data.dbStats} totalSize={data.totalSize} />
        )}
        <ActivityHeatmap data={data.activityHeatmap} minuteData={data.minuteActivity} />
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

      {/* Created Tools (agent-learned) */}
      {(data.learnedTools.length > 0 || data.learnedToolsCount > 0) && (
        <LearnedToolsSection tools={data.learnedTools} total={data.learnedToolsCount} />
      )}

      {/* Working Memory (active_state) */}
      <WorkingMemorySection entries={data.workingMemory || []} />

      {/* Resource Usage (Agents, Skills, Memory, Knowledge) */}
      <ResourceUsage
        agents={data.topAgents}
        skills={data.topSkills}
        memoryTools={data.topMemoryTools}
        knowledge={data.topKnowledge}
      />

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

      {/* Danger Zone */}
      {onClearAll && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="card p-4 border-gh-danger-emphasis/30"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle size={14} className="text-gh-danger-fg" />
              <div>
                <h3 className="text-xs font-semibold text-gh-danger-fg">Danger Zone</h3>
                <p className="text-2xs text-gh-fg-muted mt-0.5">
                  Permanently delete all data from all GAIA databases. The databases themselves are preserved.
                </p>
              </div>
            </div>
            <button
              onClick={async () => {
                const totalRows = data.dbStats.reduce((s, d) => s + d.totalRows, 0);
                if (!confirm(`Delete ALL data from ALL ${data.dbCount} databases (${totalRows.toLocaleString()} total rows)? This cannot be undone.`)) return;
                await onClearAll();
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md border border-gh-danger-emphasis/50 text-gh-danger-fg hover:bg-gh-danger-emphasis/10 transition-colors shrink-0 ml-4"
            >
              <Trash2 size={12} />
              Clear All Databases
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
}

// ============================================================================
// Created Tools Section
// ============================================================================

function LearnedToolsSection({ tools, total }: { tools: LearnedToolEntry[]; total: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.18 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <Hammer size={14} className="text-gh-success-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Created Tools
        </h3>
        <Badge variant="success">{total}</Badge>
        <span className="text-2xs text-gh-fg-subtle ml-auto">Agent-learned · source=learned in tools.db</span>
      </div>
      {tools.length === 0 ? (
        <div className="py-6 text-center text-xs text-gh-fg-subtle">
          No tools created yet. Use <code className="font-mono text-gh-accent-fg">create_tool()</code> to build and persist custom tools.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-1 max-h-64 overflow-y-auto">
          {tools.map((tool, i) => (
            <motion.div
              key={tool.name}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2, delay: i * 0.02 }}
              className="flex items-start justify-between p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-mono text-gh-accent-fg">{tool.name}</span>
                  {tool.category && (
                    <span className="text-2xs px-1 py-0.5 rounded bg-gh-canvas-subtle border border-gh-border-muted text-gh-fg-subtle">
                      {tool.category}
                    </span>
                  )}
                </div>
                {tool.description && (
                  <div className="text-2xs text-gh-fg-muted truncate mt-0.5">
                    {truncate(tool.description, 65)}
                  </div>
                )}
              </div>
              <div className="flex flex-col items-end ml-2 shrink-0">
                <span className="text-2xs font-mono text-gh-fg-muted">{tool.use_count} uses</span>
                {tool.last_used && (
                  <span className="text-2xs text-gh-fg-subtle">{formatRelative(tool.last_used)}</span>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Working Memory Section
// ============================================================================

function WorkingMemorySection({ entries }: { entries: WorkingMemoryEntry[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.19 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <MemoryStick size={14} className="text-gh-accent-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Working Memory
        </h3>
        <Badge variant="info">{entries.length}</Badge>
        <span className="text-2xs text-gh-fg-subtle ml-auto">active_state · injected into every LLM prompt</span>
      </div>
      {entries.length === 0 ? (
        <div className="py-6 text-center text-xs text-gh-fg-subtle">
          No working memory entries. Use <code className="font-mono text-gh-accent-fg">remember()</code> to store facts.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-1 max-h-64 overflow-y-auto">
          {entries.map((entry) => (
            <motion.div
              key={entry.key}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-start gap-2 p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
            >
              <span className="text-xs font-mono text-gh-accent-fg shrink-0 mt-0.5">{entry.key}</span>
              <span className="text-2xs text-gh-fg-subtle shrink-0 mt-0.5">→</span>
              <div className="flex-1 min-w-0">
                <span className="text-xs text-gh-fg-default block truncate">{entry.value}</span>
                {entry.tags && entry.tags !== 'null' && (
                  <span className="text-2xs text-gh-fg-subtle">{entry.tags}</span>
                )}
              </div>
              <span className="text-2xs text-gh-fg-subtle shrink-0 ml-1">
                {entry.stored_at ? entry.stored_at.slice(11, 16) : ''}
              </span>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Resource Usage Section
// ============================================================================

function truncate(text: string, max: number): string {
  if (!text) return '';
  if (text.length <= max) return text;
  return text.slice(0, max) + '...';
}

interface ResourceUsageProps {
  agents: AgentEntry[];
  skills: SkillEntry[];
  memoryTools: MemoryToolEntry[];
  knowledge: KnowledgeInsightEntry[];
}

function ResourceUsage({ agents, skills, memoryTools, knowledge }: ResourceUsageProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      {/* Top Agents */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.25 }}
        className="card card-hover p-4"
      >
        <div className="flex items-center gap-2 mb-3">
          <Users size={14} className="text-gh-accent-fg" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Top Agents
          </h3>
          <Badge variant="info">{agents.length}</Badge>
        </div>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {agents.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No data</div>
          ) : (
            agents.map((agent, i) => (
              <motion.div
                key={agent.name}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-center justify-between p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
              >
                <span className="text-xs text-gh-fg-default truncate">
                  {truncate(agent.name, 25)}
                </span>
                <span className="text-xs font-mono text-gh-fg-muted ml-2 shrink-0">
                  {agent.usage_count}
                </span>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>

      {/* Top Skills */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.3 }}
        className="card card-hover p-4"
      >
        <div className="flex items-center gap-2 mb-3">
          <Zap size={14} className="text-gh-attention-fg" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Top Skills
          </h3>
          <Badge variant="warning">{skills.length}</Badge>
        </div>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {skills.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No data</div>
          ) : (
            skills.map((skill, i) => (
              <motion.div
                key={skill.name}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-center justify-between p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-xs text-gh-fg-default truncate block">
                    {truncate(skill.name, 25)}
                  </span>
                  {skill.category && (
                    <span className="text-2xs text-gh-fg-subtle">{skill.category}</span>
                  )}
                </div>
                <div className="flex items-center gap-2 ml-2 shrink-0">
                  <span className="text-xs font-mono text-gh-success-fg">{skill.success_count}</span>
                  <span className="text-2xs text-gh-fg-subtle">/</span>
                  <span className="text-xs font-mono text-gh-danger-fg">{skill.failure_count}</span>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>

      {/* Memory Accesses */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.35 }}
        className="card card-hover p-4"
      >
        <div className="flex items-center gap-2 mb-3">
          <BrainCircuit size={14} className="text-gh-fg-muted" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Memory Accesses
          </h3>
          <Badge variant="neutral">{memoryTools.length}</Badge>
        </div>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {memoryTools.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No data</div>
          ) : (
            memoryTools.map((mt, i) => (
              <motion.div
                key={mt.tool_name}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-center justify-between p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
              >
                <span className="text-xs text-gh-fg-default truncate">
                  {truncate(mt.tool_name, 25)}
                </span>
                <span className="text-xs font-mono text-gh-fg-muted ml-2 shrink-0">
                  {mt.call_count}
                </span>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>

      {/* Knowledge Used */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.4 }}
        className="card card-hover p-4"
      >
        <div className="flex items-center gap-2 mb-3">
          <BookOpen size={14} className="text-gh-success-fg" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Knowledge Used
          </h3>
          <Badge variant="success">{knowledge.length}</Badge>
        </div>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {knowledge.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No data</div>
          ) : (
            knowledge.map((k, i) => (
              <motion.div
                key={k.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-center justify-between p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-xs text-gh-fg-default truncate block">
                    {truncate(k.content, 25)}
                  </span>
                  {k.category && (
                    <span className="text-2xs text-gh-fg-subtle">{k.category}</span>
                  )}
                </div>
                <span className="text-xs font-mono text-gh-fg-muted ml-2 shrink-0">
                  {k.use_count}
                </span>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>
    </div>
  );
}
