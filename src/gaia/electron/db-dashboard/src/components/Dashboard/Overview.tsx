// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
import {
  Wrench, Zap, BrainCircuit,
  BookOpen, AlertTriangle, Trash2, Hammer, MemoryStick, Bot, CheckCircle,
  XCircle, ScrollText, TreePine, Clipboard, Check, ChevronDown, ChevronRight,
  Filter, ListOrdered, MessageSquare, Cog, ArrowRight, MessagesSquare, User, Cpu,
  Clock, Bug, Microscope, TestTube, BarChart2,
} from 'lucide-react';
import type {
  DashboardData, SkillEntry, MemoryToolEntry,
  KnowledgeInsightEntry, LearnedToolEntry, WorkingMemoryEntry,
  AgentSpecialistEntry, AgentCallEntry, RuntimeLogEntry, ReasoningStepEntry,
  ConversationTurnEntry,
  PlanTreeTask, DispatchActivityEntry, AgentDetailStats,
} from '../../types/database';
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

function truncate(text: string, max: number): string {
  if (!text) return '';
  if (text.length <= max) return text;
  return text.slice(0, max) + '...';
}

/** Format a duration in milliseconds as a compact human string */
function formatDuration(ms: number): string {
  if (ms < 0) return '';
  const totalSec = Math.floor(ms / 1000);
  if (totalSec < 60) return `${totalSec}s`;
  const mins = Math.floor(totalSec / 60);
  const secs = totalSec % 60;
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  const hours = Math.floor(mins / 60);
  const remainMins = mins % 60;
  return remainMins > 0 ? `${hours}h ${remainMins}m` : `${hours}h`;
}

// ============================================================================
// Copy-to-clipboard helper
// ============================================================================

function CopyButton({ text, className = '' }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Fallback for environments without clipboard API
    }
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className={`inline-flex items-center justify-center w-5 h-5 rounded text-gh-fg-subtle hover:text-gh-fg-muted hover:bg-gh-canvas-subtle/80 transition-colors shrink-0 ${className}`}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? <Check size={10} className="text-gh-success-fg" /> : <Clipboard size={10} />}
    </button>
  );
}

// ============================================================================
// Recharts Tooltips
// ============================================================================

function ToolsBarTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { name: string; value: number } }> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg text-xs">
      <div className="text-gh-fg-default font-medium">{payload[0].payload.name}</div>
      <div className="text-gh-fg-muted">{payload[0].payload.value} calls</div>
    </div>
  );
}

function ContextTooltip({ active, payload, isStepFallback }: { active?: boolean; payload?: Array<{ payload: { step_number: number; context_tokens: number } }>; isStepFallback?: boolean }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg text-xs">
      <div className="text-gh-fg-default font-medium">Step {payload[0].payload.step_number}</div>
      <div className="text-gh-fg-muted">
        {isStepFallback
          ? `Step ${payload[0].payload.step_number}`
          : `${payload[0].payload.context_tokens.toLocaleString()} tokens`}
      </div>
    </div>
  );
}

// ============================================================================
// Main Overview Component
// ============================================================================

export default function Overview({ data, workspacePath, onClearAll }: OverviewProps) {
  const totalRows = data.dbStats.reduce((s, d) => s + d.totalRows, 0);
  const isStepFallback = (data as unknown as Record<string, unknown>)._contextFallback === 'steps';

  // Filter topTools to only those with use_count > 0 (already filtered in query, but be safe)
  const activeTools = data.topTools.filter((t) => t.usage_count > 0);

  const toolBarData = activeTools.map((t) => ({
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

      {/* Execution Steps -- LLM reasoning, tool calls, results per step */}
      <ExecutionStepsSection entries={data.reasoningSteps || []} />

      {/* Conversation History -- full LLM input/output per step */}
      <ConversationHistorySection turns={data.conversationTurns || []} />

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
        {/* Tool Usage -- only show tools with use_count > 0 */}
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
            <div className="py-8 text-center text-xs text-gh-fg-subtle">No activity yet</div>
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
            {isStepFallback ? 'Steps Over Time' : 'Context Token Usage'}
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
                    tickFormatter={(v: number) => isStepFallback ? String(v) : (v >= 1000 ? (v / 1000).toFixed(0) + 'K' : String(v))}
                  />
                  <Tooltip content={<ContextTooltip isStepFallback={isStepFallback} />} />
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

      {/* Agent Activity -- merged from Agent Dispatch + Top Agents */}
      <AgentActivitySection
        specialists={data.agentSpecialists || []}
        calls={data.recentAgentCalls || []}
        dispatchActivity={data.dispatchActivity || []}
        detailStats={data.agentDetailStats || []}
      />

      {/* Execution Log Panel */}
      <ExecutionLogSection entries={data.executionLog || []} />

      {/* Plan Tree Panel */}
      <PlanTreeSection tasks={data.planTreeTasks || []} />

      {/* Resource Usage (Skills, Memory, Knowledge) -- Top Agents removed, merged into Agent Activity */}
      <ResourceUsage
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
                    <td className="font-medium text-gh-accent-fg" title={db.name}>{db.label}</td>
                    <td className="font-mono text-gh-fg-muted">
                      <span className="flex items-center gap-1">
                        {formatBytes(db.sizeBytes)}
                        <CopyButton text={`${db.label}: ${formatBytes(db.sizeBytes)}, ${db.tableCount} tables, ${db.totalRows} rows`} />
                      </span>
                    </td>
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
                  <div className="text-2xs text-gh-fg-muted truncate mt-0.5" title={tool.description}>
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
              <span className="text-2xs text-gh-fg-subtle shrink-0 mt-0.5">{'->'}</span>
              <div className="flex-1 min-w-0">
                <span className="text-xs text-gh-fg-default block truncate" title={entry.value}>{entry.value}</span>
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
// Agent Activity Section (merged from Agent Dispatch + Top Agents)
// ============================================================================

interface AgentActivityProps {
  specialists: AgentSpecialistEntry[];
  calls: AgentCallEntry[];
  dispatchActivity: DispatchActivityEntry[];
  detailStats: AgentDetailStats[];
}

function AgentActivitySection({ specialists, calls, dispatchActivity, detailStats }: AgentActivityProps) {
  const [showRegistered, setShowRegistered] = useState(false);

  // Separate active (used) vs registered-but-unused agents
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;
  const activeAgents = specialists.filter(
    (a) => (a.use_count || 0) > 0
  );
  // SQLite stores timestamps as 'YYYY-MM-DD HH:MM:SS' (space-separated, no timezone).
  // Replace space with 'T' to ensure ISO 8601 parsing works in all V8 environments.
  const parseAgentDate = (d: string | null | undefined) =>
    d ? new Date(d.replace(' ', 'T')) : null;
  const registeredOnly = specialists.filter(
    (a) => (a.use_count || 0) === 0 && (!a.created_at || ((parseAgentDate(a.created_at)?.getTime() ?? 0) > 0 && (now - (parseAgentDate(a.created_at)?.getTime() ?? 0)) > oneDayMs))
  );
  const recentlyRegistered = specialists.filter(
    (a) => (a.use_count || 0) === 0 && a.created_at && (now - (parseAgentDate(a.created_at)?.getTime() ?? Infinity)) <= oneDayMs
  );

  // Combine active + recently registered for the main view
  const mainAgents = [...activeAgents, ...recentlyRegistered];
  const maxUse = Math.max(...mainAgents.map((a) => a.use_count || 0), 1);

  const hasAnything = mainAgents.length > 0 || calls.length > 0 || dispatchActivity.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.22 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-4">
        <Bot size={14} className="text-gh-accent-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Agent Activity
        </h3>
        {activeAgents.length > 0 && (
          <Badge variant="info">{activeAgents.length} active</Badge>
        )}
        {calls.length > 0 && (
          <span className="text-2xs text-gh-fg-subtle ml-auto">{calls.length} recent invocations</span>
        )}
      </div>

      {!hasAnything ? (
        <div className="py-8 text-center">
          <Bot size={28} className="text-gh-fg-subtle mx-auto mb-2 opacity-40" />
          <div className="text-xs text-gh-fg-subtle">No activity yet</div>
          <div className="text-2xs text-gh-fg-subtle mt-1">
            Agent activity will appear here once agents are invoked.
          </div>
        </div>
      ) : (
        <>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left: Active Agents */}
          <div>
            <div className="text-2xs font-semibold text-gh-fg-subtle uppercase tracking-wider mb-2">
              Active Agents
            </div>
            {mainAgents.length === 0 && dispatchActivity.length === 0 ? (
              <div className="py-4 text-center text-xs text-gh-fg-subtle">
                No agents used yet.
              </div>
            ) : (
              <div className="space-y-1 max-h-72 overflow-y-auto">
                {mainAgents.map((agent, i) => {
                  const confidence = agent.confidence != null ? Math.round(agent.confidence * 100) : 0;
                  const pct = Math.round(((agent.use_count || 0) / maxUse) * 100);
                  const total = (agent.success_count ?? 0) + (agent.failure_count ?? 0);
                  const successRate = total > 0 ? Math.round(((agent.success_count ?? 0) / total) * 100) : null;
                  const isNewlyRegistered = (agent.use_count || 0) === 0;
                  return (
                    <motion.div
                      key={agent.name}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2, delay: i * 0.03 }}
                      className={`p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors ${isNewlyRegistered ? 'opacity-60' : ''}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium text-gh-fg-default truncate" title={agent.description || agent.name}>
                          {agent.name}
                          {isNewlyRegistered && (
                            <span className="text-2xs text-gh-fg-subtle ml-1">(new)</span>
                          )}
                        </span>
                        <div className="flex items-center gap-2 shrink-0 ml-2">
                          <span className="text-2xs text-gh-fg-muted font-mono">{agent.use_count} calls</span>
                          {successRate !== null && (
                            <span className={`text-2xs font-mono ${successRate >= 80 ? 'text-gh-success-fg' : successRate >= 50 ? 'text-gh-attention-fg' : 'text-gh-danger-fg'}`}>
                              {successRate}% ok
                            </span>
                          )}
                          <span className="text-2xs text-gh-fg-subtle">{confidence}% conf</span>
                        </div>
                      </div>
                      <div className="h-1 bg-gh-canvas-subtle rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gh-accent-fg rounded-full transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      {agent.last_used && (
                        <div className="text-2xs text-gh-fg-subtle mt-0.5">
                          last: {formatRelative(agent.last_used)}
                        </div>
                      )}
                    </motion.div>
                  );
                })}

                {/* Dispatch Activity fallback when agent_usage is empty */}
                {calls.length === 0 && dispatchActivity.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gh-border-muted">
                    <div className="text-2xs font-semibold text-gh-fg-subtle uppercase tracking-wider mb-1">
                      Dispatch Activity
                    </div>
                    {dispatchActivity.map((d) => (
                      <div key={d.name} className="flex items-center justify-between p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors">
                        <span className="text-xs font-mono text-gh-accent-fg">{d.name}</span>
                        <div className="flex items-center gap-2 shrink-0 ml-2">
                          <span className="text-2xs text-gh-fg-muted font-mono">{d.use_count} calls</span>
                          {d.avg_duration_ms != null && (
                            <span className="text-2xs font-mono text-gh-fg-subtle">{Math.round(d.avg_duration_ms)}ms avg</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Collapsed "Registered Specialists" section for zero-usage agents */}
            {registeredOnly.length > 0 && (
              <div className="mt-2">
                <button
                  onClick={() => setShowRegistered((v) => !v)}
                  className="flex items-center gap-1 text-2xs text-gh-fg-subtle hover:text-gh-fg-muted transition-colors"
                >
                  {showRegistered ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                  {registeredOnly.length} registered specialists (unused)
                </button>
                <AnimatePresence>
                  {showRegistered && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-0.5 mt-1 max-h-40 overflow-y-auto"
                    >
                      {registeredOnly.map((agent) => (
                        <div
                          key={agent.name}
                          className="flex items-center justify-between p-1.5 rounded-md opacity-50 text-2xs"
                          title={agent.description || agent.name}
                        >
                          <span className="text-gh-fg-subtle truncate">{agent.name}</span>
                          <span className="text-gh-fg-subtle shrink-0 ml-2">
                            {agent.confidence != null ? `${Math.round(agent.confidence * 100)}% conf` : ''}
                          </span>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>

          {/* Right: Recent Invocations */}
          <div>
            <div className="text-2xs font-semibold text-gh-fg-subtle uppercase tracking-wider mb-2">
              Recent Invocations
            </div>
            {calls.length === 0 ? (
              <div className="py-6 text-center text-xs text-gh-fg-subtle">
                No invocations yet. Run the agent on a task to trigger sub-agent dispatch.
              </div>
            ) : (
              <div className="space-y-0.5 max-h-72 overflow-y-auto">
                {calls.map((call, i) => {
                  const ok = call.success === 1 || (call.success as unknown as boolean) === true;
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: 8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2, delay: i * 0.02 }}
                      className="flex items-start gap-2 p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
                    >
                      {ok ? (
                        <CheckCircle size={12} className="text-gh-success-fg shrink-0 mt-0.5" />
                      ) : (
                        <XCircle size={12} className="text-gh-danger-fg shrink-0 mt-0.5" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-2xs font-medium text-gh-accent-fg shrink-0">{call.name}</span>
                          {call.task_type && (
                            <span className="text-2xs text-gh-fg-muted truncate" title={call.task_type}>{truncate(call.task_type, 55)}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          {call.duration_ms != null && (
                            <span className="text-2xs font-mono text-gh-fg-subtle">{formatDuration(call.duration_ms)}</span>
                          )}
                          {call.timestamp && (
                            <span className="text-2xs text-gh-fg-subtle">{formatRelative(call.timestamp)}</span>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Bottom: Agent Performance Stats (full width) */}
        {detailStats.length > 0 && (
          <AgentPerfStats stats={detailStats} />
        )}
        </>
      )}
    </motion.div>
  );
}

// ============================================================================
// Agent Performance Stats (per-agent breakdown)
// ============================================================================

function agentIcon(name: string) {
  const lower = name.toLowerCase();
  if (lower.includes('bug') || lower.includes('bash')) return <Bug size={12} className="text-gh-danger-fg shrink-0" />;
  if (lower.includes('test')) return <TestTube size={12} className="text-gh-success-fg shrink-0" />;
  if (lower.includes('analysis') || lower.includes('analys')) return <Microscope size={12} className="text-gh-accent-fg shrink-0" />;
  return <BarChart2 size={12} className="text-gh-fg-muted shrink-0" />;
}

function agentLabel(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('bug') || lower.includes('bash')) return 'fix cycles';
  if (lower.includes('test')) return 'test runs';
  if (lower.includes('analysis') || lower.includes('analys')) return 'analysis runs';
  return 'invocations';
}

function AgentPerfStats({ stats }: { stats: AgentDetailStats[] }) {
  const maxTotal = Math.max(...stats.map((s) => s.total_duration_ms ?? 0), 1);

  return (
    <div className="mt-4 pt-4 border-t border-gh-border-muted">
      <div className="flex items-center gap-2 mb-3">
        <BarChart2 size={13} className="text-gh-fg-muted" />
        <span className="text-2xs font-semibold text-gh-fg-subtle uppercase tracking-wider">
          Agent Performance
        </span>
        <span className="text-2xs text-gh-fg-subtle ml-auto">
          Total runtime across all invocations
        </span>
      </div>

      <div className="space-y-2">
        {stats.map((s, i) => {
          const successRate = s.invocations > 0
            ? Math.round((s.successes / s.invocations) * 100)
            : 0;
          const totalPct = Math.round(((s.total_duration_ms ?? 0) / maxTotal) * 100);
          const label = agentLabel(s.name);

          return (
            <motion.div
              key={s.name}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, delay: i * 0.04 }}
              className="rounded-md p-2 hover:bg-gh-canvas-subtle/40 transition-colors"
            >
              {/* Row header */}
              <div className="flex items-center gap-2 mb-1.5">
                {agentIcon(s.name)}
                <span className="text-xs font-medium text-gh-fg-default">{s.name}</span>
                <div className="flex items-center gap-3 ml-auto shrink-0">
                  {/* Invocation count + label */}
                  <span className="text-2xs font-mono text-gh-fg-muted">
                    <span className="text-gh-fg-default font-semibold">{s.invocations}</span> {label}
                  </span>
                  {/* Success / fail */}
                  <span className="text-2xs font-mono flex items-center gap-1">
                    <CheckCircle size={10} className="text-gh-success-fg" />
                    <span className="text-gh-success-fg">{s.successes}</span>
                    {s.failures > 0 && (
                      <>
                        <XCircle size={10} className="text-gh-danger-fg ml-0.5" />
                        <span className="text-gh-danger-fg">{s.failures}</span>
                      </>
                    )}
                  </span>
                  {/* Success % */}
                  <span className={`text-2xs font-mono font-semibold ${successRate >= 80 ? 'text-gh-success-fg' : successRate >= 50 ? 'text-gh-attention-fg' : 'text-gh-danger-fg'}`}>
                    {successRate}%
                  </span>
                  {/* Avg duration */}
                  {s.avg_duration_ms != null && (
                    <span className="text-2xs font-mono text-gh-fg-subtle flex items-center gap-1">
                      <Clock size={9} />
                      avg {formatDuration(s.avg_duration_ms)}
                    </span>
                  )}
                  {/* Total */}
                  {s.total_duration_ms != null && (
                    <span className="text-2xs font-mono text-gh-fg-subtle">
                      total {formatDuration(s.total_duration_ms)}
                    </span>
                  )}
                </div>
              </div>

              {/* Total runtime bar */}
              <div className="h-1.5 bg-gh-canvas-subtle rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    agentLabel(s.name) === 'fix cycles'
                      ? 'bg-gh-danger-fg/70'
                      : agentLabel(s.name) === 'test runs'
                        ? 'bg-gh-success-fg/70'
                        : 'bg-gh-accent-fg/70'
                  }`}
                  style={{ width: `${totalPct}%` }}
                />
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================================
// Execution Steps Section (LLM reasoning, tool calls, results per step)
// ============================================================================

interface StepGroup {
  step_number: number;
  reasoning: string | null;
  goal: string | null;
  tool: string | null;
  toolArgs: string | null;
  success: boolean | null;
  result: string | null;
  timestamp: string;
  agent_name: string | null;
  hasError: boolean;
}

function groupReasoningSteps(entries: ReasoningStepEntry[]): StepGroup[] {
  const stepMap = new Map<number, StepGroup>();

  for (const entry of entries) {
    const step = entry.step_number ?? 0;
    if (!stepMap.has(step)) {
      stepMap.set(step, {
        step_number: step,
        reasoning: null,
        goal: null,
        tool: null,
        toolArgs: null,
        success: null,
        result: null,
        timestamp: entry.timestamp,
        agent_name: entry.agent_name,
        hasError: false,
      });
    }
    const group = stepMap.get(step)!;

    const msg = entry.message;
    if (msg.startsWith('[STEP_REASONING] ')) {
      group.reasoning = msg.slice('[STEP_REASONING] '.length);
    } else if (msg.startsWith('[STEP_GOAL] ')) {
      group.goal = msg.slice('[STEP_GOAL] '.length);
    } else if (msg.startsWith('[STEP_TOOL] ')) {
      // Parse: tool=<name> args=<json>
      const toolMatch = msg.match(/\[STEP_TOOL\] tool=(\S+)\s+args=(.*)/);
      if (toolMatch) {
        group.tool = toolMatch[1];
        group.toolArgs = toolMatch[2];
      }
    } else if (msg.startsWith('[STEP_RESULT] ')) {
      // Parse: tool=<name> success=<bool> result=<text>
      const resultMatch = msg.match(/\[STEP_RESULT\] tool=(\S+)\s+success=(True|False)\s+result=(.*)/);
      if (resultMatch) {
        group.success = resultMatch[2] === 'True';
        group.result = resultMatch[3];
        if (!group.success) group.hasError = true;
      }
    }

    // Update timestamp to latest entry in group
    if (entry.timestamp > group.timestamp) {
      group.timestamp = entry.timestamp;
    }
  }

  // Sort by step_number ascending
  return Array.from(stepMap.values()).sort((a, b) => a.step_number - b.step_number);
}

function ExecutionStepsSection({ entries }: { entries: ReasoningStepEntry[] }) {
  const steps = groupReasoningSteps(entries);

  if (steps.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.12 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <ListOrdered size={14} className="text-gh-accent-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Execution Steps
        </h3>
        <Badge variant="info">{steps.length}</Badge>
        <span className="text-2xs text-gh-fg-subtle ml-auto">LLM reasoning, tool calls, and results per step</span>
      </div>

      <div className="max-h-96 overflow-y-auto space-y-1">
        {steps.map((step, i) => (
          <motion.div
            key={step.step_number}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2, delay: i * 0.02 }}
            className={`rounded-md border transition-colors ${
              step.hasError
                ? 'border-gh-danger-emphasis/30 bg-gh-danger-emphasis/5'
                : step.success === true
                  ? 'border-gh-success-emphasis/20 bg-gh-success-emphasis/5'
                  : 'border-gh-border-muted bg-gh-canvas-subtle/30'
            }`}
          >
            {/* Step header */}
            <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gh-border-muted/50">
              <span className={`inline-flex items-center justify-center w-5 h-5 rounded text-2xs font-bold ${
                step.hasError
                  ? 'bg-gh-danger-emphasis/20 text-gh-danger-fg'
                  : step.success === true
                    ? 'bg-gh-success-emphasis/20 text-gh-success-fg'
                    : 'bg-gh-accent-fg/15 text-gh-accent-fg'
              }`}>
                {step.step_number}
              </span>
              {step.agent_name && (
                <span className="text-2xs font-mono text-gh-fg-subtle">{step.agent_name}</span>
              )}
              <span className="flex-1" />
              {step.success === true && (
                <CheckCircle size={11} className="text-gh-success-fg" />
              )}
              {step.success === false && (
                <XCircle size={11} className="text-gh-danger-fg" />
              )}
              <span className="text-2xs text-gh-fg-subtle font-mono">
                {step.timestamp ? step.timestamp.slice(11, 19) : ''}
              </span>
            </div>

            {/* Step body */}
            <div className="px-3 py-2 space-y-1.5">
              {/* Reasoning */}
              {step.reasoning && (
                <div className="flex items-start gap-1.5">
                  <MessageSquare size={10} className="text-gh-accent-fg shrink-0 mt-0.5" />
                  <span className="text-2xs text-gh-fg-default leading-relaxed">{step.reasoning}</span>
                </div>
              )}

              {/* Goal */}
              {step.goal && (
                <div className="flex items-start gap-1.5">
                  <ArrowRight size={10} className="text-gh-fg-subtle shrink-0 mt-0.5" />
                  <span className="text-2xs text-gh-fg-muted italic">{step.goal}</span>
                </div>
              )}

              {/* Tool call */}
              {step.tool && (
                <div className="flex items-start gap-1.5">
                  <Cog size={10} className="text-gh-attention-fg shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <span className="text-2xs font-mono font-semibold text-gh-attention-fg">{step.tool}</span>
                    {step.toolArgs && step.toolArgs !== '{}' && (
                      <span className="text-2xs font-mono text-gh-fg-subtle ml-1 truncate block" title={step.toolArgs}>
                        {step.toolArgs.length > 120 ? step.toolArgs.slice(0, 120) + '...' : step.toolArgs}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Result */}
              {step.result && (
                <div className="flex items-start gap-1.5">
                  {step.success === false ? (
                    <XCircle size={10} className="text-gh-danger-fg shrink-0 mt-0.5" />
                  ) : (
                    <CheckCircle size={10} className="text-gh-success-fg shrink-0 mt-0.5" />
                  )}
                  <span
                    className={`text-2xs font-mono truncate block ${
                      step.success === false ? 'text-gh-danger-fg' : 'text-gh-fg-muted'
                    }`}
                    title={step.result}
                  >
                    {step.result.length > 150 ? step.result.slice(0, 150) + '...' : step.result}
                  </span>
                </div>
              )}

              {/* Empty state for steps with no content */}
              {!step.reasoning && !step.goal && !step.tool && !step.result && (
                <div className="text-2xs text-gh-fg-subtle italic">No reasoning captured for this step</div>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// ============================================================================
// Execution Log Section (from logs.db runtime_logs)
// ============================================================================

type LogFilter = 'ALL' | 'ERROR' | 'WARNING' | 'INFO';

function ExecutionLogSection({ entries }: { entries: RuntimeLogEntry[] }) {
  const [filter, setFilter] = useState<LogFilter>('ALL');

  const filtered = filter === 'ALL'
    ? entries
    : entries.filter((e) => e.level === filter || (filter === 'ERROR' && e.level === 'CRITICAL'));

  const levelColor = (level: string) => {
    switch (level) {
      case 'ERROR':
      case 'CRITICAL':
        return 'text-gh-danger-fg';
      case 'WARNING':
        return 'text-gh-attention-fg';
      case 'INFO':
        return 'text-gh-accent-fg';
      case 'DEBUG':
        return 'text-gh-fg-subtle';
      default:
        return 'text-gh-fg-muted';
    }
  };

  const rowBg = (level: string) => {
    switch (level) {
      case 'ERROR':
      case 'CRITICAL':
        return 'bg-gh-danger-emphasis/5 border-l-2 border-l-gh-danger-fg';
      case 'WARNING':
        return 'bg-gh-attention-emphasis/5 border-l-2 border-l-gh-attention-fg';
      default:
        return '';
    }
  };

  const filters: LogFilter[] = ['ALL', 'ERROR', 'WARNING', 'INFO'];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.24 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <ScrollText size={14} className="text-gh-fg-muted" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Execution Log
        </h3>
        <Badge variant="neutral">{entries.length}</Badge>
        <span className="flex-1" />
        {/* Filter buttons */}
        <div className="flex items-center gap-1">
          <Filter size={10} className="text-gh-fg-subtle" />
          {filters.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-1.5 py-0.5 text-2xs rounded transition-colors ${
                filter === f
                  ? 'bg-gh-accent-fg/15 text-gh-accent-fg font-medium'
                  : 'text-gh-fg-subtle hover:text-gh-fg-muted hover:bg-gh-canvas-subtle'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {entries.length === 0 ? (
        <div className="py-8 text-center text-xs text-gh-fg-subtle">No activity yet</div>
      ) : filtered.length === 0 ? (
        <div className="py-6 text-center text-xs text-gh-fg-subtle">No {filter} entries</div>
      ) : (
        <div className="max-h-80 overflow-y-auto space-y-0.5">
          {filtered.map((entry, i) => (
            <div
              key={i}
              className={`flex items-start gap-2 px-2 py-1.5 rounded-sm text-2xs hover:bg-gh-canvas-subtle/30 transition-colors ${rowBg(entry.level)}`}
            >
              {/* Step */}
              <span className="text-gh-fg-subtle font-mono w-8 shrink-0 text-right">
                {entry.step_number != null ? `#${entry.step_number}` : ''}
              </span>
              {/* Level */}
              <span className={`font-mono font-semibold w-12 shrink-0 ${levelColor(entry.level)}`}>
                {entry.level}
              </span>
              {/* Agent */}
              {entry.agent_name && (
                <span className="text-gh-accent-fg shrink-0 font-mono" title={entry.agent_name}>
                  {truncate(entry.agent_name, 12)}
                </span>
              )}
              {/* Message */}
              <span className="flex-1 min-w-0 text-gh-fg-default truncate" title={entry.message}>
                {entry.message}
              </span>
              {/* Copy + Timestamp */}
              <CopyButton text={entry.message} />
              <span className="text-gh-fg-subtle shrink-0 font-mono">
                {entry.timestamp ? entry.timestamp.slice(11, 19) : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Conversation History Section (from logs.db conversation_turns)
// ============================================================================

function ConversationHistorySection({ turns }: { turns: ConversationTurnEntry[] }) {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [agentFilter, setAgentFilter] = useState<string>('ALL');

  const agents = ['ALL', ...Array.from(new Set(turns.map((t) => t.agent_name || 'unknown').filter(Boolean)))];

  const filtered = agentFilter === 'ALL'
    ? turns
    : turns.filter((t) => (t.agent_name || 'unknown') === agentFilter);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.22 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <MessagesSquare size={14} className="text-gh-accent-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Conversation History
        </h3>
        <Badge variant="neutral">{filtered.length}</Badge>
        <span className="flex-1" />
        {agents.length > 1 && (
          <div className="flex items-center gap-1">
            <Filter size={10} className="text-gh-fg-subtle" />
            {agents.map((a) => (
              <button
                key={a}
                onClick={() => setAgentFilter(a)}
                className={`px-1.5 py-0.5 text-2xs rounded transition-colors ${
                  agentFilter === a
                    ? 'bg-gh-accent-fg/15 text-gh-accent-fg font-medium'
                    : 'text-gh-fg-subtle hover:text-gh-fg-muted hover:bg-gh-canvas-subtle'
                }`}
              >
                {a === 'ALL' ? 'All' : a.replace('GaiaCodeAgent', 'GaiaCode').slice(0, 14)}
              </button>
            ))}
          </div>
        )}
      </div>

      {filtered.length === 0 ? (
        <div className="py-8 text-center">
          <MessagesSquare size={28} className="text-gh-fg-subtle mx-auto mb-2 opacity-40" />
          <div className="text-xs text-gh-fg-subtle">No conversation turns yet</div>
          <div className="text-2xs text-gh-fg-subtle mt-1">
            Turns appear here once the agent starts processing tasks.
          </div>
        </div>
      ) : (
        <div className="max-h-96 overflow-y-auto space-y-1">
          {filtered.map((turn) => {
            const isUser = turn.role === 'user';
            const isExpanded = expandedId === turn.id;
            const preview = turn.content.length > 120
              ? turn.content.slice(0, 120) + '…'
              : turn.content;

            return (
              <motion.div
                key={turn.id}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.12 }}
                className={`rounded-md border transition-colors cursor-pointer ${
                  isUser
                    ? 'border-gh-accent-fg/20 bg-gh-accent-fg/5 hover:bg-gh-accent-fg/10'
                    : 'border-gh-success-fg/20 bg-gh-success-fg/5 hover:bg-gh-success-fg/10'
                }`}
                onClick={() => setExpandedId(isExpanded ? null : turn.id)}
              >
                {/* Header row */}
                <div className="flex items-center gap-2 px-2.5 py-1.5">
                  {isUser ? (
                    <User size={11} className="text-gh-accent-fg shrink-0" />
                  ) : (
                    <Cpu size={11} className="text-gh-success-fg shrink-0" />
                  )}
                  <span className={`text-2xs font-semibold shrink-0 ${isUser ? 'text-gh-accent-fg' : 'text-gh-success-fg'}`}>
                    {isUser ? 'User' : 'Assistant'}
                  </span>
                  {turn.step_number != null && (
                    <span className="text-2xs font-mono text-gh-fg-subtle shrink-0">
                      step #{turn.step_number}
                    </span>
                  )}
                  {turn.agent_name && (
                    <span className="text-2xs text-gh-fg-subtle shrink-0 truncate max-w-[80px]" title={turn.agent_name}>
                      {turn.agent_name.replace('GaiaCodeAgent', 'GaiaCode')}
                    </span>
                  )}
                  <span className="flex-1 min-w-0 truncate text-2xs text-gh-fg-muted font-mono" title={turn.content}>
                    {!isExpanded && preview}
                  </span>
                  <span className="text-2xs text-gh-fg-subtle shrink-0 font-mono">
                    {turn.timestamp ? turn.timestamp.slice(11, 19) : ''}
                  </span>
                  {isExpanded
                    ? <ChevronDown size={10} className="shrink-0 text-gh-fg-subtle" />
                    : <ChevronRight size={10} className="shrink-0 text-gh-fg-subtle" />
                  }
                </div>

                {/* Expanded content */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.15 }}
                      className="px-2.5 pb-2"
                    >
                      <pre className="text-2xs font-mono text-gh-fg-default whitespace-pre-wrap break-words max-h-48 overflow-y-auto bg-gh-canvas-subtle rounded p-2 border border-gh-border-subtle">
                        {turn.content}
                      </pre>
                      {turn.model_id && (
                        <div className="mt-1 text-2xs text-gh-fg-subtle">
                          model: {turn.model_id}
                          {turn.token_count != null && ` · ${turn.token_count} tokens`}
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Plan Tree Section (from memory.db plan_tasks)
// ============================================================================

function PlanTreeSection({ tasks }: { tasks: PlanTreeTask[] }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const onToggle = useCallback((id: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  if (tasks.length === 0) {
    return null;
  }

  // Build parent-child map
  const childMap = new Map<string | null, PlanTreeTask[]>();
  for (const t of tasks) {
    const parentKey = t.parent_id || null;
    if (!childMap.has(parentKey)) childMap.set(parentKey, []);
    childMap.get(parentKey)!.push(t);
  }

  // Find roots (depth=0 or no parent_id)
  const roots = childMap.get(null) || tasks.filter((t) => t.depth === 0);

  const statusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-gh-success-fg';
      case 'in_progress': return 'text-blue-400';
      case 'failed': return 'text-gh-danger-fg';
      case 'pending': return 'text-gh-fg-subtle';
      case 'blocked': return 'text-gh-attention-fg';
      default: return 'text-gh-fg-subtle';
    }
  };

  const statusDot = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-gh-success-fg';
      case 'in_progress': return 'bg-blue-400';
      case 'failed': return 'bg-gh-danger-fg';
      case 'pending': return 'bg-gh-canvas-subtle border border-gh-border-muted';
      case 'blocked': return 'bg-gh-attention-fg';
      default: return 'bg-gh-canvas-subtle border border-gh-border-muted';
    }
  };

  function renderTask(task: PlanTreeTask) {
    const children = childMap.get(task.id) || [];
    const hasChildren = children.length > 0;
    const isCollapsed = collapsed.has(task.id);
    const indent = task.depth * 16;

    // Compute duration
    let duration = '';
    if (task.started_at && task.completed_at) {
      const ms = new Date(task.completed_at).getTime() - new Date(task.started_at).getTime();
      if (ms > 0) duration = formatDuration(ms);
    } else if (task.started_at && task.status === 'in_progress') {
      const ms = Date.now() - new Date(task.started_at).getTime();
      if (ms > 0) duration = formatDuration(ms);
    }

    return (
      <React.Fragment key={task.id}>
        <div
          className="flex items-center gap-1.5 py-1 px-2 rounded-sm hover:bg-gh-canvas-subtle/30 transition-colors group"
          style={{ paddingLeft: `${8 + indent}px` }}
        >
          {/* Expand/collapse */}
          {hasChildren ? (
            <button
              onClick={() => onToggle(task.id)}
              className="w-4 h-4 flex items-center justify-center text-gh-fg-subtle hover:text-gh-fg-muted transition-colors shrink-0"
            >
              {isCollapsed ? <ChevronRight size={10} /> : <ChevronDown size={10} />}
            </button>
          ) : (
            <span className="w-4 h-4 shrink-0" />
          )}

          {/* Status dot */}
          <span className={`w-2 h-2 rounded-full shrink-0 ${statusDot(task.status)}`} />

          {/* Title */}
          <span
            className={`flex-1 min-w-0 truncate text-2xs ${
              task.depth === 0
                ? 'font-semibold text-gh-fg-default'
                : 'text-gh-fg-muted'
            }`}
            title={task.title}
          >
            {task.title}
          </span>

          {/* Duration */}
          {duration && (
            <span className={`text-2xs font-mono shrink-0 ${statusColor(task.status)}`}>
              {duration}
            </span>
          )}

          {/* Status label */}
          <span className={`text-2xs shrink-0 ${statusColor(task.status)}`}>
            {task.status}
          </span>

          {/* Copy */}
          <CopyButton text={task.title} className="opacity-0 group-hover:opacity-100" />
        </div>

        {/* Children */}
        {hasChildren && !isCollapsed && children.map(renderTask)}
      </React.Fragment>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.26 }}
      className="card card-hover p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <TreePine size={14} className="text-gh-success-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Plan Tree
        </h3>
        <Badge variant="neutral">{tasks.length} tasks</Badge>
      </div>
      <div className="max-h-80 overflow-y-auto">
        {roots.map(renderTask)}
      </div>
    </motion.div>
  );
}

// ============================================================================
// Resource Usage Section (Skills, Memory, Knowledge -- Top Agents removed)
// ============================================================================

interface ResourceUsageProps {
  skills: SkillEntry[];
  memoryTools: MemoryToolEntry[];
  knowledge: KnowledgeInsightEntry[];
}

function ResourceUsage({ skills, memoryTools, knowledge }: ResourceUsageProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
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
                  <span className="text-xs text-gh-fg-default truncate block" title={skill.name}>
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
                <span className="text-xs text-gh-fg-default truncate" title={mt.tool_name}>
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
                  <span className="text-xs text-gh-fg-default truncate block" title={k.content}>
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
