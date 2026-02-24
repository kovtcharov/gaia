// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  MinusCircle,
  Loader2,
  ChevronRight,
  ChevronDown,
  ListTodo,
  FolderOpen,
  History,
  Timer,
} from 'lucide-react';
import type { ActivePlan, PlanTask, PlanHistoryEntry } from '../../types/database';
import Badge from '../shared/Badge';

// ============================================================================
// Time helpers
// ============================================================================

/** Format a duration in milliseconds as a compact human string: "42s", "3m 12s", "2h 15m" */
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

/** Duration between two timestamp strings (or start → now if end is null). */
function durationBetween(startStr: string | null, endStr: string | null): string {
  if (!startStr) return '';
  const start = new Date(startStr).getTime();
  if (isNaN(start)) return '';
  const end = endStr ? new Date(endStr).getTime() : Date.now();
  if (isNaN(end)) return '';
  return formatDuration(end - start);
}

function formatRelativeMinutes(dateStr: string | null): string {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  const diff = Date.now() - d.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  return Math.floor(diff / 86400000) + 'd ago';
}

// ============================================================================
// Status helpers
// ============================================================================

const STATUS_CONFIG: Record<string, { icon: React.ElementType; color: string; label: string }> = {
  pending:     { icon: Clock,        color: 'text-gh-fg-subtle',    label: 'Pending' },
  in_progress: { icon: Loader2,      color: 'text-blue-400',        label: 'In Progress' },
  completed:   { icon: CheckCircle2, color: 'text-gh-success-fg',   label: 'Completed' },
  failed:      { icon: XCircle,      color: 'text-gh-danger-fg',    label: 'Failed' },
  blocked:     { icon: AlertCircle,  color: 'text-gh-attention-fg', label: 'Blocked' },
  cancelled:   { icon: MinusCircle,  color: 'text-gh-fg-subtle',    label: 'Cancelled' },
  active:      { icon: Loader2,      color: 'text-blue-400',        label: 'Active' },
  abandoned:   { icon: MinusCircle,  color: 'text-gh-danger-fg',    label: 'Abandoned' },
};

function getStatusConfig(status: string) {
  return STATUS_CONFIG[status] || STATUS_CONFIG.pending;
}

function planBadgeVariant(status: string): 'success' | 'danger' | 'info' | 'warning' {
  if (status === 'completed') return 'success';
  if (status === 'abandoned') return 'danger';
  if (status === 'active' || status === 'in_progress') return 'info';
  return 'warning';
}

// ============================================================================
// Live elapsed timer (ticks every second for in-progress items)
// ============================================================================

function useLiveElapsed(startStr: string | null, isActive: boolean): string {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!isActive || !startStr) return;
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, [isActive, startStr]);
  if (!isActive || !startStr) return durationBetween(startStr, null);
  return durationBetween(startStr, null);
}

// ============================================================================
// Progress Bar
// ============================================================================

function PlanProgressBar({
  progress,
  duration,
}: {
  progress: ActivePlan['progress'];
  duration: string;
}) {
  if (progress.total === 0) return null;

  const pct = (n: number) => (n / progress.total) * 100;
  const completedPct = pct(progress.completed);
  const inProgressPct = pct(progress.in_progress);
  const failedPct = pct(progress.failed);
  const blockedPct = pct(progress.blocked);

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs text-gh-fg-muted">
          {progress.completed}/{progress.total} tasks completed
        </span>
        <div className="flex items-center gap-2">
          {duration && (
            <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle font-mono">
              <Timer size={10} className="text-gh-fg-subtle" />
              {duration}
            </span>
          )}
          <span className="text-xs font-mono text-gh-fg-muted">
            {Math.round((progress.completed / progress.total) * 100)}%
          </span>
        </div>
      </div>
      <div className="w-full h-2 bg-gh-canvas-subtle rounded-full overflow-hidden flex">
        {completedPct > 0 && (
          <div className="h-full bg-gh-success-fg transition-all duration-500" style={{ width: `${completedPct}%` }} />
        )}
        {inProgressPct > 0 && (
          <div className="h-full bg-blue-400 transition-all duration-500" style={{ width: `${inProgressPct}%` }} />
        )}
        {failedPct > 0 && (
          <div className="h-full bg-gh-danger-fg transition-all duration-500" style={{ width: `${failedPct}%` }} />
        )}
        {blockedPct > 0 && (
          <div className="h-full bg-gh-attention-fg transition-all duration-500" style={{ width: `${blockedPct}%` }} />
        )}
      </div>
      {/* Legend */}
      <div className="flex items-center gap-3 mt-1.5 flex-wrap">
        {progress.completed > 0 && (
          <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle">
            <span className="w-2 h-2 rounded-full bg-gh-success-fg inline-block" />
            {progress.completed} done
          </span>
        )}
        {progress.in_progress > 0 && (
          <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle">
            <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
            {progress.in_progress} active
          </span>
        )}
        {progress.failed > 0 && (
          <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle">
            <span className="w-2 h-2 rounded-full bg-gh-danger-fg inline-block" />
            {progress.failed} failed
          </span>
        )}
        {progress.blocked > 0 && (
          <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle">
            <span className="w-2 h-2 rounded-full bg-gh-attention-fg inline-block" />
            {progress.blocked} blocked
          </span>
        )}
        {progress.pending > 0 && (
          <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle">
            <span className="w-2 h-2 rounded-full bg-gh-canvas-subtle border border-gh-border-muted inline-block" />
            {progress.pending} pending
          </span>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Single Task Row
// ============================================================================

interface TaskRowProps {
  task: PlanTask;
  collapsed: Set<string>;
  onToggle: (id: string) => void;
}

function TaskRow({ task, collapsed, onToggle }: TaskRowProps) {
  const cfg = getStatusConfig(task.status);
  const StatusIcon = cfg.icon;
  const hasChildren = task.children.length > 0;
  const isCollapsed = collapsed.has(task.id);
  const indent = task.depth * 16;

  const isMilestone = task.depth === 0;
  const isSubtask = task.depth >= 2;
  const isActive = task.status === 'in_progress';

  // Live elapsed for in-progress; static duration for finished tasks
  const elapsed = useLiveElapsed(task.started_at, isActive);
  const taskDuration = isActive
    ? elapsed
    : task.status === 'completed' || task.status === 'failed'
    ? durationBetween(task.started_at, task.completed_at)
    : '';

  return (
    <>
      <motion.div
        initial={{ opacity: 0, x: -6 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.15 }}
        className={`flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors group ${
          isMilestone ? 'mt-1' : ''
        }`}
        style={{ paddingLeft: `${8 + indent}px` }}
      >
        {/* Expand/collapse toggle */}
        {hasChildren ? (
          <button
            onClick={() => onToggle(task.id)}
            className="w-4 h-4 flex items-center justify-center text-gh-fg-subtle hover:text-gh-fg-muted transition-colors shrink-0"
          >
            {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
          </button>
        ) : (
          <span className="w-4 h-4 shrink-0" />
        )}

        {/* Status icon */}
        <StatusIcon
          size={isMilestone ? 16 : 14}
          className={`${cfg.color} shrink-0 ${isActive ? 'animate-spin' : ''}`}
          style={isActive ? { animationDuration: '2s' } : undefined}
        />

        {/* Title */}
        <span
          className={`flex-1 min-w-0 truncate ${
            isMilestone
              ? 'text-xs font-semibold text-gh-fg-default'
              : isSubtask
              ? 'text-2xs text-gh-fg-muted'
              : 'text-xs text-gh-fg-default'
          }`}
        >
          {task.title}
        </span>

        {/* Duration chip — shown for completed/failed/in_progress tasks that have started */}
        {taskDuration && (
          <span
            className={`flex items-center gap-0.5 text-2xs font-mono shrink-0 ${
              isActive
                ? 'text-blue-400'
                : task.status === 'failed'
                ? 'text-gh-danger-fg'
                : 'text-gh-fg-subtle'
            }`}
            title={isActive ? 'Elapsed time' : 'Task duration'}
          >
            <Timer size={9} />
            {taskDuration}
          </span>
        )}

        {/* Agent badge for in_progress tasks */}
        {task.owner && isActive && (
          <span className="text-2xs px-1.5 py-0.5 rounded-full bg-blue-400/15 text-blue-400 border border-blue-400/30 shrink-0">
            {task.owner}
          </span>
        )}

        {/* Error indicator for failed tasks */}
        {task.status === 'failed' && task.error && (
          <span
            className="text-2xs text-gh-danger-fg truncate max-w-[100px] shrink-0"
            title={task.error}
          >
            {task.error.length > 18 ? task.error.slice(0, 18) + '…' : task.error}
          </span>
        )}
      </motion.div>

      {/* Render children if not collapsed */}
      <AnimatePresence>
        {hasChildren && !isCollapsed && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
          >
            {task.children.map(child => (
              <TaskRow key={child.id} task={child} collapsed={collapsed} onToggle={onToggle} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

// ============================================================================
// Plan History Row (compact)
// ============================================================================

function PlanHistoryRow({ entry, isFirst }: { entry: PlanHistoryEntry; isFirst: boolean }) {
  const cfg = getStatusConfig(entry.status);
  const StatusIcon = cfg.icon;
  const isActive = entry.status === 'active' || entry.status === 'in_progress';
  const completedPct = entry.task_count > 0
    ? Math.round((entry.completed_tasks / entry.task_count) * 100)
    : 0;

  // Duration: created → completed (or now for active)
  const planDuration = durationBetween(
    entry.created_at,
    isActive ? null : entry.completed_at,
  );

  return (
    <div
      className={`flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors ${
        isFirst ? 'border border-gh-border-muted/50' : ''
      }`}
    >
      <StatusIcon
        size={13}
        className={`${cfg.color} shrink-0 ${isActive ? 'animate-spin' : ''}`}
        style={isActive ? { animationDuration: '2s' } : undefined}
      />

      {/* Title */}
      <span className="flex-1 min-w-0 truncate text-xs text-gh-fg-default" title={entry.title}>
        {entry.title}
      </span>

      {/* Task counts */}
      {entry.task_count > 0 && (
        <span className="text-2xs text-gh-fg-subtle shrink-0 font-mono">
          {entry.completed_tasks}/{entry.task_count}
          {entry.failed_tasks > 0 && (
            <span className="text-gh-danger-fg ml-1">({entry.failed_tasks}✗)</span>
          )}
        </span>
      )}

      {/* Mini progress bar */}
      {entry.task_count > 0 && (
        <div className="w-10 h-1.5 bg-gh-canvas-subtle rounded-full overflow-hidden shrink-0">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              completedPct === 100
                ? 'bg-gh-success-fg'
                : entry.status === 'abandoned'
                ? 'bg-gh-danger-fg'
                : 'bg-blue-400'
            }`}
            style={{ width: `${completedPct}%` }}
          />
        </div>
      )}

      {/* Duration */}
      {planDuration && (
        <span className="flex items-center gap-0.5 text-2xs font-mono text-gh-fg-subtle shrink-0">
          <Timer size={9} />
          {planDuration}
        </span>
      )}

      {/* Relative timestamp */}
      <span className="text-2xs text-gh-fg-subtle shrink-0">
        {formatRelativeMinutes(entry.completed_at || entry.created_at)}
      </span>
    </div>
  );
}

// ============================================================================
// PlanView (main export)
// ============================================================================

export default function PlanView({
  plan,
  planHistory = [],
}: {
  plan: ActivePlan | null;
  planHistory?: PlanHistoryEntry[];
}) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [showHistory, setShowHistory] = useState(false);

  const onToggle = useCallback((id: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // History excludes the current active plan
  const historyEntries = plan
    ? planHistory.filter(e => e.id !== plan.id)
    : planHistory;

  // Plan total runtime
  const planIsActive = plan?.status === 'active' || plan?.status === 'in_progress';
  const planDuration = plan
    ? durationBetween(plan.created_at, planIsActive ? null : plan.completed_at)
    : '';

  // Live tick for active plan duration
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!planIsActive) return;
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, [planIsActive]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="card card-hover p-4"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <ListTodo size={14} className="text-gh-accent-fg" />
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
          Plan
        </h3>
        {plan && (
          <Badge variant={planBadgeVariant(plan.status)}>{plan.status}</Badge>
        )}
        <span className="flex-1" />
        {historyEntries.length > 0 && (
          <button
            onClick={() => setShowHistory(v => !v)}
            className="flex items-center gap-1 text-2xs text-gh-fg-subtle hover:text-gh-fg-muted transition-colors"
          >
            <History size={11} />
            {historyEntries.length} previous
            {showHistory ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          </button>
        )}
      </div>

      {/* Empty state */}
      {(!plan || plan.tasks.length === 0) && planHistory.length === 0 && (
        <div className="py-8 text-center">
          <ListTodo size={32} className="text-gh-fg-subtle mx-auto mb-2 opacity-40" />
          <div className="text-xs text-gh-fg-subtle">No plans yet.</div>
          <div className="text-2xs text-gh-fg-subtle mt-1">
            The agent creates a plan when it starts working on a task.
          </div>
        </div>
      )}

      {/* No active plan but has history */}
      {(!plan || plan.tasks.length === 0) && planHistory.length > 0 && (
        <div className="mb-2 text-2xs text-gh-fg-subtle italic">
          No current plan — showing history
        </div>
      )}

      {/* Active plan content */}
      {plan && plan.tasks.length > 0 && (
        <>
          {/* Plan title + meta row */}
          <div className="mb-3">
            <div className="text-sm font-medium text-gh-fg-default">{plan.title}</div>
            <div className="flex items-center gap-3 mt-1 flex-wrap">
              {plan.project_dir && (
                <div className="flex items-center gap-1">
                  <FolderOpen size={10} className="text-gh-fg-subtle" />
                  <span className="text-2xs text-gh-fg-subtle font-mono truncate max-w-[200px]">
                    {plan.project_dir}
                  </span>
                </div>
              )}
              <span className="text-2xs text-gh-fg-subtle">
                {formatRelativeMinutes(plan.created_at)}
              </span>
            </div>
          </div>

          {/* Progress bar with total plan duration */}
          <PlanProgressBar progress={plan.progress} duration={planDuration} />

          {/* Task tree */}
          <div className="max-h-96 overflow-y-auto -mx-2">
            {plan.tasks.map(task => (
              <TaskRow
                key={task.id}
                task={task}
                collapsed={collapsed}
                onToggle={onToggle}
              />
            ))}
          </div>
        </>
      )}

      {/* Plan history */}
      <AnimatePresence>
        {(showHistory || (!plan || plan.tasks.length === 0)) && planHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
          >
            {plan && plan.tasks.length > 0 && historyEntries.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gh-border-subtle">
                <div className="text-2xs text-gh-fg-subtle uppercase tracking-wider mb-2 flex items-center gap-1">
                  <History size={10} />
                  Previous Plans
                </div>
              </div>
            )}
            <div className="space-y-0.5">
              {(!plan || plan.tasks.length === 0)
                ? planHistory.map((entry, i) => (
                    <PlanHistoryRow key={entry.id} entry={entry} isFirst={i === 0} />
                  ))
                : historyEntries.map(entry => (
                    <PlanHistoryRow key={entry.id} entry={entry} isFirst={false} />
                  ))
              }
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
