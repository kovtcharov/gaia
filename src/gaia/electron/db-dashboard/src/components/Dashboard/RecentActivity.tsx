// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Lightbulb,
  ListTodo,
  AlertTriangle,
  XCircle,
  Loader2,
} from 'lucide-react';
import Badge from '../shared/Badge';
import type { ErrorEntry, TaskEntry, InsightEntry } from '../../types/database';

interface RecentActivityProps {
  errors: ErrorEntry[];
  tasks: TaskEntry[];
  insights: InsightEntry[];
}

function formatRelative(timestamp: string | null): string {
  if (!timestamp) return '';
  const diff = Date.now() - new Date(timestamp).getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  return Math.floor(diff / 86400000) + 'd ago';
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + '...';
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 size={14} className="text-gh-success-fg" />;
    case 'in_progress':
      return <Loader2 size={14} className="text-gh-accent-fg animate-spin" />;
    case 'failed':
      return <XCircle size={14} className="text-gh-danger-fg" />;
    case 'pending':
      return <Clock size={14} className="text-gh-fg-subtle" />;
    default:
      return <Clock size={14} className="text-gh-fg-subtle" />;
  }
}

function statusVariant(status: string): 'success' | 'warning' | 'danger' | 'info' | 'neutral' {
  switch (status) {
    case 'completed': return 'success';
    case 'in_progress': return 'info';
    case 'failed': return 'danger';
    case 'pending': return 'neutral';
    default: return 'neutral';
  }
}

export default function RecentActivity({ errors, tasks, insights }: RecentActivityProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      {/* Recent Errors */}
      <div className="card card-hover p-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle size={14} className="text-gh-danger-fg" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Recent Errors
          </h3>
          <Badge variant={errors.length > 0 ? 'danger' : 'success'}>
            {errors.length}
          </Badge>
        </div>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {errors.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No recent errors</div>
          ) : (
            errors.map((err, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-start gap-2 p-2 rounded-md bg-gh-danger-emphasis/5 border border-gh-danger-emphasis/10"
              >
                <AlertTriangle size={12} className="text-gh-danger-fg shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-gh-fg-default truncate">{truncate(err.message, 80)}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <Badge variant="danger">{err.level}</Badge>
                    {err.step_number != null && (
                      <span className="text-2xs text-gh-fg-subtle">Step {err.step_number}</span>
                    )}
                    <span className="text-2xs text-gh-fg-subtle">{formatRelative(err.timestamp)}</span>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>

      {/* Active Tasks */}
      <div className="card card-hover p-4">
        <div className="flex items-center gap-2 mb-3">
          <ListTodo size={14} className="text-gh-accent-fg" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Active Tasks
          </h3>
          <Badge variant="info">{tasks.length}</Badge>
        </div>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {tasks.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No tasks found</div>
          ) : (
            tasks.map((task, i) => (
              <motion.div
                key={task.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-start gap-2 p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
              >
                <StatusIcon status={task.status} />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-gh-fg-default truncate">
                    {truncate(task.description, 70)}
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <Badge variant={statusVariant(task.status)}>{task.status}</Badge>
                    {task.priority != null && (
                      <span className="text-2xs text-gh-fg-subtle">P{task.priority}</span>
                    )}
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>

      {/* Recent Insights */}
      <div className="card card-hover p-4">
        <div className="flex items-center gap-2 mb-3">
          <Lightbulb size={14} className="text-gh-attention-fg" />
          <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Recent Insights
          </h3>
          <Badge variant="warning">{insights.length}</Badge>
        </div>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {insights.length === 0 ? (
            <div className="py-6 text-center text-xs text-gh-fg-subtle">No insights yet</div>
          ) : (
            insights.map((ins, i) => (
              <motion.div
                key={ins.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-start gap-2 p-2 rounded-md hover:bg-gh-canvas-subtle/50 transition-colors"
              >
                <Lightbulb size={12} className="text-gh-attention-fg shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-gh-fg-default">{truncate(ins.content, 80)}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {ins.category && <Badge variant="purple">{ins.category}</Badge>}
                    {ins.confidence != null && (
                      <span className="text-2xs text-gh-fg-subtle">
                        {Math.round(ins.confidence * 100)}%
                      </span>
                    )}
                    <span className="text-2xs text-gh-fg-subtle">
                      {formatRelative(ins.created_at)}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
