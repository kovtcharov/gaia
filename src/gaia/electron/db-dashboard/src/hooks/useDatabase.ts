// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * React Query hooks for database operations.
 * Uses keepPreviousData to prevent flashing on refetch.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type {
  QueryOptions,
  DatabaseInfo,
  TableInfo,
  Row,
  PaginationInfo,
  DashboardData,
  TrendStats,
  ExecuteSQLResponse,
  WorkingMemoryEntry,
  PlanTask,
  PlanHistoryEntry,
} from '../types/database';

const api = () => window.dbAPI;

// ============================================================================
// Database List
// ============================================================================

export function useDatabases(workspacePath: string | null) {
  return useQuery({
    queryKey: ['databases', workspacePath],
    queryFn: async () => {
      const result = await api().listDatabases(workspacePath || undefined);
      if (!result.success) throw new Error(result.error);
      return result.databases;
    },
    enabled: !!workspacePath,
    placeholderData: (prev) => prev,
  });
}

// ============================================================================
// Table List
// ============================================================================

export function useTables(dbPath: string | null) {
  return useQuery({
    queryKey: ['tables', dbPath],
    queryFn: async () => {
      if (!dbPath) return [];
      const result = await api().getTables(dbPath);
      if (!result.success) throw new Error(result.error);
      return result.tables;
    },
    enabled: !!dbPath,
    placeholderData: (prev) => prev,
  });
}

// ============================================================================
// Table Data (paginated)
// ============================================================================

export function useTableData(
  dbPath: string | null,
  tableName: string | null,
  options: QueryOptions,
  refreshInterval: number,
) {
  return useQuery({
    queryKey: ['tableData', dbPath, tableName, options],
    queryFn: async () => {
      if (!dbPath || !tableName) return { rows: [], pagination: { page: 1, limit: 50, totalRows: 0, totalPages: 0 } };
      const result = await api().queryTable(dbPath, tableName, options);
      if (!result.success) throw new Error(result.error);
      return { rows: result.rows, pagination: result.pagination };
    },
    enabled: !!dbPath && !!tableName,
    refetchInterval: refreshInterval > 0 ? refreshInterval : false,
    placeholderData: (prev) => prev,
  });
}

// ============================================================================
// SQL Execution
// ============================================================================

export function useExecuteSQL() {
  return useMutation({
    mutationFn: async ({
      dbPath,
      sql,
      readOnly,
    }: {
      dbPath: string;
      sql: string;
      readOnly: boolean;
    }) => {
      const result = await api().executeSQL(dbPath, sql, readOnly);
      if (!result.success) throw new Error(result.error);
      return result;
    },
  });
}

// ============================================================================
// Schema
// ============================================================================

export function useSchema(dbPath: string | null) {
  return useQuery({
    queryKey: ['schema', dbPath],
    queryFn: async () => {
      if (!dbPath) return [];
      const result = await api().getSchema(dbPath);
      if (!result.success) throw new Error(result.error);
      return result.schemas;
    },
    enabled: !!dbPath,
  });
}

// ============================================================================
// FTS5 Search
// ============================================================================

export function useFTS5Search() {
  return useMutation({
    mutationFn: async ({
      dbPath,
      ftsTable,
      query,
      limit,
    }: {
      dbPath: string;
      ftsTable: string;
      query: string;
      limit?: number;
    }) => {
      const result = await api().searchFTS5(dbPath, ftsTable, query, { limit });
      if (!result.success) throw new Error(result.error);
      return result;
    },
  });
}

// ============================================================================
// Row Mutations
// ============================================================================

export function useInsertRow() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({
      dbPath,
      tableName,
      data,
    }: {
      dbPath: string;
      tableName: string;
      data: Record<string, unknown>;
    }) => {
      const result = await api().insertRow(dbPath, tableName, data);
      if (!result.success) throw new Error(result.error);
      return result;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tableData'] });
      queryClient.invalidateQueries({ queryKey: ['tables'] });
    },
  });
}

export function useUpdateRow() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({
      dbPath,
      tableName,
      primaryKey,
      updates,
    }: {
      dbPath: string;
      tableName: string;
      primaryKey: { column: string; value: unknown };
      updates: Record<string, unknown>;
    }) => {
      const result = await api().updateRow(dbPath, tableName, primaryKey, updates);
      if (!result.success) throw new Error(result.error);
      return result;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tableData'] });
    },
  });
}

export function useDeleteRow() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({
      dbPath,
      tableName,
      primaryKey,
    }: {
      dbPath: string;
      tableName: string;
      primaryKey: { column: string; value: unknown };
    }) => {
      const result = await api().deleteRow(dbPath, tableName, primaryKey);
      if (!result.success) throw new Error(result.error);
      return result;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tableData'] });
      queryClient.invalidateQueries({ queryKey: ['tables'] });
    },
  });
}

export function useClearTable() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ dbPath, tableName }: { dbPath: string; tableName: string }) => {
      const result = await api().clearTable(dbPath, tableName);
      if (!result.success) throw new Error(result.error);
      return result;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tableData'] });
      queryClient.invalidateQueries({ queryKey: ['tables'] });
      queryClient.invalidateQueries({ queryKey: ['dashboardData'] });
    },
  });
}

// ============================================================================
// Dashboard Data (aggregate)
// ============================================================================

export function useDashboardData(
  workspacePath: string | null,
  databases: DatabaseInfo[] | undefined,
  refreshInterval: number,
) {
  return useQuery({
    queryKey: ['dashboardData', workspacePath, databases?.map((d) => d.name).join(',')],
    queryFn: async (): Promise<DashboardData> => {
      if (!databases) throw new Error('No databases');
      const existingDbs = databases.filter((d) => d.exists);

      const totalSize = existingDbs.reduce((sum, db) => sum + db.sizeBytes, 0);
      const lastActivity = existingDbs.reduce((latest: Date | null, db) => {
        if (!db.lastModified) return latest;
        const t = new Date(db.lastModified);
        return !latest || t > latest ? t : latest;
      }, null);

      const data: DashboardData = {
        totalSize,
        dbCount: existingDbs.length,
        lastActivity,
        dbStats: [],
        recentErrors: [],
        activeTasks: [],
        recentInsights: [],
        topTools: [],
        activityHeatmap: [],
        minuteActivity: [],
        contextUsage: [],
        trendStats: null,
        topAgents: [],
        topSkills: [],
        topMemoryTools: [],
        topKnowledge: [],
        learnedTools: [],
        learnedToolsCount: 0,
        workingMemory: [],
        activePlan: null,
        planHistory: [],
        agentSpecialists: [],
        recentAgentCalls: [],
      };

      // Gather per-database stats
      for (const db of existingDbs) {
        try {
          const tablesResult = await api().getTables(db.path);
          if (tablesResult.success) {
            const totalRows = tablesResult.tables.reduce((sum, t) => sum + (t.rowCount || 0), 0);
            data.dbStats.push({
              name: db.name,
              label: db.label || db.name.replace('.db', ''),
              sizeBytes: db.sizeBytes,
              lastModified: db.lastModified,
              tableCount: tablesResult.tables.length,
              totalRows,
            });
          }
        } catch {
          // Skip databases that fail
        }
      }

      // Declare DB refs early so they're available throughout
      const logsDb = existingDbs.find((d) => d.name === 'logs.db');
      const memoryDb = existingDbs.find((d) => d.name === 'memory.db');

      // Query logs.db
      if (logsDb) {
        try {
          const errResult = await api().executeSQL(
            logsDb.path,
            `SELECT level, message, timestamp, step_number FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL') ORDER BY timestamp DESC LIMIT 10`,
            true,
          );
          if (errResult.success && 'rows' in errResult) {
            data.recentErrors = errResult.rows as DashboardData['recentErrors'];
          }
        } catch { /* ignore */ }

        try {
          const heatmapResult = await api().executeSQL(
            logsDb.path,
            `SELECT STRFTIME('%H:00', timestamp) as hour_label, COUNT(*) as count FROM runtime_logs WHERE timestamp >= DATETIME('now', '-24 hours') GROUP BY STRFTIME('%Y-%m-%d %H', timestamp) ORDER BY timestamp ASC`,
            true,
          );
          if (heatmapResult.success && 'rows' in heatmapResult) {
            data.activityHeatmap = heatmapResult.rows as DashboardData['activityHeatmap'];
          }
        } catch { /* ignore */ }

        try {
          const minuteResult = await api().executeSQL(
            logsDb.path,
            `SELECT STRFTIME('%H:%M', DATETIME((CAST(STRFTIME('%s', timestamp) AS INTEGER) / 300) * 300, 'unixepoch')) as bucket_label, COUNT(*) as count FROM runtime_logs WHERE timestamp >= DATETIME('now', '-60 minutes') GROUP BY CAST(STRFTIME('%s', timestamp) AS INTEGER) / 300 ORDER BY bucket_label ASC`,
            true,
          );
          if (minuteResult.success && 'rows' in minuteResult) {
            data.minuteActivity = minuteResult.rows as DashboardData['minuteActivity'];
          }
        } catch { /* ignore */ }

        try {
          const usageResult = await api().executeSQL(
            logsDb.path,
            `SELECT step_number, context_tokens, max_context_tokens, timestamp FROM runtime_logs WHERE context_tokens IS NOT NULL ORDER BY step_number ASC LIMIT 100`,
            true,
          );
          if (usageResult.success && 'rows' in usageResult) {
            data.contextUsage = usageResult.rows as DashboardData['contextUsage'];
          }
        } catch { /* ignore */ }

        // Trend stats
        data.trendStats = await computeTrendStats(logsDb.path);

        // Tool calls count for trend
        const toolsDb = existingDbs.find((d) => d.name === 'tools.db');
        if (toolsDb && data.trendStats) {
          try {
            const totalToolResult = await api().executeSQL(
              toolsDb.path,
              `SELECT COUNT(*) as total FROM tool_usage`,
              true,
            );
            if (totalToolResult.success && 'rows' in totalToolResult && totalToolResult.rows.length > 0) {
              data.trendStats.totalToolCalls = (totalToolResult.rows[0] as Record<string, number>).total || 0;
            }
          } catch { /* ignore */ }
        }
      }

      // Query memory.db — active plan + full task tree (plans/plan_tasks live here)
      // Note: plan data was consolidated from plan.db into memory.db so all agent
      // working state is co-located in one file.
      if (memoryDb) {
        try {
          // Active tasks summary for the activeTasks widget (legacy widget)
          const taskResult = await api().executeSQL(
            memoryDb.path,
            `SELECT id, title as description, status, priority FROM plan_tasks
             ORDER BY CASE status WHEN 'in_progress' THEN 0 WHEN 'pending' THEN 1
               WHEN 'completed' THEN 2 WHEN 'failed' THEN 3 ELSE 4 END,
             created_at DESC LIMIT 10`,
            true,
          );
          if (taskResult.success && 'rows' in taskResult) {
            data.activeTasks = taskResult.rows as DashboardData['activeTasks'];
          }
        } catch { /* ignore */ }

        // Full plan tree for PlanView
        try {
          const planResult = await api().executeSQL(
            memoryDb.path,
            `SELECT id, title, status, project_dir, target_dir, created_at, completed_at
             FROM plans ORDER BY created_at DESC LIMIT 1`,
            true,
          );
          if (planResult.success && 'rows' in planResult && planResult.rows.length > 0) {
            const plan = planResult.rows[0] as Record<string, string>;

            const tasksResult = await api().executeSQL(
              memoryDb.path,
              `SELECT id, plan_id, parent_id, title, description, status, priority, depth,
                      owner, created_by, result, error, dependencies, order_index,
                      created_at, updated_at, started_at, completed_at
               FROM plan_tasks
               WHERE plan_id = (SELECT id FROM plans ORDER BY created_at DESC LIMIT 1)
               ORDER BY depth ASC, order_index ASC, created_at ASC`,
              true,
            );

            const flatTasks: PlanTask[] = tasksResult.success && 'rows' in tasksResult
              ? (tasksResult.rows as Record<string, unknown>[]).map(r => ({ ...r, children: [] } as unknown as PlanTask))
              : [];

            // Build tree client-side via parent_id links
            const taskMap = new Map<string, PlanTask>();
            flatTasks.forEach(t => taskMap.set(t.id, t));
            const roots: PlanTask[] = [];
            flatTasks.forEach(t => {
              if (t.parent_id && taskMap.has(t.parent_id)) {
                taskMap.get(t.parent_id)!.children.push(t);
              } else {
                roots.push(t);
              }
            });

            // Compute progress counts
            const progress = {
              total: flatTasks.length,
              completed: flatTasks.filter(t => t.status === 'completed').length,
              in_progress: flatTasks.filter(t => t.status === 'in_progress').length,
              pending: flatTasks.filter(t => t.status === 'pending').length,
              failed: flatTasks.filter(t => t.status === 'failed').length,
              blocked: flatTasks.filter(t => t.status === 'blocked').length,
            };

            data.activePlan = {
              id: plan.id,
              title: plan.title,
              status: plan.status,
              project_dir: plan.project_dir || null,
              target_dir: plan.target_dir || null,
              created_at: plan.created_at,
              completed_at: plan.completed_at || null,
              tasks: roots,
              progress,
            };
          }
        } catch { /* ignore */ }

        // Plan history: all plans with stats (no full task trees needed)
        try {
          const historyResult = await api().executeSQL(
            memoryDb.path,
            `SELECT p.id, p.title, p.status, p.project_dir, p.created_at, p.completed_at,
                    COUNT(pt.id) as task_count,
                    SUM(CASE WHEN pt.status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
                    SUM(CASE WHEN pt.status = 'failed' THEN 1 ELSE 0 END) as failed_tasks
             FROM plans p
             LEFT JOIN plan_tasks pt ON p.id = pt.plan_id
             GROUP BY p.id
             ORDER BY p.created_at DESC, p.rowid DESC
             LIMIT 20`,
            true,
          );
          if (historyResult.success && 'rows' in historyResult) {
            data.planHistory = historyResult.rows as PlanHistoryEntry[];
          }
        } catch { /* ignore */ }
      }

      // Query knowledge.db
      const knowledgeDb = existingDbs.find((d) => d.name === 'knowledge.db');
      if (knowledgeDb) {
        try {
          const insightResult = await api().executeSQL(
            knowledgeDb.path,
            `SELECT id, content, category, confidence, created_at FROM insights ORDER BY created_at DESC LIMIT 8`,
            true,
          );
          if (insightResult.success && 'rows' in insightResult) {
            data.recentInsights = insightResult.rows as DashboardData['recentInsights'];
          }
        } catch { /* ignore */ }
      }

      // Query tools.db
      const toolsDb = existingDbs.find((d) => d.name === 'tools.db');
      if (toolsDb) {
        try {
          const toolResult = await api().executeSQL(
            toolsDb.path,
            `SELECT t.name, COUNT(tu.id) as usage_count, SUM(CASE WHEN tu.success THEN 1 ELSE 0 END) as success_count, AVG(tu.duration_ms) as avg_duration_ms FROM tools t LEFT JOIN tool_usage tu ON t.id = tu.tool_id GROUP BY t.id ORDER BY usage_count DESC, t.name ASC LIMIT 10`,
            true,
          );
          if (toolResult.success && 'rows' in toolResult) {
            data.topTools = toolResult.rows as DashboardData['topTools'];
          }
        } catch { /* ignore */ }

        // Agent-created tools (source='learned')
        try {
          const learnedCountResult = await api().executeSQL(
            toolsDb.path,
            `SELECT COUNT(*) as cnt FROM tools WHERE source = 'learned'`,
            true,
          );
          if (learnedCountResult.success && 'rows' in learnedCountResult && learnedCountResult.rows.length > 0) {
            data.learnedToolsCount = (learnedCountResult.rows[0] as Record<string, number>).cnt || 0;
          }
        } catch { /* ignore */ }

        try {
          const learnedResult = await api().executeSQL(
            toolsDb.path,
            `SELECT name, category, description, created_at, use_count, last_used, code_path FROM tools WHERE source = 'learned' ORDER BY created_at DESC LIMIT 20`,
            true,
          );
          if (learnedResult.success && 'rows' in learnedResult) {
            data.learnedTools = learnedResult.rows as DashboardData['learnedTools'];
          }
        } catch { /* ignore */ }
      }

      // Query agents.db
      const agentsDb = existingDbs.find((d) => d.name === 'agents.db');
      if (agentsDb) {
        // Compact summary for ResourceUsage widget (topAgents)
        try {
          const agentResult = await api().executeSQL(
            agentsDb.path,
            `SELECT name, description, use_count as usage_count, last_used FROM agents ORDER BY use_count DESC LIMIT 10`,
            true,
          );
          if (agentResult.success && 'rows' in agentResult) {
            data.topAgents = agentResult.rows as DashboardData['topAgents'];
          }
        } catch { /* ignore */ }

        // Full specialist registry for Agent Dispatch panel
        try {
          const specResult = await api().executeSQL(
            agentsDb.path,
            `SELECT name, description, confidence, use_count, success_count, failure_count, created_at, last_used FROM agents ORDER BY use_count DESC`,
            true,
          );
          if (specResult.success && 'rows' in specResult) {
            data.agentSpecialists = specResult.rows as DashboardData['agentSpecialists'];
          }
        } catch { /* ignore */ }

        // Recent sub-agent invocations for Agent Dispatch panel
        try {
          const callResult = await api().executeSQL(
            agentsDb.path,
            `SELECT a.name, au.timestamp, au.success, au.task_type, au.duration_ms FROM agent_usage au JOIN agents a ON au.agent_id = a.id ORDER BY au.timestamp DESC LIMIT 20`,
            true,
          );
          if (callResult.success && 'rows' in callResult) {
            data.recentAgentCalls = callResult.rows as DashboardData['recentAgentCalls'];
          }
        } catch { /* ignore */ }
      }

      // Query skills.db
      const skillsDb = existingDbs.find((d) => d.name === 'skills.db');
      if (skillsDb) {
        try {
          const skillResult = await api().executeSQL(
            skillsDb.path,
            `SELECT name, description, category, success_count, failure_count FROM skills ORDER BY (success_count + failure_count) DESC LIMIT 10`,
            true,
          );
          if (skillResult.success && 'rows' in skillResult) {
            data.topSkills = skillResult.rows as DashboardData['topSkills'];
          }
        } catch { /* ignore */ }
      }

      // Query memory.db - tool_results, active_state (and plan data via plan_tasks)
      if (memoryDb) {
        try {
          const memoryResult = await api().executeSQL(
            memoryDb.path,
            `SELECT tool_name, COUNT(*) as call_count FROM tool_results GROUP BY tool_name ORDER BY call_count DESC LIMIT 10`,
            true,
          );
          if (memoryResult.success && 'rows' in memoryResult) {
            data.topMemoryTools = memoryResult.rows as DashboardData['topMemoryTools'];
          }
        } catch { /* ignore */ }

        // Working memory (active_state) — facts injected into every LLM prompt
        try {
          const activeStateResult = await api().executeSQL(
            memoryDb.path,
            `SELECT key, value, tags, stored_at, last_accessed FROM active_state ORDER BY stored_at DESC`,
            true,
          );
          if (activeStateResult.success && 'rows' in activeStateResult) {
            data.workingMemory = activeStateResult.rows as WorkingMemoryEntry[];
          }
        } catch { /* ignore */ }
      }

      // Query knowledge.db - top insights by use_count
      if (knowledgeDb) {
        try {
          const knowledgeResult = await api().executeSQL(
            knowledgeDb.path,
            `SELECT id, content, category, confidence, use_count, last_used FROM insights ORDER BY use_count DESC LIMIT 10`,
            true,
          );
          if (knowledgeResult.success && 'rows' in knowledgeResult) {
            data.topKnowledge = knowledgeResult.rows as DashboardData['topKnowledge'];
          }
        } catch { /* ignore */ }
      }

      return data;
    },
    enabled: !!workspacePath && !!databases && databases.length > 0,
    refetchInterval: refreshInterval > 0 ? refreshInterval : false,
    placeholderData: (prev) => prev,
  });
}

async function computeTrendStats(logsDbPath: string): Promise<TrendStats> {
  const stats: TrendStats = {
    totalLogs24h: 0,
    totalLogsPrev24h: 0,
    errors24h: 0,
    errorsPrev24h: 0,
    avgContext24h: 0,
    avgContextPrev24h: 0,
    totalToolCalls: 0,
  };

  const queries = [
    { sql: `SELECT COUNT(*) as cnt FROM runtime_logs WHERE timestamp >= DATETIME('now', '-1 day')`, key: 'totalLogs24h', field: 'cnt' },
    { sql: `SELECT COUNT(*) as cnt FROM runtime_logs WHERE timestamp >= DATETIME('now', '-2 days') AND timestamp < DATETIME('now', '-1 day')`, key: 'totalLogsPrev24h', field: 'cnt' },
    { sql: `SELECT COUNT(*) as cnt FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL') AND timestamp >= DATETIME('now', '-1 day')`, key: 'errors24h', field: 'cnt' },
    { sql: `SELECT COUNT(*) as cnt FROM runtime_logs WHERE level IN ('ERROR', 'CRITICAL') AND timestamp >= DATETIME('now', '-2 days') AND timestamp < DATETIME('now', '-1 day')`, key: 'errorsPrev24h', field: 'cnt' },
    { sql: `SELECT AVG(context_tokens) as avg_ctx FROM runtime_logs WHERE context_tokens IS NOT NULL AND timestamp >= DATETIME('now', '-1 day')`, key: 'avgContext24h', field: 'avg_ctx' },
    { sql: `SELECT AVG(context_tokens) as avg_ctx FROM runtime_logs WHERE context_tokens IS NOT NULL AND timestamp >= DATETIME('now', '-2 days') AND timestamp < DATETIME('now', '-1 day')`, key: 'avgContextPrev24h', field: 'avg_ctx' },
  ];

  for (const q of queries) {
    try {
      const r = await api().executeSQL(logsDbPath, q.sql, true);
      if (r.success && 'rows' in r && r.rows.length > 0) {
        (stats as Record<string, number>)[q.key] = (r.rows[0] as Record<string, number>)[q.field] || 0;
      }
    } catch { /* ignore */ }
  }

  return stats;
}
