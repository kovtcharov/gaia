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

      // Query logs.db
      const logsDb = existingDbs.find((d) => d.name === 'logs.db');
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

      // Query plan.db
      const planDb = existingDbs.find((d) => d.name === 'plan.db');
      if (planDb) {
        try {
          const taskResult = await api().executeSQL(
            planDb.path,
            `SELECT id, description, status, priority FROM tasks ORDER BY CASE status WHEN 'in_progress' THEN 0 WHEN 'pending' THEN 1 WHEN 'completed' THEN 2 WHEN 'failed' THEN 3 ELSE 4 END, id DESC LIMIT 10`,
            true,
          );
          if (taskResult.success && 'rows' in taskResult) {
            data.activeTasks = taskResult.rows as DashboardData['activeTasks'];
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
      }

      // Query agents.db
      const agentsDb = existingDbs.find((d) => d.name === 'agents.db');
      if (agentsDb) {
        try {
          const agentResult = await api().executeSQL(
            agentsDb.path,
            `SELECT a.name, a.description, COUNT(au.id) as usage_count, a.last_used FROM agents a LEFT JOIN agent_usage au ON a.id = au.agent_id GROUP BY a.id ORDER BY usage_count DESC LIMIT 10`,
            true,
          );
          if (agentResult.success && 'rows' in agentResult) {
            data.topAgents = agentResult.rows as DashboardData['topAgents'];
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

      // Query memory.db - tool_results
      const memoryDb = existingDbs.find((d) => d.name === 'memory.db');
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
