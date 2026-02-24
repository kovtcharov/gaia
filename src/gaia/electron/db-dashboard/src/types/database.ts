// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * TypeScript interfaces for the GAIA DB Dashboard.
 * Mirrors the IPC response shapes from main.js / dev-server.js.
 */

// ============================================================================
// Database Discovery
// ============================================================================

export interface DatabaseInfo {
  name: string;
  label: string;
  description?: string;
  path: string;
  exists: boolean;
  sizeBytes: number;
  lastModified: string | null;
}

export interface ListDatabasesResponse {
  success: boolean;
  error?: string;
  databases: DatabaseInfo[];
  workspacePath?: string;
}

// ============================================================================
// Table Operations
// ============================================================================

export interface ColumnInfo {
  name: string;
  type: string;
  notNull: boolean;
  defaultValue: string | null;
  primaryKey: boolean;
}

export interface TableInfo {
  name: string;
  type: 'table' | 'view';
  isFts: boolean;
  rowCount: number;
  columns: ColumnInfo[];
}

export interface GetTablesResponse {
  success: boolean;
  error?: string;
  tables: TableInfo[];
}

// ============================================================================
// Query & Pagination
// ============================================================================

export interface QueryOptions {
  page?: number;
  limit?: number;
  sortColumn?: string | null;
  sortDirection?: 'ASC' | 'DESC';
  filterColumn?: string | null;
  filterValue?: string | null;
}

export interface PaginationInfo {
  page: number;
  limit: number;
  totalRows: number;
  totalPages: number;
}

export type Row = Record<string, unknown>;

export interface QueryTableResponse {
  success: boolean;
  error?: string;
  rows: Row[];
  pagination: PaginationInfo;
}

// ============================================================================
// SQL Execution
// ============================================================================

export interface SQLQueryResult {
  success: boolean;
  error?: string;
  type: 'query';
  rows: Row[];
  columns: string[];
  rowCount: number;
  duration: number;
}

export interface SQLStatementResult {
  success: boolean;
  error?: string;
  type: 'statement';
  changes: number;
  lastInsertRowid: number;
  duration: number;
}

export type ExecuteSQLResponse = SQLQueryResult | SQLStatementResult | { success: false; error: string };

// ============================================================================
// Row CRUD
// ============================================================================

export interface PrimaryKey {
  column: string;
  value: unknown;
}

export interface MutationResponse {
  success: boolean;
  error?: string;
  changes?: number;
  lastInsertRowid?: number;
}

// ============================================================================
// Schema
// ============================================================================

export interface SchemaEntry {
  name: string;
  sql: string;
}

export interface GetSchemaResponse {
  success: boolean;
  error?: string;
  schemas: SchemaEntry[];
}

// ============================================================================
// FTS5 Search
// ============================================================================

export interface FTS5SearchResponse {
  success: boolean;
  error?: string;
  rows: Row[];
  rowCount: number;
}

// ============================================================================
// Export & Backup
// ============================================================================

export interface ExportResponse {
  success: boolean;
  error?: string;
  filePath?: string;
  rowCount?: number;
}

// ============================================================================
// Database API interface (exposed via preload.js or browser mock)
// ============================================================================

export interface DatabaseAPI {
  listDatabases(workspacePath?: string): Promise<ListDatabasesResponse>;
  getWorkspacePath(): Promise<string>;
  selectWorkspace(): Promise<{ success: boolean; path?: string }>;
  selectDatabaseFile(): Promise<{ success: boolean; path?: string }>;
  getTables(dbPath: string): Promise<GetTablesResponse>;
  queryTable(dbPath: string, tableName: string, options?: QueryOptions): Promise<QueryTableResponse>;
  executeSQL(dbPath: string, sql: string, readOnly?: boolean): Promise<ExecuteSQLResponse>;
  updateRow(dbPath: string, tableName: string, primaryKey: PrimaryKey, updates: Record<string, unknown>): Promise<MutationResponse>;
  deleteRow(dbPath: string, tableName: string, primaryKey: PrimaryKey): Promise<MutationResponse>;
  clearTable(dbPath: string, tableName: string): Promise<MutationResponse>;
  insertRow(dbPath: string, tableName: string, data: Record<string, unknown>): Promise<MutationResponse>;
  getSchema(dbPath: string): Promise<GetSchemaResponse>;
  searchFTS5(dbPath: string, ftsTable: string, query: string, options?: { limit?: number }): Promise<FTS5SearchResponse>;
  exportTable(dbPath: string, tableName: string, format: 'json' | 'csv'): Promise<ExportResponse>;
  backup(dbPath: string): Promise<ExportResponse>;
  closeConnection(dbPath: string): Promise<{ success: boolean }>;
  onMenuEvent(channel: string, callback: (...args: unknown[]) => void): void;
}

// ============================================================================
// Dashboard Data
// ============================================================================

export interface DbStats {
  name: string;
  label: string;
  sizeBytes: number;
  lastModified: string | null;
  tableCount: number;
  totalRows: number;
}

export interface TrendStats {
  totalLogs24h: number;
  totalLogsPrev24h: number;
  errors24h: number;
  errorsPrev24h: number;
  avgContext24h: number;
  avgContextPrev24h: number;
  totalToolCalls: number;
}

export interface ErrorEntry {
  level: string;
  message: string;
  timestamp: string;
  step_number: number | null;
}

export interface TaskEntry {
  id: number;
  description: string;
  status: string;
  priority: number | null;
}

export interface InsightEntry {
  id: number;
  content: string;
  category: string | null;
  confidence: number | null;
  created_at: string;
}

export interface ToolEntry {
  name: string;
  usage_count: number;
  success_count: number | null;
  avg_duration_ms: number | null;
}

export interface HourlyActivityEntry {
  hour_label: string; // 'HH:00' e.g. '14:00'
  count: number;
}

export interface MinuteActivityEntry {
  bucket_label: string; // 'HH:MM' e.g. '14:35'
  count: number;
}

/** @deprecated Use HourlyActivityEntry */
export interface HeatmapEntry {
  date: string;
  count: number;
}

export interface ContextUsageEntry {
  step_number: number;
  context_tokens: number;
  max_context_tokens: number;
  timestamp: string;
}

export interface AgentEntry {
  name: string;
  description: string;
  usage_count: number;
  last_used: string | null;
}

export interface SkillEntry {
  name: string;
  description: string;
  category: string;
  success_count: number;
  failure_count: number;
}

export interface MemoryToolEntry {
  tool_name: string;
  call_count: number;
}

export interface KnowledgeInsightEntry {
  id: string;
  content: string;
  category: string | null;
  confidence: number | null;
  use_count: number;
  last_used: string | null;
}

/** Agent-created tool stored in tools.db with source='learned' */
export interface LearnedToolEntry {
  name: string;
  category: string;
  description: string;
  created_at: string;
  use_count: number;
  last_used: string | null;
  code_path: string | null;
}

export interface DashboardData {
  totalSize: number;
  dbCount: number;
  lastActivity: Date | null;
  dbStats: DbStats[];
  recentErrors: ErrorEntry[];
  activeTasks: TaskEntry[];
  recentInsights: InsightEntry[];
  topTools: ToolEntry[];
  activityHeatmap: HourlyActivityEntry[];
  minuteActivity: MinuteActivityEntry[];
  contextUsage: ContextUsageEntry[];
  trendStats: TrendStats | null;
  topAgents: AgentEntry[];
  topSkills: SkillEntry[];
  topMemoryTools: MemoryToolEntry[];
  topKnowledge: KnowledgeInsightEntry[];
  /** Tools created by the agent (source='learned') */
  learnedTools: LearnedToolEntry[];
  learnedToolsCount: number;
}

// ============================================================================
// App State
// ============================================================================

export type ViewTab = 'data' | 'sql' | 'fts';

export interface AppTab {
  id: string;
  type: 'dashboard' | 'database';
  dbPath?: string;
  dbName?: string;
  label: string;
}

// Augment the Window interface
declare global {
  interface Window {
    dbAPI: DatabaseAPI;
  }
}
