// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * GAIA DB Dashboard - Root Application Component
 *
 * Manages top-level state: workspace path, database list, tab management,
 * auto-refresh, and read/write mode. Renders the layout shell with
 * Header, Tabs, Sidebar, and the active content panel (Dashboard or DatabaseView).
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'framer-motion';
import { Database as DbIcon, FolderSearch } from 'lucide-react';

import type {
  AppTab,
  DatabaseInfo,
  QueryOptions,
  ViewTab,
  SchemaEntry,
} from './types/database';

import {
  useDatabases,
  useTables,
  useTableData,
  useDashboardData,
  useSchema,
} from './hooks/useDatabase';
import { useAutoRefresh } from './hooks/useAutoRefresh';

import Header from './components/Layout/Header';
import Tabs from './components/Layout/Tabs';
import Sidebar from './components/Layout/Sidebar';
import Overview from './components/Dashboard/Overview';
import DataGrid from './components/DatabaseView/DataGrid';
import SQLConsole from './components/DatabaseView/SQLConsole';
import FTSSearch from './components/DatabaseView/FTSSearch';
import Modal from './components/shared/Modal';
import Button from './components/shared/Button';

const DASHBOARD_TAB: AppTab = {
  id: 'dashboard',
  type: 'dashboard',
  label: 'Dashboard',
};

export default function App() {
  // ---- Workspace ----
  const [workspacePath, setWorkspacePath] = useState<string | null>(null);

  // ---- Tabs ----
  const [tabs, setTabs] = useState<AppTab[]>([DASHBOARD_TAB]);
  const [activeTabId, setActiveTabId] = useState('dashboard');

  // ---- Database View State ----
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [viewTab, setViewTab] = useState<ViewTab>('data');
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    page: 1,
    limit: 50,
    sortColumn: null,
    sortDirection: 'ASC',
    filterColumn: null,
    filterValue: null,
  });

  // ---- Mode ----
  const [readOnly, setReadOnly] = useState(true);
  const [schemaModalOpen, setSchemaModalOpen] = useState(false);

  // ---- Auto-Refresh ----
  const autoRefresh = useAutoRefresh(5000);
  const queryClient = useQueryClient();

  // ---- Derived: active tab's database path ----
  const activeTab = useMemo(
    () => tabs.find((t) => t.id === activeTabId) || DASHBOARD_TAB,
    [tabs, activeTabId],
  );
  const currentDbPath = activeTab.type === 'database' ? activeTab.dbPath! : null;

  // ---- Data Hooks ----
  const { data: databases, isLoading: dbsLoading } = useDatabases(workspacePath);
  const { data: tables, isLoading: tablesLoading } = useTables(currentDbPath);
  const { data: tableData } = useTableData(
    currentDbPath,
    selectedTable,
    queryOptions,
    autoRefresh.interval,
  );
  const { data: dashboardData } = useDashboardData(
    workspacePath,
    databases,
    autoRefresh.interval,
  );
  const { data: schemaEntries } = useSchema(schemaModalOpen ? currentDbPath : null);

  // ---- Initialize workspace path ----
  useEffect(() => {
    if (window.dbAPI) {
      window.dbAPI.getWorkspacePath().then((path: string) => {
        setWorkspacePath(path);
      });
    }
  }, []);

  // ---- Menu event listeners (Electron) ----
  useEffect(() => {
    if (window.dbAPI?.onMenuEvent) {
      window.dbAPI.onMenuEvent('menu:openWorkspace', () => handleChangeWorkspace());
      window.dbAPI.onMenuEvent('menu:refresh', () => handleRefresh());
    }
  }, []);

  // ---- Mark auto-refresh update time ----
  useEffect(() => {
    if (dashboardData || tableData) {
      autoRefresh.markUpdated();
    }
  }, [dashboardData, tableData]);

  // ---- Handlers ----

  const handleChangeWorkspace = useCallback(async () => {
    const result = await window.dbAPI.selectWorkspace();
    if (result.success && result.path) {
      setWorkspacePath(result.path);
      // Reset to dashboard
      setTabs([DASHBOARD_TAB]);
      setActiveTabId('dashboard');
      setSelectedTable(null);
    }
  }, []);

  const handleOpenFile = useCallback(async () => {
    const result = await window.dbAPI.selectDatabaseFile();
    if (result.success && result.path) {
      const name = result.path.split(/[/\\]/).pop() || 'database';
      openDatabaseTab(result.path, name);
    }
  }, []);

  const handleRefresh = useCallback(() => {
    queryClient.invalidateQueries();
    autoRefresh.markUpdated();
  }, [queryClient]);

  const openDatabaseTab = useCallback(
    (dbPath: string, dbName: string) => {
      const existing = tabs.find((t) => t.dbPath === dbPath);
      if (existing) {
        setActiveTabId(existing.id);
      } else {
        const newTab: AppTab = {
          id: `db-${Date.now()}`,
          type: 'database',
          dbPath,
          dbName,
          label: dbName.replace('.db', ''),
        };
        setTabs((prev) => [...prev, newTab]);
        setActiveTabId(newTab.id);
      }
      setSelectedTable(null);
      setViewTab('data');
      setQueryOptions({ page: 1, limit: 50, sortColumn: null, sortDirection: 'ASC', filterColumn: null, filterValue: null });
    },
    [tabs],
  );

  const handleCloseTab = useCallback(
    (tabId: string) => {
      if (tabId === 'dashboard') return;
      setTabs((prev) => {
        const updated = prev.filter((t) => t.id !== tabId);
        if (activeTabId === tabId) {
          setActiveTabId(updated[updated.length - 1]?.id || 'dashboard');
        }
        return updated;
      });
      setSelectedTable(null);
    },
    [activeTabId],
  );

  const handleSelectDatabase = useCallback(
    (db: DatabaseInfo) => {
      openDatabaseTab(db.path, db.name);
    },
    [openDatabaseTab],
  );

  const handleSelectTable = useCallback((name: string) => {
    setSelectedTable(name);
    setViewTab('data');
    setQueryOptions((prev) => ({ ...prev, page: 1, sortColumn: null, filterColumn: null, filterValue: null }));
  }, []);

  const handleExport = useCallback(
    async (format: 'json' | 'csv') => {
      if (!currentDbPath || !selectedTable) return;
      await window.dbAPI.exportTable(currentDbPath, selectedTable, format);
    },
    [currentDbPath, selectedTable],
  );

  const handleBackup = useCallback(async () => {
    if (!currentDbPath) return;
    await window.dbAPI.backup(currentDbPath);
  }, [currentDbPath]);

  // ---- Render ----

  const isDashboard = activeTab.type === 'dashboard';

  return (
    <div className="flex flex-col h-screen bg-gh-bg text-gh-fg-default overflow-hidden">
      <Header
        workspacePath={workspacePath || ''}
        readOnly={readOnly}
        onToggleReadOnly={() => setReadOnly((r) => !r)}
        onRefresh={handleRefresh}
        onChangeWorkspace={handleChangeWorkspace}
        onOpenFile={handleOpenFile}
        autoRefreshEnabled={autoRefresh.enabled}
        onToggleAutoRefresh={() => autoRefresh.setEnabled((e) => !e)}
        refreshInterval={autoRefresh.interval}
        onChangeInterval={autoRefresh.setInterval}
        intervalOptions={autoRefresh.intervalOptions}
        lastUpdateText={autoRefresh.relativeTime}
      />

      <Tabs
        tabs={tabs}
        activeTabId={activeTabId}
        onSelectTab={setActiveTabId}
        onCloseTab={handleCloseTab}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - only visible in database view */}
        {!isDashboard && (
          <Sidebar
            tables={tables || []}
            selectedTable={selectedTable}
            onSelectTable={handleSelectTable}
            databases={databases || []}
            currentDbPath={currentDbPath}
            onSelectDatabase={handleSelectDatabase}
            onViewSchema={() => setSchemaModalOpen(true)}
            onBackup={handleBackup}
            isLoading={tablesLoading}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <AnimatePresence mode="wait">
            {isDashboard ? (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="flex-1 overflow-hidden"
              >
                {!workspacePath || dbsLoading ? (
                  <EmptyState
                    title="Loading workspace..."
                    subtitle="Detecting GAIA workspace path"
                    loading
                  />
                ) : !dashboardData ? (
                  <EmptyState
                    title="No databases found"
                    subtitle="Run a GAIA agent to create databases, or change the workspace directory."
                    action={
                      <Button variant="primary" onClick={handleChangeWorkspace}>
                        Change Workspace
                      </Button>
                    }
                  />
                ) : (
                  <Overview data={dashboardData} workspacePath={workspacePath} />
                )}
              </motion.div>
            ) : (
              <motion.div
                key={activeTabId}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="flex-1 flex flex-col overflow-hidden"
              >
                {!selectedTable ? (
                  <EmptyState
                    title="Select a table"
                    subtitle="Choose a table from the sidebar to view its data."
                  />
                ) : (
                  <>
                    {/* View tabs: Data | SQL | FTS */}
                    <DatabaseViewTabs
                      viewTab={viewTab}
                      onChangeViewTab={setViewTab}
                      tableName={selectedTable}
                      isFts={tables?.find((t) => t.name === selectedTable)?.isFts || false}
                      onExportJSON={() => handleExport('json')}
                      onExportCSV={() => handleExport('csv')}
                    />

                    <div className="flex-1 overflow-hidden">
                      {viewTab === 'data' && (
                        <DataGrid
                          rows={tableData?.rows || []}
                          pagination={tableData?.pagination || { page: 1, limit: 50, totalRows: 0, totalPages: 0 }}
                          columns={tables?.find((t) => t.name === selectedTable)?.columns || []}
                          queryOptions={queryOptions}
                          onChangeOptions={setQueryOptions}
                          readOnly={readOnly}
                          dbPath={currentDbPath!}
                          tableName={selectedTable}
                        />
                      )}
                      {viewTab === 'sql' && currentDbPath && (
                        <SQLConsole dbPath={currentDbPath} readOnly={readOnly} />
                      )}
                      {viewTab === 'fts' && currentDbPath && (
                        <FTSSearch
                          dbPath={currentDbPath}
                          ftsTables={(tables || []).filter((t) => t.isFts)}
                        />
                      )}
                    </div>
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* Schema Modal */}
      <Modal
        open={schemaModalOpen}
        onClose={() => setSchemaModalOpen(false)}
        title="Database Schema"
        size="lg"
      >
        <div className="space-y-3">
          {(schemaEntries || []).map((entry: SchemaEntry) => (
            <div key={entry.name} className="rounded-md bg-gh-canvas border border-gh-border-muted p-3">
              <div className="text-xs font-semibold text-gh-accent-fg mb-1">{entry.name}</div>
              <pre className="text-2xs text-gh-fg-muted font-mono whitespace-pre-wrap break-all">
                {entry.sql}
              </pre>
            </div>
          ))}
          {(!schemaEntries || schemaEntries.length === 0) && (
            <div className="py-8 text-center text-xs text-gh-fg-subtle">
              No schema information available.
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}

// ============================================================================
// Sub-components
// ============================================================================

function EmptyState({
  title,
  subtitle,
  action,
  loading,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  loading?: boolean;
}) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
      <div className="w-12 h-12 rounded-full bg-gh-canvas-subtle border border-gh-border flex items-center justify-center">
        {loading ? (
          <DbIcon size={20} className="text-gh-fg-subtle animate-pulse" />
        ) : (
          <FolderSearch size={20} className="text-gh-fg-subtle" />
        )}
      </div>
      <div className="text-center">
        <h3 className="text-sm font-semibold text-gh-fg-default">{title}</h3>
        {subtitle && <p className="text-xs text-gh-fg-muted mt-1 max-w-sm">{subtitle}</p>}
      </div>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

function DatabaseViewTabs({
  viewTab,
  onChangeViewTab,
  tableName,
  isFts,
  onExportJSON,
  onExportCSV,
}: {
  viewTab: ViewTab;
  onChangeViewTab: (tab: ViewTab) => void;
  tableName: string;
  isFts: boolean;
  onExportJSON: () => void;
  onExportCSV: () => void;
}) {
  return (
    <div className="flex items-center justify-between h-10 px-4 border-b border-gh-border bg-gh-canvas-subtle shrink-0">
      <div className="flex items-center gap-1">
        <span className="text-xs font-semibold text-gh-fg-default mr-3">{tableName}</span>
        <ViewTabButton label="Data" value="data" current={viewTab} onChange={onChangeViewTab} />
        <ViewTabButton label="SQL" value="sql" current={viewTab} onChange={onChangeViewTab} />
        {isFts && (
          <ViewTabButton label="FTS Search" value="fts" current={viewTab} onChange={onChangeViewTab} />
        )}
      </div>
      <div className="flex items-center gap-1">
        <Button size="sm" variant="ghost" onClick={onExportJSON}>
          Export JSON
        </Button>
        <Button size="sm" variant="ghost" onClick={onExportCSV}>
          Export CSV
        </Button>
      </div>
    </div>
  );
}

function ViewTabButton({
  label,
  value,
  current,
  onChange,
}: {
  label: string;
  value: ViewTab;
  current: ViewTab;
  onChange: (v: ViewTab) => void;
}) {
  const active = value === current;
  return (
    <button
      onClick={() => onChange(value)}
      className={`
        relative px-3 py-1.5 text-xs font-medium rounded-md transition-colors
        ${active ? 'text-gh-fg-default bg-gh-border-muted' : 'text-gh-fg-muted hover:text-gh-fg-default'}
      `}
    >
      {label}
    </button>
  );
}
