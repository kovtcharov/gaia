// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * DataGrid - Virtualized data table with sorting, filtering, pagination,
 * and inline editing support.
 *
 * Uses react-window for virtualized scrolling to handle 100K+ rows without lag.
 * Framer Motion provides smooth highlight animations for new/changed rows.
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { FixedSizeList as List } from 'react-window';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Filter,
  X,
  Pencil,
  Trash2,
  Save,
  RotateCcw,
  Plus,
  Copy,
  Check,
} from 'lucide-react';

import type { Row, ColumnInfo, PaginationInfo, QueryOptions } from '../../types/database';
import { useDeleteRow, useUpdateRow, useInsertRow } from '../../hooks/useDatabase';
import Button from '../shared/Button';
import Modal from '../shared/Modal';

interface DataGridProps {
  rows: Row[];
  pagination: PaginationInfo;
  columns: ColumnInfo[];
  queryOptions: QueryOptions;
  onChangeOptions: React.Dispatch<React.SetStateAction<QueryOptions>>;
  readOnly: boolean;
  dbPath: string;
  tableName: string;
}

const ROW_HEIGHT = 36;
const HEADER_HEIGHT = 36;

export default function DataGrid({
  rows,
  pagination,
  columns,
  queryOptions,
  onChangeOptions,
  readOnly,
  dbPath,
  tableName,
}: DataGridProps) {
  // ---- State ----
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterCol, setFilterCol] = useState('');
  const [filterVal, setFilterVal] = useState('');
  const [editingRow, setEditingRow] = useState<number | null>(null);
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [insertModalOpen, setInsertModalOpen] = useState(false);
  const [insertValues, setInsertValues] = useState<Record<string, string>>({});
  const [copiedCell, setCopiedCell] = useState<string | null>(null);

  const listRef = useRef<List>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState(500);

  // ---- Mutations ----
  const deleteRow = useDeleteRow();
  const updateRow = useUpdateRow();
  const insertRow = useInsertRow();

  // ---- Resize observer ----
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerHeight(entry.contentRect.height - HEADER_HEIGHT - 48); // header + pagination
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // ---- Column names ----
  const colNames = useMemo(() => {
    if (columns.length > 0) return columns.map((c) => c.name);
    if (rows.length > 0) return Object.keys(rows[0]);
    return [];
  }, [columns, rows]);

  // ---- Primary key detection ----
  const pkColumn = useMemo(() => {
    const pk = columns.find((c) => c.primaryKey);
    return pk?.name || colNames[0] || null;
  }, [columns, colNames]);

  // ---- Handlers ----

  const handleSort = useCallback(
    (col: string) => {
      onChangeOptions((prev) => ({
        ...prev,
        sortColumn: col,
        sortDirection: prev.sortColumn === col && prev.sortDirection === 'ASC' ? 'DESC' : 'ASC',
        page: 1,
      }));
    },
    [onChangeOptions],
  );

  const handleApplyFilter = useCallback(() => {
    onChangeOptions((prev) => ({
      ...prev,
      filterColumn: filterCol || null,
      filterValue: filterVal || null,
      page: 1,
    }));
  }, [filterCol, filterVal, onChangeOptions]);

  const handleClearFilter = useCallback(() => {
    setFilterCol('');
    setFilterVal('');
    onChangeOptions((prev) => ({
      ...prev,
      filterColumn: null,
      filterValue: null,
      page: 1,
    }));
  }, [onChangeOptions]);

  const handlePageChange = useCallback(
    (page: number) => {
      onChangeOptions((prev) => ({ ...prev, page }));
      listRef.current?.scrollTo(0);
    },
    [onChangeOptions],
  );

  const handleStartEdit = useCallback(
    (rowIndex: number) => {
      if (readOnly) return;
      const row = rows[rowIndex];
      const values: Record<string, string> = {};
      colNames.forEach((col) => {
        values[col] = row[col] != null ? String(row[col]) : '';
      });
      setEditingRow(rowIndex);
      setEditValues(values);
    },
    [rows, colNames, readOnly],
  );

  const handleSaveEdit = useCallback(async () => {
    if (editingRow === null || !pkColumn) return;
    const row = rows[editingRow];
    const updates: Record<string, unknown> = {};
    colNames.forEach((col) => {
      if (col !== pkColumn && editValues[col] !== String(row[col] ?? '')) {
        updates[col] = editValues[col];
      }
    });
    if (Object.keys(updates).length > 0) {
      await updateRow.mutateAsync({
        dbPath,
        tableName,
        primaryKey: { column: pkColumn, value: row[pkColumn] },
        updates,
      });
    }
    setEditingRow(null);
  }, [editingRow, editValues, rows, colNames, pkColumn, dbPath, tableName, updateRow]);

  const handleDeleteRow = useCallback(
    async (rowIndex: number) => {
      if (readOnly || !pkColumn) return;
      const row = rows[rowIndex];
      if (!confirm(`Delete row where ${pkColumn} = ${row[pkColumn]}?`)) return;
      await deleteRow.mutateAsync({
        dbPath,
        tableName,
        primaryKey: { column: pkColumn, value: row[pkColumn] },
      });
    },
    [rows, pkColumn, readOnly, dbPath, tableName, deleteRow],
  );

  const handleInsert = useCallback(async () => {
    const data: Record<string, unknown> = {};
    colNames.forEach((col) => {
      if (insertValues[col]?.trim()) {
        data[col] = insertValues[col];
      }
    });
    await insertRow.mutateAsync({ dbPath, tableName, data });
    setInsertModalOpen(false);
    setInsertValues({});
  }, [insertValues, colNames, dbPath, tableName, insertRow]);

  const handleCopyCell = useCallback((value: string, key: string) => {
    navigator.clipboard?.writeText(value);
    setCopiedCell(key);
    setTimeout(() => setCopiedCell(null), 1500);
  }, []);

  // ---- Cell renderer ----
  const formatCell = (value: unknown): string => {
    if (value === null || value === undefined) return 'NULL';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  // ---- Virtual row renderer ----
  const RowRenderer = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const row = rows[index];
      if (!row) return null;

      const isEditing = editingRow === index;

      return (
        <motion.div
          style={style}
          initial={{ backgroundColor: 'rgba(0,0,0,0)' }}
          animate={{ backgroundColor: 'rgba(0,0,0,0)' }}
          className={`flex items-center border-b border-gh-border-muted text-sm hover:bg-gh-canvas-subtle/50 transition-colors group ${
            isEditing ? 'bg-gh-accent-emphasis/5' : ''
          }`}
        >
          {/* Row actions */}
          {!readOnly && (
            <div className="w-16 shrink-0 flex items-center justify-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              {isEditing ? (
                <>
                  <button
                    onClick={handleSaveEdit}
                    className="p-1 rounded hover:bg-gh-success-emphasis/20 text-gh-success-fg"
                    title="Save"
                  >
                    <Save size={12} />
                  </button>
                  <button
                    onClick={() => setEditingRow(null)}
                    className="p-1 rounded hover:bg-gh-border-muted text-gh-fg-muted"
                    title="Cancel"
                  >
                    <RotateCcw size={12} />
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => handleStartEdit(index)}
                    className="p-1 rounded hover:bg-gh-accent-emphasis/20 text-gh-accent-fg"
                    title="Edit"
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    onClick={() => handleDeleteRow(index)}
                    className="p-1 rounded hover:bg-gh-danger-emphasis/20 text-gh-danger-fg"
                    title="Delete"
                  >
                    <Trash2 size={12} />
                  </button>
                </>
              )}
            </div>
          )}

          {colNames.map((col) => {
            const cellKey = `${index}-${col}`;
            const rawValue = row[col];
            const displayValue = formatCell(rawValue);
            const isNull = rawValue === null || rawValue === undefined;

            return (
              <div
                key={col}
                className="flex-1 min-w-[120px] max-w-[300px] px-3 py-1 truncate relative group/cell"
                title={displayValue}
              >
                {isEditing ? (
                  <input
                    value={editValues[col] || ''}
                    onChange={(e) =>
                      setEditValues((prev) => ({ ...prev, [col]: e.target.value }))
                    }
                    className="w-full bg-gh-canvas border border-gh-border rounded px-2 py-0.5 text-xs text-gh-fg-default focus:outline-none focus:ring-1 focus:ring-gh-accent-emphasis"
                    autoFocus={col === colNames[0]}
                  />
                ) : (
                  <span
                    className={`text-xs ${isNull ? 'italic text-gh-fg-subtle' : 'text-gh-fg-default'} cursor-default`}
                    onDoubleClick={() => {
                      if (!readOnly) handleStartEdit(index);
                    }}
                  >
                    {displayValue}
                  </span>
                )}

                {/* Copy button */}
                {!isEditing && (
                  <button
                    onClick={() => handleCopyCell(displayValue, cellKey)}
                    className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover/cell:opacity-100 p-0.5 rounded bg-gh-canvas-subtle hover:bg-gh-border-muted transition-opacity"
                    title="Copy"
                  >
                    {copiedCell === cellKey ? (
                      <Check size={10} className="text-gh-success-fg" />
                    ) : (
                      <Copy size={10} className="text-gh-fg-subtle" />
                    )}
                  </button>
                )}
              </div>
            );
          })}
        </motion.div>
      );
    },
    [
      rows,
      colNames,
      editingRow,
      editValues,
      readOnly,
      copiedCell,
      handleSaveEdit,
      handleStartEdit,
      handleDeleteRow,
      handleCopyCell,
    ],
  );

  return (
    <div ref={containerRef} className="flex flex-col h-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gh-border-muted shrink-0">
        <div className="flex items-center gap-2">
          {/* Filter */}
          <Button
            size="sm"
            variant={queryOptions.filterColumn ? 'primary' : 'ghost'}
            icon={<Filter size={12} />}
            onClick={() => setFilterOpen(!filterOpen)}
          >
            Filter
          </Button>
          {queryOptions.filterColumn && (
            <button
              onClick={handleClearFilter}
              className="flex items-center gap-1 px-2 py-0.5 text-2xs rounded-full bg-gh-accent-emphasis/15 text-gh-accent-fg hover:bg-gh-accent-emphasis/25 transition-colors"
            >
              {queryOptions.filterColumn}: {queryOptions.filterValue}
              <X size={10} />
            </button>
          )}

          {/* Insert */}
          {!readOnly && (
            <Button
              size="sm"
              variant="ghost"
              icon={<Plus size={12} />}
              onClick={() => {
                setInsertValues({});
                setInsertModalOpen(true);
              }}
            >
              Insert
            </Button>
          )}
        </div>

        <div className="text-2xs text-gh-fg-subtle">
          {pagination.totalRows.toLocaleString()} rows
        </div>
      </div>

      {/* Filter bar */}
      <AnimatePresence>
        {filterOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden border-b border-gh-border-muted"
          >
            <div className="flex items-center gap-2 px-4 py-2">
              <select
                value={filterCol}
                onChange={(e) => setFilterCol(e.target.value)}
                className="bg-gh-canvas border border-gh-border rounded text-xs text-gh-fg-default px-2 py-1 focus:outline-none focus:ring-1 focus:ring-gh-accent-emphasis"
              >
                <option value="">Column...</option>
                {colNames.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <input
                value={filterVal}
                onChange={(e) => setFilterVal(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleApplyFilter()}
                placeholder="Filter value..."
                className="flex-1 max-w-xs bg-gh-canvas border border-gh-border rounded text-xs text-gh-fg-default px-2 py-1 focus:outline-none focus:ring-1 focus:ring-gh-accent-emphasis placeholder:text-gh-fg-subtle"
              />
              <Button size="sm" variant="primary" onClick={handleApplyFilter}>
                Apply
              </Button>
              <Button size="sm" variant="ghost" onClick={handleClearFilter}>
                Clear
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Table header */}
      <div
        className="flex items-center bg-gh-canvas-subtle border-b border-gh-border shrink-0"
        style={{ height: HEADER_HEIGHT }}
      >
        {!readOnly && <div className="w-16 shrink-0" />}
        {colNames.map((col) => {
          const isSorted = queryOptions.sortColumn === col;
          return (
            <button
              key={col}
              onClick={() => handleSort(col)}
              className="flex-1 min-w-[120px] max-w-[300px] flex items-center gap-1 px-3 text-left text-xs font-semibold text-gh-fg-muted uppercase tracking-wider hover:text-gh-fg-default transition-colors select-none"
            >
              <span className="truncate">{col}</span>
              {isSorted ? (
                queryOptions.sortDirection === 'ASC' ? (
                  <ArrowUp size={11} className="text-gh-accent-fg shrink-0" />
                ) : (
                  <ArrowDown size={11} className="text-gh-accent-fg shrink-0" />
                )
              ) : (
                <ArrowUpDown size={11} className="opacity-0 group-hover:opacity-30 shrink-0" />
              )}
            </button>
          );
        })}
      </div>

      {/* Virtual scrolling body */}
      <div className="flex-1 overflow-hidden">
        {rows.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs text-gh-fg-subtle">
            No data to display
          </div>
        ) : (
          <List
            ref={listRef}
            height={Math.max(containerHeight, 200)}
            itemCount={rows.length}
            itemSize={ROW_HEIGHT}
            width="100%"
            overscanCount={10}
          >
            {RowRenderer}
          </List>
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-gh-border bg-gh-canvas-subtle shrink-0">
        <div className="text-2xs text-gh-fg-subtle">
          Page {pagination.page} of {pagination.totalPages || 1}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => handlePageChange(1)}
            disabled={pagination.page <= 1}
            className="p-1 rounded hover:bg-gh-border-muted text-gh-fg-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronsLeft size={14} />
          </button>
          <button
            onClick={() => handlePageChange(pagination.page - 1)}
            disabled={pagination.page <= 1}
            className="p-1 rounded hover:bg-gh-border-muted text-gh-fg-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft size={14} />
          </button>

          {/* Page numbers */}
          {getPageNumbers(pagination.page, pagination.totalPages).map((p, i) =>
            p === '...' ? (
              <span key={`dots-${i}`} className="px-1 text-2xs text-gh-fg-subtle">
                ...
              </span>
            ) : (
              <button
                key={p}
                onClick={() => handlePageChange(p as number)}
                className={`min-w-[28px] h-7 px-1.5 text-xs rounded transition-colors ${
                  p === pagination.page
                    ? 'bg-gh-accent-emphasis text-white'
                    : 'text-gh-fg-muted hover:bg-gh-border-muted'
                }`}
              >
                {p}
              </button>
            ),
          )}

          <button
            onClick={() => handlePageChange(pagination.page + 1)}
            disabled={pagination.page >= pagination.totalPages}
            className="p-1 rounded hover:bg-gh-border-muted text-gh-fg-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight size={14} />
          </button>
          <button
            onClick={() => handlePageChange(pagination.totalPages)}
            disabled={pagination.page >= pagination.totalPages}
            className="p-1 rounded hover:bg-gh-border-muted text-gh-fg-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronsRight size={14} />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-2xs text-gh-fg-subtle">Rows per page:</span>
          <select
            value={queryOptions.limit || 50}
            onChange={(e) =>
              onChangeOptions((prev) => ({ ...prev, limit: Number(e.target.value), page: 1 }))
            }
            className="bg-gh-canvas border border-gh-border rounded text-xs text-gh-fg-muted px-1.5 py-0.5 focus:outline-none"
          >
            {[25, 50, 100, 250, 500].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Insert Row Modal */}
      <Modal
        open={insertModalOpen}
        onClose={() => setInsertModalOpen(false)}
        title={`Insert Row into ${tableName}`}
        size="md"
        footer={
          <>
            <Button variant="ghost" onClick={() => setInsertModalOpen(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleInsert} disabled={insertRow.isPending}>
              {insertRow.isPending ? 'Inserting...' : 'Insert'}
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          {colNames.map((col) => {
            const colInfo = columns.find((c) => c.name === col);
            return (
              <div key={col}>
                <label className="block text-xs text-gh-fg-muted mb-1">
                  {col}
                  {colInfo?.primaryKey && (
                    <span className="ml-1 text-2xs text-gh-accent-fg">(PK)</span>
                  )}
                  {colInfo?.notNull && (
                    <span className="ml-1 text-2xs text-gh-danger-fg">*</span>
                  )}
                  <span className="ml-1 text-2xs text-gh-fg-subtle">{colInfo?.type || 'TEXT'}</span>
                </label>
                <input
                  value={insertValues[col] || ''}
                  onChange={(e) =>
                    setInsertValues((prev) => ({ ...prev, [col]: e.target.value }))
                  }
                  placeholder={colInfo?.defaultValue ? `Default: ${colInfo.defaultValue}` : ''}
                  className="w-full bg-gh-canvas border border-gh-border rounded px-2.5 py-1.5 text-xs text-gh-fg-default focus:outline-none focus:ring-1 focus:ring-gh-accent-emphasis placeholder:text-gh-fg-subtle"
                />
              </div>
            );
          })}
        </div>
      </Modal>
    </div>
  );
}

// ---- Helpers ----

function getPageNumbers(
  current: number,
  total: number,
): (number | '...')[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }

  const pages: (number | '...')[] = [1];
  if (current > 3) pages.push('...');

  const start = Math.max(2, current - 1);
  const end = Math.min(total - 1, current + 1);

  for (let i = start; i <= end; i++) {
    pages.push(i);
  }

  if (current < total - 2) pages.push('...');
  pages.push(total);

  return pages;
}
