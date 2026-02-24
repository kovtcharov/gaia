// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * DataGrid - Virtualized data table with sorting, filtering, pagination,
 * column resizing, and inline editing support.
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
import { useDeleteRow, useUpdateRow, useInsertRow, useClearTable } from '../../hooks/useDatabase';
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
const ACTIONS_WIDTH = 64;
const MIN_COL_WIDTH = 60;
const DEFAULT_COL_WIDTH = 150;

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

  // ---- Column widths ----
  const [colWidths, setColWidths] = useState<Record<string, number>>({});
  const resizingRef = useRef<{ col: string; startX: number; startWidth: number } | null>(null);

  const listRef = useRef<List>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState(500);

  // ---- Mutations ----
  const deleteRow = useDeleteRow();
  const updateRow = useUpdateRow();
  const insertRow = useInsertRow();
  const clearTable = useClearTable();

  // ---- Resize observer ----
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerHeight(entry.contentRect.height - HEADER_HEIGHT - 48);
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

  // ---- Initialize column widths when table/columns change ----
  useEffect(() => {
    if (colNames.length === 0) return;
    const containerWidth = containerRef.current?.clientWidth || 900;
    const actionsArea = readOnly ? 0 : ACTIONS_WIDTH;
    const available = Math.max(containerWidth - actionsArea - 2, colNames.length * DEFAULT_COL_WIDTH);
    const defaultW = Math.max(DEFAULT_COL_WIDTH, Math.floor(available / colNames.length));

    setColWidths((prev) => {
      // Only reset if the column set changed (different table)
      const prevCols = Object.keys(prev);
      const sameTable = colNames.every((c) => prevCols.includes(c)) && prevCols.every((c) => colNames.includes(c));
      if (sameTable && prevCols.length > 0) return prev; // keep user's sizes
      const next: Record<string, number> = {};
      for (const col of colNames) {
        next[col] = prev[col] ?? defaultW;
      }
      return next;
    });
  }, [colNames, readOnly]);

  // ---- Column resize handlers ----
  const getColWidth = useCallback(
    (col: string) => colWidths[col] ?? DEFAULT_COL_WIDTH,
    [colWidths],
  );

  const handleResizeStart = useCallback(
    (e: React.MouseEvent, col: string) => {
      e.preventDefault();
      e.stopPropagation();
      resizingRef.current = { col, startX: e.clientX, startWidth: getColWidth(col) };

      const onMouseMove = (ev: MouseEvent) => {
        if (!resizingRef.current) return;
        const delta = ev.clientX - resizingRef.current.startX;
        const newWidth = Math.max(MIN_COL_WIDTH, resizingRef.current.startWidth + delta);
        setColWidths((prev) => ({ ...prev, [resizingRef.current!.col]: newWidth }));
      };

      const onMouseUp = () => {
        resizingRef.current = null;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
      };

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    },
    [getColWidth],
  );

  const handleResizeDoubleClick = useCallback(
    (e: React.MouseEvent, col: string) => {
      e.preventDefault();
      e.stopPropagation();
      // Reset to default width on double-click
      setColWidths((prev) => ({ ...prev, [col]: DEFAULT_COL_WIDTH }));
    },
    [],
  );

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

  const handleClearTable = useCallback(async () => {
    if (readOnly) return;
    if (!confirm(`Delete ALL ${pagination.totalRows} rows from "${tableName}"? This cannot be undone.`)) return;
    await clearTable.mutateAsync({ dbPath, tableName });
  }, [readOnly, pagination.totalRows, tableName, dbPath, clearTable]);

  const handleCopyCell = useCallback((value: string, key: string) => {
    navigator.clipboard?.writeText(value);
    setCopiedCell(key);
    setTimeout(() => setCopiedCell(null), 1500);
  }, []);

  // ---- Tooltip ----
  const [tooltip, setTooltip] = useState<{ content: string; x: number; y: number } | null>(null);

  const handleCellMouseEnter = useCallback((e: React.MouseEvent<HTMLSpanElement>, content: string) => {
    const el = e.currentTarget;
    if (el.scrollWidth > el.offsetWidth + 1) {
      const rect = el.getBoundingClientRect();
      setTooltip({ content, x: rect.left, y: rect.top - 6 });
    }
  }, []);

  const handleCellMouseLeave = useCallback(() => {
    setTooltip(null);
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
            <div className="shrink-0 flex items-center justify-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ width: ACTIONS_WIDTH }}>
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
            const width = getColWidth(col);

            return (
              <div
                key={col}
                className="shrink-0 px-3 py-1 relative group/cell border-r border-gh-border-muted/30 overflow-hidden"
                style={{ width }}
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
                    className={`text-xs block truncate ${isNull ? 'italic text-gh-fg-subtle' : 'text-gh-fg-default'} cursor-default`}
                    onMouseEnter={(e) => handleCellMouseEnter(e, displayValue)}
                    onMouseLeave={handleCellMouseLeave}
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
      colWidths,
      handleSaveEdit,
      handleStartEdit,
      handleDeleteRow,
      handleCopyCell,
      handleCellMouseEnter,
      handleCellMouseLeave,
      getColWidth,
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

          {/* Clear table */}
          {!readOnly && pagination.totalRows > 0 && (
            <Button
              size="sm"
              variant="ghost"
              icon={<Trash2 size={12} />}
              onClick={handleClearTable}
              disabled={clearTable.isPending}
            >
              Clear All
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
        className="flex items-center bg-gh-canvas-subtle border-b border-gh-border shrink-0 overflow-hidden"
        style={{ height: HEADER_HEIGHT }}
      >
        {!readOnly && <div className="shrink-0" style={{ width: ACTIONS_WIDTH }} />}
        {colNames.map((col) => {
          const isSorted = queryOptions.sortColumn === col;
          const width = getColWidth(col);
          return (
            <div
              key={col}
              className="relative shrink-0 flex items-center border-r border-gh-border-muted/40"
              style={{ width }}
            >
              <button
                onClick={() => handleSort(col)}
                className="flex-1 flex items-center gap-1 px-3 h-full text-left text-xs font-semibold text-gh-fg-muted uppercase tracking-wider hover:text-gh-fg-default transition-colors select-none overflow-hidden"
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

              {/* Resize handle */}
              <div
                className="absolute right-0 top-0 h-full w-2 cursor-default z-10 flex items-center justify-center group/resize"
                onMouseDown={(e) => handleResizeStart(e, col)}
                onDoubleClick={(e) => handleResizeDoubleClick(e, col)}
                title="Drag to resize · Double-click to reset"
              >
                <div className="w-px h-4 bg-gh-border-muted group-hover/resize:bg-gh-accent-emphasis/60 group-hover/resize:h-full transition-all" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Virtual scrolling body */}
      <div className="flex-1 overflow-auto">
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

      {/* Cell tooltip (only when text is truncated) */}
      {tooltip && (
        <div
          className="fixed z-50 max-w-sm px-3 py-2 rounded-md shadow-xl border border-gh-border bg-gh-canvas text-xs text-gh-fg-default whitespace-pre-wrap break-words pointer-events-none"
          style={{
            left: Math.min(tooltip.x, window.innerWidth - 320),
            top: tooltip.y,
            transform: 'translateY(-100%)',
          }}
        >
          {tooltip.content}
        </div>
      )}

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
