// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * SQLConsole - Interactive SQL query editor with syntax-highlighted results.
 *
 * Features:
 * - Multi-line SQL textarea with monospace font
 * - Query history with keyboard navigation (Up/Down arrows)
 * - Results displayed in a scrollable table
 * - Execution timing and row count
 * - Ctrl+Enter / Cmd+Enter to execute
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Clock, History, AlertCircle, CheckCircle2, ChevronUp } from 'lucide-react';
import Button from '../shared/Button';
import Badge from '../shared/Badge';
import { useExecuteSQL } from '../../hooks/useDatabase';
import type { ExecuteSQLResponse, Row } from '../../types/database';

interface SQLConsoleProps {
  dbPath: string;
  readOnly: boolean;
}

interface HistoryEntry {
  sql: string;
  timestamp: Date;
  success: boolean;
  duration?: number;
  rowCount?: number;
}

export default function SQLConsole({ dbPath, readOnly }: SQLConsoleProps) {
  const [sql, setSql] = useState('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showHistory, setShowHistory] = useState(false);
  const [result, setResult] = useState<ExecuteSQLResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const executeMutation = useExecuteSQL();

  const handleExecute = useCallback(async () => {
    const trimmed = sql.trim();
    if (!trimmed) return;

    setError(null);
    setResult(null);

    try {
      const res = await executeMutation.mutateAsync({ dbPath, sql: trimmed, readOnly });
      setResult(res);

      const entry: HistoryEntry = {
        sql: trimmed,
        timestamp: new Date(),
        success: true,
        duration: 'duration' in res ? res.duration : undefined,
        rowCount: 'rowCount' in res ? res.rowCount : ('changes' in res ? res.changes : undefined),
      };
      setHistory((prev) => [entry, ...prev.slice(0, 49)]); // Keep last 50
      setHistoryIndex(-1);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
      setHistory((prev) => [
        { sql: trimmed, timestamp: new Date(), success: false },
        ...prev.slice(0, 49),
      ]);
      setHistoryIndex(-1);
    }
  }, [sql, dbPath, readOnly, executeMutation]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      // Ctrl+Enter or Cmd+Enter to execute
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleExecute();
        return;
      }

      // Up arrow at beginning of input to navigate history
      if (e.key === 'ArrowUp' && textareaRef.current) {
        const { selectionStart } = textareaRef.current;
        if (selectionStart === 0 && history.length > 0) {
          e.preventDefault();
          const newIndex = Math.min(historyIndex + 1, history.length - 1);
          setHistoryIndex(newIndex);
          setSql(history[newIndex].sql);
        }
      }
      if (e.key === 'ArrowDown' && textareaRef.current) {
        const { selectionStart, value } = textareaRef.current;
        if (selectionStart === value.length && historyIndex > 0) {
          e.preventDefault();
          const newIndex = historyIndex - 1;
          setHistoryIndex(newIndex);
          setSql(history[newIndex].sql);
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          setSql('');
        }
      }
    },
    [handleExecute, history, historyIndex],
  );

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 300) + 'px';
    }
  }, [sql]);

  const resultRows = result && 'rows' in result ? result.rows : [];
  const resultColumns = result && 'columns' in result ? result.columns : [];
  const resultType = result && 'type' in result ? result.type : null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Editor area */}
      <div className="p-4 border-b border-gh-border shrink-0">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter SQL query... (Ctrl+Enter to execute)"
            className="sql-textarea min-h-[100px]"
            spellCheck={false}
          />
        </div>

        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              icon={<Play size={12} />}
              onClick={handleExecute}
              disabled={executeMutation.isPending || !sql.trim()}
            >
              {executeMutation.isPending ? 'Running...' : 'Execute'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              icon={showHistory ? <ChevronUp size={12} /> : <History size={12} />}
              onClick={() => setShowHistory(!showHistory)}
            >
              History ({history.length})
            </Button>
            {readOnly && (
              <Badge variant="warning" dot>
                Read-only mode
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-3 text-2xs text-gh-fg-subtle">
            {result && 'duration' in result && (
              <span className="flex items-center gap-1">
                <Clock size={10} />
                {result.duration}ms
              </span>
            )}
            {result && 'rowCount' in result && resultType === 'query' && (
              <span>{result.rowCount} rows</span>
            )}
            {result && 'changes' in result && resultType === 'statement' && (
              <span>{result.changes} rows affected</span>
            )}
            <span className="text-gh-fg-subtle">Ctrl+Enter to run</span>
          </div>
        </div>
      </div>

      {/* History panel */}
      <AnimatePresence>
        {showHistory && history.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden border-b border-gh-border"
          >
            <div className="max-h-48 overflow-y-auto p-2">
              {history.map((entry, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setSql(entry.sql);
                    setShowHistory(false);
                  }}
                  className="w-full flex items-start gap-2 px-3 py-2 rounded-md hover:bg-gh-canvas-subtle/50 text-left transition-colors group"
                >
                  {entry.success ? (
                    <CheckCircle2 size={12} className="text-gh-success-fg shrink-0 mt-0.5" />
                  ) : (
                    <AlertCircle size={12} className="text-gh-danger-fg shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-gh-fg-default font-mono truncate">
                      {entry.sql}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5 text-2xs text-gh-fg-subtle">
                      <span>{entry.timestamp.toLocaleTimeString()}</span>
                      {entry.duration != null && <span>{entry.duration}ms</span>}
                      {entry.rowCount != null && <span>{entry.rowCount} rows</span>}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-4 mt-3 p-3 rounded-md bg-gh-danger-emphasis/10 border border-gh-danger-emphasis/20"
        >
          <div className="flex items-start gap-2">
            <AlertCircle size={14} className="text-gh-danger-fg shrink-0 mt-0.5" />
            <div className="text-xs text-gh-danger-fg font-mono break-all">{error}</div>
          </div>
        </motion.div>
      )}

      {/* Statement result */}
      {result && resultType === 'statement' && (
        <motion.div
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-4 mt-3 p-3 rounded-md bg-gh-success-emphasis/10 border border-gh-success-emphasis/20"
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 size={14} className="text-gh-success-fg" />
            <span className="text-xs text-gh-success-fg">
              Statement executed successfully.{' '}
              {'changes' in result && `${result.changes} row(s) affected.`}
              {'lastInsertRowid' in result &&
                result.lastInsertRowid != null &&
                ` Last insert ID: ${result.lastInsertRowid}.`}
            </span>
          </div>
        </motion.div>
      )}

      {/* Query results table */}
      {result && resultType === 'query' && resultRows.length > 0 && (
        <div className="flex-1 overflow-auto m-4 border border-gh-border rounded-md">
          <table className="data-table">
            <thead>
              <tr>
                {resultColumns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {resultRows.map((row, i) => (
                <motion.tr
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.15, delay: Math.min(i * 0.01, 0.3) }}
                >
                  {resultColumns.map((col) => (
                    <td key={col} className="font-mono text-xs">
                      {row[col] === null || row[col] === undefined ? (
                        <span className="italic text-gh-fg-subtle">NULL</span>
                      ) : typeof row[col] === 'object' ? (
                        JSON.stringify(row[col])
                      ) : (
                        String(row[col])
                      )}
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty query result */}
      {result && resultType === 'query' && resultRows.length === 0 && (
        <div className="flex-1 flex items-center justify-center text-xs text-gh-fg-subtle">
          Query returned 0 rows
        </div>
      )}

      {/* Initial empty state */}
      {!result && !error && (
        <div className="flex-1 flex items-center justify-center text-xs text-gh-fg-subtle">
          Execute a query to see results
        </div>
      )}
    </div>
  );
}
