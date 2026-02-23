// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * FTSSearch - Full-text search interface for FTS5 virtual tables.
 *
 * Features:
 * - Select from available FTS5 tables
 * - Real-time search as you type (debounced)
 * - Results displayed in a clean table with match highlighting
 * - Search timing and result count
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Table2, AlertCircle, Clock, Hash } from 'lucide-react';
import Button from '../shared/Button';
import Badge from '../shared/Badge';
import { useFTS5Search } from '../../hooks/useDatabase';
import type { TableInfo, Row } from '../../types/database';

interface FTSSearchProps {
  dbPath: string;
  ftsTables: TableInfo[];
}

export default function FTSSearch({ dbPath, ftsTables }: FTSSearchProps) {
  const [selectedFts, setSelectedFts] = useState<string>(ftsTables[0]?.name || '');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Row[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const searchMutation = useFTS5Search();

  // Update selectedFts when ftsTables change
  useEffect(() => {
    if (ftsTables.length > 0 && !ftsTables.find((t) => t.name === selectedFts)) {
      setSelectedFts(ftsTables[0].name);
    }
  }, [ftsTables, selectedFts]);

  const executeSearch = useCallback(
    async (q: string) => {
      if (!q.trim() || !selectedFts) {
        setResults([]);
        setSearchTime(null);
        return;
      }

      setIsSearching(true);
      setError(null);
      const start = Date.now();

      try {
        const res = await searchMutation.mutateAsync({
          dbPath,
          ftsTable: selectedFts,
          query: q.trim(),
          limit: 100,
        });
        setResults(res.rows);
        setSearchTime(Date.now() - start);
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Search failed';
        setError(msg);
        setResults([]);
        setSearchTime(null);
      } finally {
        setIsSearching(false);
      }
    },
    [dbPath, selectedFts, searchMutation],
  );

  // Debounced search
  const handleQueryChange = useCallback(
    (value: string) => {
      setQuery(value);
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        executeSearch(value);
      }, 300);
    },
    [executeSearch],
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const resultColumns = results.length > 0 ? Object.keys(results[0]).filter((k) => k !== 'rank') : [];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Search controls */}
      <div className="p-4 border-b border-gh-border shrink-0 space-y-3">
        <div className="flex items-center gap-3">
          {/* FTS table selector */}
          <div className="flex items-center gap-2">
            <Table2 size={14} className="text-gh-fg-muted" />
            <select
              value={selectedFts}
              onChange={(e) => {
                setSelectedFts(e.target.value);
                setResults([]);
                setSearchTime(null);
                setError(null);
              }}
              className="bg-gh-canvas border border-gh-border rounded text-xs text-gh-fg-default px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-gh-accent-emphasis"
            >
              {ftsTables.length === 0 && <option value="">No FTS tables</option>}
              {ftsTables.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>

          {/* Search input */}
          <div className="flex-1 relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gh-fg-subtle"
            />
            <input
              value={query}
              onChange={(e) => handleQueryChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  if (debounceRef.current) clearTimeout(debounceRef.current);
                  executeSearch(query);
                }
              }}
              placeholder="Search full-text index..."
              className="w-full bg-gh-canvas border border-gh-border rounded-md pl-9 pr-3 py-1.5 text-sm text-gh-fg-default focus:outline-none focus:ring-2 focus:ring-gh-accent-emphasis focus:border-transparent placeholder:text-gh-fg-subtle"
              autoFocus
            />
            {isSearching && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <div className="w-4 h-4 border-2 border-gh-accent-fg border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
        </div>

        {/* Search info */}
        <div className="flex items-center gap-3 text-2xs text-gh-fg-subtle">
          {searchTime != null && (
            <span className="flex items-center gap-1">
              <Clock size={10} />
              {searchTime}ms
            </span>
          )}
          {results.length > 0 && (
            <span className="flex items-center gap-1">
              <Hash size={10} />
              {results.length} results
            </span>
          )}
          <span>
            Enter search terms separated by spaces. Results are ranked by relevance.
          </span>
        </div>
      </div>

      {/* Error */}
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

      {/* Results */}
      {results.length > 0 ? (
        <div className="flex-1 overflow-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="w-12">#</th>
                {resultColumns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((row, i) => (
                <motion.tr
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.15, delay: Math.min(i * 0.015, 0.3) }}
                >
                  <td className="text-2xs text-gh-fg-subtle font-mono">{i + 1}</td>
                  {resultColumns.map((col) => (
                    <td key={col} className="text-xs">
                      {row[col] === null || row[col] === undefined ? (
                        <span className="italic text-gh-fg-subtle">NULL</span>
                      ) : (
                        <HighlightText text={String(row[col])} query={query} />
                      )}
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : query.trim() && !isSearching && !error ? (
        <div className="flex-1 flex items-center justify-center text-xs text-gh-fg-subtle">
          No results found for "{query}"
        </div>
      ) : !query.trim() ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-gh-fg-subtle">
          <Search size={28} className="opacity-30" />
          <span className="text-xs">Enter a search query to search the FTS5 index</span>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Simple text highlighter that bolds matching terms.
 */
function HighlightText({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;

  const terms = query
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

  if (terms.length === 0) return <>{text}</>;

  const pattern = new RegExp(`(${terms.join('|')})`, 'gi');
  const parts = text.split(pattern);

  return (
    <>
      {parts.map((part, i) =>
        pattern.test(part) ? (
          <mark key={i} className="bg-gh-attention-emphasis/30 text-gh-fg-default rounded-sm px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}
