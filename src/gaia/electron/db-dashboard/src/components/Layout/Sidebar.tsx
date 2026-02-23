// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { motion } from 'framer-motion';
import {
  Table2,
  Search,
  Eye,
  FileCode2,
  Download,
  Save,
} from 'lucide-react';
import type { TableInfo, DatabaseInfo } from '../../types/database';
import Badge from '../shared/Badge';
import Button from '../shared/Button';

interface SidebarProps {
  tables: TableInfo[];
  selectedTable: string | null;
  onSelectTable: (name: string) => void;
  databases: DatabaseInfo[];
  currentDbPath: string | null;
  onSelectDatabase: (db: DatabaseInfo) => void;
  onViewSchema: () => void;
  onBackup: () => void;
  isLoading: boolean;
}

function formatRowCount(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

export default function Sidebar({
  tables,
  selectedTable,
  onSelectTable,
  databases,
  currentDbPath,
  onSelectDatabase,
  onViewSchema,
  onBackup,
  isLoading,
}: SidebarProps) {
  const existingDbs = databases.filter((d) => d.exists);
  const regularTables = tables.filter((t) => !t.isFts);
  const ftsTables = tables.filter((t) => t.isFts);

  return (
    <aside className="flex flex-col w-60 bg-gh-canvas-subtle border-r border-gh-border shrink-0 overflow-hidden">
      {/* Database picker */}
      <div className="px-3 py-3 border-b border-gh-border">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Databases
          </span>
          <Badge variant="info">{existingDbs.length}</Badge>
        </div>
        <div className="space-y-0.5 max-h-40 overflow-y-auto">
          {existingDbs.map((db) => (
            <button
              key={db.path}
              onClick={() => onSelectDatabase(db)}
              className={`
                w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors
                ${
                  db.path === currentDbPath
                    ? 'bg-gh-accent-emphasis/15 text-gh-accent-fg'
                    : 'text-gh-fg-muted hover:bg-gh-border-muted hover:text-gh-fg-default'
                }
              `}
            >
              <Search size={12} className="shrink-0 opacity-60" />
              <span className="truncate">{db.label || db.name.replace('.db', '')}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Table list */}
      <div className="flex-1 overflow-y-auto px-3 py-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider">
            Tables
          </span>
          <Badge variant="neutral">{tables.length}</Badge>
        </div>

        {isLoading ? (
          <div className="py-8 text-center text-xs text-gh-fg-subtle animate-pulse-subtle">
            Loading tables...
          </div>
        ) : tables.length === 0 ? (
          <div className="py-8 text-center text-xs text-gh-fg-subtle">
            Select a database
          </div>
        ) : (
          <div className="space-y-0.5">
            {regularTables.map((table) => (
              <motion.button
                key={table.name}
                onClick={() => onSelectTable(table.name)}
                initial={false}
                animate={{
                  backgroundColor:
                    table.name === selectedTable
                      ? 'rgba(31, 111, 235, 0.15)'
                      : 'transparent',
                }}
                transition={{ duration: 0.15 }}
                className={`
                  w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors group
                  ${
                    table.name === selectedTable
                      ? 'text-gh-accent-fg'
                      : 'text-gh-fg-muted hover:bg-gh-border-muted hover:text-gh-fg-default'
                  }
                `}
              >
                <Table2 size={12} className="shrink-0 opacity-60" />
                <span className="truncate flex-1 text-left">{table.name}</span>
                <span className="text-2xs text-gh-fg-subtle group-hover:text-gh-fg-muted shrink-0">
                  {formatRowCount(table.rowCount)}
                </span>
              </motion.button>
            ))}

            {ftsTables.length > 0 && (
              <>
                <div className="pt-3 pb-1 text-2xs font-semibold text-gh-fg-subtle uppercase tracking-wider">
                  FTS5 Virtual
                </div>
                {ftsTables.map((table) => (
                  <button
                    key={table.name}
                    onClick={() => onSelectTable(table.name)}
                    className={`
                      w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors
                      ${
                        table.name === selectedTable
                          ? 'bg-gh-done-emphasis/15 text-gh-done-fg'
                          : 'text-gh-fg-muted hover:bg-gh-border-muted hover:text-gh-fg-default'
                      }
                    `}
                  >
                    <Search size={12} className="shrink-0 opacity-60" />
                    <span className="truncate flex-1 text-left">{table.name}</span>
                    <span className="text-2xs text-gh-fg-subtle">{formatRowCount(table.rowCount)}</span>
                  </button>
                ))}
              </>
            )}
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="flex items-center gap-1 px-3 py-2 border-t border-gh-border">
        <Button size="sm" variant="ghost" onClick={onViewSchema} icon={<FileCode2 size={13} />}>
          Schema
        </Button>
        <Button size="sm" variant="ghost" onClick={onBackup} icon={<Save size={13} />}>
          Backup
        </Button>
      </div>
    </aside>
  );
}
