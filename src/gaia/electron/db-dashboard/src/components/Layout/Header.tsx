// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import {
  RefreshCw,
  FolderOpen,
  FileUp,
  Clock,
  Lock,
  Unlock,
  Database,
} from 'lucide-react';
import Button from '../shared/Button';

interface HeaderProps {
  workspacePath: string;
  readOnly: boolean;
  onToggleReadOnly: () => void;
  onRefresh: () => void;
  onChangeWorkspace: () => void;
  onOpenFile: () => void;
  autoRefreshEnabled: boolean;
  onToggleAutoRefresh: () => void;
  refreshInterval: number;
  onChangeInterval: (val: number) => void;
  intervalOptions: { value: number; label: string }[];
  lastUpdateText: string;
}

export default function Header({
  workspacePath,
  readOnly,
  onToggleReadOnly,
  onRefresh,
  onChangeWorkspace,
  onOpenFile,
  autoRefreshEnabled,
  onToggleAutoRefresh,
  refreshInterval,
  onChangeInterval,
  intervalOptions,
  lastUpdateText,
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between h-12 px-4 bg-gh-canvas-subtle border-b border-gh-border shrink-0">
      {/* Left: Logo */}
      <div className="flex items-center gap-2">
        <Database size={18} className="text-gh-accent-fg" />
        <span className="text-sm font-bold text-gh-fg-default tracking-tight">GAIA</span>
        <span className="text-sm text-gh-fg-muted font-medium">DB Dashboard</span>
      </div>

      {/* Center: Workspace path */}
      <div className="flex items-center gap-2 text-xs text-gh-fg-muted">
        <span className="max-w-[300px] truncate font-mono" title={workspacePath}>
          {workspacePath || 'No workspace'}
        </span>
        <Button size="sm" variant="ghost" onClick={onChangeWorkspace} icon={<FolderOpen size={13} />}>
          Change
        </Button>
        <Button size="sm" variant="ghost" onClick={onOpenFile} icon={<FileUp size={13} />}>
          Open File
        </Button>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-3">
        {/* Auto-refresh */}
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleAutoRefresh}
            className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${
              autoRefreshEnabled ? 'bg-gh-success-emphasis' : 'bg-gh-border'
            }`}
          >
            <span
              className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform duration-200 ${
                autoRefreshEnabled ? 'left-[18px]' : 'left-0.5'
              }`}
            />
          </button>
          <select
            value={refreshInterval}
            onChange={(e) => onChangeInterval(Number(e.target.value))}
            className="bg-gh-bg border border-gh-border rounded text-xs text-gh-fg-muted px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-gh-accent-emphasis"
          >
            {intervalOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          {lastUpdateText && (
            <span className="flex items-center gap-1 text-2xs text-gh-fg-subtle">
              <Clock size={10} />
              {lastUpdateText}
            </span>
          )}
        </div>

        <div className="w-px h-5 bg-gh-border" />

        {/* Refresh button */}
        <Button size="sm" variant="ghost" onClick={onRefresh} icon={<RefreshCw size={14} />} title="Refresh (Ctrl+R)" />

        {/* Read-only toggle */}
        <button
          onClick={onToggleReadOnly}
          className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-colors ${
            readOnly
              ? 'bg-gh-attention-emphasis/15 text-gh-attention-fg'
              : 'bg-gh-success-emphasis/15 text-gh-success-fg'
          }`}
          title={readOnly ? 'Read-only mode (click to enable writes)' : 'Write mode (click to go read-only)'}
        >
          {readOnly ? <Lock size={12} /> : <Unlock size={12} />}
          {readOnly ? 'Read Only' : 'Read/Write'}
        </button>
      </div>
    </header>
  );
}
