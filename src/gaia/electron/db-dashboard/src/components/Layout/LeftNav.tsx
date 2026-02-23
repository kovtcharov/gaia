// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  Database,
  ChevronLeft,
  ChevronRight,
  Circle,
} from 'lucide-react';
import type { DatabaseInfo } from '../../types/database';

interface LeftNavProps {
  databases: DatabaseInfo[];
  activeTabId: string;
  onSelectTab: (id: string) => void;
  isCollapsed: boolean;
  onToggleCollapsed: () => void;
}

const DB_COLORS: Record<string, string> = {
  'memory.db': '#a371f7',
  'knowledge.db': '#f0883e',
  'tools.db': '#58a6ff',
  'skills.db': '#3fb950',
  'agents.db': '#ffa657',
  'plan.db': '#ff7b72',
  'logs.db': '#8b949e',
};

function getDbColor(name: string): string {
  return DB_COLORS[name] || '#58a6ff';
}

export default function LeftNav({
  databases,
  activeTabId,
  onSelectTab,
  isCollapsed,
  onToggleCollapsed,
}: LeftNavProps) {
  const existingDbs = databases.filter((d) => d.exists);

  return (
    <motion.aside
      animate={{ width: isCollapsed ? 44 : 200 }}
      transition={{ type: 'spring', stiffness: 400, damping: 35 }}
      className="flex flex-col bg-gh-canvas-subtle border-r border-gh-border shrink-0 overflow-hidden"
      style={{ minWidth: isCollapsed ? 44 : 200 }}
    >
      {/* Dashboard item */}
      <NavItem
        id="dashboard"
        label="Dashboard"
        icon={<LayoutDashboard size={15} />}
        isActive={activeTabId === 'dashboard'}
        isCollapsed={isCollapsed}
        onClick={() => onSelectTab('dashboard')}
      />

      {/* Divider + Databases section */}
      <div className="mx-2 my-1 border-t border-gh-border-muted" />

      {!isCollapsed && (
        <div className="px-3 py-1">
          <span className="text-2xs font-semibold text-gh-fg-subtle uppercase tracking-wider">
            Databases
          </span>
        </div>
      )}

      {/* Database items */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {existingDbs.map((db) => {
          const tabId = `db-${db.name}`;
          const color = getDbColor(db.name);
          return (
            <NavItem
              key={db.path}
              id={tabId}
              label={db.label || db.name.replace('.db', '')}
              icon={<Circle size={8} fill={color} color={color} />}
              isActive={activeTabId === tabId}
              isCollapsed={isCollapsed}
              onClick={() => onSelectTab(tabId)}
              tooltip={db.description}
            />
          );
        })}

        {existingDbs.length === 0 && !isCollapsed && (
          <div className="px-3 py-4 text-2xs text-gh-fg-subtle">
            No databases found.
            <br />Run GAIA to create them.
          </div>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={onToggleCollapsed}
        title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className="flex items-center justify-center h-9 border-t border-gh-border-muted text-gh-fg-subtle hover:text-gh-fg-default hover:bg-gh-border-muted transition-colors shrink-0"
      >
        {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>
    </motion.aside>
  );
}

function NavItem({
  id,
  label,
  icon,
  isActive,
  isCollapsed,
  onClick,
  tooltip,
}: {
  id: string;
  label: string;
  icon: React.ReactNode;
  isActive: boolean;
  isCollapsed: boolean;
  onClick: () => void;
  tooltip?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={isCollapsed ? label : tooltip}
      className={`
        relative flex items-center gap-2.5 w-full px-3 py-2 text-xs transition-colors
        ${isActive
          ? 'bg-gh-accent-emphasis/15 text-gh-accent-fg'
          : 'text-gh-fg-muted hover:bg-gh-border-muted hover:text-gh-fg-default'
        }
      `}
    >
      {/* Active indicator bar */}
      {isActive && (
        <motion.div
          layoutId="activeNav"
          className="absolute left-0 top-1 bottom-1 w-0.5 bg-gh-accent-fg rounded-full"
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      )}

      <span className="shrink-0 flex items-center justify-center w-4 h-4">
        {icon}
      </span>

      <AnimatePresence>
        {!isCollapsed && (
          <motion.span
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: 'auto' }}
            exit={{ opacity: 0, width: 0 }}
            transition={{ duration: 0.15 }}
            className="truncate overflow-hidden whitespace-nowrap font-medium"
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
