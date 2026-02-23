// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { motion } from 'framer-motion';
import { LayoutDashboard, Database, X } from 'lucide-react';
import type { AppTab } from '../../types/database';

interface TabsProps {
  tabs: AppTab[];
  activeTabId: string;
  onSelectTab: (id: string) => void;
  onCloseTab: (id: string) => void;
}

export default function Tabs({ tabs, activeTabId, onSelectTab, onCloseTab }: TabsProps) {
  return (
    <div className="flex items-center h-9 px-2 bg-gh-bg border-b border-gh-border overflow-x-auto shrink-0">
      {tabs.map((tab) => {
        const isActive = tab.id === activeTabId;
        return (
          <button
            key={tab.id}
            onClick={() => onSelectTab(tab.id)}
            className={`
              relative flex items-center gap-1.5 px-3 h-full text-xs font-medium
              transition-colors duration-150 shrink-0
              ${
                isActive
                  ? 'text-gh-fg-default'
                  : 'text-gh-fg-muted hover:text-gh-fg-default'
              }
            `}
          >
            {tab.type === 'dashboard' ? (
              <LayoutDashboard size={13} className={isActive ? 'text-gh-accent-fg' : ''} />
            ) : (
              <Database size={13} className={isActive ? 'text-gh-accent-fg' : ''} />
            )}
            <span className="max-w-[120px] truncate">{tab.label}</span>

            {/* Close button for database tabs */}
            {tab.type === 'database' && (
              <span
                onClick={(e) => {
                  e.stopPropagation();
                  onCloseTab(tab.id);
                }}
                className="ml-1 p-0.5 rounded hover:bg-gh-border-muted text-gh-fg-subtle hover:text-gh-fg-default transition-colors"
              >
                <X size={11} />
              </span>
            )}

            {/* Active indicator */}
            {isActive && (
              <motion.div
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-gh-accent-fg rounded-full"
                layoutId="activeTab"
                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}
