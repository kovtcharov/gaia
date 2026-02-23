// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Activity, AlertTriangle, Cpu, Wrench } from 'lucide-react';
import type { TrendStats } from '../../types/database';

interface StatsCardsProps {
  trendStats: TrendStats | null;
  totalRows: number;
  totalSize: number;
  dbCount: number;
  lastActivity: Date | null;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatRelative(date: Date | null): string {
  if (!date) return 'N/A';
  const diff = Date.now() - date.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  return Math.floor(diff / 86400000) + 'd ago';
}

interface StatCardData {
  label: string;
  value: string;
  trend: 'up' | 'down' | 'stable';
  trendLabel: string;
  trendGood: boolean | null;
  icon: React.ReactNode;
}

export default function StatsCards({ trendStats, totalRows, totalSize, dbCount, lastActivity }: StatsCardsProps) {
  const ts = trendStats;

  const logsDiff = ts ? ts.totalLogs24h - ts.totalLogsPrev24h : 0;
  const errRate24 = ts && ts.totalLogs24h > 0 ? (ts.errors24h / ts.totalLogs24h) * 100 : 0;
  const errRatePrev = ts && ts.totalLogsPrev24h > 0 ? (ts.errorsPrev24h / ts.totalLogsPrev24h) * 100 : 0;
  const errDiff = errRate24 - errRatePrev;

  const cards: StatCardData[] = [
    {
      label: 'Total Logs (24h)',
      value: ts ? ts.totalLogs24h.toLocaleString() : totalRows.toLocaleString(),
      trend: logsDiff > 0 ? 'up' : logsDiff < 0 ? 'down' : 'stable',
      trendLabel: logsDiff !== 0 ? (logsDiff > 0 ? '+' : '') + logsDiff.toLocaleString() : 'stable',
      trendGood: null,
      icon: <Activity size={16} />,
    },
    {
      label: 'Error Rate',
      value: ts ? errRate24.toFixed(1) + '%' : '--',
      trend: errDiff > 0.1 ? 'up' : errDiff < -0.1 ? 'down' : 'stable',
      trendLabel: Math.abs(errDiff) > 0.1 ? (errDiff > 0 ? '+' : '') + errDiff.toFixed(1) + '%' : 'stable',
      trendGood: errDiff < -0.1 ? true : errDiff > 0.1 ? false : null,
      icon: <AlertTriangle size={16} />,
    },
    {
      label: 'Avg Context',
      value: ts && ts.avgContext24h > 0
        ? Math.round(ts.avgContext24h / 1000) > 0
          ? Math.round(ts.avgContext24h / 1000) + 'K tok'
          : Math.round(ts.avgContext24h) + ' tok'
        : '--',
      trend: 'stable',
      trendLabel: ts && ts.avgContext24h > 0 ? 'tracking' : 'no data',
      trendGood: null,
      icon: <Cpu size={16} />,
    },
    {
      label: 'Tool Calls',
      value: ts ? (ts.totalToolCalls || 0).toLocaleString() : '0',
      trend: ts && ts.totalToolCalls > 0 ? 'up' : 'stable',
      trendLabel: ts && ts.totalToolCalls > 0 ? 'active' : 'no data',
      trendGood: ts && ts.totalToolCalls > 0 ? true : null,
      icon: <Wrench size={16} />,
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: i * 0.05 }}
          className="card card-hover p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-gh-fg-muted">{card.icon}</span>
            <TrendIndicator trend={card.trend} trendGood={card.trendGood} />
          </div>
          <div className="text-xl font-bold text-gh-fg-default mb-1">{card.value}</div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gh-fg-muted">{card.label}</span>
            <span
              className={`text-2xs ${
                card.trendGood === true
                  ? 'text-gh-success-fg'
                  : card.trendGood === false
                  ? 'text-gh-danger-fg'
                  : 'text-gh-fg-subtle'
              }`}
            >
              {card.trendLabel}
            </span>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function TrendIndicator({ trend, trendGood }: { trend: 'up' | 'down' | 'stable'; trendGood: boolean | null }) {
  if (trend === 'stable') return <Minus size={14} className="text-gh-fg-subtle" />;
  const color =
    trendGood === true ? 'text-gh-success-fg' : trendGood === false ? 'text-gh-danger-fg' : 'text-gh-fg-muted';
  return trend === 'up' ? <TrendingUp size={14} className={color} /> : <TrendingDown size={14} className={color} />;
}
