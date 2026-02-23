// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { HeatmapEntry } from '../../types/database';

interface ActivityHeatmapProps {
  data: HeatmapEntry[];
}

/**
 * GitHub-style contribution/activity heatmap.
 * Displays up to 52 weeks of daily activity in a calendar grid.
 */
export default function ActivityHeatmap({ data }: ActivityHeatmapProps) {
  const { grid, maxCount, months } = useMemo(() => {
    // Build a map of date -> count
    const countMap = new Map<string, number>();
    data.forEach((entry) => {
      countMap.set(entry.date, entry.count);
    });

    // Generate the last 52 weeks of dates
    const today = new Date();
    const days: { date: string; count: number; dayOfWeek: number; weekIndex: number }[] = [];
    const totalDays = 52 * 7;

    // Start from (totalDays - 1) days ago
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - totalDays + 1);
    // Align to the nearest Sunday
    startDate.setDate(startDate.getDate() - startDate.getDay());

    const monthLabels: { label: string; weekIndex: number }[] = [];
    let lastMonth = -1;

    for (let i = 0; i < totalDays + 7; i++) {
      const d = new Date(startDate);
      d.setDate(d.getDate() + i);
      if (d > today) break;

      const dateStr = d.toISOString().split('T')[0];
      const dayOfWeek = d.getDay();
      const weekIndex = Math.floor(i / 7);
      const count = countMap.get(dateStr) || 0;

      days.push({ date: dateStr, count, dayOfWeek, weekIndex });

      // Track month labels
      const month = d.getMonth();
      if (month !== lastMonth && dayOfWeek <= 3) {
        monthLabels.push({
          label: d.toLocaleDateString('en-US', { month: 'short' }),
          weekIndex,
        });
        lastMonth = month;
      }
    }

    const maxVal = Math.max(...days.map((d) => d.count), 1);

    return { grid: days, maxCount: maxVal, months: monthLabels };
  }, [data]);

  function getColor(count: number): string {
    if (count === 0) return '#161b22';
    const ratio = count / maxCount;
    if (ratio <= 0.25) return '#0e4429';
    if (ratio <= 0.5) return '#006d32';
    if (ratio <= 0.75) return '#26a641';
    return '#39d353';
  }

  // Group by weeks
  const weeks = useMemo(() => {
    const result: typeof grid[] = [];
    let currentWeek: typeof grid = [];
    let currentWeekIndex = -1;

    for (const day of grid) {
      if (day.weekIndex !== currentWeekIndex) {
        if (currentWeek.length > 0) result.push(currentWeek);
        currentWeek = [];
        currentWeekIndex = day.weekIndex;
      }
      currentWeek.push(day);
    }
    if (currentWeek.length > 0) result.push(currentWeek);
    return result;
  }, [grid]);

  const dayLabels = ['', 'Mon', '', 'Wed', '', 'Fri', ''];

  return (
    <div className="card card-hover p-4">
      <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider mb-3">
        Activity
      </h3>
      <div className="overflow-x-auto">
        {/* Month labels */}
        <div className="flex ml-6 mb-1">
          {months.map((m, i) => (
            <span
              key={i}
              className="text-2xs text-gh-fg-subtle"
              style={{ marginLeft: i === 0 ? m.weekIndex * 13 : undefined, width: '52px' }}
            >
              {m.label}
            </span>
          ))}
        </div>

        <div className="flex gap-0">
          {/* Day labels */}
          <div className="flex flex-col gap-[2px] mr-1 shrink-0">
            {dayLabels.map((label, i) => (
              <span key={i} className="text-2xs text-gh-fg-subtle h-[11px] leading-[11px]">
                {label}
              </span>
            ))}
          </div>

          {/* Grid */}
          <div className="flex gap-[2px]">
            {weeks.map((week, wi) => (
              <div key={wi} className="flex flex-col gap-[2px]">
                {Array.from({ length: 7 }).map((_, di) => {
                  const day = week.find((d) => d.dayOfWeek === di);
                  if (!day) {
                    return <div key={di} className="w-[11px] h-[11px]" />;
                  }
                  return (
                    <motion.div
                      key={day.date}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3, delay: wi * 0.005 }}
                      className="w-[11px] h-[11px] rounded-[2px] cursor-default"
                      style={{ backgroundColor: getColor(day.count) }}
                      title={`${day.date}: ${day.count} entries`}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-end gap-1 mt-2">
          <span className="text-2xs text-gh-fg-subtle mr-1">Less</span>
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
            <div
              key={ratio}
              className="w-[11px] h-[11px] rounded-[2px]"
              style={{ backgroundColor: getColor(ratio * maxCount) }}
            />
          ))}
          <span className="text-2xs text-gh-fg-subtle ml-1">More</span>
        </div>
      </div>
    </div>
  );
}
