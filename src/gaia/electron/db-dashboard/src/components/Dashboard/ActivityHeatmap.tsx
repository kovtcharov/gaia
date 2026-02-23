// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { HourlyActivityEntry, MinuteActivityEntry } from '../../types/database';

interface ActivityHeatmapProps {
  data: HourlyActivityEntry[];
  minuteData?: MinuteActivityEntry[];
}

function HourTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { hour_label: string; count: number } }> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg text-xs">
      <div className="text-gh-fg-default font-medium">{payload[0].payload.hour_label}</div>
      <div className="text-gh-fg-muted">{payload[0].payload.count} events</div>
    </div>
  );
}

function MinuteTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { bucket_label: string; count: number } }> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gh-canvas-subtle border border-gh-border rounded-md px-3 py-2 shadow-lg text-xs">
      <div className="text-gh-fg-default font-medium">{payload[0].payload.bucket_label}</div>
      <div className="text-gh-fg-muted">{payload[0].payload.count} events</div>
    </div>
  );
}

/**
 * Activity chart showing last 24 hours by hour + last 60 min by 5-min bucket.
 */
export default function ActivityHeatmap({ data, minuteData = [] }: ActivityHeatmapProps) {
  // Fill in all 24 hours so empty hours show as zero bars
  const hourlyData = useMemo(() => {
    const map = new Map<string, number>();
    data.forEach((e) => map.set(e.hour_label, e.count));

    const now = new Date();
    const result: { hour_label: string; count: number; isCurrent: boolean }[] = [];
    for (let h = 23; h >= 0; h--) {
      const d = new Date(now);
      d.setHours(d.getHours() - h, 0, 0, 0);
      const label = `${String(d.getHours()).padStart(2, '0')}:00`;
      result.push({
        hour_label: h === 0 ? 'now' : `-${h}h`,
        count: map.get(label) || 0,
        isCurrent: h === 0,
      });
    }
    return result;
  }, [data]);

  // Fill in 12 five-minute buckets for last 60 min (show empty buckets too)
  const minuteBuckets = useMemo(() => {
    const map = new Map<string, number>();
    minuteData.forEach((e) => map.set(e.bucket_label, e.count));

    const now = new Date();
    // Round down to nearest 5-min
    const base = new Date(now);
    base.setSeconds(0, 0);
    base.setMinutes(Math.floor(base.getMinutes() / 5) * 5);

    const result: { bucket_label: string; display: string; count: number }[] = [];
    for (let i = 11; i >= 0; i--) {
      const d = new Date(base);
      d.setMinutes(d.getMinutes() - i * 5);
      const key = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
      result.push({
        bucket_label: key,
        display: i === 0 ? 'now' : `-${i * 5}m`,
        count: map.get(key) || 0,
      });
    }
    return result;
  }, [minuteData]);

  const maxHourly = Math.max(...hourlyData.map((d) => d.count), 1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
      className="card card-hover p-4 flex flex-col gap-4"
    >
      {/* Last 24 hours */}
      <div>
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider mb-2">
          Last 24 Hours
        </h3>
        {data.length === 0 ? (
          <div className="py-6 text-center text-xs text-gh-fg-subtle">No activity data</div>
        ) : (
          <div className="h-28">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData} margin={{ top: 2, right: 4, bottom: 0, left: -20 }}>
                <XAxis
                  dataKey="hour_label"
                  tick={{ fontSize: 9, fill: '#6e7681' }}
                  axisLine={false}
                  tickLine={false}
                  interval={3}
                />
                <YAxis
                  tick={{ fontSize: 9, fill: '#6e7681' }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                />
                <Tooltip content={<HourTooltip />} cursor={{ fill: 'rgba(88, 166, 255, 0.06)' }} />
                <Bar dataKey="count" radius={[2, 2, 0, 0]} animationDuration={400}>
                  {hourlyData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.count === 0 ? '#21262d' : entry.count / maxHourly > 0.66 ? '#39d353' : entry.count / maxHourly > 0.33 ? '#26a641' : '#0e4429'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Divider */}
      <div className="border-t border-gh-border-muted" />

      {/* Last 60 minutes */}
      <div>
        <h3 className="text-xs font-semibold text-gh-fg-muted uppercase tracking-wider mb-2">
          Last 60 Minutes
        </h3>
        {minuteData.length === 0 ? (
          <div className="py-4 text-center text-xs text-gh-fg-subtle">No recent activity</div>
        ) : (
          <div className="h-20">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={minuteBuckets} margin={{ top: 2, right: 4, bottom: 0, left: -20 }}>
                <XAxis
                  dataKey="display"
                  tick={{ fontSize: 9, fill: '#6e7681' }}
                  axisLine={false}
                  tickLine={false}
                  interval={2}
                />
                <YAxis
                  tick={{ fontSize: 9, fill: '#6e7681' }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                />
                <Tooltip content={<MinuteTooltip />} cursor={{ fill: 'rgba(88, 166, 255, 0.06)' }} />
                <Bar dataKey="count" fill="#58a6ff" radius={[2, 2, 0, 0]} animationDuration={300} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </motion.div>
  );
}
