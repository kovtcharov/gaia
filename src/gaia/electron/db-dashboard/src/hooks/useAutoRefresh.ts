// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Auto-refresh hook with smooth updates and no flashing.
 * Manages the refresh interval state and provides a human-readable
 * last-updated display.
 */

import { useState, useCallback, useEffect, useRef } from 'react';

export interface AutoRefreshState {
  enabled: boolean;
  interval: number; // milliseconds
  lastUpdate: Date | null;
  relativeTime: string;
}

const INTERVAL_OPTIONS = [
  { value: 1000, label: '1s' },
  { value: 2000, label: '2s' },
  { value: 5000, label: '5s' },
  { value: 10000, label: '10s' },
  { value: 30000, label: '30s' },
  { value: 0, label: 'Off' },
];

export function useAutoRefresh(defaultInterval = 5000) {
  const [enabled, setEnabled] = useState(true);
  const [interval, setInterval_] = useState(defaultInterval);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [relativeTime, setRelativeTime] = useState('');
  const tickRef = useRef<ReturnType<typeof setInterval>>();

  const markUpdated = useCallback(() => {
    setLastUpdate(new Date());
  }, []);

  // Update relative time display every second
  useEffect(() => {
    const updateRelative = () => {
      if (!lastUpdate) {
        setRelativeTime('');
        return;
      }
      const diff = Date.now() - lastUpdate.getTime();
      if (diff < 1000) setRelativeTime('just now');
      else if (diff < 60000) setRelativeTime(`${Math.floor(diff / 1000)}s ago`);
      else if (diff < 3600000) setRelativeTime(`${Math.floor(diff / 60000)}m ago`);
      else setRelativeTime(`${Math.floor(diff / 3600000)}h ago`);
    };

    updateRelative();
    tickRef.current = setInterval(updateRelative, 1000);
    return () => clearInterval(tickRef.current);
  }, [lastUpdate]);

  const effectiveInterval = enabled ? interval : 0;

  return {
    enabled,
    interval: effectiveInterval,
    lastUpdate,
    relativeTime,
    setEnabled,
    setInterval: setInterval_,
    markUpdated,
    intervalOptions: INTERVAL_OPTIONS,
  };
}
