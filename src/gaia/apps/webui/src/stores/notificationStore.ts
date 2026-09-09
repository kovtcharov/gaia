// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Zustand store for notification management.
 *
 * Handles permission requests, security alerts, status changes, and general
 * notifications from OS agents. Notifications are persisted in memory and
 * cleared on dismiss or on session end.
 */

import { create } from 'zustand';
import type { GaiaNotification, NotificationType } from '../types/agent';
import { confirmTool } from '../services/api';

// ── Constants ────────────────────────────────────────────────────────────

/** Maximum notifications kept in the center to prevent unbounded growth. */
const MAX_NOTIFICATIONS = 500;

/**
 * Legacy localStorage key for the "always allow" tool list.
 *
 * Always-allow is now SESSION-scoped and lives in this store only: a tick on
 * `run_shell_command` must not silently approve every shell command in every
 * future session for the life of the browser profile. The key is still named
 * here so `purgeLegacyAlwaysAllow()` can delete any grant a previous build
 * persisted.
 */
export const LEGACY_ALWAYS_ALLOW_TOOLS_KEY = 'gaia_always_allow_tools';

/**
 * Drop any always-allow list persisted by an older build. Called once at app
 * start; grants from a previous run are not carried into this session.
 */
export function purgeLegacyAlwaysAllow(): void {
    try {
        if (localStorage.getItem(LEGACY_ALWAYS_ALLOW_TOOLS_KEY) !== null) {
            localStorage.removeItem(LEGACY_ALWAYS_ALLOW_TOOLS_KEY);
            console.warn(
                '[notificationStore] Discarded a persisted "always allow" tool list from an ' +
                'earlier version — always-allow is now session-scoped and revocable in ' +
                'Settings → Tools & Permissions.'
            );
        }
    } catch (err) {
        console.error('[notificationStore] Could not clear the legacy always-allow list:', err);
    }
}

// ── State Interface ──────────────────────────────────────────────────────

interface NotificationState {
  /** All notifications (newest first). */
  notifications: GaiaNotification[];
  /** Whether the notification panel is open. */
  showPanel: boolean;
  /** Active type filter for the notification center (null = all). */
  typeFilter: NotificationType | null;

  // ── Actions ─────────────────────────────────────────────────────────
  addNotification: (notification: GaiaNotification) => void;
  dismiss: (id: string) => void;
  markRead: (id: string) => void;
  markAllRead: () => void;
  clearAll: () => void;
  setShowPanel: (show: boolean) => void;
  setTypeFilter: (type: NotificationType | null) => void;

  /**
   * Tools the user chose to always allow, for THIS session only. Cleared on
   * reload; revocable from Settings → Tools & Permissions.
   */
  alwaysAllowTools: string[];

  /** Respond to a permission request notification. */
  respondToPermission: (id: string, action: 'allow' | 'deny', remember: boolean) => Promise<void>;

  /** Whether a tool carries a session always-allow grant. */
  isAlwaysAllowed: (tool: string) => boolean;

  /** Revoke one session always-allow grant. */
  revokeAlwaysAllow: (tool: string) => void;

  /** Revoke every session always-allow grant. */
  revokeAllAlwaysAllow: () => void;
}

// ── Store Implementation ─────────────────────────────────────────────────

export const useNotificationStore = create<NotificationState>((set, get) => ({
  notifications: [],
  showPanel: false,
  typeFilter: null,
  alwaysAllowTools: [],

  addNotification: (notification) =>
    set((state) => ({
      notifications: [notification, ...state.notifications].slice(0, MAX_NOTIFICATIONS),
    })),

  dismiss: (id) =>
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === id ? { ...n, dismissed: true } : n
      ),
    })),

  markRead: (id) =>
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === id ? { ...n, read: true } : n
      ),
    })),

  markAllRead: () =>
    set((state) => ({
      notifications: state.notifications.map((n) => ({ ...n, read: true })),
    })),

  clearAll: () => set({ notifications: [] }),

  setShowPanel: (show) => set({ showPanel: show }),

  setTypeFilter: (type) => set({ typeFilter: type }),

  respondToPermission: async (id, action, remember) => {
    // Find the notification to get the session ID for the REST call
    const notification = get().notifications.find((n) => n.id === id);
    const sessionId = notification?.agentId;

    // Try Electron IPC first, then fall back to REST API
    const electronApi = window.gaiaAPI;
    if (electronApi?.notification?.respondPermission) {
      try {
        await electronApi.notification.respondPermission(id, action, remember);
      } catch (err) {
        console.error('[notificationStore] Failed to send permission response via IPC:', err);
        // Don't update local state — the agent didn't receive the response.
        // The permission prompt remains actionable so the user can retry.
        return;
      }
    } else if (sessionId) {
      // Web browser mode — call the REST endpoint
      try {
        await confirmTool(sessionId, action === 'allow');
      } catch (err) {
        console.error('[notificationStore] Failed to send permission response via REST:', err);
        return;
      }
    }
    // Record "always allow" for this session only — never persisted.
    if (action === 'allow' && remember && notification?.tool) {
      const tool = notification.tool;
      set((state) =>
        state.alwaysAllowTools.includes(tool)
          ? state
          : { alwaysAllowTools: [...state.alwaysAllowTools, tool] }
      );
    }
    // Update local state after response is delivered
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === id
          ? { ...n, response: action, respondedAt: Date.now(), read: true }
          : n
      ),
    }));
  },

  isAlwaysAllowed: (tool) => get().alwaysAllowTools.includes(tool),

  revokeAlwaysAllow: (tool) =>
    set((state) => ({
      alwaysAllowTools: state.alwaysAllowTools.filter((t) => t !== tool),
    })),

  revokeAllAlwaysAllow: () => set({ alwaysAllowTools: [] }),

}));

// ── Selectors ────────────────────────────────────────────────────────────

/** Get unread count (excluding dismissed). */
export const selectUnreadCount = (state: NotificationState): number =>
  state.notifications.filter((n) => !n.read && !n.dismissed).length;

/** Get pending permission requests. */
export const selectPendingPermissions = (state: NotificationState): GaiaNotification[] =>
  state.notifications.filter(
    (n) => n.type === 'permission_request' && !n.response && !n.dismissed
  );

/** Get visible (non-dismissed) notifications, optionally filtered by type. */
export const selectVisibleNotifications = (state: NotificationState): GaiaNotification[] => {
  const visible = state.notifications.filter((n) => !n.dismissed);
  if (state.typeFilter) {
    return visible.filter((n) => n.type === state.typeFilter);
  }
  return visible;
};

/** Tools always-allowed for this session. */
export const selectAlwaysAllowTools = (state: NotificationState): string[] =>
  state.alwaysAllowTools;

/** Get the first pending permission request (reactive selector for PermissionPrompt). */
export const selectActivePermissionPrompt = (state: NotificationState): GaiaNotification | null =>
  state.notifications.find(
    (n) => n.type === 'permission_request' && !n.response && !n.dismissed
  ) ?? null;
