// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Session tool grants — the list of tools the user ticked "allow for the rest
 * of this session" on, and the only place those grants can be revoked.
 *
 * Grants live in `notificationStore` for the lifetime of the session; nothing
 * is persisted, so this list is empty again after a restart.
 */

import { ShieldCheck, RotateCcw } from 'lucide-react';
import { useNotificationStore, selectAlwaysAllowTools } from '../stores/notificationStore';
import './SessionToolGrants.css';

export function SessionToolGrants() {
    const alwaysAllowTools = useNotificationStore(selectAlwaysAllowTools);
    const revokeAlwaysAllow = useNotificationStore((s) => s.revokeAlwaysAllow);
    const revokeAllAlwaysAllow = useNotificationStore((s) => s.revokeAllAlwaysAllow);

    return (
        <section className="perm-session-grants" aria-labelledby="perm-session-grants-title">
            <div className="perm-session-grants-head">
                <ShieldCheck size={14} className="perm-session-grants-icon" />
                <h4 id="perm-session-grants-title" className="perm-session-grants-title">
                    Always-allowed this session
                </h4>
                {alwaysAllowTools.length > 0 && (
                    <button
                        className="btn-secondary perm-session-revoke-all"
                        onClick={revokeAllAlwaysAllow}
                    >
                        <RotateCcw size={13} />
                        Revoke All
                    </button>
                )}
            </div>
            {alwaysAllowTools.length === 0 ? (
                <p className="perm-session-grants-empty">
                    No tools are auto-approved. Ticking &ldquo;allow for the rest of this
                    session&rdquo; on a permission prompt adds one here; every grant ends when GAIA
                    restarts.
                </p>
            ) : (
                <ul className="perm-session-grant-list">
                    {alwaysAllowTools.map((tool) => (
                        <li key={tool} className="perm-session-grant">
                            <code className="perm-tool-name">{tool}</code>
                            <span className="perm-session-grant-note">
                                runs without prompting until restart
                            </span>
                            <button
                                className="perm-session-grant-revoke"
                                onClick={() => revokeAlwaysAllow(tool)}
                                aria-label={`Revoke the session grant for ${tool}`}
                            >
                                Revoke
                            </button>
                        </li>
                    ))}
                </ul>
            )}
        </section>
    );
}
