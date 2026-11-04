// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Popover } from './Popover';
import { ContextPanel } from './ContextPanel';
import type { ContextUsage } from '../types';
import { useT } from '../i18n/useT';
import './ContextMeter.css';

export function formatTokens(n: number): string {
    if (n < 1000) return String(n);
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`;
    return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`;
}

const Tokens = ({ n }: { n: number }) => <bdi dir="ltr">{formatTokens(n)}</bdi>;

const PARTS = [
    { key: 'system', label: 'meter.system' },
    { key: 'tools', label: 'meter.tools' },
    { key: 'documents', label: 'meter.documents' },
    { key: 'history', label: 'meter.history' },
    { key: 'query', label: 'meter.query' },
] as const;

interface ContextMeterProps {
    usage: ContextUsage | null;
    unavailable?: string | null;
    /** With a session, the popover links to the full Context & performance panel. */
    sessionId?: string;
    busy?: boolean;
    onUsage?: (usage: ContextUsage) => void;
    /** Show the details somewhere else (the session panel) instead of a sheet. */
    onDetails?: () => void;
}

/** How full the model's context window is, with a breakdown on click. */
export function ContextMeter({ usage, unavailable, sessionId, busy = false, onUsage, onDetails }: ContextMeterProps) {
    const t = useT();
    const anchor = useRef<HTMLButtonElement>(null);
    const [open, setOpen] = useState(false);
    const [details, setDetails] = useState(false);
    const panel = details && sessionId
        ? <ContextPanel sessionId={sessionId} busy={busy} onUsage={onUsage} onClose={() => setDetails(false)} />
        : null;
    if (!usage) {
        if (!unavailable) return null;
        return (
            <span className="context-meter is-empty" title={unavailable}>
                <span className="context-meter-bar" aria-hidden="true"><span style={{ width: 0 }} /></span>
                <span className="context-meter-label">{t('ctx.context')}</span>
            </span>
        );
    }
    const pct = Math.min(100, Math.round((usage.used / usage.window) * 100));
    const warn = usage.at_risk.length > 0;
    const free = Math.max(0, usage.window - usage.used - usage.reply_reserve);
    return (
        <div className="composer-chip-wrap context-meter-wrap">
            <button
                ref={anchor}
                type="button"
                className={`context-meter${warn ? ' is-warning' : ''}`}
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                aria-label={t('meter.aria', { pct, used: formatTokens(usage.used), window: formatTokens(usage.window) })}
                title={t('meter.title')}
            >
                <span className="context-meter-bar" aria-hidden="true"><span style={{ width: `${pct}%` }} /></span>
                <span className="context-meter-label">{pct}%</span>
            </button>
            <Popover open={open} onClose={() => setOpen(false)} anchor={anchor} label={t('meter.popover')} align="end">
                <div className="context-meter-panel">
                    <div className="context-meter-head">
                        <strong><Tokens n={usage.used} /></strong> {t.bare('ctx.of')} <Tokens n={usage.window} /> {t.bare('ctx.tokensWord')}
                    </div>
                    {usage.model_window !== undefined && (usage.model_window == null || usage.model_window > usage.window) && (
                        <p className="context-meter-window" title={t('ctx.windowTitle')}>
                            {t.bare('ctx.gaiaUses')} <Tokens n={usage.window} />
                            {usage.model_window == null
                                ? <>. {t('meter.windowUnknown')}</>
                                : <> {t.bare('ctx.ofModel')} <Tokens n={usage.model_window} /></>}
                        </p>
                    )}
                    <div className="context-meter-stack" aria-hidden="true">
                        {PARTS.map(({ key }) => (
                            <span key={key} className={`part-${key}`} style={{ width: `${(usage[key] / usage.window) * 100}%` }} />
                        ))}
                    </div>
                    <dl className="context-meter-rows">
                        {PARTS.filter(({ key }) => key !== 'query' || usage.query > 0).map(({ key, label }) => (
                            <div key={key} className="context-meter-row">
                                <dt><span className={`swatch part-${key}`} aria-hidden="true" />{t(label)}</dt>
                                <dd><Tokens n={usage[key]} /></dd>
                            </div>
                        ))}
                        <div className="context-meter-row is-quiet">
                            <dt>{t('meter.replyReserve')}</dt>
                            <dd><Tokens n={usage.reply_reserve} /></dd>
                        </div>
                        <div className="context-meter-row is-quiet">
                            <dt>{t('ctx.freeRow')}</dt>
                            <dd><Tokens n={free} /></dd>
                        </div>
                    </dl>
                    <p className="context-meter-note">
                        {t.plural('meter.inView', usage.turns_in_view)}
                        {usage.turns_hidden > 0 && <>{' '}{t.plural('meter.outOfView', usage.turns_hidden)}</>}
                        {' '}{usage.tokenizer?.exact ? t('meter.exact') : t('meter.estimates')}
                        {' '}{t('meter.recall')}
                    </p>
                    {sessionId && (
                        <button
                            type="button"
                            className="context-meter-details"
                            onClick={() => {
                                setOpen(false);
                                if (onDetails) onDetails();
                                else setDetails(true);
                            }}
                        >
                            {t('meter.details')}
                        </button>
                    )}
                </div>
            </Popover>
            {panel}
        </div>
    );
}

/** Says which turns leave the model's view next, before they go. */
export function ContextWarning({ usage }: { usage: ContextUsage | null }) {
    const t = useT();
    if (!usage || usage.at_risk.length === 0) return null;
    const n = usage.at_risk.length;
    return (
        <div className="context-warning" role="status">
            <AlertTriangle size={13} aria-hidden="true" />
            <span>
                {t.plural('meter.warn', n)}{' '}
                {usage.at_risk.map((q, i) => (
                    <span key={i}>{i > 0 && ', '}<q dir="auto">{q}</q></span>
                ))}
                . {t('meter.warnTail')}
            </span>
        </div>
    );
}
