// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Remark plugin: show Windows paths in prose exactly as written.
 *
 * CommonMark reads `\` before ASCII punctuation as an escape, so
 * `C:\Users\me\.gaia` renders as `C:\Users\me.gaia`. This walks the parsed
 * text nodes and puts the source spelling of each drive (`C:\...`) or UNC
 * (`\\server\...`) path back. Inline code and code blocks are separate node
 * types and are never touched; escapes outside a path keep working.
 * Must run before `remarkSoftBreaks`, whose split text nodes carry no source position.
 */

import { findWindowsPaths, type PathSpan } from './windowsPaths';

interface MdNode {
    type: string;
    value?: string;
    children?: MdNode[];
    position?: { start: { offset?: number }; end: { offset?: number } };
}

/**
 * A drive or UNC path in markdown source, up to whitespace, a quote, backtick,
 * `]`, a char Windows forbids, or a `)` that closes no `(` inside the path.
 * An escaped `\*` stays in, so a glob like `C:\logs\*.txt` keeps its backslash;
 * a bare `*` ends the path so emphasis around it still parses. A path with spaces
 * is extended past the first space by `findWindowsPaths`.
 */
const RAW_WIN_PATH_RE = /(?:(?<![A-Za-z0-9])[A-Za-z]:\\|\\\\[A-Za-z0-9_.$-]+\\)(?:\\\*|[^\s"`()\]<>|*]|\((?:\\\*|[^\s"`()\]<>|*])*\))*/g;

const ESCAPE_RE = /\\([!-/:-@[-`{-~])/g;

/** Escaped chars that end a path segment; masked (same length) so the spaced-path scan reads past them. */
const SEGMENT_ENDING_ESCAPE_RE = /\\[*\]]/g;

/** What CommonMark makes of a path's source text: each `\` + punctuation becomes the punctuation. */
function unescaped(raw: string): string {
    return raw.replace(ESCAPE_RE, '$1');
}

/** Path spans in markdown source: the escape-aware regex merged with the spaced-path scan. */
function rawPathSpans(raw: string): PathSpan[] {
    const spans: PathSpan[] = [
        ...Array.from(raw.matchAll(RAW_WIN_PATH_RE), (m) => ({ start: m.index, end: m.index + m[0].length })),
        ...findWindowsPaths(raw.replace(SEGMENT_ENDING_ESCAPE_RE, '\\x')),
    ].sort((a, b) => a.start - b.start);
    const merged: PathSpan[] = [];
    for (const span of spans) {
        const last = merged[merged.length - 1];
        if (last && span.start < last.end) last.end = Math.max(last.end, span.end);
        else merged.push({ ...span });
    }
    return merged;
}

function restorePaths(node: MdNode, source: string): void {
    if (node.type === 'text' && typeof node.value === 'string') {
        const start = node.position?.start.offset;
        const end = node.position?.end.offset;
        if (start === undefined || end === undefined) return;
        const raw = source.slice(start, end);
        if (!raw.includes('\\')) return;

        let value = node.value;
        let cursor = 0;
        for (const { start: from, end: to } of rawPathSpans(raw)) {
            const token = raw.slice(from, to);
            const shown = unescaped(token);
            if (shown === token) {
                // Step past it, or a later path's display form could match here instead.
                const same = value.indexOf(token, cursor);
                if (same !== -1) cursor = same + token.length;
                continue;
            }
            const at = value.indexOf(shown, cursor);
            // Not found means the parser read it differently (an entity, a hard break): leave it.
            if (at === -1) continue;
            value = value.slice(0, at) + token + value.slice(at + shown.length);
            cursor = at + token.length;
        }
        node.value = value;
        return;
    }
    node.children?.forEach((child) => restorePaths(child, source));
}

export default function remarkWindowsPaths() {
    return (tree: MdNode, file: { value?: unknown }) => {
        if (typeof file.value !== 'string') return;
        restorePaths(tree, file.value);
    };
}
