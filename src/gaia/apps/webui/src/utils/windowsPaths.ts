// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/** Finds drive (`C:\...`, `C:/...`) and UNC (`\\server\share\...`) paths in message text. */

export interface PathSpan {
    start: number;
    end: number;
}

const PATH_START_RE = /(?<![A-Za-z0-9_\\])(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.$-]+\\)/g;
const SEGMENT_CHAR_RE = /[^\s*?"<>|,;)}\]\\/]/;
const SEPARATOR_RE = /[\\/]/;
/** Sentence punctuation a path swallows: "see C:\a\b." ends at `b`. */
const TRAILING_PUNCT_RE = /[.:!']+$/;
const EXTENSION_RE = /\.(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{1,8}$/;
const MAX_SPACED_WORDS = 3;

const isSegmentChar = (ch: string | undefined) => ch !== undefined && SEGMENT_CHAR_RE.test(ch);
const isSeparator = (ch: string | undefined) => ch !== undefined && SEPARATOR_RE.test(ch);
const hasExtension = (word: string) => EXTENSION_RE.test(word.replace(TRAILING_PUNCT_RE, ''));

/** End of the run of segment characters from `i`; spaced words may not hold a `:` Windows forbids. */
function wordEnd(text: string, i: number, noColon: boolean): number {
    while (isSegmentChar(text[i]) && !(noColon && text[i] === ':')) i++;
    return i;
}

/**
 * Where a space after a segment ending at `i` stops being part of the path, or -1 if it ends here.
 * Unquoted, it continues only if within three words the path reaches a separator or a word with
 * a file extension; quoted, it always continues. Either way, never after a word with an extension.
 */
function spacedRunEnd(text: string, i: number, wordStart: number, quoted: boolean): number {
    if (hasExtension(text.slice(wordStart, i))) return -1;
    let at = i;
    for (let words = 0; quoted || words < MAX_SPACED_WORDS; words++) {
        if (text[at] !== ' ' || !isSegmentChar(text[at + 1])) return -1;
        const end = wordEnd(text, at + 1, true);
        if (end === at + 1) return -1;
        if (quoted || isSeparator(text[end]) || hasExtension(text.slice(at + 1, end))) return end;
        at = end;
    }
    return -1;
}

/** End of a path whose root ends at `rootEnd`, or -1 if no segment follows the root. */
function scanPath(text: string, rootEnd: number, quoted: boolean): number {
    let i = rootEnd;
    if (!isSegmentChar(text[i])) return -1;
    let wordStart = i;
    for (;;) {
        i = wordEnd(text, i, false);
        if (isSeparator(text[i])) {
            i++;
            if (!isSegmentChar(text[i])) return i;
            wordStart = i;
            continue;
        }
        const spaced = spacedRunEnd(text, i, wordStart, quoted);
        if (spaced === -1) return i;
        // The spaced words join the current segment; keep scanning from their end.
        wordStart = text.lastIndexOf(' ', spaced - 1) + 1;
        i = spaced;
    }
}

/** True when `text` is one path and nothing else, read as a quoted path (inline code). */
export function isWholeWindowsPath(text: string): boolean {
    const root = new RegExp(PATH_START_RE.source).exec(text);
    if (!root || root.index !== 0) return false;
    return scanPath(text, root[0].length, true) === text.length;
}

/** Every path in `text`, in order, without trailing sentence punctuation. */
export function findWindowsPaths(text: string): PathSpan[] {
    const spans: PathSpan[] = [];
    const re = new RegExp(PATH_START_RE.source, 'g');
    let root: RegExpExecArray | null;
    while ((root = re.exec(text)) !== null) {
        const start = root.index;
        const rootEnd = start + root[0].length;
        let end = -1;
        const quote = text[start - 1];
        if (quote === '"' || quote === "'") {
            const close = text.indexOf(quote, rootEnd);
            const line = text.indexOf('\n', rootEnd);
            if (close !== -1 && (line === -1 || close < line) && scanPath(text.slice(0, close), rootEnd, true) === close) {
                end = close;
            }
        }
        if (end === -1) {
            end = scanPath(text, rootEnd, false);
            if (end === -1) continue;
            end -= (text.slice(start, end).match(TRAILING_PUNCT_RE)?.[0].length ?? 0);
        }
        spans.push({ start, end });
        re.lastIndex = end;
    }
    return spans;
}
