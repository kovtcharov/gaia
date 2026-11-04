// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * A Windows path with a space in it is one chip, in the user's message and in
 * the answer. Prose after a path stays prose.
 */

import { fireEvent, render } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MessageBubble } from '../MessageBubble';
import type { Message } from '../../types';
import * as api from '../../services/api';

vi.mock('../../services/api');
const mockedApi = vi.mocked(api);

function renderMessage(content: string, role: Message['role'] = 'assistant') {
    const message: Message = { id: 1, session_id: 's', role, content, created_at: '', rag_sources: null };
    return render(<MessageBubble message={message} />);
}

const chips = () => Array.from(document.querySelectorAll('.file-path-link')).map((l) => l.textContent);

beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.openFileOrFolder.mockResolvedValue({ status: 'ok', path: '' });
});

describe('paths with spaces', () => {
    it.each(['user', 'assistant'] as const)('links the whole path in a %s message', (role) => {
        const path = 'C:\\Users\\test\\Documents\\Q3 report.txt';
        renderMessage(`Summarize ${path} please`, role);
        expect(chips()).toEqual([path]);

        fireEvent.click(document.querySelector('.file-path-link')!);
        expect(mockedApi.openFileOrFolder).toHaveBeenCalledWith(path);
    });

    it.each([
        ['a folder with a space mid-path', 'C:\\Program Files\\GAIA\\gaia.exe'],
        ['several words before the extension', 'D:\\My Documents\\Q3 final report.docx'],
        ['a trailing folder separator', 'C:\\Program Files\\'],
        ['forward slashes', 'C:/Users/me/My Notes/todo.md'],
        ['a UNC path', '\\\\server\\share\\notes.md'],
        ['a UNC path with a space', '\\\\nas\\team share\\Q3 report.pdf'],
    ])('links %s as one chip', (_label, path) => {
        renderMessage(`Open ${path} now.`);
        expect(chips()).toEqual([path]);
    });

    it('takes a quoted path whole even without an extension', () => {
        renderMessage('Install into "C:\\Program Files\\My App" first.');
        expect(chips()).toEqual(['C:\\Program Files\\My App']);
    });

    it('takes a backticked path whole', () => {
        renderMessage('Saved to `C:\\Users\\me\\My Docs\\Q3 report.txt`.');
        expect(chips()).toEqual(['C:\\Users\\me\\My Docs\\Q3 report.txt']);
    });

    it('keeps a backticked command with a flag as code', () => {
        renderMessage('Run `C:\\tools\\fix.exe --force` here.');
        expect(chips()).toEqual([]);
        expect(document.querySelector('code.inline-code')?.textContent).toBe('C:\\tools\\fix.exe --force');
    });
});

describe('prose after a path', () => {
    it.each([
        ['see C:\\x\\y.txt for details', 'C:\\x\\y.txt'],
        ['Open C:\\x\\y.txt and then notes.md', 'C:\\x\\y.txt'],
        ['In C:\\Users\\me\\notes find it.', 'C:\\Users\\me\\notes'],
        ['Copy \\\\server\\share\\notes.md to your desktop.', '\\\\server\\share\\notes.md'],
        ['Saved to C:\\a\\Q3 report.txt.', 'C:\\a\\Q3 report.txt'],
    ])('in "%s" links only %s', (content, path) => {
        renderMessage(content);
        expect(chips()).toEqual([path]);
        expect(document.querySelector('.md-content')?.textContent).toBe(content);
    });

    it('links two paths in one sentence separately', () => {
        renderMessage('Compare C:\\a\\Q3 report.txt with C:\\b\\Q4 report.txt today');
        expect(chips()).toEqual(['C:\\a\\Q3 report.txt', 'C:\\b\\Q4 report.txt']);
    });
});
