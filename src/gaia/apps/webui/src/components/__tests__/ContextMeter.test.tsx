// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/** The composer's context meter speaks the chosen language and names a cloud model's own window. */

import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { ContextMeter } from '../ContextMeter';
import { useChatStore } from '../../stores/chatStore';
import type { ContextUsage } from '../../types';

function usage(over: Partial<ContextUsage> = {}): ContextUsage {
    return {
        window: 65536, reply_reserve: 0, system: 12000, tools: 8000, documents: 0, history: 1000, query: 850,
        used: 21850, history_budget: 32768, history_used: 1000, turns_in_view: 1, turns_hidden: 0,
        at_risk: [], at_risk_tokens: 0, ...over,
    };
}

afterEach(() => useChatStore.setState({ locale: 'en' }));

describe('ContextMeter', () => {
    it('labels the empty meter in Arabic', () => {
        useChatStore.setState({ locale: 'ar' });
        render(<ContextMeter usage={null} unavailable="Measured once GAIA has answered in this chat." />);
        expect(document.querySelector('.context-meter-label')?.textContent).toContain('السياق');
    });

    it('breaks the window down in Arabic', () => {
        useChatStore.setState({ locale: 'ar' });
        render(<ContextMeter usage={usage()} />);
        fireEvent.click(document.querySelector('button.context-meter')!);
        const rows = document.querySelector('.context-meter-rows')!.textContent!;
        expect(rows).toContain('المستندات');
        expect(rows).toContain('رسالتك');
    });

    it('names a cloud model\'s own window next to the part GAIA uses', () => {
        render(<ContextMeter usage={usage({ model_window: 1048576 })} />);
        fireEvent.click(screen.getByRole('button', { name: /^Context \d+% full/ }));
        expect(document.querySelector('.context-meter-window')?.textContent).toMatch(/GAIA uses 66K of the model's 1M/);
    });

    it('says when a cloud model\'s own window is not known', () => {
        render(<ContextMeter usage={usage({ model_window: null })} />);
        fireEvent.click(screen.getByRole('button', { name: /^Context \d+% full/ }));
        expect(document.querySelector('.context-meter-window')?.textContent).toContain('window is not known');
    });

    it('adds no window note for a local model', () => {
        render(<ContextMeter usage={usage()} />);
        fireEvent.click(screen.getByRole('button', { name: /^Context \d+% full/ }));
        expect(document.querySelector('.context-meter-window')).toBeNull();
    });
});
