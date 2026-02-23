// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // GitHub dark theme palette
        gh: {
          bg: '#0d1117',
          canvas: '#010409',
          'canvas-subtle': '#161b22',
          'canvas-inset': '#010409',
          border: '#30363d',
          'border-muted': '#21262d',
          'fg-default': '#e6edf3',
          'fg-muted': '#8b949e',
          'fg-subtle': '#6e7681',
          'accent-fg': '#58a6ff',
          'accent-emphasis': '#1f6feb',
          'success-fg': '#3fb950',
          'success-emphasis': '#238636',
          'attention-fg': '#d29922',
          'attention-emphasis': '#9e6a03',
          'danger-fg': '#f85149',
          'danger-emphasis': '#da3633',
          'done-fg': '#a371f7',
          'done-emphasis': '#8957e5',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Noto Sans',
          'Helvetica',
          'Arial',
          'sans-serif',
          'Apple Color Emoji',
          'Segoe UI Emoji',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'SF Mono',
          'Menlo',
          'Consolas',
          'Liberation Mono',
          'monospace',
        ],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-subtle': 'pulseSubtle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
};
