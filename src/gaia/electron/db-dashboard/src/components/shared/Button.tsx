// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: React.ReactNode;
}

const variantClasses: Record<Variant, string> = {
  primary:
    'bg-gh-accent-emphasis text-white hover:bg-gh-accent-emphasis/80 border border-transparent',
  secondary:
    'bg-gh-canvas-subtle text-gh-fg-default hover:bg-gh-border-muted border border-gh-border',
  danger:
    'bg-gh-danger-emphasis text-white hover:bg-gh-danger-emphasis/80 border border-transparent',
  ghost:
    'bg-transparent text-gh-fg-muted hover:text-gh-fg-default hover:bg-gh-canvas-subtle border border-transparent',
};

const sizeClasses: Record<Size, string> = {
  sm: 'px-2 py-1 text-xs gap-1',
  md: 'px-3 py-1.5 text-sm gap-1.5',
  lg: 'px-4 py-2 text-sm gap-2',
};

export default function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  children,
  className = '',
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={`
        inline-flex items-center justify-center rounded-md font-medium
        transition-all duration-150 ease-in-out
        focus:outline-none focus:ring-2 focus:ring-gh-accent-emphasis/50 focus:ring-offset-1 focus:ring-offset-gh-bg
        disabled:opacity-50 disabled:cursor-not-allowed
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${className}
      `}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </button>
  );
}
