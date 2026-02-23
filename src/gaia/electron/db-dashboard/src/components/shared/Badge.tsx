// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import React from 'react';

type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'neutral' | 'purple';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
}

const variantClasses: Record<BadgeVariant, string> = {
  success: 'badge-success',
  warning: 'badge-warning',
  danger: 'badge-danger',
  info: 'badge-info',
  neutral: 'badge-neutral',
  purple: 'badge-purple',
};

export default function Badge({ variant = 'neutral', children, className = '', dot }: BadgeProps) {
  return (
    <span className={`${variantClasses[variant]} ${className}`}>
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full mr-1 ${
            variant === 'success'
              ? 'bg-gh-success-fg'
              : variant === 'danger'
              ? 'bg-gh-danger-fg'
              : variant === 'warning'
              ? 'bg-gh-attention-fg'
              : variant === 'info'
              ? 'bg-gh-accent-fg'
              : 'bg-gh-fg-muted'
          }`}
        />
      )}
      {children}
    </span>
  );
}
