// Copyright(C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from 'vitest';

import {
  artifactName,
  detectOs,
  findInstallers,
  resolveArch,
  resolvePlatform,
  type NavigatorLike,
} from './download-target';

const UA = {
  win: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36',
  mac: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36',
  macSafari:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
  linux:
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36',
  linuxArm:
    'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36',
  iphone:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile Safari/604.1',
  android:
    'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Mobile Safari/537.36',
};

const nav = (
  userAgent: string,
  opts: { architecture?: string; hintsThrow?: boolean; maxTouchPoints?: number } = {}
): NavigatorLike => ({
  userAgent,
  maxTouchPoints: opts.maxTouchPoints ?? 0,
  userAgentData:
    opts.architecture !== undefined || opts.hintsThrow
      ? {
          getHighEntropyValues: async () => {
            if (opts.hintsThrow) throw new Error('refused');
            return { architecture: opts.architecture };
          },
        }
      : undefined,
});

describe('detectOs', () => {
  it('places the three desktop platforms', () => {
    expect(detectOs(UA.win)).toBe('win');
    expect(detectOs(UA.mac)).toBe('darwin');
    expect(detectOs(UA.linux)).toBe('linux');
  });

  // Mobile has no build, and iOS/Android UAs both carry a desktop OS token —
  // matching them would hand a phone an x64 desktop binary.
  it('refuses mobile', () => {
    expect(detectOs(UA.iphone)).toBeNull();
    expect(detectOs(UA.android)).toBeNull();
  });
});

describe('resolveArch', () => {
  it('trusts client hints over the UA string', async () => {
    // The UA says "Intel"/"x64" on both; only the hint is truthful.
    expect(await resolveArch(nav(UA.mac, { architecture: 'arm' }), 'darwin')).toBe('arm64');
    expect(await resolveArch(nav(UA.win, { architecture: 'arm' }), 'win')).toBe('arm64');
    expect(await resolveArch(nav(UA.win, { architecture: 'x86' }), 'win')).toBe('x64');
  });

  it('reads an explicit arm64 token when hints are absent', async () => {
    expect(await resolveArch(nav(UA.linuxArm), 'linux')).toBe('arm64');
  });

  it('falls back to x64 on Windows and Linux', async () => {
    expect(await resolveArch(nav(UA.win), 'win')).toBe('x64');
    expect(await resolveArch(nav(UA.linux), 'linux')).toBe('x64');
  });

  // Safari reports "Intel Mac OS X" on Apple Silicon and offers no hints, so
  // the UA alone cannot decide; touch points are the one honest signal.
  it('uses touch points to place an Apple Silicon Mac behind Safari', async () => {
    expect(await resolveArch(nav(UA.macSafari, { maxTouchPoints: 5 }), 'darwin')).toBe('arm64');
  });

  it('refuses to guess a Mac architecture with no evidence', async () => {
    expect(await resolveArch(nav(UA.macSafari), 'darwin')).toBeNull();
    expect(await resolveArch(nav(UA.mac, { hintsThrow: true }), 'darwin')).toBeNull();
  });
});

describe('resolvePlatform', () => {
  it('returns the hub platform key', async () => {
    expect(await resolvePlatform(nav(UA.win))).toBe('win-x64');
    expect(await resolvePlatform(nav(UA.linuxArm))).toBe('linux-arm64');
    expect(await resolvePlatform(nav(UA.mac, { architecture: 'arm' }))).toBe('darwin-arm64');
  });

  it('returns null for anything unplaceable, so the caller shows every option', async () => {
    expect(await resolvePlatform(nav(UA.iphone))).toBeNull();
    expect(await resolvePlatform(nav(UA.macSafari))).toBeNull();
  });
});

describe('artifactName', () => {
  // These are the exact filenames release_components.yml publishes; a wrong one
  // is not a build failure anywhere, it is a 404 on the visitor's click.
  it('matches the hub filenames, .exe on Windows only', () => {
    expect(artifactName('win-x64')).toBe('gaia-win-x64.exe');
    expect(artifactName('win-arm64')).toBe('gaia-win-arm64.exe');
    expect(artifactName('darwin-arm64')).toBe('gaia-darwin-arm64');
    expect(artifactName('linux-x64')).toBe('gaia-linux-x64');
  });
});

describe('findInstallers', () => {
  // A real release carries the flagship installers, the Agent UI's installers,
  // and the raw terminal-hub binaries side by side under the same `gaia-`
  // prefix. Matching the wrong one hands the visitor a different product.
  const asset = (name: string) => ({
    name,
    browser_download_url: `https://github.com/amd/gaia/releases/download/v0.23.0/${name}`,
  });

  const RELEASE = [
    asset('gaia-0.1.1-x64-setup.exe'),
    asset('gaia-0.1.1-arm64.dmg'),
    asset('gaia-0.1.1-x64.dmg'),
    asset('gaia-0.1.1-x64.deb'),
    asset('gaia-0.1.1-x64.AppImage'),
    asset('gaia-agent-ui-0.23.0-x64-setup.exe'),
    asset('gaia-agent-ui-0.23.0-arm64.dmg'),
    asset('gaia-win-x64.exe'),
    asset('gaia-darwin-arm64'),
    asset('gaia-linux-x64'),
  ];

  it('resolves the flagship installer for each supported platform', () => {
    expect(findInstallers('win-x64', RELEASE).map((i) => i.name)).toEqual([
      'gaia-0.1.1-x64-setup.exe',
    ]);
    expect(findInstallers('darwin-arm64', RELEASE).map((i) => i.name)).toEqual([
      'gaia-0.1.1-arm64.dmg',
    ]);
    expect(findInstallers('darwin-x64', RELEASE).map((i) => i.name)).toEqual([
      'gaia-0.1.1-x64.dmg',
    ]);
  });

  it('offers the .deb first and the AppImage second on Linux', () => {
    expect(findInstallers('linux-x64', RELEASE).map((i) => i.name)).toEqual([
      'gaia-0.1.1-x64.deb',
      'gaia-0.1.1-x64.AppImage',
    ]);
  });

  it('never matches the Agent UI installer or a raw binary', () => {
    const decoysOnly = RELEASE.filter((a) => !/^gaia-\d/.test(a.name));
    for (const platform of ['win-x64', 'darwin-arm64', 'darwin-x64', 'linux-x64'] as const) {
      expect(findInstallers(platform, decoysOnly)).toEqual([]);
    }
  });

  it('returns nothing for platforms with no frozen sidecar', () => {
    expect(findInstallers('win-arm64', RELEASE)).toEqual([]);
    expect(findInstallers('linux-arm64', RELEASE)).toEqual([]);
  });

  // The separator dots must be ESCAPED. A plain template literal collapses `\.`
  // to `.`, and the patterns then match any character there — which is how a
  // near-miss filename gets offered to a visitor as the real installer.
  it('treats the extension separator as a literal dot, not a wildcard', () => {
    const nearMisses = [
      asset('gaia-0.1.1-x64Xdeb'),
      asset('gaia-0.1.1-arm64-dmg'),
      asset('gaia-0.1.1-x64-setup-exe'),
      asset('gaia-0.1.1-x64_AppImage'),
    ];
    expect(findInstallers('linux-x64', nearMisses)).toEqual([]);
    expect(findInstallers('darwin-arm64', nearMisses)).toEqual([]);
    expect(findInstallers('win-x64', nearMisses)).toEqual([]);
  });

  it('returns nothing for a release cut before the installers existed', () => {
    const old = [asset('gaia-agent-ui-0.23.0-x64-setup.exe'), asset('gaia-win-x64.exe')];
    expect(findInstallers('win-x64', old)).toEqual([]);
  });
});
