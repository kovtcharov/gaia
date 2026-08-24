// Copyright(C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Which published terminal-hub binary this visitor's machine can run.
//
// The hub publishes six: {win,darwin,linux} x {x64,arm64}, so unlike a
// single-architecture installer this has to resolve the ARCHITECTURE too, and
// only the client hints report it truthfully — every OS masks arm64 in the UA
// string to keep old sites working. A machine we cannot place resolves to null
// and the caller keeps its full platform list on screen, which is the honest
// outcome: an arm64 user handed an x64 binary gets a failure at exec time, not
// a download error they can act on.

export type Os = 'win' | 'darwin' | 'linux';
export type Arch = 'x64' | 'arm64';
/** Matches the hub's platform keys, e.g. `win-x64`. */
export type PlatformKey = `${Os}-${Arch}`;

export interface UserAgentDataLike {
  getHighEntropyValues?(hints: string[]): Promise<{ architecture?: string }>;
}

export interface NavigatorLike {
  userAgent: string;
  platform?: string;
  maxTouchPoints?: number;
  userAgentData?: UserAgentDataLike;
}

export const OS_LABELS: Record<Os, string> = {
  win: 'Windows',
  darwin: 'macOS',
  linux: 'Linux',
};

export const PLATFORM_LABELS: Record<PlatformKey, string> = {
  'win-x64': 'Windows (x64)',
  'win-arm64': 'Windows (ARM64)',
  'darwin-x64': 'macOS (Intel)',
  'darwin-arm64': 'macOS (Apple Silicon)',
  'linux-x64': 'Linux (x64)',
  'linux-arm64': 'Linux (ARM64)',
};

export function detectOs(userAgent: string): Os | null {
  if (/Windows/i.test(userAgent)) return 'win';
  if (/Mac/i.test(userAgent) && !/iPhone|iPad|iPod/i.test(userAgent)) return 'darwin';
  if (/Linux/i.test(userAgent) && !/Android/i.test(userAgent)) return 'linux';
  return null;
}

/**
 * Apple Silicon without client hints (Safari): the UA says "Intel" on every
 * Mac, so the UA cannot decide it. A Mac reporting touch points is an M-series
 * machine — Intel Macs report 0 — which is the one signal Safari does expose.
 */
function appleSiliconBySideChannel(nav: NavigatorLike): boolean {
  return (nav.maxTouchPoints ?? 0) > 0;
}

export async function resolveArch(nav: NavigatorLike, os: Os): Promise<Arch | null> {
  const hints = nav.userAgentData?.getHighEntropyValues;
  if (hints) {
    try {
      const { architecture } = await nav.userAgentData!.getHighEntropyValues!([
        'architecture',
      ]);
      if (architecture === 'arm') return 'arm64';
      if (architecture === 'x86') return 'x64';
    } catch {
      // Hints can be refused outright; fall through to the UA evidence.
    }
  }

  // Explicit arm64 in the UA — Linux and Windows do sometimes say so.
  if (/aarch64|arm64/i.test(nav.userAgent)) return 'arm64';
  if (os === 'darwin') return appleSiliconBySideChannel(nav) ? 'arm64' : null;
  // Windows and Linux on arm64 without hints are rare enough that x64 is the
  // safe read; on macOS it is not, which is why that case returns null above.
  if (/x86_64|win64|wow64|x64|amd64|intel/i.test(nav.userAgent)) return 'x64';
  return null;
}

export async function resolvePlatform(nav: NavigatorLike): Promise<PlatformKey | null> {
  const os = detectOs(nav.userAgent);
  if (!os) return null;
  const arch = await resolveArch(nav, os);
  if (!arch) return null;
  return `${os}-${arch}`;
}

/** The hub's artifact filename for a platform, e.g. `win-x64` -> `gaia-win-x64.exe`. */
export function artifactName(platform: PlatformKey): string {
  return platform.startsWith('win-') ? `gaia-${platform}.exe` : `gaia-${platform}`;
}

// ---- One-click installers ----
//
// The raw binaries above are the payload; these are the installers that wrap
// them. `.github/workflows/build-flagship-installers.yml` builds one per
// platform, bundles the frozen agent sidecar and (on Windows) Lemonade Server,
// and uploads them to the GitHub Release. The names below are that workflow's
// artifact contract — a mismatch is not a build failure anywhere, it is a
// visitor who never sees the installer and gets handed a bare binary instead.

/** Platforms an installer is published for. The agent sidecar is only frozen
 *  for these four, so win-arm64 and linux-arm64 have no installer to offer. */
export const INSTALLER_PLATFORMS = [
  'win-x64',
  'darwin-arm64',
  'darwin-x64',
  'linux-x64',
] as const;

export type InstallerPlatform = (typeof INSTALLER_PLATFORMS)[number];

export interface InstallerAssetSpec {
  /** What the visitor is downloading, e.g. "Installer (.exe)". */
  label: string;
  pattern: RegExp;
}

// Anchored, and the character after `gaia-` must be a digit — that is what
// keeps these from matching the raw binaries (`gaia-win-x64.exe`) or the Agent
// UI's own installers (`gaia-agent-ui-0.23.0-x64-setup.exe`), which share the
// release and the prefix.
// Starts with a digit (that is what separates these from `gaia-agent-ui-*` and
// the raw `gaia-win-x64.exe`), and may carry a prerelease suffix — the workflow
// publishes `-rc.`/`-beta.` tags, so `0.1.1-rc.1` has to match too.
const VER = String.raw`\d[\w.]*(?:-[\w.]+)?`;

// String.raw on every pattern below is load-bearing: a plain template literal
// collapses `\.` to `.`, which turns the extension separator into "any
// character" and lets a near-miss filename match.

export const INSTALLER_ASSETS: Record<InstallerPlatform, InstallerAssetSpec[]> = {
  'win-x64': [
    { label: 'Installer (.exe)', pattern: new RegExp(String.raw`^gaia-${VER}-x64-setup\.exe$`, 'i') },
  ],
  'darwin-arm64': [
    { label: 'Disk image (.dmg)', pattern: new RegExp(String.raw`^gaia-${VER}-arm64\.dmg$`, 'i') },
  ],
  'darwin-x64': [
    { label: 'Disk image (.dmg)', pattern: new RegExp(String.raw`^gaia-${VER}-x64\.dmg$`, 'i') },
  ],
  'linux-x64': [
    { label: 'Debian package (.deb)', pattern: new RegExp(String.raw`^gaia-${VER}-x64\.deb$`, 'i') },
    { label: 'AppImage', pattern: new RegExp(String.raw`^gaia-${VER}-x64\.AppImage$`, 'i') },
  ],
};

export interface ReleaseAssetLike {
  name: string;
  browser_download_url: string;
}

export interface InstallerDownload {
  label: string;
  name: string;
  url: string;
}

/**
 * The installers a release publishes for one platform, in preference order.
 *
 * Empty means "no installer for this visitor" — either the platform has no
 * installer lane, or the release predates the installer. The caller keeps the
 * raw-binary list in that case rather than rendering a dead button.
 */
export function findInstallers(
  platform: PlatformKey,
  assets: ReleaseAssetLike[],
): InstallerDownload[] {
  const specs = INSTALLER_ASSETS[platform as InstallerPlatform];
  if (!specs) return [];
  const found: InstallerDownload[] = [];
  for (const spec of specs) {
    const asset = assets.find((a) => spec.pattern.test(a.name));
    if (asset) {
      found.push({ label: spec.label, name: asset.name, url: asset.browser_download_url });
    }
  }
  return found;
}
