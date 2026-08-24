// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"strconv"
	"strings"
)

// CompareSemver orders two versions: -1 if a < b, 0 if equal, 1 if a > b.
//
// Enough of semver for release ordering: numeric core compared field by field,
// and a pre-release sorting below the release it qualifies (1.0.0-rc.1 < 1.0.0).
// It mirrors workers/agent-hub/src/manifest.ts so the TUI and the hub never
// disagree about which published version is newest.
func CompareSemver(a, b string) int {
	aCore, aPre := splitPreRelease(a)
	bCore, bPre := splitPreRelease(b)

	aNums, bNums := numericFields(aCore), numericFields(bCore)
	for i := 0; i < len(aNums) || i < len(bNums); i++ {
		av, bv := 0, 0
		if i < len(aNums) {
			av = aNums[i]
		}
		if i < len(bNums) {
			bv = bNums[i]
		}
		if av != bv {
			if av < bv {
				return -1
			}
			return 1
		}
	}

	switch {
	case aPre == "" && bPre == "":
		return 0
	case aPre == "":
		return 1
	case bPre == "":
		return -1
	case aPre < bPre:
		return -1
	case aPre > bPre:
		return 1
	}
	return 0
}

// IsNewer reports whether candidate is strictly newer than current.
func IsNewer(candidate, current string) bool { return CompareSemver(candidate, current) > 0 }

func splitPreRelease(v string) (core, pre string) {
	v = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(v), "v"))
	// Build metadata never affects precedence.
	if i := strings.IndexByte(v, '+'); i >= 0 {
		v = v[:i]
	}
	core, pre, _ = strings.Cut(v, "-")
	return core, pre
}

func numericFields(core string) []int {
	parts := strings.Split(core, ".")
	nums := make([]int, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			// A non-numeric field ("dev", "unknown") sorts as 0 rather than
			// aborting: this compares release labels, and a build that reports
			// "dev" must still be told a published version exists.
			n = 0
		}
		nums = append(nums, n)
	}
	return nums
}
