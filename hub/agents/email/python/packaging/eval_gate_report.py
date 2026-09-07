# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Email-triage eval gate-reader (report mode) — CI helper for test_email_agent_eval.yml.

Reads the gate bars ONLY through the harness loaders (the single committed
thresholds source under tests/fixtures/email/ — no thresholds inlined here),
runs the benchmark's own gate machinery, logs the result, and writes
``eval-out/gate_report.json``. Exits non-zero ONLY if a gate's ``should_fail``
is true (= its manifest has ``enforce: true`` AND it breached). In report mode
(``enforce: false``) it never fails the build.

Config comes from the environment so the workflow step stays shell-agnostic:
  EMAIL_EVAL_MODEL        Lemonade model id (required)
  EMAIL_EVAL_LIMIT        max messages to triage (default 50)
  EMAIL_EVAL_EXPERIMENTS  repeat count for variance (default 1)
  EMAIL_EVAL_IMPL         which triage implementation to score: "python"
                          (default, the frozen sidecar agent) or "node" (the
                          SDK in hub/agents/email/node). Each implementation
                          reads its OWN committed threshold manifests, so one
                          implementation's bars can never gate another's.
  EMAIL_EVAL_NODE_DIR     path to the Node SDK (default: the repo checkout);
                          only read when EMAIL_EVAL_IMPL=node
  EMAIL_EVAL_FORCE_LLM    "1" to disable the Node SDK's heuristic short-circuit
                          so every message goes through the LLM classifier;
                          only read when EMAIL_EVAL_IMPL=node

Extracted verbatim from the former inline ``python - <<'PY'`` step so the eval
can run on the Windows ``stx`` runner pool (PowerShell, no heredocs).
"""

import json
import os
import sys
from pathlib import Path

from gaia.eval.benchmark import (
    default_perf_thresholds_path,
    default_quality_thresholds_path,
    load_default_perf_thresholds,
    load_default_quality_thresholds,
    load_ground_truth,
    run_benchmark,
    summarize_benchmark,
)

MBOX = "tests/fixtures/email/synthetic_inbox.mbox"
GROUND_TRUTH = "tests/fixtures/email/ground_truth.json"


def _agent_factory(impl: str, model: str, limit: int):
    """Build the harness's ``agent_factory`` for ``impl``.

    ``None`` keeps ``run_benchmark`` on its built-in Python path; anything else
    is an explicit construction the harness drives instead.
    """
    if impl == "python":
        return None
    if impl != "node":
        raise ValueError(
            f"EMAIL_EVAL_IMPL={impl!r} is not a known implementation; "
            'expected "python" or "node".'
        )
    from gaia.eval.node_email_adapter import make_node_agent_factory
    from gaia.llm.lemonade_client import _get_lemonade_config

    base_url = _get_lemonade_config()[2]
    node_dir = os.environ.get("EMAIL_EVAL_NODE_DIR") or None
    force_llm = os.environ.get("EMAIL_EVAL_FORCE_LLM", "") == "1"
    print(
        f"[IMPL] node SDK at {node_dir or 'hub/agents/email/node'} -> "
        f"{base_url} (force_llm_classify={force_llm})"
    )
    return make_node_agent_factory(
        mbox_path=MBOX,
        model_id=model,
        base_url=base_url,
        limit=limit,
        node_dir=node_dir,
        force_llm_classify=force_llm,
    )


def main() -> int:
    model = os.environ["EMAIL_EVAL_MODEL"]
    limit = int(os.environ.get("EMAIL_EVAL_LIMIT", "50"))
    experiments = int(os.environ.get("EMAIL_EVAL_EXPERIMENTS", "1"))
    impl = os.environ.get("EMAIL_EVAL_IMPL", "python")

    quality_thresholds = load_default_quality_thresholds(impl)
    perf_thresholds = load_default_perf_thresholds(impl)
    print(f"[GATE] implementation:   {impl}")
    print(f"[GATE] quality manifest: {default_quality_thresholds_path(impl)}")
    print(f"[GATE] perf manifest:    {default_perf_thresholds_path(impl)}")
    print(
        f"[GATE] quality enforce={quality_thresholds.enforce} "
        f"fp_max={quality_thresholds.fp_max} fn_max={quality_thresholds.fn_max} "
        f"axis={quality_thresholds.axis}"
    )
    print(
        f"[GATE] perf enforce={perf_thresholds.enforce} "
        f"ttft_max_s={perf_thresholds.ttft_max_s} "
        f"throughput_min_tps={perf_thresholds.throughput_min_tps} "
        f"pipeline_max_s={perf_thresholds.pipeline_max_s} "
        f"peak_memory_max_gb={perf_thresholds.peak_memory_max_gb}"
    )

    ground_truth = load_ground_truth(GROUND_TRUTH)

    # Synthetic corpus only — FakeGmailBackend, never a live mailbox.
    results = run_benchmark(
        model,
        mbox_path=MBOX,
        limit=limit,
        experiments=experiments,
        ground_truth=ground_truth,
        agent_factory=_agent_factory(impl, model, limit),
    )
    summary = summarize_benchmark(
        results,
        run_id=f"email-eval-{impl}-{model.replace('/', '-').lower()}",
        thresholds=quality_thresholds,
        perf_thresholds=perf_thresholds,
    )

    quality_gate = summary.get("quality_gate", {})
    perf_gate = summary.get("perf_gate", {})
    quality = summary.get("quality", {})
    perf = summary.get("scorecard", {}).get("performance", {})

    # Reported-only accuracy numbers (gates pending #1266 / #1271 — no committed
    # accuracy-threshold manifest yet, so we log, not gate). Draft-approval
    # (#1269) is scored by the drafting eval step.
    cat_acc = quality.get("category_accuracy")
    phishing = quality.get("phishing", {})
    phishing_precision = (
        phishing.get("precision") if isinstance(phishing, dict) else None
    )

    print("\n================ EMAIL EVAL GATE REPORT (report mode) ================")
    print(f"  Categorization accuracy : {cat_acc}   (target >=0.85 #1266, REPORTED)")
    print(
        f"  Phishing precision      : {phishing_precision}   (target >=0.90 #1271, REPORTED)"
    )
    print(
        "  Draft-approval rate     : scored by the voice-drafting eval step (#1269, REPORTED)"
    )
    print("  ----------------------------------------------------------------")
    print(
        f"  Quality gate (FP/FN)    : passed={quality_gate.get('passed')} "
        f"breaches={len(quality_gate.get('breaches', []))} "
        f"enforce={quality_gate.get('enforce')} "
        f"should_fail={quality_gate.get('should_fail')}"
    )
    print(
        f"  Perf gate (Strix Halo)  : passed={perf_gate.get('passed')} "
        f"breaches={len(perf_gate.get('breaches', []))} "
        f"enforce={perf_gate.get('enforce')} "
        f"should_fail={perf_gate.get('should_fail')}"
    )
    print("====================================================================\n")

    report = {
        "model": model,
        "impl": impl,
        "limit": limit,
        "experiments": experiments,
        "corpus": MBOX,
        "reported": {
            "category_accuracy": cat_acc,
            "phishing_precision": phishing_precision,
        },
        "quality_gate": quality_gate,
        "perf_gate": perf_gate,
        "quality": quality,
        "performance": perf,
    }
    out = Path("eval-out")
    out.mkdir(parents=True, exist_ok=True)
    report_name = "gate_report.json" if impl == "python" else f"gate_report_{impl}.json"
    (out / report_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OUT] wrote eval-out/{report_name}")

    # The ONLY hook CI keys off. Report mode (enforce:false) -> always false ->
    # green. Flip enforce:true in the manifests to gate.
    failing = [
        name
        for name, gate in (("quality", quality_gate), ("perf", perf_gate))
        if gate.get("should_fail")
    ]
    if failing:
        print(f"[GATE] enforced gate breach: {failing} — failing the build.")
        return 1
    print("[GATE] report mode (or no breach) — gates do not block the build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
