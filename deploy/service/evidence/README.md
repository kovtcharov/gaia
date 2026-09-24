# Local qualification evidence

M4 macOS ARM64 qualification uses Docker Desktop with native Linux ARM64 workers.

- `m4-macos-arm64-reliability.json`: 1,000 sequential tasks and 600 seconds of saturation. The source/image digests are recorded in the report. This run preceded the final request-validation-only changes.
- `m4-macos-arm64-final-image.json`: rebuilt final candidate with the non-finite JSON/invalid Unicode fixes; real adversarial HTTP, backup/recovery, 10 tasks and 10 seconds of saturation.

The first endurance attempt failed around task 459; the runtime stopped without
orphans. Its root cause was not conclusively recovered. Safe failure diagnostics
were added, and the complete rerun passed. The final image has short real-runtime
verification; promotion still requires the release workflow's complete endurance
qualification on the exact release image and native architecture gates. These
reports do not establish a supported release, hardware matrix, or suspend/resume
qualification.


## M5 final candidate

`m5-macos-arm64-final-image.json` records the final frozen image with real external
tool profiles, 10 sequential tasks and 10 seconds of saturation. HTTP controls,
crash/replay, full clone restore and zero orphan executors also passed.
`m5-macos-arm64-rag.json` records five independent frozen processes against a real
prepared EmbeddingGemma model via Lemonade/Metal. It qualifies retrieval/index
lifecycle, not generated-answer quality or controller-mediated live RAG restoration.

A preceding concurrent local runtime attempt exceeded its 45-second first-run
deadline and closed admission after guardian communication/termination uncertainty.
Its cleanup removed all owned containers. A concurrent regression attempt had one
10-second child-process startup timeout (455 tests passed). Sequential reruns passed
all 456 tests and the real runtime checks without weakening deadlines. These failed
attempts remain qualification context; no claim of availability under arbitrary host
starvation or suspend is made. Native Linux internal-network tests and exact-release
endurance remain CI/promotion gates.
