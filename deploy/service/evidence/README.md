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
