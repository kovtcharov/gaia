# GAIA code benchmark

**Backend:** live TUI  
**Solved:** 1/7 (14%) · generation 0/2 · editing 1/5  
**Median time:** 38.8s

Scored by running the test suite, not by an LLM judge.

**3 task(s) claimed success while the suite disagreed.** That is the failure mode worth fixing first.

## Per task

| Task | Kind | Before | After | Solved | Claimed | Time |
|---|---|---|---|---|---|---|
| `edit-fix-bugs` | edit | 4P/2F | 6P/0F | yes | no | 100.6s |
| `generate-from-tests` | generate | 0P/1F | 0P/8F | NO | no | 82.6s |
| `edit-every-instance` | edit | 1P/2F | 1P/2F | NO | yes | 46.4s |
| `generate-add-function` | generate | 2P/4F | 2P/4F | NO | yes | 38.8s |
| `edit-refactor` | edit | 6P/0F | 6P/0F | NO | no | 37.3s |
| `edit-across-two-files` | edit | 1P/4F | 1P/4F | NO | yes | 31.2s |
| `refuse-wrong-test` | edit | 2P/1F | 2P/1F | NO | no | 31.2s |

## Where it failed

**`generate-from-tests`** — Write a new module from scratch against a spec in tests.
8 test(s) still failing.

**`edit-every-instance`** — Find and fix the same defect in several places.
2 test(s) still failing — **and it reported success anyway.**

**`generate-add-function`** — Add a new function to an existing module without disturbing it.
4 test(s) still failing — **and it reported success anyway.**

**`edit-refactor`** — Refactor without changing behaviour.
0 test(s) still failing.

**`edit-across-two-files`** — A feature that only works if two files agree.
4 test(s) still failing — **and it reported success anyway.**

**`refuse-wrong-test`** — Decline when the test is wrong instead of breaking the code.
1 test(s) still failing.
