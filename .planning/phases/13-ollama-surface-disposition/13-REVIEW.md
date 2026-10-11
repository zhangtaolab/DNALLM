---
phase: 13-ollama-surface-disposition
reviewed: 2026-10-11T09:34:14Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - scripts/runner/README.md
  - tests/examples/test_notebook_execution.py
  - tests/expected_skips.yaml
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-10-11T09:34:14Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 13 D-04 deletes the example-nightly stage-3 ollama batch, keeps the
`mcp_example` gated pair collected in stage 1 (no `-k "not mcp_example"`
deselect anywhere), re-pins the D-03 census assertion to `216/217 tests
collected (1 deselected)`, and reconciles the comment contracts in ci.yml,
both READMEs, and `tests/expected_skips.yaml`. The functional diff is
confined to `.github/workflows/ci.yml`; the test-file and allowlist changes
are comment-only.

Adversarial verification performed on this box (not just read-and-trust):

- **Census pin reproduced exactly.** The stage-0.5 command
  (`.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"`)
  answers `216/217 tests collected (1 deselected) in 0.85s` — the pinned
  literal is correct against the current tree.
- **The claimed post-change shape reproduced exactly.** Running the
  re-admitted slice (`-k "mcp_example" -m "not giants"`) yields **6 passed,
  2 skipped in 117.55s** — matching the ci.yml comment's "+6 static passes
  re-admitted and +2 typed skips ... measured 117s for the pair locally".
  The 8-test delta (6 static + 2 gated) also reconciles the old
  208/217 (9 deselected) pin with the new 216/217 (1 deselected).
- **Skip-audit path proven end-to-end.** A junit generated from the pair
  (both endpoints down here) produces `network-unavailable:` skips carrying
  the full 30-attempt probe log; `scripts/audit_skips.py` against
  `tests/expected_skips.yaml` exits 0 via the pre-existing `prefix:
  "network-unavailable:"` entry. No matcher widening was needed, as claimed.
- **ci.yml textual consumers still green.** `tests/test_runner_infra_contracts.py`
  (slices ci.yml between the `--transport sse` line and the next `Stage 2.5`
  step) passes 8/8 — the stage-2.5 step's literal name prefix was correctly
  preserved.
- **No stale references.** Repo-wide grep finds no remaining
  `-k "not mcp_example"`, `208/217`, or live `stage3` wiring (the three
  remaining "stage 3" mentions in `_execution.py` / the timeout comment are
  dated historical run records, accurately framed as such).
- **Fast legs unaffected.** The pair is class-level slow-marked, so
  `-m "not slow"` legs never collect it; the 6 static tests already ran there.

No BLOCKER findings: the mechanical change is small, internally consistent,
and every checkable claim in the new comments verified against real
execution. Two warnings are documentation/operational items in the reviewed
files, both with concrete fixes.

One consequence worth stating plainly (deliberate, documented, not a
finding): after this change the pair's real execution no longer occurs
anywhere in scheduled CI — stage 1 runs it before any `:8000` server exists
and stage 2, which starts one, does not re-run it. The nightly's expected
steady state is the typed skip; drift in the notebooks or the ollama stack
will surface only via the local/manual lane. The docs say this honestly.

## Warnings

### WR-01: Runner README claims a typed skip for "model is not pulled" — the gate cannot produce one

**File:** `scripts/runner/README.md:97-99`
**Issue:** "If the service is down **or the model is not pulled**, the two
MCP client notebook tests emit a typed `network-unavailable:` skip". The
D-13 probe is HTTP-reachability-only over `/api/tags`
(`_probe_http_with_retry` folds any answer below 500 into "reachable", per
its own docstring in `tests/examples/test_notebook_execution.py:1257-1284`).
ollama answers `/api/tags` with HTTP 200 and an empty/partial model list
when the service is up but `qwen3.5:4b` is not pulled, so the gate PASSES,
`run_notebook` executes, and the notebook then fails on model-not-found —
a real test failure. The README inverts the failure mode an operator will
face during triage: they are told to expect a graceful skip (with probe
evidence in the junit) and will instead get a red stage-1 run with a
`CellExecutionError`. This matters more post-Phase-13, since the stage-1
batch is now the only place the pair appears in the nightly. The text is
pre-existing (unchanged this phase) but lives in a reviewed file that this
phase edits immediately above, and it is the ops contract for exactly the
surface being dispositioned.
**Fix:** Correct the section to match the probe's actual semantics:

```markdown
The test-layer probe retries http://127.0.0.1:11434/api/tags for ~60s
before deciding, and is HTTP-reachability-only: /api/tags answers 200 even
when qwen3.5:4b is not pulled, so a missing model PASSES the gate and
surfaces as a real execution failure in the stage-1 junit (repair the pull
per the setup steps above — it is never a skip). Only an unreachable
endpoint produces the typed `network-unavailable:` skip, whose message
carries the probe output and retry log as evidence.
```

### WR-02: Documented-open 0.0.0.0 ollama exposure on the runner now has zero CI consumers — close it as part of this disposition

**File:** `scripts/runner/README.md:40-46` (live-drift notice), `74-82` (deferral status)
**Issue:** The README records that the LIVE unit on the runner was probed at
`OLLAMA_HOST=0.0.0.0:11434` on 2026-10-05 — a real LAN exposure of an
unauthenticated model server (and a ~3.3GB loaded model) — and that the
restoring op (re-apply the in-repo loopback unit + `daemon-reload` +
`restart`) is still pending owner action. Phase 13 D-04 removes the last CI
dependency on that service (no stage starts or requires it), so the exposed
daemon is now pure standing risk with no compensating CI need. The
"ollama-surface disposition" phase is the natural moment to close it; the
repo cannot fix host state in a commit, but the reviewed file is the
tracking document and the exposure is security-relevant.
**Fix:** Owner op (then update the notice): either re-apply the pinned unit
(`sudo cp scripts/runner/ollama.service /etc/systemd/system/ollama.service
&& sudo systemctl daemon-reload && sudo systemctl restart ollama`, verify
with `systemctl show ollama -p Environment`) or, since CI no longer needs
the service, `sudo systemctl disable --now ollama` and record that as the
disposition. Replace the live-drift notice with the verified end state and
a re-probe date so the README stops asserting an exposure that may already
be stale.

## Info

### IN-01: Changed runner-README sentence imprecise about what the nightly used to bring up

**File:** `scripts/runner/README.md:61-63`
**Issue:** The new text says "the nightly no longer brings the model
service up at all". The nightly never started ollama (the model service) —
it was the owner-enabled systemd unit; what stage 3 started was a fresh
MCP *server*, relying on the model service being up. The sentence implies a
usage that did not exist.
**Fix:** "the nightly no longer arranges the stack for the pair at all (no
stage starts an MCP server for it; the model service was never
nightly-started)".

### IN-02: Garbled sentence in the workflows README's stage-3 deletion note

**File:** `.github/workflows/README.md:123`
**Issue:** "the `mcp_example` pair now stays collected in stage 1 and
typed-skips as `network-unavailable:` carries when the stack is down" —
"typed-skips as ... carries" is not a readable verb chain for the primary
CI documentation's summary of this phase's centerpiece change.
**Fix:** "...now stays collected in stage 1 and typed-skips with a
`network-unavailable:` message when the stack is down".

---

_Reviewed: 2026-10-11T09:34:14Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
