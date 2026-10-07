---
phase: 09-ci-wiring-census-verification
plan: 02
subsystem: testing
tags: [runtime-cuts, num-ctx, ollama, systemd, seed-sandbox, yaml-patch, epochs-cut, census, d-03-pin]

requires:
  - phase: 09-01
    provides: giants exit topology + the D-03 census literal 188/197 (9 deselected) this plan bumps
provides:
  - seed_sandbox(src_dir, tmp_path, extra_inputs=None, yaml_overrides=None) — sandbox-only YAML patch seam (D-05): fail-closed ValueError on missing target file / missing section; sandbox copy only, committed content untouched
  - NOTEBOOK_EXEC_SPECS yaml_patch spec key (finetune_custom_head: finetune_config.yaml num_train_epochs 3->1) consumed by the gated_sandbox fixture exactly like the spec env key
  - TestSeedSandboxYamlOverrides — 5 kernel-free contract tests incl. the source-still-reads-3 honesty pin; grows tests/examples 197 -> 202
  - ci.yml D-03 triple bumped 188/197 -> 193/202 tests collected (9 deselected; deselect counts unchanged at 8 mcp + 1 giants) in the SAME commit that grows the census — the designed bump-point in action
  - OLLAMA_CONTEXT_LENGTH=8192 Environment pin in scripts/runner/ollama.service beside the byte-preserved D-12 loopback pin (D-06) + header precedence record (per-request > Modelfile > env > default; qwen3.8:latest has no Modelfile num_ctx)
  - scripts/runner/README.md — Why num_ctx 8192 (D-06) section + owner re-apply step (cp + daemon-reload + restart ollama + read-only systemctl verify) + explicit live 0.0.0.0 drift notice (T-09-03)
  - tests/test_runner_infra_contracts.py — fast-leg contract file pinning both unit Environment lines and the README re-apply op (D-07)
  - 09-USER-SETUP.md — the owner sudo op that makes the pin live AND restores the drifted loopback bind (PENDING until executed; 09-04 Task 2 asserts the read-only evidence)
affects: [09-ci-wiring-census-verification (09-04 measures both cuts' effects in the D-02 baseline and gates on the ollama re-apply precondition), v1.1 ship]

actuals:
  tokens: 5240     # chars/4 over the realized diff 3557e0b..e85eb73 (estimate was 28000 — plan overestimated)
  tasks: 2
  commits: 4       # MEASURED: git rev-list --count 3557e0b..HEAD (2 RED + 2 GREEN)
plan_head_before: 3557e0b9e363194eb38c9564e22b89ae0eec46db
plan_head_after: e85eb73b42eea52df9cf1090fdd530a4846a72cb

tech-stack:
  added: []         # PyYAML is an existing core dependency (audit_skips.py precedent); test/infra/docs edits only (T-09-SC)
  patterns:
    - "Sandbox-only runtime cut: spec yaml_patch key consumed via spec.get() at the fixture, applied inside seed_sandbox after the copytree — one implementation point, kernel-free unit-testable, committed content never touched"
    - "Census-growth arithmetic owned in-commit: the commit adding collected tests also bumps BOTH D-03 number carriers (grep pattern + plain-text FAIL echo) from a fresh measurement with the exact stage-1 selector flags"
    - "Unit-file contract pin: fast file-content assertions keep infra pins honest between owner re-applies (fail in seconds, not at the next 25-min execution)"

key-files:
  created:
    - tests/test_runner_infra_contracts.py
  modified:
    - tests/examples/_execution.py                 # import yaml + seed_sandbox yaml_overrides + spec yaml_patch key
    - tests/examples/test_notebook_execution.py    # TestSeedSandboxYamlOverrides + gated fixture forwarding
    - .github/workflows/ci.yml                     # D-03 triple 193/202 (both carriers)
    - scripts/runner/ollama.service                # OLLAMA_CONTEXT_LENGTH pin + header precedence record
    - scripts/runner/README.md                     # re-apply op + Why num_ctx 8192 + drift notice

key-decisions:
  - "D-05 landed at BOTH seams the research offered: seed_sandbox owns application (one fail-closed implementation point) while the SPEC owns the value (yaml_patch key) — fixture forwards spec.get('yaml_patch') like spec.get('env'); no other gated notebook changes behavior (exactly 1 spec carries the key)"
  - "Census bump landed in the RED commit (696937b), the commit whose collected tests cause the growth — not in GREEN — so no commit ever exists where the census and the D-03 pin disagree"
  - "The pre-existing fast-lane failure (test_plot_for_regression) was proven pre-existing by re-running the full fast lane at plan-start HEAD in an isolated worktree (1825P/1F/1S -> 1833P/1F/1S = exactly +8 passed) instead of being silently absorbed or auto-fixed out of scope"
  - "The stale-remote dispatch (37333370370 at e9056c2) was cancelled and re-dispatched (37335797121 at e85eb73b) after pushing phs — a dispatch that runs 12-commits-stale code exercises nothing from this phase"

patterns-established:
  - "Delta-proof against a dirty baseline: when a plan verify greps a whole-lane summary, isolate plan-caused drift by running the same command at plan_head_before in a throwaway worktree"
  - "RED commit may carry the census-triple bump alongside the failing tests — collection counts are commitment of the test FILE, not of its greenness"

requirements-completed: [CI-03, CI-06]

coverage:
  - id: D1
    description: "seed_sandbox yaml_overrides seam: sandbox copy patched to 1 / source still 3, fail-closed missing file + missing section, default no-op byte-identity (D-05/D-07)"
    requirement: CI-06
    verification:
      - kind: unit
        ref: "tests/examples/test_notebook_execution.py::TestSeedSandboxYamlOverrides (5/5 passed; -k 'YamlOverride or yaml_patch')"
        status: pass
    human_judgment: false
  - id: D2
    description: "NOTEBOOK_EXEC_SPECS yaml_patch pin + spec-driven fixture forwarding; no other gated notebook's seeding changes (exactly 1 spec carries the key)"
    requirement: CI-06
    verification:
      - kind: unit
        ref: "test_finetune_custom_head_spec_pins_the_epochs_cut + live spec scan (1/21 specs carry yaml_patch)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Census arithmetic owned in-commit: tests/examples 197 -> 202 collected; stage-1 triple 193/202 (9 deselected, deselect counts unchanged); both ci.yml carriers bumped in the growth commit"
    requirement: CI-03
    verification:
      - kind: unit
        ref: "Task 1 verify CUT-OK: fresh collect-only measurements 202 / 193-202(9) + grep -qF of the measured triple in ci.yml; zero diff under example/; fast examples subset 171P/1S (pre-existing benign skip only)"
        status: pass
    human_judgment: false
  - id: D4
    description: "ollama unit pins both live in-repo: OLLAMA_CONTEXT_LENGTH=8192 added, OLLAMA_HOST=127.0.0.1:11434 byte-preserved (loopback line + comment untouched, exactly one new Environment line)"
    requirement: CI-06
    verification:
      - kind: unit
        ref: "tests/test_runner_infra_contracts.py (3/3 passed) + git diff purity check (additions only)"
        status: pass
    human_judgment: false
  - id: D5
    description: "README owner re-apply op (daemon-reload + restart ollama + read-only systemctl verify), Why num_ctx 8192 section, and the 0.0.0.0 live-drift notice; no 0.0.0.0 suggested anywhere"
    requirement: CI-06
    verification:
      - kind: unit
        ref: "TestRunnerReadmeReapply contract test + Task 2 verify greps (unit pins, loopback pin, daemon-reload)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Owner sudo op delivered (09-USER-SETUP.md): re-apply unit + restart makes the 8k default live AND restores the drifted live loopback bind (T-09-03); PENDING until executed"
    verification: []
    human_judgment: true
    rationale: "The live systemd re-apply requires owner sudo Claude does not have; the in-repo artifacts are machine-verified (D4/D5) but the live service state (systemctl show ollama -p Environment showing both pins, qwen3.8 served at num_ctx 8192 on loopback only) is verifiable only after the owner runs the op — 09-04 Task 2 asserts that read-only evidence as a precondition."
  - id: D7
    description: "D-16 incremental dispatch of example-nightly from fresh phs so the bumped D-03 triple is exercised before 09-04's baseline"
    verification: []
    human_judgment: true
    rationale: "Dispatch SUCCEEDED after the stale-remote correction: run 37335797121 queued at exactly e85eb73b (HEAD; first run exercising the 193/202 pin and the D-05 sandbox patch on the runner). The multi-hour outcome is deliberately NOT a plan-verify gate (never per plan); the green-run gate is D-17 in 09-04. Interim: the first dispatch attempt (37333370370) had silently run 12-commits-stale remote phs (e9056c2) — cancelled; phs then pushed and re-dispatched."

duration: 38min
completed: 2026-10-05
status: complete
---

# Phase 9 Plan 02: Runtime Cuts (D-05 epochs / D-06 num_ctx) Summary

**Both owner-approved nightly runtime cuts landed at their honest seams — finetune_custom_head epochs 3->1 as a spec-driven seed_sandbox YAML patch (sandbox copy only) and the mcp pair's num_ctx 256k->8k as the OLLAMA_CONTEXT_LENGTH=8192 server-default pin in the in-repo ollama unit — each with same-change contract tests (D-07), committed example content byte-identical, and the D-03 census triple re-measured and bumped to 193/202 in the same commit that grows the census.**

## Performance

- **Duration:** 38 min
- **Started:** 2026-10-05T15:17:19Z
- **Completed:** 2026-10-05T15:55:00Z
- **Tasks:** 2/2
- **Files modified:** 5 (+1 new test file +2 plan artifacts)

## TDD Evidence (both tasks tdd="true")

**Task 1 (D-05) — RED:** `TestSeedSandboxYamlOverrides` written first (5 kernel-free tests, `_seed_dir` fake-dir idiom). Run against pre-change `seed_sandbox`: **4 failed, 1 passed** — tests 1-3 fail with `TypeError: seed_sandbox() got an unexpected keyword argument 'yaml_overrides'` (test-level failure, the planned behavior gap), test 4 fails the assertion (`spec.get("yaml_patch") is None`), test 5 (default no-op regression pin) passes by design on pre-change behavior. Classifier: `RED_EVIDENCE_OK` (JUnit-XML record, target `tests.examples.test_notebook_execution.TestSeedSandboxYamlOverrides#test_override_patches_sandbox_copy_only`; record at `.scratch/tdd-red-0902/record-t1.json`). Semantic assessment: the target executed and failed on the planned missing feature, not a load/fixture fault. **GREEN:** implementation landed; 5/5 pass.

**Task 2 (D-06) — RED:** `tests/test_runner_infra_contracts.py` written first (3 tests, extras-guard structure). Run against the un-edited unit/README: **2 failed, 1 passed** — D-06 pin absent (assertion), README re-apply/rationale absent (assertion); the loopback-preservation test passes by design (the pin exists and must survive). Classifier: `RED_EVIDENCE_OK` (target `...TestOllamaUnitPins#test_unit_pins_context_length_8192`; record at `.scratch/tdd-red-0902/record-t2.json`). **GREEN:** unit + README edited; 3/3 pass.

## Accomplishments

- **D-05 epochs cut:** `seed_sandbox` gained `yaml_overrides` (applied after the extra-inputs loop; `yaml.safe_load`/`safe_dump`; fail-closed `ValueError` naming the missing target file or section). The `finetune_custom_head` spec carries `yaml_patch = {"finetune_config.yaml": {"finetune": {"num_train_epochs": 1}}}` (both `trainer.train` calls read this single load — RESEARCH A1), and `gated_sandbox` forwards it `spec.get()`-style exactly like the spec env key. The committed notebook + YAML stay byte-identical (`git status example/` clean; line 38 still reads `num_train_epochs: 3`), so the census executability claim keeps resting on committed content; the loop body is identical, only the epoch count changes.
- **D-03 census arithmetic owned in-commit (CI-03):** the 5 contract tests grow tests/examples **197 -> 202** collected; the stage-1 selector triple re-measured with the exact flags (`-m "not giants" -k "not mcp_example"`) to **193/202 tests collected (9 deselected)** — deselect counts unchanged (8 mcp + 1 giants), zero new skip messages. BOTH ci.yml Stage 0.5 carriers (grep pattern + plain-text FAIL echo) bumped inside the growth commit itself. Expected effect (~31 -> ~11 min) is 09-04's to measure, not asserted here.
- **D-06 num_ctx cut:** `Environment="OLLAMA_CONTEXT_LENGTH=8192"` added directly after the byte-preserved D-12 loopback pin (one new Environment line; server default covers BOTH mcp client stacks — the pydantic_ai sibling talks OpenAI-compat `/v1` with no per-request context parameter). The unit header records the precedence fact (per-request > Modelfile > env > default; `qwen3.8:latest` has NO Modelfile `num_ctx`, probed live 2026-10-05 — re-probe if re-pulled) and the README carries the owner re-apply op plus the explicit live-drift notice (LIVE unit probed at `OLLAMA_HOST=0.0.0.0:11434`, 2026-10-05 — the re-apply restores loopback and closes T-09-03). No cache cleanup anywhere (owner rule); no mypy/ty config touched (D-09 boundary held).
- **D-16 dispatch (best-effort, never a gate):** first dispatch fired instantly but was discovered running **stale remote phs** (`e9056c2`, 12 commits behind — 09-01/09-03 commits were also unpushed), so it exercised nothing from this phase; it was cancelled. `phs` was pushed (owner default; one SSL-timeout retry through the flaky github.com egress, gh API route unaffected), then re-dispatched: **run 37335797121 queued at exactly `e85eb73b` (HEAD)** — the first run exercising the bumped 193/202 pin and the D-05 sandbox patch. Outcome observation belongs to 09-04/D-17.

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1 RED: D-05 contract tests + census triple bump** - `696937b` (test)
2. **Task 1 GREEN: implement the sandbox-only YAML patch seam** - `2c67be6` (feat)
3. **Task 2 RED: D-06/D-12 runner-infra contract tests** - `5541f9c` (test; amended once for a ruff parenthesization fix before anything built on it)
4. **Task 2 GREEN: num_ctx 8192 pin + README re-apply/drift docs** - `e85eb73` (feat)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified

- `tests/examples/_execution.py` — `import yaml`; `seed_sandbox(..., yaml_overrides: dict[str, dict] | None = None)` with the D-05 docstring contract; `yaml_patch` spec key on the finetune_custom_head entry
- `tests/examples/test_notebook_execution.py` — `TestSeedSandboxYamlOverrides` (5 tests) beside `TestSeedSandbox`; `gated_sandbox` fixture forwards `spec.get("yaml_patch")`
- `.github/workflows/ci.yml` — Stage 0.5 D-03 triple `193/202 tests collected (9 deselected)` in both number carriers
- `scripts/runner/ollama.service` — `OLLAMA_CONTEXT_LENGTH=8192` + D-06 comment + RUNTIME precedence header block
- `scripts/runner/README.md` — step-5 re-apply op, live-drift notice, "Why num_ctx 8192 (D-06)" section; stale "two Environment lines" comment corrected
- `tests/test_runner_infra_contracts.py` — NEW fast-leg contract file (provenance docstring, REPO_ROOT anchor, private loaders, 2 Test classes)
- `.planning/phases/09-.../09-USER-SETUP.md` — owner sudo op + read-only verification
- `.planning/phases/09-.../deferred-items.md` — pre-existing benchmark failure record

## Decisions Made

- The census bump lives in the RED commit (the commit whose collected tests cause the growth), keeping every commit census-consistent.
- The pre-existing fast-lane failure was delta-proven (worktree run at plan-start HEAD: `1825P/1F/1S` vs now `1833P/1F/1S` = exactly +8 passed, +0 failed, +0 skipped) rather than fixed out of scope or hand-waved.
- The stale-ref dispatch was cancelled instead of left to burn hours of the single queue-serialized runner on 12-commits-old code.

## Deviations from Plan

**1. [Rule 1 - verify-command defect] Task 1 verify extraction misses pytest's `=`-padded summary line**
- **Found during:** Task 1 verify
- **Issue:** The plan's `sed -E 's/ in [0-9.]+s?$//'` assumes the bare `-q` summary; addopts `-v` cancels CLI `-q` so pytest prints `==== 193/202 tests collected (9 deselected) in 0.88s ====` and the anchored greps false-negative against a correct implementation. Identical to 09-01 deviation #1.
- **Fix:** Extended the extraction sed to strip `^=+ *` / ` *=+$` padding before the time strip (verification-only; the ci.yml step itself was already padding-tolerant).
- **Files modified:** none
- **Verification:** CUT-OK with the corrected extraction
- **Commit:** n/a (no code change)

**2. [Rule 3 - environment] Task 2 fast-lane verify grep confounded by a PRE-EXISTING failure**
- **Found during:** Task 2 verify
- **Issue:** `.venv/bin/python -m pytest tests/ -m "not slow" -q` ends `1 failed, 1833 passed, 1 skipped` — `tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression` (pandas `TypeError: float() argument ... not 'dict'` in the plot path). Reproduced at plan-start HEAD `3557e0b` in an isolated worktree (same venv): `1 failed, 1825 passed, 1 skipped` — pre-existing, not caused by any Phase 09 change (this plan's delta is exactly +8 passed / +0 failed / +0 skipped). Likely quick-task 13/14 Mapping fallout (86022f7/16a9ffb).
- **Fix:** Intent of the gate proven by the delta measurement; failure logged to `deferred-items.md` and the WINDOWS.md ledger (id 16, kind unmet-truth) for owner triage. Not auto-fixed (out of scope).
- **Files modified:** `.planning/phases/09-.../deferred-items.md`, `.planning/WINDOWS.md`
- **Verification:** worktree baseline run + full fast lane at HEAD
- **Commit:** docs (this commit)

**3. [Rule 1 - lint] ruff `parenthesize-chained-operators` slipped into the first Task 2 RED commit**
- **Found during:** Task 2 RED commit
- **Issue:** No pre-commit hooks are installed in this clone and the compound lint-then-commit command sequenced past a failing `ruff check`; the committed test carried an unparenthesized `a or b and c` chain.
- **Fix:** Parenthesized the subexpression, re-confirmed RED (2 failed, 1 passed), amended the RED commit → `5541f9c` (lint-clean). Subsequent commits ran ruff as a standalone gated step.
- **Files modified:** `tests/test_runner_infra_contracts.py`
- **Verification:** `ruff format --check` + `ruff check` clean; RED still held post-fix
- **Commit:** `5541f9c` (amend)

---

**Total deviations:** 3 auto-fixed (1 x Rule 1 verify-defect, 1 x Rule 3 environment with out-of-scope disposition, 1 x Rule 1 lint amend)
**Impact on plan:** None on delivered behavior — all plan verifies pass in intent (two with corrected instrumentation), every acceptance criterion PASS, and the pre-existing failure is now visible in the deferred + windows ledgers instead of silently absorbing into this plan's red/green.

## Issues Encountered

- **Stale-remote dispatch (resolved):** the first D-16 dispatch (37333370370) ran remote phs at `e9056c2` — 12 commits behind local (09-01's and 09-03's commits were also unpushed, so its step list still showed the pre-surgery evo provisioning). Cancelled; phs pushed (one SSL-timeout retry through the flaky github.com egress); re-dispatched as 37335797121 at `e85eb73b`. Lesson recorded for 09-04: push before dispatch.
- **Egress flakiness (resolved):** `git` HTTPS to github.com timed out while the `gh` API route worked (the documented split); a bounded retry loop landed the push on attempt 1 after recovery.
- **Pre-existing benchmark fast-lane failure** — see Deviation 2; open for owner triage.

## Authentication Gates

None.

## User Setup Required

**PENDING (see `.planning/phases/09-ci-wiring-census-verification/09-USER-SETUP.md`):** the owner must re-apply the in-repo ollama unit on the runner host (`sudo cp scripts/runner/ollama.service /etc/systemd/system/ollama.service && sudo systemctl daemon-reload && sudo systemctl restart ollama`) — the num_ctx 8192 default is read at server START, and the SAME op restores the live `OLLAMA_HOST=0.0.0.0:11434` drift to the loopback pin (T-09-03). Read-only verify: `systemctl show ollama -p Environment` must show both pins; `curl -s http://127.0.0.1:11434/api/tags` must list qwen3.8:latest. Plan 09-04 Task 2 asserts this evidence before the baseline dispatch.

## Known Stubs

None — no stubs, placeholders, or unwired data paths were introduced.

## Next Phase Readiness

- Ready for 09-04 (Wave 3): run **37335797121** queued at `e85eb73b` is the first live exercise of the bumped 193/202 D-03 pin and the D-05 sandbox patch; its Stage 0.5 verdict + full outcome feed the D-02 baseline / D-17 green-run gate. 09-04 should also record the pre-existing fast-lane failure disposition (WINDOWS id 16) and the still-open D-14 item (census-collect.txt upload path).
- The owner re-apply (user_setup) is the declared precondition for 09-04 Task 2's baseline dispatch — without it the num_ctx cut is in-repo but not live, and the loopback exposure stays open.
- Expected effects for 09-04 to measure (not asserted here): finetune_custom_head stage ~31 -> ~11 min; mcp-pair latency and the VRAM trough shrink materially.

## Self-Check: PASSED

- Files: all 5 modified + 1 created source files present; SUMMARY/USER-SETUP/deferred-items written
- Commits: 696937b, 2c67be6, 5541f9c, e85eb73 — all ancestors of HEAD
- Protected paths: `example/` and `tests/expected_skips.yaml` byte-identical to plan start; no cache cleanup anywhere

---
*Phase: 09-ci-wiring-census-verification*
*Completed: 2026-10-05*
