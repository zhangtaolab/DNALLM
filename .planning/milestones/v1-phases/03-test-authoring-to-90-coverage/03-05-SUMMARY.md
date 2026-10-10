---
phase: 03-test-authoring-to-90-coverage
plan: 05
subsystem: testing
tags: [pytest, coverage, click-clirunner, transformers-compat, bitsandbytes, logging, cli, final-gate]

requires:
  - phase: 03-test-authoring-to-90-coverage
    provides: wave-4 datahandling/finetune closeout (10/100), suite at 91.24%, coverage-wave4-missing.txt ranked worklist
  - phase: 01-harness-integrity-measured-baseline
    provides: measured baseline + coverage tooling of record + the census command shape
provides:
  - FINAL GATE PASS at 96.28% total line coverage (7,124/7,399 statements) — strictly above the >90.5% milestone target, reproducible with one config-only command
  - tests/utils/test_transformers_compat.py — the phase's named behavior-contract requirement proven on the live patched transformers class (TEST-05)
  - tests/cli/test_cli.py — every dnallm CLI entry point exercised in-process via CliRunner with lazily-imported cores patched at their binding sites (TEST-05)
  - The 60-line orphans closed to zero (tasks/metrics.py 31→0, configuration/configs.py 9→0) plus a 17-line straggler pass (support, dnabert2, special registries, head)
  - coverage-wave5-missing.txt — the final term-missing snapshot and accepted-uncovered residual ledger (Phase 4's ratchet-floor input)
affects: [04-coverage-gate-ci]

actuals:
  tokens: 21554  # chars/4 over the realized tests diff (86,214 chars); estimate was 50,000 (confidence: low)
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Live-class behavior contract: assert against the already-patched transformers.PreTrainedModel (object identity across re-apply, proxy semantics, swap/restore flow with bitsandbytes.functional monkeypatched) — never unpatch the class; the transformers-version guard arms are covered by monkeypatching transformers.modeling_utils.PreTrainedModel with a bare class so the function-local import binds the fake"
    - "CliRunner + lazy-import patch sites: `from ..finetune import DNATrainer` inside a command body resolves at call time against the ORIGIN package attribute — patch dnallm.finetune.DNATrainer / dnallm.inference.DNAInference / dnallm.mcp.server.main and assert call args; argv-assembling commands record sys.argv inside the mock and assert restoration"
    - "Network-free evaluate seams: metrics_for_dnabert2's bare metric names (evaluate.load('r_squared')) would hit the HF hub — patch evaluate.load/combine with shaped fakes keyed by name"
    - "Mock equality on tensor args is ambiguous (element-wise bool) — assert call_args fields by identity/torch.equal instead of assert_called_with"

key-files:
  created:
    - tests/utils/test_transformers_compat.py
    - tests/utils/test_logger.py
    - tests/utils/test_support.py
    - tests/cli/test_cli.py
    - .planning/phases/03-test-authoring-to-90-coverage/coverage-wave5-missing.txt
  modified:
    - tests/utils/test_cuda_compat.py
    - tests/utils/test_sequence.py
    - tests/utils/test_training_plots.py
    - tests/tasks/test_metrics.py
    - tests/configuration/test_configs.py
    - tests/models/test_head.py
    - tests/models/test_special/test_family_handlers.py

key-decisions:
  - "Compat guard arms covered without touching the live class: monkeypatch transformers.modeling_utils.PreTrainedModel with a bare class so the shim's function-local `from transformers.modeling_utils import PreTrainedModel` binds the fake — covers the method-absent early returns that would otherwise require destructive unpatching (T-3-14 mitigation)"
  - "The logger module is stdlib logging + colorama, not loguru as the plan text assumed — per-test hygiene implemented as stdlib handler add/remove on fresh loggers plus monkeypatch.chdir(tmp_path) for the logs/dnallm.log CWD sink (FIX-04 discipline)"
  - "metrics_for_dnabert2 covered via patched evaluate.load/combine because its bare metric names would resolve against the HF hub; the arm behavior (dict composition, argmax predictions, ovr/ovo AUROC kwargs) is asserted through the mocks"
  - "CLI tests use loopback hosts only (127.0.0.1) after ruff's hardcoded-bind-all-interfaces flagged 0.0.0.0; the bare-group no-args exit code is asserted leniently (0 or 2) because click versions differ — the help content is the pinned behavior"
  - "Task 3's straggler pass closed 17 lines beyond the plan's file list (utils/support.py 7, dnabert2 triton probe 4, lucaone/omnidna/space extra-append arms 3, head.py 3) — every one an already-loaded-context cheap gap per the plan's straggler-pass instruction"

patterns-established:
  - "Lazy-import patch rule for CLI tests: patch the origin-module attribute the function-local from-import resolves at invocation time, then assert the core's call args — verified per command by reading the command body first"
  - "Vacuous-test detector: an assertion that passes via the default path (e.g. a helper defaulting level='INFO') proves nothing about the target branch — construction args must actually reach the branch; caught by per-file coverage before commit"

requirements-completed: [TEST-05, TEST-06]

coverage:
  - id: D1
    description: "tests/utils/test_transformers_compat.py (20 tests): apply_patches idempotency by object identity + guard-flag short-circuits + method-absent early returns; the patched get_parameter_or_buffer (passthrough, proxy via get_parameter and via get_buffer, re-raise, non-str tuple, TypeError-inert); _QuantStatProxy read forwarding / regular sets / _is_hf_initialized drop; candidate/marked classification; initialize_weights passthrough, bitsandbytes-unavailable passthrough (sys.modules None), swap→dequantize→original→restore ordering proof, restore-on-exception"
    requirement: TEST-05
    verification:
      - kind: unit
        ref: "tests/utils/test_transformers_compat.py (20 passed; transformers_compat.py 90/90 lines = 100%)"
        status: pass
    human_judgment: false
  - id: D2
    description: "tests/utils/test_logger.py (15) + straggler extensions: handler setup branches (console INFO + file DEBUG under tmp_path, no duplication), invalid-level fallbacks (constructor and LoggingContext), setup_logging extra file handler, all 7 convenience functions, log_function_call success/failure; sequence.py n_ratio>1 / padding back-off / random-length GC filter; training_plots stepless-entry skip; cuda_compat find_spec-miss / glob-miss / CDLL-OSError tolerance"
    requirement: TEST-05
    verification:
      - kind: unit
        ref: "tests/utils/ (58 passed; logger 109/109, sequence 75/75, training_plots 66/66, cuda_compat 30/30 = 100%)"
        status: pass
    human_judgment: false
  - id: D3
    description: "tests/cli/test_cli.py (54 tests): the click group help/version/subcommand surface; train/inference/benchmark/mutagenesis/model-config-generator/mcp-server subcommands and the train/inference/mutagenesis/model_config_generator console-script modules — happy paths with cores patched at their lazy-import binding sites and call-args asserted, assembled sys.argv forwarding + restoration, minimal-config dict construction, missing-option/import/core failure exits, and the standalone mutagenesis JSON payload end to end (numpy serialization, delta math, batch results, output file); zero process spawning"
    requirement: TEST-05
    verification:
      - kind: unit
        ref: "tests/cli/test_cli.py (54 passed; cli area 225 missing -> 5, all if __name__ script guards)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The 60-line orphans: tests/tasks/test_metrics.py +5 (macro pearson/spearman skip constant target columns with hand-computed -0.5 spearman, tensor-logits scatter branch via .numpy(), and the real metrics_for_dnabert2 regression/classification/generic arms network-free) and tests/configuration/test_configs.py +6 (four task_type alias normalizations, multilabel num_labels<2, early-stopping negative threshold, float+step and log+nonpositive-low search-space rejections, report_to 'all' combination)"
    requirement: TEST-05
    verification:
      - kind: unit
        ref: "tests/tasks/test_metrics.py 43 passed + tests/configuration/ 60 passed (metrics.py 277/277, configs.py 254/254 = 100%)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Wave-5 straggler pass: utils/support.py flash_attn availability + FP8 capability arms (7 lines), dnabert2 triton trans_b probe arms incl. the disable-marker-live-during-load proof (4), lucaone/omnidna/space extra= registry appends on copied lists (3), head.py MegaDNA default dims + UNet skip-pad branch via an unevenly-halving length (3)"
    requirement: TEST-06
    verification:
      - kind: unit
        ref: "tests/utils/test_support.py 4 + test_family_handlers.py 43 + test_head.py 34 passed (support 17/17, dnabert2 36/36, head 221/221 = 100%)"
        status: pass
    human_judgment: false
  - id: D6
    description: "FINAL GATE: full census (both roots, slow included) 1653 passed / 7 allowlisted skips / 0 failed / exit 0 in 934s; GATE PASS total 96.28% (7,124/7,399) strictly > 90.5 on the locked denominator; scripts/audit_skips.py exit 0; pragma census exactly 3 with zero additions; pyproject coverage/pytest tables diff-free against phase-start ref cbbebf8; tree clean"
    requirement: TEST-06
    verification:
      - kind: command
        ref: ".venv/bin/python -m pytest -ra --durations=0 --junitxml=/tmp/p3-05-junit.xml --cov -p no:cacheprovider -p no:progress -> coverage json -> assert totals.percent_covered > 90.5 (printed GATE PASS 96.28%)"
        status: pass
    human_judgment: false

duration: 51 min
completed: 2026-09-30
status: complete
commits: 4
plan_head_before: 63feec1e4fa16e47461213c585b9fce902529041
plan_head_after: cb00c50fc58e13fc07009b3b05afb6b8dae463be
---

# Phase 3 Plan 5: CLI/Compat Wave + Final Gate Summary

**The phase's named behavior-contract requirement proven on the live patched transformers class, all five CLI entry points driven in-process through CliRunner with asserted call args, both 60-line orphans closed to zero — and the FINAL GATE landed at 96.28% (7,124/7,399), strictly above the >90.5% target with audit-clean skips, pragma stability at exactly 3, and a diff-free denominator**

## Performance

- **Duration:** 51 min (incl. 15.5-min full census)
- **Started:** 2026-09-30T12:27:11Z
- **Completed:** 2026-09-30T13:18:40Z
- **Tasks:** 3/3
- **Files modified:** 11 test files + 1 coverage artifact

## FINAL GATE (the phase's definition of done — TEST-06)

**Command (reproducible from the repo root; identical shape to every wave census):**

```
.venv/bin/python -m pytest -ra --durations=0 --junitxml=/tmp/p3-05-junit.xml --cov -p no:cacheprovider -p no:progress
```

- **GATE PASS total 96.28% (7,124 covered / 7,399 statements)** — strictly greater than 90.5; was 91.24% after wave 4, 45.92% at the Phase-1 baseline
- Census: 1653 passed / 7 skipped (all allowlisted, zero new skips) / 0 failed / exit 0 — 934s
- `scripts/audit_skips.py` exit 0 on the gate junit
- Pragma census across `dnallm/`: **exactly 3, zero additions** (all three are the transformers/bitsandbytes-not-installed guards in `transformers_compat.py:87,156,181`)
- `pyproject.toml` coverage/pytest tables diff-free against the phase-start ref `cbbebf8` (last commit touching the phase's PLAN files)
- Working tree clean of strays (only the known gitignored `logs/` sink)
- Residual: 275 missing lines across 20 files — the complete accepted-uncovered ledger below

## Accomplishments

- `transformers_compat` is verified as the BEHAVIOR CONTRACT TEST-05 demanded, not line completion: idempotency by object identity on the real patched `PreTrainedModel`, the patched accessor's proxy/passthrough/re-raise arms, the `_is_hf_initialized` drop, the candidate/marked classifier, and the dequantize→original→re-quantize flow with `bitsandbytes.functional` monkeypatched — 90/90 lines, never unpatching the class
- Every CLI surface is now driven by the official runner: the group plus five command modules, with lazily-imported cores patched at the binding site the from-import actually resolves (origin-package attributes), asserted exit codes/output/call args, argv assembly and restoration, and the standalone mutagenesis JSON payload asserted field-by-field including numpy serialization and delta math
- Both orphans landed at exactly 100%: `tasks/metrics.py` (the real `metrics_for_dnabert2` arms, network-free) and `configuration/configs.py` (all nine validator branches)
- The straggler pass closed 17 additional cheap lines (`support.py`, the dnabert2 triton probe, three registry-append arms, two head branches); every remaining wave-4 row is documented with a per-file justification
- The whole utils area target set (transformers_compat, logger, sequence, training_plots, cuda_compat) plus support/dnabert2/lucaone/omnidna/space/head sits at 100%

## Task Commits

1. **Task 1: transformers_compat behavior contract + logger + utils stragglers (tracer)** — `2a662ea` (test)
2. **Task 2: CliRunner across all five CLI modules + the 60-line orphans** — `a4da561` (test)
3. **Task 3: straggler pass** — `bf491d3` (test)
4. **Task 3: FINAL GATE measurement + artifact** — `cb00c50` (test + artifact)

**Plan metadata:** this commit (docs)

Tracer feedback gate (Task 1, interactive + end-of-phase + automated-only verify): the full `<automated>` block was re-run end-to-end on the committed HEAD — green (58 utils passed, 20/15 ≥ 8 collected, pragma 3, fast leg 1552 passed) — expanded without a checkpoint per the #3299 precedence chain.

## Files Created/Modified

- `tests/utils/test_transformers_compat.py` — NEW, 20 contract tests on the live patched class
- `tests/utils/test_logger.py` — NEW, 15 tests (singleton, handlers, setup_logging, convenience fns, LoggingContext, decorator)
- `tests/utils/test_support.py` — NEW, 4 tests (flash_attn availability, FP8 capability)
- `tests/cli/test_cli.py` — NEW, 54 in-process CLI tests
- `tests/utils/test_cuda_compat.py` (+3), `tests/utils/test_sequence.py` (+3), `tests/utils/test_training_plots.py` (+1)
- `tests/tasks/test_metrics.py` (+5), `tests/configuration/test_configs.py` (+6)
- `tests/models/test_head.py` (+2), `tests/models/test_special/test_family_handlers.py` (+6)
- `.planning/phases/03-test-authoring-to-90-coverage/coverage-wave5-missing.txt` — final term-missing snapshot

## Decisions Made

- Cover the compat guard arms by monkeypatching `transformers.modeling_utils.PreTrainedModel` with a bare class (the function-local import binds the fake) instead of destructively removing methods from the live class
- Logger hygiene follows the actual implementation (stdlib logging + colorama, not loguru): handler add/remove per fresh logger + `chdir(tmp_path)` for the `logs/` sink
- `metrics_for_dnabert2`'s bare `evaluate.load("...")` names would resolve against the HF hub — patched `evaluate.load`/`combine` keep the arms network-free while asserting dict composition and prediction args
- Loopback-only hosts in CLI tests (ruff `hardcoded-bind-all-interfaces`); bare-group exit code asserted as 0-or-2 because click versions differ, with the help content pinned

## Deviations from Plan

### Verify-command substitutions (no code impact)

**1. [Verify substitution] `grep -c "::"` collect tripwires**
- **Found during:** Tasks 1-2 verify blocks
- **Issue:** pytest 9.1.1's `--collect-only -q` emits no `::` separators (established waves 1-4; already recorded in the windows ledger by wave 4)
- **Fix:** enforced the equivalent "N tests collected" counts — 20 ≥ 8 and 15 ≥ 8 (Task 1), 54 ≥ 15 (Task 2)

### Plan-text vs source reality

**2. [Rule 3 - Blocking] Logger is stdlib logging, not loguru**
- **Found during:** Task 1
- **Issue:** The plan's action text prescribed "per-test loguru handler cleanup via logger.remove"; `dnallm/utils/logger.py` is stdlib `logging` + colorama with no loguru anywhere
- **Fix:** implemented the equivalent hygiene for the real implementation (fresh-logger handler tracking + close/remove in an autouse fixture, `monkeypatch.chdir(tmp_path)` so `_setup_handlers`' `logs/dnallm.log` sink never lands in the repo)
- **Files modified:** tests/utils/test_logger.py
- **Verification:** 15 tests green; logger.py 109/109 lines

### Objective-required straggler closes beyond the plan's file list

**3. [Task 3 straggler pass] 17 lines in files the plan did not enumerate**
- support.py 7, dnabert2.py 4, lucaone/omnidna/space 3, head.py 3 — every one a cheap gap from already-loaded context, closed under the plan's own straggler-pass instruction ("close what is cheap, document what is not")

**4. [Measured-fact adjustment] The metrics orphan was 31 lines, not the plan's 51**
- The wave-4 term-missing artifact (the plan's named authority) listed 180, 191, 224, 592-650 = 31 statements; all 31 closed

---

**Total deviations:** 1 verify-command substitution (pre-recorded), 1 implementation-substitution (loguru→stdlib logging), 2 straggler/measurement adjustments inside the plan's own instructions
**Impact on plan:** No scope creep — no new dependencies, no new skips (allowlist untouched at 7), pragma budget intact at 3, denominator untouched.

## Accepted-Uncovered Residual Ledger (275 lines across 20 files — documented, never pragma'd)

| File | Missing | Justification |
|------|---------|---------------|
| dnallm/inference/inference.py | 93 | Deep `generate`/scoring branches and device corners (CUDA-only mamba paths, GPU memory-estimation branches, batch-streaming corners) — research-projected residual (Assumption A1); CUDA-only lines cannot cover on the CPU CI gate environment anyway |
| dnallm/models/special/borzoi.py | 42 | Handler bodies behind absent `borzoi`-toolchain imports — stub-shape limits; timeboxed in wave 2, slack banked elsewhere exactly as resolved in research Open Question 2 |
| dnallm/inference/plot.py | 40 | Altair spec corners and the `chart.show()` browser branches — driving a browser render is out of reach in-process |
| dnallm/inference/interpret.py | 23 | Captum attribution edge kernels (LayerDeepLift fallback arms, non-attributable target shapes) |
| dnallm/models/special/crossdna.py | 14 | Exotic forward corners in the ported architecture (dropout/module-list arms not hit by the real-torch forwards) |
| dnallm/inference/benchmark.py | 11 | StratifiedKFold fallback + plot residual branches |
| dnallm/models/special/megadna.py | 10 | `seqmodels`-gated handler branches (dependency absent) |
| dnallm/models/model.py | 10 | Defensive fallback arms in the loader chain that are dead in this environment (tokenizer-tier fallbacks already satisfied earlier, mirror/env branches) |
| dnallm/inference/mutagenesis.py | 9 | Visualization/browser branches of the effect plots |
| dnallm/datahandling/data.py | 8 | Wave-4 ledger carried forward verbatim: dead defensive branches (unconditional pad_id harvest, tokenize-closure None guard, unreachable `__data_type__` exit, `_create_final_chart` non-dict else) + the `chart.show()` browser branch |
| dnallm/mcp/start_server.py | 4 | Direct-execution import shim (needs module-reload machinery that would mutate shared state mid-suite) + `__main__` guard |
| dnallm/mcp/server.py | 2 | Line 1286 is dead-defensive (the both-None case already returns an error dict one branch earlier); 2065 is the `__main__` guard |
| dnallm/finetune/trainer.py | 2 | Module-level `except ImportError: optuna = None` — reachable only without optuna installed (wave-4 ledger) |
| dnallm/models/tokenizer.py | 1 | `return_dict=False` embeds arm of DNAOneHotTokenizer |
| dnallm/models/special/evo.py | 1 | Wave-2 ledger line (86) |
| dnallm/cli/{cli,inference,train,mutagenesis,model_config_generator}.py | 5 | `if __name__ == "__main__":` script guards each — reachable only by running the module as a script, which the in-process discipline (AUDIT-04) forbids |

Environment-bound classes represented above: CUDA-only, browser-render, absent-toolchain (borzoi/seqmodels/lucagplm-adjacent), dead-defensive, and script guards. None silenced with pragmas — the count stays at the locked 3.

## Known Stubs

None — every new test asserts observable behavior (identities, call args, exact/approx values, file contents, JSON fields); no placeholder logic introduced.

## Threat Flags

None — test-only changes; no new network listeners, auth paths, or schema changes. The T-3-12/13/14 mitigations all verified at the gate (pragma 3, pyproject diff-free, cores patched with call assertions, zero spawn references in CLI tests).

## Issues Encountered

- First draft of the invalid-level logger test was vacuous (the helper's `level="INFO"` default took the valid path, so the fallback branch never ran) — caught by per-file coverage before commit and fixed by passing the bogus level explicitly; recorded as a pattern (vacuous-test detector)
- `Mock.assert_called_once_with` on torch-tensor args raises `RuntimeError: Boolean value of Tensor is ambiguous` — replaced with field-by-field `call_args` assertions
- Ruff's `hardcoded-bind-all-interfaces` flagged the `0.0.0.0` host literal in CLI argv tests — switched to loopback `127.0.0.1`
- A first straggler fake for the dnabert2 triton probe lacked `float32`, so line 28 raised AttributeError and took the wrong except arm — coverage per-file check exposed it; the fake now carries `float32`, and a dedicated TypeError-arm test pins the no-disable behavior

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 3 is COMPLETE: every area gate green and the FINAL GATE at 96.28% leaves 5.78 points of headroom over the strict >90.5% requirement (6.28 over Phase 4's 90% gate) — the CI threshold can ratchet on the same census command with margin for drift
- The identical invocation reproduces the number locally; Phase 4's CI job should run exactly this command (network-included slow leg, ~16 min) before adding the threshold
- `coverage-wave5-missing.txt` is the residual worklist if a future phase wants to push beyond 96% — the two largest blocks (inference.py 93, borzoi.py 42) are environment-bound or stub-limited by design
- Suite runtime: census 934s (+3s vs wave 4 despite +122 tests); slow real-model legs remain the dominant cost (deferred-items has both)

---
*Phase: 03-test-authoring-to-90-coverage*
*Completed: 2026-09-30*

## Self-Check: PASSED

Created files exist on disk (tests/utils/test_transformers_compat.py, test_logger.py, test_support.py, tests/cli/test_cli.py, coverage-wave5-missing.txt); all four task commits (2a662ea, a4da561, bf491d3, cb00c50) present in history; commits measured from the plan ledger (63feec1 → cb00c50 = 4).
