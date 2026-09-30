---
phase: 04-ci-gate-enforcement
reviewed: 2026-09-30T18:30:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - pyproject.toml
  - models.lock
  - .github/workflows/ci.yml
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
  - .github/workflows/README.md
findings:
  critical: 1
  warning: 4
  info: 4
  total: 9
status: issues_found
---

# Phase 04: Code Review Report — CI Gate Enforcement

**Reviewed:** 2026-09-30T18:30:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed the coverage ratchet (`fail_under = 90`), the `models.lock` manifest, the two-job
CI restructure (`coverage-gate` / `coverage-nightly`), the 7 per-test timeout marks, and the
workflow README — with cross-file verification against every slow test in the suite, the MCP
test configs, `scripts/audit_skips.py`, `tests/expected_skips.yaml`, and live GitHub Actions
runs on `dev` and on the GATE-04 probe branch.

Verified working (evidence-backed, not assumed):

- **Gate proven live both directions.** Run 36748818884 (push, dev): `coverage-gate` green in
  12m53s, all six `test` matrix legs green, `coverage-nightly` correctly skipped. Probe run
  36749810723: `coverage-gate` **failure** on the synthetic coverage regression, with the
  py3.11 matrix leg failing under the same floor (confirming `fail_under` rides every
  `--cov` invocation, including the matrix — as the README documents).
- **Event guards behave.** On `workflow_dispatch`, legacy jobs + `deploy` are skipped and only
  `coverage-nightly` runs; on push, `coverage-nightly` is skipped. The amended
  `deploy` guard (`event_name == 'push' && ...`) correctly prevents schedule/dispatch from
  triggering docs deploys (the un-amended condition would have matched on dispatch against
  `main`).
- **Timeout marks fire on unittest methods.** Verified empirically in this repo's venv:
  `@pytest.mark.timeout(1)` on a `unittest.TestCase` method fails with
  `Failed: Timeout (>1.0s) from pytest-timeout`. All 6 trainer marks + 1 integration mark are
  on unittest methods, so the mechanism is sound.
- **models.lock is accurate.** All 9 entries verified against actual call sites:
  `DialoGPT-small` (tests/models/test_model.py:184, HF), `plant-dnagpt-BPE-promoter` HF
  (test_inference_real_model.py:48-50) and MS (test_inference.py:470-477),
  `plant-dnabert-BPE` (exactly the 12 `load_model_and_tokenizer` call sites in
  test_trainer_real_model.py), the three MCP models (exactly the three per-model configs
  referenced by `dnallm/mcp/tests/configs/mcp_server_config.yaml`; the unused
  h3k27ac/h3k27me3 configs are correctly not listed), `DNA_bert_4` (test_model.py:194-198),
  and the promoter dataset (`MsDataset.load` caches under `~/.cache/modelscope/hub`, which
  the nightly cache path covers). The `test` matrix py3.13+numpy1.26.4 leg (no cp313 wheels
  exists upstream) was confirmed green on live CI, so no finding there.

One blocker: a slow test that cannot fail now sits inside the newly-gated nightly census.
Four warnings: six slow MCP live-server tests can never run in CI while the census reports
green; the timeout layering covers only 7 of 21 slow tests; the nightly's 480-minute kill
is below the sum of its own per-test timeout ceilings and a kill also forfeits the model
cache; and the README documents a quality toolchain (Black/isort/Flake8, XML coverage
reports) the workflow no longer runs.

Note: at review time the first live `coverage-nightly` dispatch (run 36747594207, started
16:54Z) was still in progress (~1.5h elapsed) — within projection; its outcome is the one
remaining unproven leg of this phase.

## Critical Issues

### CR-01: Slow test `test_with_config_file` can never fail — reports green on any regression inside the nightly census

**File:** `tests/finetune/test_trainer_real_model.py:704-773`
**Issue:** Every failure path in `test_with_config_file` ends in `return False` (missing
config file at line 723, and the catch-all at line 768-773), and the success path returns
`True`. Pytest ignores non-None return values from test functions — the returned value does
not fail the test. The `PytestReturnNotNoneWarning` this triggers is silently swallowed
because `PytestReturnNotNoneWarning` subclasses `UserWarning` and `pyproject.toml:493`
filters `ignore::UserWarning`. Net effect: this `@pytest.mark.slow` test (given
`@pytest.mark.timeout(7200)` this phase) passes unconditionally in the
`coverage-nightly` census — its only CI execution point — even when config loading, model
loading, dataset encoding, training, or prediction raises. A regression in the exact
load→dataset→train→infer workflow it claims to verify prints a `❌` and reports PASS. This
is precisely the exit-code-masking failure class this phase guards against (the `test` job
even carries a dedicated exit-code canary for it), and it directly undermines the census of
record the nightly is supposed to be.

**Fix:**

```python
@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_with_config_file():
    """Test with the provided finetune config file."""
    ...
    except Exception as e:
        print(f"❌ Error during config file testing: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Config-file training workflow failed: {e}")  # was: return False
```

(Also replace the early `return False` at line 723 with `pytest.fail(...)` /
`pytest.skip(...)` as appropriate. If the `__main__` block needs the boolean contract,
wrap its call in try/except instead of weakening the test.)

## Warnings

### WR-01: The 6 slow MCP live-server probes can never execute in any CI job, and the skip allowlist makes their absence invisible

**File:** `.github/workflows/ci.yml:348-356` (cross-file:
`dnallm/mcp/tests/test_sse_client.py:26,74,94`,
`dnallm/mcp/tests/test_streamable_http_client.py:36,63,95`,
`tests/expected_skips.yaml` — `prefix: "network-unavailable:"` entry)
**Issue:** The three SSE probes target `http://localhost:8000/sse` and the three
streamable-http probes target `http://localhost:8000/mcp`. Nothing in the suite or the
workflow ever starts a server on localhost:8000 (no uvicorn/serve/subprocess/Popen anywhere
under `dnallm/mcp/tests/`; the nightly job runs pytest only). So in `coverage-nightly` these
six `@pytest.mark.slow` tests deterministically hit connection-refused
(`httpx.ConnectError` is a `TransportError`), `skip_if_unreachable` converts it to a skip
with the stable `network-unavailable:` prefix, and that prefix is allowlisted in
`tests/expected_skips.yaml` — whose comment explicitly says it "serves … the Phase-4 slow
leg". The result: the "full census incl. slow" systematically and silently excludes 6 of the
21 slow tests while reporting green, and the README claim "Full coverage census including
the `slow` tests" (`.github/workflows/README.md:106,108,115`) overstates what the census
contains. This is a coverage gap, not a false-pass on executed assertions — hence WARNING,
not BLOCKER.
**Fix:** Either start the MCP server in the nightly before pytest (a `run: uv … &
dnallm-mcp-server --transport sse …` background step, or a session-scoped fixture that
boots the server and yields the URL), or scope the `network-unavailable:` allowlist entry
out of CI (e.g. env-gated) and document in the README that live-server probes are
local-only so the census claim is accurate.

### WR-02: Timeout layering incomplete — 14 slow tests still under the global 300s cap, including every cold-cache network download outside the trainer file

**File:** `pyproject.toml:475` (`--timeout=300` global; cross-file:
`tests/models/test_model.py:178-200`, `tests/inference/test_inference_real_model.py:22-232`,
`dnallm/mcp/tests/test_mcp_functionality.py:40-163`)
**Issue:** The phase added per-test marks to 7 long tests (6 trainer + 1 integration), but
the nightly runs 21 slow tests; the rest remain capped at the global 300s per test. That
includes all remaining real-network work: the two hub-download tests
(`test_download_real_huggingface_connection`, `test_download_real_modelscope_connection`,
`max_try=1`), the `TestRealModelInference` class whose `setUpClass` downloads
`zhangtaolab/plant-dnagpt-BPE-promoter` and builds the engine (that time lands inside the
first test item's timeout window), and `test_mcp_functionality`, which loads all three
ModelScope MCP models in-process. The first nightly (and any cache-key rotation) runs with a
cold `Linux-models-<hash>` cache, so per-test wall time includes full multi-hundred-MB
downloads from US runners; 300s for download + load + inference is tight, and a breach is a
red nightly (fail-correct, but it kills the census on a duration, not a regression). The
trainer-file treatment shows the intent; it is just not applied to the other network-bound
slow tests.
**Fix:** Add per-test marks mirroring the trainer treatment, e.g.
`@pytest.mark.timeout(900)` on the two download tests and `test_mcp_functionality`, and
`@pytest.mark.timeout(1800)` on `TestRealModelInference` (class-level, so the `setUpClass`
download is covered).

### WR-03: Nightly job kill at 480min is below the sum of its own per-test timeout ceilings — and a kill also forfeits that night's model cache

**File:** `.github/workflows/ci.yml:294` (`timeout-minutes: 480`, cross-file:
`tests/finetune/test_trainer_real_model.py:54,313,406,485,560,703`,
`tests/inference/test_inference.py:460`)
**Issue:** The new per-test ceilings sum to 600min
(7200+7200+3600+3600+3600+7200+3600s) before counting the fast suite or any other slow
test. A single hung trainer test burns up to 2h before its mark fires; two long straggers
push past 480 and GitHub kills the job mid-census — no junit (the skip audit never runs),
no coverage total, and the `actions/cache` post-save (success-only, as the README states at
line 110) discards the multi-GB downloads performed that night. While the hang persists,
every subsequent nightly repeats the full download cost. Even the README's own projection
("4-7.5h", line 108) leaves only ~30min of headroom to the kill.
**Fix:** Raise `timeout-minutes` to ≥660 (comfortably above the ceiling sum plus fast-suite
time), or split the cache into `actions/cache/restore` + an explicit
`actions/cache/save` step with `if: always()` so a census kill does not also throw away the
model downloads.

### WR-04: README documents a quality toolchain and reporting pipeline the workflow no longer runs

**File:** `.github/workflows/README.md:34-38, 133, 137-140, 179-183, 194-205`
**Issue:** The `test` job section, "Quality Standards", "Troubleshooting", and "Local
Testing" all reference **Black / isort / Flake8** (and tell contributors to run
`black --check .`, `isort --check-only .`, `flake8 .`), but the workflow runs
`ruff format --check .` and `ruff check . --statistics` (`ci.yml:77-83`). Additionally,
line 133 claims "Coverage reports are generated in XML and terminal formats" — this phase
removed both `coverage xml` and the codecov uploader from the `test` job, so no XML
coverage report is produced anywhere in CI (junit XML is test results, not coverage). A
contributor following the README validates with the wrong tools and expects an artifact
that no longer exists.
**Fix:** Replace the Black/isort/Flake8 references (steps list, Quality Standards,
Troubleshooting, Local Testing block) with the ruff commands (`ruff format --check .`,
`ruff check .`), and drop or correct the XML-coverage claim to match the junit-only
reporting.

## Info

### IN-01: models.lock header points at the wrong job for the model cache

**File:** `models.lock:2`
**Issue:** "Keys the gated CI job's model cache" — the models.lock-keyed cache exists only
in `coverage-nightly` (`ci.yml:325-333`); the gated PR fast leg (`coverage-gate`) has no
model cache. A maintainer rotating cache keys would look in the wrong job.
**Fix:** Reword to "Keys the coverage-nightly model cache (actions/cache hashFiles)".

### IN-02: README names a "develop" branch; the workflow filters on `dev`

**File:** `.github/workflows/README.md:13-14` (vs `.github/workflows/ci.yml:8,10`)
**Issue:** Push/PR trigger docs say `main`, `master`, and `develop`; the actual filters are
`main`, `master`, `dev`.
**Fix:** Change "develop" to "dev".

### IN-03: `deploy` does not `need` `coverage-gate`

**File:** `.github/workflows/ci.yml:359`
**Issue:** `needs: [test, test-cuda, test-mamba]` omits the dedicated gate job. Coverage is
still transitively enforced on main pushes (every `test` matrix leg runs `--cov` under the
90 floor — the probe run shows matrix legs failing alongside the gate), so this is a
defense-in-depth gap, not a bypass.
**Fix:** Add `coverage-gate` to the `needs` list so the dedicated gate is authoritative for
deploys.

### IN-04: `coverage-gate` duplicates the `test` (py3.12, numpy2.2.0) matrix leg

**File:** `.github/workflows/ci.yml:233-288`
**Issue:** Same interpreter, same env (`.[base]` + numpy 2.2.0), same fast census
(`-m "not slow"` + `--cov`), same skip audit as one matrix cell — an extra ~13-minute leg
on every push and PR (live: gate 12m53s vs matrix leg ~8m). If the single stable leg is
intentional (isolation from matrix churn), that rationale is worth stating; otherwise the
gate could be a matrix cell.
**Fix:** Document the intended redundancy in the README job description, or consolidate.

---

_Reviewed: 2026-09-30T18:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
