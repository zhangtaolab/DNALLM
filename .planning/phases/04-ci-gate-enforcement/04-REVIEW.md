---
phase: 04-ci-gate-enforcement
reviewed: 2026-10-01T05:07:13Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - .github/dependabot.yml
  - .github/workflows/ci.yml
  - dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
  - dnallm/mcp/tests/test_mcp_functionality.py
findings:
  critical: 0
  warning: 0
  info: 3
  total: 3
status: issues_found
incremental: true
diff_base: 8a6c69e477b80c8667af5436b7f10179e49113a6
---

# Phase 04: Code Review Report (incremental re-review #3661)

**Reviewed:** 2026-10-01T05:07:13Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found (0 critical, 0 warning, 3 info)

## Summary

Incremental re-review of the delta since `8a6c69e` only: (a) the open_chromatin MCP test model
swap (mamba remote code → cached non-mamba `plant-dnagpt-BPE-promoter`, multiclass/3 → binary/2),
(b) the `runner.environment == 'github-hosted'` free-disk guards added to all four jobs, (c) the
`coverage-nightly` move to `[self-hosted, dnallm-nightly]`, and (d) the dependabot comment/ignore
block documenting the mamba-ssm/causal-conv1d pins and the transformers MambaCache risk.

Every changed hunk was verified against source or runtime evidence, not just read:

- **Model swap is sound and consistent.** `Plant DNAGPT BPE promoter` / `zhangtaolab/plant-dnagpt-BPE-promoter`
  exists in the registry (`dnallm/models/model_info.yaml:173-174`); `binary` + `num_labels: 2` +
  two `label_names` passes the Pydantic `model_post_init` binary branch
  (`dnallm/configuration/configs.py:118-136`); the test's `num_labels=2` matches. The stale
  `model_name: "Plant DNAMamba BPE open chromatin"` in `dnallm/mcp/tests/configs/mcp_server_config.yaml:34`
  is descriptive only — `MCPConfigManager` consumes just `enabled` and `config_path`
  (`dnallm/mcp/config_manager.py:63-75`); loading uses `model.path` from the per-model config.
  `mcp_server_config_2.yaml` shares the swapped inference config; no other test asserts the old
  multiclass/3-label values (all count/label assertions in `test_config_manager.py` target inline
  synthetic configs).
- **Test hardening claims hold.** `_assert_prediction`'s docstring contract ("ModelManager
  swallows load/predict failures, returns None, never raises") matches
  `dnallm/mcp/model_manager.py:233-247`; `assert result_map` catches per-model load failure even
  when `assert manager.loaded_models` passes on a partial load.
- **CI guards are correct.** `runner.environment` is a documented runner-context property;
  the guard fails safe (undefined on old runners → step skipped, never a sudo failure on the
  self-hosted box). The self-hosted nightly is restricted to `schedule`/`workflow_dispatch`, so
  fork PRs cannot execute test code on it; job permissions remain `contents: read`.
- **Comment arithmetic re-derived, not trusted.** Timeout marks measured across `tests/` and
  `dnallm/mcp/tests/`: 3×7200 + 4×3600 = 600 min (phase marks), 2×900 + 5×1800 + 1×3600 =
  240 min — the 1800 s mark is class-level on `TestRealModelInference`
  (`tests/inference/test_inference_real_model.py:23-24`) which has exactly 5 test items, matching
  the comment. 840 min total < 900 min job kill. Accurate.
- **dependabot factual claims verified.** `pyproject.toml:66` is `transformers>=4.49.0,<6`
  ("<6" comment accurate); empirically, installed transformers 5.17.0 does **not** export
  `MambaCache` and the cached ModelScope `modeling_mamba.py` imports it — the "removed as of
  5.17" claim is correct (the earlier commit message d8130c1 saying "5.18" was the wrong one,
  and that bound was reverted in 95c9ba0 anyway). The mamba-ssm/causal-conv1d ignore entries
  without `update-types` are valid YAML with real effect (exact pins would otherwise be bumped).
- No secrets, no injection, no new untrusted interpolation in the delta.

Three Info-level findings below (numbered IN-05.. to avoid colliding with the open IN-01..IN-04
rows already recorded in 04-REVIEW-DISPOSITION.md — per that file's own rules, reusing an ID
silently drops the earlier open row).

Adjacent out-of-scope context (no finding issued, recorded for awareness): the **production**
MCP configs still point at mamba models that cannot load under the installed transformers 5.17.0
(`dnallm/mcp/configs/open_chromatin_inference_config.yaml:24`, `h3k27me3`/`h3k27ac` configs), so
`dnallm-mcp-server` run with the shipped configs would fail to load those three slots today.
Owner-accepted this cycle (the transformers bound was deliberately reverted as spurious); worth
a WINDOWS.md entry of its own (see IN-05).

## Critical Issues

None.

## Warnings

None.

## Info

### IN-05: dependabot comment points at a WINDOWS.md entry that does not exist

**File:** `.github/dependabot.yml:26`
**Issue:** The transformers risk comment says the MambaCache breakage is recorded "(see
`.planning/WINDOWS.md`)". The file exists and is git-tracked, but its Broken Windows Ledger (all
10 entries read) contains no transformers/MambaCache/mamba-load entry — every row is a phase
deviation or an inference/mutagenesis bug. The pointer implies the risk is ledgered where
`/gsd-ship` gates and waiver decisions look; it is not. The risk text itself is co-located in
the comment (and factually correct — verified against the installed transformers), so no
information is lost, but the cross-reference misdirects the maintainer evaluating a transformers
bump.
**Fix:** Either record the actual entry (`gsd-tools windows add unmet-truth "ModelScope mamba
remote code imports MambaCache, removed in transformers 5.17 — mamba-family loading broken
(production MCP configs affected)"`) or repoint the comment at an artifact that does record it
(e.g. this phase's 04-REVIEW.md / commit 95c9ba0).

### IN-06: swapped open_chromatin config keeps the old model's semantics and performance metrics

**File:** `dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml:12,33-38`
**Issue:** The swap note documents the model change, but `description` still says "…in open
chromatin regions in plants" (line 12), `task_category` is still `chromatin_state_prediction`
(line 33), and the `performance_metrics` block (accuracy 0.82 / f1 0.79 / precision 0.78 /
recall 0.81, lines 34-38) was carried over unchanged from the removed DNAMamba open-chromatin
model — those figures are not the promoter model's, yet they are surfaced verbatim through
`ModelManager.get_model_info` → the MCP `model_info` tool. The label semantics
("Not promoter"/"Core promoter") now contradict the surrounding open-chromatin description.
Purely cosmetic in a test fixture (none of these fields gate loading or assertions), but this
fixture doubles as the MCP example config. The sibling `mcp_server_config.yaml:34`
`model_name: "Plant DNAMamba BPE open chromatin"` has the same drift (out-of-scope file).
**Fix:** Either update `description`/`task_category`/`performance_metrics` to the promoter
model's actual values, or extend the existing NOTE with "slot metadata intentionally left
open-chromatin; model is a promoter model" so the next reader does not have to diff to find out.

### IN-07: deploy job pins deprecated `actions/cache@v3` while the rest of the file uses @v4 (pre-existing, outside this cycle's delta)

**File:** `.github/workflows/ci.yml:410`
**Issue:** Every other cache use in this workflow is `actions/cache@v4`; the deploy job alone
pins `@v3`, the deprecated major of the action. Latent, not hypothetical-broken: deploy only
runs on push to `main`/`master` and this branch is `dev`, so the step has not executed since the
divergence and a v3 failure/brownout would surface only at the next release docs deploy. Not
introduced by this delta (pre-dates `8a6c69e`); recorded because it is a real robustness defect
in a reviewed file and the new github-actions dependabot lane will not necessarily catch a
major pin it is told to group only minor/patch.
**Fix:** Bump to `actions/cache@v4` (the `key`/`path`/`restore-keys` syntax in use is
compatible as-is).

---

_Reviewed: 2026-10-01T05:07:13Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (incremental, diff_base 8a6c69e477b80c8667af5436b7f10179e49113a6)_
