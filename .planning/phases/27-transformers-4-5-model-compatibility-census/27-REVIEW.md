---
phase: 27-transformers-4-5-model-compatibility-census
reviewed: 2026-10-27T10:14:04Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - dnallm/models/model_info.yaml
  - dnallm/models/registry.py
  - dnallm/utils/transformers_compat.py
  - docs/user_guide/model_compatibility.md
  - .github/dependabot.yml
  - .github/workflows/census_compat.yml
  - mkdocs.yml
  - scripts/census_compat.py
  - tests/models/compat_census.json
  - tests/models/test_census_compat.py
  - tests/models/test_compat_docs_consistency.py
  - tests/models/test_model_info_registry.py
  - tests/models/test_registry.py
  - tests/utils/test_transformers_compat_census.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 1
  warning: 3
  info: 6
  total: 10
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-10-27T10:14:04Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Phase 27 delivers a transformers 4/5 compatibility census: `scripts/census_compat.py` (census / write-back / docs generation / verification), two new compat rungs in `dnallm/utils/transformers_compat.py`, three `tf_compat_*` schema fields in `dnallm/models/registry.py` plus a 49-row write-back into `model_info.yaml`, a generated docs page with an offline drift gate, and a dispatch-only CI lane.

Verification performed during this review: `ruff check` and `ruff format --check` pass on all changed Python files; all 304 tests across the five reviewed test files pass (`191 + 113`); `python scripts/census_compat.py --verify --artifact tests/models/compat_census.json` (dry mode) exits 0 against the committed registry. The invariants that hold: subprocess isolation for remote-code probes and loads (both child programs import `dnallm` first, absolute paths, no shell), anchored text insertion in `--write-registry` (no YAML round-trip; idempotent; provenance blocks pinned), REG-01 sole-parser rule (the docs gate reads typed registry accessors only; no second YAML reader inside `dnallm/`), runner-lane surface posture (workflow_dispatch only, `contents: read`, no masked steps), and committed data consistency (registry / artifact / docs page agree; 49/187 split; no `tf_compat_*` keys on finetuned rows).

However, the fail-closed **composition** of the dispatch lane is structurally broken: a fresh census run can never satisfy `compat_verdict`, so `--run --verify` can never exit 0 (CR-01, proven by simulation below). Two robustness issues around the frozen evidence artifact (disposition loss on re-run, HF endpoint leakage into the test process) and a set of minor quality items follow.

## Critical Issues

### CR-01: `--run --verify` can never exit 0 — the dispatch lane is permanently red and cannot act as a drift gate

**File:** `scripts/census_compat.py:1278-1336` (`compat_verdict`), `scripts/census_compat.py:1840-1901` (`verify_registry`), `scripts/census_compat.py:1963-2001` (`main`), `.github/workflows/census_compat.yml:88-90`
**Issue:** The workflow runs `python scripts/census_compat.py --run --with-loads --verify --artifact census-artifacts/census-fresh.json`. `compat_verdict` requires a `disposition` (`skip`/`fix`) to map any record that has neither a green load+forward nor non-representative status — but `disposition` keys exist **only** in the frozen committed artifact (hand-authored 27-02 judgments); nothing in the tool ever writes them (grep: only `compat_verdict` reads `record.get("disposition")`; `merge_records` does not carry them either). A fresh census therefore produces unmappable records for every parked failure. Verified live in this review by simulating fresh records through `compat_verdict`:

```
RAISED: census record for songlab/gpn-brassicales (status 'import-fail') carries no green load and no skip disposition - cannot map it to a registry status
RAISED: census record for JadenLong/MutBERT-Human-Ref (status 'import-ok') carries no green load and no skip disposition - cannot map it to a registry status
```

At minimum GPN, Omni-DNA, DNABERT-2, GENA-LM, GENA-LM-BigBird and MutBERT (all family representatives that are import-fail or failed-load today) hit the `raise ValueError` inside `verify_registry`, so `verify` returns 1 unconditionally. Compounding this, `run_census` returns 1 whenever any repo is unreachable on both hubs — `GenerTeam/GENERanno-eukaryote-1.2b-base` is committed as exactly that (census-blocked, "repo id exists on neither hub"), so `run_rc` is also 1 on every run — and `main()` then returns `run_rc if run_rc != 0 else verify_rc`, discarding the verification result entirely. Net effect: the lane can never be green; "red" cannot distinguish real drift from baseline, defeating its documented purpose ("fails red the moment the world ... disagrees" implies green when the world agrees; the workflow comment "a red step means real drift or infrastructure failure" is false for at least three structural reasons). Note the dry form (`--verify` alone against the committed artifact) does pass today — the defect is specifically the composed fresh-evidence form the workflow ships.
**Fix:** Make the fresh-evidence verification self-contained, e.g. (a) have `run_census` carry `disposition`/`previous_status` forward from any existing artifact (see WR-01) and seed the lane's artifact from the committed file (`--artifact tests/models/compat_census.json` or copy it to `census-artifacts/` first), or (b) add a fresh-run mapping fallback in `compat_verdict` (import-fail rep with no disposition → compare against committed `remote-code-broken` via a dedicated lane-side mapping), and stop treating a committed-known unreachable repo as a lane failure (record it, compare verdicts, and let matching verdicts pass). At minimum, surface `verify_rc` independently in `main()` (e.g. `return run_rc or verify_rc`) so the two failure classes are distinguishable, and document the expected-red entries in the workflow.

## Warnings

### WR-01: `merge_records` silently drops `disposition` and `previous_status` — a default-artifact re-run destroys the frozen 27-02 judgments

**File:** `scripts/census_compat.py:1124-1173`
**Issue:** `merge_records` replaces each existing record with the fresh one and carries over only `load`/`load_note` (lines 1156-1164). The hand-authored `disposition` objects and the `previous_status` stamps (the pre-shim evidence trail) are dropped even when the verdict is unchanged. Verified by simulation in this review: an unchanged `import-ok` record with `disposition: {skip: "checkpoint data defect: ..."}` and `previous_status: "import-fail"` merges to a record with neither key (load carried, disposition lost). The committed provenance block invites exactly this re-run ("Tooling: scripts/census_compat.py --write-registry --apply (committed, re-runnable ...)"), and `--artifact` defaults to `tests/models/compat_census.json` — running plain `--run` against the default path rewrites the frozen artifact without dispositions, immediately turning `TestDispositionCompleteness` (`tests/models/test_census_compat.py:572-637`) red and losing the skip/fix judgment evidence (git-recoverable only).
**Fix:** In `merge_records`, when `previous.get("status") == record.get("status")`, carry `disposition` (and `previous_status`) forward the same way `load` is carried:

```python
if (
    record.get("load") is None
    and previous is not None
    and previous.get("load") is not None
    and previous.get("status") == record.get("status")
):
    record["load"] = previous["load"]
    if "load_note" in previous:
        record["load_note"] = previous["load_note"]
    if "disposition" in previous:
        record["disposition"] = previous["disposition"]
    if "previous_status" in previous:
        record["previous_status"] = previous["previous_status"]
```

### WR-02: Loading the census script in tests sets `HF_ENDPOINT=https://hf-mirror.com` process-wide at collection time

**File:** `scripts/census_compat.py:17`, `tests/models/test_census_compat.py:28-39`
**Issue:** `scripts/census_compat.py` sets `os.environ["HF_ENDPOINT"] = HF_MIRROR` at module import (line 17), and `tests/models/test_census_compat.py` executes the module at collection time (`census = _load_census_module()`, line 39). Every test that runs afterwards in the same pytest process — including the `slow` real-model lanes in the nightly run — has its Hugging Face downloads silently rerouted through the third-party `hf-mirror.com` endpoint, and any user-/CI-provided `HF_ENDPOINT` is overridden. On hosts where the mirror is unreachable or slow, real-model tests fail for an environment reason no test-local change can explain; downloads also traverse an unintended third party.
**Fix:** In the test loader, snapshot and restore the environment around `exec_module` (e.g. `monkeypatch` at session scope, or in `_load_census_module`: save `os.environ.pop("HF_ENDPOINT", None)`, exec, restore) — the child probe/load programs set `HF_ENDPOINT` inside their generated source anyway (`build_probe_program`/`build_load_program`), so the parent-process mutation is not needed by any offline test.

### WR-03: Remote-code probe subprocesses inherit the full runner environment, including `GITHUB_TOKEN`, while executing untrusted repo code

**File:** `scripts/census_compat.py:629-635` (`run_import_probe`), `scripts/census_compat.py:1002-1008` (`run_load_probe`), `.github/workflows/census_compat.yml:30,88-90`
**Issue:** Both probe subprocesses call `subprocess.run([sys.executable, "-c", program], ...)` without an `env` argument, so the child inherits the parent's environment. On the dispatch lane that parent is a GitHub runner process whose environment carries `GITHUB_TOKEN` (scoped `contents: read`), `RUNNER_*` credentials/paths, and any configured secrets. The child executes arbitrary `trust_remote_code` Python fetched from 49 third-party repos (the census's whole purpose). The subprocess isolation protects the census process's Python state (shims), not the host or the token: a malicious repo's code runs with read access to the repo token and the self-hosted box's runner context. The lane's dispatch-only trigger bounds who can *schedule* this, but not what the fetched code can *reach* once scheduled.
**Fix:** Pass a sanitized environment to both probes, keeping only what the children need (`PATH`, `PYTHONPATH`/venv, `HF_HOME`/`HF_ENDPOINT`/`HF_HUB_*`, CUDA-related vars, `HOME`):

```python
_BLOCKED_ENV_PREFIXES = ("GITHUB_", "RUNNER_", "ACTIONS_")
clean_env = {
    k: v for k, v in os.environ.items()
    if not k.startswith(_BLOCKED_ENV_PREFIXES)
}
completed = subprocess.run([sys.executable, "-c", program], ..., env=clean_env)
```

## Info

### IN-01: New rung code adds mypy errors under a `ty:`-only ignore convention

**File:** `dnallm/utils/transformers_compat.py:1594-1601`
**Issue:** `_materialize_tied_weights_bookkeeping` uses `# ty: ignore[unresolved-attribute]` comments, which mypy does not honor; the added lines produce new mypy `attr-defined` errors (1596, 1597, 1601: `get_expanded_tied_weights_keys` / `named_children` / `all_tied_weights_keys` on `object`). This matches the module's pre-existing posture (lines 718/725) and mypy is advisory in CI (`mypy dnallm/ || true`), but the phase claimed mypy-clean code. If that claim matters, use a checker-agnostic shape (e.g. `typing.cast` or a `Protocol` receiver) or add a mypy-compatible ignore.

### IN-02: Independent artifact-vs-registry checker omits the `fix`-disposition mapping branch

**File:** `tests/models/test_model_info_registry.py:445-488`
**Issue:** `test_compat_census_artifact_matches_registry` re-implements the artifact→status mapping but has no `disposition: {fix: ...}` branch (its green-load path special-cases only PlantCaduceus). The writer (`compat_verdict`) maps a fix-disposition green load to `shim-required-tf5`; the checker would derive `verified-tf5` and go red on any future fix-disposition entry. It fails loudly (by design of independent checkers), but the checker's docstring claims equivalence with the writer mapping — a comment noting the deliberate omission (or implementing the branch) would prevent confusion during the next census refresh.

### IN-03: Machine-specific evidence text committed into the packaged registry

**File:** `dnallm/models/model_info.yaml:315,485`
**Issue:** `GENERanno`'s evidence carries a double-`@` stamp (`...@unpinned@2026-10-27T08:39:46Z`), and `CrossDNA`'s evidence embeds a dev-box absolute cache path (`/home/forrest/.cache/huggingface/hub/models--chengCCC--.../config.json: [Errno 2] ...`). Both ship in the packaged `model_info.yaml` (distributed to PyPI). Honest evidence, and the docs gate correctly keeps the raw path off the generated page, but any re-census on another machine produces different evidence text, guaranteeing write-back drift for those two rows; consider normalizing the stamp form and trimming the path in `describe_config_problem`/`_hub_revision` output at the next refresh.

### IN-04: `_finalize_model_loading` wrapper skips materialization on a kwargs-only call

**File:** `dnallm/utils/transformers_compat.py:1650-1653`
**Issue:** The wrapper materializes bookkeeping only under `if args:`; a caller invoking `_finalize_model_loading(model=..., load_config=..., loading_info=...)` with no positional args delegates without materializing, and the load then fails later with the original `AttributeError`. transformers 5.17 calls positionally, so this is latent. Consider resolving the receiver from `kwargs` as a fallback (first of `model`/`self`/`cls` in `kwargs`).

### IN-05: Unpinned `curl | sh` for uv on the self-hosted box

**File:** `.github/workflows/census_compat.yml:60-62`
**Issue:** `curl -LsSf https://astral.sh/uv/install.sh | sh` fetches and executes an unpinned install script on the self-hosted GPU runner (the box already has uv configured with a mirror per runner ops; this step can also silently replace it). Pin the version (`UV_VERSION=...` with the script's env var) or skip installation when `command -v uv` succeeds.

### IN-06: `docs_blocked_reason` fails open to the raw first clause for unforeseen blocked shapes

**File:** `scripts/census_compat.py:1634`
**Issue:** The two committed blocking shapes map to biologist-friendly reasons; any future blocked entry with a different evidence shape falls through to `detail.split(";")[0]`, which can put raw error text (including local paths, as in the CrossDNA evidence) onto the generated page. A fail-closed default (raise on unknown shape, like `docs_status_cell` does for unknown statuses) would keep the page's plain-language contract.

---

_Reviewed: 2026-10-27T10:14:04Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
