# Phase 17: Publish Chain Robustness - Pattern Map

**Mapped:** 2026-10-14
**Files analyzed:** 6 (4 modified, 2 created)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `.github/workflows/publish.yml` | config (CI workflow) | event-driven (release/dispatch/cron → build/upload) | `.github/workflows/publish.yml` itself (self-edit) + `.github/workflows/ci.yml:286,611` (D-19 gates) | exact |
| `pyproject.toml` (line 86 mypy cap) | config (package metadata) | n/a (declarative) | `pyproject.toml` dev-extra block itself (adjacent pins `ruff==0.16.10`, `pre-commit>=3.6.0`) | exact |
| `.github/workflows/README.md` (trigger docs) | config (docs) | n/a | existing trigger notes in `.github/workflows/README.md` (cron default-branch note) | exact |
| `.planning/WINDOWS.md` (row 20 → fixed) | config (ledger) | n/a | existing row-20 entry + prior "fixed" transitions (e.g. #22/#23 post-merge unrun-verify items) | exact |
| Local dev venv `numpy==2.2.0` pin | environment mutation | n/a | CI legs' explicit `numpy==2.2.0` matrix pins (ci.yml legs) | role-match |
| Owner human-gate task artifacts (PyPI registration checklist + PYPI_API_TOKEN deletion gate) | plan task / ledger item | n/a | 16-01-PLAN.md task/SC-table shape; WINDOWS #22/#23 carried-ledger precedents | role-match |

All analog paths are git-tracked (verified via `git ls-files`).

## Pattern Assignments

### `.github/workflows/publish.yml` (workflow config, event-driven)

**Analogs:** itself (60 lines, fully read) + ci.yml D-19 gates. This is configuration surgery on an existing file — the pattern is "keep the working parts byte-compatible, change only the identified lines."

**Keep byte-compatible (fix history — Pitfall 6, do not break):**

Current file, lines 34-44 (install + test steps; d104e9e `-e '.[base]'`, 98f7aa5 `--system`, a8102da check_code CI path, 0281c66 tests-as-default):

```yaml
    - name: Install dependencies
      env:
        UV_HTTP_TIMEOUT: 120
        UV_CONCURRENT_DOWNLOADS: 1
      run: |
        uv pip install --system -e '.[base]'
        uv pip install --system build twine
    - name: Run tests
      run: |
        python scripts/check_code.py
```

On the release job: keep `-e '.[base]'` and check_code.py; change only line 40 (`build twine` → `'twine>=7,<8'`, drop `build`) and line 48 (`uv build` → `uv build --no-sources`). Also keep the checkout/setup-python@v7/py3.12/setup-uv@v7/cache@v4 header block (lines 12-32) unchanged.

**Upload-guard pattern (job-level event gate, replaces lines 50-55):**

```yaml
  publish:
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/dnallm/
    permissions:
      id-token: write   # job-level only, never workflow-global
      contents: read
    ...
    - name: Publish to PyPI (OIDC)
      uses: pypa/gh-action-pypi-publish@release/v1
```

The `TWINE_USERNAME`/`TWINE_PASSWORD`/`twine upload` block (lines 50-55) is deleted entirely. Gate must be job-level, not step-level (PUBS-01 "can never reach"). Must land in the SAME commit as the `schedule:` trigger.

**Smoke-gate pattern — verbatim D-19 precedent** from ci.yml:286 (03:00 entry; twin at :611 selecting `'30 5 * * *'`):

```yaml
    if: github.event_name == 'workflow_dispatch' || (github.event_name == 'schedule' && github.event.schedule == '0 3 * * *')
```

Smoke copies this shape with its own cron string `'0 8 * * 0'` (Sunday 08:00 UTC, extending the 03:00 → 05:30 stagger chain). Note ci.yml's explanatory comment style above each gate (lines 283-286, 608-611) — replicate it.

**Smoke job skeleton:** use RESEARCH.md "Pattern 3" verbatim (checkout@v4 → setup-python@v7 '3.12' → setup-uv@v7 → `uv build --no-sources` → `uv pip install --system 'twine>=7,<8'` + `twine check dist/*` → clean venv install `dist/*.whl` → import probe printing `dnallm.__version__`). No `twine upload` string anywhere in the job. runs-on `ubuntu-latest` (never the self-hosted dnallm-nightly label — see ci.yml:280,617 for the label form being avoided).

**Verify-step surface (current lines 57-60 `pip install dnallm` from PyPI):** restructure — smoke's install-from-dist IS the verify; release job either drops it or keeps it inside the gated release job (planner discretion, RESEARCH Open Question 2).

### `pyproject.toml` (config, one-line edit)

**Analog:** its own dev-extra block. Current line 86:

```toml
    "mypy>=1.15.0",
```

becomes `"mypy>=1.15.0,<2.5",`. Sibling pins show both accepted forms: `ruff==0.16.10` (exact) and `pre-commit>=3.6.0` (floor) — the cap form `>=X,<Y` matches the floor style plus an upper bound. No uv.lock in repo (`.gitignore:49`), so no lock regen. Do not touch `[tool.coverage.run] source_pkgs = ["dnallm"]` at pyproject.toml:552-553.

### `.github/workflows/README.md` (docs)

**Analog:** existing cron/default-branch knowledge already documented there. Add: new triggers (weekly cron + dispatch = smoke-only, never uploads; release event = the only upload path), the "owner does not dispatch expecting an upload" note (RESEARCH Consequence for the plan), and the coverage-CLI backup note (bare `--cov` and `coverage run -m pytest` forms — verbatim basis: `.planning/milestones/v1.2-phases/10-evaluation-contract-layer-shared-scaffolding/deferred-items.md:8-17`, quoted in RESEARCH.md Code Examples). Proper docs sweep is Phase 19 — keep this additive and trigger-scoped.

### `.planning/WINDOWS.md` row 20 (ledger edit)

**Analog:** row 20's current entry (status `open`, dated 2026-10-09). Edit pattern: status → `fixed`, add fixed-date/notes columns per the table's existing column set; reference the numpy pin + clean reprobe. Precedent for carried items: #22/#23 post-merge unrun-verify entries (also the model for the scheduled-cron-leg ledger item, Pitfall 3, and the PYPI_API_TOKEN-deletion carried item, Pitfall 5).

### Local dev venv numpy pin (environment mutation, not a committed file)

**Analog:** CI legs pin `numpy==2.2.0` explicitly (matrix legs cited ci.yml; RESEARCH: "CI-matrix-identical"). Fix: `uv pip install 'numpy==2.2.0'` in `.venv`. Ceiling must be `<2.4`, NEVER `<2.5` (Pitfall 2: 2.4.6 crashes, matrix 2.2.0 OK / 2.3.5 OK / 2.4.6 CRASH / 2.5.3 CRASH). Verification probes (must come back clean):

```bash
.venv/bin/python -m pytest tests/inference/test_mutagenesis.py --cov=dnallm.utils.sequence --co -q
.venv/bin/python -c "import numpy; assert numpy.__version__ < '2.4'"
```

Plan as an explicit task with a restore note (mutation of the shared dev venv). If the torch facet persists post-pin (A1), fall back to the documented coverage-CLI form.

### Owner human-gate task artifacts (plan tasks, not repo files)

**Analogs:** 16-01-PLAN.md structure — front-matter block (phase/plan/type/wave/depends_on/files_modified/requirements/estimate), `must_haves.truths` as numbered TRUTH-N strings with inline citations, artifacts list, key_links, and the security table (T-16-SC row at 16-01-PLAN.md:199). Copy that shape for the owner-gate plan: (1) create GitHub environment `pypi` via `gh api -X PUT repos/zhangtaolab/DNALLM/environments/pypi` (scriptable, admin verified), (2) PyPI trusted-publisher registration checklist pinned to exact strings — owner `zhangtaolab`, repo `DNALLM`, workflow filename `.github/workflows/publish.yml`, environment `pypi` (environment FIRST), (3) PYPI_API_TOKEN deletion encoded as a carried ledger item dependent on an observed green OIDC release run ID — never a phase-blocking task (Pitfall 5).

## Shared Patterns

### Event gating (D-19)
**Source:** `.github/workflows/ci.yml:286` and `:611`, with explanatory comment blocks at 283-286 / 608-611.
**Apply to:** both new publish.yml jobs (upload gate: `github.event_name == 'release'`; smoke gate: dispatch OR schedule-matching-own-cron-string disjunct, verbatim shape above).

### Atomicity invariant
**Apply to:** the single commit touching publish.yml — the upload gate is the first hunk of the same diff that adds `schedule:`. Any pushed state with `schedule:` and an ungated upload job is a defect (PUBS-01, Pitfall 1).

### Byte-compatibility set (fix history)
**Apply to:** every release-job step except the build command and upload step:
- d104e9e: `uv pip install --system -e '.[base]'` (check_code.py needs ruff/pytest/mypy; mcp notebook imports need extras)
- 98f7aa5: `--system` flag on uv pip
- a8102da: check_code.py CI=true non-interactive path
- 0281c66: tests are check_code.py's default mode (no `--with-tests` flag)
- env `UV_HTTP_TIMEOUT: 120` / `UV_CONCURRENT_DOWNLOADS: 1`

### Never `--cov=<dotted.target>` in any new CI step
**Source:** ci.yml bare `--cov` form (ci.yml:103,182,420,589) + `[tool.coverage.run] source_pkgs = ["dnallm"]` (pyproject.toml:552-553).
**Apply to:** all workflow steps; the dotted form is the WINDOWS #20 crash class.

## No Analog Found

None — all six edit surfaces have in-repo analogs (the workflow edits are self-analogous; OIDC action usage and PyPI registration are external, covered by RESEARCH.md Patterns 1-3 with verified official-doc citations).

## Metadata

**Analog search scope:** .github/workflows/, pyproject.toml, .planning/WINDOWS.md, .planning/phases/16-*/, .planning/milestones/v1.2-phases/10-*/
**Files scanned:** 8 (publish.yml full read; ci.yml targeted reads at 6 ranges; pyproject.toml 2 ranges; 16-01-PLAN.md targeted; WINDOWS.md row 20; workflows README verified tracked)
**Pattern extraction date:** 2026-10-14
