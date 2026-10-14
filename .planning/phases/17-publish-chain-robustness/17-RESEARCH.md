# Phase 17: Publish Chain Robustness - Research

**Researched:** 2026-10-14
**Domain:** GitHub Actions publish-chain hardening (event gating, OIDC Trusted Publishing, PyPI tooling) + local dev-tooling failure caps (coverage/numpy interplay, mypy version bound)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
(owner-accepted 2026-10-14)
- Upload guard = EVENT GATING: dispatch/cron can never reach `twine upload`; real publishing only on the tag/push event path; the verify step reads `dist/`, never pip-installs from PyPI on non-publish runs
- Smoke schedule: WEEKLY cron (off-peak) + manual workflow_dispatch
- WINDOWS #20 fix: LOCAL NUMPY PIN (fix the cause) with the coverage-CLI workaround documented as the backup note — NOT a coverage version bump
- OIDC sequencing: workflow-side ready first (job-level id-token: write, pypa/gh-action-pypi-publish@release/v1, publisher registered for exact owner/repo/workflow-filename/environment, no reusable-workflow factoring, no token fallback); PyPI-side trusted-publisher REGISTRATION and PYPI_API_TOKEN deletion are OWNER human-gates — token deleted only after the first observed green OIDC release

### Carried constraints (roadmap)
- twine pinned >=7,<8; redundant build dep dropped; `uv build --no-sources`; mypy capped >=1.15.0,<2.5 in dev extra
- Smoke dispatches validate Phase 18's extras surgery (position note)

### Claude's Discretion
Workflow yaml mechanics, guard predicate shape, pin placement, smoke job structure within the locked contracts.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

Verbatim from `.planning/REQUIREMENTS.md:30,34-38`:

| ID | Description | Research Support |
|----|-------------|------------------|
| PUBS-01 | "The unguarded `workflow_dispatch`→`twine upload` path is event-gated or split BEFORE any smoke trigger exists — a dry-run dispatch can never upload (PyPI filenames are immutable)" | Current publish.yml anatomy mapped line-by-line; exact guard insertion points identified (see Current Publish Chain Anatomy); guard + smoke must land in ONE atomic commit (no intermediate state where `schedule:` exists ungated) |
| PUBS-02 | "A publish smoke runs the never-uploading chain on dispatch: `uv build --no-sources` → `twine check` (twine pinned `>=7,<8`; redundant `build` dep dropped) → clean-venv install-from-dist → import probe — verification reads `dist/`, never PyPI" | `uv build --no-sources` verified present on uv 0.13.1; twine 7.0.0 verified latest on PyPI; `twine check` scope documented; smoke job skeleton provided |
| PUBS-03 | "The smoke also runs on a periodic cron schedule, schedule-aware, still never uploading" | D-19 cron-string gate pattern quoted verbatim from ci.yml:286/611; slot recommendation Sunday 08:00 UTC; default-branch-only cron caveat documented |
| PUBS-04 | "OIDC Trusted Publishing migration via `pypa/gh-action-pypi-publish@release/v1` (job-level `id-token: write`; publisher registered for the exact owner/repo/workflow-filename/environment; no reusable-workflow factoring); `PYPI_API_TOKEN` deleted only after the first observed green OIDC release" | Repo facts verified (owner/repo, no `pypi` environment yet, secret exists, token has admin); registration form fields confirmed from PyPI docs; action inputs confirmed from official README; owner human-gate checklist written |
| PUBS-05 | "WINDOWS #20 remediated — local-venv `pytest --cov` dotted-target crash fixed via a local numpy pin or the documented coverage-CLI workaround (not fixable by a coverage version bump)" | Crash reproduced live this session; numpy version boundary empirically probed (2.2.0/2.3.5 pass, 2.4.6/2.5.3 crash) — **pin ceiling must be `<2.4`, not `<2.5`**; fix placement options analyzed |
| PKGH-08 | "mypy is capped (research: `>=1.15.0,<2.5`) so CI cannot resolve an unvalidated stricter version" | Current spec `mypy>=1.15.0` at pyproject.toml:86; PyPI landscape verified (2.4.0 latest, observed resolving on the publish runner); cap semantics documented |
</phase_requirements>

## Summary

The publish chain today is a single unguarded job in `.github/workflows/publish.yml` triggered by both `release: [published]` and `workflow_dispatch`. Every step — install, check_code.py, `uv build`, `twine upload dist/*`, and a verify step that `pip install dnallm` from PyPI — runs unconditionally on either event. Run history proves the danger is real, not theoretical: the release-event run of v1.2.1 (38070839824) failed at "Install dependencies", and PyPI 1.2.1 was actually uploaded by a **workflow_dispatch run** (38112794446, 2026-10-11 04:49Z) after a 403 retry — dispatch is today the de-facto publisher of record. The run also documents the mypy drift class: the fresh runner resolved `mypy==2.4.0` from the unbounded `mypy>=1.15.0` spec and reported 136 informational errors. The remediation shape is settled by owner decision: event-gate the upload (release-only), add a never-uploading smoke (dispatch + weekly cron, D-19 schedule-aware gate), migrate the release upload to `pypa/gh-action-pypi-publish@release/v1` OIDC, pin numpy locally to fix WINDOWS #20, and cap mypy.

The highest-value empirical finding of this research: **the numpy pin ceiling must be `<2.4`, not the naive `<2.5`.** WINDOWS #20's crash was reproduced live in the current venv (numpy 2.5.3, coverage 7.16.2, py3.13), and an isolated-venv version matrix pinned the boundary precisely: numpy 2.2.0 and 2.3.5 tolerate coverage's dotted-`source_pkgs` double-import with only a `UserWarning`, while 2.4.6 and 2.5.3 raise `ImportError: cannot load module more than once per process` at `numpy/_core/multiarray.py:11` — the exact error in the ledger. numpy turned the reload warning into a hard guard in the 2.4 series (numpy PR #29030). A `<2.5` pin still resolves 2.4.x and keeps crashing. Recommended local pin: `numpy==2.2.0` (identical to the CI matrix leg, satisfying the `>=2.0.0` floor).

**Primary recommendation:** One atomic commit to publish.yml that (a) gates the upload job `if: github.event_name == 'release'` with OIDC upload via gh-action-pypi-publish, (b) adds the never-uploading smoke job gated to dispatch + its own cron string, (c) drops `build`, pins `'twine>=7,<8'`, builds with `uv build --no-sources`; plus the one-line mypy cap in pyproject, the local `numpy==2.2.0` venv pin (verified by a re-run of the dotted-target probe), and an owner-gate checklist for the PyPI-side registration and token deletion.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Upload guard (event gating) | CI workflow (publish.yml job-level `if:`) | — | PyPI filenames are immutable; the only safe place to enforce "dispatch can never upload" is before the upload step can ever be reached — the job-level gate |
| Real publishing (release event) | CI workflow → PyPI via OIDC | GitHub environment `pypi` (to be created) | Short-lived OIDC token exchange replaces the long-lived PYPI_API_TOKEN secret |
| PyPI trusted-publisher registration | External service (PyPI) — OWNER human-gate | GitHub repo admin (environment creation is scriptable) | Registration requires the PyPI account UI under 2FA; environment creation is a gh api call the phase can make |
| Publish smoke (build→check→install→import) | CI workflow (ubuntu-latest, GitHub-hosted) | — | Never touches PyPI; reads dist/ only; must not run on the self-hosted dnallm-nightly box |
| Test/coverage execution | ci.yml (existing) | Local dev venv | Phase 17 does not add pytest surface; the numpy pin only restores the local dotted-target form CI already avoids via bare `--cov` |
| mypy version bound | Package metadata (pyproject dev extra) | CI resolver | The cap acts at resolution time on every fresh runner |

## Current Publish Chain Anatomy (VERIFIED)

Full current file read this session — `.github/workflows/publish.yml` (60 lines total) [VERIFIED: .github/workflows/publish.yml:1-60]:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  publish:
    runs-on: ubuntu-latest
```

Steps (verbatim key lines):

| Lines | Step | Verbatim command | Phase-17 relevance |
|-------|------|------------------|--------------------|
| 39-40 | Install dependencies | `uv pip install --system -e '.[base]'` / `uv pip install --system build twine` | `build` is the redundant dep (uv builds natively); `twine` is UNPINNED today → becomes `'twine>=7,<8'`; `-e '.[base]'` is the d104e9e fix — keep (check_code.py needs ruff/pytest/mypy) |
| 42-44 | Run tests | `python scripts/check_code.py` | Where runs 38097323032/38106936910/38109158525 failed; mypy 2.4.0 informational drift surfaces here; keep on the release path |
| 46-48 | Build package | `uv build` | becomes `uv build --no-sources` (flag verified present on uv 0.13.1) |
| 50-55 | Publish to PyPI | env `TWINE_USERNAME: __token__` / `TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}` / `twine upload dist/*` | THE unguarded path — runs on dispatch today; replaced by event-gated OIDC job |
| 57-60 | Verify installation | `pip install dnallm` / `python -c "import dnallm; ..."` | Installs from PyPI on EVERY run incl. dispatch — the second guard surface; becomes install-from-dist |

Other anatomy facts: `actions/checkout@v4`, `actions/setup-python@v7` (python 3.12), `astral-sh/setup-uv@v7` (version "latest"), `actions/cache@v4` on `~/.cache/uv`, install env `UV_HTTP_TIMEOUT: 120` / `UV_CONCURRENT_DOWNLOADS: 1` [VERIFIED: publish.yml:14-37].

### Publish-run history (VERIFIED via gh)

| Run | Event | When (UTC) | Result | Failed step / note |
|-----|-------|-----------|--------|--------------------|
| 38070839824 | release (v1.2.1) | 2026-10-10 17:13 | failure | "Install dependencies" — the release-event path never uploaded |
| 38097323032 | dispatch | 2026-10-11 00:06 | failure | "Run tests" |
| 38106936910 | dispatch | 2026-10-11 02:59 | failure | "Run tests" |
| 38109158525 | dispatch | 2026-10-11 03:40 | failure | "Run tests" — the run that surfaced BOTH failure classes (see below) |
| 38110694132 | dispatch | 2026-10-11 04:09 | failure | "Publish to PyPI" — `HTTPError: 403 Forbidden from https://upload.pypi.org/legacy/` (upload attempted and reached PyPI) |
| 38112794446 | dispatch | 2026-10-11 04:49 | **success** | **This dispatch run uploaded dnallm 1.2.1 to PyPI** (PyPI shows 1.2.1; only versions 1.2.1, 0.5.2, 0.4.0, 0.3.0, 0.1.0 exist [VERIFIED: pip index versions dnallm]) |

Run 38109158525's two failure classes, both confirmed from its logs [VERIFIED: gh run view 38109158525 --log / --log-failed]:
1. **Blocking:** `tests/examples/test_examples.py::TestNotebookExamples::test_notebook_imports[mcp_example/...]` — missing `nest_asyncio` / `langchain_mcp_adapters` / `pydantic_ai` (fixed after by d104e9e installing `.[base]`).
2. **Informational (the PKGH-08 driver):** the runner resolved `mypy==2.4.0` from unbounded `mypy>=1.15.0` and mypy reported `Found 136 errors in 17 files (checked 83 source files)` — non-blocking (`check=False` in check_code.py, matches CI's `|| true`) but it is the unvalidated-stricter-version drift class.

**Consequence for the plan:** after PUBS-01's event gate, real publishing moves to the release path only. The owner's current working habit (dispatch-to-publish, proven above) changes: dispatch becomes smoke-only. The workflows README and/or the run book should say this explicitly so the owner does not dispatch expecting an upload.

### Exact guard insertion points

1. **Job-level gate on the upload job** (release-only): `if: github.event_name == 'release'` — release events are `types: [published]` only, i.e. a real tag/release publish. A step-level `if:` on just the twine/action step is weaker (an editing mistake one step below re-opens the path); the owner decision says "event gating... can never reach `twine upload`" — job-level is the strongest form. Splitting into two jobs (publish vs smoke) in one workflow file is the natural shape and keeps the workflow filename `publish.yml` (required for the PyPI registration to stay exact).
2. **The verify step** (lines 57-60): on non-publish runs it must read `dist/`. In the two-job split this is structural: the smoke job's install-from-dist IS the verify; the release job may keep a post-upload `pip install dnallm` (legitimate on publish runs) or also read dist/ — mechanics are Claude's discretion.
3. **Atomicity:** the `schedule:` trigger for the weekly cron and the upload guard must land in the SAME commit (PUBS-01: "BEFORE any smoke trigger exists"). Never an intermediate pushed state where `schedule:` exists without the gate — a nightly-driven `twine upload` against immutable PyPI filenames is exactly the hazard.

## Standard Stack

### Core

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| pypa/gh-action-pypi-publish | `@release/v1` (moving tag maintained by PyPA) | OIDC upload on the release path | PyPA's official trusted-publishing action; the only supported OIDC upload path for GitHub Actions [CITED: github.com/pypa/gh-action-pypi-publish] |
| twine | `>=7,<8` (7.0.0 is the only 7.x today) | `twine check dist/*` metadata/readme validation in the smoke | PyPA-official; check command validates long_description rendering, never uploads [VERIFIED: pip index versions twine → 7.0.0 latest] [CITED: twine.readthedocs.io] |
| uv (`uv build`) | runner "latest" via setup-uv@v7; `--no-sources` verified on local 0.13.1 | Build sdist+wheel against publishable metadata | `--no-sources` = "Ignore the `tool.uv.sources` table when resolving dependencies... lock against the standards-compliant, publishable package metadata" [VERIFIED: `uv build --help` on uv 0.13.1] — exactly right for a repo whose `[tool.uv.sources]` pins torch to the cu130 index |
| GitHub environment | `pypi` (to be created) | Scopes the OIDC claim; matches the PyPI registration | Optional but strongly recommended by PyPI docs (enables protections like required reviewers) [CITED: docs.pypi.org/trusted-publishers/adding-a-publisher/] |

### Supporting

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| `uv publish --dry-run` | exists on uv 0.13.1 | Optional never-uploading upload-rehearsal ("Perform a dry run without uploading files") [VERIFIED: `uv publish --help` on 0.13.1] | Optional addition to the smoke if the planner wants upload-path rehearsal beyond `twine check`; NOT required by PUBS-02 (resolves the old roadmap research flag about the flag's existence) |
| numpy (local venv only) | `==2.2.0` recommended (ceiling `<2.4`) | WINDOWS #20 fix | Local dev venv pin; NOT a pyproject dependency change (owner decision: LOCAL pin) |
| mypy | `>=1.15.0,<2.5` | PKGH-08 cap in dev extra | One-line pyproject edit; no uv.lock in repo (gitignored, `.gitignore:49` [VERIFIED]) so no lock regen |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Job-level event gate | Step-level `if:` on the upload step only | Weaker: any future step added below re-opens the dispatch→upload path; job-level matches the owner's "can never reach" wording |
| Two jobs in publish.yml | Separate smoke.yml workflow | PUBS-04 requires the publisher registration to name "the exact... workflow-filename" for the upload job — keeping upload in publish.yml means the existing filename stays the registration target; a second file would also duplicate build setup |
| `twine check` in smoke | `uv publish --dry-run` | dry-run rehearses the upload protocol but against the registry endpoint; twine check is the locked decision and validates rendering offline |

**Installation (CI-side, inside the smoke job):**
```bash
uv pip install --system 'twine>=7,<8'   # build dep dropped — uv builds natively
```

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| twine | PyPI | ~13 yrs (4.0 era 2023, project since 2013) | not retrievable via seam | github.com/pypa/twine | SUS* | Approved — see note |
| pypa/gh-action-pypi-publish | GitHub Actions marketplace (not a pypi package) | — | — | github.com/pypa/gh-action-pypi-publish | — (action, pinned by `release/v1` moving tag) | Approved — official PyPA org |
| numpy (local pin) | PyPI | — | — | github.com/numpy/numpy | — (already a project dependency; floor `>=2.0.0`) | Approved |

\* The seam's `package-legitimacy check --ecosystem pypi twine` returned verdict `SUS` with ALL signals null (`exists: null, publishedAt: null, weeklyDownloads: null, repoUrl: null`) — the checker had no PyPI registry data in this environment, so the verdict reflects data absence, not suspicion. Compensating verification performed per protocol Step 2: `pip index versions twine` returns `twine (7.0.0)` from the live PyPI registry [VERIFIED: pip index]. twine is referenced by the official pypa/gh-action-pypi-publish README (its `verify-metadata` input "runs `twine check` before upload") [CITED: github.com/pypa/gh-action-pypi-publish] and is ALREADY installed by the repo's publish.yml today (unpinned, line 40) — the phase only adds a pin, it introduces no new dependency.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** twine — disposition approved with the null-signal explanation above; no `checkpoint:human-verify` install task needed (no new install; the package is already in the working chain).

## Architecture Patterns

### System Architecture Diagram

```
                    ┌──────────────────────────────────────────────────┐
                    │  .github/workflows/publish.yml (one file)        │
                    │                                                  │
  GitHub release ───►  on: release [published]                         │
  (published)        │        │                                        │
                    │        ▼                                        │
                    │  [publish job]  if: github.event_name=='release' │
                    │   checkout → py3.12 → uv → install .[base]       │
                    │   → check_code.py → uv build --no-sources        │
                    │   → pypa/gh-action-pypi-publish@release/v1       │
                    │        │            (environment: pypi,          │
                    │        │             permissions: id-token:write)│
                    │        ▼                                        │
                    │   PyPI upload via OIDC short-lived token ────────┼──► pypi.org/project/dnallm
                    │                                                  │
  workflow_dispatch─►  on: workflow_dispatch                            │
  weekly cron ───────►  on: schedule (0 8 * * 0)                        │
                    │        │                                        │
                    │        ▼                                        │
                    │  [smoke job]  if: event=='workflow_dispatch'     │
                    │               || (event=='schedule'              │
                    │                   && event.schedule=='0 8 * * 0')│
                    │   checkout → uv → uv build --no-sources          │
                    │   → twine check dist/*   (twine>=7,<8)           │
                    │   → clean venv → install dist/*.whl              │
                    │   → import dnallm probe                          │
                    │   (NO upload step exists in this job —           │
                    │    reads dist/ only, never PyPI)                 │
                    └──────────────────────────────────────────────────┘

  Owner human-gates (outside the workflow):
   1. create GitHub environment "pypi"          (scriptable: gh api, admin verified)
   2. register trusted publisher on pypi.org    (PyPI account, 2FA)
   3. first green OIDC release observed ────────► delete PYPI_API_TOKEN secret
```

Trace the primary use case: a dispatch lands in the smoke job only; the publish job's `if:` is false, so no upload-capable step is even instantiated. A release event lands in the publish job only; the smoke's gate is false. Neither job can reach the other's path.

### Recommended Project Structure

```
.github/workflows/publish.yml   # both jobs; filename is the PyPI registration key — do NOT rename
.github/workflows/README.md     # optional: document the new triggers (docs sweep proper is Phase 19)
pyproject.toml                  # line 86: mypy cap (one line)
.planning/WINDOWS.md            # #20 → fixed after the pin + probe
(local dev venv)                # numpy==2.2.0 pin — venv-local, not committed metadata
```

### Pattern 1: D-19 schedule-aware cron gate (verbatim precedent)

**What:** Each scheduled job selects exactly its own cron entry; dispatch is the on-demand disjunct.
**When to use:** the smoke job's gate.
**Example** — verbatim from ci.yml (the 03:00 entry) [VERIFIED: .github/workflows/ci.yml:286]:

```yaml
if: github.event_name == 'workflow_dispatch' || (github.event_name == 'schedule' && github.event.schedule == '0 3 * * *')
```

The 05:30 twin at ci.yml:611 selects `'30 5 * * *'` [VERIFIED: ci.yml:611]. The smoke copies this shape with its own cron string.

### Pattern 2: OIDC publish job skeleton (release-only)

**What:** job-level permissions + environment + the pypa action.
**When to use:** the publish job.
**Example** (input names verified against the action README [CITED: github.com/pypa/gh-action-pypi-publish]):

```yaml
  publish:
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/dnallm/
    permissions:
      id-token: write   # mandatory for trusted publishing; job-level, never global
      contents: read
    steps:
      # ... checkout, python, uv, install .[base], check_code.py, uv build --no-sources ...
      - name: Publish to PyPI (OIDC)
        uses: pypa/gh-action-pypi-publish@release/v1
        # packages-dir defaults to dist/; repository-url defaults to PyPI;
        # verify-metadata runs twine check before upload by default;
        # attestations are ON by default (set `attestations: false` explicitly if unwanted)
```

Action facts the planner must honor [CITED: github.com/pypa/gh-action-pypi-publish]:
- `id-token: write` is mandatory and belongs at **job** level — "only set the `id-token: write` permission in the job that does publishing, not globally" (matches the owner's "job-level id-token: write").
- "Trusted publishing cannot be used from within a reusable workflow at this time" — matches the owner's "no reusable-workflow factoring". Composite actions and calling the action more than once per job are also unsupported.
- Defaults: `packages-dir: dist/`, `verify-metadata` on (runs `twine check` pre-upload), `attestations` on, `skip-existing` off.
- No API token or username/password is used on this path (custom registries excepted) — so the `TWINE_USERNAME`/`TWINE_PASSWORD` env block disappears with the twine upload step.

### Pattern 3: never-uploading smoke chain

```yaml
  smoke:
    if: github.event_name == 'workflow_dispatch' || (github.event_name == 'schedule' && github.event.schedule == '0 8 * * 0')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v7
        with: { python-version: '3.12' }
      - uses: astral-sh/setup-uv@v7
        with: { version: "latest" }
      - name: Build (publishable metadata)
        run: uv build --no-sources
      - name: Check distributions
        run: |
          uv pip install --system 'twine>=7,<8'
          twine check dist/*
      - name: Install from dist in a clean venv
        run: |
          uv venv /tmp/smoke-venv
          uv pip install --python /tmp/smoke-venv/bin/python dist/*.whl
      - name: Import probe
        run: /tmp/smoke-venv/bin/python -c "import dnallm; print('DNALLM', dnallm.__version__, 'installed from dist')"
```

Notes: installing the wheel pulls dnallm's full dep tree including torch (default PyPI index on the runner) — expect a few minutes; that cost IS the test (wheel metadata genuinely installable). The import probe must print `dnallm.__version__` to also catch version-metadata regressions. No `twine upload` string may appear anywhere in this job.

### Anti-Patterns to Avoid

- **Step-level-only gating of the upload:** any later-added step (or a moved upload) re-opens the dispatch path. Gate the JOB.
- **Renaming publish.yml:** the PyPI trusted-publisher registration binds to the exact workflow filename; a rename invalidates the registration (uploads start failing 403-style after working).
- **`skip-existing: true` by default:** hides immutable-filename collisions instead of surfacing them; this repo's publishing history (the 403 retry loop) argues for fail-loud.
- **A pushed intermediate state with `schedule:` before the guard:** the weekly tick itself becomes an uploader. One commit, both changes.
- **`--cov=<dotted.target>` in any new CI step:** the WINDOWS #20 class; CI uses bare `--cov` (reads `source_pkgs = ["dnallm"]` from pyproject) [VERIFIED: ci.yml:103,182,420,589; pyproject.toml:552-553].

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PyPI OIDC token exchange | Custom OIDC flow / curl to the token endpoint | pypa/gh-action-pypi-publish@release/v1 | The subject-claim matching, attestation generation, and retry semantics are the action's whole job; hand-rolled versions are the classic trusted-publishing footgun |
| Package metadata/readme validation | Ad-hoc `unzip dist/*.whl` + render checks | `twine check dist/*` | Renders the long description exactly as PyPI will; one command |
| Building distributions | `python -m build` + separate sdist/wheel steps | `uv build --no-sources` | uv is already the workflow's installer; `--no-sources` guarantees the built metadata matches what pip users see (no `[tool.uv.sources]` cuda-index leakage) |
| Cron routing | Separate workflows per schedule | `github.event.schedule == '<cron>'` gates (D-19) | Established repo pattern with two working precedents in ci.yml |
| The numpy fix | Editing coverage config / bumping coverage / patching numpy | Local venv pin `numpy==2.2.0` | PUBS-05 explicitly says not fixable by a coverage bump; the root cause is numpy's 2.4+ one-init guard (probed this session) |

**Key insight:** every hand-rolled replacement here either re-implements a security boundary (OIDC) or re-creates an existing, already-integrated repo pattern (D-19 gates). The phase is almost entirely *configuration surgery on verified facts*, which is why this research leads with anatomy and probes rather than library choices.

## Runtime State Inventory

> This phase migrates the publish path (token → OIDC) — runtime-state categories answered explicitly.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | PyPI project dnallm (versions 1.2.1, 0.5.2, 0.4.0, 0.3.0, 0.1.0 [VERIFIED: pip index versions dnallm]) — filenames immutable | None (no data migration; the guard prevents NEW bad uploads, cannot affect existing ones) |
| Live service config | (a) GitHub environments: ONLY `github-pages` exists — no `pypi` environment yet [VERIFIED: gh api repos/zhangtaolab/DNALLM/environments]; (b) PyPI trusted publishers for dnallm: NONE registered (no OIDC ever used) [ASSUMED — PyPI project manage page not inspectable without owner login]; (c) repo secret `PYPI_API_TOKEN` exists [VERIFIED: gh api .../actions/secrets] | (a) scriptable: `gh api -X PUT repos/zhangtaolab/DNALLM/environments/pypi` (token has admin:true [VERIFIED: gh api repos/zhangtaolab/DNALLM --jq .permissions]); (b) OWNER human-gate on pypi.org; (c) OWNER deletes only after first green OIDC release |
| OS-registered state | None — no OS registrations reference the publish chain | None — verified by scope review (workflow + venv + PyPI/GitHub only) |
| Secrets/env vars | `PYPI_API_TOKEN` (repo secret, used ONLY in publish.yml:53 — grep of workflows confirms the sole reference [VERIFIED]); no SOPS/other env coupling | Delete after first green OIDC release (owner gate); until then it becomes simply unused once the twine upload step is removed |
| Build artifacts / installed packages | Local dev venv `.venv` (py3.13.15): numpy 2.5.3, mypy 2.4.0, coverage 7.16.2, pytest 9.1.1, pytest-cov 7.1.0, torch 2.11.0+cu130 [VERIFIED: pip list]; no uv.lock tracked (`.gitignore:49`) | numpy pin to ==2.2.0 in this venv (mutation of the shared dev venv — plan as an explicit task with restore note); mypy already at 2.4.0 (cap-compatible) |

**The canonical question** — after every file in the repo is updated, what still holds old state? (1) PyPI's publisher table (owner must register), (2) the GitHub environments list (scriptable), (3) the PYPI_API_TOKEN secret (owner deletes later), (4) the local venv numpy (explicit pin task). All four are in the plan.

## Common Pitfalls

### Pitfall 1: The intermediate-state upload
**What goes wrong:** the schedule trigger and the upload guard land in separate commits/runs; the nightly cron fires between them and `twine upload dist/*` executes against an already-published version.
**Why it happens:** natural task-splitting of "add smoke" and "gate upload".
**How to avoid:** one atomic commit — the guard is the first hunk of the same diff that adds `schedule:`. PUBS-01's "BEFORE any smoke trigger exists" is satisfied only atomically.
**Warning signs:** any pushed state where publish.yml contains `schedule:` but the upload step/job lacks `if: github.event_name == 'release'`.

### Pitfall 2: The `<2.5` numpy ceiling is WRONG — it must be `<2.4`
**What goes wrong:** pinning `numpy<2.5` per the roadmap's phrasing still resolves 2.4.x — and every 2.4.x crashes with the one-init guard.
**Why it happens:** the roadmap deferred-note said "numpy 2.5.x cannot be instrumented" (the observed venv version), so `<2.5` looks right — but the guard shipped in the 2.4 series.
**How to avoid:** pin `numpy==2.2.0` (CI-matrix-identical) or cap `<2.4`. Empirical matrix this session: 2.2.0 OK / 2.3.5 OK / 2.4.6 CRASH / 2.5.3 CRASH.
**Warning signs:** after the pin, `python -c "import numpy; print(numpy.__version__)"` in the dev venv shows anything ≥2.4.0.

### Pitfall 3: Cron fires only from the default branch
**What goes wrong:** the weekly smoke never fires while the work lives on the v1.3 branch; the phase looks "done" with the schedule leg unobserved.
**Why it happens:** "GitHub runs cron schedules only from the default branch" [VERIFIED: .github/workflows/README.md — established repo knowledge].
**How to avoid:** verify the dispatch leg on the working branch; record the scheduled leg as a post-merge unrun-verify ledger item (precedents: WINDOWS #22/#23 phase-13/14 nightly confirmations).
**Warning signs:** planning a phase-close gate that requires an observed scheduled run.

### Pitfall 4: Registration/workflow mismatch (filename, environment)
**What goes wrong:** OIDC upload fails with permission errors because the PyPI registration's workflow filename or environment name does not byte-match the workflow (or the environment doesn't exist when PyPI validates).
**Why it happens:** rename, reusable-workflow factoring, or forgetting to create the `pypi` environment before registering.
**How to avoid:** registration checklist pinned to exact strings: owner `zhangtaolab`, repository `DNALLM`, workflow filename `.github/workflows/publish.yml`, environment `pypi` (create the environment FIRST — the phase can do this via gh api).
**Warning signs:** first release run failing at the publish step with a trusted-publisher claim mismatch.

### Pitfall 5: Deleting PYPI_API_TOKEN too early
**What goes wrong:** token deleted before any green OIDC release; if the OIDC path has a defect, there is no working publisher at all.
**How to avoid:** strictly the owner's sequencing (already locked): delete only after the first observed green OIDC release. Realistically the first OIDC release is the v1.3 milestone close release — the deletion gate likely outlives phase-17 close; encode it as a carried ledger item, not a phase-blocking task.
**Warning signs:** a plan task that deletes the secret without a dependency on an observed green release run ID.

### Pitfall 6: Breaking the working parts of the current chain
**What goes wrong:** the refactor removes things that recent runs depended on: `-e '.[base]'` install (d104e9e — check_code.py needs ruff/pytest/mypy and the mcp notebook import checks need the extras), the `--system` flag on uv pip (98f7aa5 — new uv no longer implicit system env), `UV_HTTP_TIMEOUT`/`UV_CONCURRENT_DOWNLOADS`, check_code.py's CI=true non-interactive path (a8102da), and the 0281c66 fact that tests are check_code.py's default mode (no `--with-tests` flag).
**How to avoid:** keep the release job's install/test steps byte-compatible; only the upload step and the build command change on that path.
**Warning signs:** any release-job diff touching lines other than the build command, the upload step, and the job header.

## Code Examples

### WINDOWS #20 — verified repro and boundary probes (run this session)

Current venv repro (numpy 2.5.3) — dotted cov target, verbatim output:

```
$ .venv/bin/python -m pytest tests/inference/test_mutagenesis.py --cov=dnallm.utils.sequence --co -q
ImportError while loading conftest '/home/forrest/Github/DNALLM/tests/conftest.py'.
tests/conftest.py:8: in <module>    import pandas as pd
.venv/lib/python3.13/site-packages/pandas/__init__.py:13: in <module>    raise ImportError(
E   ImportError: Unable to import required dependency numpy. Please see the traceback for details.
```

Minimal probes isolating the two terminal errors (numpy-only and torch facets):

```
$ .venv/bin/python -I -c "
from coverage import Coverage
cov = Coverage(source_pkgs=['dnallm.utils.sequence'])
cov.start()
import numpy"
ImportError: cannot load module more than once per process
  (numpy/_core/multiarray.py:11: from . import _multiarray_umath, overrides)

$ ... same, import torch ...
  File ".../torch/overrides.py", line 1779, in <module>
    has_torch_function = _add_docstr(...)
RuntimeError: function '_has_torch_function' already has a docstring
```

Isolated-venv version matrix (py3.13, coverage 7.16.2, synthetic package whose `__init__` imports numpy under a dotted `source_pkgs` tracer):

```
numpy 2.2.0 → PROBE OK  (UserWarning: The NumPy module was reloaded)
numpy 2.3.5 → PROBE OK  (same warning)
numpy 2.4.6 → ImportError: cannot load module more than once per process
numpy 2.5.3 → ImportError: cannot load module more than once per process
```

Boundary confirmed externally: the hard guard comes from numpy PR #29030, first shipped in the 2.4 series; 2.3.x still warns [CITED: DataDog/dd-trace-py#18276 and numpy/_core sources — same error class]. The mechanism matches coverage's dotted-`source_pkgs` resolution importing the target package chain, then pytest-cov's early import re-importing it: ddtrace hit the identical guard via module-unload-then-reimport.

### The fix and its verification probe

```bash
# Fix (local dev venv only — owner decision: LOCAL pin):
uv pip install 'numpy==2.2.0'          # CI-matrix-identical; satisfies >=2.0.0 floor
# (alternative ceiling form: 'numpy<2.4' — NOT '<2.5', which still resolves 2.4.x)

# Verification probe (plan task — must come back clean):
.venv/bin/python -m pytest tests/inference/test_mutagenesis.py --cov=dnallm.utils.sequence --co -q
.venv/bin/python -c "import numpy; assert numpy.__version__ < '2.4'"
```

Note the residual warning on the fixed path: numpy <2.4 emits `UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result in small but subtle issues and is discouraged.` — benign, and CI has run this shape (bare `--cov`) for months.

### Backup note (documented workaround — owner decision says document, not prefer)

Verbatim from the phase-10 deferred-items record [VERIFIED: .planning/milestones/v1.2-phases/10-evaluation-contract-layer-shared-scaffolding/deferred-items.md:8-17]:

> `pytest --cov=<dotted.dnallm.target>` (pytest-cov 7.1.0, torch 2.11.0+cu130, Python 3.13 ...) crashes at conftest load with `RuntimeError: function '_has_torch_function' already has a docstring` (torch.overrides double-execution) once pytest-cov resolves the dotted cov target and imports the `dnallm` package early; the pandas/numpy ImportError shown first is a masking cascade. ... bare `--cov` (source_pkgs from pyproject) works, and `coverage run -m pytest` + `coverage report` works.

I.e. two fallback forms that work on ANY numpy: bare `--cov` (config-driven `source_pkgs = ["dnallm"]`) and the coverage-CLI form `coverage run --include=<path> -m pytest` / `coverage run -m pytest` + `coverage report` — the form Phases 10/15 already used for module-row proofs.

### mypy cap

Current dev-extra spec [VERIFIED: pyproject.toml:86]: `"mypy>=1.15.0",`
Becomes: `"mypy>=1.15.0,<2.5",`

Registry landscape [VERIFIED: pip index versions mypy]: latest is 2.4.0 (series: 2.4.0, 2.3.1, 2.3.0, 2.2.0, 2.1.0, 2.0.0, 1.20.2, ... 1.15.0, ...). The cap admits the OBSERVED-and-tolerated runner resolution (mypy==2.4.0 resolved in run 38109158525 [VERIFIED: run log]; local venv also has 2.4.0) while excluding the next unvalidated stricter release — exactly PKGH-08's "cannot resolve an unvalidated stricter version". The 136 informational errors under 2.4.0 remain non-blocking (`check=False` in check_code.py Step 4, matching CI's `|| true`) — the cap does NOT need to silence them, only to freeze the class of behavior.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Long-lived PyPI API tokens in CI secrets | OIDC Trusted Publishing (short-lived tokens minted per-run) | PyPI GA since 2024; pypa action `release/v1` is the standard | The `master` branch version of the action was sunset — migrate to `release/v1`, a tag, or a pinned SHA [CITED: action README]; PYPI_API_TOKEN becomes removable |
| `twine upload` with `TWINE_PASSWORD` | `pypa/gh-action-pypi-publish` with `id-token: write` | — | The env-var auth block disappears entirely on the release path |
| numpy reload tolerated with a warning | Hard `ImportError` guard (one-init) | numpy 2.4 series (PR #29030) [CITED: web sources + local probe] | Any tool that imports→unloads→re-imports numpy (ddtrace's cleanup, coverage's dotted source_pkgs) breaks on numpy ≥2.4 |
| `python -m build` | `uv build` (+`--no-sources` for publishable metadata) | uv build GA | The `build` pip package is redundant in a uv-based workflow |

**Deprecated/outdated:**
- pypa/gh-action-pypi-publish `@master`: sunset — use `@release/v1`.
- Trusted publishing from reusable workflows: explicitly unsupported today — do not factor the publish job out.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The numpy pin fixes the FULL dotted-target pytest crash in the real dev venv (torch facet included), not just the numpy-guard facet | WINDOWS #20 | Medium — root cause (numpy one-init guard) is verified in isolation and the torch docstring error is its downstream symptom, but the end-to-end `pytest --cov=dnallm.utils.sequence` was NOT re-run post-pin (would require mutating the shared dev venv mid-research). Mitigation: the plan's first #20 task is pin-then-reprobe; if the torch facet somehow persists, fall back to the documented coverage-CLI form (PUBS-05 permits either) |
| A2 | Weekly slot Sunday 08:00 UTC (`0 8 * * 0`) is acceptable | Smoke schedule | Low — mechanics are Claude's discretion; chosen to extend the 2.5h stagger chain (03:00 → 05:30 → 08:00) and the smoke runs on GitHub-hosted ubuntu-latest, so it never contends for the single self-hosted dnallm-nightly runner |
| A3 | Environment name `pypi` (the action README convention) | OIDC | Low — any name works if the workflow, the GitHub environment, and the PyPI registration all byte-match; `pypi` is the documented convention |
| A4 | Attestations left ON (default) on the OIDC publish | OIDC | Low — PEP 740 attestations are supported by pypi.org; planner may set `attestations: false` explicitly if the owner dislikes the provenance files on the project page. Decide explicitly rather than by omission |
| A5 | The first real OIDC release will be the v1.3 milestone close release; the PYPI_API_TOKEN-deletion gate therefore likely outlives phase-17 close as a carried ledger item | OIDC sequencing | Low — if a release happens sooner, the gate closes sooner; plan encodes the dependency, not the date |
| A6 | twine 7.0.0 `check` semantics unchanged from documented behavior | Smoke chain | Low — twine docs are for current stable (7.x); the command's contract (render-check, no upload) has been stable across majors |
| A7 | No trusted publisher is currently registered on PyPI for dnallm | Runtime State Inventory | None operationally (registering a fresh publisher is idempotent-ish: one project can have multiple publishers) — unverifiable without owner's PyPI login; stated as assumed |
| A8 | Runner uv (setup-uv "latest") ≥ 0.13.1, so `--no-sources` (verified locally on 0.13.1) exists on the runner | Smoke chain | Low — runners resolve newer-than-local; the flag has existed across recent releases |

## Open Questions

1. **Exact numpy pin form — `==2.2.0` vs `<2.4`?**
   - What we know: both fix the crash (probed); CI pins `numpy==2.2.0` explicitly on its legs; pyproject floor is `>=2.0.0`.
   - What's unclear: whether the owner wants the venv frozen to the CI-identical version or allowed to track 2.3.x.
   - Recommendation: `==2.2.0` (maximal CI-parity, zero drift surface); record the alternative in the plan.
2. **Does the release job keep a post-upload verify step, and from where?**
   - What we know: the old verify pip-installs from PyPI (forbidden on non-publish runs only); the release job IS a publish run.
   - Recommendation: drop the old verify on the release job and rely on the smoke chain's install-from-dist + import probe shape for artifact verification (uniform, and one less PyPI-polluting moving part); alternatively keep `pip install dnallm` gated inside the release job. Either satisfies the locked decision — planner picks.
3. **Smoke python version** — 3.12 (matches the current publish job) vs 3.13 (matches local dev where the #20 crash lives)? Recommendation: 3.12, byte-compatible with today's job; the smoke is about packaging, not interpreter coverage (ci.yml owns 3.11–3.13).
4. **Attestations on/off** (A4) and `print-hash: true` for release-run evidence? Recommendation: attestations on (default), `print-hash: true` for the evidence trail — decide at plan time.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| gh CLI (authenticated, admin on zhangtaolab/DNALLM) | environment creation, run observation, secret deletion | ✓ | gh, account forrestzhang, permissions admin:true [VERIFIED] | — |
| uv | local build rehearsal, venv ops | ✓ | 0.13.1 (aarch64) [VERIFIED] | — |
| Python 3.13 + project venv | #20 probes | ✓ | 3.13.15; numpy 2.5.3 / mypy 2.4.0 / coverage 7.16.2 [VERIFIED] | — |
| PyPI registry reachability | version checks | ✓ | pip index queries succeeded this session [VERIFIED] | — |
| GitHub-hosted ubuntu-latest runners | smoke + publish jobs | ✓ (assumed, standard pool) | — | The smoke must NOT use the self-hosted dnallm-nightly label (that box is the nightly queue) |
| Owner PyPI account (2FA) | trusted-publisher registration | ✗ (owner human-gate by design) | — | None — sequencing locked: workflow-side first |

**Missing dependencies with no fallback:** none blocking execution — the PyPI-account items are owner human-gates by the locked sequencing, not blockers.
**Missing dependencies with fallback:** none.

## Security Domain

`security_enforcement: true`, ASVS level 1 [VERIFIED: .planning/config.json]. This phase is CI/CD supply-chain surface — the applicable categories:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes (CI→PyPI auth) | OIDC Trusted Publishing replaces the long-lived bearer secret — short-lived per-run tokens minted only for the exact registered publisher (owner/repo/workflow/environment) |
| V3 Session Management | yes (token lifecycle) | 15-minute OIDC token lifetime; PYPI_API_TOKEN (non-expiring) deleted after first green OIDC release |
| V4 Access Control | yes | Job-level `id-token: write` (minimum scope, publish job only — never workflow-global); GitHub environment `pypi` as the claim anchor; no reusable-workflow indirection |
| V5 Input Validation | yes (artifact integrity) | `twine check dist/*` + action's default `verify-metadata` before upload; build from clean checkout with `--no-sources` (no local-source leakage into published metadata) |
| V6 Cryptography | no (no custom crypto) | OIDC exchange handled entirely by the pypa action |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Compromised/dispatched workflow uploading to PyPI | Spoofing/Elevation | Event-gated upload job (PUBS-01) — dispatch/cron structurally cannot instantiate the upload step |
| Long-lived API token exfiltration | Information Disclosure | Token removed from the release path immediately; secret deleted after first green OIDC release (owner gate) |
| Overbroad OIDC claim (workflow reused/renamed) | Elevation | Registration binds exact owner/repo/workflow-filename/environment; no reusable-workflow factoring; filename never renamed |
| Typosquatted/mistyped action pin | Spoofing | `pypa/gh-action-pypi-publish@release/v1` (official PyPA org; optionally pin a SHA at plan time) |
| Unvalidated metadata reaching PyPI (immutable) | Tampering | `twine check` in smoke AND action's default `verify-metadata`; fail-loud upload (`skip-existing` off) |

## Sources

### Primary (HIGH confidence)
- `.github/workflows/publish.yml:1-60` — full current anatomy, read this session
- `gh run list/view` on workflow publish.yml (runs 38070839824, 38097323032, 38106936910, 38109158525, 38110694132, 38112794446) — trigger/failure/upload history, resolved `mypy==2.4.0`, 136-error output
- `gh api` — environments (`github-pages` only), secrets (`PYPI_API_TOKEN`), repo permissions (admin:true)
- Live probes this session — WINDOWS #20 repro (both facets) + numpy matrix 2.2.0/2.3.5/2.4.6/2.5.3 in isolated venvs
- `pip index versions` — mypy (2.4.0 latest), twine (7.0.0), numpy (2.5.4 latest), dnallm (1.2.1 published versions)
- `uv build --help` / `uv publish --help` on uv 0.13.1 — `--no-sources` and `--dry-run` verified present
- `.github/workflows/ci.yml:286,611,103,182,420,589` — D-19 gate pattern and bare `--cov` form
- `.planning/WINDOWS.md` row 20; phase-10 `deferred-items.md:8-17` — ledger and workaround verbatim
- `pyproject.toml:86,552-553` — mypy spec and coverage source_pkgs verbatim
- git log/show: d104e9e, a8102da, 0281c66, 98f7aa5 — the publish-chain fix history that must not be broken

### Secondary (MEDIUM confidence)
- [pypa/gh-action-pypi-publish README](https://github.com/pypa/gh-action-pypi-publish) — inputs, defaults, job-level id-token, reusable-workflow restriction
- [PyPI docs — adding a trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) — registration fields (owner/repo/workflow filename required; environment optional-but-recommended)
- [PyPI docs — trusted publishers overview](https://docs.pypi.org/trusted-publishers/) — 15-minute token exchange
- [twine docs](https://twine.readthedocs.io/en/stable/#twine-check) — `twine check` scope

### Tertiary (LOW confidence)
- numpy 2.4 guard introduction (PR #29030) — corroborated via [dd-trace-py issue #18276](https://github.com/DataDog/dd-trace-py/issues/18276) and [PR #18919](https://github.com/DataDog/dd-trace-py/pull/18919) (same error class, numpy ≥2.4); the boundary itself is PRIMARY here (own probe)

## Metadata

**Confidence breakdown:**
- Publish-chain anatomy + run history: HIGH — file read + gh evidence, both facets of the drift note confirmed
- WINDOWS #20 root cause + fix boundary: HIGH — reproduced and probed in four numpy versions; only the full-venv end-to-end post-pin re-run is outstanding (A1)
- OIDC specifics: HIGH on repo facts (gh api), MEDIUM-HIGH on action/registration contract (official docs cited; not yet exercised in this repo)
- Cron/schedule mechanics: HIGH — D-19 precedents in-repo; default-branch caveat documented

**Research date:** 2026-10-14
**Valid until:** 2026-11-14 (stable domain; revisit if pypa action or PyPI publisher policy changes before execution)
