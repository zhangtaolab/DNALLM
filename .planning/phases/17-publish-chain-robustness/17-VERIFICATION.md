---
phase: 17-publish-chain-robustness
verified: 2026-10-15T15:42:40Z
status: passed
score: 7/8 must-haves verified
covered_files:
  - .github/workflows/publish.yml
  - .planning/WINDOWS.md
  - .planning/phases/17-publish-chain-robustness/17-01-PLAN.md
  - .planning/phases/17-publish-chain-robustness/17-01-SUMMARY.md
  - .planning/phases/17-publish-chain-robustness/17-02-PLAN.md
  - .planning/phases/17-publish-chain-robustness/17-02-SUMMARY.md
  - .planning/phases/17-publish-chain-robustness/17-03-PLAN.md
  - .planning/phases/17-publish-chain-robustness/17-03-SUMMARY.md
  - pyproject.toml
covered_digest: "v3:sha256:9150d4113e128ab67bb23cb67d60564473a24f2c60a52fcb3f04d2848d544808"
behavior_unverified: 0
overrides_applied: 0
deferred: # Disclosed standing items — NOT gaps (owner human-gates / time-gated observations outside phase control)
  - truth: "PUBS-04 owner-side clause: PyPI trusted-publisher registration (and PYPI_API_TOKEN deletion after the first green OIDC release)"
    addressed_in: "Owner action at v1.3 milestone close — WINDOWS #25 standing item; 17-03 halted at the blocking-human checkpoint by owner deferral (稍后再做, 2026-10-14)"
    evidence: "WINDOWS #25 open with resume path; 17-03-SUMMARY status: halted, requirements-completed deliberately empty, five exact registration strings recorded (dnallm / zhangtaolab / DNALLM / .github/workflows/publish.yml / pypi); gh api repos/zhangtaolab/DNALLM/actions/secrets still lists PYPI_API_TOKEN — the deletion gate is honored by NOT deleting (no green OIDC release exists yet)"
  - truth: "Scheduled smoke leg (weekly cron 0 8 * * 0) first live firing"
    addressed_in: "Post-merge to the default branch (17-01 AS-5 carried obligation — GitHub runs cron schedules only from the default branch; workflow currently on v1.3)"
    evidence: "publish.yml schedule trigger + D-19 schedule-aware smoke gate landed atomically in 2c649fa (pickaxe-verified, single introducing commit for gate + cron strings); README documents the default-branch-only cron semantics; the same D-19 gate pattern is live-proven in ci.yml nightly jobs"
  - truth: "17-03 Task 3: .github/workflows/README.md publish-workflow trigger-semantics section + carried-ledger heading"
    addressed_in: "WINDOWS #25 resume unit — 17-03 'completes in minutes' once the owner registers (registration record + README trigger docs + token-deletion carried item)"
    evidence: "17-03-SUMMARY 'Remaining Work at Resume' item 2; current .github/workflows/README.md headings enumerated — documents ci.yml jobs only, no publish-workflow section (deliberately not written past the owner gate)"
---

# Phase 17: Publish Chain Robustness Verification Report

**Phase Goal:** The publish chain can be exercised end-to-end — build, check, install-from-dist, import — on demand and on schedule with zero possibility of uploading, the release path migrates to OIDC Trusted Publishing, and the known dev-tooling failure classes (mypy drift, the #20 cov crash) are capped or fixed
**Verified:** 2026-10-15T15:42:40Z (against current tree at HEAD 6b397d1, branch v1.3)
**Status:** passed
**Re-verification:** No — fresh goal-backward verification. No prior 17-VERIFICATION.md existed on the tree or in git history; this report supersedes the stale-verification condition (phase 18/19 commits 399f9f1/165800d touched publish.yml/pyproject/README after the earlier verifier run).

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1/PUBS-01(a): a dispatch/cron event can structurally never instantiate an upload-capable step — the upload job is gated at JOB level | ✓ VERIFIED | `publish.yml:18` `if: github.event_name == 'release'` on the `publish` job; release trigger is `types: [published]` only (`:4-5`). The legacy token-auth upload step (`TWINE_USERNAME: __token__` / `TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}` / `twine upload dist/*`) is deleted — transform diff `2c649fa^..2c649fa` shows the removal. Live: dispatch runs 38678155010 (final tree, headSha 15bf6460, publish.yml byte-identical to HEAD) and 38554913144 (headSha = 2c649fa itself) both show `publish` job conclusion **skipped**, zero steps instantiated |
| 2 | SC1/PUBS-01(b): no run installs the package from PyPI outside a publish run — verify reads `dist/` | ✓ VERIFIED | Old `pip install dnallm` + registry verify step deleted entirely (transform diff); the only artifact-install path is `uv pip install --python /tmp/smoke-venv/bin/python dist/*.whl` (`:123`) inside the smoke job. Negative greps clean: no `PYPI_API_TOKEN`, no `twine upload`, no package registry install, no `index-url` |
| 3 | SC2/PUBS-02: a dispatched run executes the never-uploading chain green — `uv build --no-sources` → `twine check` (twine `>=7,<8`, build dep dropped) → clean-venv install-from-dist → import probe | ✓ VERIFIED | Live run 38678155010 (workflow_dispatch, v1.3): smoke job **success**, every step green — Build package / Check distributions / Install from dist in a clean venv / **Import probe**, run log line `DNALLM 1.2.1 installed from dist` (2026-10-15T03:49:27Z). Twine spec `'twine>=7,<8'` at `:60`/`:114`; transform diff shows `build twine` → `'twine>=7,<8'` (build dep dropped) and `uv build` → `uv build --no-sources`. Identical proof at transform commit via run 38554913144 |
| 4 | SC3/PUBS-03: smoke also runs on a weekly cron, schedule-aware, landed atomically WITH the upload guard | ✓ VERIFIED | Trigger `schedule: cron '0 8 * * 0'` (`:7-8`); smoke gate `github.event_name == 'workflow_dispatch' \|\| (github.event_name == 'schedule' && github.event.schedule == '0 8 * * 0')` (`:89`) — D-19 byte-equal cron-string shape. Pickaxe: the cron string, the release gate, and the smoke gate all have the SAME single introducing commit **2c649fa** (atomicity — the guard could never exist without the trigger). The scheduled leg's first live firing is a carried post-merge item (cron fires only from the default branch; AS-5) — see Deferred |
| 5 | SC3/PUBS-04 (workflow-side): the release path is OIDC-ready | ✓ VERIFIED | `pypa/gh-action-pypi-publish@release/v1` with `print-hash: true` + `attestations: true` (`:76-79`); job-level `permissions: id-token: write, contents: read` (`:28-30`); NO workflow-global permissions key (grep clean); `environment: name: pypi` (`:23-25`); no `workflow_call` trigger (reusable-workflow factoring absent, grep clean); filename unchanged (`.github/workflows/publish.yml`, continuous history); no token fallback (secret name absent from file). GitHub environment **pypi verified live**: `gh api .../environments` → id 24275955008, created 2026-10-14T11:48:46Z, 0 protection rules |
| 6 | SC4/PUBS-05: WINDOWS #20 remediated via a local numpy pin OR the documented coverage-CLI workaround (not a coverage bump) | ✓ VERIFIED (coverage-CLI arm — the sanctioned alternative; see note below) | WINDOWS.md row 20 **fixed** (resolved 2026-10-14T11:37:34Z) with the live two-facet probe record: matrix 2.2.0 OK / 2.3.5 OK / 2.4.6 CRASH / 2.5.3 CRASH (ceiling <2.4 recorded); the numpy==2.2.0 pin cleared the pandas/numpy facet but the torch facet (`torch/overrides.py:1779`) proved numpy-independent → per the plan's own AS-1 fallback contract, numpy was restored to 2.5.3 (venv dist-info confirms numpy-2.5.3) and the two always-working coverage-CLI forms are THE recorded fix. **Verifier re-proved form 1 live on the current tree:** `pytest tests/inference/test_mutagenesis.py --cov --co -q` → exit 0, 45 tests collected (10.4s) — matching the ledger record exactly. Not a coverage bump: the only pyproject change of the phase's dev-tooling commit (8e2e1b9) is the mypy line (numstat 1/1); the coverage spec `coverage[toml]>=7.10.6` and `source_pkgs = ["dnallm"]` are untouched on the current tree |
| 7 | SC5/PKGH-08: mypy capped `>=1.15.0,<2.5` in the dev extra | ✓ VERIFIED | `pyproject.toml:86` `"mypy>=1.15.0,<2.5"` inside the `dev` extra; introducing commit 8e2e1b9 numstat `1 1 pyproject.toml` (one-line edit as planned). Semantics: 2.4.0 (the observed runner drift) admitted, 2.5.0 excluded. Reachable from the publish runner via `full` → `dnallm[dev,...]` chain (post-phase-18 rename; deprecated `base` alias still installs the full set) |
| 8 | SC3/PUBS-04 (owner-side): trusted-publisher registered for the exact owner/repo/workflow-filename/environment; PYPI_API_TOKEN deleted only after the first green OIDC release | ⏸ DEFERRED (owner human-gate — disclosed, not a gap) | Registration NOT done: owner explicitly deferred (稍后再做, 2026-10-14) at the 17-03 blocking-human checkpoint — WINDOWS **#25 open** with the resume path; 17-03-SUMMARY `status: halted`, `requirements-completed: []` (honest non-completion); the five exact strings are on record in the SUMMARY Task 2 table. Token-deletion sequencing HONORED: `gh api .../actions/secrets` still lists `PYPI_API_TOKEN` and no workflow references it — correctly not deleted before a green OIDC release exists. REQUIREMENTS.md honestly shows PUBS-04 unchecked/Pending. Per the phase contract (17-CONTEXT owner-accepted decisions), workflow-side-ready is the phase deliverable; PyPI-side registration and token deletion are owner human-gates |

**Score:** 7/8 truths verified (1 disclosed owner-deferred item; 0 behavior-unverified)

**Note on truth 6 (mechanism switch, disclosed):** 17-02 PLAN TRUTH-1 as literally written ("the shared dev venv numpy is pinned ==2.2.0") is NOT the delivered state — the live reprobe disproved the single-cause hypothesis (torch facet survives any numpy), so the plan's own AS-1 fallback clause ("fall back to documenting the coverage-CLI workaround as THE fix — PUBS-05 sanctions either path") was exercised. The roadmap SC4 and PUBS-05 requirement text sanction BOTH arms ("via a local numpy pin **or** the documented coverage-CLI workaround"), so the contract-level truth is satisfied by the delivered workaround arm; the switch is fully recorded in WINDOWS row 20 and 17-02-SUMMARY. No override required — this is in-plan sanctioned branching, not a deviation.

### Deferred Items

Disclosed standing items — intentionally outside what this phase could close, tracked in the WINDOWS ledger; NOT gaps and NOT counted against the score.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | PyPI trusted-publisher registration (five exact strings) + eventual PYPI_API_TOKEN deletion | Owner action at v1.3 milestone close (WINDOWS #25; 17-03 resume: register at pypi.org/manage/project/dnallm/publishing, then re-run /gsd-execute-phase 17) | WINDOWS #25 open; 17-03-SUMMARY halted-state record; token still present (correct) |
| 2 | Scheduled smoke leg first live firing (cron 0 8 * * 0) | Post-merge to default branch (17-01 AS-5 carried obligation) | Trigger + D-19 gate landed atomically (2c649fa); unobservable pre-merge by GitHub design |
| 3 | 17-03 Task 3 README trigger docs + carried-ledger heading | WINDOWS #25 resume unit (minutes of work once the owner registers) | Current .github/workflows/README.md has no publish-workflow section (ci.yml jobs only); 17-03-SUMMARY Remaining Work item 2 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/publish.yml` | Three triggers (release/workflow_dispatch/schedule) + two jobs (publish = release-only OIDC; smoke = dispatch + weekly cron, never uploading), one atomic commit | ✓ VERIFIED | All elements present on current tree; atomicity pickaxe-proven (2c649fa, single commit, 1 file, 76+/9-); post-phase-18 change (399f9f1, extras rename) preserves every phase-17 contract element — verified line-by-line on current tree |
| `pyproject.toml` | mypy `>=1.15.0,<2.5` in dev extra (one-line edit) | ✓ VERIFIED | Line 86; commit 8e2e1b9 numstat 1/1; coverage config + numpy floor untouched |
| `.planning/WINDOWS.md` | Row 20 open → fixed with evidence + backup note; row 25 tracking the owner gate | ✓ VERIFIED | Row 20 fixed with two-facet evidence, <2.4 ceiling, both coverage-CLI forms; row 25 open with resume path; frontmatter counts consistent (6 open + 3 waived + 18 fixed = 27) |
| GitHub environment `pypi` (external state) | Exists, created before any registration ask (Pitfall 4 ordering) | ✓ VERIFIED | Live `gh api`: id 24275955008, created 2026-10-14T11:48:46Z, 0 protection rules (deliberate — no manual-approval dependency on the release path) |
| 17-01/17-02/17-03-SUMMARY.md | Live records (run ids, probe evidence, halted-state disclosure) | ✓ VERIFIED | All three present and substantive; 17-01/17-02 claims independently cross-checked against gh run data, git history, and re-run probes — all match |
| `.venv` numpy | (Plan-as-written: pinned ==2.2.0; delivered: restored 2.5.3 per sanctioned AS-1 fallback) | ✓ VERIFIED (as delivered) | numpy-2.5.3 dist-info present; the pin was a live probe step whose insufficiency is the recorded root-cause finding, not a regression |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| publish job if-gate | `github.event_name == 'release'` | job-level `if` (publish.yml:18) | ✓ WIRED | Release trigger is `types: [published]` only; live dispatch runs prove the gate denies dispatch events (publish skipped) |
| schedule cron `0 8 * * 0` | smoke gate's `github.event.schedule` match | D-19 byte-equal cron-string gate (publish.yml:8 ↔ :89) | ✓ WIRED | Strings byte-equal; same-pattern gates live-proven in ci.yml nightlies |
| publish job `environment: pypi` | GitHub environment `pypi` → PyPI trusted-publisher registration | OIDC claim environment anchor | ✓ WIRED (to environment) / ⏸ registration end pending (owner, #25) | Environment verified live via gh api (id 24275955008); registration is the disclosed owner gate |
| `uv build --no-sources` → `dist/` | twine check / smoke wheel install / pypa action packages-dir | `dist/*` consumed at :115, :123; action default packages-dir | ✓ WIRED | Live log proves the full chain: build → check → install → `DNALLM 1.2.1 installed from dist` |
| pyproject mypy cap | publish runner resolution (check_code.py path) | publish job installs `-e '.[full]'` → `dnallm[dev,...]` → capped mypy | ✓ WIRED | full extra self-references dev (pyproject:149); cap reaches the runner that exhibited the 2.4.0 drift |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| publish.yml smoke chain | `dist/*` | `uv build --no-sources` on the checked-out ref | Yes — live run 38678155010 consumed real artifacts through import probe | ✓ FLOWING |
| publish.yml OIDC publish | id-token exchange | GitHub OIDC issuer → PyPI registered publisher | Workflow-side ready; exchange end gated on owner registration (#25) | ✓ FLOWING (workflow side) |
| WINDOWS row 20 record | repro matrix + probe outputs | live venv probes (17-02 Task 1, re-proven by this verifier) | Yes — exit 0 / 45 collected reproduced 2026-10-15 | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Dispatch can never upload — final tree | `gh run view 38678155010` (headSha 15bf6460, ancestor of HEAD; publish.yml diff to HEAD empty) | run success; smoke **success** (all steps incl. Import probe); publish **skipped**, 0 steps | ✓ PASS |
| Dispatch can never upload — transform commit | `gh run view 38554913144` (headSha = 2c649fa) | smoke success; publish skipped | ✓ PASS |
| Import probe output | `gh run view 38678155010 --log` (carriage-return cleaned) | `DNALLM 1.2.1 installed from dist` | ✓ PASS |
| GitHub environment `pypi` exists | `gh api repos/zhangtaolab/DNALLM/environments` | pypi, id 24275955008, created 2026-10-14T11:48:46Z | ✓ PASS |
| WINDOWS #20 coverage-CLI form 1 works on current venv | `COVERAGE_FILE=/tmp/... .venv/bin/python -m pytest tests/inference/test_mutagenesis.py --cov --co -q` | exit 0, 45 tests collected in 10.4s (no conftest ImportError) | ✓ PASS |
| Token-deletion sequencing honored | `gh api .../actions/secrets` | PYPI_API_TOKEN still listed (deletion correctly gated on first green OIDC release) | ✓ PASS |

### Probe Execution

No phase-declared probes (`scripts/*/tests/probe-*.sh`) and none conventional for this phase — the phase's live-proof mechanism is the dispatched GitHub run, executed above as behavioral spot-checks.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PUBS-01 | 17-01 | Unguarded dispatch→upload path event-gated BEFORE any smoke trigger; dry-run can never upload | ✓ SATISFIED | Job-level release gate; atomic single-commit landing (pickaxe); live dispatch proofs ×2 |
| PUBS-02 | 17-01 | Never-uploading smoke chain on dispatch: build → twine check → install-from-dist → import probe, reads dist/ never PyPI | ✓ SATISFIED | Live run 38678155010 green through Import probe; twine `>=7,<8`, build dep dropped, `--no-sources` |
| PUBS-03 | 17-01 | Smoke also on periodic cron, schedule-aware, still never uploading | ✓ SATISFIED | Weekly `0 8 * * 0` + D-19 gate, atomic with the guard; scheduled-leg first firing carried post-merge (AS-5, disclosed) |
| PUBS-04 | 17-01 + 17-03 | OIDC Trusted Publishing: workflow-side + registration + token-deletion sequencing | ◐ PARTIAL (as disclosed) | Workflow-side SATISFIED (truth 5). Registration clause PENDING — owner human-gate, WINDOWS #25, 17-03 halted; token correctly NOT deleted. REQUIREMENTS.md honestly tracks Pending |
| PUBS-05 | 17-02 | WINDOWS #20 remediated via numpy pin or documented coverage-CLI workaround (not a coverage bump) | ✓ SATISFIED | Row 20 fixed via the sanctioned coverage-CLI arm; form 1 re-proven live by the verifier; no coverage-spec change in phase commits |
| PKGH-08 | 17-02 | mypy capped so CI can never resolve an unvalidated stricter version | ✓ SATISFIED | `>=1.15.0,<2.5` at pyproject:86; one-line commit 8e2e1b9 |

Orphaned requirements: none — all six phase-17 IDs are claimed by plans (17-01: PUBS-01..04; 17-02: PUBS-05, PKGH-08; 17-03: PUBS-04).

### Decision Coverage

`check.decision-coverage-verify` ran: `skipped: true, blocking: false — "No trackable decisions in CONTEXT.md."` The 17-CONTEXT `<decisions>` block records four owner-accepted decisions in prose form (not the trackable tag format); each was nonetheless checked above — upload-guard event gating (truth 1), weekly-cron + dispatch smoke schedule (truth 4), #20 local-pin-with-fallback (truth 6, fallback arm exercised), OIDC workflow-side-first sequencing with owner gates (truths 5/8). Non-blocking by gate design.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found in phase-modified files (publish.yml, pyproject.toml, WINDOWS.md: zero TBD/FIXME/XXX/TODO/placeholder matches; no empty run steps — all 7 `run:` blocks carry scripts) | — | — |

### Test Quality Audit

No requirement-linked test files in this phase (CI/tooling + ledger phase; the phase's evidence is live-run + live-prove, cross-checked above). The phase census claim (zero test-surface movement) is proven by the atomic commit stat: 2c649fa touched exactly one file, `.github/workflows/publish.yml` (76+/9-) — tests/ and ci.yml untouched by the transform.

### Human Verification Required

N/A — infrastructure/CI-CD phase with no user-facing elements. All acceptance criteria were verified programmatically (live gh run/api evidence, git pickaxe/numstat, re-run probes). The single outstanding human action — the PyPI trusted-publisher registration — is tracked owner WORK (WINDOWS #25, blocking /gsd-ship via windows_enforce while open), not a phase-verification item; it is disclosed under Deferred Items above.

### Gaps Summary

No gaps. The phase's core goal — the publish chain exercisable end-to-end on demand and on schedule with zero upload possibility — is structurally guaranteed (job-level release gate; no upload step or token reference exists anywhere in the workflow) and live-proven on both the transform commit (run 38554913144) and the final current tree (run 38678155010, publish.yml byte-identical to HEAD 6b397d1): smoke green through the import probe, publish job skipped. OIDC workflow-side readiness is complete and live-verified (environment pypi exists; pypa action wired; no token fallback). WINDOWS #20 is closed via the PUBS-05-sanctioned coverage-CLI arm (re-proven live by this verifier: exit 0, 45 collected). The mypy cap is in place as a verified one-line edit.

Two disclosed standing items remain, both correctly tracked and neither hidden: (1) the owner's PyPI trusted-publisher registration (WINDOWS #25 open; 17-03 honestly halted with the five exact strings and resume path recorded; PYPI_API_TOKEN correctly retained until the first green OIDC release), and (2) the scheduled smoke leg's first post-merge firing. REQUIREMENTS.md bookkeeping is honest (PUBS-04 Pending). Phase 18's later extras-surgery touch of publish.yml (399f9f1) preserved every phase-17 contract element — re-verified line-by-line on the current tree, which is what invalidates the stale prior verification without introducing any regression.

---

_Verified: 2026-10-15T15:42:40Z_
_Verifier: Claude (gsd-verifier)_
