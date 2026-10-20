---
phase: 18-packaging-install-honesty-atomic-extras-surgery
plan: 01
subsystem: packaging
tags: [pyproject, extras, pip, uv, pep621, environment-markers, github-actions, deprecation-alias, cuda121-retirement, rocm-restore-path, triton-windows]

requires:
  - phase: 17-publish-chain-robustness
    provides: never-uploading smoke chain (dispatchable against any ref; publish job release-gated)
provides:
  - "`full` extra (former `base` member set verbatim) + the one-cycle deprecated alias `base = [\"dnallm[full]\"]` (remove in v1.4) in pyproject.toml"
  - triton-windows Windows-marker declaration in the fla extra (WPAR-07 declaration half; the Windows execution proof rides 18-03's KDA checkpoint)
  - cuda121 retired everywhere (extra, uv index/sources/conflicts, CI matrix leg, docs rows) with the frozen cu121 index recorded
  - rocm removed with the rocm7.1 restore path documented against 2026-10-14 facts
  - repo-scan residue gates in tests/test_extras_guard.py holding the surface to verbatim sanctioned lists (extendable by 18-02's same-change list update)
affects: [18-02 (honesty-matrix page must update the Pattern-1 sanctioned list same-change), 18-03 (Windows KDA execution retires the triton-windows SUS flag), 19-docs-sweep (describes this final extras state)]

actuals:
  tokens: 18500      # chars/4 over files actually changed: surgery diff 62702 chars + SUMMARY ~12k chars
  tasks: 3
  commits: 1         # measured: git rev-list --count e850aac..HEAD (docs commit excluded by construction)

plan_head_before: e850aac74fe6b931dd5a829905e9f6ffbda6bd9f
plan_head_after: 399f9f15ab1790860ac175a87a6eab4905a6849d

tech-stack:
  added:
    - "triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows' (declared in fla; triton-lang org fork, exact band, no floating)"
  patterns:
    - "one-cycle deprecated extra alias: self-referential base = [\"dnallm[full]\"] converts pip/uv's post-removal warn-and-exit-0 under-install into a correct install while docs carry the deprecation loudly"
    - "sanctioned-residue repo-scan gates in pytest (phase-13 verbatim discipline): exact (path, line-content) pairs; empty/short = deletion creep = fail; extra line = fail"
    - "pickaxe 3-way atomicity proof: alias body + renamed CI literal + guard constant all resolve to the identical introducing hash"

key-files:
  created:
    - .planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-01-SUMMARY.md
  modified:
    - pyproject.toml
    - .github/workflows/ci.yml
    - .github/workflows/publish.yml
    - .github/workflows/feasibility.yml
    - .github/workflows/README.md
    - tests/test_extras_guard.py
    - tests/models/test_plant_helixseek_fla_kernels.py
    - README.md
    - docs/getting_started/installation.md
    - docs/getting_started/quick_start.md
    - docs/index.md
    - docs/user_guide/getting_started.md
    - docs/user_guide/tutorial/end_to_end_workflow.md
    - docs/faq/models_troubleshooting.md
    - example/notebooks/overview.md (+ its byte-identical docs mirror)
    - docs/example/**/*.md (27 wrapper files)

key-decisions:
  - "Pattern-2 sanctioned residue finalized at THREE lines, not the plan's two: the pyproject fla comment (kept verbatim per Task 1(3)/Pitfall 4) is a third member of the same fla-upstream [cuda]/[rocm] vocabulary class the plan sanctions — the plan's enumeration undercounted its own sanctioned class (Pitfall-1-type line drift)"
  - "Fresh-venv resolves run on conda python3.12 (headers present) with CPATH/LIBRARY_PATH exposing the miniconda-base curl headers to isolated sdist builds — no user env mutated; system /usr/bin/python3.12 lacks Python.h and pyfastx/pybigwig/scikit-allel have no linux-aarch64 wheels"
  - "Pip-side alias full-closure resolve DEFERRED as an unrun-verify (box network pathology against pip's urllib3 stack, 5 attempts); mechanism already live-probed (pip 26.2.1, 18-RESEARCH), the built wheel's METADATA alias line asserted, uv-side parity green — recorded in WINDOWS ledger with the exact re-run command"
  - "Verify-literal corrections applied with intent preserved: METADATA grep matches setuptools-normalized form (triton-windows<3.8,>=3.7.1.post27; platform_system == \"Windows\" and extra == \"fla\"); pickaxe CI literal is '.[full,fla]' (the plan's quoted form cannot occur in the file)"

patterns-established:
  - "Deprecated-extra-alias cycle: over-install is the safe direction for one release; docs (not installers) are the loud channel during the cycle (probe-verified: no installer warns on an existing extra)"
  - "Retirement note vocabulary rule: write the index-lane token (cu121, rocm7.1) — never the extra-name/bracket form — so the residue gates stay clean"

requirements-completed: [PKGH-01, PKGH-02, PKGH-03, WPAR-07]

coverage:
  - id: D1
    description: "Atomic base→full rename with one-cycle alias across pyproject, 10 workflow install lines + comments, guard tests, and 49 docs occurrences — ONE commit"
    requirement: PKGH-01
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py::TestFullExtraAndBaseAlias (4 guards) + ::TestExtrasSurgeryResidueGates (2 repo-scan gates) — 17 passed"
        status: pass
      - kind: other
        ref: "pickaxe 3-way: git log -S over pyproject alias body, ci.yml '.[full,fla]', EXPECTED_FULL_MEMBERS — all = 399f9f1 = HEAD -> ATOMIC-COMMIT-OK"
        status: pass
      - kind: other
        ref: "fresh-venv wheel resolves: $W[full] and $W[base] (uv) exit 0, import dnallm + isort (isort reaches through the alias — no under-install)"
        status: pass
    human_judgment: false
  - id: D2
    description: "cuda121 retired (extra + uv index/sources/conflicts + CI matrix leg + docs rows) with the frozen cu121 index recorded; rocm removed with the rocm7.1 restore path on 2026-10-14 facts"
    requirement: PKGH-02
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py::test_cuda121_and_rocm_extras_absent (extras keys + [[tool.uv.index]] + sources via tomllib)"
        status: pass
      - kind: other
        ref: "residue gate Pattern 2 equals the 3-line sanctioned fla-upstream list; conflicts shrunk to the 10 pairwise rows; test-cuda matrix single 12.4 leg"
        status: pass
    human_judgment: false
  - id: D3
    description: "triton-windows declared behind platform_system == 'Windows' in fla (WPAR-07 declaration half); Linux marker-false proven; Windows execution proof rides 18-03"
    requirement: WPAR-07
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py::test_triton_windows_marker_declared_in_fla (exact-string)"
        status: pass
      - kind: other
        ref: "built-wheel METADATA carries exactly one triton-windows line, normalized form 'triton-windows<3.8,>=3.7.1.post27; platform_system == \"Windows\" and extra == \"fla\"'"
        status: pass
      - kind: other
        ref: "fresh /tmp/xe-fla venv (Linux): resolve green, import dnallm OK, triton-windows ABSENT from uv pip list (marker false)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Phase-17 smoke dispatch green against the post-surgery tree with publish skipped (SC5)"
    verification:
      - kind: other
        ref: "run 38578071738 (dispatch 2026-10-14T14:09:54Z on 399f9f1): jobs API publish=skipped AND smoke=success -> SMOKE-DISPATCH-OK; probe line 'DNALLM 1.2.1 installed from dist'"
        status: pass
      - kind: other
        ref: "fast lane 2556 passed / 1 skipped / 75 deselected, exit 0; census triple 216/217 (1 deselected) with exact stage-0.5 flags — no re-pin needed"
        status: pass
    human_judgment: false
  - id: D5
    description: "Pip-side alias resolve of the full real closure (pip-side edge of the alias proof)"
    verification: []
    human_judgment: true
    rationale: "Could not execute on this box today: pip's urllib3 stack repeatedly stalls against both direct PyPI and mirrors (5 attempts) while uv/gh/curl are unaffected — a box-network artifact, not a repo defect. Adjacent evidence: 18-RESEARCH live probe (pip 26.2.1 resolves wheel[base]->pkg[full] full set, exit 0), the built wheel's METADATA alias line 'dnallm[full]; extra == \"base\"', the pure-self-reference guard, and uv-side parity. Re-run command recorded in WINDOWS ledger; verifier or owner should execute it on a healthy-network host."

duration: 65 min
completed: 2026-10-14
status: complete
---

# Phase 18 Plan 01: Atomic Extras Surgery Summary

**ONE commit (399f9f1) renamed base→full with a one-cycle self-referential alias, retired cuda121 (frozen cu121 index recorded), removed rocm with the rocm7.1 restore path, and declared triton-windows behind the Windows marker in fla — proven by fresh-venv wheel resolves, a green never-uploading smoke dispatch (run 38578071738, publish skipped), and 3-way pickaxe atomicity**

## Performance

- **Duration:** 65 min (3871 s) — most of it the multi-GB fresh-venv resolve battery and the deferred pip edge
- **Started:** 2026-10-14T13:08:08Z
- **Completed:** 2026-10-14T14:13:29Z
- **Tasks:** 3/3
- **Files modified:** 41 (+ this SUMMARY)

## Accomplishments

- The extras surface is honest in ONE atomic commit: `full` carries the former `base` members verbatim (`dnallm[dev,test,notebook,mcp]`, `isort>=6.0.1`, `types-transformers>=0.1.0`); the deprecated alias `base = ["dnallm[full]"]` installs the full set under uv (proven live from the built wheel: isort reachable through `[base]`) and converts the post-v1.4 pip/uv warn-and-exit-0 into a correct install for one cycle
- cuda121 is gone from extras, `[[tool.uv.index]]`, `[tool.uv.sources]`, the conflicts table (10 pairwise rows remain among cpu/cuda124/126/128/130), the CI test-cuda matrix (single 12.4 leg), and every docs table; the retirement note records the frozen index fact: cu121 tops at torch 2.5.1 (re-verified 2026-10-14) — the extra's `>=2.2.0,<2.12` spec could never resolve past 2.5.1 on its own index
- rocm removed with the restore path written against today's facts (rocm7.1 ships torch 2.10/2.11, Linux x86_64, resolution-tested-only; rocm6.4 tops at 2.9.1; ROCm 7.2+ needs torch ≥2.12 > the cap; never execution-tested on AMD — dnallm makes no AMD claim)
- WPAR-07 declaration half: `triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'` rides the fla extra exactly once in built-wheel METADATA (normalized form asserted); the Linux fla venv proves the marker evaluates false (triton-windows ABSENT)
- Guard surface extended and census-stable: 6 new tests in tests/test_extras_guard.py (member preservation, pure-alias, exact marker, retirement regression, 2 repo-scan residue gates with verbatim sanctioned lists); fast lane 2556 passed; D-03 census triple STILL 216/217 with the exact stage-0.5 flags — zero ci.yml census re-pin needed
- Phase-17 smoke dispatched against the surgery commit: publish job skipped by the release gate, smoke chain green through the import probe (`DNALLM 1.2.1 installed from dist`)

## Task Commits

1. **Task 1 (tracer): pyproject surgery + guards + thinnest end-to-end proof** — no commit by design (tracer verified end-to-end, then expanded)
2. **Task 2: consumer sweep (workflows/docs/README + residue gates)** — no commit by design
3. **Task 3: THE atomic commit** - `399f9f1` (chore) — the ONE commit carrying the entire Task 1-2 surface; pickaxe 3-way proof ATOMIC-COMMIT-OK; publish.yml diff = the single install-line rename (D=0, trigger surface byte-untouched)

**Plan metadata:** (docs commit — follows this SUMMARY)

## Files Created/Modified

- `pyproject.toml` — full extra + alias + all-chain swap; cuda121/rocm blocks, 2 uv indexes, 1 source row, 9 conflict rows deleted; fla marker; :116 comment reworded
- `tests/test_extras_guard.py` — +227 lines: EXPECTED_FULL_MEMBERS / EXPECTED_BASE_ALIAS / EXPECTED_TRITON_WINDOWS_SPEC + TestFullExtraAndBaseAlias (4) + TestExtrasSurgeryResidueGates (2)
- `tests/models/test_plant_helixseek_fla_kernels.py` — parser literals flipped base→full (:45, :56, :61-62)
- `.github/workflows/ci.yml` — 8 install sites, matrix collapse 12.1+12.4→12.4, if/elif→unconditional `.[full,cuda124]`, 3 comment sites; census pins untouched
- `.github/workflows/publish.yml` — install line only (`-e '.[full]'`; --system and quoting preserved)
- `.github/workflows/feasibility.yml` + `.github/workflows/README.md` — install line + comment/prose flips
- `README.md`, `docs/getting_started/installation.md`, `quick_start.md`, `docs/index.md`, user_guide ×2, faq — 49 install occurrences renamed; cuda121/rocm rows retired; deprecation notes (v1.4) + cu121 freeze + rocm7.1 restore path
- `docs/example/**/*.md` + `example/notebooks/overview.md` — 27 wrapper flips (mirror pair byte-identical after edit); plant_helixseek_cre.md block-5 re-synced

## Sanctioned Residue Records (verbatim, as enforced by the gates)

**Pattern 1 — bracket install forms of base** (`.\[base` / `dnallm\[base`; exactly 2 lines):

| File | Verbatim line |
|---|---|
| README.md:179 | `> **Deprecation note (v1.3):** the \`base\` extra was renamed to \`full\` (same members: \`dev\` + \`test\` + \`notebook\` + \`mcp\` + isort + types-transformers). The old \`.[base]\` spelling still resolves the full member set through a one-cycle alias and will be removed in v1.4 — after that removal pip/uv only print a warning that the extra does not exist and exit 0, leaving a core-only under-install (the silent hazard the alias exists to prevent). Use \`.[full]\`.` |
| docs/getting_started/installation.md:214 | `> **Deprecation note (v1.3):** the \`base\` group was renamed to \`full\` (same members: \`dev\` + \`test\` + \`notebook\` + \`mcp\` + isort + types-transformers). The \`.[base]\` alias still installs the full set and will be removed in v1.4 — after removal pip/uv only print a warning that the extra does not exist and exit 0, leaving a core-only under-install. Use \`.[full]\` in new environments.` |

**Pattern 2 — cuda121 | torch-rocm | [rocm]** (exactly 3 lines, all fla-UPSTREAM vocabulary, Pitfall 4):

| File | Verbatim line (leading) |
|---|---|
| pyproject.toml:185 | `# ([cuda]/[rocm]), they pin their own torch and would downgrade this environment.` |
| README.md:255 | `Two install notes: install the package **bare** (\`flash-linear-attention\`, not its \`[cuda]\`/\`[rocm]\` extras) — …` |
| docs/faq/models_troubleshooting.md:13 | `With the kernels installed the same checkpoints separate cleanly (p(CRE) 0.77 in-DHS vs 0.22 non-DHS on the probe set). Install the package **bare** — since v0.5 its \`[cuda]\`/\`[rocm]\` extras …` |

(The gates assert (path, full-line-content) pairs byte-exactly; the full verbatim strings live in `tests/test_extras_guard.py` SANCTIONED_* frozensets. 18-02's honesty-matrix page extends the Pattern-1 list same-change.)

## Resolve Battery (fresh 3.12 venvs, wheel from `uv build --no-sources`)

| Extra | Installer | Result |
|---|---|---|
| `[full]` | uv | exit 0; `import dnallm` OK |
| `[base]` (alias) | uv | exit 0; `import dnallm, isort` OK — full-only member reaches the alias (no under-install) |
| `[base]` (alias) | pip | **DEFERRED** — box-network pathology against pip's HTTP stack (5 attempts); see Issues; mechanism probe-proven in 18-RESEARCH + METADATA line asserted |
| `[all]` | uv | exit 0; `import dnallm` OK |
| `[fla]` | uv | exit 0; `import dnallm` OK; **triton-windows ABSENT on Linux** (marker false) |
| `[cuda124]` (control) | uv | exit 0; `import dnallm` OK |

Wheel METADATA: exactly one triton-windows Requires-Dist, normalized `triton-windows<3.8,>=3.7.1.post27; platform_system == "Windows" and extra == "fla"` (MARKER-EXACTLY-ONCE-NORMALIZED-OK).

## Smoke Dispatch Evidence (Task 3)

- Push: `13d5a30..399f9f1 v1.3 -> v1.3`; HEAD == origin/v1.3 == 399f9f1 (REF-SYNC-OK)
- Pre-flight: remote publish.yml@v1.3 carries the release gate + weekly cron + the `.[full]` line (REMOTE-GUARD-PRESENT-OK); no in-flight runs
- Dispatch: `gh workflow run publish.yml --ref v1.3` at **2026-10-14T14:09:54Z** → run **38578071738** on headSha 399f9f1
- Jobs API: `publish` = **skipped** (release gate held — no upload-capable step instantiated); `smoke` = **success** (SMOKE-DISPATCH-OK)
- Probe line (verbatim, 14:11:02Z): `DNALLM 1.2.1 installed from dist`
- Honest scope: the smoke's install step is extras-free (and the publish job carrying the `.[full]` line is the skipped one) — the smoke proves the artifact core; the fresh-venv resolves above are the extras proof (18-CONTEXT locked depth)

## Pickaxe Atomicity Proofs

`git log -S '"dnallm[full]"' -- pyproject.toml` = `git log -S '.[full,fla]' -- .github/workflows/ci.yml` = `git log -S 'EXPECTED_FULL_MEMBERS' -- tests/test_extras_guard.py` = **399f9f15ab1790860ac175a87a6eab4905a6849d** = HEAD → ATOMIC-COMMIT-OK. publish.yml staged-diff trigger/gate/OIDC check: D=0 captured then explicitly compared (R4 hygiene) → PUBLISH-TRIGGERS-UNTOUCHED.

## cuda121 Retirement Fact (SC2 record)

The `https://download.pytorch.org/whl/cu121` index is FROZEN at torch 2.5.1 (re-verified 2026-10-14, 18-RESEARCH live fetch): the retired extra's `torch>=2.2.0,<2.12` spec could never resolve past 2.5.1 on its own index — a dead lane under the core `torch>=2.4.0,<2.12` floor. Recorded in installation.md's hardware-groups section (cu121 token only, never a bracket form).

## Decisions Made

- Pattern-2 sanctioned list finalized at 3 lines (see key-decisions frontmatter): the plan's 2-line enumeration missed the pyproject fla comment its own Task 1(3) mandates keeping — same sanctioned class, gate extended, deviation documented
- Fresh-venv python = conda 3.12 (mineru env) + CPATH/LIBRARY_PATH for isolated sdist builds; zero mutation of any user environment
- Deprecation notes phrase post-removal behavior as the documented native warning (Pitfall 2 honored — no in-cycle warning assertion anywhere)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Headerless system python3.12 broke fresh-venv sdist builds**
- **Found during:** Task 1 (tracer proof)
- **Issue:** `uv venv --python 3.12` resolved to /usr/bin/python3.12 (no Python.h); pyfastx/scikit-allel/pybigwig publish no linux-aarch64 wheels and must build from sdist
- **Fix:** point uv at the conda `mineru` env's python3.12 (ships headers); no user env modified
- **Verification:** full/alias resolves green twice (tracer + re-run)

**2. [Rule 3 - Blocking] pybigwig sdist build needs curl/curl.h absent from the conda env**
- **Found during:** Task 1 (tracer proof)
- **Fix:** export CPATH=/home/forrest/miniconda3/include + LIBRARY_PATH=…/lib (the conda BASE prefix has libcurl) on the install commands only — ephemeral, no env mutation
- **Verification:** pyfastx/pybigwig/scikit-allel compiled; resolves green

**3. [Rule 1 - Bug] Tracer METADATA grep literal unmatchable**
- **Found during:** Task 1 (verify)
- **Issue:** the plan greps for `triton-windows>=3.7.1.post27,<3.8; platform_system == "Windows"` but setuptools NORMALIZES published metadata (`<3.8,>=3.7.1.post27; … and extra == "fla"`) — the literal can never match
- **Fix:** assert the normalized form with the same intent (exactly one triton-windows METADATA line, carrying the Windows marker and the fla extra)
- **Verification:** MARKER-EXACTLY-ONCE-NORMALIZED-OK

**4. [Rule 1 - Bug] Pattern-2 sanctioned list undercounted its own class**
- **Found during:** Task 2 (residue gate design)
- **Issue:** the plan sanctions "exactly the two fla-upstream lines" but Task 1(3)/Pitfall 4 mandate keeping the pyproject fla comment — a third `[rocm]`-matching line of the SAME sanctioned class; a 2-line gate would fail on the plan's own instructions
- **Fix:** sanctioned list finalized at 3 verbatim lines (README:255, faq:13, pyproject:185); recorded here and in the gate
- **Verification:** residue gate green; echo counts S=3 (within the plan's 1-4 review band)

**5. [Rule 3 - Blocking] Pre-existing notebook-md sync failure blocked the Task 2 verify gate**
- **Found during:** Task 2 (baseline, BEFORE any docs edit)
- **Issue:** docs/example/notebooks/plant_helixseek_cre.md block 5 drifted from the notebook's phase-16 cell (d05a3b9 added the sys.platform dispatch; wrapper never updated) — check_notebook_md_sync red pre-surgery
- **Fix:** wrapper block 5 re-synced verbatim to notebook cell 11
- **Verification:** NBMD-SYNC-GREEN (24 pairs, all in sync)

**6. [Rule 1 - Bug] installation.md:517 cuda121 install line missed by the research enumeration**
- **Found during:** Task 2 (execution-time re-grep, Pitfall 1 discipline)
- **Issue:** a troubleshooting-section `uv pip install -e '.[cuda121]'` line absent from every planning-artifact list
- **Fix:** re-pointed to `'.[cuda124]'` (surviving extra, "Choose based on actual situation" context intact)
- **Verification:** Pattern-2 residue gate green (no cuda121 anywhere in scope)

**7. [Rule 1 - Bug] Task 3 pickaxe literal unmatchable**
- **Found during:** Task 3 (verify)
- **Issue:** `git log -S '"[full,fla]"'` cannot match `".[full,fla]"` (the dot sits between quote and bracket)
- **Fix:** corrected literal `.[full,fla]`; intent unchanged (the renamed ci.yml install line's introducing hash)
- **Verification:** ATOMIC-COMMIT-OK (A=B=C=H)

---

**Total deviations:** 7 auto-fixed (3 blocking-environment, 3 plan-literal/verification bugs, 1 pre-existing drift) + 1 deferral (below)
**Impact on plan:** All fixes preserve plan intent verbatim; two verify literals and one sanctioned-list count corrected against executable reality. No scope creep — commit 399f9f1 carries exactly the planned surface plus the two documented extras (block-5 re-sync, :517 re-point).

## Issues Encountered

**Pip-side alias full-closure resolve deferred (unrun-verify, WINDOWS ledger entry written).** pip's urllib3 stack repeatedly stalled against this box's network across 5 attempts (direct PyPI ×2 — dead connections, zero transfer; Tsinghua mirror — 11.6 GB bulk download then resolver stall; pinned-to-uv-set with dual index — metadata crawl at ~1.4 MB/min), while uv (rust reqwest), gh, and curl all work normally — a box-network artifact, not a repo defect. The 5th attempt also revealed pip resolver backtracking through 2023-era candidates on the unconstrained closure. Adjacent evidence carried instead: the 18-RESEARCH live probe (pip 26.2.1 resolves `wheel[base]` → `pkg[full]` full set, exit 0, from a local wheel — the exact alias mechanism), the built wheel's METADATA line `dnallm[full]; extra == "base"`, the pure-self-reference guard test, and green uv-side parity on the identical member set. Re-run on a healthy-network host: `uv venv --seed /tmp/xp --python 3.12 && /tmp/xp/bin/python -m pip install "$(ls dist/*.whl)[base]" && /tmp/xp/bin/python -c "import dnallm, isort"`.

Also noted (out of scope, no action): a self-hosted nightly runner executed the pre-surgery origin/v1.3 tree concurrently on this box throughout — no interference; `.planning/state.json` was dirty before execution began and stays out of the surgery commit.

## Authentication Gates

None — gh CLI authenticated with workflow scope throughout (verified in Task 1 preconditions).

## User Setup Required

None for this plan. (WPAR-07's Windows execution half is 18-03's owner checkpoint.)

## Known Stubs

None — no stubs or placeholders. The one deferred verification (pip-side edge) is recorded above and in the WINDOWS ledger as unrun-verify, with its evidence chain and exact re-run command.

## Next Phase Readiness

- 18-02 (honesty-matrix page) can proceed; it MUST extend the Pattern-1 sanctioned list same-change when its page adds `.[base]` rows — the gate's SANCTIONED_BASE_BRACKET_RESIDUE frozenset is the single edit point
- 18-03's Windows KDA checkpoint retires the triton-windows SUS-with-null-signals flag; the declaration it executes is live in METADATA exactly once
- Carried: the pip-side alias re-run (healthy-network host); the publish job's `.[full]` install line executes for the first time on the next RELEASE event (dispatch can't reach it by design)

## Self-Check: PASSED

- Files exist: 18-01-SUMMARY.md (this file), tests/test_extras_guard.py, all 41 committed surgery files — FOUND
- Task commit `399f9f1` is an ancestor of HEAD — FOUND
- `gsd_run check evaluation-scope --plan 18-01 --commits-only` resolves to exactly [399f9f1] on this branch — FOUND
- Guard gates re-run post-commit: 11 passed
- Pickaxe re-run post-commit: ATOMIC-COMMIT-OK (A=B=C=H=399f9f1…)
- Live run re-check: 38578071738 — publish=skipped / smoke=success / probe line present
- Census triple re-verified: 216/217 (1 deselected), exact stage-0.5 flags
