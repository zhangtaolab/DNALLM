---
phase: 18-packaging-install-honesty-atomic-extras-surgery
plan: 02
subsystem: packaging
tags: [docs, install-honesty, uv, pip, gpu-support-matrix, extras, lane-split, mkdocs, residue-gate, deprecation-alias]

requires:
  - phase: 18-packaging-install-honesty-atomic-extras-surgery (plan 01)
    provides: the post-surgery extras state (full + one-cycle base alias, cuda121 retired, rocm removed, triton-windows declared) and the Pattern-1/2 sanctioned residue gates this plan extends
provides:
  - installation.md uv/pip lane split with the [tool.uv.sources] asymmetry stated before any extras table (PKGH-04)
  - docs/getting_started/gpu_support.md — the GPU support matrix from the 0542a6b pre-verified record with the cu121-retirement adjustment (PKGH-06)
  - docs/getting_started/install_honesty.md — the five-row cross-installer honesty matrix from the 2026-10-14 live probes (PKGH-07)
  - mkdocs nav + README wiring for both pages; Pattern-1 sanctioned list extended same-change with the honesty page's two dnallm[base] rows
affects: [18-03 (Windows KDA checkpoint — links to the honesty page's triton-windows row), 19-docs-sweep (DOCR strict build; the lane labels are the sweep's vocabulary)]

actuals:
  tokens: 8800      # chars/4 over files actually changed: docs/guards diff 22842 chars + this SUMMARY ~12k chars (plan estimate 60000 — the estimate carried the full publication-writing effort at confidence:low)
  tasks: 3
  commits: 1        # measured: git rev-list --count 7ebd557..HEAD

plan_head_before: 7ebd55732c85df2bf1287aa37fa79878de66013f
plan_head_after: 29aaa683c50371652f56ad6ddf5beb674519ef2a

tech-stack:
  added: []          # docs-only plan; the isolated mkdocs verification venv is /tmp tooling, not a repo dependency
  patterns:
    - "lane-label vocabulary: every dnallm/torch install block in installation.md carries '# [uv lane]' / '# [pip lane]'; third-party model-dep blocks are covered by a section-level prose note"
    - "provenance-carried matrix rows: no GPU/honesty row without its trace (0542a6b record / live-probe date + installer versions) — publishing unverified rows stays a recorded anti-feature"
    - "residue-gate list extension is same-change: new sanctioned bracket-form lines land in the SAME commit as the page that introduces them"

key-files:
  created:
    - docs/getting_started/gpu_support.md
    - docs/getting_started/install_honesty.md
  modified:
    - docs/getting_started/installation.md
    - mkdocs.yml
    - README.md
    - tests/test_extras_guard.py

key-decisions:
  - "Commit granularity follows the plan's own Task 3(4) 'Commit once' (the 18-01 atomic pattern): Tasks 1-2 edit without committing, Task 3 validates everything and lands the single docs(18-02) commit 29aaa68 — the plan's verify asserts HEAD is that commit"
  - "GPU matrix published by pasting the 0542a6b record recovered verbatim from git (richer than the research excerpt: it carries the sm_61-after-cu118 Pascal fact and the CUDA-13-floor-is-sm_75 note); the only re-derivation is the planned cu121-retirement narrowing of the 20xx row to cuda124–cuda128"
  - "Honesty page keeps bracket-form base usage to exactly the two matrix rows so the sanctioned-list extension is exactly two lines; prose elsewhere says 'the deprecated base spelling' without brackets"
  - "mkdocs verified in an isolated /tmp venv reproducing the CI docs lane (docs-extra deps + marimo, which the CI lane gets transitively via .[full,docs]); jupytext shells out to the marimo CLI, so the venv's bin must be on PATH — recorded here because it will recur for Phase 19's DOCR-03 strict work"

patterns-established:
  - "During-cycle deprecation honesty phrasing: warn-and-exit-0 is ALWAYS attributed to the post-removal state ('after removal', 'once the alias is dropped'); no sentence claims an installer warning while the alias exists (Pitfall 2)"
  - "Index-lane token rule held in new prose: cu121 (never cuda121), rocm7.1 — keeps Pattern 2 green by construction"

requirements-completed: [PKGH-04, PKGH-06, PKGH-07]

coverage:
  - id: D1
    description: "installation.md uv/pip lane split: [tool.uv.sources] asymmetry callout before the extras table (uv-docs quote), lane labels on every dnallm/torch install block, both new pages linked, pip-lane torch-LAST guidance intact, table prose matching the post-surgery pyproject"
    requirement: PKGH-04
    verification:
      - kind: other
        ref: "Task 1 automated verify: lane-labels=29 asymmetry=11 page-links=6 -> LANE-SPLIT-PRESENT; content re-read confirmed real statements, not tokens"
        status: pass
      - kind: manual_procedural
        ref: "Review item (e): every pip-lane block enumerated (7 install sites); none promises a cuda index pin; uv-lane blocks annotate the pip fallback"
        status: pass
    human_judgment: false
  - id: D2
    description: "GPU support matrix page from the pre-verified boundaries: Pascal cpu-only with the Turing minimum-GPU line explicit, 20xx cuda124–cuda128 (cu121 retirement adjustment), 30/40xx every extra, 50xx cuda128+, cuda130 default = Ampere+; every row with provenance"
    requirement: PKGH-06
    verification:
      - kind: other
        ref: "Task 2 automated verify: gpu-pascal=3 gpu-20xx=4 -> matrix rows present; provenance section names 0542a6b verbatim recovery + torch 2.11.0+cu130 arch list"
        status: pass
      - kind: manual_procedural
        ref: "Review item (b): each of the 5 GPU rows traced to the 0542a6b record or the 2026-10-14 cu121-retirement adjustment; zero rows re-derived from vendor charts"
        status: pass
      - kind: other
        ref: "mkdocs build exit 0, warning set byte-identical to the pre-change baseline (15 pre-existing, 0 new)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Cross-installer honesty matrix page: five command x installer rows from the 2026-10-14 live probes (pip 26.2.1 / uv 0.13.1, clean venvs); during-cycle rows state NO warning, post-removal rows warn-and-exit-0; composes with the pip-lane torch-LAST rule; nav + README wired"
    requirement: PKGH-07
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py (11 passed) — Pattern-1 gate green over the same-change extended sanctioned list; Pattern-2 unchanged at 3 lines"
        status: pass
      - kind: other
        ref: "Task 2 automated verify (corrected literal): honesty-warn=1 wired-total=3 (2 nav + 1 README) -> PAGES-WIRED"
        status: pass
      - kind: manual_procedural
        ref: "Review items (a)/(c): every row carries its Provenance cell with probe date, installer versions in the provenance section; every 'warning' mention on the page is post-removal-attributed — no during-cycle warning claim anywhere"
        status: pass
    human_judgment: false
  - id: D4
    description: "No-regression battery: docs sync + notebook-md sync green, fast lane at the 18-01 baseline, census 216/217 stable, residue gate green"
    verification:
      - kind: other
        ref: "check_docs_sync.py exit 0 (docs/example in sync); check_notebook_md_sync.py exit 0 (24 pairs in sync)"
        status: pass
      - kind: other
        ref: "fast lane: 2556 passed / 1 skipped / 75 deselected, exit 0 (identical to the 18-01 baseline; test COUNT unchanged by construction)"
        status: pass
      - kind: other
        ref: "census: '216/217 tests collected (1 deselected)' with the exact stage flags -> CENSUS-STABLE"
        status: pass
    human_judgment: false

duration: 24 min
completed: 2026-10-14
status: complete
---

# Phase 18 Plan 02: Install-Honesty Publications Summary

**installation.md split into lane-labeled uv/pip commands with the [tool.uv.sources] asymmetry stated before any extras table, plus two new provenance-carried pages — the GPU support matrix (0542a6b record + cu121-retirement adjustment) and the five-row cross-installer honesty matrix (live probes: during-cycle no-warning, post-removal warn-and-exit-0) — nav/README-wired, residue gate extended same-change, all docs gates green at the stable census**

## Performance

- **Duration:** 24 min (1450 s)
- **Started:** 2026-10-14T14:19:08Z
- **Completed:** 2026-10-14T14:43:18Z
- **Tasks:** 3/3
- **Files modified:** 6 (4 modified + 2 new pages) + this SUMMARY

## Accomplishments

- **PKGH-04:** the asymmetry callout now opens the Dependency Groups section BEFORE any table — the uv-docs quote ("Sources are only respected by uv…", cited via the v1.3 stack research) with the two-lane consequence spelled out (uv lane: pinned torch-cuda124…130 indexes; pip lane: default PyPI torch — Linux cu130-bundled, Windows CPU-only silently). Every dnallm/torch install block through the page carries `# [uv lane]` / `# [pip lane]`; the third-party model-deps section carries a section-level lane note. The torch-LAST pip rule and the verify-torch-suffix guidance are intact and now cross-linked from the honesty page.
- **PKGH-06:** `docs/getting_started/gpu_support.md` published from the 0542a6b record recovered verbatim from git — Pascal (10xx) cpu-only with the sm_61-after-cu118 fact and the Turing minimum-GPU line explicit; 20xx served by cuda124–cuda128 (sm_75 IS in cu128 — torch 2.11 dropped only Volta sm_70, pytorch#172352 — but NOT cu130, arch list starts sm_80, no PTX); 30/40xx run every extra via sm_80 same-major binary compat; 50xx needs cuda128+; cuda130 default = Ampere+; forward note (cu128 deprecates from torch 2.12) and the driver-based chooser (cuda124–cuda130). Provenance section names the record, the commit, the local basis, and the v1.3 narrowing.
- **PKGH-07:** `docs/getting_started/install_honesty.md` — the five command × installer rows with a per-row Provenance column: `dnallm[full]` (identical both installers); `dnallm[base]` during the cycle (full set, exit 0, NO warning — probe-verified, the loud surface is docs); `dnallm[base]` after v1.4 (pip `WARNING: … does not provide the extra 'base'` / uv `warning: … no extra named 'base'`, core-only, exit 0 — pre-explained); `dnallm[cuda130]` (uv: pinned index; pip: default PyPI, Windows CPU-only); `dnallm[fla]` on Windows (both pull triton-windows via the PEP 508 marker; the community-fork status stated plainly — triton-lang org, attested builds since 3.7.1.post27).
- **Wiring:** mkdocs nav entries after Installation; README pointer line next to the 18-01 deprecation note naming both pages (Pascal cpu-only, 20xx → cuda124–cuda128, 50xx cuda128+).
- **Residue gate extended same-change:** the honesty page's two `dnallm[base]` matrix rows are the ONLY new bracket-form residue in the repo — added verbatim to `SANCTIONED_BASE_BRACKET_RESIDUE` in the same commit; the gate's failure message updated to match. Test count unchanged (11 collected), census triple untouched.

## Task Commits

1. **Task 1: installation.md uv/pip lane split** — no commit by design (the plan's Task 3(4) single-commit pattern)
2. **Task 2: both pages + wiring + sanctioned-list extension** — no commit by design
3. **Task 3: validation + THE commit** - `29aaa68` (docs) — carries the entire six-file surface; verify battery green post-commit

**Plan metadata:** (docs commit — follows this SUMMARY)

## Files Created/Modified

- `docs/getting_started/installation.md` — asymmetry callout subsection before the Core Dependency Groups table; lane labels on all dnallm/torch install blocks (Methods 1-3, GPU Support, plain-pip, Windows cu130, Scenarios 1-8, troubleshooting, native mamba); links to both new pages from the callout, the GPU intro, the plain-pip section, and the Method-3 note; extras-table prose verified against the post-surgery pyproject
- `docs/getting_started/gpu_support.md` — NEW: the GPU support matrix + forward notes + verification provenance + related pages
- `docs/getting_started/install_honesty.md` — NEW: the five-row matrix + "reading the rows honestly" + probe provenance
- `mkdocs.yml` — two Getting Started nav entries (GPU Support Matrix, Install Honesty Matrix)
- `README.md` — one pointer paragraph after the 18-01 deprecation note
- `tests/test_extras_guard.py` — Pattern-1 sanctioned frozenset +2 entries (verbatim honesty rows), comment + failure message updated; no test functions added or removed

## Final Sanctioned Residue Lists (verbatim, as enforced by the gates)

**Pattern 1 — bracket install forms of base** (`.\[base` / `dnallm\[base`; exactly 4 lines):

| File | Verbatim line (leading) |
|---|---|
| README.md | `> **Deprecation note (v1.3):** the \`base\` extra was renamed to \`full\` (same members: …) Use \`.[full]\`.` |
| docs/getting_started/installation.md | `> **Deprecation note (v1.3):** the \`base\` group was renamed to \`full\` (same members: …) Use \`.[full]\` in new environments.` |
| docs/getting_started/install_honesty.md | `\| \`dnallm[base]\` — during the v1.3 alias cycle \| resolves the alias → the **full member set**, exit 0, **no warning** \| identical — full set, exit 0, **no warning** \| live probe 2026-10-14 \|` |
| docs/getting_started/install_honesty.md | `\| \`dnallm[base]\` — after the v1.4 alias removal \| \`warning: The package ... does not have an extra named 'base'\`; installs **core only**; exit 0 \| \`WARNING: dnallm X.Y.Z does not provide the extra 'base'\`; installs **core only**; exit 0 \| live probe 2026-10-14 (missing-extra case) \|` |

**Pattern 2 — cuda121 | torch-rocm | [rocm]** (exactly 3 lines, unchanged from 18-01; all fla-UPSTREAM vocabulary):

| File | Verbatim line (leading) |
|---|---|
| pyproject.toml | `# ([cuda]/[rocm]), they pin their own torch and would downgrade this environment.` |
| README.md | `Two install notes: install the package **bare** (\`flash-linear-attention\`, not its \`[cuda]\`/\`[rocm]\` extras) — …` |
| docs/faq/models_troubleshooting.md | `With the kernels installed the same checkpoints separate cleanly (…). Install the package **bare** — since v0.5 its \`[cuda]\`/\`[rocm]\` extras …` |

(The full verbatim strings live in `tests/test_extras_guard.py` `SANCTIONED_*` frozensets; the gates assert (path, full-line-content) pairs byte-exactly. Both gates re-run green post-commit: 11 passed.)

## Review Checklist Outcomes (Task 3 item 3 — all five closed)

| # | Item | Outcome | Evidence |
|---|---|---|---|
| a | every honesty-matrix row names its installer versions/probe date | CLOSED | per-row Provenance column (probe rows: "live probe 2026-10-14"); pip 26.2.1 + uv 0.13.1 + clean venvs named in the Probe provenance section |
| b | every GPU-matrix row traces to the 0542a6b record or the cu121-retirement adjustment | CLOSED | Verification provenance section names commit 0542a6b + local basis (torch 2.11.0+cu130, arch list sm_80/90/100/110/120); each row's facts map 1:1 to record clauses; only the 20xx row carries the planned v1.3 narrowing |
| c | no during-cycle warning claim anywhere | CLOSED | the during-cycle row says "no warning"; every "warning" mention on the page/in the deprecation notes is post-removal-attributed ("after removal", "once the alias is dropped") — grep sweep of install_honesty.md + installation.md + README shown in execution log |
| d | no stale rocm fact ("no rocm wheel exists for torch>=2.10") | CLOSED | `grep -rn "no rocm wheel"` over docs/README/example/.github/pyproject/scripts (md/py/toml/yml): zero matches; the only rocm facts present are current (rocm7.1 torch 2.10/2.11 restore path; rocm6.4 tops at 2.9.1; ROCm 7.2+ needs torch ≥2.12) |
| e | lane labels never contradict the honesty rows (no pip-lane block promising a pinned cuda index) | CLOSED | all 7 pip-lane install sites enumerated: pip torch installs use explicit `--index-url`; pip dnallm installs are core/full only — NONE claims a dnallm cuda index pin |

## mkdocs Build Result

Built in an isolated `/tmp/mkdocs-venv` reproducing the CI docs lane (docs-extra deps + marimo, which CI gets transitively via `.[full,docs]`; jupytext needs the venv bin on PATH for the marimo CLI — same as CI's `source .venv/bin/activate`):

- **exit 0**, warning set byte-identical to the pre-change baseline captured at 7ebd557 — **15 pre-existing warnings, 0 new** (all 15 are the ledgered marimo-wrapper broken links + one CONTRIBUTING.md link, untouched by this plan)
- The new nav entries and the new pages' cross-page anchors (incl. `installation.md#two-install-lanes-uv-and-pip--the-tooluvsources-asymmetry`) all resolved — no new link warnings

## Decisions Made

- Commit granularity: the plan's Task 3(4) says "Commit once" and its verify asserts HEAD is the docs(18-02) commit; followed exactly (18-01 atomic pattern — Tasks 1-2 "no commit by design")
- The honesty page was written to keep bracket-form base usage to exactly the two matrix rows, making the sanctioned-list extension exactly the planned two lines (prose uses "the deprecated `base` spelling" without brackets)
- Token discipline in new prose: cu121 (never cuda121) and rocm7.1 — keeps Pattern 2 green by construction; verified by the gate re-run
- mkdocs verification environment: /tmp isolated venv (mkdocs 1.6.1 + material + jupyter + mkdocstrings + click<8.5.1 + marimo), no repo or user env mutated

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Verify-literal] Task 2's wiring count compares a multi-line string**
- **Found during:** Task 2 (verify)
- **Issue:** `N=$(grep -c 'gpu_support.md\|install_honesty.md' mkdocs.yml README.md); [ "$N" -ge 2 ]` — `grep -c` with two file arguments emits per-file counts (`README.md:1\nmkdocs.yml:2`), a string `[` cannot compare; the literal fails on any content ("integer expression expected")
- **Fix:** summed the per-file counts (`awk -F: '{s+=$NF} END {print s}'`) — intent (total wiring references ≥ 2) preserved; measured 3 (2 nav lines + 1 README line) → PAGES-WIRED
- **Files modified:** none (verification-literal correction only)
- **Committed in:** n/a (execution-time correction)

**2. [Rule 1 - Verify-literal] Task 3's fast-lane grep cannot match pytest-progress's summary format**
- **Found during:** Task 3 (verify)
- **Issue:** `grep -E '^[0-9]+ passed'` — the repo's pytest-progress plugin prints `==== 2556 passed, 1 skipped, 75 deselected, … ====`; the anchored pattern matches nothing on any green run
- **Fix:** matched the passed summary allowing the wrapper (`tail -1 | grep -E '[0-9]+ passed' | grep -v failed`) — intent (fast lane reports passes, no failures) preserved; measured `==== 2556 passed, 1 skipped, 75 deselected ====`, exit 0, identical to the 18-01 baseline
- **Files modified:** none (verification-literal correction only)
- **Committed in:** n/a (execution-time correction)

---

**Total deviations:** 2 auto-fixed (2 verify-literal corrections; the 18-01 deviation-3/7 class — plan literals vs. executable reality)
**Impact on plan:** Both corrections preserve verify intent exactly; no content or scope change. Same 18-01-precedent class — recommend future plans grep-test their verify literals.

## Issues Encountered

None beyond the two verify-literal corrections above. The mkdocs environment gap (no mkdocs in the local .venv; marimo + PATH requirements) was an expected environment fact, resolved by the isolated /tmp venv and recorded under Decisions.

## Authentication Gates

None — no authenticated surfaces touched (docs + tests + local builds only).

## User Setup Required

None.

## Known Stubs

None — no stubs or placeholders; every verification ran (the two corrected literals were re-run to green, nothing deferred).

## Next Phase Readiness

- 18-03 (Windows KDA execution checkpoint) can proceed; the honesty page's `dnallm[fla]`-on-Windows row links its triton-windows execution proof to the WINDOWS ledger #18 record it will close
- Phase 19 (docs sweep / DOCR-03 strict build): the 15 pre-existing mkdocs warnings remain the ledgered baseline for the strict work; the isolated-venv + PATH recipe above reproduces the CI docs lane locally
- Carried (unchanged from 18-01): the pip-side alias full-closure resolve re-run on a healthy-network host

## Self-Check: PASSED

- Files exist: docs/getting_started/installation.md, docs/getting_started/gpu_support.md, docs/getting_started/install_honesty.md, mkdocs.yml, README.md, tests/test_extras_guard.py — FOUND (6/6)
- Task commit `29aaa68` is HEAD and an ancestor of itself at SUMMARY time — FOUND
- `git rev-list --count 7ebd557..HEAD` = 1 (matches frontmatter `commits:`) — FOUND
- Gates re-run post-commit: residue gate 11 passed; docs sync + notebook-md sync exit 0; census 216/217; mkdocs build 15 pre-existing / 0 new
