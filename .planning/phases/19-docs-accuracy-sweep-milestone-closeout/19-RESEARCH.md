# Phase 19: Docs Accuracy Sweep & Milestone Closeout - Research

**Researched:** 2026-10-14
**Domain:** docs/ + example/ accuracy verification against the final v1.3 code state; mkdocs strict build; cuda130-default re-pointing; milestone-close CI honesty
**Confidence:** HIGH (nearly every load-bearing claim was verified empirically this session: two mkdocs builds, a full 15-warning fix experiment, gate re-runs, CI run inspection)

## Summary

The phase is an infrastructure-shaped sweep: **309 files** (docs/ 227 + example/ 82) must each carry a per-file accuracy disposition, the cuda124-era default install commands must be re-pointed to cuda130 everywhere, `mkdocs build --strict` must reach zero warnings, and the milestone must close with the CI matrix honest-green. The good news proven this session: the DOCR-03 surface is **tiny and fully solved** — all 15 ledgered warnings were reproduced byte-exactly, and a 4-file docs-only edit (13 relative-path fixes + 1 target correction + 1 GitHub-URL substitution) drove a strict build from 15 warnings to 1 warning in one pass, with the last warning's fix mechanically identified (the `api/inference/visualization.md` link target does not exist; the real page is `user_guide/inference/visualization.md`). No example/ file participates in the strict fix at all. The DOCR-01/PKGH-05 bulk is the ~120 docs prose pages + ~29 example wrapper pages human-grade review, of which the stale spots are already enumerable: 2 cuda124 sites in the mirrored `overview.md`, ~10 cuda124-defaulted install sites in `installation.md`, ~17 docs-side notebook wrappers carrying `.[full,cuda124]`, plus prose staleness confirmed at `docs/user_guide/continuous_integration.md:17` (describes the phase-13-deleted ollama stage 3), `README.md:170` ("5090 → cuda128 & torch==2.7" contradicting the 18-02 GPU matrix), `README.md:30/558/580` ("1,900+ tests" vs the 2556 fast-lane baseline), `.github/workflows/README.md:59` ("CUDA versions: 12.1, 12.4" — cuda121 retired in 18-01), and two broken anchors the 18-02 pages link into `installation.md`.

The one structural trap for the planner is the **CI topology**: v1.3 is NOT in ci.yml's push/PR branch filter (only main/master/dev), so pushes to v1.3 fire nothing and a manual dispatch fires only the three nightly legs. "Milestone closes honest-green (fast lane, coverage gate, full CI matrix)" therefore requires an explicit choice — PR into main/master/dev (matrix legs fire), adding v1.3 to the push filter for the closeout, or taking the merge-to-main push itself as the matrix-green evidence. The nightly legs run serialized on the single self-hosted `dnallm-nightly` runner (measured: coverage-nightly ~3h43m, example-nightly ~4h01m, test-mamba ~5 min; ~8h serial wall).

**Primary recommendation:** structure the phase as (1) one strict-clean commit (the 4-file, 15-warning fix — fully pre-solved below), (2) one cuda130-default re-point commit over the enumerated install surface with the mirrored-file discipline for `overview.md`, (3) batched per-file review waves over the ~120 prose pages with the disposition ledger growing in the phase dir, (4) the milestone-close evidence battery (local fast lane + coverage gate + dispatched nightlies + a matrix-legs path chosen at plan time).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (carried constraints, roadmap, phases 16-18)
- Sweep lands AFTER every surface it describes stopped moving (deps 16/17/18 — satisfied; the two owner-deferred acceptance items, WINDOWS #25/#27, do not move docs surfaces)
- cuda130 recommended default everywhere: overview.md's 3 stale `.[base,notebook,cuda124]`-class mentions + all install surfaces, lane-labeled per the phase-18 split
- mkdocs --strict zero warnings with the nav/mirror interplay respected; marimo + PATH requirement recorded by 18-02 for exactly this work
- Automated sync/masking gates stay green throughout; example edits mirrored byte-identical same-commit

### Claude's Discretion
Disposition-record shape (chosen: phase-dir review ledger per the 16-EVIDENCE precedent — one row per file, verdict + fix summary), sweep ordering, warning-clearance mechanics.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description (REQUIREMENTS.md verbatim) | Research Support |
|----|----------------------------------------|------------------|
| PKGH-05 | "cuda130 is the recommended default install across all surfaces (overview.md's 3 stale `.[base,notebook,cuda124]`-class mentions + docs + examples)" | Full install-surface census below (§ cuda130 Re-Point Surface); the "3 mentions" is the pre-18-01 count — current overview.md carries 2 cuda124 sites (lines 9, 25, base→full already swept by 18-01); the 18-RESEARCH OQ3 resolution explicitly parks the ci.yml test-cuda leg re-point in this phase |
| DOCR-01 | "Every file under docs/ and example/ is reviewed per-file for accuracy vs current code and post-numpy2/py311 staleness, with fixes in the same pass (beyond the automated sync/masking gates)" | The Inventory below is the per-file task structure; stale-spot classes enumerated with file:line; the automated-gate blind spots documented in § Automated Gates |
| DOCR-02 | "Install commands are re-verified against the final extras surface — the sweep lands after the packaging phase so reviewed docs describe the final state" | Final extras surface recorded verbatim from pyproject.toml; base→full residue verified zero beyond the sanctioned 4+3 lines (extras gate 11 passed this session) |
| DOCR-03 | "`mkdocs --strict` builds with zero warnings (the 15 ledgered pre-existing warnings cleared; nav/mirror interplay respected)" | Exact 15-warning byte-record reproduced this session + a proven fix mapping that reaches zero (§ DOCR-03 Mechanics) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Docs accuracy content (DOCR-01/02) | Repo docs tree (docs/, example/, README) | — | Static markdown/notebooks; correctness is a content property, no runtime |
| Strict build verification (DOCR-03) | Local isolated mkdocs venv (build tool) | CI `deploy` job (gh-pages, non-strict today) | mkdocs is a build-time checker; CI never runs strict (verified: deploy runs `mkdocs gh-deploy --force` only) |
| Mirror integrity (example edits) | Repo gate `scripts/check_docs_sync.py` | `check_notebook_md_sync.py` (AST) | Byte-identical mirror is enforced by script, not review |
| cuda130 default publication (PKGH-05) | docs prose + README + ci.yml test-cuda leg | GPU matrix page (18-02, accurate already) | Default-recommendation is guidance content; the CI leg is an install surface |
| Milestone-close green evidence | GitHub Actions matrix | Self-hosted `dnallm-nightly` runner (nightly legs) | Push legs run on GitHub-hosted; nightly legs serialize on the single runner |

## The Inventory (the plan's task structure)

**Totals [VERIFIED: `find docs -type f | wc -l` = 227; `find example -type f | wc -l` = 82; 309 files]**

| Class | Count | Review disposition | Gate coverage |
|-------|-------|--------------------|---------------|
| docs/ prose pages (everything outside docs/example) | **120 md** | **Human-grade accuracy review — the core DOCR-01 workload** | None (only validate_docs_snippets syntax check) |
| docs/example md wrappers (24 `notebook:`-frontmatter pairs + 3 marimo See-Also wrappers + overview.md + selection.md) | **29 md** | Accuracy review; install-command re-point lives here (17 wrappers carry `.[full,cuda124]`) | check_notebook_md_sync (24 pairs, AST statement-level) |
| Mirrored both-sides md | 2 (`example/notebooks/overview.md`, `.../plant_helixseek_shared/data/selection.md`) | overview.md is a PKGH-05 headline file; selection.md is the frozen truth source — **do not touch its data content** | check_docs_sync byte-exact |
| Notebooks | 48 ipynb (24 × both sides) | Executed green in nightly run 38441258701 (2026-10-14, rice-caveat lanes aside) — low staleness risk; review for prose cells only | check_docs_sync + nightly execution |
| Example YAML configs | 42 (21 × 2) | Loaded by real `load_config()` in the nightly — low risk | validate_yaml + nightly YAML leg |
| Marimo .py apps | 6 (3 × 2) | Executed in nightly stage 1; no cuda124 mentions inside (verified grep) | check_docs_sync + nightly |
| Data files (csv/tsv/fas/gff/gtf/pkl/xlsx/bedGraph/fxi) | 48 (24 × 2) | No accuracy review (not prose); mirror-verified | check_docs_sync |
| README.md (repo root) + .github/workflows/README.md | 2 | Outside DOCR-01's literal set but inside PKGH-05's "all surfaces" — both carry stale claims (below); include them | extras residue gates grep the repo incl. README |

Per-subdirectory prose counts (docs/ excl. example): user_guide 40, api 35, concepts 15, resources 15, faq 9, getting_started 5, index 1 [VERIFIED: find per dir].

### Known stale spots (verified this session, the expected catch classes)

**Class 1 — cuda124-as-default install commands (PKGH-05 re-point surface):**

| File | Sites | Form |
|------|-------|------|
| example/notebooks/overview.md (MIRRORED — edit both sides same-commit) | :9, :25 | `uv pip install -e '.[full,notebook,cuda124]'` (the requirement's headline file; the "3 mentions" was the pre-18-01 `.[base,...]` count — 18-01's 399f9f1 swept base→full, leaving 2 cuda124 sites) |
| 17 docs-side notebook wrappers (finetune_binary, finetune_multi_labels, finetune_NER_task, finetune_NER_data_generation, finetune_custom_head, finetune_generation, lora_finetune, lora_inference, inference, inference_evo_models, inference_megaDNA, inference_generation, inference_trna, mutagenesis, interpretation, embedding_attention, benchmark, data_prepare_finetune, data_prepare_predict — docs/example/notebooks/*.md, line ~17-24 each) | 1 each | `uv pip install -e '.[full,cuda124]'` (these wrappers are docs-only files — no mirroring needed; install commands sit in ```bash blocks, invisible to the AST sync gate) |
| 3 marimo wrappers (docs/example/marimo/{benchmark,finetune,inference}/*_demo.md, line 18 each) | 1 each | same form |
| docs/getting_started/quick_start.md | :96 (+ :328 hardware-group list, fine) | `uv pip install -e '.[cuda124]'` |
| docs/user_guide/getting_started.md | :67 | `uv pip install -e '.[cuda124]'` |
| docs/index.md | :78-79 | menu comment + example `uv pip install -e '.[all,cuda124]'` |
| docs/user_guide/tutorial/end_to_end_workflow.md | :983 | `uv pip install -e '.[full,cuda124]' --system` |
| docs/getting_started/installation.md | ~10 sites (:120 GPU Support lead example, :274, :309, :370, :474, :497, :573, scenario blocks) | cuda124 leads the GPU section; cuda130 appears second — **flip the default order**, keep the lane labels |
| README.md | :167 (`.[all,cuda124]` headline), :202 (extras table row "cuda124 … Most modern GPUs (recommended)" — contradicts gpu_support.md), :212, :177 (`.[cuda124,mamba]` example) | :170 "**Nvidia 5090 Please use cuda128 & torch==2.7**" — stale pre-18-02 guidance contradicting gpu_support.md ("50xx needs cuda128+; cuda130 works, torch 2.9+") |
| .github/workflows/ci.yml test-cuda | matrix `cuda-version: ['12.4']`, install line (:263) `uv pip install -e ".[full,cuda124]"`, cache-key prefix `uv-cuda-12.4` | The OQ3-carried PKGH-05 surface — see Open Questions |

**Re-point rule (honesty constraint):** cuda130 becomes the *default recommendation* everywhere generic; the 20xx (Turing) guidance "cuda124–cuda128, NOT cu130" from `docs/getting_started/gpu_support.md:15` is correct and must survive — blanket substitution without the GPU-matrix pointer would create a NEW inaccuracy (20xx cannot run cu130; arch list starts sm_80). gpu_support.md itself (published 18-02 from the 0542a6b record) is accurate as-is — verify, don't rewrite.

**Class 2 — claims about removed machinery:**
- `docs/user_guide/continuous_integration.md:17` — "torch-heavy execution, then MCP live-server probes, **then the ollama-backed `mcp_example` pair**" — phase 13 (D-04, 2026-10-11) DELETED nightly stage 3; the authoritative `.github/workflows/README.md:123/125` records the deletion. The docs page contradicts the workflows README — fix the docs page to match. [VERIFIED: both files read this session]
- `.github/workflows/README.md:59` — "CUDA versions: 12.1, 12.4" — cuda121 retired 18-01; the matrix is single-leg `['12.4']` today.
- ollama mentions per se are NOT stale (the mcp_example notebooks stay in-repo per WINDOWS #13; `langchain-ollama>=1.1.0` stays declared in the mcp extra [VERIFIED: pyproject.toml:131]) — only execution/topology claims about it are.

**Class 3 — dated-number drift:**
- `README.md:30/:558` "1,900+ test cases", `README.md:580` "~1,900 tests" — fast-lane baseline is now **2556 passed / 1 skipped / 75 deselected** [VERIFIED: 18-02-SUMMARY D4].
- `docs/user_guide/continuous_integration.md:27/:33/:40` — "96.72%" carries dated v1.2-closeout provenance (honest per the D-07 dated-provenance discipline, but the per-file review should re-measure or re-date; the gate floor itself is `fail_under = 90` [VERIFIED: pyproject.toml:551]).

**Class 4 — 18-02-introduced broken anchors (found by this session's build):** `docs/getting_started/gpu_support.md` and `docs/getting_started/install_honesty.md` both link `installation.md#two-install-lanes-uv-and-pip--the-tooluvsources-asymmetry` (double dash) but the real slug is `#two-install-lanes-uv-and-pip-the-tooluvsources-asymmetry` (single dash — mkdocs's default slugify DROPS the em-dash rather than converting it to a separator). [VERIFIED: `grep -o 'id="two-install-lanes[^"]*"' site/getting_started/installation/index.html` after this session's build; heading at installation.md:213 `### Two install lanes: uv and pip — the `[tool.uv.sources]` asymmetry`] These log as INFO (do not fail strict) but are genuine accuracy defects — in-scope for DOCR-01.

**Class 5 — post-numpy2/py311 staleness:** no `numpy 1.26`/`1.26.4` claims remain in md prose; no Python 3.10 claims outside torch-version tokens [VERIFIED: greps this session]. The 2026-10-10 floors (numpy>=2.0, py>=3.11) are correctly stated in README:83, quick_start:7, installation:7. Expect scattered older torch-version prose to verify per-file, but no confirmed hits beyond README:170 above.

**Class 6 — base→full residue: ZERO beyond sanctioned.** The extras residue gate passed this session (11 passed) — the sanctioned lists are exactly 4 Pattern-1 lines + 3 Pattern-2 lines as byte-recorded in 18-02-SUMMARY [VERIFIED: `pytest tests/test_extras_guard.py` → 11 passed, this session; lists at tests/test_extras_guard.py `SANCTIONED_*` frozensets].

## DOCR-03 Mechanics (the 15 warnings — byte-record and the proven fix)

### The exact warning set [VERIFIED: reproduced this session]

Built in `/tmp/mkdocs-venv19` (mkdocs 1.6.1, mkdocs-jupyter 0.26.3, mkdocs-material 9.7.7, mkdocstrings 1.0.7 / mkdocstrings-python 2.0.9, marimo 0.25.1, jupytext 1.19.6, click 8.5.0 — matching the `docs` extra floors `mkdocs-jupyter>=0.24.0, mkdocs-material>=9.6.1, mkdocstrings-python>=1.16.10, click<8.5.1` plus marimo as CI gets it transitively via `.[full,docs]`). Non-strict build: exit 0, **exactly 15 WARNING lines**, build ~5.5s. Strict build: exit 1, "Aborted with 15 warnings in strict mode!".

All 15, verbatim class:
1-4. `example/marimo/benchmark/benchmark_demo.md` links `example/notebooks/benchmark.md`, `example/marimo/inference/inference_demo.md`, `user_guide/benchmark/getting_started.md`, `api/inference/benchmark.md` — resolved against the page's own dir, not found.
5-9. `example/marimo/finetune/finetune_demo.md` links `example/notebooks/finetune_binary.md`, `example/notebooks/finetune_multi_labels.md`, `example/notebooks/lora_finetune.md`, `user_guide/fine_tuning/getting_started.md`, `api/finetune/trainer.md`.
10-14. `example/marimo/inference/inference_demo.md` links `example/notebooks/inference.md`, `example/marimo/benchmark/benchmark_demo.md`, `user_guide/inference/getting_started.md`, `api/inference/inference.md`, `api/inference/visualization.md`.
15. `user_guide/benchmark/advanced_techniques.md` links `../../../CONTRIBUTING.md` — target outside docs/.

Root cause: the 14 marimo links live in hand-maintained `## See Also` tails of the three docs-side wrappers (NOT generated — `scripts/generate_md_from_marimo.py` emits no See-Also section [VERIFIED: script read this session], so a regen would destroy them; hand-edit, don't regenerate). They use repo-root-relative paths where mkdocs resolves page-relative. Warning 14 is doubly rotten: `docs/api/inference/` contains only benchmark/inference/interpret/mutagenesis/plot — **no visualization.md**; the real page is `docs/user_guide/inference/visualization.md` [VERIFIED: `ls` both dirs]. Warning 15 escapes docs_dir (mkdocs refuses to link outside docs/).

### The proven fix (experimentally driven 15 → 1 → mechanically 0 this session)

| Warning(s) | Fix | Evidence |
|-----------|-----|----------|
| 1-13 (13 links, valid targets) | rewrite repo-root-relative → page-relative (mkdocs's own suggested form, e.g. `../../notebooks/benchmark.md`, `../inference/inference_demo.md`, `../../../user_guide/...`) | Applied to all 3 files this session: warnings 15→1, zero residual hints |
| 14 (rotten target) | re-point target `api/inference/visualization.md` → `../../../user_guide/inference/visualization.md` | Target-existence verified; same mechanism as the 13 |
| 15 (docs-dir escape) | substitute absolute GitHub URL `https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md` (the existing in-repo pattern — continuous_integration.md links `.github/workflows/README.md` this way) | Applied this session: warning cleared |

Do NOT use root-absolute `/example/...` form: it also clears the WARNING but mkdocs logs "contains an absolute link … it was left as is" and `site_url` is commented out in mkdocs.yml — under the gh-pages subpath deployment the links would break at runtime [VERIFIED: both variants tested this session; mkdocs.yml read].

**Nav/mirror interplay, resolved:** the three marimo wrappers are docs-ONLY files (example/marimo holds .py/.yaml/.csv only — the `.md` wrappers pass check_docs_sync via the `DOCS_ONLY_SUFFIXES = (".md",)` right_only exemption [VERIFIED: scripts/check_docs_sync.py:30]) and carry `marimo:` (not `notebook:`) frontmatter so check_notebook_md_sync ignores them. **The entire strict fix is 4 docs-only files; zero example/ edits; zero mirror constraints; zero census risk.**

### Strict-mode semantics [VERIFIED: empirical, both modes run]

- `--strict` promotes WARNING → abort (exit 1 with "Aborted with N warnings in strict mode!").
- INFO lines do NOT abort: broken-anchor hints (the two Class-4 anchors), "pages exist … not included in the nav" (the mirrored .py/.ipynb sources under docs/example), and absolute-link hints all stay INFO. Empirically: strict failed on exactly the 15 WARNINGs while 2 anchor INFOs + the not-in-nav INFO list were present.
- The two Class-4 anchors are therefore DOCR-01 accuracy catches, not DOCR-03 blockers — but fix them in the same pass (they're 2 one-character edits: `--` → `-`).

### The reproducible verification recipe (from 18-02, re-proven this session)

```bash
uv venv /tmp/mkdocs-venv19 --python 3.12
uv pip install --python /tmp/mkdocs-venv19/bin/python \
  "mkdocs-jupyter>=0.24.0" "mkdocs-material>=9.6.1" \
  "mkdocstrings-python>=1.16.10" "click<8.5.1" "marimo>=0.16.3"
PATH="/tmp/mkdocs-venv19/bin:$PATH" mkdocs build --strict   # marimo CLI must be on PATH (jupytext shells out to it)
```
[VERIFIED: exact commands executed this session; recorded by 18-02-SUMMARY key-decisions as "will recur for Phase 19's DOCR-03 strict work"]

## Standard Stack

### Core (all already in-repo — this phase installs NOTHING new)

| Tool | Version (resolved this session) | Purpose | Why |
|------|------|---------|-----|
| mkdocs | 1.6.1 | strict build verification | the docs toolchain already pinned via the `docs` extra |
| mkdocs-jupyter | 0.26.3 | notebook wrappers in nav | provides the marimo/notebook conversion (jupytext 1.19.6 → marimo CLI) |
| mkdocs-material / mkdocstrings(-python) | 9.7.7 / 1.0.7+2.0.9 | theme + API pages | unchanged |
| marimo | 0.25.1 | marimo CLI for jupytext shell-out | must be on PATH during build |
| pytest (repo .venv) | existing | fast lane + gates | 2556-passed baseline intact |

**Installation:** none. `pip` on this box stalls (urllib3; recorded 18-01 decision) — use `uv` for any venv work.

## Package Legitimacy Audit

No packages are installed by this phase (docs/config-only; the /tmp mkdocs venv is throwaway tooling re-created from the recipe above, packages already vetted by the `docs` extra floors and 18-02's identical lane). **Packages removed/flagged: none.**

## Architecture Patterns

### Pattern 1: Per-file disposition ledger (the 16-EVIDENCE precedent, CONTEXT-chosen)
**What:** a phase-dir evidence file (`.planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-EVIDENCE.md` or equivalent) with one row per reviewed file: `| File | Verdict | Fix summary |`. 16-EVIDENCE's table shape is two-column `| Assumption | Disposition |` with bold verdict + one-line evidence; 13-01 carried its 59-ref per-assertion ledger inside the PLAN. For 309 files, rows can group mirror pairs and data-file classes (the CONTEXT wording is "one row per file" — a wrapper+its mirror is one logical file).
**When to use:** every review batch lands rows in the same commit as the fixes.

### Pattern 2: Mirror discipline (WPAR-08 / check_docs_sync)
Any edit to a file present on BOTH sides (overview.md is the only md; also any .ipynb/.yaml/.py if the sweep touches them) lands example-side and docs-side **byte-identical in the same commit**. The gate compares with `shallow=False` (always reads bytes) [VERIFIED: scripts/check_docs_sync.py:79-81].

### Pattern 3: Lane-label vocabulary (18-02, "the sweep's vocabulary")
Every dnallm/torch install block in installation.md carries `# [uv lane]` / `# [pip lane]` (29 labels at 18-02). New/edited install commands during the re-point must keep or add labels; pip-lane blocks must never promise a cuda index pin [VERIFIED: 18-02-SUMMARY review item e].

### Pattern 4: Census stability (D-03)
The sweep's docs edits must not move the census triple. The pinned collection line is `216/217 tests collected (1 deselected)` under the exact stage flags `pytest tests/examples --collect-only -q -m "not giants"` [VERIFIED: ci.yml:850-867; assertion greps `^=* *216/217 tests collected \(1 deselected\) in `]. Nothing in the strict fix or install re-point touches test collection — but the ci.yml test-cuda edit (if taken) is a workflow install-line change, which also leaves collection untouched.

### Recommended task shape (maps 1:1 to the inventory)

```
Wave 1  strict-clean commit (4 docs-only files, pre-solved above) + the 2 anchor one-char fixes
Wave 2  cuda130 re-point: overview.md (mirrored, both sides) + 17+3 wrappers + quick_start/index/
        user_guide-getting_started/tutorial + installation.md default-flip + README (incl. :170,
        :202 "recommended" row, test counts) + workflows README :59
Wave 3+ per-file review batches over the 120 prose pages (user_guide 40, api 35, concepts 15,
        resources 15, faq 9, getting_started 5, index 1) + 29 wrappers; ledger rows same-commit
Final   milestone-close evidence battery (see below)
```

### Anti-Patterns to Avoid
- **Regenerating the marimo wrappers** via generate_md_from_marimo.py to "fix" the See-Also links — the generator emits no See-Also section; regeneration destroys hand-maintained content.
- **Blanket cuda124→cuda130 substitution** without the Turing exception — creates a fresh inaccuracy for 20xx GPUs (gpu_support.md:15).
- **Root-absolute `/path` links** — clear the warning but break under the gh-pages subpath (site_url unset).
- **Editing `selection.md` data content** — it is the frozen single truth source parsed by tests on both platforms.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mirror verification | ad-hoc diff scripts | `python scripts/check_docs_sync.py` | byte-exact, shallow=False, ignores the sanctioned runtime artifacts |
| Notebook/md code sync | manual reading | `python scripts/check_notebook_md_sync.py` (24 pairs) | AST statement-level, tolerates formatting |
| Bracket-residue policing | fresh greps | `pytest tests/test_extras_guard.py` (11 tests) | byte-exact sanctioned frozensets already enforced |
| Link validation | custom crawler | `mkdocs build --strict` exit code | the checker this requirement names |
| Skip honesty | eyeballing | `python scripts/audit_skips.py <junit> tests/expected_skips.yaml` | the established typed-skip audit |

**Key insight:** every mechanical guarantee in this phase already has a gate; the sweep's unique work is precisely what the gates CANNOT see (prose accuracy, stale claims, default-recommendation honesty).

## Automated Gates: exact commands and blind spots

| Gate | Command | Catches | Blind to (the sweep must catch) |
|------|---------|---------|----------------------------------|
| Mirror sync | `python scripts/check_docs_sync.py` | byte drift example/ ↔ docs/example/ | accuracy of the shared content itself |
| Notebook-md sync | `python scripts/check_notebook_md_sync.py` | md ```python statements missing from the notebook (AST) | ```bash install commands (unchecked), prose claims, markdown cells |
| Snippet syntax | `python scripts/validate_docs_snippets.py` | Python-syntax errors in docs code blocks | semantics/staleness of the snippets |
| YAML validity | `python scripts/validate_yaml.py` | config syntax | prose about configs |
| Example tests | `pytest tests/examples/test_examples.py -v` | execution breakage | docs pages never executed |
| Extras residue | `pytest tests/test_extras_guard.py` | bracket-form base/cuda121/rocm residue | prose claims, default recommendations |
| mkdocs build (deploy lane) | CI `deploy` job: `mkdocs gh-deploy --force` | build failure | warnings (non-strict today) |

The CI lane that runs the doc gates is `.github/workflows/docs-validation.yml` (sync + snippets + YAML + example/yaml tests) — but its push filter is main/master/dev + PRs, so on v1.3 it runs only via PR or after merge [VERIFIED: docs-validation.yml:3-10].

## Runtime State Inventory

> The cuda124→cuda130 re-point is a string-replacement-class change; categories answered explicitly.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — docs-only phase; no DB/store keys on cuda124 (grep over docs+example+workflows shows only the enumerated prose/CI sites) | none |
| Live service config | ci.yml test-cuda leg (matrix `['12.4']`, install line :263, cache key `uv-cuda-12.4`) — the only executable cuda124 consumer; runner `.env` already carries `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130` + `UV_INDEX_STRATEGY=unsafe-best-match` (cu130-aligned) | ci.yml edit only if PKGH-05 re-point taken; runner .env needs nothing |
| OS-registered state | Self-hosted runner `dnallm-nightly` (id 22, `~/github-runners/dnallm-nightly`, nohup-started — Listener PID alive this session) | none for the sweep; milestone-close dispatch needs it online (recovery manual in project memory `phase13-uat-nightly-recovery`) |
| Secrets/env vars | None reference cuda124 | none |
| Build artifacts | `site/` (gitignored, regenerable), `/tmp/mkdocs-venv19` (may not persist — recreation recipe recorded), uv cache | none — rebuild via recipe |

## Common Pitfalls

### Pitfall 1: The verify-literal class (18-01 deviation-3/7, twice in 18-02)
**What goes wrong:** plan verify literals that don't match executable reality (grep -c multi-file output compared with `[`, `^[0-9]+ passed` vs pytest-progress's `==== N passed ====` wrapper).
**How to avoid:** grep-test every verify literal before writing it into the plan; prefer the pytest-progress-tolerant form `tail -1 | grep -E '[0-9]+ passed' | grep -v failed`.

### Pitfall 2: The em-dash slug trap
**What goes wrong:** hand-derived anchors assume punctuation → hyphen; mkdocs's default slugify DROPS the em-dash (single dash in the slug), so `--the-` links break quietly at INFO level.
**How to avoid:** never hand-compute anchors — copy the `id="..."` from a built `site/**/index.html` (the exact working command is recorded in § DOCR-03).

### Pitfall 3: The Turing exception on the cuda130 re-point
Covered above — the re-point changes the DEFAULT, and every re-pointed block should keep/link the GPU-matrix pointer. The extras table row at README:202 ("cuda124 … (recommended)") must move the "recommended" label to cuda130, not just swap strings.

### Pitfall 4: Expecting push legs on v1.3
**What goes wrong:** planning "full CI matrix green" as `git push origin v1.3` + wait — pushes to v1.3 fire NOTHING (branch filter main/master/dev), and a workflow_dispatch fires only coverage-nightly + example-nightly + test-mamba (D-19 cron gates).
**How to avoid:** choose the matrix-green path explicitly (PR / branch-filter line / merge-to-main push) at plan time — see Open Questions.

### Pitfall 5: Runner serialization at milestone close
The single dnallm-nightly runner serializes coverage-nightly (~3h43m) + example-nightly (~4h01m) + test-mamba (~5m) behind any queued foreign/scheduled work (~hours of queue latency observed in 16-EVIDENCE A.3). Budget wall-clock accordingly; cancel-and-re dispatch only runs you own. Expected: ~8h serial from a clean queue.

### Pitfall 6: Census movement by accident
Any edit that changes test collection (e.g., "while I'm here" test edits) breaks the `216/217 (1 deselected)` hard gate and violates one-census-move-per-phase. The sweep's edits are docs-only by construction — keep them that way; the only ci.yml edit on the table (test-cuda leg) does not touch collection.

## State of the Art (repo-internal currency table)

| Old (pre-v1.3) | Current (final state the sweep verifies against) | Changed |
|----------------|---------------------------------------------------|---------|
| `.[base]` extras spelling | `.[full]` + one-cycle alias `base = ["dnallm[full]"]` (remove v1.4) | 18-01 (399f9f1) |
| cuda121 extra + uv index/sources/conflicts + matrix 12.1 leg | all retired; index frozen torch 2.5.1 | 18-01 |
| rocm extra | removed; manual rocm7.1 restore path documented (installation.md:272) | 18-01 |
| triton-windows undocumented | `triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'` in fla | 18-01 |
| cuda124 as example default | cuda130 default everywhere — **this phase's job** | Phase 19 |
| mkdocs 15 warnings tolerated | strict zero — **this phase's job** | Phase 19 |
| ollama nightly stage 3 | deleted; pair typed-skips via network-unavailable | Phase 13 |

Final extras surface (verbatim member names) [VERIFIED: pyproject.toml:81-191]: `dev, test, notebook, docs, ui, mcp, full, base(alias), all, cpu, cuda124, cuda126, cuda128, cuda130, mamba, fla`; uv indexes `pypi, torch-cpu, torch-cuda124, torch-cuda126, torch-cuda128, torch-cuda130`; cuda130 spec `torch>=2.9.0,<2.12`; local verified basis torch 2.11.0+cu130.

## Milestone-Close CI Matrix (concrete honest-green recipe)

**Local (fast, minutes):**
- Fast lane: `.venv` activated, `pytest -m "not slow" --cov` — baseline **2556 passed / 1 skipped / 75 deselected**, exit 0 [VERIFIED: 18-02-SUMMARY D4].
- Gates: extras guard (11 passed — re-run this session), check_docs_sync, check_notebook_md_sync (24 pairs), validate_docs_snippets, validate.
- Census: `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"` → `216/217 tests collected (1 deselected)` [VERIFIED: ci.yml:850-867 flags].
- Strict: the §DOCR-03 recipe → exit 0.

**Push legs (GitHub-hosted, parallel; durations from the last full push run 38110680128, 2026-10-11):** test ×3 (py3.11/3.12/3.13 × numpy 2.2.0) ~7-11 min each; test-windows (py3.12) ~7 min; test-cuda (py3.11) ~9 min; coverage-gate (py3.12) ~6 min; deploy ~2.5 min (main/master push only, gh-pages). Whole push run ≈ 13 min wall.
**Nightly legs (self-hosted runner, serialized; durations from run 38441258701 records):** coverage-nightly ~3h43m (full census incl. slow — 2607 passed at the 16-04 record); example-nightly ~4h01m (staged census, stage-0.5 hard gate, selection.md bands); test-mamba ~5 min. Dispatch: `gh workflow run ci.yml --repo zhangtaolab/DNALLM --ref v1.3` — fires exactly these three (push/PR legs skip on dispatch).

**Known standing caveat:** the rice.uga.edu genome-download fragility (owner-accepted since phase 14; 4th/5th manifestations in run 38441258701) can red two input-download lanes unrelated to any repo change — the 16-EVIDENCE A.5 triage-disposition pattern (classify before retry, never silent) is the closeout precedent.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The Phase 19 plan takes the test-cuda re-point (OQ3 parked it here) — direction (13.0 vs keep 12.4) is an owner call at plan time | PKGH-05 / Open Questions | Minor: one-line matrix + install + cache-key edit either way; "all surfaces" wording pulls toward 13.0 |
| A2 | README.md and .github/workflows/README.md are in scope despite sitting outside DOCR-01's literal docs/+example/ set (PKGH-05 "all surfaces" + both carry confirmed stale claims) | Inventory | Low: if excluded, PKGH-05 is under-delivered; recommend including |
| A3 | The mkdocs toolchain resolves to the same majors when the plan executes (mkdocs 1.6.1 line; strict semantics stable) | DOCR-03 | Low: recipe pins by the extra floors; re-run the build to re-baseline if >30 days |
| A4 | ~120 prose pages is the effective human-review surface (data/notebook/yaml classes disposed via gates + nightly execution evidence) | Inventory / ledger shape | Low: planner may tighten to literal per-file rows for all 309 — heavier but mechanical |
| A5 | The matrix-legs green path for milestone close is chosen at plan time (PR vs branch-filter vs merge push) — no repo evidence locks one | CI Matrix | Medium: choosing "wait for merge" defers SC4 evidence past phase close; recommend PR or a temporary v1.3 push-filter entry decided with the owner |

## Open Questions (RESOLVED 2026-10-14 — all three owner-answered at discuss-phase; dispositions below)

1. **test-cuda leg direction (A1) — RESOLVED: 13.0 (default-everywhere consistency).** Owner accepted re-pointing the leg to CUDA 13.0 (19-CONTEXT "收尾方向"): matrix `cuda-version: ['13.0']`, install `.[full,cuda130]`, the templated cache key follows automatically; `.github/workflows/README.md:59` updated in the SAME change. The 20xx lane stays covered by uv resolves + the GPU-matrix prose, and cuda124 remains a supported extra (the Turing-row exception survives in the docs matrix only). Implemented: 19-07 Task 1.
2. **Matrix-green path at closeout (A5) — RESOLVED: temporary branch-filter vehicle.** Owner declined dev-merge for now; the vehicle is a temporary `- v1.3` entry in ci.yml's push branches filter (GitHub-hosted legs only — push never reaches the self-hosted box per the D-19 invariant; deploy stays main/master-only). It stays until actual merge or explicit removal; the removal decision is recorded at closeout in 19-EVIDENCE.md. Implemented: 19-07 Task 1 (entry + dated lesson comment) and Task 2 (the push-fired matrix evidence).
3. **Ledger granularity — RESOLVED: ~150 logical-file rows.** Owner decision (19-CONTEXT): data/mirror-pair classes collapsed by category — per-batch phase-dir review ledgers (19-REVIEW-*.md, one row per reviewed file with verdict + fix summary) consolidated into 19-EVIDENCE.md at closeout, not a literal 309-row dump.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| uv | venvs, resolves | ✓ | 0.13.1 | — |
| mkdocs toolchain (isolated venv) | DOCR-03 verification | ✓ (built this session at /tmp/mkdocs-venv19) | mkdocs 1.6.1 / material 9.7.7 / marimo 0.25.1 | recreate via recorded recipe |
| repo .venv | fast lane, gates | ✓ (extras gate 11 passed this session) | py3.13 | — |
| gh CLI (authenticated) | nightly dispatch, run inspection | ✓ (forrestzhang, workflow scope) | — | web UI dispatch (manual) |
| dnallm-nightly runner | nightly legs | ✓ Listener alive (PID 31376, since 2026-10-12) | id 22 | recovery manual in project memory |
| pip on this box | — | stalls (urllib3; 18-01 record) | — | use uv everywhere |

**Missing dependencies with no fallback:** none.
**Missing with fallback:** pip → uv (already the convention).

## Validation Architecture

SKIPPED — `workflow.nyquist_validation: false` in .planning/config.json. (The phase's verification is the gate battery + CI matrix above, per the roadmap's SC4.)

## Security Domain

ASVS level 1 (`security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`). Docs-only phase — no new code paths.

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | nothing added |
| V3 Session Management | no | nothing added |
| V4 Access Control | no | nothing added |
| V5 Input Validation | no | no new input surfaces (markdown content only) |
| V6 Cryptography | no | nothing added |
| V12 File & Resources | indirectly | edits confined to docs/, example/, README, ci.yml leg; no secrets present (verify no tokens leak into published pages — trivial grep at closeout) |

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious external links injected via docs edits | Tampering/Phishing | link targets limited to github.com/zhangtaolab/DNALLM + the pytorch/uv doc domains already used; per-file review is the control |
| Secrets in published pages | Information Disclosure | grep sweep for token-shaped strings in the closeout battery |

## Sources

### Primary (HIGH confidence — all verified by direct execution/inspection this session)
- Live mkdocs builds (non-strict + strict + two fix experiments): /tmp/mkdocs-build-19-base.txt, -strict.txt, -test2.txt — exact 15-warning byte-record, fix mechanics, INFO/WARNING boundary
- scripts/check_docs_sync.py, scripts/check_notebook_md_sync.py, scripts/generate_md_from_marimo.py, scripts/validate_docs_snippets.py — read in full
- .github/workflows/ci.yml (push filter, jobs, stage-0.5 census, test-cuda leg, deploy), docs-validation.yml, pyproject.toml (extras surface), mkdocs.yml (nav, site_url)
- docs/user_guide/continuous_integration.md vs .github/workflows/README.md (stage-3 contradiction); docs/getting_started/{installation,gpu_support,install_honesty}.md; example/notebooks/overview.md; README.md
- Phase artifacts: 18-02-SUMMARY.md (recipe, sanctioned lists, lane vocabulary), 18-RESEARCH.md (install-line inventory, OQ3), 16-EVIDENCE.md (disposition precedent, nightly durations, queue forensics), v1.2 deferred-items.md (original 15-warning ledger)
- CI records via gh: run 38110680128 (push-leg durations), run 38441258701 records (nightly durations), run list on v1.3 (topology proof)

### Secondary (MEDIUM)
- None required — no external claims needed; all mkdocs behavior verified empirically rather than cited

## Metadata

**Confidence breakdown:**
- DOCR-03 mechanics: HIGH — warnings reproduced and the fix driven to (mechanically-identified) zero this session
- Inventory/stale-spot census: HIGH — file:line-cited greps over the whole surface
- CI topology/durations: HIGH — filters read; durations from real runs; v1.3 non-triggering proven by run list
- cuda130 re-point edge cases: HIGH on the census; MEDIUM on per-file prose beyond the enumerated spots (that residual is exactly what the per-file review exists to catch)

**Research date:** 2026-10-14
**Valid until:** 2026-11-13 (stable repo-internal facts; re-run the mkdocs build to re-baseline if the docs extra floors move)
