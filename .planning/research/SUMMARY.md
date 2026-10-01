# Project Research Summary

**Project:** DNALLM — milestone v1.1 "Example Execution Testing & Repair"
**Domain:** Real-execution testing of Jupyter/marimo example artifacts under pytest + PlantHelixSeek genomics showcase (prediction-vs-experimental-truth presentation) on an existing CI-hardened ML toolkit
**Researched:** 2026-10-01
**Confidence:** HIGH overall — findings are grounded in direct repo inspection, empirical probes of the installed venv (nbclient 0.11.0, marimo 0.25.0, kernelspec, CLI behavior), PyPI JSON API, and official docs. Environment-specific unknowns (aarch64 toolchains, transformers-5 remote code) are named as gaps below rather than guessed at.

## Executive Summary

This milestone adds two things to an already-mature pytest/CI system: (1) a real-execution test layer that runs all ~21 `example/` notebooks, 3 marimo apps, `generate_bpe_dataset.py`, and every example YAML through real `load_config()` on the self-hosted nightly GPU runner — fixing every error that surfaces; and (2) a PlantHelixSeek showcase vertical — CRE and Anno notebooks that run sliding-window inference on committed ≤200 kb Arabidopsis loci and assert *substantial* agreement against PlantDHS / TAIR10 ground truth. Experts build exactly this as a thin pytest layer over **nbclient used as a library** (the engine under nbmake/nbconvert, minus the plugin), one kernel per notebook, tmp-sandbox cwd isolation, partial-notebook failure artifacts, layered per-cell/per-test timeouts, and a preregistered model cache (`models.lock`). The existing repo machinery — `slow` marker leg split, typed-skip allowlist + `audit_skips.py`, junit artifacts, `hashFiles('models.lock')` cache key — already provides every CI seam this needs; no new markers, no new frameworks.

The recommended approach: harness mechanics in a private `tests/examples/_execution.py` + locally-scoped `conftest.py` (never root conftest, never `dnallm/`); marimo apps via subprocess; PlantHelixSeek-CRE/-Anno onboarded through **generic registry entries + the existing task-type loader route** (verified: no special-handler substring collision, only hard remote-code dep is `einops` which is core; `fla`/`flash_attn` are guarded with pure-torch fallbacks); showcase rendering via **static altair** (already installed, survives nbconvert→HTML into the docs mirror) with **stdlib GFF3 parsing + pyfastx** (1-based inclusive, same convention as GFF3) and pyBigWig as an *optional* emission step; agreement asserted as **thresholded Jaccard/F1 floors**, never exact outputs (transformers spans 4.49–5.x). WR-08/WR-09 (false-green docs gate, missing `mcp` extra) are fixed first so all subsequent repair rides an enforced lane.

The dominant risks are environmental, not conceptual: (a) **leaked jupyter kernels** after hung/timed-out tests poison the *persistent* runner's VRAM (pytest-timeout kills the pytest process, not the process group) — mitigate with `NotebookClient` context-manager use, timeout layering where the nbclient cell timeout fires *first*, and a nightly `pkill`/VRAM hygiene step; (b) **timeout arithmetic** — ~24 new slow tests with naive ceilings sum to 14–48 h against a 900-min job, so per-artifact budgets must be measured, not assumed; (c) **GitHub's 10 GB cache quota** — evo-1 is a 29.7 GB repo whose blind inclusion evicts the existing warm cache (filter to safetensors ≈12.9 GB, hold giants on-disk outside the cached paths); (d) **evo/evo2/megaDNA toolchain infeasibility on the aarch64 GB10 box** (flash-attn x86 wheels, Transformer Engine pre-Blackwell, FP8-needs-Hopper) — these are environment-gated, not content-gated: feasibility-spike first, typed `environment-unavailable:` skips otherwise; (e) **silently-empty genomic results** from 0-based/1-based and `Chr1` vs `1` chrom-naming mismatches — one shared, unit-tested normalization helper plus non-emptiness assertions and a negative-control locus; and (f) **showcase overfitting framing** — loci are chosen because they agree, so the notebook must present "illustrative loci + selection criteria," never accuracy claims. Flipping WR-08 honest detonates accumulated docs-mirror drift — inventory and close drift in the same unit as the gate flip.

## Key Findings

### Recommended Stack

Every addition was verified against the installed venv and PyPI; the pre-validated stack (pytest 8.4+/cov/timeout/asyncio, torch 2.11 cu130, transformers 4.49–5.x, HF/ModelScope loading, `dnallm-nightly` runner) is untouched. Only three extras change: `nbclient>=0.10` → `notebook`, `langchain-ollama>=1.1.0` → `mcp` (the actually-missing import, currently shell-magic-installed by a notebook cell), `pyBigWig>=0.3.26; platform_system != 'Windows'` → `dev`.

**Core technologies:**
- **nbclient 0.11.0 (already installed) as a library** — executes the ~21 `.ipynb` as parametrized pytest tests; default `allow_errors=False` stops at the first failing cell with `CellTimeoutError`/`CellExecutionError` carrying source + traceback (the exact repair signal); `resources={"metadata": {"path": ...}}` sets kernel cwd. Chosen over nbmake (new plugin semantics, duplicate timeout machinery — violates no-new-frameworks) and `nbconvert --execute` (subprocess opacity).
- **marimo via subprocess** — `marimo export html app.py` (0.25.0, verified: executes headlessly, HTML artifact) or script mode `python app.py` (verified: UI elements yield `value=` defaults, exit 1 on cell error). Subprocess is mandatory — `App.run()` executes in-process, leaking app state/CUDA into pytest (Pitfall 7).
- **pyfastx 2.3.1 (already in `dev`)** — FASTA region slicing; `fetch(name, (start, end))` is 1-based inclusive, identical to GFF3 coordinates, with reverse-complement in the same call.
- **stdlib GFF3 reader/writer + bisect interval math** — GFF3 is nine tab-separated columns; at ≤200 kb loci a ~60-line strict parser beats gffutils/BCBio (dependency weight for nothing) and feeds agreement metrics *and* track rendering.
- **altair 6.3 + vl-convert-python 1.9 (already installed via `altair[all]`)** — static track/gene-model figures; `chart.save("png")` embeds `image/png` that survives nbconvert→HTML and the mkdocs-jupyter mirror. **Ruled out:** pyGenomeTracks (GPL-3.0 in an MIT project, `matplotlib<3.9` pin conflicts with installed 3.11.2, external bedtools binary, no documented GFF3), jbrowse-anywidget (not on PyPI — CI-non-reproducible), igv-notebook in gated cells (widget output doesn't survive nbconvert; optional non-gated appendix only).
- **pyBigWig 0.3.26** — BigWig emission from CRE score tracks; default zoom levels on `close()` (never `maxZooms=0`). **See Gap 1: aarch64 wheel availability on the GB10 runner is unverified — treat as optional/guarded.**
- **ollama as runner infrastructure (not a pip package)** — systemd service on `127.0.0.1:11434`, `OLLAMA_MODELS` drop-in cache dir, pre-pulled `qwen3.6:latest`, readiness probe `curl -sf localhost:11434/api/tags` → typed `network-unavailable:` skip.
- **Env hygiene** — `MPLBACKEND=Agg`, `MPLCONFIGDIR=$(mktemp -d)`, `WANDB_MODE=disabled` in the kernel env (belt-and-braces against config drift to wandb hanging the nightly on a login prompt).

### Expected Features

**Must have (table stakes, all P1):**
- [EXEC] nbclient harness: per-notebook parametrized tests, per-cell + per-test timeouts, tmp-cwd isolation, kernel cleanup, partial-notebook failure artifacts uploaded `if: always()` — the spine everything hangs on
- [CI] Execution tests `slow`-marked into the nightly census only (marker deselection = zero new fast-leg skips); `models.lock` extended with revision pins; typed skips reused/extended (`network-unavailable:`, one new `optional-dep:`/`environment-unavailable:` prefix)
- [EXEC] Real execution of 20–21 notebooks, 3 marimo apps, `generate_bpe_dataset.py`, all YAMLs through real `load_config()`
- [REPAIR] Every surfaced error fixed (examples, docs mirror, library) with regression tests — the milestone's core value
- [CI] WR-08 (remove docs-validation `continue-on-error` false green) + WR-09 (add `mcp` extra) — cheap, unblock honest enforcement
- [REG] `model_info.yaml` entries for PlantHelixSeek-CRE (binary, 2 labels) and -Anno (token, 17 BILOU) via the generic loader route
- [SHOW] Both showcase notebooks: committed ≤200 kb loci, side-by-side prediction-vs-truth presentation, Jaccard/F1 statistics, test-asserted agreement floors, docs-mirror sync

**Should have (differentiators, v1.x):**
- models.lock consistency guard (fast-leg test cross-checking notebook model literals against lock entries)
- Executed-notebook write-back / rendered figures in the docs mirror for the two showcase notebooks only (highest-leverage credibility artifact)
- In-notebook peak calling (mean±1.5σ → BED/narrowPeak) so Jaccard operates on called peaks; RC-TTA toggle

**Defer (v2+):**
- Notebook parallelism (only if a second GPU runner appears), output-regression testing (nbval-style), additional species' loci, per-cell timing diagnostics

**Anti-features (research is emphatic):** `allow_errors=True` (cascading noise — stop-at-first-error is the repair signal); xdist on the single GPU (VRAM flake on exactly the run that must be trustworthy); mutating notebooks under test; committing executed notebooks for all 21; IDR scoring (replicate-concordance tool, wrong question); exact-output assertions (guaranteed-flaky across the transformers span); a `special/` PlantHelixSeek handler (dispatch risk for zero capability — the CrossDNA bug lived in that chain); full-genome inference (CUDA-mandatory, hours-scale, blows the 200 kB rule).

### Architecture Approach

The additions slot in as **one new nightly-only test layer** and **one new example vertical**; the fast leg's semantics are untouched (structural tests + `load_config()` auto-cover new files via `rglob` from commit #1). The loader needs **no changes** — registry metadata plus the generic `_load_model_by_task_type` route covers both checkpoints (`trust_remote_code=True` already forwarded to model and tokenizer).

**Major components:**
1. `tests/examples/_execution.py` + `tests/examples/conftest.py` — private harness mechanics (nbclient wrapper, marimo subprocess runner, `seed_sandbox`, `assert_tree_clean`) and locally-scoped fixtures; mirrors the `dnallm/mcp/tests/_network_skip.py` seam. A `NOTEBOOK_EXEC_SPECS` dict is the single per-artifact timeout/input tuning table.
2. Four new test modules — `test_notebook_execution.py` (parametrized over all notebooks), `test_marimo_execution.py`, `test_example_script_execution.py`, `test_plant_helixseek_examples.py` (execution + truth-agreement asserts, including `id2label` equality).
3. Showcase vertical — `example/notebooks/plant_helixseek_{cre,anno}/` with notebook + YAML + committed `data/` fragments (FASTA + truth GFF/GFF3 slices) + per-dir `.gitignore` for download scratch.
4. Support-contract extensions — `models.lock` (~8+ ids, `hf`/`ms` prefix matching each notebook's actual `source=`, revision pins), `model_info.yaml` `finetuned:` entries, `expected_skips.yaml` (+1 typed prefix), docs mirror + `mkdocs.yml` nav (currently drifted — see below).

**Key patterns:** tmp-copy + cwd-redirect sandboxing (kernel cwd = sandbox; never execute in-place; belt-and-braces `git status --porcelain` guard); timeout ladder with nbclient cell timeout (~600–900 s) as the *inner* guard below the pytest mark (1800 inference / 3600 showcase / 7200 finetune) below the 900-min job; truth-in-the-loop — loci selected during the phase for substantial agreement, floors asserted nightly thereafter.

### Critical Pitfalls

1. **Leaked kernels poison the persistent runner** (Critical #1) — pytest-timeout kills the pytest process, never the process group; orphan `ipykernel_launcher` holds VRAM and failures appear as random OOM in *unrelated* tests the next night. Use `NotebookClient` as a context manager, `shutdown_kernel="immediate"` where graceful hangs, layer timeouts so the cell timeout fires first, and add a nightly `pkill -f ipykernel_launcher || true` + `nvidia-smi` hygiene step. Ship with a deliberate-hang kill test.
2. **Timeout-arithmetic and cache-quota economics** (Critical #2, #4) — naive per-test ceilings sum to 14–48 h vs the 900-min job (job kill = no junit + cache forfeit); GitHub's 10 GB/repo cache quota is LRU-evicted *regardless of runner type* — one 29.7 GB evo-1 fetch evicts the existing warm cache. Measure budgets in Phase 1; filter evo-1 to safetensors via `allow_patterns`; split cache tiers (quota-bounded cache for small/medium, persistent on-disk dir for giants).
3. **Environment-gated families: evo/evo2/megaDNA cannot ever run on aarch64 GB10 as-shipped** (Critical #5) — flash-attn ships x86_64 wheels, Transformer Engine 2.3.0 predates Blackwell, evo2's big tiers need FP8-on-Hopper; megaDNA's `git clone && pip install -e .` is unpinned arbitrary setup.py. Feasibility-spike first, typed `environment-unavailable:` skips for the infeasible, smallest-viable variants otherwise, and revision-pin everything (`trust_remote_code` + mutable refs = unreviewed code execution on the box — Critical #8).
4. **Hermeticity and the cwd false-repair** (Critical #3) — notebooks assume their own directory (`load_config("./x.yaml")`), write `ath_cds.csv`/tensorboard events/BEDs to cwd, and shell-magic `!wget`/`!uv pip install`; the #1 false positive is a `FileNotFoundError` misread as a notebook bug and "fixed" by editing the notebook. Copy-to-tmp + cwd redirect + tree-clean guard + pre-extended `.gitignore`; triage harness-bug vs content-bug explicitly.
5. **Silent emptiness in genomics code** (Critical #11, #12, #13) — 0-based half-open bigWig vs 1-based closed GFF3, case-sensitive `Chr1` vs `1` chrom names (wrong name returns `[]`, not an error), arabidopsis.org serving HTML-as-`.gz` to non-browser clients, and cherry-picked-loci-as-benchmark framing. One shared unit-tested normalization helper, non-emptiness assertions everywhere, magic-byte download validation, committed-artifact loci selection with a rationale doc, and a negative-control region.
6. **Mirror drift detonates when WR-08 flips honest** (Critical #14) — `check_docs_sync.py` exits 1 *today* (wrapper `.md` handling, drifted outputs, missing script mirror); removing `continue-on-error` without closing drift blocks unrelated PRs. Inventory drift first, flip the gate with the closure in one reviewable unit, then regenerate mirror MD as part of every notebook repair.

## Implications for Roadmap

Suggested phase structure (5 phases; Phases 1 and 2 are independent and parallelizable; 3 depends on 2; 4 depends on 1; 5 validates all):

### Phase 1: Execution Harness, Honest Gates & Runner Feasibility
**Rationale:** The harness is the spine everything hangs on, and the two honesty repairs (WR-08/09) must land before any repair work rides an enforced lane. The GB10 feasibility verdicts must exist *before* execution tests are written for evo/megaDNA/marimo families — tests written for models that cannot ever run there are pure waste.
**Delivers:** `tests/examples/_execution.py` + local `conftest.py` (tmp-sandbox, timeout layering, kernel-cleanup kill test, tree-clean guard, artifact capture); `.gitignore` extensions; typed skip prefixes (`optional-dep:`/`environment-unavailable:`) + allowlist entries; `models.lock` revision-pin scheme; WR-08/WR-09 fixed together with the mirror-drift inventory and closure; feasibility verdict matrix for evo-1/evo2/megaDNA/marimo/pyBigWig on the actual box; harness piloted green on 1–2 already-healthy notebooks.
**Addresses:** [CI] WR-08/09, [EXEC] harness table-stakes, [CI] marker/skip discipline
**Avoids:** Pitfalls 1, 2, 3, 5, 6, 7 (pattern), 8, 14 — all marked "Phase 1" in the pitfall-to-phase mapping

### Phase 2: Model Registry & Showcase Data Curation
**Rationale:** Registry entries are dependency-free and unblock everything; loci selection is the long pole of the showcase (prediction-truth agreement must be *measurable* to select loci, and the committed fragments are a precondition of both notebooks' final form). Coordinate/chrom normalization must exist before any notebook is written against the data.
**Delivers:** `model_info.yaml` entries for CRE/-Anno with label lists frozen from checkpoint `config.id2label` (dnallm *overrides* id2label from config — a permuted 17-BILOU list silently permutes predictions); smoke-load test via the generic route (transformers 5.17 dev env first — the one genuine compat risk); shared GFF3/BigWig coordinate + chrom-name normalization helper with unit-tested tiny fixtures; loci-selection script (live downloads OK inside it, magic-byte validated) producing committed ≤200 kb fragments + truth slices + selection-rationale doc + one negative-control locus.
**Addresses:** [REG] entries, [SHOW] committed loci precondition
**Avoids:** Pitfalls 11, 12, 13 (data-acquisition half), Architecture Anti-Pattern 4 (label-order guessing)

### Phase 3: PlantHelixSeek Showcase Notebooks
**Rationale:** Needs registry (Phase 2) for inference and committed loci for truth; produces the milestone's flagship user-facing artifacts.
**Delivers:** CRE notebook (500 bp/50 stride/50 bin sliding-window scan via `DNAInference`, score track + optional guarded pyBigWig emission + altair side-by-side vs PlantDHS, peak-overlap Jaccard); Anno notebook (8192/4096 both-strand scan, BILOU span decode → structurally valid GFF3, nucleotide/exon-level P/R/F1 vs TAIR10, exon/intron block diagrams); YAMLs valid for the fast-leg `load_config` gate from the first commit; "illustrative loci + selection criteria" framing; docs-mirror + mkdocs nav entries.
**Addresses:** [SHOW] CRE + Anno table-stakes; differentiator: rendered figures path
**Avoids:** Pitfalls 11 (rendering half), 13 (framing), 10 (tolerance bands not exact asserts)

### Phase 4: Full Execution Rollout & Repair Loop
**Rationale:** Needs the proven harness (Phase 1) and repaired gates; the repair loop is the open-ended unknown-unknowns sink, so it comes last-among-content phases with the harness stable beneath it.
**Delivers:** Parametrized execution of all notebooks + 3 marimo apps (subprocess, cheap-default assertions) + `generate_bpe_dataset.py` + real YAML validation; ollama runner setup (loopback systemd unit, idempotent `qwen3.6:latest` pre-pull, port plan vs the 6 MCP live-server probes on :8000); `models.lock` extension with correct `hf`/`ms` prefixes per actual route; fix-everything-surfaced loop with regression tests and per-fix mirror regeneration; invariant/tolerance assertion conventions (valid DNA alphabet, finite in-range scores, `defs` shape asserts — never prose or exact values).
**Addresses:** [EXEC] real execution of everything, [REPAIR] fix-all loop, [CI] models.lock preregistration
**Avoids:** Pitfalls 4 (cache tiers — the first evo-class run is the one that evicts), 7, 9, 10; per-notebook timeboxing keeps the repair loop bounded

### Phase 5: CI Wiring & Census Verification
**Rationale:** Last because it validates everything: nightly collection of the new slow tests, warm-cache behavior, runtime budget, and audit gates can only be verified against the finished test set.
**Delivers:** Nightly census collecting all new slow tests; junit + `audit_skips.py` green *with* the new skip categories present; cache-size/`gh cache list` report step; kernel/VRAM hygiene steps; timeout-arithmetic comment updated and sum-of-ceilings review checklist; documented coverage expectation (kernel subprocesses are unmeasured by design — the 96.30% gate neither rises nor falls from example execution).
**Addresses:** [CI] nightly integration end-to-end
**Avoids:** Pitfalls 1 (hygiene), 2 (arithmetic review), 6 (skip-count growth as review trigger)

### Phase Ordering Rationale

- **Dependency chains from FEATURES.md:** showcase notebooks ← registry entries + committed loci + harness; repair loop ← harness + artifact capture + honest gates (WR-08/09 first, or fixes ride an unenforced lane); rendered docs figures ← both notebooks green.
- **Pitfalls demand harness-before-rollout:** every "Phase 1" pitfall (kernel leaks, timeout layering, hermeticity, skip taxonomy, feasibility, provenance pins) is a property of the harness, and each is cheapest to build in on day one — retrofitting isolation into 24 red tests is the failure mode.
- **Grouping follows the architecture's two verticals:** test-layer work (Phases 1, 4, 5) and example-vertical work (Phases 2, 3) touch disjoint files and can proceed in parallel or in either order where staffing allows.
- **Post-MVP differentiators stay out:** models.lock consistency guard, executed-notebook write-back, peak calling, per-cell timing are v1.x — after the lock entry set stabilizes and the showcase content stops churning.

### Research Flags

Phases likely needing deeper research during planning (`/gsd-plan-phase --research-phase`):
- **Phase 2 (showcase data curation):** the densest open questions converge here — pyBigWig-on-aarch64 resolution (Gap 1), chrom-naming harmonization against the *actual* committed files, threshold-calibration methodology, arabidopsis.org/plantdhs.org acquisition reliability. Loci selection is a methodology, not a pattern.
- **Phase 4 (partially — ollama/GB10 specifics):** runner service ops (systemd drop-ins, VRAM co-residency with torch models, port serialization vs MCP probes) are environment-specific and under-documented; a short research pass or in-phase spike is warranted. The repair loop itself is not researchable.

Phases with standard patterns (skip research-phase):
- **Phase 1 (harness):** nbclient/marimo semantics were verified *empirically in the project venv* (traits, cwd handling, per-cell timeout firing, script-mode exit codes); the repo's own precedents (`_network_skip.py`, tmp-path PDF fix, timeout ladder) are the spec.
- **Phase 3 (notebooks):** upstream pipeline parameters (500/50/50 CRE; 8192/4096 17-BILOU Anno) are verified from the upstream repo; altair rendering is already exercised by `dnallm/inference/plot.py`.
- **Phase 5 (CI wiring):** pure extension of existing workflow patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Everything cross-verified against installed venv + PyPI JSON + official docs; weak seams are web-only single-channel claims (pyBigWig write API details) and the aarch64 gap below |
| Features | HIGH | Project-grounded items read directly from repo (test_examples.py, expected_skips.yaml, ci.yml, models.lock, model_info.yaml); ecosystem patterns cross-checked |
| Architecture | HIGH | Repo-verified + empirical probes (marimo script mode, nbclient semantics, dispatch-chain inspection, remote-code reading); upstream model facts MEDIUM (single-org) |
| Pitfalls | HIGH | Direct repo inspection + official docs (pytest-timeout issues, GitHub cache limits, pyBigWig coordinates note, ollama security, evo2 requirements); runner arch verified via `nvidia-smi` |

**Overall confidence:** HIGH — an unusually well-grounded research set. Residual uncertainty is concentrated in the named environment facts below; none block the roadmap structure.

### Gaps to Address

- **Gap 1 — pyBigWig on aarch64 (cross-file conflict, resolve in Phase 1 spike):** STACK.md verified pyBigWig 0.3.26 ships manylinux **x86_64** wheels only and proposed `platform_system != 'Windows'` on the assumption of an x86_64 runner; PITFALLS.md verified the nightly runner is **aarch64 GB10**, where that marker does not protect (sdist compile, needs toolchain + libcurl headers); FEATURES.md's "linux/mac/win wheels" claim is the outlier. Mitigation is already structural — BigWig *emission* is the only pyBigWig consumer (plots need numpy/altair only), so keep it an optional guarded cell until the spike proves the import on the runner.
- **Gap 2 — evo/evo2/megaDNA feasibility on GB10 (LOW confidence):** flash-attn x86-only wheels, TE pre-Blackwell, FP8-requires-Hopper, unpinned megaDNA clone. Default posture: typed `environment-unavailable:` skip; enabling any of them is a stretch goal behind a Phase 1 capability spike.
- **Gap 3 — PlantHelixSeek remote code under transformers 5.x:** upstream pins 4.49; remote code imports `transformers.cache_utils.Cache`. Dev env is 5.17 — smoke-load locally in Phase 2 before freezing anything; fallback is a typed skip + upstream issue, not a dnallm shim.
- **Gap 4 — marimo execution flavor:** STACK prefers `marimo export html` subprocess (full runtime path, HTML artifact); ARCHITECTURE prefers script-mode `python app.py` (empirically verified). Both subprocess-based. Spot-check one `mo.ui`-bearing app under both in the Phase 1 pilot and standardize.
- **Gap 5 — nightly runtime budget:** ~24 new slow tests; worst-case serial ceilings (14–48 h) far exceed the 900-min job. Phase 1 must measure real per-artifact runtimes; the escalation path (separate/sharded nightly job for example execution) should be pre-approved in the roadmap.
- **Gap 6 — ollama/`langchain-ollama` wiring:** STACK verified the notebook shell-magics `!uv pip install -U langchain-ollama` (add to `mcp` extra + repair the cell); ARCHITECTURE's import scan didn't see it. Also unresolved: VRAM co-residency ordering with torch models, and the :8000 port plan vs the 6 MCP live-server probes. Settle in Phase 4 planning.
- **Gap 7 — agreement thresholds:** inherently un-researchable — calibrated at loci-selection time with recorded observed values, tolerance bands for cross-device float drift, and re-derivation tied to any model-revision rotation.

## Sources

### Primary (HIGH confidence)
- Direct repo inspection (2026-10-01): `tests/examples/test_examples.py`, `tests/configuration/test_yaml_load.py`, `tests/expected_skips.yaml`, `scripts/audit_skips.py`, `.github/workflows/ci.yml` + `docs-validation.yml`, `models.lock`, `pyproject.toml`, `dnallm/models/{model_info.yaml,model.py,tokenizer.py}`, `dnallm/models/special/`, `scripts/check_docs_sync.py` (run: exit 1), `example/` notebooks/marimo/apps (side-effect greps, committed-output audit, evo/megaDNA/ollama cell sources), `.gitignore`, per-dir example `.gitignore`s, `dnallm/utils/support.py`
- Empirical probes in the project venv: nbclient 0.11.0 traits + cwd handling + per-cell `CellTimeoutError`; marimo 0.25.0 script mode (UI defaults, exit codes) + CLI help + `App.run` signature; venv kernelspec content; vl-convert importability; `pytest.mark.timeout` availability; `nvidia-smi` (GB10); bedtools absent from runner PATH
- PyPI JSON API (fetched 2026-10-01): versions/wheels for nbclient, pyBigWig, pyfastx, marimo, ipykernel, vl-convert-python, langchain-ollama, gffutils, bcbio-gff, pyfaidx, papermill, nbmake, ollama, pygenometracks, igv-notebook; jbrowse-anywidget absent from PyPI (deterministic check)
- Official docs: nbclient client/reference; pytest-timeout issues #134/#159; GitHub Actions limits + actions/cache (+Nov 2025 >10 GB changelog); pyBigWig README coordinate note; GFF3 spec v1.26; ollama docs (install/OpenAI-compat/security issue #16236); W&B headless mode; evo2 PyPI/GitHub requirements; HF `togethercomputer/evo-1-131k-base` file listing (29.7 GB incl. redundant 16.8 GB `.pt`)

### Secondary (MEDIUM confidence)
- Upstream PlantHelixSeek: github.com/zhangtaolab/PlantHelixSeek (`scripts/cis_regulatory` 500/50/50, `scripts/gene_annotation` 8192/4096 17-BILOU) + HF model cards + raw remote-code inspection (single-org, cross-checked); ModelScope twin existence
- marimo docs (testing/CLI), jupyter_client kernelspec docs, matplotlib backends FAQ, altair saving docs, nbmake docs, upload-artifact #328
- Field conventions: bedtools jaccard; GffCompare sensitivity/precision/F1; Enformer/ChromBPNet/AlphaGenome track-presentation norms; pyGenomeTracks docs
- Community evidence for TAIR `Chr1` vs Ensembl `1` naming (verify against actual committed files at selection time)

### Tertiary (LOW confidence)
- arabidopsis.org SPA/login-wall behavior (repo-internal milestone knowledge + TAIR portal state reports; mitigations valid regardless)
- evo2-on-GB10 impact (requirements HIGH, arch inference MEDIUM); pyBigWig under matrix numpy 1.26.4 (flagged for phase spike)

---
*Research completed: 2026-10-01*
*Ready for roadmap: yes*
