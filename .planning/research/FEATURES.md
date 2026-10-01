# Feature Research

**Domain:** Real-execution testing of Jupyter/marimo examples + genomics showcase notebooks (prediction-vs-ground-truth presentation) for an existing pytest/CI-hardened ML toolkit
**Researched:** 2026-10-01
**Confidence:** HIGH for project-grounded items (read directly from repo: `tests/examples/test_examples.py`, `tests/expected_skips.yaml`, `scripts/audit_skips.py`, `.github/workflows/ci.yml`, `models.lock`, `dnallm/models/model_info.yaml`); MEDIUM for ecosystem patterns (cross-checked across independent sources); LOW where individually noted

"Users" are the DNALLM maintainer, the CI system, and future contributors. A "feature" here is a capability of the v1.1 example-execution program, not a library feature. Areas are tagged so REQUIREMENTS.md can group them:

- **[EXEC]** notebook/marimo/script execution harness mechanics
- **[CI]** gating, markers, skips, caching, artifacts
- **[REPAIR]** error-repair workflow
- **[REG]** PlantHelixSeek model registry integration
- **[SHOW]** showcase notebooks: prediction-vs-truth presentation + agreement assertions

## Feature Landscape

### Table Stakes (Users Expect These)

How mature ML projects execute notebooks in CI (verified pattern, MEDIUM confidence): a pytest layer over **nbclient** — nbmake is the most popular packaging of it; pytest-notebook/nbval do output-regression instead. The consistent behaviors: **per-cell timeout** inside the executor, each notebook an **independent test** (one failure never halts the others), **fresh kernel per notebook**, executed/partial notebooks **saved and uploaded as artifacts** (`if: always()`), and model downloads satisfied from a **pre-seeded cache**.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| [EXEC] nbclient-based execution of all 20–21 `.ipynb` under `example/`, each notebook one parametrized pytest test | PROJECT.md commits to nbclient. nbclient is the engine nbmake/nbconvert sit on; using it directly in a parametrized suite mirrors the existing `tests/examples/test_examples.py` discovery pattern (module-level `rglob`, `ids=relative path`) and respects the "no new test frameworks" constraint (skip nbmake — it would add a pytest plugin for conveniences we get from pytest itself) | MEDIUM | `NotebookClient(nb, timeout=..., kernel_name=...)`; default `allow_errors=False` stops at the first failing cell with `CellExecutionError` carrying cell source + traceback — exactly the repair signal we want. Do NOT set `allow_errors=True` |
| [EXEC] Per-notebook (per-test) timeout overrides on top of the per-cell timeout | nbclient `timeout` is **per cell** (default 30s in the API; `None`/`-1` disables) and `startup_timeout` (60s) bounds kernel start. The repo addopts `--timeout=300` is **per test** and will kill any real-model notebook. Both layers are needed: per-cell catches a hung cell; per-test bounds total notebook wall time | LOW | pytest-timeout supports a per-test `@pytest.mark.timeout(N)` marker (MEDIUM confidence — verify against installed pytest-timeout during implementation). Budget: 21 notebooks inside the nightly job's existing 180-min cap → calibrate per-family (finetune notebooks cost most) |
| [EXEC] Fail-soft across notebooks (continue-on-error at suite level), fail-fast within a notebook | Ecosystem norm: nbmake treats each notebook as an independent pytest item; pytest continues past failures. A full nightly census is worthless if notebook 3 kills the run and notebooks 4–21 never execute — one flaky GPU alloc would blind the census. Within a notebook, stop at first error (default nbclient behavior) so the saved artifact pinpoints the failing cell | LOW | Free from pytest semantics (no `-x` in nightly census command — current nightly line is `pytest -ra` full suite, correct). This is the answer to the orchestrator's continue-on-error question: **yes across notebooks, no within a notebook** |
| [EXEC] Fresh kernel per notebook + guaranteed kernel shutdown on failure | Kernel bleed (stale globals, GPU memory not released) is the classic flake source across sequential notebook runs. nbclient shuts the kernel down even on error (`on_notebook_error` fires *before* kernel cleanup; `on_notebook_complete` after), so relying on its lifecycle is sufficient | LOW | Add a per-test fixture teardown with `gc.collect()` + `torch.cuda.empty_cache()` between GPU notebooks — cheap insurance for VRAM fragmentation across 20+ model loads |
| [EXEC] Working-directory isolation per notebook (tmp dir), with copy-in of needed inputs | nbclient sets execution cwd via `resources={'metadata': {'path': <dir>}}`. v1 already paid for this class of bug (Phase 2: PDF tests left 9 stray files in the tree). Notebooks that `mkdir`/write outputs must not dirty the repo or race each other | LOW-MEDIUM | Copy the notebook + its sibling data into `tmp_path`, execute there, then assert the git tree is clean after the suite (reuse the Phase 2 pattern). Resource isolation: **sequential** execution on the single nightly GPU (no xdist) — see Anti-Features |
| [EXEC/CI] Artifact + log capture on failure: save the (partially) executed notebook and captured cell outputs; upload via `actions/upload-artifact` guarded with `if: always()` | The documented nbclient pattern: wrap `client.execute()` in `try/except CellExecutionError` and in `finally` write the notebook with outputs up to the failing cell. Unguarded artifact uploads are skipped on failure — the exact opposite of when they're needed (documented failure mode, upload-artifact issue #328). The existing nightly job already uploads `pytest.log` on failure; extend with executed-notebook artifacts | LOW | Executed notebooks are the *only* way to debug "which cell died on the GPU box at 3am". Name artifacts by notebook path |
| [CI] Execution tests marked `slow`, joining the nightly census — never the fast `-m 'not slow'` leg | Real-model execution needs GPU + network + tens of minutes; the fast PR leg must stay fast. Deselection via `-m "not slow"` means the fast leg never even *collects* these tests → **zero new expected-skips entries on the fast leg** (marker deselection is not a skip; `audit_skips.py` sees nothing). Nightly already runs the full suite including slow | LOW | Direct dependency on existing marker system (`slow` registered, `--strict-markers`). Typed local-run skips reuse the existing `prefix: "network-unavailable:"` allowlist entry — no new skip categories needed unless a GPU-availability guard is added (then one new typed prefix, e.g. `gpu-unavailable:`, must be added to `expected_skips.yaml`) |
| [CI] Preregistered model/dataset dependencies in `models.lock` | The nightly model cache is keyed on `hashFiles('models.lock')`; every remote artifact an execution test touches must be listed or the first run pays full downloads and cache-key drift hides provenance. The lock file is already the reviewable "who fetches what" registry with per-entry test provenance comments | LOW | Add entries for every model each notebook loads (finetune notebooks' backbones, the PlantHelixSeek-CRE/-Anno checkpoints, the PlantHelixSeek-CRE dataset if used). Differentiator below adds the consistency guard |
| [EXEC] Headless execution of the 3 marimo apps | marimo apps are pure Python: `python app.py` runs cells in topological order headlessly; `marimo export html/script --include-outputs` executes and renders; marimo's own CI (marimo-integration-ci) validates a whitelist of example notebooks via export. Interactivity (`mo.ui`) is the wrinkle — headless runs see uninitialized UI elements, so apps needing interaction may require the export path or small mock/arg handling | MEDIUM | Run in tmp cwd with per-test timeout + artifact capture, same harness discipline as ipynb. Apps already survive `ast.parse` + import-exec structural tests, so real execution is the next rung |
| [EXEC] Real `load_config()` validation of every example YAML | Structural tests only `yaml.safe_load` them today; the milestone requires every YAML through real Pydantic validation. Cheap (no network), so it can even run unmarked in the fast leg | LOW | Extends `TestYamlConfigs` in `tests/examples/test_examples.py` or a sibling test module. Watch for example YAMLs that are intentionally partial (CLI-fill-in templates) — those need `pytest.raises(ValidationError)`-style expectation or explicit exclusion with a documented reason |
| [EXEC] Real execution of `generate_bpe_dataset.py` | It is an example artifact like the notebooks; a syntax-checked-but-broken helper is exactly the false confidence this milestone removes | LOW | `runpy.run_path` or subprocess in tmp dir with a tiny input fasta; assert the output dataset file exists and parses |
| [REPAIR] Fix-everything-it-surfaces loop with regression tests | The milestone's core value: execution errors in `example/` code, `docs/example/` mirror, and dnallm library bugs each get a fix + a test that pins it. v1 precedent: AUROC and CrossDNA bugs found by unskipping tests | HIGH | Inherently open-ended — the unknown-unknowns sink. Timebox per notebook; keep the harness independent of repair progress (a red notebook test is a valid intermediate state only on a branch, never main) |
| [CI] Close WR-08 (remove `continue-on-error` false-green in docs-validation) and WR-09 (add missing `mcp` extra) | A gate that reports green while steps fail is worse than no gate — this is the same exit-code honesty principle v1 Phase 1 established for pytest | LOW | Direct edits in `.github/workflows/ci.yml`; keep the exit-code canary untouched |
| [CI] `docs/example/` mirror stays in sync (`scripts/check_docs_sync.py`) including the two new notebooks | Existing enforcement; new notebooks and repaired cells must flow to the mirror or the sync gate fails | LOW | Repo has `scripts/generate_md_from_notebook.py` / `generate_md_from_marimo.py` for regeneration |
| [REG] Registry entries for `PlantHelixSeek-CRE` (binary sequence classification) and `PlantHelixSeek-Anno` (token classification, 17 BILOU) | Only the base `PlantHelixSeek` (task_type `mask`) exists in `dnallm/models/model_info.yaml` (verified line 152); the two task checkpoints are absent and `modeling_auto.py` has no PlantHelixSeek family map. Without entries, `load_model_and_tokenizer` cannot route them | LOW | Generic loading should suffice (upstream loads via `AutoModelForSequenceClassification` / `AutoModelForTokenClassification` with `trust_remote_code=True`); avoid a `special/` handler unless a quirk forces it. Keep task_type values aligned with existing taxonomy (`binary`, token/NER) so `compute_metrics` dispatch works |
| [SHOW] CRE notebook: sliding-window scan + per-bin score track, mirroring upstream `scripts/cis_regulatory` | Upstream convention (verified from repo READMEs, MEDIUM-HIGH): 500 bp window / 50 bp stride / 50 bp bins, class-1 probability averaged over covering windows, emitted as BigWig + npy; optional reverse-complement TTA. The notebook should reproduce this on a committed Arabidopsis locus using the dnallm API (`DNAInference`/`load_model_and_tokenizer`), not the upstream standalone scripts | MEDIUM | For ≤200 kb loci the scan is bounded (~4k windows) — feasible in-notebook on the nightly GPU. BigWig writing needs `pyBigWig` (not currently a dependency — see Dependency Notes) |
| [SHOW] CRE agreement presentation: side-by-side track plot (predicted CRE track vs PlantDHS `TAIR10_DHSs.gff`) + peak-overlap statistic (Jaccard/IoU) | Field convention for prediction-vs-truth at a locus: Enformer's usage Colab plots predicted vs observed tracks stacked on one region; ChromBPNet/AlphaGenome report per-track Pearson on held-out intervals; for *peak-set* agreement, `bedtools jaccard` (intersection/union in bp) is the standard statistic. A showcase notebook that predicts but never overlays truth is decoration, not demonstration | MEDIUM | Compute Jaccard in pure Python/numpy over the two interval sets (avoid a bedtools binary dependency — `pybedtools` is dev-extra and non-Windows). Table-stakes threshold: assert agreement exceeds a calibrated floor in the example test |
| [SHOW] Anno notebook: sliding-window token-classification scan + BILOU→gene-model decode on a committed locus | Upstream `scripts/gene_annotation` convention (verified, MEDIUM-HIGH): 8192 bp window / 4096 stride (50% overlap), both strands, valid-middle stitching, 17 BILOU labels (O + B/I/L/U for CDS, INTRON, UTR5, UTR3), decoded to GFF3 gene/mRNA/CDS/UTR records (upstream default `viterbi+orf`; simpler heuristic decode acceptable in-notebook). On a ≤200 kb locus this is ~50 windows — tractable | HIGH | In-notebook decode can be simpler than upstream's numba Viterbi (direct BILOU-run extraction is the honest minimum), but the GFF3 output must be structurally valid (ID/Parent hierarchy) for the comparison step |
| [SHOW] Anno agreement presentation: side-by-side gene-model diagram (predicted vs TAIR10 GFF3) + match statistics | Gene-prediction convention (MEDIUM): gffcompare-style sensitivity/precision/F1 at nucleotide and exon (and gene) level vs reference GFF3; visualization is exon/intron block diagrams for predicted vs annotated models aligned at the same locus (pyGenomeTracks-style stacking is the field standard for track+gene figures; pure matplotlib gene-model rendering is a well-trodden notebook fallback) | HIGH | Compute nucleotide/exon-level precision/recall/F1 in-notebook from the two GFF3 interval sets (no gffcompare binary). Assert F1 exceeds a calibrated floor on the committed loci |
| [SHOW] In-repo showcase data: committed ≤200 kb Arabidopsis regions (FASTA + PlantDHS GFF + TAIR10 GFF3 slices), gitignored full-genome intermediates | PROJECT.md hard requirement. Loci must be *selected for* substantial prediction-truth agreement and that agreement *asserted by tests* — the curated-loci guarantee is the notebook's credibility | HIGH | Selection is iterative: run predictions across candidate loci, keep those with margin above threshold, commit slices. Add a guard test asserting committed region files exist and are ≤200 kb; `.gitignore` entries for the download scratch dir |
| [SHOW] Test-asserted agreement thresholds (not exact-output assertions) | Transformers spans 4.49–5.x by constraint; logits shift slightly across minors. Assert `jaccard >= floor` / `F1 >= floor` with margin calibrated at loci-selection time; never assert array equality | MEDIUM | Floors live next to the loci data (or in the test module) with a comment recording the observed value at selection time, so drift is diagnosable |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| [CI] models.lock consistency guard: a fast test that cross-checks models referenced by execution tests/notebooks against lock entries | Makes the lock self-verifying instead of convention-maintained — a new notebook silently loading an unlocked model becomes a fast-leg failure instead of a surprise nightly download | LOW-MEDIUM | Simplest honest form: parse each execution-test module for model-name literals, or maintain an explicit per-notebook requirement map the test checks. Keep it literal-matching (no notebook execution) so it runs in `-m 'not slow'` |
| [EXEC] Executed-notebook write-back *option* for docs freshness | nbmake's flagship extra: committing executed outputs speeds docs builds and proves examples run. Here the docs mirror is generated from source notebooks by script, so full write-back duplicates that machinery | MEDIUM | Defer unless the owner wants rendered outputs in docs/example/ (nice showcase value for the PlantHelixSeek notebooks specifically — rendered track plots in the docs mirror are strong marketing). If adopted: write back only showcase notebooks, never all 21 |
| [EXEC] Per-cell timeout granularity reported per cell (`--durations`-style) | Turns the nightly log into a worklist of which *cells* burn the budget — trims the worst notebooks first during repair | LOW | nbclient hooks (`on_cell_executed` with timing) make this a few lines; purely diagnostic |
| [SHOW] Rendered agreement figures committed via the docs mirror | The two showcase notebooks' side-by-side track/gene-model plots appearing on the GitHub Pages docs site is the single highest-leverage credibility artifact this milestone can ship | LOW (once write-back exists) MEDIUM (standalone) | Alternative without write-back: save PNGs as CI artifacts and link from the milestone record; or commit pre-rendered figures for the two showcase notebooks only (small, stable binary cost) |
| [SHOW] Peak-calling reproduction in the CRE notebook (upstream `call_peaks_from_bigwig.py` semantics: mean±k·std, k=1.5; min 50 bp / max 5000 bp; min score 0.6; merge gap 50 bp → BED/narrowPeak) | Elevates the notebook from "score track" to "peaks you can overlap with PlantDHS" — the Jaccard statistic then operates on called peaks, matching how a user would actually validate | MEDIUM | All doable in numpy + interval ops; narrowPeak output is a nice-to-have, BED is the table-stakes subset |
| [SHOW] Reverse-complement TTA toggle in the CRE scan | Mirrors an upstream option (~2x cost, slight accuracy gain) and demonstrates toolkit maturity | LOW | Off by default in the notebook; a single parameterized cell |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| [EXEC] `allow_errors=True` notebook execution ("run the whole thing, collect all errors") | Feels like better triage — one pass, every broken cell listed | Cascading noise: cell 4's failure garbles state for cells 5–40; 30 "errors" from one root cause. nbclient's default stop-at-first-error + saved partial notebook gives the precise failing cell with clean traceback | Default `allow_errors=False`; iterate repair notebook-by-notebook |
| [EXEC] Parallel notebook execution (xdist) on the nightly GPU | "21 notebooks × N minutes is slow" | VRAM contention → flaky OOM on exactly the run that must be trustworthy; model-cache races; nondeterministic reds burn the census's credibility | Sequential execution + per-test timeouts + `torch.cuda.empty_cache()` teardown; parallelism belongs to the future, not the trust-building phase |
| [EXEC] Mutating notebooks under test (nbmake.mock-style epoch shrinking, cell rewriting) | "Make CI fast by shrinking epochs" | Tests something other than what users run — the false confidence this milestone exists to destroy. Repo examples are already small-config | Run notebooks as-is on the nightly GPU; runtime cost accepted by owner per Constraints |
| [CI] Committing executed notebooks / outputs for all 21 notebooks back to the repo | "Prove they ran, in-repo" | Diff churn, merge conflicts, repo bloat; the docs mirror already regenerates from source | Executed notebooks as failure artifacts only; optional curated write-back for the 2 showcase notebooks |
| [SHOW] IDR-based agreement scoring for CRE | "It's the ENCODE reproducibility standard" | IDR scores *replicate concordance* using peak rankings — wrong question for prediction-vs-truth on a single model; also needs scores the showcase may not calibrate | Jaccard/IoU (+ overlap fraction) vs PlantDHS; that is what bedtools-using pipelines do for cross-dataset similarity |
| [SHOW] Asserting exact model outputs / array equality | "Strongest possible guarantee" | Breaks across transformers 4.49–5.x and torch versions — guaranteed-flaky gate across the compatibility span the project must keep | Thresholded agreement assertions (Jaccard/F1 floors) with recorded selection-time values |
| [EXEC] Adopting nbmake (or pytest-notebook/nbval) as the harness | "Popular plugin, less code" | Violates the "no new test frameworks" constraint; hides cwd/artifact control we need (tmp-dir execution, partial-notebook save, per-family timeouts); output-regression tools (nbval/pytest-notebook) solve a different problem (they pin outputs — the anti-feature above) | nbclient directly inside parametrized pytest tests in `tests/examples/` |
| [SHOW] Full-genome inference in the showcase/tests | "Match the upstream genome-wide pipelines exactly" | Upstream CRE/Anno inference is CUDA-mandatory, flash-linear-attention-dependent, hours-scale; blows the 200 kb in-repo rule and the nightly budget | ≤200 kb curated loci; full-genome stays an upstream-scripts concern, referenced by link from the notebook |
| [CI] New GPU-availability skip category without allowlist discipline | "Skip cleanly when no GPU" | An untyped skip is exactly what `audit_skips.py` exists to fail; a silent GPU skip on the nightly box would hollow the census | Reuse typed `network-unavailable:` prefix for network; if a GPU guard is added, add exactly one new typed prefix + `expected_skips.yaml` entry, fail-closed |
| [REG] A `special/` handler for PlantHelixSeek | "Consistency with other families" | Upstream loads fine through generic Auto* + `trust_remote_code`; an unnecessary handler adds dispatch surface to maintain (and v1 found the one real bug in exactly that chain) | Generic registry entries; add a handler only on demonstrated need, mirroring the first-resolved-wins contract |

## Feature Dependencies

```
[REG] PlantHelixSeek-CRE/-Anno registry entries
    └──requires──> nothing new (existing model_info.yaml + modeling_auto maps)

[SHOW] CRE notebook ──────requires──> [REG] CRE entry
    └──requires──> [SHOW] committed Arabidopsis loci + PlantDHS slices
    └──requires──> [EXEC] notebook execution harness (to be tested at all)
    └──enhances──> [SHOW] agreement assertion (Jaccard floor)

[SHOW] Anno notebook ─────requires──> [REG] Anno entry
    └──requires──> [SHOW] committed loci + TAIR10 GFF3 slices
    └──requires──> [EXEC] execution harness
    └──enhances──> [SHOW] agreement assertion (F1 floor)

[EXEC] execution harness (nbclient, timeouts, tmp-cwd isolation, artifacts)
    └──requires──> [CI] slow marker placement (nightly-only)
    └──requires──> [CI] models.lock entries for every fetched model
    └──requires──> [CI] typed-skip allowlist entries (reuse network-unavailable)

[REPAIR] error repair loop
    └──requires──> [EXEC] harness + artifact capture (the failure signals)
    └──requires──> [CI] WR-08/09 gate repairs (so fixes are actually enforced)

[SHOW] rendered figures in docs mirror ──requires──> [SHOW] both notebooks green
    └──requires──> [CI] docs sync machinery (exists)

[CI] models.lock consistency guard ──enhances──> [CI] models.lock preregistration

[SHOW] peak calling in CRE notebook ──enhances──> [SHOW] Jaccard-on-peaks assertion
```

### Dependency Notes

- **Execution harness → markers/skips:** the `slow` marker and `network-unavailable:` typed-skip prefix already exist; the harness must *use* them (marker on every execution test; typed skip guard at test start) rather than invent new mechanisms — this is what keeps `audit_skips.py` green with zero-to-one new allowlist entries.
- **Harness → models.lock:** every model the 21 notebooks + 2 new notebooks load becomes a lock entry with provenance comment; the nightly cache key rotates automatically via `hashFiles`.
- **Showcase → loci selection:** the ≤200 kb committed regions are a *precondition* of the agreement assertions; loci selection (predict across candidates, keep high-agreement ones) is the long pole and gates both notebooks' final form.
- **Showcase ↔ pyBigWig dependency:** writing a BigWig in-notebook needs `pyBigWig` (wheels exist for linux/mac/win; not currently a dependency). Cleanest: add to the `dev` extra (nightly runner installs it; casual users see a guarded import with install hint). An overlay track *plot* does not require BigWig at all (numpy + matplotlib), so only BigWig *emission* pulls the dependency — the notebook can make it an optional final cell. This is a REQUIREMENTS-level decision to pin.
- **Repair loop ↔ WR-08/09:** fixing example code without the CI gate repair means fixes ride an unenforced lane — do the gate repair early.
- **Conflicts:** none structural; the only tension is nightly runtime budget (21 notebooks + showcase scans inside 180 min) versus running notebooks *unmodified* — resolved by per-family timeout calibration and the owner-accepted runtime cost, not by mutation.

## MVP Definition

### Launch With (v1.1)

- [ ] [CI] WR-08/WR-09 gate repairs — cheap, unblock honest enforcement of everything after
- [ ] [EXEC] nbclient harness: per-notebook parametrized tests, per-cell + per-test timeouts, tmp-cwd isolation, kernel cleanup, failure artifacts — the spine everything else hangs on
- [ ] [CI] Execution tests `slow`-marked into the nightly census; models.lock extended; typed skips reused
- [ ] [EXEC] Real execution: 20–21 notebooks, 3 marimo apps, `generate_bpe_dataset.py`, all YAMLs through `load_config()`
- [ ] [REPAIR] Every surfaced error fixed (examples, mirror, library) with regression tests
- [ ] [REG] PlantHelixSeek-CRE/-Anno registry entries
- [ ] [SHOW] Both showcase notebooks with committed ≤200 kb loci, side-by-side prediction-vs-truth presentation, Jaccard/F1 statistics, and test-asserted agreement floors
- [ ] [CI] Docs mirror sync for new/repaired notebooks

### Add After Validation (v1.x)

- [ ] models.lock consistency guard — once the lock entry set has stabilized post-repair
- [ ] Executed-notebook write-back for the two showcase notebooks (rendered docs figures) — after their content has stopped churning
- [ ] Peak-calling reproduction + narrowPeak emission in CRE notebook — after the score-track version is green
- [ ] Per-cell timing report — diagnostic nicety once nightly runs are routine

### Future Consideration (v2+)

- [ ] TTA toggle and additional species' showcase loci — showcase breadth, not correctness
- [ ] Any notebook-parallelism — only if a second GPU runner appears
- [ ] Output-regression testing (nbval-style) for showcase notebooks — only if output stability across transformers versions proves tractable

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| [CI] WR-08/09 gate repair | HIGH | LOW | P1 |
| [EXEC] nbclient harness (timeouts, isolation, artifacts) | HIGH | MEDIUM | P1 |
| [CI] slow-marker nightly integration + models.lock extension | HIGH | LOW | P1 |
| [EXEC] notebook real execution (21) | HIGH | MEDIUM | P1 |
| [REPAIR] fix-all loop | HIGH | HIGH | P1 |
| [REG] CRE/-Anno registry entries | HIGH | LOW | P1 |
| [SHOW] committed loci + agreement assertions | HIGH | HIGH | P1 |
| [SHOW] CRE notebook (scan + track + Jaccard) | HIGH | MEDIUM | P1 |
| [SHOW] Anno notebook (scan + GFF3 + F1) | HIGH | HIGH | P1 |
| [EXEC] marimo headless execution (3 apps) | MEDIUM | MEDIUM | P1 |
| [EXEC] YAML load_config validation | MEDIUM | LOW | P1 |
| [EXEC] generate_bpe_dataset.py execution | MEDIUM | LOW | P1 |
| [SHOW] rendered figures in docs mirror | MEDIUM | LOW-MEDIUM | P2 |
| [CI] models.lock consistency guard | MEDIUM | LOW-MEDIUM | P2 |
| [SHOW] in-notebook peak calling | MEDIUM | MEDIUM | P2 |
| [EXEC] per-cell timing report | LOW | LOW | P3 |
| [SHOW] TTA toggle | LOW | LOW | P3 |

## Competitor / Prior-Art Feature Analysis

| Feature | Prior art | Our approach |
|---------|-----------|---------------|
| Notebook-as-test packaging | nbmake (pytest plugin over nbclient; per-cell `--nbmake-timeout`, `execution.allow_errors` metadata, `raises-exception`/`skip-execution` cell tags, xdist); pytest-notebook/nbval (output regression); pytest-nb-as-test | nbclient called directly from parametrized pytest tests — same semantics, no new plugin, full control of cwd/artifacts/timeouts per family |
| Failure artifacts | upload-artifact with `if: always()`/`if: failure()`; nbclient try/except/finally partial-notebook save | Same, integrated into the nightly job's existing artifact step |
| Headless marimo | marimo: `python app.py`, `marimo export html/script --include-outputs`, `marimo check`; marimo-integration-ci whitelist-and-export pipeline | pytest-driven headless execution in tmp cwd; export path only if `mo.ui` interactivity blocks plain script runs |
| Prediction-vs-truth track presentation | Enformer usage Colab (side-by-side predicted/observed tracks per locus); ChromBPNet (log-counts Pearson + profile metrics); AlphaGenome (per-track Pearson violins on held-out intervals) | Side-by-side matplotlib track overlay on committed loci + Jaccard/IoU on called peaks vs PlantDHS; no exact-output assertions |
| Peak-set agreement statistic | `bedtools jaccard` (bp-intersection/bp-union; the standard similarity metric); IDR reserved for replicate concordance | Pure-Python interval Jaccard (no bedtools binary dep), asserted ≥ floor in tests |
| Gene-model agreement | gffcompare (sensitivity/precision/F1 at nucleotide/exon/gene levels vs reference GFF) | In-notebook nucleotide/exon-level P/R/F1 vs TAIR10 GFF3 slices + exon/intron block diagrams; gffcompare linked, not required |
| Track/gene figures | pyGenomeTracks (tracks.ini stacking of bigwig + bed/gtf/gff) | matplotlib rendering inside notebooks (keeps deps light); pyGenomeTracks mentioned as the external standard |
| CRE/Anno inference protocol | Upstream `scripts/cis_regulatory` (500/50/50, mean class-1 prob → BigWig; peak call mean±1.5sd) and `scripts/gene_annotation` (8192/4096, both strands, middle-stitch, BILOU→GFF3 viterbi+orf) | Same window/stride/bin parameters reproduced in-notebook via dnallm API on ≤200 kb loci; simplified decode acceptable if GFF3 stays valid |

## Sources

Project-grounded (HIGH confidence, read directly):

- `/home/forrest/Github/DNALLM/tests/examples/test_examples.py` — existing structural example tests (discovery pattern, skip reasons already allowlisted)
- `/home/forrest/Github/DNALLM/tests/expected_skips.yaml`, `/home/forrest/Github/DNALLM/scripts/audit_skips.py` — typed-skip allowlist + fail-closed audit
- `/home/forrest/Github/DNALLM/.github/workflows/ci.yml` — fast leg `-m "not slow"`, nightly full census on `[self-hosted, dnallm-nightly]` (180-min cap, models.lock-keyed model cache, junit + skip audit, log artifact on failure)
- `/home/forrest/Github/DNALLM/models.lock` — artifact registry format with provenance comments
- `/home/forrest/Github/DNALLM/dnallm/models/model_info.yaml` — PlantHelixSeek present only as base/mask entry (line 152)
- `/home/forrest/Github/DNALLM/pyproject.toml` — pytest addopts (`--timeout=300`), markers, pybedtools/pyfastx dev extras

Ecosystem (MEDIUM unless noted):

- [nbclient docs — executing notebooks / client reference](https://nbclient.readthedocs.io/en/latest/client.html) — per-cell `timeout`, `startup_timeout`, `allow_errors`, `CellExecutionError`/`CellTimeoutError`, `resources` metadata `path` (cwd), kernel cleanup ordering (LOW tier per seam, but cross-checked against nbmake's documented mapping of `--nbmake-timeout` to per-cell timeout; treated as MEDIUM jointly)
- [nbmake (computationalmodelling/nbmake, ex treebeardtech)](https://github.com/computationalmodelling/nbmake) — notebook-as-pytest-test semantics, per-cell timeout flag, `execution.allow_errors` metadata, `raises-exception`/`skip-execution` tags, `nbmake.mock`, xdist parallelism
- [pytest-notebook](https://pypi.org/project/pytest-notebook), [nbval ecosystem coverage via IQMO blog](https://blog.iqmo.com/blog/python/jupyter_notebook_testing), [Semaphore nbmake CI tutorial](https://semaphore.io/blog/test-jupyter-notebooks-with-pytest-and-nbmake) — the pytest-plugin landscape and CI patterns
- [GitHub docs — workflow artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data) + upload-artifact issue #328 (via search) — `if: always()`/`if: failure()` artifact-upload guard pattern
- [marimo docs — run as scripts](https://docs.marimo.io/guides/scripts), [CLI export](https://docs.marimo.io/cli), [testing guide](https://docs.marimo.io/guides/testing), [marimo-integration-ci](https://github.com/marimo-team/marimo-integration-ci) — headless marimo execution and CI precedent
- [bedtools jaccard documentation](https://bedtools.readthedocs.io/en/latest/content/tools/jaccard.html), [Quinlan lab tutorial](http://quinlanlab.org/tutorials/bedtools.html), [ENCODE IDR portal](https://www.encodeproject.org/software/idr) — Jaccard as peak-similarity standard; IDR as replicate-only tool
- [GffCompare (CCB JHU)](https://ccb.jhu.edu/software/stringtie/gffcompare.shtml), [GFF Utilities paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7222033) — sensitivity/precision/F1 at nucleotide/exon/gene levels for gene-prediction evaluation
- [pyGenomeTracks docs](https://pygenometracks.readthedocs.io), [Bioinformatics paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8058774) — stacked track+gene figure standard
- [ChromBPNet preprint](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v4.full), [Enformer usage Colab](https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer-usage.ipynb), [AlphaGenome paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12851941) — per-track correlation + side-by-side track presentation conventions
- Upstream PlantHelixSeek (HIGH for pipeline parameters, fetched from the repo's own READMEs): [zhangtaolab/PlantHelixSeek](https://github.com/zhangtaolab/PlantHelixSeek) (`scripts/cis_regulatory`, `scripts/gene_annotation`), HF model cards [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek), [PlantHelixSeek-CRE](https://huggingface.co/zhangtaolab/PlantHelixSeek-CRE), [PlantHelixSeek-Anno](https://huggingface.co/zhangtaolab/PlantHelixSeek-Anno)

---
*Feature research for: DNALLM v1.1 — example execution testing + PlantHelixSeek showcase examples*
*Researched: 2026-10-01*
