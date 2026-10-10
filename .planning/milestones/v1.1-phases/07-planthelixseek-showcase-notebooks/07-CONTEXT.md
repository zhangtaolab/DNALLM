# Phase 07: PlantHelixSeek Showcase Notebooks - Context

**Gathered:** 2026-10-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Two flagship showcase notebooks (CRE + Anno) run REAL sliding-window inference on the Phase-6 committed Arabidopsis loci through the dnallm public API, present prediction-vs-truth honestly under illustrative-loci framing, compute the selection-time metrics in-notebook, and are asserted against the Phase-6 frozen floors by example execution tests. Executed notebooks (with embedded figures) are written back into the docs mirror following the established wrapper-.md pattern. Delivers SHOW-03, SHOW-04, SHOW-05, SHOW-06, SHOW-07.

</domain>

<decisions>
## Implementation Decisions

### Notebook narrative & honesty
- **D-01:** Each notebook opens with ONE consolidated provenance markdown cell: selected loci coordinates, one-line selection methodology + link to `example/notebooks/plant_helixseek_shared/data/selection.md`, the floors/bands table, environment versions (transformers/torch/fla), and the illustrative-loci disclaimer.
- **D-02:** Negative controls appear IN-notebook: reference the selection.md recorded values (flanking 0.0325 / intergenic 0.0000) plus one lightweight recompute cell — evidence the model is quiet where it should be.
- **D-03:** Disclaimer coverage is dual: full statement in the opening provenance cell AND a short caption line on every metric figure / conclusion cell ("illustrative locus, not genome-wide accuracy").
- **D-04:** Narrative is tutorial-style: walk the dnallm API flow (load → sliding scan → decode → metrics → visualization) with copyable steps — matches the docs tutorial positioning.

### Assertion ownership (SHOW-05)
- **D-05:** Two-layer design: notebooks compute and PRINT the metrics + a floors/bands comparison table (transparent, NO in-notebook asserts); the authoritative assertions live in `tests/examples/` execution tests that parse output-cell metrics and assert the tolerance bands. — **Reversibility:** costly — moving assertions between layers touches both the notebooks and the test suite.
- **D-06:** Floors/bands are PARSED from the committed `selection.md` at test startup (single source of truth; frozen keys `jaccard=`, `genes_above_floor=`, `neg_cre_fraction=`, `neg_anno_fraction=`, band table), plus a parse-guard test that fails red when any expected key is missing.
- **D-07:** Drift semantics: bands verbatim from selection.md (jaccard ∈ [0.3, 1.00], flanking ≤ 0.05, intergenic genic ≤ 0.10, ≥3 genes) — "substantially consistent" per the Phase-6 frozen contract; observed-headroom-is-the-margin.
- **D-08:** Assertion failure messages are named-cause: which metric, observed value, expected band, plus a pointer "re-run selection (Phase-6 methodology) if environment drift is suspected".

### Docs-mirror write-back (SHOW-06)
- **D-09:** Write-back timing: local execution then manual commit — the executed notebook (with figures) is committed to `example/`, the docs mirror follows byte-identically (`scripts/check_docs_sync.py`); nightly only VERIFIES reproducibility, produces no commits.
- **D-10:** Figures are altair embedded vega JSON inside the ipynb (KB-scale, renders natively in GitHub blob view, no binaries in repo).
- **D-11:** Docs presentation follows the ESTABLISHED wrapper-.md pattern (owner-corrected premise, verified live 2026-10-03: mkdocs.yml nav points exclusively to wrapper .md files; mkdocs-jupyter is configured but renders no pages): one wrapper tutorial .md per notebook (tutorial-style excerpts, env prerequisites, static figure descriptions), executed ipynb byte-synced into the mirror, "View Full Notebook" button to GitHub where figures render natively. SHOW-06 "rendered figures written back into the docs mirror" is satisfied by the executed nb entering the mirror byte-identically — zero new rendering mechanism. — **Reversibility:** reversible — switching to mkdocs-jupyter page rendering later is an additive docs change.
- **D-12:** Size budget: executed ipynb ≤ 2MB per notebook (downsample plot data: track ~1 point/bin ≈ 4000 points; gene models per-locus gene count). Exceeding the budget is a plan violation.

### Execution lanes
- **D-13:** The two showcase notebooks execute ONLY on the nightly lane (`@pytest.mark.slow` + the Phase-5 nbclient harness: tmp-sandbox cwd isolation, per-cell timeout inside per-test timeout, kernel-kill discipline). Fast lane gets structure tests only (collection, provenance-cell format presence, no execution).
- **D-14:** Per-test timeout budgets: CRE notebook 40 minutes, Anno notebook 90 minutes (Phase-6 measured single-model verify ~10-20 / ~30-60 min; 2x headroom).
- **D-15:** GPU batch sizes frozen into the notebooks per Phase-6 measured ceilings, provenance noted: CRE bs=4 (eager-attention ceiling through the dnallm route; bs=64 OOMs), Anno bs=1.
- **D-16:** First code cell is a HARD environment guard: raise if `fla` is not importable (missing fla = silently-dead outputs — the Phase-6 lesson; never degrade silently), and print transformers/torch/fla versions as key=value evidence lines (provenance + greppable by execution tests).

### Claude's Discretion
- Exact cell ordering within the tutorial narrative beyond the locked first-cell constraints (D-01, D-16).
- Plot styling (altair themes/colors) within the size budget.
- Structure-test granularity on the fast lane (beyond presence checks).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Frozen selection contract (single source of truth for floors/bands/decode/match)
- `example/notebooks/plant_helixseek_shared/data/selection.md` — Phase-6 frozen methodology: verification contracts (CRE bin/peak rules, Anno stitching/decode/match), observed values, thresholds and tolerance bands, provenance, budget. Phase-7 assertions parse this file.

### Phase-6 context and research
- `.planning/phases/06-model-registry-showcase-data-curation/06-CONTEXT.md` — owner decisions incl. selection methodology, argmax decode, negative-control design, constraint addenda
- `.planning/phases/06-model-registry-showcase-data-curation/06-RESEARCH.md` — upstream contract sources, memory/OOM bounds, timings
- `.planning/phases/06-model-registry-showcase-data-curation/06-VERIFICATION.md` — what was proven about the committed artifacts

### Execution harness (Phase-5)
- `tests/examples/_execution.py` — nbclient harness: tmp-sandbox cwd isolation, per-cell timeout, kernel-kill, assert_tree_clean (scoped git status guard)
- `tests/examples/test_notebook_execution.py` — execution-test shape; typed-skip prefixes; the notebook_sandbox fixture lives HERE (never add a conftest.py under tests/examples/)

### Docs mirror mechanism
- `scripts/check_docs_sync.py` — byte-identical mirror contract between example/ and docs/example/
- `docs/example/notebooks/finetune_binary.md` — the wrapper-.md pattern to follow (frontmatter `notebook:` + `sync_check: true`, tutorial body, GitHub full-notebook button)
- `mkdocs.yml` § plugins + nav — nav points to wrapper .md files only (mkdocs-jupyter configured but renders no pages — verified live 2026-10-03)

### Registry / models / environment
- `dnallm/models/model_info.yaml` — the two PlantHelixSeek finetuned entries (repo ids, frozen label order, provenance comments)
- `pyproject.toml` § optional-dependencies — `fla` extra (`flash-linear-attention>=0.5.2,<0.6`), wired into `all`
- `README.md` § Flash-Linear-Attention (KDA) Kernels — the silent-fallback warning and bare-install contract
- `tests/expected_skips.yaml` — typed-skip whitelist (`environment-unavailable:` prefix)
- `.github/workflows/ci.yml` — nightly legs install `.[base,fla]` (WR-01 fix)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm.utils.genomic_coords` (six helpers, re-exported at `dnallm.utils`): all coordinate/chrom conversions, CRLF-robust as of quick-261003-r73
- Committed showcase data: `.fas` fragments + truth slices + negative controls under `example/notebooks/plant_helixseek_{cre,anno,shared}/data/` (≤200kb/set, no re-download)
- `load_model_and_tokenizer` generic route with registry ids (no special handler)
- Phase-5 nbclient harness (see canonical refs) — the two notebooks join `test_notebook_execution.py`'s slow lane

### Established Patterns
- Wrapper-.md + byte-sync mirror for docs (D-11)
- Typed `environment-unavailable:` skip contract with expected_skips.yaml whitelist
- key=value evidence prints (T20-compliant version logging)
- Floors parsed from committed artifacts, not copied constants (D-06)

### Integration Points
- `tests/examples/test_notebook_execution.py` gains the two slow execution tests (floors parsed from selection.md)
- `example/` + `docs/example/` mirror pair gains the two notebook dirs + wrapper pages; `check_docs_sync.py` stays green
- nav in `mkdocs.yml` gains two wrapper entries

</code_context>

<specifics>
## Specific Ideas

- Owner's live-site reference for the wrapper pattern: https://zhangtaolab.org/DNALLM/example/notebooks/finetune_binary/ (the presentation form the showcase pages should match)
- The fla silent-fallback incident (Phase-6 checkpoint, B+ decision) is the motivating story for D-16 — the notebooks must fail loudly where the models would silently degrade

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-PlantHelixSeek Showcase Notebooks*
*Context gathered: 2026-10-03*
