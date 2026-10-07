---
phase: 261004-dyw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - mkdocs.yml
  - tests/examples/_execution.py
  - tests/examples/test_plant_helixseek_showcase.py
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
  - example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph
  - example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
  - docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_anno.md
  - docs/example/notebooks/plant_helixseek_combined.md
autonomous: true
requirements:
  - SHOW-07
  - D-12
  - D-13

estimate:
  tokens: 55000
  raw_tokens: 55000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - The EXISTING altair full-locus figures (CRE cell 13, Anno cell 19) keep their vega v6 object + vegalite v6 JSON string and GAIN an image/png mime (vl_convert) — at least one vega and at least two image/png display outputs per committed main notebook (full-locus + zoom).
    - Main notebooks are SYMMETRIC and SIMPLIFIED (no sibling-model runs, no combined sections): CRE's new zoom figure is a pgt 2-track view (p(CRE) bedGraph filtered to >= 0.5 sliced from the ALREADY-COMPUTED full-locus bin_scores — zero extra inference — plus the leaf-DNase track fed DIRECTLY from the committed bedGraph); Anno's new zoom figure is a pgt 4-track view (predicted transcripts +/− #1f77b4 decoded from the EXISTING cell-11 label_tracks, TAIR10 truth + #2ca02c / − #d62728 fed DIRECTLY from the owner-pre-converted committed truth GTF), window slice by overlap-with-clipping.
    - A NEW third notebook example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb is the ONLY place both modalities meet: standard provenance cell (data links, combined_locus=Chr1:5220001-5265000, env versions, illustrative disclaimer) + D-16 fla hard guard (find_spec + RuntimeError + version prints, same shape) + loads BOTH models (models.lock, warm cache) + scans ONLY the display region Chr1:5220001-5265000 (owner window + 5 kb downstream flank so edge-crossing genes complete, e.g. AT1G15290.1 ends 5264942) — CRE 500/50/50 on the 45 kb (~1 min) and Anno 8192/4096 both strands on the 45 kb (~1 min; plan_windows semantics on the slice, BOS offset, _B_SWAP_L permutation) + display decode to GTF (non-O islands → gene/transcript/exon/CDS/UTR features, INTRON runs as spacers, transcripts < 100 bp dropped and disclosed) + the OWNER-PRE-CONVERTED committed truth GTF (data/TAIR10_GTF_chr1_5100001_5300000.gtf) fed DIRECTLY to pgt — NO runtime GFF3→GTF conversion anywhere; the original GFF3 stays committed for the Anno metrics cell — + the p(CRE) bedGraph (>= 0.5-filtered, min_value 0.5 / max_value 1 / #1f77b4) with the leaf-DNase track fed DIRECTLY from the committed bedGraph (NO binned DHS track) + the six-track TITLED pgt combined figure (approved v5 render .scratch/zoom-candidates/combined-pgt-v5-titled.png: titles on, pred blue, TAIR10 truth green + / red −, p(CRE) floor 0.5) with zoom_window=Chr1:5220001-5260000 and combined_window=Chr1:5220001-5265000 key=value prints and a caption disclosing the +5 kb display flank.
    - The leaf-DNase artifact is a proper UCSC bedGraph (NOT a tsv — .gitignore:86 `*.tsv` would silently exclude it from the atomic commit): example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph (111,791 bytes, on disk, uncommitted) — # provenance header block (source URL, extraction date 2026-10-04, 0-based 50 bp bins aligned to the CRE grid) plus Chr1 start end signal rows; pgt renders it WITH the # header (verified live, 0 issues), so NO sidecar file and NO pandas conversion anywhere: notebooks pass the file directly as the pgt bed_graph track input (pgt crops to the rendered region). A second OWNER-DIRECTED committed artifact, example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf (127,310 bytes, 1,425 features, # provenance header documenting source + conversion rule + date), replaces ALL runtime GFF3→GTF conversion — the combined notebook and the Anno zoom read it DIRECTLY for pgt. git check-ignore verified NEGATIVE for both (.gtf and .bedGraph escape every ignore pattern). NO pyBigWig use in notebooks, NO runtime network, NO dnallm code change; provenance cells cite BOTH artifacts' extraction/conversion provenance (internal preparation steps, not re-done in-notebook).
    - Every new figure (all three notebooks) uses the owner-selected literals with ZERO selection/scoring code: no "jaccard" and no "gene_f1" strings in any new figure cell (both banned in the combined notebook entirely), no best/optimal claims anywhere.
    - pygenometracks>=3.9 is declared in the pyproject.toml `notebook` extra (GPL-3.0 — owner explicitly overrides the v1-milestone exclusion on 2026-10-04; dnallm package code never imports it; example-notebook use only; pyBigWig rides along; fresh-env/CI install needs the 05-FEASIBILITY CFLAGS deviation — recorded for the Phase-8 example job). pgt 3.9 pins matplotlib down (3.11.2 → 3.8.4, live-observed); the downgrade gate is PRE-SATISFIED by the halted executor's run (full fast suite: 1741 passed / 1 skipped / exit 0, log /tmp/261004-dyw-downgrade-gate.log) — the executor cites it and cheaply re-confirms the examples lane; the full fast suite re-runs WITH the new assertions at close.
    - Lane wiring: NOTEBOOK_EXEC_SPECS gains the combined entry (cell_timeout 1200, strictly below its 2400s slow mark); the parametrized structure tests (provenance / fla guard / no-fla-import / SHOW-07 denylist / caption-after-figure) cover all THREE notebooks; the combined notebook gets its own committed-blob structure test (image/png present, <= 2 MB, stream outputs) and its own slow sandboxed execution test asserting the combined figure + window evidence lines; the combined notebook NEVER joins ACTIVE_NOTEBOOKS/GATED_NOTEBOOKS; the CRE slow test seeds ONLY the bedGraph extra and the Anno slow test seeds the bedGraph + truth-GTF extras (its zoom reads both shared artifacts).
    - Re-execution on the GB10 reproduces the frozen metrics exactly: jaccard=0.3247 and neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000 (Anno).
    - All three committed notebooks stay at most 2,097,152 bytes (D-12); docs mirrors byte-identical (three notebooks + the bedGraph); md-sync/snippets/docs-sync stay exactly as green as baseline; the combined wrapper page (finetune_binary pattern) joins the mkdocs Showcase nav.
    - All seventeen files (pyproject, mkdocs, _execution.py, showcase test module, three notebooks, bedGraph + truth GTF, three notebook mirrors, bedGraph + GTF mirrors, three wrappers) land in ONE atomic commit with no attribution trailers — the bedGraph and GTF are committable precisely because their extensions escape the *.tsv ignore (never use git add -f).
  artifacts:
    - pyproject.toml — `notebook` extra gains "pygenometracks>=3.9".
    - mkdocs.yml — Showcase nav gains the combined wrapper entry.
    - tests/examples/_execution.py — NOTEBOOK_EXEC_SPECS combined entry (cell_timeout 1200).
    - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb — 23 cells (20 + 3): figure cell 13 gains image/png; new pgt zoom section after caption cell 14; provenance line for the bedGraph.
    - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb — 29 cells (26 + 3): figure cell 19 gains image/png; new pgt zoom section after caption cell 20; provenance line for the bedGraph.
    - example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb — NEW: provenance markdown, D-16 guard, both-model loads, 45 kb display-region scans, decode + GTF/bedGraph writers, six-track pgt figure, SHOW-07-compliant captions; PNG-only figure embeds.
    - example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph — committed (orchestrator-created UCSC bedGraph with # provenance header, 111,791 bytes); mirrored byte-identically under docs/.
    - example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf — committed (owner-directed pre-converted truth GTF, 127,310 bytes / 1,425 features, # provenance header); mirrored byte-identically under docs/.
    - tests/examples/test_plant_helixseek_showcase.py — SHOWCASE_NOTEBOOKS gains the combined param; committed-outputs test rekeyed (vega for the two altair notebooks, image/png for all); new combined structure + slow execution tests; main-test extras — bedGraph only (CRE) and bedGraph + truth GTF (Anno).
    - docs/example/notebooks/plant_helixseek_{cre,anno}/plant_helixseek_{cre,anno}.ipynb, docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb, and docs/example/notebooks/plant_helixseek_shared/data/{Ath_leaf_DNase_chr1_5100001_5300000.bedGraph, TAIR10_GTF_chr1_5100001_5300000.gtf} — byte-identical mirrors.
    - docs/example/notebooks/plant_helixseek_cre.md / _anno.md — prose passage updates; docs/example/notebooks/plant_helixseek_combined.md — NEW wrapper (finetune_binary pattern, front-matter notebook + sync_check).
  key_links:
    - Owner selection (SELECTION.txt, already written) → plain literals 5220000-5260000 (zoom) and 5220000-5265000 (combined display region) → zoom_window=/combined_window= evidence lines printing exactly Chr1:5220001-5260000 / Chr1:5220001-5265000 — the only window data the committed notebooks derive.
    - pyproject notebook extra → .venv pgt 3.9 + pyBigWig 0.3.26 (live-verified) → notebook cells invoke the CLI as Path(sys.executable).with_name("pgt") (PATH-independent; console scripts are pgt/pyGenomeTracks — there is NO `pygenometracks` script) with check=True → PNG bytes → base64 image/png display.
    - Committed bedGraph (example + docs mirrors) → fed DIRECTLY as the pgt bed_graph track file in the CRE zoom and the combined figure (pgt crops to the rendered region; no parsing/conversion cell); its # provenance header rides in the file itself.
    - Combined notebook data flow: ../plant_helixseek_cre/data/chr1_5100001_5300000.fas (fetch_sequence slice for the 45 kb region; the ONLY cross-notebook read) + data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph and data/TAIR10_GTF_chr1_5100001_5300000.gtf (its own dir, fed directly to pgt) → predicted GTF + p(CRE) bedGraph + tracks ini under gitignored outputs/ → pgt render → embedded PNG.
    - seed_sandbox whole-dir copy semantics (verified): the combined notebook's own plant_helixseek_shared/data/ (selection.md + bedGraph + truth GTF) rides along; ONLY the single cross-notebook CRE FASTA needs a tuple extra in its slow test.
    - matplotlib downgrade (3.8.4) → pre-satisfied gate (1741P/1S/exit 0, /tmp/261004-dyw-downgrade-gate.log) → cheap examples-lane re-confirm in Task 1 → full fast suite WITH the new assertions at Task 3 close.
    - Committed blobs → structure tests + the one-liner gates (vega/image/png counts, window evidence lines exact, bands, size, guard, no-scoring greps).
---

<objective>
Enhance the PlantHelixSeek showcase display surface (owner request; final architecture after five
revision rounds — owner-selected window, pygenometracks renderer, per-notebook simplification, a
dedicated combined notebook, and a bedGraph-format data artifact):

1. The existing altair full-locus figures (CRE cell 13, Anno cell 19) gain an image/png mime
   (vl_convert) next to vega/vegalite — GitHub renders the vega object but local JupyterLab/VS
   Code do not; PNG renders everywhere.
2. Each main notebook gains ONLY its own pgt zoom figure over the owner-selected window
   Chr1:5220001-5260000 (CRE: confident-band p(CRE) + leaf DNase, zero extra inference; Anno:
   predicted vs truth gene tracks from already-computed labels, the truth side fed directly from
   the committed GTF). NO sibling-model runs, NO combined sections in the main notebooks.
3. A NEW combined notebook (example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb)
   is the single place both modalities meet: it loads both models, scans ONLY the 45 kb display
   region (window + 5 kb downstream flank), and renders the approved six-track titled pgt figure
   (CRE confident band, official leaf DNase, predicted transcripts, TAIR10 gene models on one
   genomic axis). Wired into the showcase lane: exec spec, parametrized structure tests, its own
   slow execution test, wrapper page + mkdocs nav + byte-identical mirror.
4. The leaf-DNase evidence artifact is a UCSC bedGraph (the original .tsv hit the `*.tsv`
   gitignore — .gitignore:86 — and would have been silently dropped from the atomic commit);
   the orchestrator regenerates it as .bedGraph before dispatch, and every notebook feeds it to
   pgt DIRECTLY (verified: pgt's bedGraph parser tolerates the # provenance header).
5. pygenometracks joins the notebook extra (GPL-3.0 owner override); the matplotlib-pin risk is
   pre-proven harmless (full fast suite 1741P/1S on the downgraded env, log preserved) and
   re-gated with the new assertions at close.
6. All three notebooks are executed on the GB10; the main notebooks must reproduce the frozen
   metrics exactly (jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59,
   neg_anno_fraction=0.0000).
7. Notebooks + bedGraph byte-synced to docs mirrors, wrappers updated/created (prose-only edits;
   new combined wrapper), 2 MB budget held, fast structure lane + nightly extras + docs-sync
   gates green, everything in one atomic commit (seventeen files) with no attribution trailers.

Purpose: the showcase notebooks are the public evidence surface for the PlantHelixSeek work
(Phases 6-7); today their figures are invisible to anyone opening them locally, and no single
view aligns CRE signal, open-chromatin truth, and gene structure on one axis.
Output: re-executed CRE/Anno notebooks with dual-render full-locus figures and pgt zoom views, a
new executed combined notebook, a declared notebook-extra dependency, a properly-committable
bedGraph artifact, lane wiring, synced mirrors/wrappers/nav, extended structure tests.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Verified-live facts this plan encodes (do not re-derive):
- SELECTION COMPLETE: /tmp/planthelixseek_zoom_candidates/SELECTION.txt contains cre_zoom=,
  anno_zoom=, combined_zoom= all Chr1:5220001-5260000. Candidate phase done; no human gates remain.
- EXECUTOR STATE: the previously dispatched executor was halted BEFORE any commits — treat the
  working tree as pre-Task-1. Its downgrade-gate result IS reusable: full fast suite on the
  matplotlib-3.8.4 + pgt-3.9 environment = 1741 passed / 1 skipped / exit 0, log at
  /tmp/261004-dyw-downgrade-gate.log.
- COMMITTED ARTIFACTS (both ON DISK, uncommitted; the original .tsv is DELETED — it was
  GITIGNORED by .gitignore:86 `*.tsv` and would have been silently dropped by the atomic commit):
  * example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph
    (111,791 bytes) — UCSC bedGraph: # provenance header (source
    https://plantdhs.org/static/download/Ath_leaf_DNase.bw, extracted 2026-10-04, 0-based 50 bp
    bins aligned to the CRE scan grid) + Chr1 start end signal rows over the 200 kb locus.
    VERIFIED live: git check-ignore NEGATIVE, and pgt renders it WITH the # header, 0 issues.
  * example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf
    (127,310 bytes, 1,425 features) — owner-directed pre-converted truth GTF (deterministic
    GFF3→GTF; # provenance header documents source + rule + date); VERIFIED git check-ignore
    NEGATIVE. Notebooks read it DIRECTLY for pgt — NO runtime GFF3→GTF conversion anywhere; the
    original GFF3 stays committed (the Anno metrics cell still reads its own-dir GFF3 slice).
  Never use git add -f.
- ENVIRONMENT (live-verified this session): pygenometracks 3.9 + pyBigWig 0.3.26 installed in
  .venv; matplotlib DOWNGRADED to 3.8.4 by the pgt pin (was 3.11.2). Console scripts present:
  pgt, pyGenomeTracks, make_tracks_file — there is NO `pygenometracks` script, and the package
  has no __main__ (python -m fails). All five pgt track types render from our real data with
  zero empty-track warnings after the GTF conversion (owner-validated). Fresh-env installs need
  the 05-FEASIBILITY CFLAGS deviation for pyBigWig
  (.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md:
  CFLAGS=-I/home/forrest/miniconda3/include LDFLAGS=-L/home/forrest/miniconda3/lib pip install
  pyBigWig --no-build-isolation) — the same line must be added to the Phase-8 example-job install
  notes (cross-phase bookkeeping, record in SUMMARY).
- Owner-approved reference render for the combined figure (consult it; transient, gitignored):
  .scratch/zoom-candidates/combined-pgt-v5-titled.png — six TITLED data tracks on one genomic
  axis: CRE p(CRE) confident band (floor 0.5), PlantDHS leaf DNase, predicted transcripts (+)/(−)
  in blue, TAIR10 truth (+) green / (−) red, x-axis; titles on (never --trackLabelFraction 0).
- CRE notebook: 20 cells, altair figure = code cell 13, caption = markdown cell 14; presentation
  state: bin_scores / bin_width / genomic_offset (the zoom slices these — ZERO extra inference).
  Anno notebook: 26 cells, altair figure = code cell 19, caption = markdown cell 20; presentation
  state: label_tracks = {"+": plus_labels, "-": minus_labels} (cell 11, full-locus per-base argmax
  labels — the zoom decode reuses these, never rescans), plan_windows / predict_window_labels /
  _B_SWAP_L machinery (cells 7/9/11) transcribed onto the 45 kb slice in the combined notebook.
- Per-notebook data dirs: plant_helixseek_cre/data/{chr1_5100001_5300000.fas,
  TAIR10_DHSs_chr1_5100001_5300000.gff}; plant_helixseek_anno/data/{chr1_5100001_5300000.fas,
  TAIR10_GFF3_chr1_5100001_5300000.gff3}; plant_helixseek_shared/data/{selection.md, negative
  controls, the new bedGraph + truth GTF}. The combined notebook's ONLY cross-notebook read is
  the CRE FASTA via ../plant_helixseek_cre/data/chr1_5100001_5300000.fas (seeded as one tuple
  extra in its slow test); the bedGraph and GTF live in its own data/ dir. The Anno zoom reads
  the shared GTF via ../plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf.
- seed_sandbox semantics (verified, tests/examples/_execution.py:259): WHOLE-DIR copy of the
  notebook's parent (ignoring .ipynb_checkpoints/__pycache__/outputs*/results*) plus optional
  (src, dest-relative-to-sandbox) tuple extras — so the combined notebook's own shared data dir
  (selection.md + bedGraph + truth GTF) rides along automatically; its slow test needs only the
  single CRE-FASTA tuple extra. The CRE slow test needs only the bedGraph extra; the Anno slow
  test needs the bedGraph + truth-GTF extras (its zoom reads both shared artifacts).
- pgt GTF requirement (live-verified): readGtf needs gene_id/transcript_id attributes — feeding
  raw GFF3 ID/Parent attributes yields "No transcript found" + an empty track; the deterministic
  GFF3→GTF conversion keeps gene/mRNA/exon/CDS/UTR and maps first-Parent → transcript_id.
- Display region: Chr1:5220001-5265000 = the 40 kb owner window + a 5 kb downstream flank
  (0-based half-open 5220000-5265000); edge-crossing genes complete inside the flank (owner
  example: AT1G15290.1 ends at 5264942).
- .venv/bin/jupyter execute --inplace --timeout N exists; nbclient sets the kernel cwd to the
  notebook's own directory so relative data paths and outputs/ resolve (proven 07-01).
  NOTEBOOK_EXEC_SPECS budgets: CRE 1200, Anno 3600; the combined entry (this plan) is 1200.
- Wrappers docs/example/notebooks/plant_helixseek_{cre,anno}.md do NOT quote the figure cells —
  editing figure cells and inserting cells cannot break the AST sync check. Wrapper pattern for
  the new page: docs/example/notebooks/finetune_binary.md (front-matter notebook: + sync_check:
  true, title, Full Notebook button, Prerequisites, section prose). mkdocs.yml Showcase nav
  block sits at lines 212-214 — add the combined entry there.
- Baselines: check_notebook_md_sync fails on exactly 3 pre-existing stale wrappers
  (mcp_langchain, mcp_pydantic_ai, data_prepare_finetune — none plant_helixseek);
  validate_docs_snippets green; check_docs_sync "OK". outputs/ is gitignored (.gitignore:63);
  .scratch/ is ignored (verified via check-ignore).
- Owner rules: zoom selection stays internal (done); pygenometracks adoption is an explicit
  GPL-3.0 override recorded 2026-10-04 (dnallm code never imports it); atomic commit (all
  seventeen files together), no attribution trailers, execution on the local GB10.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Dependency adoption (gate pre-satisfied); surgery on both main notebooks; author the combined notebook; test/lane wiring</name>
  <precondition>Both orchestrator-created artifacts exist and are NOT gitignored (git check-ignore exits 1 for each): example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph (# provenance header + Chr1 start/end/signal rows, ~112 KB) and example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf (# provenance header + GTF features, ~127 KB). If either is missing or only the old .tsv is present, HALT and report — artifact preparation is an owner-side step; never git add -f, never regenerate in-notebook.</precondition>
  <files>pyproject.toml, tests/examples/_execution.py, tests/examples/test_plant_helixseek_showcase.py, example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb</files>
  <behavior>
    - Structure test: both MAIN notebooks expose at least two image/png mimes and at least one vega mime across display outputs; the COMBINED notebook exposes at least one image/png mime, stays <= 2 MB, and carries stream outputs.
    - Structure test: EVERY figure cell (keyed on a display dict naming "image/png") across all THREE notebooks is followed by a markdown cell carrying the pinned disclaimer "illustrative locus, not genome-wide accuracy".
    - Parametrized provenance / fla-guard / no-fla-import / SHOW-07 denylist tests cover all THREE notebooks (SHOWCASE_NOTEBOOKS gains the combined param with locus key combined_locus=Chr1:5220001-5265000).
    - Nightly lane: NOTEBOOK_EXEC_SPECS combined entry (1200 < the 2400s mark); combined slow test asserts the figure + window evidence lines; the CRE slow test seeds the bedGraph extra and the Anno slow test seeds the bedGraph + truth-GTF extras.
    - These go RED against the edited-but-not-yet-executed blobs — expected transient; they turn GREEN only after Task 2's execution. Do not weaken them to pass early.
  </behavior>
  <action>
    STEP 1 — DEPENDENCY ADOPTION (gate pre-satisfied):
    a. Edit pyproject.toml: the `notebook` extra (jupyter>=1.1.1, marimo>=0.16.3, nbclient>=0.10)
       gains "pygenometracks>=3.9". GPL-3.0 owner override 2026-10-04 — dnallm package code never
       imports it; example-notebook use only. Do not touch any other extra.
    b. DOWNGRADE GATE — PRE-SATISFIED: the halted executor ran the full fast suite on this exact
       downgraded environment (matplotlib 3.8.4 + pgt 3.9): 1741 passed / 1 skipped / exit 0,
       log /tmp/261004-dyw-downgrade-gate.log. Cite that log (path + counts) in the SUMMARY.
       Cheap re-confirmation: run the examples fast lane (task-end verify does this). Do NOT
       re-run the full tree here; the full suite re-runs WITH the new assertions in Task 3.
    c. Assert the precondition: the bedGraph exists, starts with # provenance comments, and
       git check-ignore exits 1 for it (the .tsv trap must not recur).

    STEP 2 — MAIN-NOTEBOOK SURGERY. MECHANICS: the Edit tool refuses .ipynb — do all edits in a
    throwaway .venv/bin/python session using nbformat (read as_version=4, exact single-match
    anchors, v4 cell constructors, write). Do NOT execute anything in this task. Window literals:
    zoom_start=5220000, zoom_end=5260000 (display Chr1:5220001-5260000). Each new figure cell
    carries a brief comment naming the window as an owner pick and pygenometracks as the renderer.

    EDIT 1 (both main notebooks, altair figure cell — CRE 13, Anno 19): add import base64; add
    one display-dict entry after the vegalite entry and before text/plain: "image/png" mapped to
    base64.b64encode(vlc.vegalite_to_png(vegalite_spec)).decode("ascii"). Extend the embed
    comment (local JupyterLab/VS Code do not render the vega v6 object + bare .json string).
    These remain the ONLY vega-mime figures. Everything else in these cells stays as-is.

    EDIT 2 (CRE — pgt zoom section, 3 cells after caption cell 14): markdown header "##
    Illustrative zoom window" (display clarity; full-locus metrics above remain authoritative);
    code cell PRESENTATION ONLY with ZERO extra inference — slice the ALREADY-COMPUTED bin_scores
    to the window and write under gitignored outputs/ a bedGraph FILTERED AT WRITE TIME to
    score >= 0.5 (sub-0.5 rows never written; print the descriptive pcres_bins_shown=<n>); a pgt
    tracks ini with TITLED tracks — CRE p(CRE) confident band (bed_graph, min_value 0.5,
    max_value 1, color #1f77b4, file = the written bedGraph) and PlantDHS leaf DNase (bed_graph,
    color #8c2d04, file = ../plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph
    passed DIRECTLY — pgt crops to the region; no parsing, no conversion) — plus the x-axis
    section; invoke the CLI as str(Path(sys.executable).with_name("pgt")) (PATH-independent)
    with subprocess check=True over region Chr1:5220001-5260000; embed the PNG via
    display(..., raw=True) with image/png (+ a text/plain line) ONLY — no vega mimes. Print
    zoom_window=Chr1:5220001-5260000 and zoom_width=40000. NEVER pass --trackLabelFraction 0.
    "jaccard" must not appear in this cell. Caption markdown: pinned disclaimer +
    display-clarity wording + the filter disclosure ("only p(CRE) >= 0.5 bins shown — confident
    band; the full-locus overview above shows the unfiltered track") + rendered-with-pygenometracks
    note; SHOW-07-compliant; no best claims.

    EDIT 3 (Anno — pgt zoom section, 3 cells after caption cell 20): same literal-window pattern,
    4-track gene view — predicted transcripts (+) and (−) GTF (#1f77b4) decoded from the EXISTING
    cell-11 label_tracks (never a rescan) via the island decode (below), and the TAIR10 truth
    (+) (#2ca02c) / truth (−) (#d62728) tracks fed DIRECTLY from the committed
    ../plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf (pgt crops to the region —
    no GFF3→GTF conversion at runtime); x-axis; titles on; PNG-only embed; print
    zoom_window=Chr1:5220001-5260000 (plus zoom_width=40000 and descriptive zoom_transcripts=<n>
    if useful). "gene_f1" and "EXON_F1_FLOOR" must not appear. Caption: pinned disclaimer +
    display-clarity + the < 100 bp transcript-drop disclosure; SHOW-07-compliant.

    SHARED DECODE HELPER (transcribed presentation-only, used by the Anno zoom and the combined
    notebook; no metric computation):
    - Island decode: scan a per-base label array for maximal islands of non-O labels; per island
      emit GTF gene + transcript spanning it and exon/CDS/five_prime_utr/three_prime_utr
      sub-features from the label runs (INTRON runs are inter-exon spacers — NOT emitted); drop
      transcripts with total span < 100 bp (captions disclose); window-slice with OVERLAP
      semantics — edge-crossing transcripts included and clipped, never inside-only;
      locus-local → genomic via the offset; 1-based closed GTF lines with gene_id/transcript_id.
      (The truth side needs NO decode — the owner-pre-converted committed GTF is used as-is.)

    STEP 3 — AUTHOR THE COMBINED NOTEBOOK (example/notebooks/plant_helixseek_shared/
    plant_helixseek_combined.ipynb, NEW). Cell plan (narrative markdown headers between code
    cells, tutorial voice per the D-04 pattern; every code line <= 100 chars):
    1. Provenance markdown cell 0: same consolidated shape as the main notebooks — link to
       plant_helixseek_shared/data/selection.md, the line combined_locus=Chr1:5220001-5265000
       (owner display window Chr1:5220001-5260000 plus a 5 kb downstream flank so edge-crossing
       genes complete, e.g. AT1G15290.1 ends 5264942), the illustrative-loci disclaimer, and the
       committed-artifact provenance citations — the leaf-DNase bedGraph (official PlantDHS
       BigWig https://plantdhs.org/static/download/Ath_leaf_DNase.bw extracted 2026-10-04 to the
       committed per-bin bedGraph) and the pre-converted TAIR10 truth GTF (deterministic
       GFF3→GTF of the committed GFF3 slice; both internal preparation steps, not re-done
       in-notebook).
    2. First code cell = D-16 fla hard guard, same shape as the main notebooks:
       importlib.util.find_spec("fla") check raising RuntimeError, printing transformers_version=,
       torch_version=, fla_version= (via importlib.metadata; NEVER an import naming fla).
    3. Data-load cell: fetch the 45 kb display region with fetch_sequence from
       ../plant_helixseek_cre/data/chr1_5100001_5300000.fas (the FASTA header carries the genomic
       interval — compute the slice offsets from it, never hardcode twice); the leaf-DNase
       bedGraph and the pre-converted TAIR10 truth GTF (both in its own data/ dir) are referenced
       directly by the tracks ini in the figure cell — no load, no conversion (the Anno GFF3 is
       never read here).
    4. Model-load cell: load BOTH checkpoints through the dnallm public route
       (load_model_and_tokenizer with their packaged registry entries — PlantHelixSeek-CRE binary
       and PlantHelixSeek-Anno 17-BILOU — same load shapes as the main notebooks' own cells;
       models.lock + warm cache).
    5. CRE scan cell: the 500/50/50 sliding-window scan on the 45 kb region only (~1 min),
       transcribed from the CRE notebook's scan cell, giving per-bin p(CRE) scores.
    6. Anno scan cell: 8192/4096 both strands on the 45 kb region only (~1 min) — transcribe the
       Anno notebook's plan_windows semantics onto the slice, predict_window_labels with the BOS
       offset, the _B_SWAP_L permutation for the minus strand (Anno cells 7/9/11) — producing
       per-base label arrays for the display region.
    7. Decode + track-file cell: island decode of the label arrays to predicted-transcript GTFs
       (+ and −), and the p(CRE) bedGraph filtered to >= 0.5 (min_value 0.5, max_value 1,
       #1f77b4; print pcres_bins_shown=<n>). NO binned DHS track; the leaf-DNase and truth-GTF
       tracks need no generated files — the committed bedGraph and GTF are used as-is.
    8. Combined figure cell: six TITLED data tracks matching the approved v5 render
       (.scratch/zoom-candidates/combined-pgt-v5-titled.png) — [CRE predicted p(CRE)] [PlantDHS
       leaf DNase] [Predicted transcripts (+)] [Predicted transcripts (−)] [TAIR10 truth (+)]
       [TAIR10 truth (−)] (+ spacers/x-axis per the render; pred blue #1f77b4, truth + #2ca02c /
       − #d62728); region Chr1:5220001-5265000; pgt invocation as EDIT 2; embed PNG only. Print
       zoom_window=Chr1:5220001-5260000 and combined_window=Chr1:5220001-5265000.
    9. Caption markdown: pinned disclaimer + display-view wording + confident-band filter
       disclosure + < 100 bp transcript-drop disclosure + the +5 kb display flank disclosure +
       leaf-DNase provenance pointer; SHOW-07-compliant; no best claims.
    Constraints: import only modules already exercised in the showcase notebooks plus
    base64/subprocess/sys (no altair at all — the combined notebook has NO alt.Chart cell); no
    metric asserts anywhere (display-only); "jaccard" and "gene_f1" must not appear ANYWHERE in
    this notebook; every line <= 100 chars; kernelspec python3.

    STEP 4 — TEST/LANE WIRING:
    - tests/examples/_execution.py: NOTEBOOK_EXEC_SPECS gains the combined entry — key str(
      EXAMPLE_DIR / "notebooks" / "plant_helixseek_shared" / "plant_helixseek_combined.ipynb"),
      cell_timeout 1200 (strictly below the 2400s slow mark, Pitfall 6 comment), extra_inputs [].
    - tests/examples/test_plant_helixseek_showcase.py:
      * SHOWCASE_NOTEBOOKS gains pytest.param(COMBINED_NB, "combined_locus=Chr1:5220001-5265000",
        id="combined") — provenance / fla-guard / no-fla-import / denylist / caption tests cover
        all three notebooks automatically.
      * test_committed_notebook_has_executed_outputs: keep the vega assertion scoped to the two
        MAIN notebooks (parametrized as today) with image/png >= 2; ADD a dedicated combined
        structure test — committed blob carries >= 1 image/png display output, stream outputs,
        <= 2,097,152 bytes (D-12), and no "jaccard"/"gene_f1" strings in any cell source.
      * Main slow-test extras — CRE: the bedGraph tuple ONLY
        ((SHARED_DATA / "Ath_leaf_DNase_chr1_5100001_5300000.bedGraph"),
        "../plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph"); Anno:
        that bedGraph tuple PLUS the truth-GTF tuple
        ((SHARED_DATA / "TAIR10_GTF_chr1_5100001_5300000.gtf",
        "../plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf")).
      * NEW combined slow test: @pytest.mark.slow + @pytest.mark.timeout(2400) (spec cell_timeout
        1200 strictly below, Pitfall 6); seed_sandbox(COMBINED_NB.parent, tmp_path, extra_inputs=
        [(EXAMPLE_DIR / "notebooks" / "plant_helixseek_cre" / "data" / "chr1_5100001_5300000.fas",
        "../plant_helixseek_cre/data/chr1_5100001_5300000.fas")]) — its own shared data dir
        (selection.md + bedGraph + truth GTF) rides along via the whole-dir copy; assert no
        error outputs, assert_tree_clean(), stream carries zoom_window=Chr1:5220001-5260000 and
        combined_window=Chr1:5220001-5265000, and >= 1 image/png display output. No band
        assertions (display-only notebook).
      * Module docstring updated; do NOT pin coordinates/colors/counts in pytest (owner display
        choices; the one-liner gates pin the windows).
    - Keep ruff clean on both test files.

    POST-SURGERY VALIDATION (same session): all three notebooks — json.loads round-trip,
    ast.parse every code cell, no line over 100 chars, "--trackLabelFraction" nowhere; main
    notebooks — cell counts exactly 23 (CRE) / 29 (Anno), first code cell still the D-16 guard,
    exactly ONE alt.Chart cell at index 13/19 naming "image/png", exactly ONE pgt figure cell
    embedding "image/png" with the 5220000/5260000 literals and zoom_window=Chr1:5220001-5260000,
    no-scoring guards (CRE: no "jaccard"; Anno: no "gene_f1"), CRE zoom cell carries the
    confident-band markers (">= 0.5" + "pcres_bins_shown=") and the bedGraph path, Anno zoom cell
    references the committed truth GTF; combined notebook — cell 0 markdown with selection.md
    link + locus line + disclaimer, first code cell = guard, exactly ONE pgt figure cell with
    image/png + both window evidence prints, ZERO alt.Chart cells, "jaccard" and "gene_f1"
    absent everywhere, the bedGraph + truth GTF + CRE-FASTA paths referenced, and NO
    plant_helixseek_anno/data reference (the GFF3 is never read there).
  </action>
  <verify>
    <automated>grep -n "pygenometracks" pyproject.toml && grep -n "plant_helixseek_combined" tests/examples/_execution.py && test -f example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph && head -1 example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph | grep -q '^#' && ! git check-ignore -q example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph && test -f example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf && head -1 example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf | grep -q '^#' && ! git check-ignore -q example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf && .venv/bin/python -m pytest tests/examples -m "not slow" -q --ignore=tests/examples/test_plant_helixseek_showcase.py && .venv/bin/python -c "
import json, ast, pathlib
main_specs = [
    ('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb', 23, 13, 'jaccard'),
    ('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb', 29, 19, 'gene_f1'),
]
for p, n, fig, banned in main_specs:
    nb = json.loads(pathlib.Path(p).read_text())
    assert len(nb['cells']) == n, (p, 'cell count', len(nb['cells']))
    srcs = [''.join(c.get('source', [])) for c in nb['cells']]
    assert 'find_spec' in srcs[1] and 'RuntimeError' in srcs[1], (p, 'guard changed')
    assert 'Ath_leaf_DNase_chr1_5100001_5300000.bedGraph' in srcs[0], (p, 'provenance bedGraph line missing')
    alts = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'alt.Chart(' in ''.join(c.get('source', []))]
    assert alts == [fig], (p, 'altair figure cells', alts)
    assert 'image/png' in srcs[fig], (p, 'altair cell missing png')
    pgts = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'pgt' in ''.join(c.get('source', [])) and 'image/png' in ''.join(c.get('source', []))]
    assert len(pgts) == 1, (p, 'pgt figure cells', pgts)
    i = pgts[0]
    assert banned not in srcs[i], (p, 'scoring code leaked')
    assert '5220000' in srcs[i] and '5260000' in srcs[i] and 'zoom_window=Chr1:5220001-5260000' in srcs[i], (p, 'window literals/prints missing')
    if banned == 'gene_f1':
        assert 'TAIR10_GTF_chr1_5100001_5300000.gtf' in srcs[i], (p, 'anno zoom does not read the committed truth GTF')
    if 'bin_scores' in srcs[i] or 'p(CRE)' in srcs[i]:
        assert '>= 0.5' in srcs[i] and 'pcres_bins_shown=' in srcs[i], (p, 'confident-band markers missing')
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        assert '--trackLabelFraction' not in s, (p, 'titles stripped')
        assert all(len(line) <= 100 for line in s.splitlines()), (p, 'line over 100 chars')
        if c['cell_type'] == 'code' and s.strip():
            ast.parse(s)
cp_ = 'example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb'
nb = json.loads(pathlib.Path(cp_).read_text())
srcs = [''.join(c.get('source', [])) for c in nb['cells']]
assert nb['cells'][0]['cell_type'] == 'markdown' and 'selection.md' in srcs[0] and 'combined_locus=Chr1:5220001-5265000' in srcs[0] and 'illustrative locus' in srcs[0].lower()
assert 'find_spec' in srcs[1] and 'RuntimeError' in srcs[1], 'combined guard missing'
assert not any('alt.Chart(' in s for s in srcs), 'combined notebook has an altair cell'
assert not any('jaccard' in s for s in srcs) and not any('gene_f1' in s for s in srcs), 'combined notebook has scoring text'
pgts = [i for i, s in enumerate(srcs) if 'pgt' in s and 'image/png' in s]
assert len(pgts) == 1, 'combined figure cells ' + str(pgts)
i = pgts[0]
assert 'zoom_window=Chr1:5220001-5260000' in srcs[i] and 'combined_window=Chr1:5220001-5265000' in srcs[i], 'window prints missing'
assert any('Ath_leaf_DNase_chr1_5100001_5300000.bedGraph' in s for s in srcs) and any('TAIR10_GTF_chr1_5100001_5300000.gtf' in s for s in srcs) and any('plant_helixseek_cre/data' in s for s in srcs), 'data sources missing'
assert not any('plant_helixseek_anno/data' in s for s in srcs), 'combined notebook reads the anno GFF3 (should use the committed GTF)'
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    assert '--trackLabelFraction' not in s
    assert all(len(line) <= 100 for line in s.splitlines()), 'line over 100 chars'
    if c['cell_type'] == 'code' and s.strip():
        ast.parse(s)
print('surgery-ok')" && .venv/bin/ruff check tests/examples/test_plant_helixseek_showcase.py tests/examples/_execution.py && .venv/bin/ruff format --check tests/examples/test_plant_helixseek_showcase.py tests/examples/_execution.py</automated>
  </verify>
  <done>pyproject declares pygenometracks (downgrade gate cited from /tmp/261004-dyw-downgrade-gate.log with the examples lane cheaply re-confirmed); the bedGraph precondition holds (exists, # header, not ignored); both main notebooks carry the PNG mime in the altair cell plus one pgt presentation-only zoom section each (23/29 cells) driven by the owner-chosen literals, the CRE zoom fed directly by the committed bedGraph and the Anno zoom truth tracks fed directly by the committed truth GTF; the combined notebook exists with provenance/guard/both-model scans/decode/six-track figure and window evidence prints (bedGraph + GTF direct, no runtime GFF3→GTF anywhere); lane wiring complete (exec spec, three-notebook parametrization, combined structure + slow tests, per-test extras); all invariants hold (RED on stale outputs is the expected state at task end).</done>
</task>

<task type="auto">
  <name>Task 2: Execute all three notebooks on the GB10 and gate the committed blobs</name>
  <files>example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb</files>
  <action>
    Run from the repo root (nbclient sets the kernel cwd to each notebook's own directory, so
    data/, ../plant_helixseek_{cre,anno}/data/, and outputs/ resolve):
    .venv/bin/jupyter execute --inplace --timeout 1200 example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
    .venv/bin/jupyter execute --inplace --timeout 3600 example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
    .venv/bin/jupyter execute --inplace --timeout 1200 example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
    Budgets mirror NOTEBOOK_EXEC_SPECS (1200/3600/1200, strictly inside the nightly marks).
    Expected wall times on the GB10: CRE ~5-6 min (no sibling inference), Anno ~12-25 min
    (no sibling inference), combined ~3-5 min (two warm model loads + the two ~1 min 45 kb scans
    + pgt render). Run in the background or sequentially with adequate tool timeouts; execution
    requires the local GPU and regenerates every output wholesale.

    After each execution: cross-check the main notebooks' exact frozen values — jaccard=0.3247,
    neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000
    (Anno). The one-liner gates assert the selection.md band mirrors (transient-gate note from
    07-02: the authoritative layer is the parsed _parse_bands() test assertions) plus the vega/
    image/png output counts, the exact window evidence lines, and the size budget. If any metric
    drifts OUTSIDE its band, HALT — environment drift; do not adjust code or literals. If a pgt
    subprocess fails inside a kernel (non-zero exit / empty-track warning), fix the track files
    and re-execute — never commit a figure with an empty-track warning.

    Size fallback: if any notebook exceeds 2,097,152 bytes, shrink ONLY PNG payloads (vl_convert
    scale for altair cells, pgt figure height/width for pgt cells) — never drop a vega mime.

    Confirm the tree: only the three .ipynb modified plus the orchestrator-created uncommitted
    bedGraph and truth GTF (generated bedGraphs/GTFs/inis/PNGs land under gitignored outputs/).
  </action>
  <verify>
    <automated>.venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); j=float(re.search(r'^jaccard=([0-9.]+)',s,re.M).group(1)); n=float(re.search(r'^neg_cre_fraction=([0-9.]+)',s,re.M).group(1)); assert 1.00>=j>=0.30, 'jaccard='+str(j); assert 0.05>=n>=0.0, 'neg_cre_fraction='+str(n); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=1, 'vega display outputs='+str(len(vega)); assert len(png)>=2, 'image/png display outputs='+str(len(png)); assert 'zoom_window=Chr1:5220001-5260000' in s, 'window evidence line missing/wrong'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('cre-notebook-ok jaccard='+str(j)+' neg_cre_fraction='+str(n)+' size='+str(p.stat().st_size))" && .venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); g=int(re.search(r'^genes_above_floor=([0-9]+)',s,re.M).group(1)); n=float(re.search(r'^neg_anno_fraction=([0-9.]+)',s,re.M).group(1)); f=float(re.search(r'^exon_f1=([0-9.]+)',s,re.M).group(1)); r=int(re.search(r'^pred_gff3_rows=([0-9]+)',s,re.M).group(1)); assert g>=3, 'genes_above_floor='+str(g); assert 0.10>=n>=0.0, 'neg_anno_fraction='+str(n); assert 1.0>=f>=0.0, 'exon_f1='+str(f); assert r>=1, 'pred_gff3_rows='+str(r); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=1, 'vega display outputs='+str(len(vega)); assert len(png)>=2, 'image/png display outputs='+str(len(png)); assert 'zoom_window=Chr1:5220001-5260000' in s, 'window evidence line missing/wrong'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('anno-notebook-ok genes='+str(g)+' exon_f1='+str(f)+' neg_anno_fraction='+str(n)+' size='+str(p.stat().st_size))" && .venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert png, 'no image/png display output'; assert 'zoom_window=Chr1:5220001-5260000' in s and 'combined_window=Chr1:5220001-5265000' in s, 'window evidence lines missing/wrong'; assert 'pcres_bins_shown=' in s, 'bins-shown evidence missing'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); blob=p.read_text(); assert 'find_spec' in blob and 'fla_version=' in s, 'guard/versions missing'; assert 'jaccard' not in blob and 'gene_f1' not in blob, 'scoring text present'; print('combined-notebook-ok size='+str(p.stat().st_size))" && test -f example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph && test -f example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf && .venv/bin/python -c "import subprocess; dirt=subprocess.run(['git','status','--porcelain','example/'],capture_output=True,text=True,check=True).stdout; bad=[l for l in dirt.splitlines() if 'Ath_leaf_DNase' not in l and 'TAIR10_GTF' not in l and not l.endswith('.ipynb')]; assert not bad, 'unexpected dirty paths: '+repr(bad); print('tree-ok')"</automated>
  </verify>
  <done>All three one-liner gates print their ok lines (cre-notebook-ok jaccard=0.3247 neg_cre_fraction=0.0325; anno-notebook-ok genes=59 exon_f1=0.7522 neg_anno_fraction=0.0; combined-notebook-ok), each main notebook showing at least one vega and at least two image/png display outputs plus the zoom_window= line, the combined notebook showing its image/png figure plus both window lines with no scoring text, no empty-track warnings, all sizes within 2,097,152 bytes, and the only example/ changes are the three notebooks plus the orchestrator-created bedGraph and truth GTF.</done>
</task>

<task type="auto">
  <name>Task 3: Mirror byte-sync, wrapper pages + mkdocs nav, full gate suite, atomic commit, cross-phase bookkeeping</name>
  <files>docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb, docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph, docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf, docs/example/notebooks/plant_helixseek_cre.md, docs/example/notebooks/plant_helixseek_anno.md, docs/example/notebooks/plant_helixseek_combined.md, mkdocs.yml</files>
  <action>
    1. Mirror sync: cp each executed notebook over its docs mirror (cre/anno dirs and
       docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb), and cp the
       two artifacts to docs/example/notebooks/plant_helixseek_shared/data/
       (Ath_leaf_DNase_chr1_5100001_5300000.bedGraph and TAIR10_GTF_chr1_5100001_5300000.gtf);
       verify byte-identity with cmp for all five pairs.
    2. Wrapper prose:
       - cre.md / anno.md: extend the "Full Notebook" paragraph — the full-locus figures embed an
         image/png mime alongside the vega/vega-lite JSON (local JupyterLab/VS Code render), and
         each notebook adds an owner-chosen illustrative pgt zoom window; point readers to the
         NEW combined notebook for the both-modality view. PROSE ONLY, no new code blocks.
       - plant_helixseek_combined.md (NEW, finetune_binary pattern): front-matter
         (notebook: example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb,
         sync_check: true), title, intro paragraph (one window, both PlantHelixSeek modalities
         and their truths on one aligned pygenometracks axis — CRE confident band, official
         PlantDHS leaf DNase signal from the committed bedGraph, predicted transcripts, and
         TAIR10 truth from the committed pre-converted GTF; display view, full-locus metrics in
         the two showcase notebooks remain
         authoritative), Full Notebook button, Prerequisites (uv pip install -e '.[base,fla]'
         plus the notebook extra; CUDA GPU), section prose mirroring the notebook narrative. Any
         code block must be copied VERBATIM from the authored notebook source (the AST sync
         check matches statements); prose-only is acceptable and preferred.
    3. mkdocs.yml: add "- PlantHelixSeek Combined View: example/notebooks/plant_helixseek_combined.md"
       to the Showcase nav block (lines 212-214 neighborhood, after the Gene Annotation entry).
    4. Full gate suite (all from repo root): fast showcase lane
       (.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q
       → passes — now covering all three notebooks); the FULL fast suite once more WITH the
       showcase module (.venv/bin/python -m pytest tests -m "not slow" -q — the downgrade gate
       re-run now includes the new assertions; must be fully green, matching the pre-satisfied
       baseline 1741P/1S modulo the new tests); python3 scripts/check_notebook_md_sync.py still
       reports exactly the 3 pre-existing stale wrappers and NO plant_helixseek line (the new
       wrapper counts as a pair — it must be clean); python3 scripts/validate_docs_snippets.py
       green; python3 scripts/check_docs_sync.py → "OK: docs/example/ is in sync with example/"
       (the combined mirror + both artifacts are part of this); .venv/bin/ruff check + format --check
       on both test files.
    5. Atomic commit — ONE commit containing exactly SEVENTEEN files: pyproject.toml, mkdocs.yml,
       tests/examples/_execution.py, tests/examples/test_plant_helixseek_showcase.py, the three
       example/ notebooks, the bedGraph + truth GTF, the three docs notebook mirrors, the
       bedGraph + GTF mirrors, and the three wrappers. Message:
       "docs(quick-261004): showcase PNG mimes, pgt zoom windows, and the combined PlantHelixSeek notebook".
       NO attribution trailers of any kind (owner rule). Push to origin phs (owner default:
       commit and push). Do not include unrelated dirty files (.planning/, .scratch/, /tmp).
    6. Cross-phase bookkeeping (into the SUMMARY, this task's output): (a) the GPL-3.0 owner
       override adopting pygenometracks into the notebook extra (2026-10-04, dnallm code never
       imports it); (b) the Phase-8 example-job install notes need the 05-FEASIBILITY CFLAGS
       deviation line for pyBigWig (default sdist build fails on the stock box toolchain);
       (c) the matplotlib pin (pgt 3.9 → matplotlib 3.8.4) with the pre-satisfied gate evidence
       (/tmp/261004-dyw-downgrade-gate.log, 1741P/1S/exit 0) and the Task-3 re-run with the new
       assertions; (d) the combined notebook's display region (window + 5 kb flank,
       owner-directed computation reduction); (e) the committed-artifact format decisions (the
       *.tsv gitignore trap at .gitignore:86; the .bedGraph and .gtf escapes with provenance
       riding in their # headers; the owner-directed pre-converted truth GTF replacing all
       runtime GFF3→GTF conversion — never git add -f).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q && .venv/bin/python -m pytest tests -m "not slow" -q && cmp example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb && cmp example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb && cmp example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb && cmp example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph && cmp example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf && out=$(python3 scripts/check_notebook_md_sync.py); ! echo "$out" | grep -q plant_helixseek && python3 scripts/validate_docs_snippets.py && dsout=$(python3 scripts/check_docs_sync.py) && echo "$dsout" | grep -q '^OK: docs/example/' && grep -q "plant_helixseek_combined" mkdocs.yml && hfiles=$(git show --name-only --format= HEAD) && diff <(printf '%s\n' "$hfiles" | sort) <(printf '%s\n' pyproject.toml mkdocs.yml tests/examples/_execution.py tests/examples/test_plant_helixseek_showcase.py example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf docs/example/notebooks/plant_helixseek_cre.md docs/example/notebooks/plant_helixseek_anno.md docs/example/notebooks/plant_helixseek_combined.md | sort) && cmsg=$(git log -1 --format=%B) && ! echo "$cmsg" | grep -qiE 'co-authored|generated-with|attribution'</automated>
  </verify>
  <done>Showcase fast tests pass across all three notebooks AND the full fast suite (matplotlib-downgraded environment) is green with the new assertions; five cmp-identical mirror pairs (three notebooks + bedGraph + truth GTF); the new wrapper is sync-clean and joins the mkdocs Showcase nav; md-sync failure set unchanged (3 pre-existing, none plant_helixseek); snippets and docs-sync green; ruff clean; HEAD is the single atomic commit touching exactly the seventeen files with no attribution trailers, pushed to origin phs; the SUMMARY records the five bookkeeping items.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| executed kernel → committed blob | notebook outputs are committed as showcase evidence; edits without re-execution would present stale/forged results |
| external packages → .venv → notebook subprocess | pygenometracks (GPL-3.0) + pyBigWig enter via the notebook extra and pin matplotlib down; dnallm code never imports them |
| external data → committed bedGraph / pre-converted GTF → notebook panels | the leaf-DNase panel derives from an externally downloaded BigWig and the truth GTF from the committed GFF3; both committed derivatives must carry provenance, stay integrity-pinned, and actually be committable (the *.tsv ignore trap) |
| committed blob → public docs surface | mirrors + wrappers + mkdocs nav are what readers consume; drift between example/ and docs/ misrepresents evidence |
| owner selection → notebook literals | the zoom window, display flank, axis filter, and renderer are human display choices; the notebook must not dress them up as computed optima, and display filtering must be disclosed |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261004-01 | Tampering | showcase notebook committed outputs (all three) | high | mitigate | Edits and executions land in one atomic commit; per-notebook one-liner gates assert the executed stream metrics in-band (main notebooks), vega/image/png output counts, and the exact window lines; the nightly lane re-executes all three and re-asserts parsed bands where applicable |
| T-261004-02 | Tampering | 2 MB notebook budget (D-12) | medium | mitigate | Size asserted in all one-liner gates and pinned by the structure tests; pgt figures embed a single PNG each; documented shrink fallback never drops a vega mime |
| T-261004-03 | Repudiation | illustrative zoom/combined framing (SHOW-07) | medium | mitigate | Captions carry the pinned disclaimer, name the main notebooks' full-locus metrics as authoritative, and DISCLOSE the confident-band filter, the < 100 bp transcript drop, and the +5 kb display flank as owner display choices; selection tooling and candidate renderings live only in /tmp and .scratch (gitignored); window evidence lines derive from plain literals, never scoring code; the genome-wide denylist test plus the caption test enforce this kernel-free across all three notebooks |
| T-261004-04 | Tampering | committed leaf-DNase bedGraph + pre-converted truth GTF (external-data derivatives + committability) | high | mitigate | The original .tsv was silently gitignored (.gitignore:86) — both artifacts use suffixes that escape every ignore (.bedGraph, .gtf; verified git check-ignore negative) with provenance INSIDE their # headers (pgt parser tolerance verified live, 0 issues); Task-1 precondition re-checks existence + non-ignored status of BOTH before surgery; integrity via byte-identical docs mirrors (cmp) inside the atomic commit; no git add -f anywhere; both read at runtime directly by pgt (no network, no pyBigWig in notebooks, no dnallm change, no runtime GFF3-to-GTF) |
| T-261004-05 | Tampering | pygenometracks adoption (GPL-3.0 + matplotlib pin) | medium | mitigate | Owner-explicit override 2026-10-04 recorded in SUMMARY; declared only in the notebook extra — dnallm package code never imports it; matplotlib 3.11.2→3.8.4 downgrade proven harmless pre-surgery (1741P/1S/exit 0, /tmp/261004-dyw-downgrade-gate.log) and re-proven WITH the new assertions at Task-3 close; CLI invoked PATH-independently as the sys.executable sibling |
| T-261004-06 | Tampering | new combined notebook lane wiring | medium | mitigate | Same contract as the showcase siblings: D-16 guard first code cell, provenance/denylist/no-fla-import structure tests parametrized over it, NOTEBOOK_EXEC_SPECS budget under its pytest-timeout mark, NEVER in ACTIVE_NOTEBOOKS/GATED_NOTEBOOKS (no census double-execution), sandbox extras explicit for its cross-notebook reads |
| T-261004-SC | Tampering | package installs (pygenometracks + pyBigWig) | high | mitigate | Owner-adopted and live-verified this session (pgt 3.9 + pyBigWig 0.3.26 installed, all five track types render from real data, zero empty-track warnings); fresh-env installs documented via the 05-FEASIBILITY CFLAGS deviation; the Phase-8 example-job install note recorded in SUMMARY — no undocumented install path exists |
</threat_model>

<verification>
- Task 1: pyproject declares pygenometracks; exec spec entry present; downgrade gate cited (log path) with the examples lane cheaply re-confirmed; bedGraph precondition verified (exists, # header first line, NOT gitignored); surgery validation — main notebooks 23/29 cells with exactly one altair cell (13/19, image/png) and one pgt zoom cell (window literals + prints, no-scoring, CRE confident-band markers, titles kept, bedGraph path wired); combined notebook with provenance/guard/one pgt figure cell (both window prints), zero altair cells, zero scoring strings, all data sources wired; lane wiring in both test files.
- Task 2 one-liner gates: CRE (jaccard [0.30, 1.00] observed 0.3247; neg_cre_fraction [0.00, 0.05] observed 0.0325) and Anno (genes >= 3 observed 59; neg_anno_fraction [0.00, 0.10] observed 0.0000; exon_f1 observed 0.7522; pred_gff3_rows >= 1), each with >= 1 vega + >= 2 image/png outputs and zoom_window= exactly Chr1:5220001-5260000; combined with its image/png figure, zoom_window=Chr1:5220001-5260000 + combined_window=Chr1:5220001-5265000 + pcres_bins_shown=, no scoring text, guard intact; all <= 2,097,152 bytes; tree clean outside the three notebooks + bedGraph; no empty-track warnings.
- Task 3 full suite: showcase fast tests green across all three notebooks AND the full fast suite green with the new assertions; five cmp-identical mirror pairs; new wrapper sync-clean + mkdocs nav entry; md-sync failure set unchanged; snippets and docs-sync green; ruff clean; single atomic seventeen-file commit, no attribution trailers, pushed; SUMMARY records the five bookkeeping items.
</verification>

<success_criteria>
- The full-locus altair figures render everywhere (image/png added; vega/vegalite retained for GitHub); the new pgt zoom figures embed PNG with titles on and the exact owner window; all figure work in the main notebooks required zero additional inference.
- The combined notebook is the single both-modality view: both models scanned only over the 45 kb display region (window + 5 kb flank), six titled tracks aligned on one genomic axis matching the approved v5 render, with the confident band, transcript drop, and flank all disclosed; no scoring code anywhere (structure tests + blob greps prove it).
- The leaf-DNase evidence artifact is a properly-committable UCSC bedGraph carrying its own provenance header, byte-identical in docs, fed directly to pgt with no conversion step.
- The main notebooks reproduce jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000; all three notebooks within 2,097,152 bytes; the nightly lane executes all three with correct budgets and seeded extras; the combined notebook never joins ACTIVE_NOTEBOOKS.
- pygenometracks>=3.9 declared in the notebook extra with the GPL override, the Phase-8 pyBigWig CFLAGS install note, the matplotlib-pin evidence (pre-satisfied + re-run), the display-region decision, and the committed-artifact gitignore-escape decisions (bedGraph + pre-converted truth GTF) recorded in the SUMMARY; docs mirrors (three notebooks + bedGraph + GTF) byte-identical; wrapper + mkdocs nav complete; single atomic seventeen-file commit without attribution trailers on origin phs.
</success_criteria>

<output>
Create `.planning/quick/261004-dyw-planthelixseek-showcase-notebook-vega-ve/261004-dyw-SUMMARY.md` when done. The SUMMARY MUST record (owner decisions, cross-phase bookkeeping): (1) the GPL-3.0 override adopting pygenometracks into the notebook extra (2026-10-04; dnallm code never imports it; example-notebook use only); (2) the Phase-8 example-job install notes need the 05-FEASIBILITY CFLAGS deviation line for pyBigWig; (3) the pgt 3.9 matplotlib pin (3.11.2 → 3.8.4) with the pre-satisfied gate evidence (/tmp/261004-dyw-downgrade-gate.log, 1741P/1S/exit 0) and the Task-3 re-run; (4) the combined notebook's display region (owner window + 5 kb downstream flank — owner-directed computation reduction); (5) the committed-artifact format decisions (.gitignore:86 `*.tsv` trap; .bedGraph and .gtf escapes with provenance in their # headers; the owner-directed pre-converted truth GTF replacing all runtime GFF3-to-GTF conversion; never git add -f).
</output>
