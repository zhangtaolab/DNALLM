---
phase: 261004-dyw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - tests/examples/test_plant_helixseek_showcase.py
  - example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_anno.md
autonomous: true
requirements:
  - SHOW-07
  - D-12
  - D-13

estimate:
  tokens: 48000
  raw_tokens: 48000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - The EXISTING altair full-locus figures (CRE cell 13, Anno cell 19) keep their vega v6 object + vegalite v6 JSON string and GAIN an image/png mime (vl_convert) — at least one vega mime and at least three image/png display outputs per committed notebook (full-locus + zoom + combined).
    - ALL NEW figures (per-modality zoom + combined, in both notebooks) render via the pygenometracks CLI (`pgt`, invoked PATH-independently as the sys.executable sibling) and embed image/png (+ text/plain) ONLY — no vega mimes for pgt figures. Track titles stay ON (never pass --trackLabelFraction 0).
    - ALL new figures use the owner-selected window Chr1:5220001-5260000 (0-based half-open 5220000-5260000, 40 kb) as plain literals. Zero window-selection/scoring code in the notebooks: no "jaccard" string in any CRE new-figure cell, no "gene_f1" string in any Anno new-figure cell, no best/optimal claims anywhere.
    - The combined figure (BOTH notebooks, matching the owner-approved v4 render .scratch/zoom-candidates/combined-pgt-v4.png) is a pgt track stack sharing the genomic axis: CRE predicted p(CRE) bedGraph (rows FILTERED to score >= 0.5 at write time — sub-0.5 bins not drawn; min_value 0.5, max_value 1, color #1f77b4; pcres_bins_shown= printed) → PlantDHS leaf DNase bedGraph from the committed tsv (#8c2d04) → spacer → predicted transcripts (+) and (−) GTF (#1f77b4) → spacer → TAIR10 truth (+) GTF (#2ca02c) and truth (−) GTF (#d62728) → x-axis. NO binned DHS track (owner removed it).
    - Predicted-transcript GTFs come from a presentation-only island decode of per-base argmax labels — the Anno notebook decodes its EXISTING full-locus label_tracks (cell 11, no rescan); the CRE notebook runs the sibling PlantHelixSeek-Anno scan over the 40 kb slice only (validated live: 8192/4096 both strands, plan_windows semantics on the slice, BOS offset, _B_SWAP_L permutation) and the Anno notebook runs the sibling PlantHelixSeek-CRE 500/50/50 scan on the window. Non-O label islands become gene/transcript/exon/CDS/five_prime_utr/three_prime_utr features (INTRON runs = inter-exon spacers, not emitted); transcripts with span < 100 bp are not drawn (caption discloses); window slicing uses overlap semantics (edge-crossing transcripts included and clipped).
    - Truth GTFs are a deterministic GFF3→GTF conversion of the committed TAIR10 GFF3 (pygenometracks readGtf requires gene_id/transcript_id — bare GFF3 ID/Parent yields "No transcript found" + empty track, verified live); types gene/mRNA/exon/CDS/UTR kept, first-Parent value becomes transcript_id.
    - The leaf-DNase track reads the committed example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv — NO pyBigWig use in notebooks, NO runtime network, NO dnallm code change; the provenance cell of both notebooks gains one line citing the extraction.
    - pygenometracks>=3.9 is declared in the pyproject.toml `notebook` extra (GPL-3.0 — owner explicitly overrides the v1-milestone exclusion on 2026-10-04; dnallm package code never imports it; example-notebook use only; pyBigWig rides along; dev-box/CI install needs the 05-FEASIBILITY CFLAGS deviation for pyBigWig — recorded for the Phase-8 example job). pygenometracks 3.9 pins matplotlib down (3.11.2 → 3.8.4, live-observed): the FULL fast suite MUST pass on the downgraded environment before any notebook edit proceeds.
    - Re-execution on the GB10 reproduces the frozen metrics exactly: jaccard=0.3247 and neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000 (Anno).
    - Both committed notebooks stay at most 2,097,152 bytes (D-12) — pgt figures embed a single PNG each, well under budget.
    - The fast showcase structure lane passes (13 tests) including the rekeyed assertions (≥1 vega + ≥3 image/png outputs; disclaimer captions after EVERY figure cell, altair or pgt); the SHOW-07 denylist stays green; the nightly slow tests' sandbox extras seed the tsv (both) and the Anno GFF3 (CRE test) so the lane keeps passing; docs mirrors stay byte-identical (notebooks AND the tsv); md-sync/snippets/docs-sync stay exactly as green as baseline.
    - pyproject + notebooks + mirrors + tsv + wrappers + test file land in ONE atomic commit (ten files) with no attribution trailers.
  artifacts:
    - pyproject.toml — `notebook` extra gains "pygenometracks>=3.9".
    - example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv — committed (orchestrator-created, 59,818 bytes, provenance header); mirrored byte-identically under docs/.
    - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb — 26 cells (20 + 6): figure cell 13 gains image/png; new pgt zoom section (p(CRE) confident band + leaf DNase) and pgt combined section after caption cell 14; provenance line for the tsv.
    - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb — 32 cells (26 + 6): figure cell 19 gains image/png; new pgt zoom section (pred + truth GTF tracks) and pgt combined section after caption cell 20; provenance line for the tsv.
    - tests/examples/test_plant_helixseek_showcase.py — structure tests rekeyed to the new figure mix; both slow tests' extras lists gain the tsv (both) and the cross-notebook Anno GFF3 (CRE test only).
    - docs/example/notebooks/plant_helixseek_{cre,anno}/plant_helixseek_{cre,anno}.ipynb and docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv — byte-identical mirrors.
    - docs/example/notebooks/plant_helixseek_{cre,anno}.md — one added prose passage each (PNG mimes, pgt-rendered zoom + combined views, leaf-DNase panel), no new code blocks.
  key_links:
    - Owner selection (SELECTION.txt, already written) → plain literals 5220000-5260000 in every new figure cell → zoom_window=/combined_window= evidence lines printing exactly Chr1:5220001-5260000 — the only window data the committed notebooks derive.
    - pyproject notebook extra → .venv pgt 3.9 + pyBigWig 0.3.26 (live-verified) → notebook cells invoke the CLI as Path(sys.executable).with_name("pgt") (PATH-independent; console scripts are pgt/pyGenomeTracks — there is NO `pygenometracks` script) with check=True → PNG bytes → base64 image/png display.
    - Computed labels/scores → per-figure bedGraph/GTF files + tracks ini under gitignored outputs/ → pgt render → embedded PNG; the committed tsv (example + docs mirrors) feeds the leaf-DNase bedGraph in both notebooks.
    - matplotlib downgrade (3.8.4) → full fast suite gate BEFORE surgery (downgrade breaks nothing) → showcase module excluded from the post-surgery variant only because its extended tests are expected-RED until re-execution.
    - Committed blobs → structure tests + the one-liner gates (≥1 vega + ≥3 image/png outputs, zoom_window=/combined_window= exact, bands, size, guard).
---

<objective>
Enhance the two PlantHelixSeek showcase notebooks' result display (owner request; final form after
three revision rounds — owner-selected window, combined CRE+Anno figure, pygenometracks renderer):

1. The existing altair full-locus figures gain an image/png mime (vl_convert) next to vega/vegalite
   — GitHub renders the vega object but local JupyterLab/VS Code do not; PNG renders everywhere.
2. The owner has ALREADY selected the window (candidate phase complete): Chr1:5220001-5260000 for
   every new figure in both notebooks. The notebooks contain ONLY presentation: literal
   coordinates, no selection or scoring code.
3. All NEW figures render via pygenometracks (owner adopted it over the altair design; v4 render
   approved): per-modality zoom figures plus a combined CRE+Anno track stack (p(CRE) confident
   band, official leaf DNase signal, predicted transcripts, TAIR10 gene models on one genomic
   axis, titles on). Each notebook computes its SIBLING model's prediction on the fixed window.
4. pygenometracks joins the notebook extra (GPL-3.0 owner override), with the matplotlib-pin risk
   gated by a full fast-suite run on the downgraded environment before any edit.
5. Both notebooks are re-executed on the GB10 and must reproduce the frozen metrics exactly
   (jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59,
   neg_anno_fraction=0.0000).
6. Notebooks + tsv byte-synced to docs mirrors, wrappers updated (prose only), 2 MB budget held,
   fast structure lane + nightly extras + docs-sync gates green, everything in one atomic commit
   (ten files) with no attribution trailers.

Purpose: the showcase notebooks are the public evidence surface for the PlantHelixSeek work
(Phases 6-7); today their figures are invisible to anyone opening them locally, and no single
view aligns CRE signal, open-chromatin truth, and gene structure on one axis.
Output: re-executed notebooks with dual-render full-locus figures and pgt-rendered zoom/combined
views over the owner-chosen window, a declared notebook-extra dependency, synced
mirrors/wrappers/tsv, extended kernel-free structure tests.
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
  .scratch/zoom-candidates/combined-pgt-v4.png — track stack top→bottom: titled p(CRE)
  bedGraph (confident band), titled leaf-DNase bedGraph, spacer, predicted transcripts (+)/(−),
  spacer, TAIR10 truth (+)/(−), x-axis. The earlier v4 preview stripped titles via
  --trackLabelFraction 0 — that flag must NOT appear.
- New committed data artifact (created by the orchestrator, currently UNCOMMITTED):
  example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv —
  59,818 bytes, # comment header (source https://plantdhs.org/static/download/Ath_leaf_DNase.bw,
  extracted 2026-10-04, 0-based half-open 50 bp bins aligned to the CRE scan grid) then a
  start/signal TSV over the whole 200 kb locus. Read with pandas, skipping the # header.
  Mirror target docs/example/notebooks/plant_helixseek_shared/data/ exists.
- CRE notebook: 20 cells, altair figure = code cell 13, caption = markdown cell 14; presentation
  state: pred_df/truth_df, bin_scores, bin_width, genomic_offset. Anno: 26 cells, altair figure =
  code cell 19, caption = markdown cell 20; presentation state: label_tracks = {"+": plus_labels,
  "-": minus_labels} (cell 11, full-locus per-base argmax labels — the zoom/combined decode MUST
  reuse these, never rescan), predicted_segments, genomic_offset, locus_length, plus the
  plan_windows / predict_window_labels / _B_SWAP_L machinery in cells 7/9/11 the CRE notebook's
  sibling scan transcribes onto the 40 kb slice.
- Per-notebook data dirs (ride along with the notebook dir in sandboxes):
  plant_helixseek_cre/data/{chr1_5100001_5300000.fas, TAIR10_DHSs_chr1_5100001_5300000.gff};
  plant_helixseek_anno/data/{chr1_5100001_5300000.fas, TAIR10_GFF3_chr1_5100001_5300000.gff3}.
  The only cross-notebook read left in the final design: the CRE combined truth-GTF conversion
  reads ../plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3 (seed it in the CRE
  slow test's extras). The Anno notebook needs only its own files + the tsv (the binned-DHS track
  is gone from the final design).
- pgt GTF requirement (live-verified): readGtf needs gene_id/transcript_id attributes — feeding
  raw GFF3 ID/Parent attributes yields "No transcript found" + an empty track; the deterministic
  GFF3→GTF conversion keeps gene/mRNA/exon/CDS/UTR and maps first-Parent → transcript_id.
- .venv/bin/jupyter execute --inplace --timeout N exists; nbclient sets the kernel cwd to the
  notebook's own directory so relative data paths and outputs/ resolve (proven 07-01).
  NOTEBOOK_EXEC_SPECS budgets: CRE cell_timeout 1200, Anno 3600 (tests/examples/_execution.py:196-220).
- Wrappers docs/example/notebooks/plant_helixseek_{cre,anno}.md do NOT quote the figure cells —
  editing figure cells and inserting cells cannot break the AST sync check.
- Baselines: check_notebook_md_sync fails on exactly 3 pre-existing stale wrappers
  (mcp_langchain, mcp_pydantic_ai, data_prepare_finetune — none plant_helixseek);
  validate_docs_snippets green; check_docs_sync "OK". outputs/ is gitignored (.gitignore:63);
  .scratch/ is ignored (verified via check-ignore).
- Owner rules: zoom selection stays internal (done); pygenometracks adoption is an explicit
  GPL-3.0 override recorded 2026-10-04 (dnallm code never imports it); atomic commit
  (pyproject + notebooks + mirrors + tsv + wrappers together), no attribution trailers,
  execution on the local GB10.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Dependency adoption + downgrade gate, notebook surgery (PNG mimes, pgt zoom + combined sections), test extension</name>
  <files>pyproject.toml, example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, tests/examples/test_plant_helixseek_showcase.py</files>
  <behavior>
    - Structure test: both committed notebooks expose at least THREE image/png mimes across display outputs and at least ONE vega mime (the full-locus altair figures) — named-cause: local JupyterLab/VS Code rendering + pgt-rendered zoom/combined views.
    - Structure test: EVERY figure cell — keyed on a display dict naming "image/png", covering both the altair cell and the pgt cells — is followed by a markdown cell whose text (lowercased) carries the pinned disclaimer "illustrative locus, not genome-wide accuracy".
    - Nightly lane keeps passing: both slow tests' extras seed the tsv; the CRE slow test also seeds the Anno GFF3.
    - These go RED against the surgically edited (not yet re-executed) blobs — expected transient; they turn GREEN only after Task 2's re-execution. Do not weaken them to pass early.
  </behavior>
  <action>
    STEP 1 — DEPENDENCY ADOPTION + DOWNGRADE GATE (before any notebook edit):
    a. Edit pyproject.toml: the `notebook` extra (currently jupyter>=1.1.1, marimo>=0.16.3,
       nbclient>=0.10) gains "pygenometracks>=3.9". GPL-3.0 owner override 2026-10-04 — dnallm
       package code never imports it; example-notebook use only. Do not touch any other extra.
    b. Confirm the .venv state (already installed live: pgt 3.9, pyBigWig 0.3.26, matplotlib
       3.8.4). If a fresh environment ever needs it, the documented install is the
       05-FEASIBILITY CFLAGS deviation for pyBigWig (path + exact line in the context block).
    c. DOWNGRADE GATE (mandatory, ordered BEFORE the surgery): run the full fast suite on the
       downgraded environment — .venv/bin/python -m pytest tests -m "not slow" -q — and require
       it fully green. pygenometracks 3.9 pins matplotlib 3.11.2 → 3.8.4; any failure here means
       the downgrade broke something — STOP and report; do NOT proceed to surgery.

    STEP 2 — NOTEBOOK SURGERY. MECHANICS: the Edit tool refuses .ipynb — do all notebook edits in
    one throwaway .venv/bin/python session using nbformat (nbformat.read as_version=4, exact
    single-match string anchors, nbformat.v4 new_markdown_cell/new_code_cell inserts,
    nbformat.write). Do NOT execute anything in this task — Task 2 regenerates outputs wholesale.
    Consult the reference render .scratch/zoom-candidates/combined-pgt-v4.png. The chosen window
    literals throughout: zoom_start=5220000, zoom_end=5260000 (display Chr1:5220001-5260000,
    40 kb). Every new figure cell carries a brief comment naming the window as an owner pick and
    names pygenometracks as the renderer.

    EDIT 1 (both notebooks, altair figure cell — CRE 13, Anno 19): add import base64; add one
    display-dict entry after the vegalite entry and before text/plain: "image/png" mapped to
    base64.b64encode(vlc.vegalite_to_png(vegalite_spec)).decode("ascii"). Extend the embed
    comment (local JupyterLab/VS Code do not render the vega v6 object + bare .json string).
    These remain the ONLY vega-mime figures. Everything else in these cells stays as-is.

    EDIT 2 (CRE — pgt zoom section, 3 cells after caption cell 14):
    a. Markdown header "## Illustrative zoom window" — shown for display clarity; full-locus
       metrics above remain the authoritative claim.
    b. Code cell, PRESENTATION ONLY, rendering via pgt: write under gitignored outputs/ a
       bedGraph of the window's bin scores FILTERED AT WRITE TIME to score >= 0.5 (sub-0.5 rows
       never written; print the descriptive count pcres_bins_shown=<n>) and a bedGraph of
       leaf-DNase rows (from the committed tsv, # header skipped); write a pgt tracks ini with
       TITLED tracks — CRE p(CRE) confident band (bed_graph, min_value 0.5, max_value 1, color
       #1f77b4) and leaf DNase (bed_graph, color #8c2d04) — plus the x-axis section; invoke the
       CLI as str(Path(sys.executable).with_name("pgt")) (PATH-independent; there is no
       `pygenometracks` console script) with subprocess check=True over region
       Chr1:5220001-5260000; read the PNG bytes and embed via display(..., raw=True) with
       image/png (+ a text/plain line) ONLY — no vega mimes. Print
       zoom_window=Chr1:5220001-5260000 and zoom_width=40000. NEVER pass --trackLabelFraction 0.
       ZERO scoring code — "jaccard" must not appear in this cell's source.
    c. Markdown caption: pinned sentence "Illustrative locus, not genome-wide accuracy." +
       chosen-for-display-clarity wording + the filter disclosure ("only p(CRE) >= 0.5 bins
       shown — confident band; the full-locus overview above shows the unfiltered track") +
       rendered-with-pygenometracks note. SHOW-07-compliant; no best claims.

    EDIT 3 (Anno — pgt zoom section, 3 cells after caption cell 20): same literal-window pattern
    for a 4-track gene view: predicted transcripts (+) GTF (#1f77b4), predicted transcripts (−)
    GTF (#1f77b4), TAIR10 truth (+) GTF (#2ca02c), TAIR10 truth (−) GTF (#d62728), x-axis —
    GTFs from the shared decode/conversion helpers (below), titles on, PNG-only embed, print
    zoom_window=Chr1:5220001-5260000 (plus zoom_width=40000 and descriptive
    zoom_transcripts=<n> if useful). ZERO scoring code — "gene_f1" and "EXON_F1_FLOOR" must not
    appear. Caption as EDIT 2c adapted (display-clarity, no filter needed here unless the < 100 bp
    transcript drop applies — it does: disclose it).

    SHARED HELPERS (define once per notebook, in the zoom code cells or a small dedicated cell —
    presentation-only, no metric computation):
    - Predicted-transcript GTF decode: scan a per-base label array (Anno: the EXISTING cell-11
      label_tracks, never a rescan) for maximal islands of non-O labels; per island emit GTF
      gene + transcript spanning it and exon/CDS/five_prime_utr/three_prime_utr sub-features from
      the label runs (INTRON runs are inter-exon spacers — NOT emitted); drop transcripts whose
      total span < 100 bp (captions disclose); window-slice with OVERLAP semantics —
      edge-crossing transcripts included and clipped to the window, never inside-only;
      locus-local → genomic via genomic_offset; 1-based closed GTF lines with gene_id/transcript_id.
    - Truth GTF conversion: deterministic GFF3→GTF of the committed TAIR10 GFF3 keeping
      gene/mRNA/exon/CDS/UTR rows, first-Parent value → transcript_id (bare GFF3 attributes
      leave pgt readGtf with "No transcript found" + an empty track).
    - Sibling scans: CRE notebook loads PlantHelixSeek-Anno (registry entry, same load shape as
      its own model cell) and scans ONLY the 40 kb slice — transcribe the Anno notebook's
      plan_windows semantics onto the slice, 8192/4096 both strands, predict_window_labels with
      the BOS offset, the _B_SWAP_L permutation for the minus strand (Anno cells 7/9/11; ~2 min
      including load, owner-validated) — then the island decode. Anno notebook loads
      PlantHelixSeek-CRE (binary CRE registry entry) and runs the 500/50/50 scan on the window
      (transcribe the CRE notebook's scan parameters) for the p(CRE) bedGraph.

    EDIT 4 (CRE — pgt combined section, 3 cells after the zoom caption):
    a. Markdown header "## Combined CRE + Anno view" — one window, both modalities and their
       truths on one aligned genomic axis (pygenometracks); display clarity; full-locus metrics
       above remain authoritative.
    b. Code cell, PRESENTATION ONLY: build the v4 track stack — [CRE predicted p(CRE)] bedGraph
       from bin_scores window slice, rows filtered to >= 0.5, min_value 0.5, max_value 1, color
       #1f77b4, pcres_bins_shown= printed; [PlantDHS leaf DNase] bedGraph from the committed tsv,
       color #8c2d04; [spacer]; [Predicted transcripts (+)] GTF #1f77b4; [Predicted transcripts
       (−)] GTF #1f77b4; [spacer]; [TAIR10 truth (+)] GTF #2ca02c; [TAIR10 truth (−)] GTF #d62728;
       [x-axis]. Predicted GTFs from the sibling Anno scan + island decode (SHARED HELPERS);
       truth GTF from the GFF3→GTF conversion of
       ../plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3. NO binned DHS track.
       Titles on (no --trackLabelFraction). pgt invocation as EDIT 2b; PNG-only embed; print
       combined_window=Chr1:5220001-5260000. ZERO scoring code — "jaccard" must not appear.
    c. Markdown caption: pinned disclaimer + display-view wording + confident-band filter
       disclosure + < 100 bp transcript-drop disclosure + leaf-DNase provenance pointer (see the
      provenance cell); full-locus metrics remain authoritative; SHOW-07-compliant; no best claims.

    EDIT 5 (Anno — pgt combined section, 3 cells after the zoom caption): the SAME track stack
    with inputs flipped to what this notebook already holds — p(CRE) bedGraph from the sibling
    CRE 500/50/50 window scan (filtered >= 0.5, min 0.5 max 1, #1f77b4, pcres_bins_shown=);
    leaf-DNase bedGraph from the committed tsv; predicted-transcript GTFs decoded from the
    EXISTING cell-11 label_tracks (NO sibling Anno scan, NO rescan); truth GTF from the
    notebook's own committed GFF3 slice. Same titles/spacers/axis, same literals, PNG-only embed,
    print combined_window=Chr1:5220001-5260000. ZERO scoring code — "gene_f1" must not appear.
    Caption as EDIT 4c.

    EDIT 6 (both notebooks, provenance markdown cell 0): append ONE line citing the tsv
    extraction — official PlantDHS leaf DNase BigWig
    (https://plantdhs.org/static/download/Ath_leaf_DNase.bw) extracted 2026-10-04 to the
    committed per-bin tsv in plant_helixseek_shared/data/ (0-based half-open, 50 bp bins aligned
    to the CRE scan grid); extraction is an internal preparation step, not re-done in-notebook.
    Keep every existing provenance line intact (structure test pins them).

    CONSTRAINTS on every new/edited code cell: import only modules already exercised in these
    notebooks plus base64/subprocess/sys for the pgt calls (json, pandas, numpy, torch, pathlib,
    re, vl_convert, altair in the altair cells only) — the fast-lane import-exec test and the
    D-16 no-fla-import rule stay satisfied; never name the fla module. No metric asserts anywhere
    new (nightly re-execution safety). Every source line at most 100 chars.

    EDIT 7 (tests/examples/test_plant_helixseek_showcase.py):
    - test_committed_notebook_has_executed_outputs: assert at least ONE display_data carries an
      application/vnd.vega mime (the full-locus altair figures) and at least THREE display_data
      outputs carry "image/png" (full-locus + zoom + combined). Update the docstring.
    - test_illustrative_caption_follows_the_metric_figure: rekey from "alt.Chart(" to figure
      cells = code cells whose source contains a display dict naming "image/png" (covers the
      altair cell AND both pgt cells; expect at least 3 per notebook) — the first markdown after
      EACH must carry ILLUSTRATIVE_DISCLAIMER. Update the docstring.
    - Slow-test extras: BOTH tests gain the tsv tuple
      ((SHARED_DATA / "Ath_leaf_DNase_chr1_5100001_5300000.tsv"),
      "../plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv"); the CRE test
      ALSO gains (EXAMPLE_DIR / "notebooks" / "plant_helixseek_anno" / "data" /
      "TAIR10_GFF3_chr1_5100001_5300000.gff3",
      "../plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3") for the truth-GTF
      read. (The Anno test needs no cross-notebook file in the final design — its binned-DHS
      track is gone.)
    - Module docstring: note the PNG-mime and pgt zoom/combined pinning. Do NOT add tests pinning
      window coordinates, track colors, or transcript counts — owner display choices; the
      one-liner gates pin the window.
    - Keep ruff clean (line length 100).

    POST-SURGERY VALIDATION (same session, before moving on): reload both files; json.loads
    round-trip; ast.parse every code cell; no source line over 100 chars; cell counts exactly 26
    (CRE) and 32 (Anno); the first code cell is still the D-16 guard; exactly ONE code cell
    contains "alt.Chart(" per notebook, at index 13/19, and its source names "image/png"; exactly
    TWO code cells per notebook reference the pgt CLI ("pgt") and each embeds "image/png", prints
    zoom_window=/combined_window= with the exact window, and contains the window literals 5220000
    and 5260000; no-scoring guards on ALL new figure cells (CRE: no "jaccard"; Anno: no
    "gene_f1"); the confident-band filter markers (">= 0.5" and "pcres_bins_shown=") appear in
    the CRE zoom cell and both combined cells; "--trackLabelFraction" appears NOWHERE; both
    combined cells reference the tsv filename; provenance cells reference the tsv; pyproject
    declares pygenometracks.
  </action>
  <verify>
    <automated>grep -n "pygenometracks" pyproject.toml && .venv/bin/python -m pytest tests -m "not slow" -q --ignore=tests/examples/test_plant_helixseek_showcase.py && .venv/bin/python -c "
import json, ast, pathlib
specs = [
    ('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb', 26, 13, 'jaccard'),
    ('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb', 32, 19, 'gene_f1'),
]
for p, n, fig, banned in specs:
    nb = json.loads(pathlib.Path(p).read_text())
    assert len(nb['cells']) == n, (p, 'cell count', len(nb['cells']))
    srcs = [''.join(c.get('source', [])) for c in nb['cells']]
    assert 'find_spec' in srcs[1] and 'RuntimeError' in srcs[1], (p, 'guard cell changed')
    assert 'Ath_leaf_DNase_chr1_5100001_5300000.tsv' in srcs[0], (p, 'provenance tsv line missing')
    alts = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'alt.Chart(' in ''.join(c.get('source', []))]
    assert alts == [fig], (p, 'altair figure cells', alts)
    assert 'image/png' in srcs[fig], (p, 'altair cell missing png mime')
    pgts = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'pgt' in ''.join(c.get('source', [])) and 'image/png' in ''.join(c.get('source', []))]
    assert len(pgts) == 2, (p, 'pgt figure cells', pgts)
    for i in pgts:
        assert banned not in srcs[i], (p, 'scoring code leaked into new figure cell:', banned)
        assert '5220000' in srcs[i] and '5260000' in srcs[i], (p, 'window literals missing', i)
        assert 'zoom_window=Chr1:5220001-5260000' in srcs[i] or 'combined_window=Chr1:5220001-5260000' in srcs[i], (p, 'window evidence print missing', i)
    for i in pgts:
        if 'bin_scores' in srcs[i] or 'p(CRE)' in srcs[i]:
            assert '>= 0.5' in srcs[i] and 'pcres_bins_shown=' in srcs[i], (p, 'confident-band filter markers missing', i)
    assert any('Ath_leaf_DNase_chr1_5100001_5300000.tsv' in srcs[i] for i in pgts), (p, 'combined cell does not read the tsv')
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        assert '--trackLabelFraction' not in s, (p, 'track titles stripped')
        assert all(len(line) <= 100 for line in s.splitlines()), (p, 'line over 100 chars')
        if c['cell_type'] == 'code' and s.strip():
            ast.parse(s)
print('surgery-ok')" && .venv/bin/ruff check tests/examples/test_plant_helixseek_showcase.py && .venv/bin/ruff format --check tests/examples/test_plant_helixseek_showcase.py && grep -c "Ath_leaf_DNase" tests/examples/test_plant_helixseek_showcase.py</automated>
  </verify>
  <done>pyproject declares pygenometracks with the full fast suite proven green on the downgraded matplotlib BEFORE surgery; both notebooks carry the PNG mime in the altair cell and two pgt-rendered presentation-only figure sections driven by the owner-chosen literal window (26/32 total cells) with the confident-band filter, titles kept, tsv wired into both combined cells plus the provenance line; all cells AST-parse with 100-char lines; guard/altair-only/pgt-pair/no-scoring/no-title-strip invariants hold; tests are extended per the behavior block including the nightly extras (RED on stale outputs is the expected state at task end).</done>
</task>

<task type="auto">
  <name>Task 2: Re-execute both notebooks on the GB10 and gate the committed blobs</name>
  <files>example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb</files>
  <action>
    Run from the repo root (nbclient sets the kernel cwd to each notebook's own directory, so
    ../plant_helixseek_shared/data/, the sibling-notebook data dir, and outputs/ resolve):
    .venv/bin/jupyter execute --inplace --timeout 1200 example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
    .venv/bin/jupyter execute --inplace --timeout 3600 example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
    The timeouts mirror the NOTEBOOK_EXEC_SPECS cell budgets (1200/3600, strictly inside the
    nightly 2400/5400 marks). Expect roughly 7-9 min for CRE (prior ~5-6 plus the sibling
    PlantHelixSeek-Anno slice scan ~2 min including load) and 14-27 min for Anno (prior ~12-25
    plus the sibling CRE window scan ~1 min). Run in the background or sequentially with
    adequate tool timeouts; execution requires the local GPU and regenerates every output
    wholesale (stale Task-1 outputs are fully replaced).

    After each execution, cross-check the exact frozen values: jaccard=0.3247,
    neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59,
    neg_anno_fraction=0.0000 (Anno) — the sibling-model/pgt additions must not perturb the
    existing cells' metrics. The one-liner gates below assert the selection.md band mirrors
    (transient-gate note from 07-02: the authoritative band layer is the parsed _parse_bands()
    test assertions), plus at least one vega and at least three image/png display outputs, the
    zoom_window=/combined_window= lines matching Chr1:5220001-5260000 exactly, and the size
    budget. If any metric drifts OUTSIDE its band, HALT — environment drift; do not adjust
    notebook code or literals to force green. If a pgt subprocess fails inside the kernel
    (non-zero exit / empty track warning), fix the track files and re-execute — never commit a
    figure with an empty-track warning.

    Size check: pgt figures embed a single PNG each — well inside budget; if either notebook
    nonetheless exceeds 2,097,152 bytes, shrink ONLY PNG payloads (vl_convert scale for the
    altair cell, pgt figure height/width for the pgt cells) — never drop a vega mime.

    Confirm the tree: only the two .ipynb modified plus the orchestrator-created uncommitted tsv
    (bedGraphs/GTFs/inis/PNGs land under gitignored outputs/).
  </action>
  <verify>
    <automated>.venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); j=float(re.search(r'^jaccard=([0-9.]+)',s,re.M).group(1)); n=float(re.search(r'^neg_cre_fraction=([0-9.]+)',s,re.M).group(1)); assert 1.00>=j>=0.30, 'jaccard='+str(j); assert 0.05>=n>=0.0, 'neg_cre_fraction='+str(n); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=1, 'vega display outputs='+str(len(vega)); assert len(png)>=3, 'image/png display outputs='+str(len(png)); assert 'zoom_window=Chr1:5220001-5260000' in s and 'combined_window=Chr1:5220001-5260000' in s, 'window evidence lines missing/wrong'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('cre-notebook-ok jaccard='+str(j)+' neg_cre_fraction='+str(n)+' size='+str(p.stat().st_size))" && .venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); g=int(re.search(r'^genes_above_floor=([0-9]+)',s,re.M).group(1)); n=float(re.search(r'^neg_anno_fraction=([0-9.]+)',s,re.M).group(1)); f=float(re.search(r'^exon_f1=([0-9.]+)',s,re.M).group(1)); r=int(re.search(r'^pred_gff3_rows=([0-9]+)',s,re.M).group(1)); assert g>=3, 'genes_above_floor='+str(g); assert 0.10>=n>=0.0, 'neg_anno_fraction='+str(n); assert 1.0>=f>=0.0, 'exon_f1='+str(f); assert r>=1, 'pred_gff3_rows='+str(r); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=1, 'vega display outputs='+str(len(vega)); assert len(png)>=3, 'image/png display outputs='+str(len(png)); assert 'zoom_window=Chr1:5220001-5260000' in s and 'combined_window=Chr1:5220001-5260000' in s, 'window evidence lines missing/wrong'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('anno-notebook-ok genes='+str(g)+' exon_f1='+str(f)+' neg_anno_fraction='+str(n)+' size='+str(p.stat().st_size))" && test -f example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv && .venv/bin/python -c "import subprocess; dirt=subprocess.run(['git','status','--porcelain','example/'],capture_output=True,text=True,check=True).stdout; bad=[l for l in dirt.splitlines() if 'Ath_leaf_DNase' not in l and not l.endswith('.ipynb')]; assert not bad, 'unexpected dirty paths: '+repr(bad); print('tree-ok')"</automated>
  </verify>
  <done>Both one-liner gates print their ok lines with the exact frozen values (cre-notebook-ok jaccard=0.3247 neg_cre_fraction=0.0325; anno-notebook-ok genes=59 exon_f1=0.7522 neg_anno_fraction=0.0), each notebook showing at least one vega and at least three image/png display outputs plus zoom_window=/combined_window=Chr1:5220001-5260000 stream lines, no empty-track warnings in the pgt cell outputs, both sizes within 2,097,152 bytes, and the only example/ changes are the two notebooks plus the orchestrator-created tsv.</done>
</task>

<task type="auto">
  <name>Task 3: Mirror byte-sync (notebooks + tsv), wrapper prose, full gate suite, atomic commit, cross-phase bookkeeping</name>
  <files>docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv, docs/example/notebooks/plant_helixseek_cre.md, docs/example/notebooks/plant_helixseek_anno.md</files>
  <action>
    1. Mirror sync: cp each executed notebook over its docs mirror
       (docs/example/notebooks/plant_helixseek_cre/ and _anno/), and cp
       example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv to
       docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv;
       verify byte-identity with cmp for all three pairs.
    2. Wrapper prose (both .md wrappers): extend the "Full Notebook" paragraph with a short
       passage stating that the full-locus figures embed an image/png mime alongside the compiled
       vega/vega-lite JSON (so local JupyterLab/VS Code render them) and that the notebook adds
       owner-chosen illustrative zoom and combined CRE+Anno views over one window, rendered with
       pygenometracks — CRE confident-band prediction (p(CRE) >= 0.5 bins only), official PlantDHS
       leaf DNase signal (committed per-bin tsv), predicted transcripts, and TAIR10 gene models
       on one aligned axis — for display clarity, with the full-locus metrics remaining the
       authoritative claim. PROSE ONLY — add no code blocks (check_notebook_md_sync AST-matches
       code blocks; neither wrapper quotes the figure cells today; keep it that way).
    3. Full gate suite (all from repo root): fast showcase lane
       (.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q
       → 13 passed, the rekeyed structure tests now GREEN against the executed blobs); the FULL
       fast suite once more WITH the showcase module
       (.venv/bin/python -m pytest tests -m "not slow" -q — the downgrade gate now includes the
       new showcase assertions; must be fully green); python3 scripts/check_notebook_md_sync.py
       still reports exactly the 3 pre-existing stale wrappers and NO plant_helixseek line;
       python3 scripts/validate_docs_snippets.py green; python3 scripts/check_docs_sync.py →
       "OK: docs/example/ is in sync with example/" (the tsv mirror is part of this);
       .venv/bin/ruff check + format --check on the test file.
    4. Atomic commit — ONE commit containing exactly TEN files: pyproject.toml, both example/
       notebooks, both docs notebook mirrors, the tsv (example + docs mirror), both wrappers,
       and the test file. Message:
       "docs(quick-261004): showcase notebooks gain PNG mimes and pygenometracks zoom/combined windows".
       NO attribution trailers of any kind (owner rule). Push to origin phs (owner default:
       commit and push). Do not include unrelated dirty files (.planning/, .scratch/, /tmp).
    5. Cross-phase bookkeeping (into the SUMMARY, this task's output): (a) the GPL-3.0 owner
       override adopting pygenometracks into the notebook extra (2026-10-04, dnallm code never
       imports it); (b) the Phase-8 example-job install notes need the 05-FEASIBILITY CFLAGS
       deviation line for pyBigWig (default sdist build fails on the stock box toolchain);
       (c) the matplotlib pin (pgt 3.9 → matplotlib 3.8.4) and the fast-suite gate that proved
       it harmless.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q && .venv/bin/python -m pytest tests -m "not slow" -q && cmp example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb && cmp example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb && cmp example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv && out=$(python3 scripts/check_notebook_md_sync.py); ! echo "$out" | grep -q plant_helixseek && python3 scripts/validate_docs_snippets.py && dsout=$(python3 scripts/check_docs_sync.py) && echo "$dsout" | grep -q '^OK: docs/example/' && hfiles=$(git show --name-only --format= HEAD) && diff <(printf '%s\n' "$hfiles" | sort) <(printf '%s\n' pyproject.toml example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv docs/example/notebooks/plant_helixseek_cre.md docs/example/notebooks/plant_helixseek_anno.md tests/examples/test_plant_helixseek_showcase.py | sort) && cmsg=$(git log -1 --format=%B) && ! echo "$cmsg" | grep -qiE 'co-authored|generated-with|attribution'</automated>
  </verify>
  <done>13 showcase fast tests pass AND the full fast suite (matplotlib-downgraded environment) is green; all three mirror pairs byte-identical (both notebooks + the tsv); md-sync failure set unchanged (3 pre-existing, none plant_helixseek); snippets and docs-sync green; ruff clean; HEAD is the single atomic commit touching exactly the ten files with no attribution trailers, pushed to origin phs; the SUMMARY records the GPL override, the Phase-8 CFLAGS install note, and the matplotlib-pin gate evidence.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| executed kernel → committed blob | notebook outputs are committed as showcase evidence; edits without re-execution would present stale/forged results |
| external packages → .venv → notebook subprocess | pygenometracks (GPL-3.0) + pyBigWig enter via the notebook extra and pin matplotlib down; dnallm code never imports them |
| external data → committed tsv → notebook panel | the leaf-DNase panel derives from an externally downloaded BigWig; its committed derivative must carry provenance and stay integrity-pinned |
| committed blob → public docs surface | mirrors + wrappers are what readers consume; drift between example/ and docs/ misrepresents evidence |
| owner selection → notebook literals | the zoom window, axis filter, and renderer are human display choices; the notebook must not dress them up as computed optima, and display filtering must be disclosed |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261004-01 | Tampering | showcase notebook committed outputs | high | mitigate | Source edits and re-execution land in one atomic commit; committed-blob one-liner gates assert the executed stream metrics in-band plus ≥1 vega + ≥3 image/png outputs and the exact window lines; nightly lane re-executes and re-asserts parsed bands (extras extended for the new data reads) |
| T-261004-02 | Tampering | 2 MB notebook budget (D-12) | medium | mitigate | Size asserted in both one-liner gates and pinned by the existing structure test; pgt figures embed a single PNG each; documented shrink fallback never drops a vega mime |
| T-261004-03 | Repudiation | illustrative zoom/combined framing (SHOW-07) | medium | mitigate | Captions carry the pinned disclaimer, name the full-locus metrics as authoritative, and DISCLOSE the confident-band display filter (only p(CRE) >= 0.5 bins drawn) and the < 100 bp transcript drop as display choices with the unfiltered full-locus overview referenced; selection tooling and candidate renderings live only in /tmp and .scratch (gitignored); window/combined evidence lines derive from plain literals, never scoring code; the genome-wide denylist test plus the rekeyed caption test enforce this kernel-free |
| T-261004-04 | Tampering | committed leaf-DNase tsv (external-data derivative) | medium | mitigate | Provenance pinned three ways — tsv header comments (source URL, extraction date, coordinate convention), a provenance-cell line in both notebooks, and wrapper prose; integrity via byte-identical docs mirror (cmp) inside the atomic commit; read at runtime like any committed shared data (no network, no pyBigWig in notebooks, no dnallm change) |
| T-261004-05 | Tampering | pygenometracks adoption (GPL-3.0 + matplotlib pin) | medium | mitigate | Owner-explicit override 2026-10-04 recorded in SUMMARY; declared only in the notebook extra — dnallm package code never imports it; matplotlib 3.11.2→3.8.4 downgrade proven harmless by the full fast suite gate run BEFORE surgery and re-run WITH the new assertions at close; CLI invoked PATH-independently as the sys.executable sibling |
| T-261004-SC | Tampering | package installs (pygenometracks + pyBigWig) | high | mitigate | Owner-adopted and live-verified this session (pgt 3.9 + pyBigWig 0.3.26 installed, all five track types render from real data, zero empty-track warnings); fresh-env installs documented via the 05-FEASIBILITY CFLAGS deviation; the Phase-8 example-job install note recorded in SUMMARY — no undocumented install path exists |
</threat_model>

<verification>
- Task 1: pyproject declares pygenometracks; full fast suite green on the downgraded matplotlib BEFORE surgery (showcase module excluded from the post-surgery variant only because its extended tests are expected-RED); surgery validation — both notebooks JSON-valid, every code cell AST-parses, 100-char lines, cell counts 26/32, exactly one altair figure cell per notebook (index 13/19) carrying image/png, exactly two pgt figure cells per notebook with the exact window literals + evidence prints, no-scoring guards on all new figure cells, confident-band filter markers, no --trackLabelFraction anywhere, tsv wired into both combined cells + provenance cells, nightly extras extended.
- Task 2 committed-blob gates: CRE (jaccard in [0.30, 1.00] observed 0.3247; neg_cre_fraction in [0.00, 0.05] observed 0.0325) and Anno (genes >= 3 observed 59; neg_anno_fraction in [0.00, 0.10] observed 0.0000; exon_f1 observed 0.7522; pred_gff3_rows >= 1), each plus ≥1 vega + ≥3 image/png display outputs, zoom_window= and combined_window= lines exactly Chr1:5220001-5260000, ≤ 2,097,152 bytes, guard intact, tsv present, tree clean outside the notebooks + tsv, no empty-track warnings.
- Task 3 full suite: 13 showcase fast tests green AND the full fast suite green; three cmp-identical mirror pairs; md-sync failure set unchanged; snippets and docs-sync green; ruff clean; single atomic ten-file commit, no attribution trailers, pushed; SUMMARY records the GPL override + Phase-8 CFLAGS note + matplotlib-pin evidence.
</verification>

<success_criteria>
- The full-locus altair figures render everywhere (image/png added; vega/vegalite retained for GitHub), and the new pgt zoom + combined figures embed PNG with titles on and the exact owner window.
- The combined view aligns the CRE confident-band prediction, official leaf DNase signal (committed, provenance-pinned tsv), predicted transcripts (sibling-model presentation decode), and TAIR10 gene models (deterministic GTF conversion) on one genomic axis in both notebooks, matching the approved v4 render; no binned DHS track; no empty-track warnings.
- The zoom window and every display filter are owner choices rendered as literals with disclosed captions (SHOW-07); no selection or scoring code entered the notebooks; the full-locus metrics remain the authoritative assertions.
- Re-execution reproduces jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000; both notebooks within 2,097,152 bytes; the matplotlib downgrade is proven harmless by the full fast suite (run before surgery and at close).
- pygenometracks>=3.9 declared in the notebook extra with the GPL override, the Phase-8 pyBigWig CFLAGS install note, and the pin evidence recorded in the SUMMARY; docs mirrors (notebooks + tsv) byte-identical; single atomic ten-file commit without attribution trailers on origin phs.
</success_criteria>

<output>
Create `.planning/quick/261004-dyw-planthelixseek-showcase-notebook-vega-ve/261004-dyw-SUMMARY.md` when done. The SUMMARY MUST record (owner decisions, cross-phase bookkeeping): (1) the GPL-3.0 override adopting pygenometracks into the notebook extra (2026-10-04; dnallm code never imports it; example-notebook use only); (2) the Phase-8 example-job install notes need the 05-FEASIBILITY CFLAGS deviation line for pyBigWig; (3) the pgt 3.9 matplotlib pin (3.11.2 → 3.8.4) and the two full-fast-suite gate runs that proved it harmless.
</output>
