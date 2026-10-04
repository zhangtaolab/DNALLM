---
phase: 261004-dyw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
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
  tokens: 40000
  raw_tokens: 40000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - Every figure output in both committed showcase notebooks (full-locus, zoom, combined) carries an image/png mime (vl_convert PNG export) alongside the existing vega v6 object + vegalite v6 JSON string — the figures render in local JupyterLab and VS Code, not only in the GitHub viewer.
    - ALL new figures use the owner-selected window Chr1:5220001-5260000 (0-based half-open 5220000-5260000, 40 kb) as plain literals — recorded in /tmp/planthelixseek_zoom_candidates/SELECTION.txt (cre_zoom/anno_zoom/combined_zoom all this window). Zero window-selection/scoring code in the notebooks: no "jaccard" string in any CRE new-figure cell, no "gene_f1" string in any Anno new-figure cell, no best/optimal claims anywhere.
    - Each notebook gains a per-modality zoom figure AND a combined CRE+Anno figure over the window: one vconcat with ALL panels sharing one aligned genomic x-axis (resolve_scale(x="shared"), single bottom axis) — CRE predicted p(CRE) track (blue) + mean+1.5σ dashed reference, official PlantDHS leaf DNase signal (dark red #8c2d04 area), binned DHS truth (#ff7f0e rects), per-strand Anno predicted-CDS strips (blue), and TAIR10 gene-model lanes (grey intron backbone rules, green + / red − exon blocks, ▶/◀ 3' direction text markers) — matching the owner-approved reference render .scratch/zoom-candidates/combined-1-final.png (159/800 window bins drawn there — the confident-band filter in action).
    - Each notebook computes its SIBLING model's prediction on the fixed window (CRE notebook runs PlantHelixSeek-Anno over the 40 kb window with the frozen 8192/4096 stitching + argmax BILOU decode; Anno notebook runs PlantHelixSeek-CRE 500/50/50 on it) — presentation on a fixed literal window, both models already in models.lock with warm cache.
    - In ZOOM and COMBINED figures, sub-0.5 p(CRE) bins are NOT DRAWN AT ALL (owner final display decision — confident band only): bins are filtered to score >= 0.5 and rendered as mark_rect bars from the 0.5 axis floor up to the score (x=start, x2=start+50) with y domain [0.5, 1] — genuinely empty gaps below threshold, not clipped-to-baseline area — and the mean+1.5σ dashed line still overlays; captions DISCLOSE the display filter ("only p(CRE) >= 0.5 bins shown — confident band; the full-locus overview above shows the unfiltered track"); the full-locus CRE figure (cell 13) stays unfiltered on its current axis, gaining only the image/png mime.
    - The leaf-DNase panel reads the committed example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv like any other shared data — NO pyBigWig import, NO runtime network, NO new dependency; the provenance cell of both notebooks gains one line citing the extraction (official BigWig → committed per-bin tsv, internal preparation step, not re-done in-notebook).
    - Re-execution on the GB10 reproduces the frozen metrics exactly: jaccard=0.3247 and neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000 (Anno).
    - Both committed notebooks stay at most 2,097,152 bytes (D-12) — CRE starts at 1.16 MB and gains a full-locus PNG (~200-400 KB) plus small window payloads; sizes are gated, with the documented PNG-scale fallback.
    - The fast showcase structure lane passes (13 tests) including the extended assertions (three vega-bearing and three image/png display outputs, disclaimer captions after EVERY figure cell); the SHOW-07 denylist stays green; the nightly slow tests' sandbox extras now seed the tsv and the cross-notebook truth files so the lane keeps passing; docs mirrors stay byte-identical (notebooks AND the tsv); check_notebook_md_sync / validate_docs_snippets / check_docs_sync stay exactly as green as baseline.
    - Notebooks + mirrors + tsv + wrappers + test file land in ONE atomic commit with no attribution trailers.
  artifacts:
    - example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv — committed (created by the orchestrator, 59,818 bytes, 4000×50 bp bins, header comments carrying source URL + extraction date 2026-10-04 + 0-based-half-open convention); mirrored byte-identically under docs/.
    - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb — 26 cells (20 + 6): figure cell 13 gains image/png; new zoom section and combined-figure section (each markdown header + code cell + markdown caption) after caption cell 14; provenance line for the tsv.
    - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb — 32 cells (26 + 6): figure cell 19 gains image/png; new zoom section and combined-figure section after caption cell 20; provenance line for the tsv.
    - tests/examples/test_plant_helixseek_showcase.py — structure tests extended to three vega + three image/png outputs and captions after every figure cell; both slow tests' extras lists gain the tsv and the cross-notebook truth file tuples.
    - docs/example/notebooks/plant_helixseek_{cre,anno}/plant_helixseek_{cre,anno}.ipynb and docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv — byte-identical mirrors.
    - docs/example/notebooks/plant_helixseek_{cre,anno}.md — one added prose passage each (PNG mimes, zoom + combined figures, leaf-DNase panel), no new code blocks.
  key_links:
    - Owner selection (SELECTION.txt, already written) → plain literals 5220000-5260000 in every new figure cell of both notebooks → zoom_window=/combined_window= evidence lines printing exactly Chr1:5220001-5260000 — the only zoom data the committed notebooks derive.
    - Figure-cell display dict → 4-mime bundle (vega v6 object + vegalite v6 JSON string + image/png base64 + text/plain) via base64.b64encode(vlc.vegalite_to_png(vegalite_spec)).decode("ascii") — vl_convert 1.9.0 verified present in .venv.
    - Sibling model loads via load_model_and_tokenizer with the packaged registry entry (same pattern as each notebook's own load cell; both models in models.lock, warm cache) → fixed-window inference → combined-figure panels; combined x-axis alignment via resolve_scale(x="shared") across all five panels.
    - Committed tsv (example + docs mirrors) → leaf-DNase panel in both notebooks; cross-notebook truth reads (CRE reads ../plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3; Anno reads ../plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff) seeded into the nightly sandboxes via the slow tests' extras.
    - Committed blobs → structure tests + the 07-01/07-02-style one-liner gates (three vega + three image/png outputs, zoom_window=/combined_window= exact match, bands, size, guard).
---

<objective>
Enhance the two PlantHelixSeek showcase notebooks' result display (owner request; revised twice
mid-planning — zoom-window selection is an internal owner-driven process, and a combined CRE+Anno
figure was added over the owner-selected window):

1. Every figure output gains an image/png mime (vl_convert PNG export) next to the existing
   vega v6 object + vegalite v6 JSON string — GitHub renders the vega object, but local JupyterLab
   and VS Code do not; the PNG mime makes the figures render everywhere.
2. The owner has ALREADY selected the zoom window from rendered candidates (candidate gallery
   phase complete, orchestrator-side): Chr1:5220001-5260000 for every new figure in both
   notebooks. The notebooks contain ONLY presentation: literal coordinates, no selection or
   scoring code.
3. Each notebook gains a per-modality zoom figure AND a combined CRE+Anno figure over the window —
   five panels on one aligned genomic x-axis (p(CRE) + mean+1.5σ dashed, official leaf DNase
   signal from a newly committed tsv, binned DHS truth, per-strand predicted-CDS strips, TAIR10
   gene-model lanes with 3' direction markers), matching the owner-approved reference render
   .scratch/zoom-candidates/combined-1-final.png. Each notebook computes its SIBLING
   model's prediction on the fixed window. In the zoom/combined figures, only p(CRE) >= 0.5 bins
   are drawn (mark_rect bars from the 0.5 floor, y domain [0.5, 1], mean+1.5σ dashed overlay),
   disclosed in the captions; the full-locus CRE figure stays unfiltered.
4. Both notebooks are re-executed on the GB10 and must reproduce the frozen metrics exactly
   (jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59,
   neg_anno_fraction=0.0000).
5. Notebooks + tsv byte-synced to docs mirrors, wrappers updated (prose only), 2 MB budget held,
   fast structure lane + nightly extras + docs-sync gates green, everything in one atomic commit
   with no attribution trailers.

Purpose: the showcase notebooks are the public evidence surface for the PlantHelixSeek work
(Phases 6-7); today their figures are invisible to anyone opening them locally, and no single
view aligns CRE signal, open-chromatin truth, and gene structure on one axis.
Output: re-executed notebooks with dual-render figures, owner-chosen zoom + combined windows,
synced mirrors/wrappers/tsv, extended kernel-free structure tests.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Verified-live facts this plan encodes (do not re-derive):
- SELECTION COMPLETE: /tmp/planthelixseek_zoom_candidates/SELECTION.txt contains cre_zoom=,
  anno_zoom=, combined_zoom= all Chr1:5220001-5260000. The candidate-gallery task from the prior
  plan revision is DONE (orchestrator rendered candidates; owner picked). No human gates remain.
- Owner-approved reference render for the combined figure (consult it; transient, gitignored):
  .scratch/zoom-candidates/combined-1-final.png — panel order top→bottom: CRE predicted p(CRE)
  as mark_rect bars over the >= 0.5 confident band only (y domain [0.5, 1]; 159/800 window bins
  drawn there) + mean+1.5σ dashed rule; leaf DNase signal as
  dark red #8c2d04 area; binned DHS truth as #ff7f0e rects; per-strand predicted-CDS strips
  (blue, +/− lanes); TAIR10 gene-model lanes (grey #999999 intron backbone rules, green +
  strand / red − strand exon blocks, ▶/◀ 3' direction text markers); all panels share one
  aligned x-axis with a single bottom axis.
- New committed data artifact (created by the orchestrator, currently UNCOMMITTED):
  example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv —
  59,818 bytes, header comment lines (# source: https://plantdhs.org/static/download/Ath_leaf_DNase.bw
  extracted 2026-10-04, 0-based half-open 50 bp bins aligned to the CRE scan grid) then a
  start/signal TSV over the whole 200 kb locus (4000 bins). Read with pandas, skipping the #
  comment header. It must be mirrored byte-identically to
  docs/example/notebooks/plant_helixseek_shared/data/ (mirror dir exists, already holds
  selection.md + negative controls).
- CRE notebook: 20 cells, figure = code cell 13, caption = markdown cell 14. Anno: 26 cells,
  figure = code cell 19, caption = markdown cell 20. Each has one display_data with vega-v6-object
  + vegalite-v6-STRING mimes. Figure cells already import json/altair/pandas/vl_convert and end
  with display({...}, raw=True) built from vegalite_spec/vega_spec.
- Per-notebook data dirs (these ride along with the notebook dir in sandboxes):
  plant_helixseek_cre/data/{chr1_5100001_5300000.fas, TAIR10_DHSs_chr1_5100001_5300000.gff};
  plant_helixseek_anno/data/{chr1_5100001_5300000.fas, TAIR10_GFF3_chr1_5100001_5300000.gff3}.
  Cross-notebook reads therefore use ../plant_helixseek_<sibling>/data/... paths AND must be
  seeded as extras in the nightly slow tests (they seed only the notebook dir + explicit tuples
  into ../plant_helixseek_shared/data/ today).
- Presentation state at insert points — CRE: pred_df/truth_df (cell 13), bin_scores, bin_width,
  genomic_offset, threshold (cell 9 locus-level mean+1.5σ); Anno: pred_track_df, span_records,
  truth_cds_df, gene_orders (cell 19), predicted_segments, genomic_offset, locus_length.
- Sibling models: both already in models.lock with warm cache (owner-verified); load through
  load_model_and_tokenizer with the packaged registry entry exactly as each notebook's own load
  cell does (TaskConfig task_type/num_labels/label_names/threshold from model_info.yaml; Anno
  sibling = 17 BILOU labels; CRE sibling = binary CRE). fla guard already present in both
  notebooks — never import the fla module directly (D-16).
- .venv/bin/jupyter execute --inplace --timeout N exists; nbclient sets the kernel cwd to the
  notebook's own directory so relative data paths and outputs/ resolve (proven 07-01).
  NOTEBOOK_EXEC_SPECS budgets: CRE cell_timeout 1200, Anno 3600 (tests/examples/_execution.py:196-220);
  the new sibling cells (~1-2 min each) fit well inside those budgets.
- Wrappers docs/example/notebooks/plant_helixseek_{cre,anno}.md do NOT quote the figure cells —
  editing figure cells and inserting cells cannot break the AST sync check.
- Baselines: check_notebook_md_sync fails on exactly 3 pre-existing stale wrappers
  (mcp_langchain, mcp_pydantic_ai, data_prepare_finetune — none plant_helixseek);
  validate_docs_snippets green; check_docs_sync "OK". outputs/ is gitignored (.gitignore:63);
  .scratch/ is ignored (verified via check-ignore).
- Owner rules: zoom selection stays internal (done); atomic commit (notebooks + mirrors + tsv +
  wrappers together), no attribution trailers, execution on the local GB10.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Notebook source surgery — PNG mimes, zoom + combined figure sections on the selected window, tsv wiring, provenance line — and test extension</name>
  <files>example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, tests/examples/test_plant_helixseek_showcase.py</files>
  <behavior>
    - Structure test: both committed notebooks expose at least three image/png mimes across display outputs (named-cause: local JupyterLab/VS Code rendering).
    - Structure test: both committed notebooks expose at least THREE display_data outputs carrying an application/vnd.vega mime (full-locus + zoom + combined).
    - Structure test: EVERY code cell containing "alt.Chart(" is followed by a markdown cell whose text (lowercased) carries the pinned disclaimer "illustrative locus, not genome-wide accuracy".
    - Nightly lane keeps passing: both slow tests' extras seed the new tsv and the cross-notebook truth file.
    - These go RED against the surgically edited (not yet re-executed) blobs — expected transient; they turn GREEN only after Task 2's re-execution. Do not weaken them to pass early.
  </behavior>
  <action>
    MECHANICS — the Edit tool refuses .ipynb: do all notebook edits in one throwaway .venv/bin/python
    session using nbformat (nbformat.read(path, as_version=4), exact single-match string anchors on
    cell sources, nbformat.v4.new_markdown_cell / new_code_cell for inserts, nbformat.write). Keep
    the existing serialization style. Do NOT execute anything in this task — Task 2 regenerates all
    outputs wholesale. Consult the owner-approved reference render
    .scratch/zoom-candidates/combined-1-final.png while authoring the combined cells.
    The chosen window literals throughout: 0-based half-open zoom_start=5220000, zoom_end=5260000
    (display form Chr1:5220001-5260000, 40 kb). A brief comment in each new figure cell names the
    window as an owner pick from candidate renderings, not a notebook-computed optimum.

    EDIT 1 (both notebooks, existing figure cell — CRE cell 13, Anno cell 19): add import base64
    to the cell's import block, and add one entry to the display dict after the vegalite entry and
    before text/plain: "image/png" mapped to base64.b64encode(vlc.vegalite_to_png(vegalite_spec))
    .decode("ascii"). Extend the embed comment: the PNG mime exists because local JupyterLab/VS
    Code do not render the vega v6 object + bare .json string combination (GitHub does). The
    full-locus figures otherwise stay exactly as they are (cell 13 keeps its current y axis).

    EDIT 2 (CRE — per-modality zoom section, 3 cells inserted after caption cell 14):
    a. Markdown header "## Illustrative zoom window" — one sentence: shown for display clarity;
       the full-locus metrics above remain the authoritative claim.
    b. Code cell, PRESENTATION ONLY: window as plain literals; slice pred_df/truth_df to the
       window; build zoom_track_chart in the SAME two-track vconcat shape as cell 13 (blue
       #4c78a8 p(CRE) panel on top, #f58518 truth rects below), x scale domain pinned to the
       window — EXCEPT the p(CRE) panel applies the owner's confident-band display filter:
       filter window bins to score >= 0.5 and render them as mark_rect bars from the 0.5 floor
       up to the score (x=start, x2=start+50) with y domain [0.5, 1] — sub-0.5 bins are NOT
       drawn at all (genuinely empty gaps) — with the mean+1.5σ dashed reference overlaying.
       Print exactly zoom_window=Chr1:5220001-5260000, zoom_width=40000, and the descriptive
       count pcres_bins_shown=<n> (all from the literals/data; the reference render drew 159
       bins in this window — eyeball cross-check only, never a gate literal). Embed via
       display(..., raw=True) with ALL FOUR mimes (vega v6 object, vegalite v6 JSON string,
       image/png base64, text/plain naming it as the zoom view). ZERO scoring code — the string
       "jaccard" must not appear in this cell's source.
    c. Markdown caption: the pinned sentence "Illustrative locus, not genome-wide accuracy." plus
       wording that the window was chosen for display clarity and the full-locus jaccard above
       remains the authoritative claim, plus the display-filter disclosure ("only p(CRE) >= 0.5
       bins shown — confident band; the full-locus overview above shows the unfiltered track").
       Any line mentioning genome-wide must contain "not genome-wide" (SHOW-07).

    EDIT 3 (Anno — per-modality zoom section, 3 cells inserted after caption cell 20): same
    literal-window pattern for the gene-model view — filter pred_track_df, span_records,
    truth_cds_df to entries overlapping the window (LOCAL zoom panel helper, lane order re-sorted
    by start; never mutate cell-19 state), zoom_gene_model_chart in the cell-19 vconcat shape with
    x domain pinned to the window. Print zoom_window=Chr1:5220001-5260000 (plus zoom_width=40000
    and a descriptive zoom_gene_count= if useful — counts only). Embed with all four mimes. ZERO
    scoring code — "gene_f1" and "EXON_F1_FLOOR" must not appear in this cell's source. Caption:
    pinned disclaimer + chosen-for-display-clarity wording, SHOW-07-compliant, no best claims.

    EDIT 4 (CRE — combined CRE+Anno section, 3 cells after the zoom caption):
    a. Markdown header "## Combined CRE + Anno view" — one sentence: one window, both modalities
       and their truths on a single aligned genomic axis, for display clarity; full-locus metrics
       above remain authoritative.
    b. Code cell, PRESENTATION ON A FIXED LITERAL WINDOW: read the committed leaf-DNase tsv
       (Path("../plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv"), pandas
       with the # comment header skipped — same shared-data read pattern as the other files); read
       TAIR10 gene models from ../plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
       (transcribe the GFF parse pattern the Anno notebook uses, CDS rows of mRNA parents,
       gff1_to_half_open); load the PlantHelixSeek-Anno SIBLING through load_model_and_tokenizer
       with its registry entry (17 BILOU labels, same load shape as the notebook's own model
       cell), run it over the 40 kb window under torch.no_grad() with the frozen 8192/4096
       stitching plan and decode predicted CDS segments with the frozen argmax BILOU rule
       (transcribe both rules from the Anno notebook's scan/decode cells). Build the combined
       chart as ONE alt.vconcat of five panels, all sharing the genomic x axis
       (resolve_scale(x="shared"), only the bottom panel draws the x axis): (1) CRE predicted
       p(CRE) panel sliced from bin_scores with the confident-band display filter — mark_rect
       bars only for score >= 0.5 bins, drawn from the 0.5 floor up to the score (x=start,
       x2=start+50), y domain [0.5, 1], mean+1.5σ dashed reference overlaying, and a descriptive
       pcres_bins_shown=<n> count printed; (2) leaf DNase signal as #8c2d04 area from
       the tsv rows inside the window; (3) binned DHS truth as #ff7f0e rects (truth_df window
       slice); (4) per-strand predicted-CDS strips (blue rects, + and − lanes, from the sibling
       decode); (5) TAIR10 gene-model lanes for genes overlapping the window — grey #999999
       intron backbone rules, exon blocks colored green (+ strand) / red (− strand), and ▶/◀
       text markers at the 3' end per strand. Print combined_window=Chr1:5220001-5260000 (plus
       descriptive counts like combined_pred_cds= if useful). Embed with all four mimes. ZERO
       scoring code — "jaccard" must not appear in this cell's source.
    c. Markdown caption: pinned disclaimer + this-is-a-display-view wording + the display-filter
       disclosure ("only p(CRE) >= 0.5 bins shown — confident band; the full-locus overview above
       shows the unfiltered track") + the leaf-DNase provenance pointer
       (official PlantDHS BigWig, committed per-bin tsv — see the provenance cell); full-locus
       metrics remain the authoritative claim; SHOW-07-compliant; no best claims.

    EDIT 5 (Anno — combined CRE+Anno section, 3 cells after the zoom caption): mirror image of
    EDIT 4 with the sibling direction flipped — load the PlantHelixSeek-CRE sibling (binary CRE
    registry entry), run the 500/50/50 scan on the fixed 40 kb window (transcribe the CRE
    notebook's scan parameters), giving per-bin p(CRE) scores for panel 1 with the SAME
    confident-band display filter (mark_rect bars for score >= 0.5 only, from the 0.5 floor,
    y domain [0.5, 1], mean+1.5σ dashed overlay, pcres_bins_shown= printed); read the same
    committed tsv for panel 2; read
    binned DHS truth from ../plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
    (transcribe the CRE notebook's GFF read + per-bin truth-coverage pattern); panels 4-5 from
    the notebook's OWN predicted_segments and truth gene models (window slice). Same five-panel
    vconcat, same palette, same literals; print combined_window=Chr1:5220001-5260000. Embed with
    all four mimes. ZERO scoring code — "gene_f1" must not appear in this cell's source. Caption
    as in EDIT 4c.

    EDIT 6 (both notebooks, provenance markdown cell 0): append ONE line citing the tsv
    extraction — official PlantDHS leaf DNase BigWig
    (https://plantdhs.org/static/download/Ath_leaf_DNase.bw) extracted 2026-10-04 to the
    committed per-bin tsv in plant_helixseek_shared/data/ (0-based half-open, 50 bp bins aligned
    to the CRE scan grid); extraction is an internal preparation step, not re-done in-notebook.
    Keep every existing provenance line intact (the structure test pins the selection.md link,
    the locus line, and the illustrative-loci phrase).

    CONSTRAINTS on every new/edited code cell: import only modules already exercised in these
    notebooks plus base64 (json, altair, pandas, vl_convert, numpy, torch, pathlib, re, and the
    dnallm public route) — the fast-lane import-exec test and the D-16 no-fla-import rule stay
    satisfied; never name the fla module. No metric asserts anywhere new (nightly re-execution
    safety). Every source line at most 100 chars.

    EDIT 7 (tests/examples/test_plant_helixseek_showcase.py):
    - test_committed_notebook_has_executed_outputs: collect display_data outputs; assert at
      least THREE carry "image/png" in their data (named-cause: local JupyterLab/VS Code
      rendering) and at least THREE carry an application/vnd.vega mime (full-locus + zoom +
      combined). Update the docstring.
    - test_illustrative_caption_follows_the_metric_figure: generalize from the first alt.Chart
      cell to EVERY code cell containing "alt.Chart(" (expect at least 3 per notebook) — the
      first markdown after EACH must carry ILLUSTRATIVE_DISCLAIMER. Update the docstring.
    - BOTH slow tests' extras lists gain two tuples each: the tsv
      ((SHARED_DATA / "Ath_leaf_DNase_chr1_5100001_5300000.tsv"),
      "../plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv") and the
      cross-notebook truth file — CRE test:
      (EXAMPLE_DIR / "notebooks" / "plant_helixseek_anno" / "data" / "TAIR10_GFF3_chr1_5100001_5300000.gff3",
      "../plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3"); Anno test:
      (EXAMPLE_DIR / "notebooks" / "plant_helixseek_cre" / "data" / "TAIR10_DHSs_chr1_5100001_5300000.gff",
      "../plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff"). Without these the
      nightly sandbox lacks the files the combined cells read and the lane fails.
    - Module docstring: note the PNG-mime, zoom + combined figure pinning. Do NOT add any test
      that pins the window coordinates or panel colors — those are owner display choices; the
      one-liner gates pin the window.
    - Keep ruff clean (line length 100).

    POST-SURGERY VALIDATION (same session, before moving on): reload both files; json.loads
    round-trip; ast.parse every code cell; no source line over 100 chars; cell counts exactly 26
    (CRE) and 32 (Anno); the first code cell is still the D-16 guard; exactly three code cells
    contain "alt.Chart(" per notebook, the FIRST being the pre-existing full-locus figure (CRE
    index 13, Anno index 19 — inserts land strictly after those indices); each of the three
    figure cells' sources names "image/png"; the new figure cells print zoom_window= and
    combined_window= with the exact window; the no-scoring guards hold on BOTH new figure cells
    (CRE: no "jaccard"; Anno: no "gene_f1"); the confident-band display filter appears in every
    new p(CRE)-bearing cell (>= 0.5 filter + domain=[0.5, 1] + pcres_bins_shown= print) and the
    caption markdowns following those cells disclose the filter; both combined cells reference
    the tsv filename.
  </action>
  <verify>
    <automated>.venv/bin/python -c "
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
    figs = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'alt.Chart(' in ''.join(c.get('source', []))]
    assert len(figs) == 3 and figs[0] == fig, (p, 'figure cells', figs)
    assert all('image/png' in srcs[i] for i in figs), (p, 'png mime missing in a figure cell')
    for i in figs[1:]:
        assert banned not in srcs[i], (p, 'scoring code leaked into new figure cell:', banned)
        assert 'zoom_window=Chr1:5220001-5260000' in srcs[i] or 'combined_window=Chr1:5220001-5260000' in srcs[i], (p, 'window literal print missing', i)
        assert '5220000' in srcs[i] and '5260000' in srcs[i], (p, 'window literals missing', i)
        if 'p(CRE)' in srcs[i] and 'alt.Y' in srcs[i]:
            assert 'domain=[0.5, 1]' in srcs[i], (p, 'p(CRE) axis floor missing', i)
            assert '>= 0.5' in srcs[i], (p, 'confident-band filter missing', i)
            assert 'pcres_bins_shown=' in srcs[i], (p, 'bins-shown evidence print missing', i)
    assert 'Ath_leaf_DNase_chr1_5100001_5300000.tsv' in srcs[figs[2]], (p, 'combined cell does not read the tsv')
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        assert all(len(line) <= 100 for line in s.splitlines()), (p, 'line over 100 chars')
        if c['cell_type'] == 'code' and s.strip():
            ast.parse(s)
print('surgery-ok')" && .venv/bin/ruff check tests/examples/test_plant_helixseek_showcase.py && .venv/bin/ruff format --check tests/examples/test_plant_helixseek_showcase.py && grep -n "Ath_leaf_DNase" tests/examples/test_plant_helixseek_showcase.py | head -4</automated>
  </verify>
  <done>Both notebooks carry the PNG mime in all three figure cells, a presentation-only zoom section and a combined five-panel figure section driven by the owner-chosen literal window (26/32 total cells), the tsv wired into both combined cells plus the provenance line, all cells AST-parse with 100-char lines, guard/figure-ordering/no-scoring/confident-band-filter invariants hold, and the tests are extended per the behavior block including the nightly extras (RED on stale outputs is the expected state at task end).</done>
</task>

<task type="auto">
  <name>Task 2: Re-execute both notebooks on the GB10 and gate the committed blobs</name>
  <files>example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb</files>
  <action>
    Run from the repo root (nbclient sets the kernel cwd to each notebook's own directory, so
    ../plant_helixseek_shared/data/, the sibling-notebook data dirs, and outputs/ resolve):
    .venv/bin/jupyter execute --inplace --timeout 1200 example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
    .venv/bin/jupyter execute --inplace --timeout 3600 example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
    The timeouts mirror the NOTEBOOK_EXEC_SPECS cell budgets (1200/3600, strictly inside the
    nightly 2400/5400 marks). Expect roughly 6-8 min for CRE and 14-27 min for Anno on the GB10 —
    the prior budgets (~5-6 / ~12-25 min) plus one sibling model load + fixed-window inference
    (~1-2 min each). Run in the background or sequentially with adequate tool timeouts; execution
    requires the local GPU and regenerates every output wholesale (stale Task-1 outputs are fully
    replaced).

    After each execution, cross-check the exact frozen values: jaccard=0.3247,
    neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59,
    neg_anno_fraction=0.0000 (Anno) — the sibling-model additions must not perturb the existing
    cells' metrics. The one-liner gates below assert the selection.md band mirrors
    (transient-gate note from 07-02: the literals are a pre-commit mirror; the authoritative band
    layer is the parsed _parse_bands() test assertions), plus three vega + three image/png
    display outputs, the zoom_window=/combined_window= lines matching Chr1:5220001-5260000
    exactly, and the size budget. If any metric drifts OUTSIDE its band, HALT — that is
    environment drift; do not adjust notebook code or literals to force green.

    Size fallback (watch CRE — it starts at 1.16 MB and gains a full-locus PNG plus window
    payloads): if either notebook exceeds 2,097,152 bytes, shrink ONLY the PNG payloads (e.g. a
    smaller scale= argument to vlc.vegalite_to_png) — never drop a vega mime — then re-execute
    that notebook and re-gate.

    Confirm git status --porcelain example/ shows only the two .ipynb modified plus the
    orchestrator-created uncommitted tsv (outputs/ artifacts land under the global gitignore
    line 63; the tsv is committed in Task 3).
  </action>
  <verify>
    <automated>.venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); j=float(re.search(r'^jaccard=([0-9.]+)',s,re.M).group(1)); n=float(re.search(r'^neg_cre_fraction=([0-9.]+)',s,re.M).group(1)); assert 1.00>=j>=0.30, 'jaccard='+str(j); assert 0.05>=n>=0.0, 'neg_cre_fraction='+str(n); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=3, 'vega display outputs='+str(len(vega)); assert len(png)>=3, 'image/png display outputs='+str(len(png)); assert 'zoom_window=Chr1:5220001-5260000' in s and 'combined_window=Chr1:5220001-5260000' in s, 'window evidence lines missing/wrong'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('cre-notebook-ok jaccard='+str(j)+' neg_cre_fraction='+str(n)+' size='+str(p.stat().st_size))" && .venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); g=int(re.search(r'^genes_above_floor=([0-9]+)',s,re.M).group(1)); n=float(re.search(r'^neg_anno_fraction=([0-9.]+)',s,re.M).group(1)); f=float(re.search(r'^exon_f1=([0-9.]+)',s,re.M).group(1)); r=int(re.search(r'^pred_gff3_rows=([0-9]+)',s,re.M).group(1)); assert g>=3, 'genes_above_floor='+str(g); assert 0.10>=n>=0.0, 'neg_anno_fraction='+str(n); assert 1.0>=f>=0.0, 'exon_f1='+str(f); assert r>=1, 'pred_gff3_rows='+str(r); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=3, 'vega display outputs='+str(len(vega)); assert len(png)>=3, 'image/png display outputs='+str(len(png)); assert 'zoom_window=Chr1:5220001-5260000' in s and 'combined_window=Chr1:5220001-5260000' in s, 'window evidence lines missing/wrong'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('anno-notebook-ok genes='+str(g)+' exon_f1='+str(f)+' neg_anno_fraction='+str(n)+' size='+str(p.stat().st_size))" && test -f example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv && .venv/bin/python -c "import subprocess; dirt=subprocess.run(['git','status','--porcelain','example/'],capture_output=True,text=True,check=True).stdout; bad=[l for l in dirt.splitlines() if 'Ath_leaf_DNase' not in l and not l.endswith('.ipynb')]; assert not bad, 'unexpected dirty paths: '+repr(bad); print('tree-ok')"
  </verify>
  <done>Both one-liner gates print their ok lines with the exact frozen values (cre-notebook-ok jaccard=0.3247 neg_cre_fraction=0.0325; anno-notebook-ok genes=59 exon_f1=0.7522 neg_anno_fraction=0.0), each notebook showing three vega and three image/png display outputs plus zoom_window=/combined_window=Chr1:5220001-5260000 stream lines, both sizes within 2,097,152 bytes, and the only example/ changes are the two notebooks plus the orchestrator-created tsv.</done>
</task>

<task type="auto">
  <name>Task 3: Mirror byte-sync (notebooks + tsv), wrapper prose, full gate suite, atomic commit</name>
  <files>docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv, docs/example/notebooks/plant_helixseek_cre.md, docs/example/notebooks/plant_helixseek_anno.md</files>
  <action>
    1. Mirror sync: cp each executed notebook over its docs mirror
       (docs/example/notebooks/plant_helixseek_cre/ and _anno/), and cp
       example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv to
       docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv;
       verify byte-identity with cmp for all three pairs.
    2. Wrapper prose (both .md wrappers): extend the "Full Notebook" paragraph with a short
       passage stating that the embedded figures now carry an image/png mime alongside the
       compiled vega/vega-lite JSON (so local JupyterLab/VS Code render them) and that the
       notebook adds an owner-chosen illustrative zoom window plus a combined CRE+Anno view over
       that window — CRE prediction, official PlantDHS leaf DNase signal (committed per-bin tsv),
       binned DHS truth, predicted CDS strips, and TAIR10 gene models on one aligned axis,
       drawing only the confident p(CRE) band (bins >= 0.5) in the zoom/combined views — for
       display clarity, with the full-locus metrics remaining the authoritative claim. PROSE
       ONLY — add no code blocks: the AST sync check (check_notebook_md_sync) matches code blocks
       against notebook cells, and neither wrapper quotes the figure cells today; keep it that
       way.
    3. Full gate suite (all from repo root): fast showcase lane
       (.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q
       → 13 passed, the extended structure tests now GREEN against the executed blobs);
       python3 scripts/check_notebook_md_sync.py still reports exactly the 3 pre-existing stale
       wrappers and NO plant_helixseek line; python3 scripts/validate_docs_snippets.py green;
       python3 scripts/check_docs_sync.py → "OK: docs/example/ is in sync with example/"
       (the tsv mirror is part of this); .venv/bin/ruff check + format --check on the test file.
    4. Atomic commit — ONE commit containing exactly NINE files: both example/ notebooks, both
       docs notebook mirrors, the tsv (example + docs mirror), both wrappers, and the test file.
       Message: "docs(quick-261004): showcase notebooks gain PNG figure mimes, zoom and combined CRE+Anno windows".
       NO attribution trailers of any kind (owner rule). Push to origin phs (owner default:
       commit and push). Do not include unrelated dirty files (.planning/, .scratch/, /tmp).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q && cmp example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb && cmp example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb && cmp example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv && out=$(python3 scripts/check_notebook_md_sync.py); ! echo "$out" | grep -q plant_helixseek && python3 scripts/validate_docs_snippets.py && dsout=$(python3 scripts/check_docs_sync.py) && echo "$dsout" | grep -q '^OK: docs/example/' && hfiles=$(git show --name-only --format= HEAD) && diff <(printf '%s\n' "$hfiles" | sort) <(printf '%s\n' example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.tsv docs/example/notebooks/plant_helixseek_cre.md docs/example/notebooks/plant_helixseek_anno.md tests/examples/test_plant_helixseek_showcase.py | sort) && cmsg=$(git log -1 --format=%B) && ! echo "$cmsg" | grep -qiE 'co-authored|generated-with|attribution'</automated>
  </verify>
  <done>13 fast tests pass; all three mirror pairs byte-identical (both notebooks + the tsv); md-sync failure set unchanged (3 pre-existing, none plant_helixseek); snippets and docs-sync green; ruff clean; HEAD is the single atomic commit touching exactly the nine files with no attribution trailers, pushed to origin phs.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| executed kernel → committed blob | notebook outputs are committed as showcase evidence; edits without re-execution would present stale/forged results |
| external data → committed tsv → notebook panel | the leaf-DNase panel derives from an externally downloaded BigWig; its committed derivative must carry provenance and stay integrity-pinned |
| committed blob → public docs surface | mirrors + wrappers are what readers consume; drift between example/ and docs/ misrepresents evidence |
| owner selection → notebook literals | the zoom window and axis floor are human display choices; the notebook must not dress them up as computed optima, and truncating display axes must be disclosed |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261004-01 | Tampering | showcase notebook committed outputs | high | mitigate | Source edits and re-execution land in one atomic commit; committed-blob one-liner gates assert the executed stream metrics in-band plus three vega + three image/png outputs and the exact window lines; nightly lane re-executes and re-asserts parsed bands (existing contract, extras extended for the new data reads) |
| T-261004-02 | Tampering | 2 MB notebook budget (D-12) | medium | mitigate | Size asserted in both one-liner gates and pinned by the existing structure test; PNG export uses vl_convert defaults with a documented shrink-scale fallback that never drops a vega mime; window payloads are inherently small (40 kb window) |
| T-261004-03 | Repudiation | illustrative zoom/combined framing (SHOW-07) | medium | mitigate | Captions carry the pinned disclaimer, name the full-locus metrics as authoritative, and DISCLOSE the confident-band display filter (only p(CRE) >= 0.5 bins drawn, y domain [0.5, 1]) as a display choice with the unfiltered full-locus overview referenced; selection tooling and candidate renderings live only in /tmp and .scratch (gitignored) and never enter the repo; window/combined evidence lines derive from plain literals, never scoring code; the genome-wide denylist test plus the generalized caption test enforce this kernel-free |
| T-261004-04 | Tampering | committed leaf-DNase tsv (external-data derivative) | medium | mitigate | Provenance pinned three ways — tsv header comments (source URL, extraction date, coordinate convention), a provenance-cell line in both notebooks, and wrapper prose; integrity via byte-identical docs mirror (cmp) inside the atomic commit; read at runtime like any committed shared data (no network, no pyBigWig, no new dependency) |
| T-261004-SC | Tampering | package installs | low | accept | No package-manager installs in this plan; vl_convert 1.9.0 and the jupyter CLI verified already present in .venv (live check this session) |
</threat_model>

<verification>
- Task 1 surgery validation: both notebooks JSON-valid, every code cell AST-parses, 100-char lines, cell counts 26/32, guard-first + first-figure-index invariants, image/png named in all three figure-cell sources, exact window literals + zoom_window=/combined_window= prints, no-scoring guards on both new figure cells per notebook, confident-band filter (>= 0.5, domain=[0.5, 1], pcres_bins_shown=) present and disclosed in every new p(CRE)-bearing cell, tsv referenced by both combined cells and the provenance cells, nightly extras extended.
- Task 2 committed-blob gates: CRE (jaccard in [0.30, 1.00] observed 0.3247; neg_cre_fraction in [0.00, 0.05] observed 0.0325) and Anno (genes >= 3 observed 59; neg_anno_fraction in [0.00, 0.10] observed 0.0000; exon_f1 observed 0.7522; pred_gff3_rows >= 1), each plus three vega + three image/png display outputs, zoom_window= and combined_window= lines exactly Chr1:5220001-5260000, <= 2,097,152 bytes, guard intact, tsv present, tree clean outside the notebooks + tsv.
- Task 3 full suite: 13 fast tests green; three cmp-identical mirror pairs (both notebooks + tsv); md-sync failure set unchanged; snippets and docs-sync green; ruff clean; single atomic nine-file commit, no attribution trailers, pushed.
</verification>

<success_criteria>
- All three figures per notebook render in local JupyterLab and VS Code (image/png mime present) while keeping the GitHub vega rendering (vega v6 object + vegalite v6 string retained).
- The zoom and combined figures use the owner-chosen window Chr1:5220001-5260000 as plain literals with only presentation code; sub-0.5 p(CRE) bins are not drawn (confident-band filter, disclosed in captions with the unfiltered full-locus overview referenced); captions frame them as display-clarity views under SHOW-07, and the full-locus metrics explicitly remain the authoritative claim.
- The combined figure aligns CRE prediction, official leaf DNase signal (from the committed, provenance-pinned tsv), binned DHS truth, predicted CDS strips, and TAIR10 gene models on one genomic axis in both notebooks, matching the approved reference render; each notebook computed its sibling model's prediction on the fixed window.
- Re-execution reproduces jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000; both notebooks within 2,097,152 bytes; nightly extras seed every new data read.
- Fast structure lane 13/13 green; docs mirrors (notebooks + tsv) byte-identical; wrapper + sync baselines unchanged; single atomic commit without attribution trailers on origin phs.
</success_criteria>

<output>
Create `.planning/quick/261004-dyw-planthelixseek-showcase-notebook-vega-ve/261004-dyw-SUMMARY.md` when done
</output>
