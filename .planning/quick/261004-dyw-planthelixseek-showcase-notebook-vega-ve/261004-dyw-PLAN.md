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
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_anno.md
autonomous: false
requirements:
  - SHOW-07
  - D-12
  - D-13

estimate:
  tokens: 34000
  raw_tokens: 34000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - Every figure output in both committed showcase notebooks carries an image/png mime (vl_convert PNG export) alongside the existing application/vnd.vega.v6+json object and vegalite v6 JSON string — the figures render in local JupyterLab and VS Code, not only in the GitHub viewer.
    - The zoom-window selection is an INTERNAL, owner-driven process that never enters the notebooks: candidates are rendered from the committed vega data into /tmp for the owner to choose from, and each notebook's zoom cell contains ONLY presentation code with the chosen window's coordinates as plain literals — zero window-scoring/selection code, no "best jaccard"/"best F1" algorithm or claim exposed in notebook source or captions (owner revision, mid-planning).
    - Each notebook carries a SECOND figure over the owner-chosen 20-50 kb illustrative zoom window, captioned under SHOW-07 as a window chosen for display clarity with the full-locus metrics named as the authoritative claim.
    - Re-execution on the GB10 reproduces the frozen metrics exactly: jaccard=0.3247 and neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000 (Anno).
    - Both committed notebooks stay at most 2,097,152 bytes (D-12).
    - The fast showcase structure lane passes (13 tests) including the extended assertions; the SHOW-07 genome-wide denylist and caption tests stay green; docs mirrors stay byte-identical; check_notebook_md_sync / validate_docs_snippets / check_docs_sync stay exactly as green as the pre-change baseline (md-sync failure set remains the 3 pre-existing stale wrappers, none plant_helixseek).
    - Notebook + mirrors + wrappers + test file land in ONE atomic commit with no attribution trailers.
  artifacts:
    - /tmp/planthelixseek_zoom_candidates/ — transient (never committed): candidate PNG gallery (4-6 per notebook), the generator script, and SELECTION.txt recording the owner's cre_zoom=/anno_zoom= coordinate picks.
    - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb — 23 cells (20 + 3 inserted after caption cell 14): figure cell 13 gains an image/png mime entry; new zoom header/figure/caption cells carrying the chosen CRE window as literals.
    - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb — 29 cells (26 + 3 inserted after caption cell 20): figure cell 19 gains an image/png mime entry; new zoom header/figure/caption cells carrying the chosen Anno window as literals.
    - tests/examples/test_plant_helixseek_showcase.py — test_committed_notebook_has_executed_outputs extended (image/png present, at least two vega-bearing display outputs); test_illustrative_caption_follows_the_metric_figure generalized to EVERY altair figure cell.
    - docs/example/notebooks/plant_helixseek_{cre,anno}/plant_helixseek_{cre,anno}.ipynb — byte-identical mirrors of the executed notebooks.
    - docs/example/notebooks/plant_helixseek_{cre,anno}.md — one added prose sentence each (PNG mimes + illustrative zoom window), no new code blocks.
  key_links:
    - Owner selection (Task 1 gallery + SELECTION.txt) → plain literals inside the notebook zoom cells → zoom_window= evidence line printed from those literals — the only zoom data the committed notebook derives.
    - Figure-cell display dict → 4-mime bundle (vega v6 object + vegalite v6 JSON string + image/png base64 + text/plain) via base64.b64encode(vlc.vegalite_to_png(vegalite_spec)).decode("ascii") — vl_convert 1.9.0 verified present in .venv.
    - Committed blob → structure tests + the 07-01/07-02-style one-liner gates (extended with image/png, second vega output, and the zoom_window= line).
    - Executed notebooks → docs mirrors (cp + cmp byte-identity) → check_docs_sync "OK".
---

<objective>
Enhance the two PlantHelixSeek showcase notebooks' result display (owner request, revised
mid-planning: zoom-window selection is an internal owner-driven process, NOT notebook code):

1. Each figure's display_data output gains an image/png mime (vl_convert PNG export) next to the
   existing vega v6 object + vegalite v6 JSON string. GitHub renders the vega object, but local
   JupyterLab and VS Code do NOT handle the vega v6 object + bare .json string combination — the
   PNG mime makes the figures render everywhere.
2. Candidate 20-50 kb zoom windows are rendered OUTSIDE the repo (from the already-committed vega
   output data) into /tmp as PNGs; the OWNER picks one window per notebook. The notebooks then
   contain ONLY the presentation: a zoom-figure cell with the chosen window's coordinates as plain
   literals, a normal SHOW-07-framed caption (illustrative window chosen for display clarity;
   full-locus metrics remain the authoritative assertions). No selection/scoring code in the
   notebooks.
3. Both notebooks are re-executed on the GB10 and must reproduce the frozen metrics exactly
   (jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59,
   neg_anno_fraction=0.0000).
4. Mirrors byte-synced, wrappers updated (prose only), 2 MB budget held, fast structure lane and
   docs-sync gates green, everything in one atomic commit with no attribution trailers.

Purpose: the showcase notebooks are the public evidence surface for the PlantHelixSeek work
(Phases 6-7); today their figures are invisible to anyone opening them locally.
Output: re-executed notebooks with dual-render figures + owner-chosen zoom windows, synced
mirrors/wrappers, extended kernel-free structure tests.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Verified-live facts this plan encodes (do not re-derive):
- CRE notebook has 20 cells, figure = code cell 13, caption = markdown cell 14; Anno has 26 cells,
  figure = code cell 19, caption = markdown cell 20. Each has exactly one display_data with
  vega-v6-object + vegalite-v6-STRING mimes (plus an unrelated widget-view output in the
  model-load cells). The vegalite v6 string carries the full inline datasets (pred track
  start/score rows + truth start/end/dhs rects for CRE; predicted segments + gene spans + CDS
  blocks for Anno) — enough to render candidate zoom windows WITHOUT any GPU or model work.
- Both figure cells already import json/altair/pandas/vl_convert and end with
  display({...}, raw=True) built from vegalite_spec = json.loads(chart.to_json()) and
  vega_spec = vlc.vegalite_to_vega(vegalite_spec).
- CRE presentation state at insert point: pred_df / truth_df (figure cell 13 DataFrames), plus
  bin_scores, bin_width, genomic_offset. Anno presentation state: pred_track_df, span_records,
  truth_cds_df, gene_orders (figure cell 19), plus genomic_offset, locus_length.
- .venv/bin/jupyter execute --inplace --timeout N exists; nbclient sets the kernel cwd to the
  notebook's own directory so ../plant_helixseek_shared/data/ and outputs/ resolve (proven 07-01).
- NOTEBOOK_EXEC_SPECS budgets: CRE cell_timeout 1200, Anno cell_timeout 3600
  (tests/examples/_execution.py:196-220).
- Wrappers docs/example/notebooks/plant_helixseek_{cre,anno}.md do NOT quote the figure cells (no
  alt.Chart/vegalite/display excerpts) — editing figure cells and inserting cells cannot break the
  AST sync check.
- Baselines: check_notebook_md_sync fails on exactly 3 pre-existing stale wrappers
  (mcp_langchain, mcp_pydantic_ai, data_prepare_finetune — none plant_helixseek);
  validate_docs_snippets green; check_docs_sync "OK". outputs/ is gitignored (.gitignore:63).
- Owner rules: zoom selection stays internal (this revision); atomic commit (notebook + mirror +
  wrapper together), no attribution trailers, execution on the local GB10.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Render candidate zoom windows from the committed vega data for owner selection (transient, /tmp only)</name>
  <files></files>
  <action>
    Fully orchestrator-side input producer — nothing in this task enters the repo, and it needs no
    GPU/model work. All artifacts go under /tmp/planthelixseek_zoom_candidates/ (create it).

    1. Write a throwaway generator script /tmp/planthelixseek_zoom_candidates/render_candidates.py
       using .venv/bin/python with altair + vl_convert (both installed). It reads the two committed
       notebooks, extracts the vegalite v6 JSON STRING mime from each figure cell's display_data
       output (CRE cell 13, Anno cell 19), and json.loads it — the inline datasets inside the spec
       are the rendered evidence (CRE: pred start/score track + truth start/end/dhs rects; Anno:
       predicted CDS segments + truth gene spans + CDS blocks).
    2. Candidate ranking (internal heuristic, only to surface promising windows — the owner makes
       the final aesthetic call): 20-50 kb window widths (e.g. 20000/35000/50000) sliding across
       the locus with a 10 kb stride. CRE: rank by per-window mean truth coverage fraction plus
       mean predicted score (both derivable from the embedded datasets). Anno: rank by count of
       truth gene lanes plus predicted segments fully inside the window. Keep the top 4-6 per
       notebook, preferring some width variety around the best-ranked region.
    3. Render each candidate as a PNG using the SAME chart shape the notebook shows (CRE:
       two-track vconcat — blue #4c78a8 score line + orange #f58518 truth rects, x domain pinned
       to the window; Anno: predicted-segment rect track + per-strand gene panels, x domain
       pinned) via vlc.vegalite_to_png. Save as
       /tmp/planthelixseek_zoom_candidates/cre_candidate_<n>_Chr1_<start>-<end>.png and
       anno_candidate_<n>_Chr1_<start>-<end>.png (1-based closed coordinates in filenames).
    4. Print a gallery table for the owner: per notebook, one row per candidate (PNG path, window
       coordinates, width, one-line heuristic note), plus the instruction: pick ONE window per
       notebook and reply with the two coordinate pairs.
    5. Selection contract: the orchestrator/owner records the choice in
       /tmp/planthelixseek_zoom_candidates/SELECTION.txt as exactly two lines, cre_zoom=Chr1:a-b
       and anno_zoom=Chr1:c-d (1-based closed, both inside Chr1:5100001-5300000). If the
       orchestrator already supplies the chosen coordinates at dispatch time, write this file
       directly and skip the interactive pick. Task 2 consumes this file.
  </action>
  <verify>
    <automated>test -d /tmp/planthelixseek_zoom_candidates && test $(ls /tmp/planthelixseek_zoom_candidates/cre_candidate_*.png 2>/dev/null | wc -l) -ge 4 && test $(ls /tmp/planthelixseek_zoom_candidates/anno_candidate_*.png 2>/dev/null | wc -l) -ge 4 && ls -la /tmp/planthelixseek_zoom_candidates/ && du -sh /tmp/planthelixseek_zoom_candidates/ && test -z "$(git status --porcelain example/ docs/ tests/)"</automated>
  </verify>
  <done>4-6 candidate PNGs per notebook exist under /tmp/planthelixseek_zoom_candidates/ with the gallery table printed for the owner, and the repo is untouched by this task; SELECTION.txt either written by the orchestrator or pending the owner's pick (Task 2's precondition governs the hand-off).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Notebook source surgery — PNG mimes + presentation-only zoom cells from the chosen literals — and fast structure-test extension</name>
  <precondition>Owner zoom-window selection recorded: /tmp/planthelixseek_zoom_candidates/SELECTION.txt exists with parseable cre_zoom=Chr1:a-b and anno_zoom=Chr1:c-d lines (1-based closed, inside Chr1:5100001-5300000). If absent, present the Task 1 gallery and halt for the owner's pick — do not choose windows autonomously.</precondition>
  <files>example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, tests/examples/test_plant_helixseek_showcase.py</files>
  <behavior>
    - Structure test: both committed notebooks expose at least one image/png mime across their display outputs (named-cause: local JupyterLab/VS Code rendering).
    - Structure test: both committed notebooks expose at least TWO display_data outputs carrying an application/vnd.vega mime (full-locus figure + zoom figure).
    - Structure test: EVERY code cell containing "alt.Chart(" is followed by a markdown cell whose text (lowercased) carries the pinned disclaimer "illustrative locus, not genome-wide accuracy".
    - These go RED against the surgically edited (not yet re-executed) blobs — expected transient; they turn GREEN only after Task 3's re-execution. Do not weaken them to pass early.
  </behavior>
  <action>
    MECHANICS — the Edit tool refuses .ipynb: do all notebook edits in one throwaway .venv/bin/python
    session using nbformat (nbformat.read(path, as_version=4), exact single-match string anchors on
    cell sources, nbformat.v4.new_markdown_cell / new_code_cell for inserts, nbformat.write). Keep
    the existing serialization style (nbformat canonical form, same as prior jupyter-execute
    writes). Do NOT execute anything in this task — Task 3 regenerates all outputs wholesale.
    Read the chosen windows from /tmp/planthelixseek_zoom_candidates/SELECTION.txt first.

    EDIT 1 (both notebooks, existing figure cell — CRE cell 13, Anno cell 19): add import base64
    to the cell's import block, and add one entry to the display dict after the vegalite entry and
    before text/plain: "image/png" mapped to base64.b64encode(vlc.vegalite_to_png(vegalite_spec))
    .decode("ascii"). Extend the embed comment: the PNG mime exists because local JupyterLab/VS
    Code do not render the vega v6 object + bare .json string combination (GitHub does).

    EDIT 2 (CRE — insert 3 cells after caption cell 14; later cells shift by 3):
    a. Markdown header "## Illustrative zoom window" — one sentence: the window below is shown for
       display clarity; the full-locus metrics above remain the authoritative claim.
    b. Code cell, PRESENTATION ONLY: the chosen window's coordinates as plain literals (zoom_start,
       zoom_end in the notebook's 0-based half-open convention, converted once from the selected
       1-based closed pair; a brief comment names the selection as an owner pick from candidate
       renderings, not a notebook-computed optimum). Slice the existing figure-cell DataFrames —
       pred_df and truth_df rows inside the window — and build zoom_track_chart with the SAME
       two-track vconcat shape as cell 13 (blue #4c78a8 prediction line, orange #f58518 truth
       rects), x scale domain pinned to the window. Print exactly one evidence line derived from
       the literals: zoom_window=Chr1:<start+1>-<end> (plus zoom_width=<int> if useful). Embed via
       display(..., raw=True) carrying ALL FOUR mimes (vega v6 object, vegalite v6 JSON string,
       image/png base64, text/plain naming it as the zoom view). ZERO scoring code: no jaccard
       computation, no threshold recomputation, no peak re-calling — the string "jaccard" must not
       appear in this cell's source.
    c. Markdown caption: reuse the pinned sentence "Illustrative locus, not genome-wide accuracy."
       plus wording that this window was chosen for display clarity and the full-locus jaccard
       above remains the authoritative claim. Any line mentioning genome-wide must contain "not
       genome-wide" (SHOW-07 denylist test). No "best"/"optimal" claims.

    EDIT 3 (Anno — insert 3 cells after caption cell 20):
    a. Markdown header, same framing sentence as CRE.
    b. Code cell, PRESENTATION ONLY, same literal-coordinates pattern: filter pred_track_df,
       span_records and truth_cds_df to entries overlapping the window (a LOCAL zoom panel helper,
       lane order re-sorted by start — never mutate the cell-19 state or its outputs), and build
       zoom_gene_model_chart with the same vconcat shape as cell 19 (predicted CDS rect track +
       per-strand truth gene panels + shared-x), x domain pinned to the window. Print
       zoom_window=Chr1:<start+1>-<end> from the literals (plus zoom_width=<int> and
       zoom_gene_count=<genes visible in the window> if useful — descriptive counts only). Embed
       with all four mimes. ZERO scoring code: no per-gene F1 recomputation — the strings
       "gene_f1" and "EXON_F1_FLOOR" must not appear in this cell's source.
    c. Markdown caption: pinned disclaimer + chosen-for-display-clarity wording; the full-locus
       exon-level metrics above remain the authoritative claim; SHOW-07-compliant; no "best"
       claims.

    CONSTRAINTS on every new/edited code cell: import only json, base64, altair, pandas,
    vl_convert, numpy — all already exercised in these notebooks (the fast-lane import-exec test
    and the D-16 no-fla-import rule stay satisfied; never name the fla module). No metric asserts
    anywhere new (nightly re-execution safety). Every source line at most 100 chars.

    EDIT 4 (tests/examples/test_plant_helixseek_showcase.py):
    - test_committed_notebook_has_executed_outputs: additionally collect display_data outputs;
      assert at least one carries "image/png" in its data (named-cause message: local
      JupyterLab/VS Code rendering of the showcase figures), and assert at least TWO display_data
      outputs carry an application/vnd.vega mime (full-locus figure + zoom figure). Update the
      docstring.
    - test_illustrative_caption_follows_the_metric_figure: generalize from the first alt.Chart cell
      to EVERY code cell containing "alt.Chart(" (expect at least 2 per notebook) — the first
      markdown after EACH must carry ILLUSTRATIVE_DISCLAIMER. Update the docstring.
    - Module docstring: one sentence noting the PNG-mime and zoom-window pinning.
    - Keep ruff clean (line length 100). Do NOT add any test that pins specific zoom coordinates —
      the window is an owner choice, not a contract value.

    POST-SURGERY VALIDATION (same session, before moving on): reload both files; json.loads
    round-trip; ast.parse every code cell; no source line over 100 chars; cell counts exactly 23
    (CRE) and 29 (Anno); the first code cell is still the D-16 guard (find_spec + RuntimeError);
    exactly two code cells contain "alt.Chart(" per notebook, the FIRST being the pre-existing
    full-locus figure (CRE index 13, Anno index 19 — inserts land strictly after those indices);
    each of the two figure cells' sources names "image/png"; the zoom cell's source prints
    zoom_window=; and the no-scoring guards hold (CRE zoom source contains no "jaccard"; Anno zoom
    source contains no "gene_f1").
  </action>
  <verify>
    <automated>.venv/bin/python -c "
import json, ast, pathlib
specs = [
    ('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb', 23, 13, 'jaccard'),
    ('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb', 29, 19, 'gene_f1'),
]
for p, n, fig, banned in specs:
    nb = json.loads(pathlib.Path(p).read_text())
    assert len(nb['cells']) == n, (p, 'cell count', len(nb['cells']))
    srcs = [''.join(c.get('source', [])) for c in nb['cells']]
    assert 'find_spec' in srcs[1] and 'RuntimeError' in srcs[1], (p, 'guard cell changed')
    figs = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'alt.Chart(' in ''.join(c.get('source', []))]
    assert len(figs) == 2 and figs[0] == fig, (p, 'figure cells', figs)
    assert all('image/png' in srcs[i] for i in figs), (p, 'png mime missing in a figure cell')
    assert 'zoom_window=' in srcs[figs[1]], (p, 'zoom evidence print missing')
    assert banned not in srcs[figs[1]], (p, 'scoring code leaked into zoom cell:', banned)
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        assert all(len(line) <= 100 for line in s.splitlines()), (p, 'line over 100 chars')
        if c['cell_type'] == 'code' and s.strip():
            ast.parse(s)
print('surgery-ok')" && .venv/bin/ruff check tests/examples/test_plant_helixseek_showcase.py && .venv/bin/ruff format --check tests/examples/test_plant_helixseek_showcase.py</automated>
  </verify>
  <done>Both notebooks carry the PNG mime in both figure cells' display dicts and three new presentation-only zoom cells each (23/29 total cells) driven by the owner-chosen literal coordinates, all cells AST-parse with 100-char lines, guard/figure ordering invariants and the no-scoring guards hold, and the structure tests are extended per the behavior block (RED on stale outputs is the expected state at task end).</done>
</task>

<task type="auto">
  <name>Task 3: Re-execute both notebooks on the GB10, gate the committed blobs, sync mirrors/wrappers, atomic commit</name>
  <files>example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb, docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb, docs/example/notebooks/plant_helixseek_cre.md, docs/example/notebooks/plant_helixseek_anno.md</files>
  <action>
    1. Re-execute from the repo root (nbclient sets the kernel cwd to each notebook's own
       directory, so ../plant_helixseek_shared/data/ and outputs/ resolve exactly as in 07-01):
       .venv/bin/jupyter execute --inplace --timeout 1200 example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
       .venv/bin/jupyter execute --inplace --timeout 3600 example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
       The timeouts mirror the NOTEBOOK_EXEC_SPECS cell budgets (1200/3600, strictly inside the
       nightly 2400/5400 marks). Expect roughly 5-6 min for CRE and 12-25 min for Anno on the GB10
       — run them in the background or sequentially with adequate tool timeouts; execution requires
       the local GPU and regenerates every output wholesale (stale Task-2 outputs are fully
       replaced).
    2. After each execution, cross-check the exact frozen values: jaccard=0.3247,
       neg_cre_fraction=0.0325 (CRE); exon_f1=0.7522, genes_above_floor=59,
       neg_anno_fraction=0.0000 (Anno). The one-liner gates below assert the selection.md band
       mirrors (transient-gate note from 07-02: the literals are a pre-commit mirror; the
       authoritative band layer is the parsed _parse_bands() test assertions). If any value drifts
       OUTSIDE its band, HALT — that is environment drift; do not adjust notebook code or literals
       to force green.
    3. Size fallback: if either notebook exceeds 2,097,152 bytes, shrink ONLY the PNG payloads
       (e.g. a smaller scale= argument to vlc.vegalite_to_png) — never drop a vega mime — then
       re-execute that notebook and re-gate.
    4. Confirm git status --porcelain example/ shows only the two .ipynb modified (outputs/
       artifacts land under the global gitignore line 63).
    5. Mirror sync: cp each executed notebook over its docs mirror
       (docs/example/notebooks/plant_helixseek_cre/ and _anno/); verify byte-identity with cmp.
    6. Wrapper prose (both .md wrappers): extend the "Full Notebook" paragraph with ONE sentence
       stating that the embedded figures now carry an image/png mime alongside the compiled
       vega/vega-lite JSON (so local JupyterLab/VS Code render them) and that the notebook adds an
       illustrative zoom window chosen for display clarity while the full-locus metrics remain the
       authoritative claim. PROSE ONLY — add no code blocks: the AST sync check
       (check_notebook_md_sync) matches code blocks against notebook cells, and neither wrapper
       quotes the figure cells today; keep it that way.
    7. Full gate suite (all from repo root): fast showcase lane
       (.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q
       → 13 passed, the extended structure tests now GREEN against the executed blobs);
       python3 scripts/check_notebook_md_sync.py still reports exactly the 3 pre-existing stale
       wrappers and NO plant_helixseek line; python3 scripts/validate_docs_snippets.py green;
       python3 scripts/check_docs_sync.py → "OK: docs/example/ is in sync with example/";
       .venv/bin/ruff check + format --check on the test file.
    8. Atomic commit — ONE commit containing exactly: both example/ notebooks, both docs mirrors,
       both wrappers, and the test file. Message:
       "docs(quick-261004): showcase notebooks gain PNG figure mimes and illustrative zoom windows".
       NO attribution trailers of any kind (owner rule). Push to origin phs (owner default:
       commit and push). Do not include unrelated dirty files (.planning/, scratch dirs, /tmp).
  </action>
  <verify>
    <automated>.venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); j=float(re.search(r'^jaccard=([0-9.]+)',s,re.M).group(1)); n=float(re.search(r'^neg_cre_fraction=([0-9.]+)',s,re.M).group(1)); assert 1.00>=j>=0.30, 'jaccard='+str(j); assert 0.05>=n>=0.0, 'neg_cre_fraction='+str(n); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=2, 'vega display outputs='+str(len(vega)); assert len(png)>=2, 'image/png display outputs='+str(len(png)); assert 'zoom_window=' in s, 'zoom evidence line missing'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('cre-notebook-ok jaccard='+str(j)+' neg_cre_fraction='+str(n))" && .venv/bin/python -c "import json,re,pathlib; p=pathlib.Path('example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb'); nb=json.loads(p.read_text()); s=''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'); g=int(re.search(r'^genes_above_floor=([0-9]+)',s,re.M).group(1)); n=float(re.search(r'^neg_anno_fraction=([0-9.]+)',s,re.M).group(1)); f=float(re.search(r'^exon_f1=([0-9.]+)',s,re.M).group(1)); r=int(re.search(r'^pred_gff3_rows=([0-9]+)',s,re.M).group(1)); assert g>=3, 'genes_above_floor='+str(g); assert 0.10>=n>=0.0, 'neg_anno_fraction='+str(n); assert 1.0>=f>=0.0, 'exon_f1='+str(f); assert r>=1, 'pred_gff3_rows='+str(r); dd=[o for c in nb['cells'] for o in (c.get('outputs') or []) if o.get('output_type')=='display_data']; vega=[o for o in dd if any(m.startswith('application/vnd.vega') for m in (o.get('data') or {}))]; png=[o for o in dd if 'image/png' in (o.get('data') or {})]; assert len(vega)>=2, 'vega display outputs='+str(len(vega)); assert len(png)>=2, 'image/png display outputs='+str(len(png)); assert 'zoom_window=' in s, 'zoom evidence line missing'; assert 2097152>=p.stat().st_size, 'size='+str(p.stat().st_size); assert 'find_spec' in p.read_text() and 'fla_version=' in s, 'guard/versions missing'; print('anno-notebook-ok genes='+str(g)+' exon_f1='+str(f)+' neg_anno_fraction='+str(n))" && .venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q && cmp example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb && cmp example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb && out=$(python3 scripts/check_notebook_md_sync.py); ! echo "$out" | grep -q plant_helixseek && python3 scripts/validate_docs_snippets.py && dsout=$(python3 scripts/check_docs_sync.py) && echo "$dsout" | grep -q '^OK: docs/example/' && hfiles=$(git show --name-only --format= HEAD) && diff <(printf '%s\n' "$hfiles" | sort) <(printf '%s\n' example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb docs/example/notebooks/plant_helixseek_cre.md docs/example/notebooks/plant_helixseek_anno.md tests/examples/test_plant_helixseek_showcase.py | sort) && cmsg=$(git log -1 --format=%B) && ! echo "$cmsg" | grep -qiE 'co-authored|generated-with|attribution'</automated>
  </verify>
  <done>Both one-liner gates print their ok lines with the exact frozen values (cre-notebook-ok jaccard=0.3247 neg_cre_fraction=0.0325; anno-notebook-ok genes=59 exon_f1=0.7522 neg_anno_fraction=0.0), both sizes within 2,097,152 bytes, each notebook showing two vega and two image/png display outputs plus the zoom_window= stream line; 13 fast tests pass; mirrors byte-identical; md-sync failure set unchanged (3 pre-existing, none plant_helixseek); snippets and docs-sync green; ruff clean; HEAD is the single atomic commit touching exactly the 7 files with no attribution trailers, pushed to origin phs.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| executed kernel → committed blob | notebook outputs are committed as showcase evidence; edits without re-execution would present stale/forged results |
| committed blob → public docs surface | mirrors + wrappers are what readers consume; drift between example/ and docs/ misrepresents evidence |
| owner selection → notebook literals | the zoom window is a human aesthetic pick; the notebook must not dress it up as a computed optimum |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261004-01 | Tampering | showcase notebook committed outputs | high | mitigate | Source edits and re-execution land in one atomic commit; committed-blob one-liner gates assert the executed stream metrics in-band plus two vega + two image/png outputs; nightly lane re-executes and re-asserts parsed bands (existing tests, unchanged contract) |
| T-261004-02 | Tampering | 2 MB notebook budget (D-12) | medium | mitigate | Size asserted in both one-liner gates and pinned by the existing structure test; PNG export uses vl_convert defaults, zoom payloads are inherently small (20-50 kb windows); documented fallback shrinks PNG scale only, never drops a vega mime |
| T-261004-03 | Repudiation | illustrative zoom window framing (SHOW-07) | medium | mitigate | Zoom captions carry the pinned disclaimer and name the full-locus metrics as authoritative with no best/optimal claims; selection tooling and candidate renderings live only in /tmp and never enter the repo; the genome-wide denylist test plus the generalized caption test enforce this kernel-free; the zoom_window= stream line derives from plain literals, never from notebook scoring code |
| T-261004-SC | Tampering | package installs | low | accept | No package-manager installs in this plan; vl_convert 1.9.0 and the jupyter CLI verified already present in .venv (live check this session) |
</threat_model>

<verification>
- Task 1: 4-6 candidate PNGs per notebook under /tmp/planthelixseek_zoom_candidates/ plus the printed gallery table; no repo writes.
- Task 2 surgery validation: both notebooks JSON-valid, every code cell AST-parses, 100-char lines, cell counts 23/29, guard-first + first-figure-index invariants, image/png named in both figure-cell sources, zoom evidence print present, no scoring strings in the zoom cells (no "jaccard" in CRE zoom, no "gene_f1" in Anno zoom).
- Task 3 committed-blob gates: CRE (jaccard in [0.30, 1.00] observed 0.3247; neg_cre_fraction in [0.00, 0.05] observed 0.0325) and Anno (genes >= 3 observed 59; neg_anno_fraction in [0.00, 0.10] observed 0.0000; exon_f1 observed 0.7522; pred_gff3_rows >= 1), each plus two vega and two image/png display outputs, the zoom_window= stream line, <= 2,097,152 bytes, guard intact; 13 fast tests green; mirrors cmp-identical; md-sync/snippets/docs-sync at baseline; one atomic 7-file commit, no attribution trailers, pushed.
</verification>

<success_criteria>
- Both showcase notebooks render their figures in local JupyterLab and VS Code (image/png mime present in every figure output) while keeping the GitHub vega rendering (vega v6 object + vegalite v6 string retained).
- The zoom-window choice is the owner's, made from rendered candidates outside the repo; the committed notebooks contain only presentation code with the chosen coordinates as plain literals and captions that frame them as display-clarity windows under SHOW-07, with the full-locus metrics explicitly remaining the authoritative claim.
- Re-execution reproduces jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000; both notebooks within 2,097,152 bytes.
- Fast structure lane 13/13 green; docs mirrors byte-identical; wrapper + sync baselines unchanged; single atomic commit without attribution trailers on origin phs.
</success_criteria>

<output>
Create `.planning/quick/261004-dyw-planthelixseek-showcase-notebook-vega-ve/261004-dyw-SUMMARY.md` when done
</output>
