# Phase 16: Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity - Research

**Researched:** 2026-10-13
**Domain:** Platform dispatch at example acquisition seams; platform-aware test gates; frozen real-pair fixture parity (numpy evaluators vs bedtools truth chain)
**Confidence:** HIGH (nearly every claim verified in-repo this session; the load-bearing adapter contract verified by executing the real pinned pybedtools 0.12.1 + bedtools v2.31.1 against the phase-15 module on this box)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**平台拆分总纲 (owner directive 2026-10-10, WINDOWS #19 — carried, binding)**
- Linux KEEPS bedtools/pybedtools; the frozen truth chain stays bedtools-computed, no re-freeze
- Windows dispatches to the phase-15 numpy evaluators only where bedtools cannot install; the NER example CODE ITSELF gains runtime platform detection
- The downstream frozen Anno metric (exon_f1=0.7522) must hold in band on BOTH paths; the loj replacement needs its OWN parity check (numpy join vs bedtools intersect -loj — jaccard parity does not transfer to join semantics)

**冻结值与测试策略 (owner-accepted 2026-10-13)**
- exon_f1 tolerance band sized to measured numpy trailing-float deltas (jaccard 0.324727 precedent ~1e-9 class; planner pins the exact band from measurement, recorded in selection.md as the single truth source)
- Frozen interval pairs committed under tests/fixtures/ (51/94-row-class real pairs regenerated from the selection.md source, git-tracked)
- Windows-branch contract tests exercise the dispatch via monkeypatched platform (roadmap wording)
- WPAR-06 dispatch-direction structure test lands as its own file tests/test_dispatch_direction.py (fail-loud direction assertion, no except-fallback ladder)

### Claude's Discretion
Adapter plumbing details, gate-edit mechanics within the inventoried seams, notebook install-cell conditionals.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. (WPAR-07 triton-windows declaration is explicitly Phase 18, not here.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WPAR-03 | `generate_bpe_dataset.py` + `data_generation_and_inference.ipynb` fork at acquisition only (runtime platform detection); Linux keeps `pybedtools.BedTool.intersect(loj=True)`; Windows dispatches to numpy through uniform row adapter (`.chrom/.start/.end/.name/.fields`); notebook install cell skips pybedtools on Windows | Seam inventory (§ Seams), adapter contract empirically pinned (§ Code Examples), install/import cell states verified |
| WPAR-04 | Three inventoried gates become platform-aware — Linux probe-then-execute unchanged, Windows executes the numpy path (fail-loud, no ever-green skip) | Gate current-state audit at exact current line numbers (§ Test Gates), gate helper mechanics, fast-contract-test precedents |
| WPAR-05 | test-windows executes the numpy evaluators over committed frozen pairs with pinned tolerances; jaccard=0.324727 and exon_f1=0.7522 (band newly pinned) hold in band on BOTH paths; selection.md single truth source parsed on both paths | Frozen-pair provenance + regeneration feasibility (§ Frozen Pairs), selection.md parse machinery (§ selection.md), test-windows lane mechanics (§ CI) |
| WPAR-06 | Same-change dispatch-direction structure test proves the numpy path cannot silently execute on Linux (fail-loud dispatch, no except-fallback ladder) | Structure-test precedents (nb JSON + ast.parse readers), placement decision (§ WPAR-06) |
| WPAR-08 | Every example edit lands with docs mirror byte-identical (`check_docs_sync` green); Linux nightly re-execution holds parsed selection.md bands | check_docs_sync mechanics + docs-validation.yml same-commit enforcement (§ Docs Sync); nightly lanes identified (§ CI) |
</phase_requirements>

## Summary

Phase 16 wires the phase-15 numpy evaluators (`dnallm.utils.jaccard` / `loj_intersect` / `parse_bed`) into the example tree's acquisition seams, makes the three inventoried test gates platform-aware, commits real frozen interval pairs under `tests/fixtures/`, and pins the exon_f1 tolerance band in `selection.md`. The change surface is small and precisely known: two pybedtools seams in the NER script (import at line 7, `BedTool(...).intersect(loj=True)` at line 249), three notebook cells in `data_generation_and_inference.ipynb` (cell 0 install, cell 2 import, cell 8 loj call — plus a discovered fourth adjacent seam, cell 1's `!which intersectBed`), one subprocess seam in `plant_helixseek_cre.ipynb` cell 11, three test gates, one new fixture directory, one new structure-test file, and a band row in `selection.md`.

The single most load-bearing technical fact, verified empirically this session by executing the repo's pinned pybedtools 0.12.1 against the nightly runner's cached bedtools v2.31.1: `pybedtools.BedTool.intersect(loj=True)` rows and `loj_intersect`'s tab-joined output lines have a byte-identical 12-column layout, including the null-B fill `('.', '-1', '-1', '.', '-1', '.')`. The row adapter is therefore a trivial split: `.chrom=f[0]`, `.start=int(f[1])`, `.end=int(f[2])`, `.name=f[3]`, `.fields=f` (all strings). The consumer's `name == "-1"` intergenic check at generate_bpe_dataset.py:283 keys on the count-6 null **score slot** — a `.`-filled slot would crash with KeyError on the second consecutive intergenic token, so the null literals must ride through verbatim (they already do; the phase-15 module pins them).

The frozen-pair situation is the main planning risk: the original 51/94-row CRE pair lived under gitignored `.scratch/` and is gone; only aggregates survive (`FROZEN_PAIR_JACCARD = 0.32472687098490116`, intersection=15872, union=48878 baked in tests/test_genomic_intervals.py:580-590). Regeneration is feasible on this box — bedtools v2.31.1 exists at the nightly runner's cached prefix (not on PATH), pybedtools imports in the repo venv, and the CRE truth side is deterministic from committed artifacts — but the **pred** side is model output and requires a GPU notebook run in the fixture-generation session, with bedtools AND numpy values recorded in the same session.

**Primary recommendation:** Put the frozen-pair tests and the dispatch-direction test at `tests/` root (platform-agnostic, not slow-marked) so they execute on every CI leg including test-windows with zero census impact; regenerate fixtures in one dev-box session with the runner-prefix bedtools on PATH; keep the exon_f1 tight band asserted only by the frozen-pair tests (the nightly re-execution keeps the four existing drift bands); mirror every example edit to docs/ in the same commit.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Platform dispatch decision (win32 vs POSIX) | Example tier (script + notebook cells) | — | Owner directive: "the NER example CODE ITSELF gains runtime platform detection". The library stays platform-agnostic |
| Interval evaluation on Windows | `dnallm.utils.genomic_intervals` (phase-15, locked surface) | — | WPAR-01 locked the import-pure numpy-only module; Phase 16 consumes it, does not extend semantics |
| Row adapter (pybedtools-row emulation) | Claude's discretion: `dnallm.utils` helper OR example-inline | — | ~8 lines; a utils helper avoids triple duplication (script + notebook) but extends the WPAR-01-locked public surface (needs its own tests); inline keeps the locked surface untouched |
| Truth bands / tolerances | `selection.md` (data, single truth source) | tests parse it | Owner decision: "tolerance lives in the comparison layer only" — bands parsed, never hardcoded (D-07) |
| Platform-aware gating | tests tier (three inventoried gates) | — | Gates decide execute-vs-dispatch; Linux probe-then-execute pattern already established |
| Frozen pairs + oracle values | `tests/fixtures/` (git-tracked data) + selection.md | — | New directory; `*.bed` is not gitignored, no .gitignore exception needed |
| Windows CI execution of evaluators | test-windows fast lane (`pytest -m "not slow"`) | — | Platform-agnostic test files run there automatically; numpy is a core dependency on every leg |
| bedtools binary provisioning (Linux) | CI rootless micromamba step (unchanged) | — | coverage-nightly + example-nightly already provision v2.31.1 from cached prefix; Phase 16 must not touch this |

## Seams — Exact Current State (re-located, not trusted from old line numbers)

### NER script: `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` (321 lines)

[VERIFIED: example/notebooks/finetune_NER_task/generate_bpe_dataset.py:7,249,258-265,283]
- Line 7: `from pybedtools import BedTool`
- Line 249: `intersection = BedTool(tokens_bed).intersect(annotation_bed, loj=True)`
- Lines 258-265, the consumer (verbatim):
```python
for iv in intersection:
    chrom = iv.chrom
    start = iv.start
    end = iv.end
    token = iv.name
    gene = iv.fields[4]
    gene2 = iv.fields[9]
    name = iv.fields[10]
```
- Line 283: `if name == "-1":` → intergenic branch.

Column semantics (verified empirically, see Code Examples): tokens_bed is BED6 (`chrom start end token_str gene strand`, written at line 226); annotation_bed is BED6 (`chrom start end gene feature strand`, written at line 130). loj output = 12 columns: A cols 1-6 + B cols 1-6. `iv.name` = A col 4 (token_str); `iv.fields[4]` = A col 5 (gene — the flush key); `iv.fields[9]` = B col 4 (annotation gene, `"."` on no-hit); `iv.fields[10]` = B col 5 (feature `exon`/`intron`, **`"-1"` on no-hit** — the count-6 null score slot).

Two structural invariants the dispatch must preserve:
1. **Gene contiguity**: the consumer flushes on `if gene != current_gene` (line 268), so all rows of a gene must be contiguous in the intersection stream. Both bedtools `-loj` and `loj_intersect` emit **A-file order** (join-not-merge, never sorted), and tokens_bed groups rows per gene, so contiguity holds on both paths. [VERIFIED: dnallm/utils/genomic_intervals.py:274-297 docstring "Every A row survives in A-file order... This is a JOIN, not a merge: rows are never sorted"]
2. **The `-1` null literal**: with a 6-column B, `_null_b_fields(6)` returns `(NULL_CHROM, NULL_START, NULL_END, NULL_NAME, NULL_SCORE, NULL_OTHER)` = `(".", "-1", "-1", ".", "-1", ".")` [VERIFIED: dnallm/utils/genomic_intervals.py:213-218,269]. `fields[10]` lands on the score slot = `"-1"`. If it were `"."`, the consumer's else-branch would look up `zero_map[".0"]` → None → intergenic on first occurrence, but on a SECOND consecutive `"."` row `name == last_name` → `one_map.get(".1")` → None → `zero_map[".0"]` → **KeyError** (zero_map keys are only `intergenic0`, `exon0`, `intron0` — generate_bpe_dataset.py:239-247). Consecutive intergenic tokens are the common case, so this is a guaranteed crash, not a corner. The adapter must pass fields through verbatim — never re-encode nulls.

### NER notebook: `example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` (22 cells, all code except none — every cell is `code`)

[VERIFIED: read this session via JSON]
- **Cell 0** (install, 385 chars): `!brew install bedtools` then `!uv pip install pyfastx pybedtools` — "Omit it if you have already installed these two packages". This is the cell that must skip pybedtools (and the brew/bedtools line) on Windows.
- **Cell 1** (`!which intersectBed`): a verify cell. **Discovered adjacent seam, not in the WINDOWS #19 inventory** — `which` does not exist on Windows, so this cell errors there. The install-cell conditional work (Claude's discretion per CONTEXT) must cover it or the notebook cannot execute on Windows at all.
- **Cell 2** (imports): `from pybedtools import BedTool` among stdlib/dnallm imports.
- **Cell 8** (`tokens_to_nerdata`, 6238 chars): identical consumer shape to the script (`iv.chrom/.start/.end/.name/.fields[4]/.fields[9]/.fields[10]`, `if name == "-1"`). Its docstring comment claims "BedTool.intersect(...) returns results in ascending genomic order" — factually it is A-file order (spike-pinned); do not "fix" the comment's claim into code that sorts.
- **Cell 9**: `!wget -c https://rice.uga.edu/osa1r7_download/...` — downloads the two rice inputs (the script-lane test's `RICE_INPUT_URLS` mirror these).

### CRE notebook: `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` (23 cells)

[VERIFIED: read this session via JSON]
- **Cell 11** (the jaccard seam, verbatim):
```python
import subprocess

result = subprocess.run(
    ["bedtools", "jaccard", "-a", str(predicted_bed_path), "-b", str(truth_bed_path)],
    capture_output=True, text=True, check=True,
)
jaccard_value = float(result.stdout.strip().splitlines()[1].split("\t")[2])
print(f"jaccard={jaccard_value:.4f}")
```
  The Windows replacement calls `dnallm.utils.jaccard(a, b)["jaccard"]` (full-precision float; bedtools prints 6 dp). The `print(f"jaccard={jaccard_value:.4f}")` line MUST be preserved on both branches — the nightly test regex `^jaccard=([0-9.eE+-]+)` (test_plant_helixseek_showcase.py:576) and CRE cell 21's comparison parse it.
- **Cell 9** writes the two BEDs the jaccard consumes: `outputs/predicted_cre_peaks.narrowPeak` (BED6 core + 4 narrowPeak cols, built ascending so sorted) and `outputs/truth_dhs.bed` (BED6, `sorted(truth_intervals)`). Both are sandbox runtime artifacts (never committed; `outputs` is check_docs_sync-ignored).
- **Cell 21** parses selection.md frozen keys (`^key=` MULTILINE) and band rows (`line.startswith(f"| {metric_label} |")`) for the CRE rows — a new exon_f1 band row in selection.md does not break this parser (it matches specific labels only).
- **md cells 0, 10, 22** are bedtools-worded (`bedtools jaccard (v2.31.1)` etc.) and stay byte-identical per owner directive. Structure tests already constrain cells 0-1: `test_provenance_markdown_cell` (cell 0 markdown, selection.md link, locus line) and `test_first_code_cell_is_the_fla_guard` (first CODE cell = fla guard with `find_spec` + version prints) [VERIFIED: tests/examples/test_plant_helixseek_showcase.py:340-365]. Cell 11 edits do not disturb these.

### Docs mirrors (byte-identical today, verified by diff this session)

- `docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py` ≡ example copy
- `docs/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` ≡ example copy
- `docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` exists (mirror)
- `docs/example/notebooks/plant_helixseek_shared/data/selection.md` ≡ example copy

## Test Gates — Current Line Numbers and Mechanics

The inventory line numbers hold for two of three; the script gate drifted +2:

| Gate | Inventoried | Current | Kind |
|------|-------------|---------|------|
| script gate | test_script_execution.py:128 | **:130** `def _gate_script_pybedtools` | find_spec probe → `optional_dep_skip` |
| notebook gate | test_notebook_execution.py:1397 | **:1397** `def _gate_pybedtools` (unchanged) | `_gate_optional_deps(nb, ("pybedtools",))` |
| showcase gate | test_plant_helixseek_showcase.py:523 | **:523** `assert shutil.which("bedtools") is not None` (unchanged) | hard assert, fail-loud (NOT a skip) |

[VERIFIED: tests/examples/test_script_execution.py:130-145; tests/examples/test_notebook_execution.py:1397-1419; tests/examples/test_plant_helixseek_showcase.py:519-527]

Call sites and shapes:
- Script gate fires at test_script_execution.py:177 inside `TestExampleScriptExecution.test_generate_bpe_dataset_produces_artifact` (slow + timeout(3600)), BEFORE rice downloads. Fast contract tests `TestScriptGateContracts` (:148-167) monkeypatch `importlib.util.find_spec` — these run on test-windows today.
- Notebook gate registered in `_ACTIVE_NOTEBOOK_GATES` (:1417-1419) keyed `"notebooks/finetune_NER_task/data_generation_and_inference.ipynb"`; consulted at :229-231 in the slow `TestNotebookExecution.test_notebook_executes_end_to_end`. Fast contract tests `TestActiveNotebookGates.test_pybedtools_gate_*` (:315-341) monkeypatch `_probe_module`.
- Showcase gate sits at the top of the slow `test_cre_notebook_executes_within_selection_bands` (:517) — on a Windows box without bedtools this test currently FAILS (assert), which is the "fail-loud" half of WPAR-04 already; the platform-aware edit routes Windows to the numpy path instead of failing on the missing binary.

Typed-skip plumbing (all [VERIFIED: tests/examples/_execution.py:1197-1240]):
- `optional_dep_skip(action, evidence)` → `pytest.skip(f"optional-dep: {action} ({evidence})")` — prefix allowlisted in tests/expected_skips.yaml.
- `environment_unavailable_skip` → `environment-unavailable:` prefix (allowlisted).
- The probe-then-execute precedent the CONTEXT cites: `_gate_ollama_stack` (:1293) — probe with evidence, typed skip only on genuine infra-missing, never on wrong output.

Notebook import tolerance: `OPTIONAL_IMPORT_MODULES = ("pybedtools", "flash_attn", "stripedhyena", "evo2")` in tests/examples/test_examples.py:95 — the fast-lane `test_notebook_imports` execs every import statement in every notebook but tolerates missing pybedtools. Conditional imports added to NER cells stay compatible on every leg. [VERIFIED: tests/examples/test_examples.py:89-95,230-292]

## selection.md — Truth Source Mechanics

File: `example/notebooks/plant_helixseek_shared/data/selection.md` (+ docs mirror). [VERIFIED: read in full this session]

- **Observed values** (§ "Observed values (this run)", lines 56-64), verbatim keys at line starts:
```
jaccard=0.3247
exon_f1=0.7522
genes_above_floor=59
```
  plus `anno_locus_detail: n_truth_cds=526 n_pred_segments=394 tp=346 fp=48 fn=180 n_genes=91 top_gene_f1=1.0` (line 64). Note exon_f1 = 2·346/(2·346+48+180) = 692/920 — a rational number, fully deterministic given the pair.
- **Band table** (§ "Thresholds and tolerance bands", lines 68-73) — four rows: CRE jaccard `[0.3, 1.00]`; Anno gene models `>= 3 genes`; neg CRE `[0.00, 0.05]`; neg Anno `[0.00, 0.1]`. **No exon_f1 row exists** — WPAR-05 adds it.
- Line 35: `- Metric: \`bedtools jaccard\` (v2.31.1) on coordinate-sorted BEDs...` — the "selection.md:35 historical, unchanged" docs-only mention from WINDOWS #19.
- Parsers that must keep working after the edit:
  - `_parse_floors` (test_plant_helixseek_showcase.py:216-238): `re.search(rf"^{key}=([0-9.]+)", text, re.MULTILINE)` over keys `("jaccard", "genes_above_floor", "neg_cre_fraction", "neg_anno_fraction")`.
  - `_parse_bands` (:241-297): finds `## Thresholds and tolerance bands`, matches table rows by substring in columns[1], extracts `[a, b]` intervals; the gene row via `>=\s*([0-9]+)\s*genes`.
  - CRE cell 21 / Anno cell 27 parse with `line.startswith(f"| {metric_label} |")` — additive rows are invisible to them (safe).
- exon_f1 today is asserted ONLY as an evidence key (`"exon_f1=" in stream`, :677-683, with the comment "exon_f1 carries no Phase-7 band (selection.md records the observed 0.7522 only)"). Anno cell 27 prints the row label `"exon_f1 (evidence only, no band)"` — if selection.md gains an exon_f1 band row, decide whether that notebook wording goes stale (see Open Questions).

## Frozen Pairs — Provenance and Regeneration Feasibility

[VERIFIED: tests/test_genomic_intervals.py:571-590] The original pair is GONE:
```
# Quick-261010-wx2 recorded oracle: the 51/94-row CRE-selection pair lived
# at example/notebooks/plant_helixseek_shared/.scratch/{pred,truth}.bed,
# which is gitignored (.gitignore .scratch/) and absent from disk — the row
# corpora are unrecoverable in-repo, so only the aggregates survive. The
# real-pair in-band gate is Phase 16 / WPAR-05 and is NOT recomputed here.
FROZEN_PAIR_INTERSECTION = 15872
FROZEN_PAIR_UNION = 48878
FROZEN_PAIR_JACCARD = 0.32472687098490116  # numpy full precision; bedtools prints 6 dp
```
Confirmed absent: `example/notebooks/plant_helixseek_shared/.scratch/` does not exist on disk. The baked anchor asserts `round(FROZEN_PAIR_JACCARD, 6) == 0.324727`.

**Regeneration on this box is feasible:**
- bedtools **v2.31.1** (the exact parity-record version) exists at `/home/forrest/github-runners/dnallm-nightly/_work/DNALLM/DNALLM/.bedtools-env/bin/bedtools` — the nightly job's cached rootless prefix. It is NOT on PATH; export it or micromamba-install rootless (ci.yml's own recipe, lines 560-578 / 759-777). [VERIFIED: executed `--version` this session]
- pybedtools 0.12.1 imports in the repo `.venv` and runs green with that PATH addition (verified by the adapter probe). [VERIFIED]
- `from dnallm.utils import jaccard, loj_intersect, parse_bed` imports green in the repo venv. [VERIFIED]
- CRE truth side: deterministic from the committed `TAIR10_DHSs_chr1_5100001_5300000.gff` (cell 5/9 logic — verbatim rows, `gff1_to_half_open`, sorted). No model needed.
- CRE pred side (51-row class): model output — requires running the CRE notebook (or just its scan/peak cells) on this GB10 box (GPU + modelscope checkpoint caches; the box is the nightly runner, HF_ENDPOINT mirror env may be needed per ci.yml:695). The regenerated pred bed will NOT be byte-identical to the lost original (transformers drift) — that is fine: the frozen oracle values are re-measured in the SAME session as regeneration (bedtools jaccard AND numpy jaccard over the new pair), and selection.md's loose band `[0.3, 1.00]` still governs the nightly re-execution.
- Anno pair (exon_f1): truth = 526 CDS rows from the committed `TAIR10_GFF3_chr1_5100001_5300000.gff3` (Anno cell 5); pred = model-derived segments (394-row class) from an Anno notebook run on this box. exon_f1 over the pair is integer-count rational arithmetic (greedy reciprocal-overlap match, Anno cell 17 — pure Python, no bedtools).
- NER loj pair: the script's own `rice_genes_tokens_BPE.bed` (A) + `rice_annotation.bed` (B) from a `generate_bpe_dataset.py` run — rice inputs must download (census cache `example/.scratch/census-out/inputs/` is cold; rice.uga.edu is flaky, the typed network-unavailable skip exists). A real subset at the 51/94-row scale suffices (the loj contract — join multiplicity, null fills, A-order — is what the fixture exercises, not genome scale).
- `tests/fixtures/` does not exist yet (new directory). `*.bed` and `*.gff`/`*.gff3` are NOT in .gitignore's data patterns (only `*.fasta *.fa *.fna *.fastq *.fq *.sam *.bam *.gz *.zip *.pkl *.fxi *.csv *.tsv *.token_sizes` are ignored) — BED fixtures commit with no .gitignore change. [VERIFIED: .gitignore:70-90]

## CI — test-windows and the Nightly Lanes

[VERIFIED: .github/workflows/ci.yml read in full]

- **test-windows** (:133-210): `windows-latest`, py3.12, bash shell, installs `.[base]` ONLY (no dev extra → no pybedtools; pyfastx also absent), runs `pytest -m "not slow" --cov --junitxml=pytest-junit.xml` then `python scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml` (skip audit is a hard gate there too) + exit-code canary. **Answer to research Q8: the example-execution tests are all slow-marked, so test-windows NEVER runs the three gates' execution tests today — only their fast contract tests** (`TestScriptGateContracts`, `TestActiveNotebookGates`) and `tests/test_genomic_intervals.py` (phase-15, platform-neutral) already run there. The WPAR-05 frozen-pair tests belong at `tests/` root, unmarked (fast) and platform-AGNOSTIC (numpy is core everywhere): they then execute on test-windows, every Linux matrix leg, coverage-gate, and both nightlies automatically, with no platform conditional and no new CI step. This is also what "frozen metrics hold in band on BOTH paths" wants — the same assertion file proves the numpy path on Windows and Linux in one artifact.
- **Census pin** (D-03): example-nightly stage 0.5 (ci.yml:854-871) asserts the collect-only triple `216/217 tests collected \(1 deselected\)` for `pytest tests/examples --collect-only -q -m "not giants"`. **Any test added under tests/examples/ changes this triple and requires the same-change ci.yml edit** (repo convention: "one census-moving edit per phase (D-03 triple + expected_skips.yaml + ci.yml stages same-change)" — STATE.md decision log). Placing new tests at tests/ root avoids the triple entirely.
- **Nightly re-execution for WPAR-08/SC4** ("Linux nightly re-execution holds the parsed selection.md bands in the same change"): BOTH self-hosted nightlies re-execute the bands — coverage-nightly (03:00 UTC, full census incl. slow, bedtools rootless step at :544-578) and example-nightly (05:30 UTC, stage 1 runs `pytest tests/examples -m "not giants"`, bedtools rootless at :747-777, `. [base,fla,dev,mcp]` venv so pybedtools present). The dispatch edits must keep `test_cre_notebook_executes_within_selection_bands`, `test_anno_notebook_executes_within_selection_bands`, the NER notebook lane, and the script lane green on Linux with pybedtools+bedtools — i.e., the Linux branches must remain byte-equivalent in behavior.
- test-windows has no bedtools and none should be added (WPAR-05 is about the numpy path there).

## Docs Sync (check_docs_sync) — WPAR-08 Mechanics

[VERIFIED: scripts/check_docs_sync.py read in full; .github/workflows/docs-validation.yml:49]

- `python scripts/check_docs_sync.py` compares `example/` vs `docs/example/` recursively, **byte-identical** (`filecmp.cmpfiles(..., shallow=False)` — always reads bytes). Ignore set: `{"__pycache__", "logs", "outputs", "outputs_multilabel", "benchmark_results", ".ipynb_checkpoints", ".gitignore", ".scratch"}` + suffixes `(".gz", ".log")` + a path-shaped `.pdf` exemption under `notebooks/<dir>/` (both sides) + docs-only `.md` wrappers (right_only). 
- Enforcement: docs-validation.yml runs it on every push/PR to main/master/dev — **the mirror must be byte-identical in the SAME commit as each example edit** (mirrors are separate files to copy, not generated at CI time). Unit tests of the checker live at tests/scripts/test_check_docs_sync.py.
- Consequence for Phase 16: every edit to `generate_bpe_dataset.py`, `data_generation_and_inference.ipynb`, `plant_helixseek_cre.ipynb`, `selection.md` (and the Anno notebook if touched) needs its `docs/example/...` twin updated in the same change. Mirrors verified byte-identical today, so a straight copy is safe.

## Standard Stack

No new packages. This phase consumes already-locked surfaces:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.2.0 (CI matrix pin) | interval math under the evaluators | core dependency, already on every leg incl. test-windows [VERIFIED: pyproject + ci.yml:45] |
| dnallm.utils.genomic_intervals (in-repo, phase-15) | current | `jaccard` / `loj_intersect` / `parse_bed` — the Windows dispatch target | WPAR-01 oracle-pinned, coverage-proven, import-pure numpy-only [VERIFIED: dnallm/utils/genomic_intervals.py; tests/test_genomic_intervals.py] |
| pybedtools (Linux only) | 0.12.1 installed / `>=0.11.0; platform_system != 'Windows'` declared | the unchanged Linux truth path | dev-extra platform marker already excludes Windows [VERIFIED: pyproject.toml:89; .venv probe] |
| bedtools binary (Linux only) | v2.31.1 (nightly-cached prefix) | the unchanged Linux truth chain | owner directive: frozen truth stays bedtools-computed [VERIFIED: runner prefix execution] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyfastx | `>=2.2.0` (unconditional, dev extra) | genome FASTA indexing in the NER script/notebook | unchanged; present on Windows dev installs — the Windows dispatch only replaces the interval join, not FASTA reading [VERIFIED: pyproject.toml:90] |

**Installation:** none — no `pip install` changes belong in this phase. (WPAR-07's triton-windows declaration is Phase 18.)

## Package Legitimacy Audit

No external packages are installed or recommended by this phase; every dependency is already pinned in pyproject.toml and verified present in the repo environment this session. Audit not applicable.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

## Architecture Patterns

### System Architecture Diagram

```
                     EXAMPLE ACQUISITION SEAMS (runtime platform detection)
                     ┌────────────────────────────────────────────────────────┐
                     │ generate_bpe_dataset.py:7,249                          │
                     │ data_generation_and_inference.ipynb cells 0/1/2/8     │
                     │ plant_helixseek_cre.ipynb cell 11                      │
                     └───────────────┬────────────────────────────────────────┘
                                     │ sys.platform == "win32"?
                        ┌────────────┴─────────────┐
                        │ no (Linux/macOS)         │ yes (Windows)
                        ▼                          ▼
              pybedtools.BedTool            dnallm.utils.loj_intersect / jaccard
              .intersect(loj=True)          (pure numpy, in-process)
              bedtools jaccard (CLI                │
              subprocess, CRE cell 11)             │ list[str] tab-joined
                        │                          ▼
                        │               UNIFORM ROW ADAPTER (split on \t)
                        │              .chrom=f[0] .start=int(f[1])
                        │              .end=int(f[2]) .name=f[3] .fields=f
                        ▼                          │
                        └──────────┬───────────────┘
                                   ▼
                     CONSUMER (byte-identical both paths)
                     tokens_to_nerdata: iv.fields[4]/[9]/[10],
                     name == "-1" intergenic check, gene flush
                                   │
                                   ▼
                     rice_gene_ner_BPE.pkl / jaccard= stream line
                     
  ───────────────────────── TEST/GATE LAYER ─────────────────────────
  three gates (script :130, notebook :1397, showcase :523)
        │ platform-aware: Linux probe-then-execute (unchanged) 
        │                 Windows → numpy path, fail-loud (no skip)
        ▼
  tests/fixtures/*.bed (committed frozen real pairs, 51/94-row class)
        │
        ▼
  frozen-pair tests (tests/ root, platform-agnostic, fast)
        parse selection.md bands → numpy jaccard/loj/exon_f1 in band
        (runs on test-windows + every Linux leg)
        +
  tests/test_dispatch_direction.py (structure test: no except-fallback,
        numpy path cannot silently execute on Linux)
```

### Pattern 1: Acquisition-seam dispatch (owner directive shape)
**What:** At the seam only, branch on `sys.platform == "win32"`; the Linux branch keeps the exact current code; the Windows branch calls the phase-15 evaluator. No try/except ImportError ladder — an ImportError on Linux must FAIL, not fall back (WPAR-06 makes this a structure test).
**When to use:** exactly the inventoried seams; nowhere else.
**Example:** see Code Examples.

### Pattern 2: Probe-then-execute gating (existing repo pattern, extend to platform-aware)
**What:** Gate functions probe (find_spec / shutil.which) and either return (execute for real) or emit a typed skip with probe evidence. Phase 16 adds the third outcome: on Windows, skip the probe and proceed into the numpy dispatch.
[VERIFIED: tests/examples/test_notebook_execution.py:1293-1419; tests/examples/_execution.py:1197-1240]

### Pattern 3: Frozen-value band pinning via single truth source
**What:** selection.md holds observed values (`^key=` lines) and the band table; tests PARSE them (never copy literals — D-07); parse guards fail red on missing keys/rows (D-06). The new exon_f1 row joins the band table; the tight tolerance is recorded there and measured, not guessed.
[VERIFIED: selection.md:54-75; test_plant_helixseek_showcase.py:216-297]

### Pattern 4: Platform-neutral evaluator tests (phase-15 precedent)
**What:** tests/test_genomic_intervals.py runs "on every CI leg — no bedtools binary required" with baked oracle literals; the file's header forbids skip machinery and custom markers. The new frozen-pair tests copy this posture (plus the selection.md parse, which that file deliberately lacks).
[VERIFIED: tests/test_genomic_intervals.py:1-25]

### Anti-Patterns to Avoid
- **except-fallback dispatch** (`try: from pybedtools import BedTool / except ImportError: <numpy>`): explicitly prohibited by WPAR-06 — it would let the numpy path silently execute on a Linux box whose pybedtools broke, corrupting the bedtools truth chain. Use platform detection only.
- **Sorting or deduplicating loj rows** "to be safe": the consumer's gene-flush depends on A-file order and join multiplicity; both paths already emit it — any normalization breaks parity.
- **Re-encoding null fields** (`.`→`-1` or vice versa): the `-1` score slot is load-bearing (KeyError analysis above).
- **Asserting the tight exon_f1 band in the nightly model re-execution**: the nightly recomputes from live model output with transformers drift; a ~1e-9 band would false-red. Tight band = frozen-pair tests only; nightly keeps the four existing drift bands (+ evidence key).
- **Touching md cells 0/10/22 of the CRE notebook**: frozen-convention docs stay bedtools-worded (owner directive + structure tests).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Interval join/jaccard math | any new interval code | `dnallm.utils.loj_intersect` / `jaccard` | oracle-pinned over 5008 bedtools v2.31.1 cases; null-fill/multiplicity/order contracts are empirically pinned where "the obvious reading is wrong" [VERIFIED: module docstring] |
| Row adapter beyond split('\t') | parsing BED into objects, dtype inference | one tiny adapter class/namedtuple over the tab-split line | pybedtools row surface is exactly chrom/start/end/name/fields (verified empirically); anything more is surface area for divergence |
| Skip/audit machinery | ad-hoc pytest.skip strings | `optional_dep_skip` / `environment_unavailable_skip` + expected_skips.yaml prefixes | audit_skips fails CI on unmatched skip messages |
| Band assertions | hardcoded tolerances in tests | parse selection.md (D-07 pattern) | single truth source, parsed on both paths (WPAR-05 wording) |

**Key insight:** every piece of interval semantics this phase needs already exists and is oracle-locked; the phase is plumbing (dispatch, gates, fixtures, one band row) — the risk is in the plumbing details (null literals, ordering, mirrors, census), not the math.

## Common Pitfalls

### Pitfall 1: The `-1` null score slot is load-bearing
**What goes wrong:** An adapter or test that treats no-hit B fields as "empty" or normalizes `.`/`-1` breaks the consumer.
**Why:** `iv.fields[10]` on a no-hit row is `"-1"` (count-6 null score slot). The consumer's `if name == "-1"` keys on it; a `"."` there KeyErrors on the second consecutive intergenic token (`zero_map[".0"]` missing).
**How to avoid:** pass fields through verbatim; pin a contract test with two consecutive no-hit rows (the phase-15 corpora already cover count-6 nulls — extend to the adapter).
**Warning signs:** KeyError `".0"` / `".1"` in NER labeling; silently all-`O` labels.

### Pitfall 2: `iv.start` is int but `iv.fields[1]` is the string
**What goes wrong:** adapter exposing `fields` with ints (or `.start` as str) diverges from pybedtools.
**Why:** verified empirically: `.chrom` str, `.start`/`.end` int, `.fields` = all 12 columns as strings.
**How to avoid:** mirror exactly: `start=int(f[1])`, `fields=f`.
**Warning signs:** TypeError in `token_id = (start, end)` set membership across paths.

### Pitfall 3: Docs mirror lands late
**What goes wrong:** example edit committed without the `docs/example/` twin → docs-validation.yml red on the PR.
**Why:** check_docs_sync is byte-identical, shallow=False, on push/PR.
**How to avoid:** same-commit mirror copy for every touched example file (script, both notebooks, selection.md).
**Warning signs:** "DIFFER: notebooks/..." from scripts/check_docs_sync.py.

### Pitfall 4: Census triple moved
**What goes wrong:** new Windows-branch contract tests added under tests/examples/ change the 216/217 collect-only pin; example-nightly stage 0.5 goes red.
**Why:** ci.yml:867 hard-greps the triple; repo convention requires the same-change triple+expected_skips+ci.yml edit, once per phase.
**How to avoid:** put new test files at tests/ root (`tests/test_dispatch_direction.py` is already so placed by decision; keep frozen-pair tests there too). If contract tests must extend existing tests/examples files, count the parametrization delta and re-pin in the same change.
**Warning signs:** stage-0.5 "census collection mismatch".

### Pitfall 5: Cell 1 (`!which intersectBed`) breaks Windows execution
**What goes wrong:** the notebook dispatches fine in cells 2/8 but cell 1's shell magic errors on Windows (`which` absent) → errored outputs → the (future/local) Windows execution test fails.
**Why:** the WINDOWS #19 inventory lists cells 0/2/8; cell 1 is adjacent but on the execution path.
**How to avoid:** fold cell 1 into the cell-0 platform conditional (Claude's discretion explicitly covers "notebook install-cell conditionals").
**Warning signs:** error output at cell 1 in any Windows-path execution.

### Pitfall 6: Tight exon_f1 band asserted against model drift
**What goes wrong:** nightly Anno re-execution fails after a transformers bump because a 1e-9-class band was wired into the nightly assertion.
**Why:** the nightly recomputes exon_f1 from live model predictions; the existing bands are drift-wide by design ("substantially consistent — never exact outputs", selection.md:75).
**How to avoid:** the new tight band is consumed by the frozen-pair tests only; the nightly keeps `genes_above_floor >= 3` + evidence key. Surface this as an owner-call if the nightly should ALSO gain a drift-wide exon_f1 band (out of the pinned wording).
**Warning signs:** nightly red with `exon_f1=0.74x outside [..., ...]`.

### Pitfall 7: Fixture regeneration session loses provenance
**What goes wrong:** fixtures committed with oracle values measured in a different session/environment than the pair was generated in → values don't reproduce.
**Why:** pred sides are model-derived; bedtools AND numpy values must be measured over the exact committed bytes in one session (bedtools from the v2.31.1 prefix).
**How to avoid:** one regeneration script/session producing pair + both oracle values + the selection.md row together; document provenance in the fixture directory (selection.md audit-section precedent at selection.md:127-153).
**Warning signs:** frozen-pair test failing only on some legs (all legs run it — a cross-leg failure is exactly this).

### Pitfall 8: CRE cell 11 output-format drift
**What goes wrong:** the Windows branch prints with different precision/format → nightly stream regex or cell 21 comparison breaks.
**Why:** `^jaccard=([0-9.eE+-]+)` and cell 21 parse the SAME printed line; bedtools prints 6 dp, numpy returns full precision — the print format `:.4f` is the normalization point.
**How to avoid:** keep `print(f"jaccard={jaccard_value:.4f}")` on both branches, computing `jaccard_value` differently.

### Pitfall 9: loj performance on the full NER run (Windows user-facing, not CI)
**What goes wrong:** a real Windows user running the example waits minutes-tens-of-minutes at cell 8.
**Why:** `loj_intersect` broadcasts per-A-row over all same-chrom B rows (spike-scale by design); the full NER run at sampling=2000 genes is ~10^5-10^6 A rows vs ~10^4-10^5 B rows per chrom.
**How to avoid:** accept it (example correctness over asymptotics; CI exercises frozen pairs, not full runs) — or note the cost in the notebook cell comment. Not a blocker; planner judgment.
**Warning signs:** Windows-local full-run wall time (not observable in CI).

## Code Examples

### The uniform row adapter (contract verified empirically this session)

Probe executed with pybedtools 0.12.1 + bedtools v2.31.1 (nightly prefix) + the phase-15 module:

```
--- pybedtools loj iteration (a.bed = 3-row BED6 tokens-like, b.bed = 2-row BED6 annotation-like) ---
chrom='Chr1' start=100 end=200 name='ATCG' fields=['Chr1', '100', '200', 'ATCG', 'geneA', '+', 'Chr1', '150', '260', 'geneX', 'exon', '+']
  types: str int str
chrom='Chr1' start=250 end=300 name='GG' fields=['Chr1', '250', '300', 'GG', 'geneA', '+', 'Chr1', '150', '260', 'geneX', 'exon', '+']
chrom='Chr1' start=500 end=560 name='TTTT' fields=['Chr1', '500', '560', 'TTTT', 'geneB', '-', '.', '-1', '-1', '.', '-1', '.']
--- numpy loj_intersect lines ---
'Chr1\t100\t200\tATCG\tgeneA\t+\tChr1\t150\t260\tgeneX\texon\t+'
'Chr1\t250\t300\tGG\tgeneA\t+\tChr1\t150\t260\tgeneX\texon\t+'
'Chr1\t500\t560\tTTTT\tgeneB\t-\t.\t-1\t-1\t.\t-1\t.'
```

Adapter (sketch — placement is Claude's discretion; if it lands in `dnallm.utils`, it joins the tested public surface):

```python
class LojRow:
    """pybedtools-row emulation over one loj_intersect output line."""
    __slots__ = ("chrom", "start", "end", "name", "fields")

    def __init__(self, line: str) -> None:
        f = line.split("\t")
        self.chrom = f[0]          # str  — A col 1
        self.start = int(f[1])     # int  — A col 2 (pybedtools exposes int)
        self.end = int(f[2])       # int  — A col 3
        self.name = f[3]           # str  — A col 4 (BED name)
        self.fields = f            # list[str] — ALL columns verbatim, start/end as strings
```

Source: empirical probe this session (pybedtools 0.12.1 / bedtools v2.31.1); null literals `[VERIFIED: dnallm/utils/genomic_intervals.py:213-218]`.

### Dispatch at the acquisition seam (shape, not final code)

```python
# generate_bpe_dataset.py — seam at line 7 / 249 (script) and notebook cells 2 / 8
import sys

if sys.platform == "win32":
    from dnallm.utils import loj_intersect  # numpy path; import-pure, no bedtools
else:
    from pybedtools import BedTool          # Linux truth chain — unchanged

def _loj_rows(path_a, path_b):
    """Platform-dispatched loj; yields pybedtools-row-like objects (fail-loud)."""
    if sys.platform == "win32":
        return [LojRow(line) for line in loj_intersect(path_a, path_b)]
    return BedTool(path_a).intersect(path_b, loj=True)
# consumer stays byte-identical: `for iv in _loj_rows(tokens_bed, annotation_bed): ...`
```

No `try/except ImportError` anywhere — that is the WPAR-06 structure-test target.

### CRE cell 11 Windows branch (shape)

```python
import sys
if sys.platform == "win32":
    from dnallm.utils import jaccard as _np_jaccard
    jaccard_value = _np_jaccard(predicted_bed_path, truth_bed_path)["jaccard"]
else:
    result = subprocess.run(
        ["bedtools", "jaccard", "-a", str(predicted_bed_path), "-b", str(truth_bed_path)],
        capture_output=True, text=True, check=True,
    )
    jaccard_value = float(result.stdout.strip().splitlines()[1].split("\t")[2])
print(f"jaccard={jaccard_value:.4f}")   # identical format on both branches (stream regex)
```

### Frozen-pair test skeleton (platform-agnostic, tests/ root)

```python
# tests/test_frozen_pairs.py (name planner's choice) — phase-15 posture: no skips, no markers
from pathlib import Path
from dnallm.utils import jaccard, loj_intersect

FIXTURES = Path(__file__).parent / "fixtures" / "genomic_intervals"

def test_frozen_cre_pair_jaccard_in_band():
    # bands parsed from selection.md (D-07: never copy literals)
    result = jaccard(FIXTURES / "cre_pred.bed", FIXTURES / "cre_truth.bed")
    low, high = _parse_selection_band("Anno/pinned exon_f1 row label TBD")  # parse guard
    assert low <= result["jaccard"] <= high
```

Pair regeneration (dev-box session, one script): export the runner-prefix bedtools PATH, run the CRE notebook scan (or reuse a nightly-run pred bed captured before `outputs/` expiry), freeze pred+truth beds, record bedtools `jaccard` (6 dp) AND numpy full precision in the same session, write the selection.md band row, commit pair + values together.

## WPAR-06 — Dispatch-Direction Structure Test Design Inputs

CONTEXT pins the file: `tests/test_dispatch_direction.py`. Precedents in-repo: structure tests read notebook JSON and AST-walk cells (test_plant_helixseek_showcase.py:366-395 `ast.parse` + Import/ImportFrom walk; :470-495 substring bans). The test should, per file (script text + notebook cells 2/8 + CRE cell 11):
1. Assert the dispatch seam exists and is platform-shaped (`sys.platform == "win32"` or equivalent) with the Linux branch still naming `BedTool`/`bedtools`.
2. Assert NO except-fallback ladder around the pybedtools import (no `try:` ... `except ImportError` enclosing a pybedtools import in the same statement block).
3. (Optional, planner) a runtime half: monkeypatch `sys.platform` to `"linux"` and assert the dispatch helper raises loudly if pybedtools is unimportable, rather than silently using numpy — the "cannot silently execute on Linux" clause. A monkeypatch-to-win32 companion exercises the numpy branch (the SC2 "Windows-branch contract tests exercising the dispatch via monkeypatched platform").

Note on placement of those runtime contract tests: `tests/test_dispatch_direction.py` at tests/ root keeps the examples census triple untouched; if the SC2 monkeypatch-platform contract tests instead extend the three tests/examples gate files, the 216/217 pin moves and ci.yml:867 must be edited in the same change (Pitfall 4).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Windows = typed skip at every bedtools/pybedtools seam (probe-then-skip, 2026-10-08 first Windows exposure) | Windows = native numpy dispatch; skip machinery removed from the Windows path | Phase 16 (this phase) | test-windows becomes an executor of interval math, not a spectator |
| Frozen-pair aggregates only (constants baked in test_genomic_intervals.py) | committed real pairs under tests/fixtures/ | Phase 16 | real-pair in-band gate becomes recomputable on every leg |

**Deprecated/outdated:** nothing in-repo deprecates in this phase; the inventory line `test_script_execution.py:128` is stale (gate now at :130) — the plan should cite current lines.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sys.platform == "win32"` is the platform predicate the owner intends ("runtime platform detection"); alternatives (platform.system()) behave equivalently for this fork | Patterns/Code Examples | None material — any monotonic predicate satisfies WPAR-06 as long as it is not exception-based; confirm at plan time |
| A2 | The regenerated CRE pred bed need not reproduce the lost original byte-for-byte — oracle values are re-measured in the regeneration session and selection.md's loose band still covers nightly drift | Frozen Pairs | If the owner wants the historical 0.324727 exact pair restored, regeneration cannot do it (original bytes gone) — would need owner call |
| A3 | The exon_f1 frozen pair requires a model run on this box (Anno scan) to source the 394 pred segments; no committed artifact carries them | Frozen Pairs | If a committed/nighly-captured pred source exists, fixture work shrinks; verify at plan time |
| A4 | Tests at tests/ root are collected by test-windows's `pytest -m "not slow"` (they are — same tree, no markers) and by coverage-gate/nightlies without count pins outside tests/examples | CI | If any other pinned count exists elsewhere, same-change re-pin needed (none found in ci.yml) |
| A5 | The tight exon_f1 band is asserted by frozen-pair tests only, not the nightly model re-execution (interpretation of "hold in band on BOTH paths") | Pitfall 6 | If the owner intends a nightly exon_f1 band too, it must be drift-wide (owner call; current Anno cell 27 wording says "evidence only, no band") |
| A6 | loj frozen pair may be sourced from the same regeneration session (CRE-scale real rows exercising null fills/multiplicity) rather than a full NER run (rice download flakiness) | Frozen Pairs | If the owner requires NER-pipeline-sourced rows, the rice-download step (with typed network-unavailable fallback) joins the fixture task |

## Open Questions (RESOLVED)

All five questions are dispositioned at plan time — each item below is tagged with the plan task/assumption that adopted its recommendation. No open question remains for execution.

1. **exon_f1 band row label + consumers** — RESOLVED → 16-03 Task 2 + 16-03 AS-5 (distinct label adopted; Anno cell 27 "evidence only" wording left unchanged)
   - What we know: band table rows are matched by substring/startswith parsers in `_parse_bands` and notebook cells 21/27; an additive row is invisible to existing parsers.
   - What's unclear: the exact row label (must not collide with "Anno gene models with exon-F1 >= 0.8" substring matching — `_row()` uses `label in columns[1]`, so the new label must not be a substring of, nor contain, that label ambiguously), and whether the nightly/Anno cell 27 "evidence only, no band" wording gets updated.
   - Recommendation: label it distinctly (e.g. "Anno pooled exon-F1 (frozen pair)"); leave cell 27 wording alone (still true for the notebook's own live comparison); surface the wording question at plan time.

2. **Frozen-pair test placement + file name** — RESOLVED → 16-03 Task 3 (one file `tests/test_frozen_pairs.py` at tests/ root, phase-15 posture; parse helper duplicated, not imported)
   - What we know: tests/ root avoids the census pin; tests/examples moves it.
   - What's unclear: owner preference for one file vs per-evaluator files.
   - Recommendation: one file at tests/ root, phase-15 posture (no skips/markers), plus the selection.md band parse helper shared or duplicated from the showcase test.

3. **Adapter location (utils helper vs example-inline)** — RESOLVED → 16-02 AS-1 (inline-in-example, duplicated script + cell 8; locked dnallm.utils surface not extended)
   - What we know: ~8 lines, needed in 2 places (script + notebook cell 8); a `dnallm.utils` helper joins a WPAR-01-locked surface (needs new tests + coverage; module currently numpy+stdlib-only — a pure-python adapter keeps that).
   - What's unclear: whether extending the locked public surface in the same milestone is acceptable to the owner.
   - Recommendation: inline-in-example duplicated twice is lowest-risk for the locked surface; a utils helper is cleaner if the owner accepts the surface growth. Plan-time decision (CONTEXT leaves it to discretion).

4. **Whether the dispatch seam is factored as a testable function** — RESOLVED → 16-02 AS-2 (tiny module-level/cell-level `_loj_rows(path_a, path_b)` helper as the contract-test extraction target)
   - What we know: SC2 wants "Windows-branch contract tests exercising the dispatch via monkeypatched platform"; top-level script code is not importable without executing the whole script.
   - What's unclear: how much refactoring of the example script into functions is acceptable (example code style is currently linear).
   - Recommendation: a tiny module-level helper (`_loj_rows(a, b)`) in the script/notebook, or put the dispatch in the adapter location chosen in Q3 so the contract tests have an import target.

5. **Nightly "in the same change" verification for SC4** — RESOLVED → 16-04 AS-1 + 16-04 Task 1 (manual `workflow_dispatch` of the CI workflow over the phase branch is the primary SC4 evidence; scheduled runs are additional)
   - What we know: both nightlies re-execute the bands with pybedtools+bedtools on Linux; the phase's dispatch edits must keep them green; the runner is this box and can be dispatched manually (`workflow_dispatch` is in both jobs' gates).
   - What's unclear: whether the owner wants a dispatch-triggered nightly verification inside the phase execution or accepts the next scheduled run.
   - Recommendation: plan a manual `workflow_dispatch` of example-nightly (or at least coverage-nightly) as the phase's SC4 evidence step; memory notes carry the recovery manual for this runner.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| bedtools binary (Linux) | fixture regeneration, Linux truth chain, nightlies | ✓ (not on PATH) | v2.31.1 at `/home/forrest/github-runners/dnallm-nightly/_work/DNALLM/DNALLM/.bedtools-env/bin/bedtools` | rootless micromamba/bioconda per ci.yml recipe |
| pybedtools | Linux truth path, fixture-session oracle | ✓ (repo .venv) | 0.12.1 (declared `>=0.11.0`) | — |
| dnallm.utils evaluators | dispatch target | ✓ | current HEAD | — |
| numpy | evaluators | ✓ | per lockfile (CI pins 2.2.0) | — |
| GPU + modelscope caches (CRE/Anno pred regeneration) | fixture pred sides | ✓ (this box is the nightly runner; GB10) | torch 2.11.0+cu130 env per memory | — |
| rice.uga.edu inputs (NER loj pair option) | NER-pipeline-sourced loj fixture | ✗ cold cache (`example/.scratch/census-out/inputs/` absent; host flaky) | — | CRE-session-sourced loj pair (A6), or typed network-unavailable skip then retry |
| tests/fixtures/ | WPAR-05 | ✗ (new directory) | — | none needed — create; `*.bed` not gitignored |

**Missing dependencies with no fallback:** none blocking.
**Missing dependencies with fallback:** rice inputs (flaky host) — fallback documented above.

## Security Domain

security_enforcement: true, ASVS level 1. This phase adds no auth/session/crypto surface; it REMOVES a subprocess call from the Windows path (attack-surface reduction: no argv construction, no shell, per the module docstring).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `parse_bed` raises loud ValueError naming file+line on malformed rows (fixtures are committed data parsed by it — a malformed fixture fails red, never silently empty) [VERIFIED: dnallm/utils/genomic_intervals.py:76-100] |
| V6 Cryptography | no | — |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Tampering via downloaded rice inputs (existing path, unchanged) | Tampering | 4xx re-raises (never converted to skip/cache hit, WR-04); existing behavior retained [VERIFIED: tests/examples/test_script_execution.py:105-127] |
| Injection via fixture bytes | Tampering | fixtures are in-repo git-tracked data (reviewable); parse_bed fail-loud; no eval/exec on data |
| Prompt/command injection via notebook cells | — | cells are committed repo content not fetched data; check_docs_sync guarantees mirror integrity |

## Sources

### Primary (HIGH confidence)
- In-repo source files read this session (all cited with path:line above): generate_bpe_dataset.py; data_generation_and_inference.ipynb (cells 0,1,2,8,9); plant_helixseek_cre.ipynb (cells 0,5,9,10,11,21,22); plant_helixseek_anno.ipynb (cells 5,17,27); dnallm/utils/genomic_intervals.py; dnallm/utils/__init__.py; tests/test_genomic_intervals.py; tests/examples/test_script_execution.py; tests/examples/test_notebook_execution.py; tests/examples/test_plant_helixseek_showcase.py; tests/examples/_execution.py; tests/examples/test_examples.py; scripts/check_docs_sync.py; .github/workflows/ci.yml; .github/workflows/docs-validation.yml; pyproject.toml; .gitignore; tests/expected_skips.yaml; selection.md
- Empirical probe this session: pybedtools 0.12.1 + bedtools v2.31.1 (runner prefix) + dnallm phase-15 module — loj row-shape and null-literal parity, executed and captured verbatim (cached in research-store, key 3cccfd1b…)
- .planning/phases/16-.../16-CONTEXT.md; .planning/REQUIREMENTS.md (WPAR-03..08); .planning/STATE.md

### Secondary (MEDIUM confidence)
- None — no external documentation was needed; no new packages or external APIs enter the phase.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all existing pins verified in-repo/venv this session
- Architecture (seams/gates/dispatch): HIGH — every seam read at current line numbers; adapter contract verified by execution
- Frozen-pair feasibility: MEDIUM-HIGH — regeneration inputs verified present; the model-run step (pred sides) is feasible on this box but not executed in this research session
- Pitfalls: HIGH — each grounded in read code (KeyError trace, census grep, gitignore rules, mirror diffs)

**Research date:** 2026-10-13
**Valid until:** 2026-11-13 (stable in-repo facts; re-verify line numbers if phases 13-15 artifacts are rebased)
