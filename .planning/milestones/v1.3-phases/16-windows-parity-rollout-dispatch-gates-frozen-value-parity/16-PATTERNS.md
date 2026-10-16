# Phase 16: Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity - Pattern Map

**Mapped:** 2026-10-13
**Files analyzed:** 12 (new/modified)
**Analogs found:** 12 / 12 (all analogs git-tracked, verified via `git ls-files`)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` (seams :7, :249 + adapter helper) | example script (dispatch seam) | file-I/O / transform | the file itself (Linux branch) + `dnallm/utils/genomic_intervals.py` (Windows target) | exact |
| `example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` (cells 0/1/2/8) | notebook cells (dispatch seam) | file-I/O / transform | `generate_bpe_dataset.py` cells' identical consumer (script :235-294) | exact |
| `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` (cell 11) | notebook cell (jaccard dispatch) | subprocess / request-response | cell 11 itself (Linux branch); numpy target `dnallm.utils.jaccard` | exact |
| `tests/examples/test_script_execution.py` (:130 gate + :148 contracts) | test gate (platform-aware) | probe-then-execute | same file, `_gate_script_pybedtools` (:130-145) + `TestScriptGateContracts` (:148-167) | exact |
| `tests/examples/test_notebook_execution.py` (:1397 gate) | test gate (platform-aware) | probe-then-execute | same file, `_gate_pybedtools` (:1397-1407) + `_ACTIVE_NOTEBOOK_GATES` (:1417-1419) | exact |
| `tests/examples/test_plant_helixseek_showcase.py` (:523 gate) | test gate (platform-aware) | probe-then-execute | same file, `shutil.which("bedtools")` assert (:519-527) | exact |
| `tests/test_dispatch_direction.py` (NEW, WPAR-06) | structure test (AST/text reader) | transform | `test_plant_helixseek_showcase.py:364-395` (ast.parse notebook-cell walk) + `test_genomic_intervals.py` posture | role-match |
| `tests/test_frozen_pairs.py` (NEW, name planner's choice, tests/ root) | test (platform-agnostic frozen-pair) | file-I/O + band-parse | `tests/test_genomic_intervals.py` (posture + frozen anchors :571-590) + `_parse_bands` (showcase :241-297) | role-match |
| `tests/fixtures/genomic_intervals/*.bed` (NEW, committed pairs) | fixture data | file-I/O | none (new directory; `*.bed` not gitignored) | none needed |
| `example/notebooks/plant_helixseek_shared/data/selection.md` (+ docs mirror) (new exon_f1 band row) | data / single truth source | — | same file band table (lines 68-73, parsed by `_parse_bands`) | exact |
| `docs/example/notebooks/...` mirrors (script, both notebooks, selection.md) | docs mirror | — | existing byte-identical twins (verified by diff in RESEARCH) | exact (mechanical copy) |
| (possible) `dnallm/utils/genomic_intervals.py` adapter helper — planner decision Q3 | utility | transform | `LojRow` sketch in RESEARCH § Code Examples; module stays numpy+stdlib-only | research-pattern |

## Pattern Assignments

### `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` (dispatch seam)

**Analog:** the file itself; Windows target verified against `dnallm/utils/genomic_intervals.py`.

**Current Linux seam — import (line 7, keep unchanged on Linux branch):**
```python
from pybedtools import BedTool
```

**Current Linux seam — loj call (line 249):**
```python
    intersection = BedTool(tokens_bed).intersect(annotation_bed, loj=True)
```

**Consumer that must stay byte-identical (lines 258-266, 268, 283):**
```python
    for iv in intersection:
        chrom = iv.chrom
        start = iv.start
        end = iv.end
        token = iv.name
        gene = iv.fields[4]
        gene2 = iv.fields[9]
        name = iv.fields[10]
        token_id = (start, end)
        ...
        if gene != current_gene:        # gene-contiguity flush (A-file order is load-bearing)
        ...
        if name == "-1":                # null score slot — fields ride through VERBATIM
```

**Dispatch shape (from RESEARCH § Code Examples, verified empirically):**
```python
import sys
if sys.platform == "win32":
    from dnallm.utils import loj_intersect
else:
    from pybedtools import BedTool

def _loj_rows(path_a, path_b):
    if sys.platform == "win32":
        return [LojRow(line) for line in loj_intersect(path_a, path_b)]
    return BedTool(path_a).intersect(path_b, loj=True)
```
Adapter contract (probed against pybedtools 0.12.1 + bedtools v2.31.1): `.chrom=f[0]` str, `.start=int(f[1])`, `.end=int(f[2])` int, `.name=f[3]` str, `.fields=f` all-strings 12 cols. Null-B literals `('.', '-1', '-1', '.', '-1', '.')` come verbatim from `loj_intersect` (dnallm/utils/genomic_intervals.py:213-218) — never re-encode.

**Forbidden:** `try/except ImportError` fallback (WPAR-06 structure test), sorting/dedup of loj rows, null re-encoding.

---

### `example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` (cells 0/1/2/8)

**Analog:** script consumer (identical shape, generate_bpe_dataset.py:235-294) + same dispatch shape as above.

- Cell 0 (install) + cell 1 (`!which intersectBed`): wrap both in the same platform conditional — `which` and `brew install bedtools` are POSIX-only; skip both pybedtools install and the verify cell on Windows (CONTEXT grants discretion over "notebook install-cell conditionals").
- Cell 2 (import): conditional import exactly like the script seam above.
- Cell 8 (`tokens_to_nerdata`): identical consumer to script :258-283 — replace only the `BedTool(...).intersect(loj=True)` acquisition; do NOT "fix" the docstring's ascending-order claim into code that sorts (A-file order is the real contract).
- Compatibility constraint: `OPTIONAL_IMPORT_MODULES = ("pybedtools", ...)` in tests/examples/test_examples.py:95 tolerates missing pybedtools in the fast-lane import-exec test — conditional imports stay compatible; a bare top-level `from pybedtools import BedTool` in an exec'd cell would already be tolerated today, but keep imports inside the platform branch.

---

### `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` (cell 11)

**Analog:** cell 11 itself; pattern:

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
print(f"jaccard={jaccard_value:.4f}")   # identical on BOTH branches — stream regex ^jaccard=([0-9.eE+-]+) (showcase test :576) and CRE cell 21 parse it
```
Do NOT touch md cells 0/10/22 (frozen bedtools-worded; structure tests constrain cells 0-1, showcase :340-362).

---

### `tests/examples/test_script_execution.py` — platform-aware gate

**Analog:** `_gate_script_pybedtools` (lines 130-145) — probe-then-execute:
```python
def _gate_script_pybedtools() -> None:
    spec = importlib.util.find_spec("pybedtools")
    if spec is None:
        optional_dep_skip(
            "execute generate_bpe_dataset.py (prerequisites install-gated)",
            evidence="find_spec('pybedtools') is None",
        )
```
Platform-aware extension (third outcome): on `sys.platform == "win32"`, skip the probe and return (proceed to execute via numpy dispatch); on POSIX the probe behavior stays byte-equivalent. Fast contract tests to copy: `TestScriptGateContracts` (:148-167) monkeypatch `importlib.util.find_spec` and assert the typed-skip message — extend with a monkeypatched-platform win32 case (roadmap wording: "Windows-branch contract tests exercise the dispatch via monkeypatched platform"). NOTE (Pitfall 4): adding tests here moves the 216/217 census triple — prefer putting monkeypatch-platform runtime contracts in `tests/test_dispatch_direction.py` at tests/ root instead.

---

### `tests/examples/test_notebook_execution.py` — platform-aware gate

**Analog:** `_gate_pybedtools` (lines 1397-1407) + registration (:1417-1419):
```python
def _gate_pybedtools(nb_name: str) -> None:
    _gate_optional_deps(nb_name, ("pybedtools",))

_ACTIVE_NOTEBOOK_GATES: dict[str, object] = {
    "notebooks/finetune_NER_task/data_generation_and_inference.ipynb": _gate_pybedtools,
}
```
Edit only inside `_gate_pybedtools` (Windows → return, execute numpy path); keep the registration map untouched. Contract-test precedent: `TestActiveNotebookGates.test_pybedtools_gate_*` (:315-341) monkeypatching `_probe_module`.

---

### `tests/examples/test_plant_helixseek_showcase.py` — platform-aware gate

**Analog:** the fail-loud gate (lines 519-527):
```python
    assert shutil.which("bedtools") is not None, (
        "bedtools is not on PATH -- the CRE notebook's jaccard agreement step "
        "requires bedtools v2.31+ (see 'Prerequisites' in "
        "docs/example/notebooks/plant_helixseek_cre.md)"
    )
```
Platform-aware edit: keep the hard assert on POSIX (fail-loud, not skip); on Windows route past the assert into numpy execution. Band assertions after execution (:570-586) stay unchanged (Linux behavior byte-equivalent).

---

### `tests/test_dispatch_direction.py` (NEW, WPAR-06 structure test)

**Analog:** notebook-JSON + AST reader pattern, `test_plant_helixseek_showcase.py:364-395`:
```python
        nb = _load_nb(nb_path)
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = _cell_text(cell)
            ...
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): ...
                elif isinstance(node, ast.ImportFrom): ...
```
Structure-test assertions (per RESEARCH § WPAR-06): (1) each seam (script text + NER cells 2/8 + CRE cell 11) contains a platform-shaped predicate (`sys.platform == "win32"`) whose Linux branch still names `BedTool`/`bedtools`; (2) no `try:`/`except ImportError` ladder enclosing a pybedtools import; (3) optional runtime half: monkeypatch `sys.platform` and assert fail-loud on Linux when pybedtools unimportable. Copy the file header posture from `tests/test_genomic_intervals.py:1-17` (no skips, no markers, platform-neutral).

---

### `tests/test_frozen_pairs.py` (NEW, tests/ root) + `tests/fixtures/genomic_intervals/*.bed`

**Analog (posture + anchors):** `tests/test_genomic_intervals.py` — header docstring (lines 1-17: "No bedtools binary, no subprocess, no skip machinery and no custom markers"), imports (line 26: `from dnallm.utils.genomic_intervals import jaccard, loj_intersect, parse_bed`), frozen-anchor precedent (lines 571-590):
```python
FROZEN_PAIR_INTERSECTION = 15872
FROZEN_PAIR_UNION = 48878
FROZEN_PAIR_JACCARD = 0.32472687098490116  # numpy full precision; bedtools prints 6 dp

def test_frozen_pair_anchor_rounds_to_selection_md_value():
    assert round(FROZEN_PAIR_JACCARD, 6) == 0.324727
```

**Analog (band parsing, D-07):** `_parse_bands`, `tests/examples/test_plant_helixseek_showcase.py:241-297` — section anchor + `_row(label)` substring match on `columns[1]` + `[a, b]` interval regex + `pytest.fail` parse guards (D-06). Duplicate or share this helper; new exon_f1 row label must not substring-collide with "Anno gene models with exon-F1" (`label in columns[1]` matching).

New tests: numpy `jaccard`/`loj_intersect`/exon_f1 over the committed fixture pairs, bands parsed from selection.md, tight exon_f1 band asserted HERE ONLY (never in the nightly — Pitfall 6). Files at tests/ root are unmarked/fast → run automatically on test-windows and every Linux leg (A4).

---

### `selection.md` band row + docs mirrors

**Analog:** existing band-table rows (lines 68-73) and the observed-keys block (lines 56-64, `jaccard=0.3247` / `exon_f1=0.7522` line-start keys parsed by `_parse_floors`, showcase :216-238). New row is additive — existing parsers (showcase `_parse_bands`/`_parse_floors`, CRE cell 21, Anno cell 27) match specific labels only and are unaffected, provided the label is distinct. Leave selection.md:35 and Anno cell 27 "evidence only" wording unchanged (research Open Question 1 recommendation).

Docs mirrors: copy each edited example file byte-identical into `docs/example/...` in the SAME commit (scripts/check_docs_sync.py, `filecmp.cmp(..., shallow=False)`; enforced by .github/workflows/docs-validation.yml:49). Mirrors verified byte-identical today → straight copy is safe.

## Shared Patterns

### Typed skips (Linux gates only)
**Source:** tests/examples/_execution.py:1197-1227
```python
def optional_dep_skip(action: str, evidence: str) -> None:
    pytest.skip(f"optional-dep: {action} ({evidence})")

def environment_unavailable_skip(action: str, evidence: str) -> None:
    pytest.skip(f"environment-unavailable: {action} ({evidence})")
```
**Apply to:** the three gates' POSIX branches only. Windows branch must NEVER skip (WPAR-04: no ever-green skip). Prefixes are allowlisted in tests/expected_skips.yaml — never invent new skip strings (audit_skips is a hard gate on test-windows).

### Platform dispatch (fail-loud, no except-fallback)
**Source:** RESEARCH § Code Examples (empirically verified seam shape)
**Apply to:** script :7/:249, NER cells 2/8, CRE cell 11. Predicate `sys.platform == "win32"` (A1). Structure-tested by tests/test_dispatch_direction.py.

### Verbatim field passthrough (row adapter)
**Source:** dnallm/utils/genomic_intervals.py:213-218 null literals + RESEARCH probe
**Apply to:** any adapter in script/notebook cell 8. `fields` all-strings; `.start`/`.end` int; nulls verbatim (KeyError `".0"` crash analysis, RESEARCH Pitfall 1).

### Probe-then-execute gating
**Source:** tests/examples/test_notebook_execution.py `_gate_ollama_stack` (:1293) and `_gate_pybedtools` (:1397); script `_gate_script_pybedtools` (:130)
**Apply to:** all three gate edits. Probe with evidence; typed skip only on genuine infra-missing; never on wrong output.

### Docs mirror same-commit copy
**Source:** scripts/check_docs_sync.py + tests/scripts/test_check_docs_sync.py
**Apply to:** every touched example file (script, both notebooks, selection.md).

### Selection.md single truth source (D-06/D-07)
**Source:** showcase `_parse_floors`/`_parse_bands` (:216-297)
**Apply to:** frozen-pair tests. Parse, never copy literals; missing key/row fails red.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/fixtures/genomic_intervals/*.bed` | fixture data | file-I/O | New directory; no fixture-data precedent needed (BED not gitignored; regeneration procedure is in RESEARCH § Frozen Pairs, one-session provenance per Pitfall 7) |
| adapter location (utils vs inline) | utility | transform | Owner-discretion (RESEARCH Q3); if utils, it joins the WPAR-01-locked surface and needs its own tests per module conventions |

## Metadata

**Analog search scope:** example/notebooks/{finetune_NER_task,plant_helixseek_cre,plant_helixseek_shared}, tests/, tests/examples/, dnallm/utils/, docs/example/, scripts/
**Files scanned:** ~20 (targeted reads at verified line numbers)
**Tracked-source gate:** all 12 named analog paths verified via `git ls-files -- <path>` (12/12 tracked)
**Pattern extraction date:** 2026-10-13
