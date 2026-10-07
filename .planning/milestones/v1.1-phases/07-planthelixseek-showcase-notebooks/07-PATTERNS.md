# Phase 7: PlantHelixSeek Showcase Notebooks - Pattern Map

**Mapped:** 2026-10-03
**Files analyzed:** 14 (8 new artifacts, 3 mirror copies, 3 modified, 1 expected-no-change)
**Analogs found:** 12 full/partial / 14 (the 2 notebooks' compute cells have no in-repo notebook analog — closest source is the frozen contract + smoke test; see "No Analog Found")

**Tracked-source gate:** every analog path below verified with `git ls-files -- <path>` (non-empty). No `.gsd/capabilities/` or other mirror paths appear.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` | notebook (example) | batch transform (sliding-window inference → decode → metrics) | `example/notebooks/finetune_binary/finetune_binary.ipynb` (narrative/API-walk shape) + `tests/models/test_plant_helixseek_smoke.py` (compute cells) | partial (no existing notebook scans) |
| `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` | notebook (example) | batch transform (both-strand scan → stitch → decode → metrics) | same pair | partial |
| `docs/example/notebooks/plant_helixseek_cre.md` | docs (wrapper tutorial) | n/a (static) | `docs/example/notebooks/finetune_binary.md` | exact |
| `docs/example/notebooks/plant_helixseek_anno.md` | docs (wrapper tutorial) | n/a (static) | `docs/example/notebooks/finetune_binary.md` | exact |
| `docs/example/notebooks/plant_helixseek_cre/{plant_helixseek_cre.ipynb,data/}` | docs (byte-identical mirror) | file copy | existing mirror pairs (e.g. `docs/example/notebooks/finetune_binary/`) verified by `scripts/check_docs_sync.py` | exact (mechanism) |
| `docs/example/notebooks/plant_helixseek_anno/{...ipynb,data/}` | docs (mirror) | file copy | same | exact |
| `docs/example/notebooks/plant_helixseek_shared/data/` | docs (mirror of Phase-6 data dir — closes today's red) | file copy | same | exact |
| `tests/examples/test_plant_helixseek_showcase.py` (or sibling; nightly lane) | test (execution + assertion layer) | request-response (run notebook → parse stream outputs → assert bands) | `tests/examples/test_notebook_execution.py` | exact |
| fast-lane structure tests (kernel-free class in the same/sibling module) | test (static notebook structure) | transform (parse ipynb JSON) | `tests/examples/test_examples.py` | exact |
| `tests/examples/_execution.py` | test infra (MODIFIED: 2 spec entries) | n/a | its own `NOTEBOOK_EXEC_SPECS` + `_NOTEBOOK_EXTRA_INPUTS` in `test_notebook_execution.py:92-96` | exact (self-extension) |
| `mkdocs.yml` | config (MODIFIED: 2 nav entries) | n/a | existing `Examples → Notebooks` nav block (lines 190-216) | exact |
| `scripts/check_docs_sync.py` | utility (MODIFIED, discretionary: `.scratch` → IGNORE) | n/a | its own `IGNORE` set (lines 8-15) | exact (self-extension) |
| `models.lock` | config (MODIFIED, optional: 2 `ms` lines) | n/a | existing entries | exact |
| `tests/expected_skips.yaml` | test config (EXPECTED NO CHANGE — prefixes already registered) | n/a | lines 33-46 of the yaml | n/a (verification only) |

Auto-discovery fact: `tests/examples/test_examples.py:62-70` (`_get_notebook_files`) rglobs `example/notebooks/**/*.ipynb` — **the two new notebooks join the fast-lane JSON/syntax/import tests with zero new code**, and `test_notebook_imports` (line 227) will exec every `ast.Import` node in them (the constraint behind the find_spec guard).

---

## Pattern Assignments

### `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` (notebook, batch transform)

**Narrative analog:** `example/notebooks/finetune_binary/finetune_binary.ipynb` (9 cells, 27,906 bytes committed **with outputs** — 5 output-bearing code cells). Cell order shape: device/env check first, single public-API import cell, then one short "load → use" cell per API step with `# comment` lead lines. First cells (verbatim structure):

```python
# cell 0 — environment check (extend with the D-16 fla guard, see Shared Patterns)
import torch
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
# cell 1 — public API imports only
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer
# cell 2+ — one API step per cell, cwd-relative sibling paths ("./finetune_config.yaml")
```

**Compute-cell analog:** `tests/models/test_plant_helixseek_smoke.py` — this is the exact code that produced the frozen floors. Load cell (adapt lines 220-241):

```python
# Source: tests/models/test_plant_helixseek_smoke.py:47-56, 220-241
task = _registry_task(CRE_REPO_ID)   # parse the packaged model_info.yaml 'finetuned:' entry
cfg = TaskConfig(task_type=task["task_type"], num_labels=task["num_labels"],
                 label_names=task["label_names"], threshold=task["threshold"])
model, tokenizer = load_model_and_tokenizer("zhangtaolab/PlantHelixSeek-CRE", cfg, source="modelscope")
enc = tokenizer([seq], return_tensors="pt", padding=True)
with torch.no_grad():   # MANDATORY — autograd overloads memory; budgets were measured no-grad
    out = model(input_ids=enc["input_ids"].to(model.device),
                attention_mask=enc["attention_mask"].to(model.device))
assert tuple(out.logits.shape) == (batch, 2)   # CRE: softmax → p(CRE) = probs[:, 1]
```

Registry-read helper to adapt (smoke test lines 47-56) — `REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml"`, `yaml.safe_load`, filter `data["finetuned"]` on `model == repo_id`, assert exactly one entry. Registry content: CRE entry at `dnallm/models/model_info.yaml:1641-1648` (`binary`, `num_labels: 2`, `label_names: ["Not CRE", "CRE"]`); Anno entry at lines 1650-1658 (`token`, 17 BILOU labels, `threshold: 0.5`).

**Frozen-rule source (transcribe verbatim, do not redesign):** `example/notebooks/plant_helixseek_shared/data/selection.md:29-36` (CRE: 500/50/50 scan bs=4; bin score = arithmetic mean of class-1 probabilities with coverage assert; mean+1.5σ threshold, merge_gap 50, min_length 50, NO upstream min_score/max_length; bedtools jaccard on sorted 0-based-half-open BEDs parsing col 3; floor ≥ 0.3) and `:54-75` (observed values + band table the notebook prints and the tests parse).

**Metric emission:** every metric as a `key=value` print line at line start (frozen keys `jaccard=`, `genes_above_floor=`, `neg_cre_fraction=`, `neg_anno_fraction=`) — key=value precedent is `_emit_env` in the smoke test (lines 41-44):

```python
sys.stdout.write(f"transformers_version={transformers.__version__}\n")
sys.stdout.write(f"torch_version={torch.__version__}\n")
```

**Committed-with-outputs precedent:** `example/notebooks/embedding_attention.ipynb` — 734,374 bytes in working tree, in the mirror, and at `HEAD` (identical), proving the declared nbstripout filter is inert here and outputs survive commits. Kernelspec: `{"display_name": ".venv", "language": "python", "name": "python3"}`.

---

### `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` (notebook, batch transform)

**Analog:** same narrative + compute sources as the CRE notebook. Differences per frozen contract (`selection.md:38-47`): 8192/4096 both-strand scan bs=1; BOS offset `logits[:, 1:window+1, :]` (smoke test asserts `(1, 8194, 17)` at line 279); minus-strand via `dnallm.utils.sequence.reverse_complement` then the upstream 17-element B↔L permutation (`predict_genome_multigpu.py:97-101`, must be transcribed from upstream per RESEARCH A4); stitching rule; argmax BILOU CDS-run decode; reciprocal-overlap-≥0.5 greedy match; pooled + per-gene exon F1. Structural-validity target: 9-column GFF3, verified against the committed truth slice `example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3`.

---

### `docs/example/notebooks/plant_helixseek_{cre,anno}.md` (docs wrapper)

**Analog:** `docs/example/notebooks/finetune_binary.md` — copy this shape exactly. Frontmatter + full-notebook button + prerequisites (lines 1-20):

```markdown
---
notebook: example/notebooks/finetune_binary/finetune_binary.ipynb
sync_check: true
---

# Binary Classification Fine-Tuning

This tutorial demonstrates how to fine-tune a DNA language model for binary classification, using promoter prediction as an example.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb){ .md-button }

## Prerequisites

Install DNALLM with the fine-tuning extras:

```bash
uv pip install -e '.[base,finetune,cuda124]'
```
```

Ending block — "## Related Tutorials" with GitHub-blob links (lines 99-102). For the CRE page the Prerequisites section must add `uv pip install -e '.[base,fla]'` **plus the bedtools prerequisite** (RESEARCH A2) and a GPU note; both pages carry the illustrative-loci disclaimer (D-03).

**Hard constraints on wrapper code blocks (CI-enforced):**
- Every ```` ```python ```` block must be valid Python — `scripts/validate_docs_snippets.py` (strips magic/comment lines, `ast.parse`s each block).
- Every python-block statement must AST-match a statement in the referenced notebook — `scripts/check_notebook_md_sync.py` (frontmatter `notebook:` is the link key). Do not paraphrase API calls in the wrapper.
- `.md` is a docs-only suffix in `scripts/check_docs_sync.py:21` (`DOCS_ONLY_SUFFIXES = (".md",)`) — wrappers may exist right-only; both-side `.md` (e.g. `selection.md` inside the shared mirror) still must match byte-for-byte.

---

### Docs mirrors: `docs/example/notebooks/plant_helixseek_{cre,anno,shared}/`

**Mechanism analog:** existing mirror pairs; contract enforced by `scripts/check_docs_sync.py`. The comparison core (lines 37-74): recursive `filecmp.dircmp`, `left_only` → "ONLY in example/:" error, `right_only` allowed only for `.md`/ignored names, and **byte-level** `filecmp.cmpfiles(..., shallow=False)` on common files. IGNORE set (lines 8-16) currently: `__pycache__, logs, outputs, outputs_multilabel, .ipynb_checkpoints, .gitignore`; suffixes `.gz`, `.log`. The mirror is RED today (`ONLY in example/: notebooks/plant_helixseek_{anno,cre,shared}`) — this phase mirrors **all three** dirs, including `data/` and the shared dir, not just the executed notebooks.

### `scripts/check_docs_sync.py` (discretionary edit)

Add `".scratch"` to the `IGNORE` set (lines 8-15) — one line, same class as existing runtime-dir entries; keeps local verification honest against gitignored scratch dirt under `plant_helixseek_shared/.scratch/`.

---

### `tests/examples/test_plant_helixseek_showcase.py` — slow execution tests + floors parser (test, request-response)

**Analog:** `tests/examples/test_notebook_execution.py`. Copy these exact shapes:

Imports (lines 46-56):

```python
from tests.examples._execution import (
    EXAMPLE_DIR,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    run_notebook,
    seed_sandbox,
)
```

Slow + timeout class ladder (lines 141-143, 600-601): `@pytest.mark.slow` + `@pytest.mark.timeout(...)` on the class/test. Per-test override precedent for the cell-timeout < test-timeout invariant (lines 587-597 `_TIMEOUT_7200_GATED` + parametrize marks at 612-621). Phase-7 values (D-14, RESEARCH Pitfall 6): CRE `cell_timeout=1200` under `@pytest.mark.timeout(2400)`; Anno `cell_timeout=3600` under `@pytest.mark.timeout(5400)`.

Execution-test body (lines 151-186, adapted — replace the generic no-error assert with floors parsing):

```python
spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
nb = run_notebook(nb_path, sandbox, cell_timeout=spec["cell_timeout"], artifact_dir=artifacts)
errored = [(i, out) for i, cell in enumerate(nb.cells) if cell.cell_type == "code"
           for out in cell.get("outputs", []) if out.get("output_type") == "error"]
assert not errored, f"{nb_path.name} executed with error outputs: {errored}"
assert_tree_clean()
```

Cross-dir sandbox seeding — the exact pattern for `../plant_helixseek_shared/data/*` reads (lines 92-96, 116-117):

```python
_NOTEBOOK_EXTRA_INPUTS: dict[str, list[tuple[Path, str]]] = {
    "notebooks/benchmark/benchmark.ipynb": [
        (EXAMPLE_DIR / "notebooks" / "inference" / "test.csv", "../inference/test.csv"),
    ],
}
extras = _NOTEBOOK_EXTRA_INPUTS.get(nb_path.relative_to(EXAMPLE_DIR).as_posix(), [])
yield seed_sandbox(nb_path.parent, tmp_path, extra_inputs=extras)
```

`seed_sandbox` tuple-extra semantics (`tests/examples/_execution.py:277-294`): `(src, dest_relative_to_sandbox)` copies anywhere under `tmp_path` (sibling escapes included), rejects destinations resolving outside `tmp_path` with `ValueError`; copies **files**, not directories — seed each shared file individually (selection.md + flanking `.fas`/`.gff` for CRE; selection.md + intergenic `.fas`/`.gff3` for Anno).

Floors parser (new code; shape from RESEARCH Pattern 5): parse `example/notebooks/plant_helixseek_shared/data/selection.md` at test startup with `re.search(rf"^{key}=([0-9.]+)", text, re.MULTILINE)` for keys `jaccard`, `genes_above_floor`, `neg_cre_fraction`, `neg_anno_fraction`; missing key → `pytest.fail` (parse-guard, D-06). Stream-text extraction: join `cell["outputs"]` entries with `output_type == "stream"`; observed values regexed with the same `^key=` anchors; assertions named-cause per D-08 (metric, observed, band, re-selection pointer). Band bounds parsed from the selection.md band table, never literals.

**Do NOT** add these notebooks to `ACTIVE_NOTEBOOKS` (`test_notebook_execution.py:66-80`) — census semantics would double-execute 40-90 min per notebook (RESEARCH Open Question 1 recommends the dedicated module).

---

### Fast-lane structure tests (test, transform)

**Analog:** `tests/examples/test_examples.py`. Parametrize + ids + `pytest.fail` style (lines 158-184):

```python
@pytest.mark.skipif(not NOTEBOOK_FILES, reason="No notebook files found")
@pytest.mark.parametrize("nb_file", NOTEBOOK_FILES, ids=lambda p: str(p.relative_to(EXAMPLE_DIR)))
def test_notebook_json_validity(self, nb_file: Path):
    nb = json.loads(nb_file.read_text(encoding="utf-8"))
    assert "cells" in nb, "Missing 'cells' key"
```

Phase-7 structure checks are phase-specific (provenance-cell presence, guard shape, disclaimer strings, ≤ 2 MB size, outputs/vega present in the committed copy, parse-guard for selection.md keys) and run kernel-free — the generic JSON/syntax/import coverage already auto-applies via rglob discovery. Notebook-source reading idiom to reuse (lines 202, 237): `"".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]`.

---

### `tests/examples/_execution.py` (MODIFIED — spec entries)

**Analog:** existing `NOTEBOOK_EXEC_SPECS` entries (lines 100-104, 141-144, and the comment-carrying entries 177-193):

```python
str(EXAMPLE_DIR / "notebooks" / "finetune_binary" / "finetune_binary.ipynb"): {
    "cell_timeout": 3600,
    "extra_inputs": [],
},
```

Add two entries with provenance comments (CRE `cell_timeout: 1200`, Anno `cell_timeout: 3600`; Phase-6 measured actuals ~10-20 / ~30-60 min, 2x headroom per D-14). Key form is `str(EXAMPLE_DIR / ...)` absolute-path strings.

### `mkdocs.yml` (MODIFIED — nav)

**Analog:** `Examples → Notebooks` block (lines 190-216). New entries follow the nested-subsection shape, e.g. a `PlantHelixSeek:` group beside `Fine-Tuning:`/`Inference:`/`Analysis:`:

```yaml
  - Examples:
    - Notebooks:
      - Overview: example/notebooks/overview.md
      - Fine-Tuning:
        - Binary Classification: example/notebooks/finetune_binary.md
```

mkdocs-jupyter stays configured (plugins section) but renders nothing — nav points only at wrapper `.md` files (D-11).

### `models.lock` (MODIFIED — optional)

**Analog:** existing entries — `ms  zhangtaolab/plant-dnabert-BPE                    # tests/finetune/test_trainer_real_model.py (...)`. Add `ms  zhangtaolab/PlantHelixSeek-CRE` / `-Anno` one-liners with test-reference comments (optional this phase; CI-04 in Phase 8 owns it).

---

## Shared Patterns

### Registry-driven model access (all notebook load cells)
**Source:** `tests/models/test_plant_helixseek_smoke.py:47-56, 221-241`
**Apply to:** both notebooks' load cells.
`_registry_task(repo_id)` parses the packaged `dnallm/models/model_info.yaml` (`finetuned:` list) → `TaskConfig(...)` → `load_model_and_tokenizer(repo_id, cfg, source="modelscope")` → `tokenizer([seq], return_tensors="pt", padding=True)` → `with torch.no_grad():` forward with tensors moved to `model.device`. Never `DNAInference` internals or dispatch edits (frozen contract).

### key=value evidence prints (T20-compliant provenance)
**Source:** `tests/models/test_plant_helixseek_smoke.py:41-44`
**Apply to:** first code cell (transformers/torch/fla versions per D-16) and every metric cell (`jaccard=`, `genes_above_floor=`, `neg_cre_fraction=`, `neg_anno_fraction=`). Line-start `key=value` form is what the execution tests regex from stream outputs, matching the frozen keys in `selection.md:58-62`.

### fla hard guard — find_spec + raise, never a bare import node
**Source shape:** `tests/examples/test_notebook_execution.py:499-502` (`_probe_module` find_spec idiom) + RESEARCH Pattern 4.
**Apply to:** first code cell of both notebooks. `tests/examples/test_examples.py:227-272` execs every `ast.Import`/`ast.ImportFrom` node (line 260: `exec(compile(stmt, str(nb_file), "exec"), {})`) on legs without `fla` (hosted `.[base]`, docs-validation `.[test,dev,mcp]`), and `OPTIONAL_IMPORT_MODULES` whitelists only `pybedtools` (line 92). Guard with `importlib.util.find_spec("fla") is None → raise RuntimeError(...)`; read the version via `from importlib.metadata import version` (a call, not an import node). Test-side gates use `pytest.importorskip("fla", reason="environment-unavailable: ...")` (smoke test lines 214-219) — never in-notebook.

### Sandbox seeding + tree-clean tripwire (all kernel-spawning tests)
**Source:** `tests/examples/_execution.py:232-294` (`seed_sandbox`), `:665-722` (`assert_tree_clean`, delta-zero against import-time baseline); fixture shape `tests/examples/test_notebook_execution.py:103-118`.
**Apply to:** both showcase execution tests; extras as `(src, "../plant_helixseek_shared/data/<file>")` tuples.

### Timeout ladder (slow tests)
**Source:** `tests/examples/_execution.py:100-99` (specs carry only cell budgets) + `tests/examples/test_notebook_execution.py:587-597` (strictly-below invariant, override mechanism).
**Apply to:** CRE 1200s cell under 2400s mark; Anno 3600s cell under 5400s mark (D-14).

### Typed-skip prefixes (only sanctioned skips)
**Source:** `tests/expected_skips.yaml:33-46` — `network-unavailable:`, `environment-unavailable:`, `optional-dep:` prefixes already registered; `scripts/audit_skips.py` fails CI on unmatched skips.
**Apply to:** any environment gate the showcase tests add must reuse these prefixes verbatim or extend the yaml. D-16's guard raises (never skips) inside notebooks.

### Wrapper-.md + byte-identical mirror (docs write-back)
**Source:** `docs/example/notebooks/finetune_binary.md` (page shape); `scripts/check_docs_sync.py` (mirror contract); committed-with-outputs precedent `example/notebooks/embedding_attention.ipynb` (734,374 bytes at HEAD = tree = mirror).
**Apply to:** both wrapper pages + all three mirror dirs. After commit, verify output retention: `git cat-file -s HEAD:<nb>.ipynb` in the hundreds-of-KB range and `git show HEAD:<nb>.ipynb | grep -c 'vega'` > 0 (RESEARCH Pitfall 3).

### Error/assertion style (tests)
**Source:** `tests/examples/test_notebook_execution.py:179-185` (collect-then-assert with descriptive message), smoke test `pytest.fail`/`pytest.raises(match=...)` idioms; project convention `ValueError` with matchable messages for helper misuse.
**Apply to:** floors parser (missing key → `pytest.fail` named-cause), band assertions (D-08 message shape: metric, observed, expected band, selection.md observed value + margin, re-selection pointer).

---

## No Analog Found

Files/parts with no close in-repo analog (planner: use RESEARCH.md patterns + frozen contract verbatim):

| Item | Role | Data Flow | Reason / Substitute Source |
|------|------|-----------|---------------------------|
| CRE sliding-window scan cells (batched 500/50/50, bin aggregation, peak calling, bedtools jaccard) | notebook compute | batch transform | No notebook or library module scans windows. Transcribe `selection.md:29-36` verbatim; load/forward shape from smoke test; RESEARCH Patterns 1/3 + Code Examples. bedtools subprocess has no repo precedent — output = header + 1 line, col 3 is jaccard (RESEARCH Don't-Hand-Roll). |
| Anno stitch/decode/match cells (8192/4096 both-strand, B↔L permutation, BILOU argmax runs, reciprocal-overlap match, exon-F1) | notebook compute | batch transform | Same. `selection.md:38-47` is the spec; permutation constants must be transcribed from upstream `predict_genome_multigpu.py:97-101` (RESEARCH A4, cross-check against exon_f1≈0.7522 / genes≈59). |
| altair track plots + gene-model diagrams inside notebooks | notebook visualization | transform | No `example/` notebook uses altair (grep: zero hits in `example/`). Closest conventions: `dnallm/inference/plot.py` (line 10 `import altair as alt`, line 16 `alt.data_transformers.enable("default", max_rows=None)`, `alt.X(...)/alt.Color(...)` construction). Size budget D-12 (~4000 pts, ≤ 2 MB). |
| selection.md floors/bands parser | test utility | transform | New. RESEARCH Pattern 5 gives the compliant shape (regex `^key=` anchors, MULTILINE, parse-guard `pytest.fail`). |

---

## Metadata

**Analog search scope:** `example/notebooks/**`, `docs/example/notebooks/**`, `tests/examples/`, `tests/models/`, `scripts/`, `dnallm/utils/`, `dnallm/inference/plot.py`, `dnallm/models/model_info.yaml`, `mkdocs.yml`, `pyproject.toml` (pytest config), `models.lock`, `tests/expected_skips.yaml`.
**Files scanned:** ~20 tracked files read; all analog paths verified via `git ls-files`.
**Key line-number anchors:** harness specs `tests/examples/_execution.py:100-194`; seeding `:232-294`; `run_notebook` `:297-363`; tripwire `:665-722`; execution-test shapes `tests/examples/test_notebook_execution.py:66-96, 103-118, 141-186, 587-597, 600-647`; fast-lane import exec `tests/examples/test_examples.py:62-70, 227-272`; smoke pattern `tests/models/test_plant_helixseek_smoke.py:41-56, 110-136, 207-279`; wrapper `docs/example/notebooks/finetune_binary.md:1-20, 99-102`; mirror contract `scripts/check_docs_sync.py:8-34, 37-74`; frozen contract `example/notebooks/plant_helixseek_shared/data/selection.md:27-47, 54-75, 79-80`.
**Pattern extraction date:** 2026-10-03
