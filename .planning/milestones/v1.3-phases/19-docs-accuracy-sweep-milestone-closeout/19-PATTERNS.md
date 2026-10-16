# Phase 19: Docs Accuracy Sweep & Milestone Closeout - Pattern Map

**Mapped:** 2026-10-14
**Files analyzed:** 8 logical change-classes (309 sweep files disposed via ~150 ledger rows)
**Analogs found:** 8 / 8

All analogs verified git-tracked (`git ls-files` non-empty for each). This phase creates no new code files — every change is markdown prose, one CI workflow edit, and phase-dir planning artifacts. "Data flow" is therefore docs-transform / config-edit / evidence-record.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| 4 docs-only marimo wrappers (`docs/example/marimo/{benchmark,finetune,inference}/*_demo.md`) + `docs/user_guide/benchmark/advanced_techniques.md` — strict-warning fix | doc | transform (link rewrite) | `docs/example/marimo/benchmark/benchmark_demo.md` lines 246-251 (the current broken See-Also tail) | exact |
| `example/notebooks/overview.md` + `docs/example/notebooks/overview.md` (MIRRORED) + ~23 wrappers + quick_start/user_guide/tutorial/index/installation — cuda130 re-point | doc | transform (install-command re-point) | `docs/getting_started/installation.md` lines 120-126 (current cuda124-leads GPU block, to be flipped) | exact |
| `README.md`, `.github/workflows/README.md` — stale claims + test counts | doc | transform | `README.md:167/:202` (current cuda124-recommended rows — self, edit-in-place) | role-match |
| `.github/workflows/ci.yml` — test-cuda leg re-point + temp `v1.3` push-filter entry | config | CI topology | `.github/workflows/ci.yml` lines 3-30 (branch filter + lesson comments) and 218/252-254/263 (test-cuda leg) | exact |
| `.planning/phases/19-.../19-EVIDENCE.md` disposition ledger (~150 logical-file rows) | evidence record | batch record | `.planning/phases/16-.../16-EVIDENCE.md` (per-assertion table shape) | exact |
| Wave review batches (commits over ~120 prose pages) | doc | review + same-pass fix | `docs/user_guide/continuous_integration.md` (the Class-2 stale-claim exemplar) | role-match |
| Broken-anchor fixes (`docs/getting_started/{gpu_support,install_honesty}.md`) | doc | transform (1-char) | existing link lines gpu_support.md:24, install_honesty.md:7 | exact |
| Optional `mkdocs.yml` (only if nav work proves needed) | config | build config | `mkdocs.yml` — research says NOT needed (fix is 4 docs-only files); touch only if a new page appears | no-change-expected |

## Pattern Assignments

### Strict-clean commit: 4 marimo wrappers + advanced_techniques.md (doc, transform)

**Analog:** `docs/example/marimo/benchmark/benchmark_demo.md` lines 246-251 (edit in place)

Current broken form (repo-root-relative):
```markdown
## See Also

- [Benchmark Notebook](example/notebooks/benchmark.md) — Static tutorial for benchmarking
- [Inference Demo](example/marimo/inference/inference_demo.md) — Interactive inference with attention visualization
- [Benchmark User Guide](user_guide/benchmark/getting_started.md) — Comprehensive benchmark documentation
- [Benchmark API](api/inference/benchmark.md) — API reference for the Benchmark class
```

Fix pattern (proven 15→0 by research): rewrite each target page-relative (`../../notebooks/benchmark.md`, `../inference/inference_demo.md`, `../../../user_guide/...`); warning 14's rotten target re-points to `../../../user_guide/inference/visualization.md`; warning 15 substitutes the GitHub absolute URL, copying the existing in-repo precedent in `docs/user_guide/continuous_integration.md` (which links `.github/workflows/README.md` as `https://github.com/zhangtaolab/DNALLM/blob/main/...`). **HAND-EDIT ONLY — never regenerate** (`scripts/generate_md_from_marimo.py` emits no See-Also section). Never use root-absolute `/example/...` (clears the warning, breaks gh-pages subpath; site_url unset).

**Verify:** the isolated-venv strict recipe (verbatim from RESEARCH §DOCR-03 / 18-02-SUMMARY):
```bash
uv venv /tmp/mkdocs-venv19 --python 3.12
uv pip install --python /tmp/mkdocs-venv19/bin/python \
  "mkdocs-jupyter>=0.24.0" "mkdocs-material>=9.6.1" \
  "mkdocstrings-python>=1.16.10" "click<8.5.1" "marimo>=0.16.3"
PATH="/tmp/mkdocs-venv19/bin:$PATH" mkdocs build --strict   # exit 0
```

### cuda130 re-point surface (doc, transform, ~30 sites)

**Analog:** `docs/getting_started/installation.md` lines 119-126 — the GPU default-order flip

Current (cuda124 leads):
```markdown
# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[cuda124]'

# CUDA 13.0 (requires torch >= 2.9, NVIDIA driver >= 580)
uv pip install -e '.[cuda130]'

# Other supported versions: cpu, cuda126, cuda128
```
Flip so cuda130 leads as recommended; keep lane labels (`# [uv lane]` / `# [pip lane]`, line 110 precedent) on every dnallm/torch block; pip-lane blocks must never promise a cuda index pin (18-02-SUMMARY review item e).

**Mirrored exemplar:** `example/notebooks/overview.md` lines 7 and 25 both carry `uv pip install -e '.[full,notebook,cuda124]'` — edit BOTH sides byte-identical same-commit (Pattern: `scripts/check_docs_sync.py` compares bytes, `shallow=False` at lines 79-81). The 17 docs-side notebook wrappers + 3 marimo wrappers are docs-ONLY (no mirror) — their install commands sit in ```bash blocks invisible to the AST sync gate, so hand-edit is safe and unguarded.

**Re-point honesty rule:** the Turing exception must survive — `docs/getting_started/gpu_support.md:15` ("20xx: cuda124–cuda128, NOT cu130") is correct and every re-pointed generic block keeps/links the GPU-matrix pointer. README:202's extras-table "(recommended)" label moves to cuda130, not a bare string swap. gpu_support.md itself: verify only, do not rewrite.

### `.github/workflows/ci.yml` test-cuda leg + branch filter (config, CI topology)

**Analog:** ci.yml itself — three edit points, all with existing comment discipline:

Branch filter (lines 3-20): the block carries dated lesson comments; the temp `v1.3` entry follows the documented pattern ("2026-10-07 lesson" comment explains matrix exposure for working branches). Add `- v1.3` with its own dated comment + note that removal decision is recorded at closeout. D-19 invariant holds: push never fires the self-hosted nightly legs (each nightly job carries a `github.event.schedule` cron gate — see lines ~24-30).

Test-cuda leg (three synchronized lines — matrix `cuda-version: ['12.4']` at :218 → `['13.0']`; install `uv pip install -e ".[full,cuda124]"` at :263 → `".[full,cuda130]"`; cache key `uv-cuda-${{ matrix.cuda-version }}-` at :252-254 is templated — changes automatically). Update `.github/workflows/README.md:59` ("CUDA versions: 12.1, 12.4") in the same change.

Install-step shape (ci.yml:257-263):
```yaml
      - name: Create virtual environment and install CUDA dependencies
        env:
          UV_HTTP_TIMEOUT: 300
          UV_CONCURRENT_DOWNLOADS: 4
        run: |
          uv venv
          uv pip install -e ".[full,cuda124]"
```

### Disposition ledger `19-EVIDENCE.md` (evidence record, ~150 logical-file rows)

**Analog:** `.planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-EVIDENCE.md` — table shape `| Item | Verdict/disposition |` with bold verdict + one-line evidence per row; class-level rows group mirror pairs and gate-disposed data-file classes (the CONTEXT-chosen "~150 logical-file rows" granularity: `| File (or class) | Verdict | Fix summary |`). Rows land in the SAME commit as the fixes they record.

### Broken-anchor fixes (doc, 1-char transform)

**Sites:** `docs/getting_started/gpu_support.md:24` and `docs/getting_started/install_honesty.md:7`, both currently:
```markdown
(lane asymmetry note](installation.md#two-install-lanes-uv-and-pip--the-tooluvsources-asymmetry))
```
Change `--the-` → `-the-` (mkdocs slugify DROPS the em-dash). Never hand-compute anchors — confirm against `id="two-install-lanes..."` in a built `site/**/index.html`.

## Shared Patterns

### Mirror discipline (apply to overview.md and any both-sides file)
Source: `scripts/check_docs_sync.py` (byte-exact, `shallow=False`); example-side and docs-side edits land byte-identical in the same commit. Verify with `python scripts/check_docs_sync.py`.

### Gate battery (run after every wave; all must stay green)
`pytest tests/test_extras_guard.py` (11 passed), `python scripts/check_docs_sync.py`, `python scripts/check_notebook_md_sync.py`, `python scripts/validate_docs_snippets.py`, `python scripts/validate_yaml.py`; census stable at `216/217 tests collected (1 deselected)` under `pytest tests/examples --collect-only -q -m "not giants"`; fast lane 2556 passed / 1 skipped / 75 deselected.

### Verify-literal honesty (18-01 pitfall, applied to every plan verify step)
Grep-test each verify literal before writing it; prefer `tail -1 | grep -E '[0-9]+ passed' | grep -v failed` for pytest output.

### Evidence-record discipline (16-EVIDENCE / 16-EVIDENCE A.5 triage pattern)
Every CI red at closeout gets classify-before-retry disposition in the ledger; the rice.uga.edu caveat class is owner-accepted with precedent (16-EVIDENCE A.5).

## No Analog Found

None — every change-class has an in-repo analog (docs prose edits are self-analogous; the ledger, CI, mirror, and lane-label patterns all have tracked precedents).

## Metadata

**Analog search scope:** docs/, example/, .github/workflows/, scripts/, .planning/phases/{16,18}-*/; tracked-verified via `git ls-files`
**Files scanned:** ~12 primary analogs; inventory facts taken from 19-RESEARCH (verified there with file:line)
**Pattern extraction date:** 2026-10-14
