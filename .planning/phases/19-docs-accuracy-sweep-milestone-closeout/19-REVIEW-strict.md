# Phase 19 Strict-Build Review Ledger (19-01)

Per-file disposition ledger for the DOCR-03 strict-clean work (16-EVIDENCE table shape: bold verdict + one-line evidence per row). Rows land in the same commit as the fixes they record. Later 19-REVIEW-*.md files follow this shape; consolidated into 19-EVIDENCE.md by 19-07.

## Provenance

- **Task:** 19-01 Task 1 — 15 warnings to zero (the four-file strict fix), proven RED first.
- **Recipe (reused, not recreated):** `/tmp/mkdocs-venv19` from the research session (research § DOCR-03 "The reproducible verification recipe"): `uv venv /tmp/mkdocs-venv19 --python 3.12` + `uv pip install --python … "mkdocs-jupyter>=0.24.0" "mkdocs-material>=9.6.1" "mkdocstrings-python>=1.16.10" "click<8.5.1" "marimo>=0.16.3"`; `PATH="/tmp/mkdocs-venv19/bin:$PATH" mkdocs build --strict` (marimo CLI must be on PATH — jupytext shells out to it).
- **Toolchain (resolved):** mkdocs 1.6.1, mkdocs-material 9.7.7, mkdocs-jupyter 0.26.3, mkdocstrings 1.0.7 / mkdocstrings-python 2.0.9, marimo 0.25.1, jupytext 1.19.6, click 8.5.0 — identical to the research session's record.
- **RED (2026-10-15, pre-fix, log `/tmp/strict-19-01-RED.txt`):** exit 1 — `Aborted with 15 warnings in strict mode!` The 15 WARNING lines, verbatim byte-record:

```
WARNING -  Doc file 'example/marimo/benchmark/benchmark_demo.md' contains a link 'example/notebooks/benchmark.md', but the target 'example/marimo/benchmark/example/notebooks/benchmark.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/benchmark/benchmark_demo.md' contains a link 'example/marimo/inference/inference_demo.md', but the target 'example/marimo/benchmark/example/marimo/inference/inference_demo.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/benchmark/benchmark_demo.md' contains a link 'user_guide/benchmark/getting_started.md', but the target 'example/marimo/benchmark/user_guide/benchmark/getting_started.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/benchmark/benchmark_demo.md' contains a link 'api/inference/benchmark.md', but the target 'example/marimo/benchmark/api/inference/benchmark.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/finetune/finetune_demo.md' contains a link 'example/notebooks/finetune_binary.md', but the target 'example/marimo/finetune/example/notebooks/finetune_binary.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/finetune/finetune_demo.md' contains a link 'example/notebooks/finetune_multi_labels.md', but the target 'example/marimo/finetune/example/notebooks/finetune_multi_labels.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/finetune/finetune_demo.md' contains a link 'example/notebooks/lora_finetune.md', but the target 'example/marimo/finetune/example/notebooks/lora_finetune.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/finetune/finetune_demo.md' contains a link 'user_guide/fine_tuning/getting_started.md', but the target 'example/marimo/finetune/user_guide/fine_tuning/getting_started.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/finetune/finetune_demo.md' contains a link 'api/finetune/trainer.md', but the target 'example/marimo/finetune/api/finetune/trainer.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/inference/inference_demo.md' contains a link 'example/notebooks/inference.md', but the target 'example/marimo/inference/example/notebooks/inference.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/inference/inference_demo.md' contains a link 'example/marimo/benchmark/benchmark_demo.md', but the target 'example/marimo/inference/example/marimo/benchmark/benchmark_demo.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/inference/inference_demo.md' contains a link 'user_guide/inference/getting_started.md', but the target 'example/marimo/inference/user_guide/inference/getting_started.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/inference/inference_demo.md' contains a link 'api/inference/inference.md', but the target 'example/marimo/inference/api/inference/inference.md' is not found among documentation files.
WARNING -  Doc file 'example/marimo/inference/inference_demo.md' contains a link 'api/inference/visualization.md', but the target 'example/marimo/inference/api/inference/visualization.md' is not found among documentation files.
WARNING -  Doc file 'user_guide/benchmark/advanced_techniques.md' contains a link '../../../CONTRIBUTING.md', but the target '../CONTRIBUTING.md' is not found among documentation files.
```

- **GREEN (post-fix, log `/tmp/strict-19-01.txt`):** exit 0, **0 WARNING lines** — `INFO - Documentation built in 4.49 seconds` (+ `STRICT-CLEAN` from the plan's automated gate).
- **Forbidden mechanics respected:** hand-edit only (no regeneration — `scripts/generate_md_from_marimo.py` emits no See-Also section); no root-absolute `/path` forms (they clear the WARNING but mkdocs logs an absolute-link INFO and break under the gh-pages subpath; `site_url` unset in mkdocs.yml); anchors never hand-computed (Task 2 confirms slugs against the built site's `id=`).

## Per-file rows (Task 1)

| File | Verdict | Fix summary |
|------|---------|-------------|
| `docs/example/marimo/benchmark/benchmark_demo.md` | **FIXED (warnings 1-4)** | See-Also tail hand-edited to page-relative: `../../notebooks/benchmark.md`, `../inference/inference_demo.md`, `../../../user_guide/benchmark/getting_started.md`, `../../../api/inference/benchmark.md` — all 4 warnings cleared |
| `docs/example/marimo/finetune/finetune_demo.md` | **FIXED (warnings 5-9)** | See-Also tail hand-edited to page-relative: `../../notebooks/{finetune_binary,finetune_multi_labels,lora_finetune}.md`, `../../../user_guide/fine_tuning/getting_started.md`, `../../../api/finetune/trainer.md` — all 5 warnings cleared |
| `docs/example/marimo/inference/inference_demo.md` | **FIXED (warnings 10-14)** | See-Also tail hand-edited page-relative + warning 14's rotten target re-pointed: `api/inference/visualization.md` → `../../../user_guide/inference/visualization.md` (`docs/api/inference/` contains no visualization page; the real page is user_guide) — all 5 warnings cleared |
| `docs/user_guide/benchmark/advanced_techniques.md` | **FIXED (warning 15)** | docs-dir escape substituted with the in-repo GitHub-URL precedent: `[CONTRIBUTING.md](../../../CONTRIBUTING.md)` → `https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md` (same form as continuous_integration.md's workflows-README link); root-absolute `/CONTRIBUTING.md` deliberately NOT used |

## Per-file rows (Task 2 — Class-4 broken anchors)

Slug source: `grep -o 'id="two-install-lanes[^"]*"' site/getting_started/installation/index.html` → `two-install-lanes-uv-and-pip-the-tooluvsources-asymmetry` (single dash — mkdocs's default slugify DROPS the em-dash in the heading `Two install lanes: uv and pip — the [tool.uv.sources] asymmetry` at installation.md:213; never hand-computed).

| File | Verdict | Fix summary |
|------|---------|-------------|
| `docs/getting_started/gpu_support.md` | **FIXED (anchor, 1 char)** | `:24` anchor `pip--the-tooluvsources` → `pip-the-tooluvsources`, slug confirmed against the built site's `id=`; rest of the file byte-untouched (18-02's verified record — verify-only file) |
| `docs/getting_started/install_honesty.md` | **FIXED (anchor, 1 char)** | `:7` anchor `pip--the-tooluvsources` → `pip-the-tooluvsources`, same built-site confirmation; no other change |
