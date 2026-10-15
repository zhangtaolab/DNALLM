# Phase 19: cuda130 Re-Point Review Ledger (19-02)

Per-surface ledger for the PKGH-05/DOCR-02 re-point (plan 19-02). Rows land in the SAME
commit as the fixes they record (19-01 ledger discipline, 16-EVIDENCE shape).
Before/after counts are grep-verified at commit time.

Re-point rule (honesty constraint): cuda130 becomes the *default recommendation* everywhere
generic; the Turing exception survives (`gpu_support.md`: 20xx → cuda124–cuda128, NOT cu130 —
verify-only file this plan). No blanket substitution.

## Task 1 — Mirrored headline + wrappers

| File (or class) | Verdict | Fix summary |
|---|---|---|
| `example/notebooks/overview.md` + `docs/example/notebooks/overview.md` (MIRRORED pair) | **FIXED** | Both `uv pip install -e '.[full,notebook,cuda124]'` sites re-pointed to `.[full,notebook,cuda130]` (:9 prose bullet, :25 bash block). Before: 2 cuda124 sites/side; after: 0 cuda124 / 2 cuda130 sites per side. Byte-identical same-commit — `check_docs_sync` OK (byte compare, shallow=False). |
| 19 docs-side notebook wrappers (`docs/example/notebooks/*.md`, ~:17 each: benchmark, data_prepare_finetune, data_prepare_predict, embedding_attention, finetune_binary, finetune_custom_head, finetune_generation, finetune_multi_labels, finetune_NER_data_generation, finetune_NER_task, inference, inference_evo_models, inference_generation, inference_megaDNA, inference_trna, interpretation, lora_finetune, lora_inference, mutagenesis) | **FIXED** | `uv pip install -e '.[full,cuda124]'` → `.[full,cuda130]`, one occurrence per file (pre-verified 22 files × exactly 1 occurrence). Docs-only files — install commands sit in ```bash blocks invisible to the AST sync gate; no mirror. |
| 3 marimo wrappers (`docs/example/marimo/benchmark/benchmark_demo.md`, `docs/example/marimo/finetune/finetune_demo.md`, `docs/example/marimo/inference/inference_demo.md`, :18 each) | **FIXED** | Same `.[full,cuda124]` → `.[full,cuda130]` re-point, install line ONLY — the 19-01 page-relative See-Also tails untouched. |

Task 1 verify (grep-verified): `example/` cuda124 = 0; `docs/example/` `[full,cuda124]` files = 0;
overview cuda130 sites C1=C2=2 per side; `check_docs_sync` OK (MIRROR-INTACT).

Post-Task-1 residue: `docs/example/` cuda124 mentions remain only in… none — the docs-side
wrappers are fully cuda130; `example/` carries zero cuda124 strings.
