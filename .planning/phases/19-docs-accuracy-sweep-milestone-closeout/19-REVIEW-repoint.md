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

## Task 2 — getting-started surfaces (installation.md default flip + 4 surfaces)

| File (or class) | Verdict | Fix summary |
|---|---|---|
| `docs/getting_started/installation.md` — GPU-section lead block (was :119-126) | **FIXED** | cuda130 now leads as "(recommended default — requires torch >= 2.9, NVIDIA driver >= 580)"; the cuda124 install command is replaced by the prose older-GPU lane ("Older GPUs (Turing 20xx) are served by the CUDA 12.4–12.8 lanes — see the GPU Support Matrix above"). Before: 1 cuda124 install cmd; after: 0. Lane labels intact. |
| `docs/getting_started/installation.md` — extras table (:262/:265 post-edit) | **FIXED (annotation moved)** | "(recommended)" MOVED from the cuda124 row to the cuda130 row — not a bare string swap. cuda124 row keeps its class description ("Most modern GPUs") minus the label; cuda130 row now reads "NVIDIA (CUDA 13.0, Windows & Linux) — recommended". |
| `docs/getting_started/installation.md` — scenario blocks :307/:455/:472/:495/:571 (post-shift) | **FIXED** | Scenario 2 `.[all,cuda124]` → `.[all,cuda130]` (+ its "using CUDA 12.4/12.4 support" comments → 13.0); Scenario 6 `.[cuda124,mamba]` → `.[cuda130,mamba]` (resolves: mamba torch>=2.6.0,<2.12 ∩ cuda130 torch>=2.9.0,<2.12); Scenario 7 `.[all,cuda124]` → cuda130; Scenario 8 `.[full,cuda124]` → cuda130; troubleshooting :571 `.[cuda124]` → cuda130 ("# Choose based on actual situation" comment kept). |
| `docs/getting_started/installation.md` — 5 sanctioned survivors | **KEPT (verify-only)** | :213 index-mechanics prose (`.[cuda124]`–`.[cuda130]` range), :223 uv-lane enumeration (`.[all,cuda124]`), :255 mamba-combination warning example, :268 Turing note (cu121 retirement), :272 hardware+feature-group note. Untouched. |
| `docs/getting_started/quick_start.md` (:96 block) | **FIXED** | Same lead-flip shape: cuda130 leads as recommended; cuda124 command replaced by the prose older-GPU lane (CUDA 12.4–12.8, pointer to the Installation guide). :328's choose-ONE hardware enumeration kept (Q=1). |
| `docs/user_guide/getting_started.md` (:67) | **FIXED** | `.[cuda124]` → `.[cuda130]`; comment notes the Turing 20xx CUDA 12.4/12.8 lanes. |
| `docs/index.md` (:79) | **FIXED** | Example comment `.[all,cuda124]` → `.[all,cuda130]`; :78's choose-ONE enumeration kept (N=1). |
| `docs/user_guide/tutorial/end_to_end_workflow.md` (:983) | **FIXED** | Dockerfile RUN line `.[full,cuda124] --system` → `.[full,cuda130] --system`. |
| `docs/getting_started/gpu_support.md` | **VERIFY-ONLY, UNTOUCHED** | cuda124 count = 4 (Turing row `cuda124`–`cuda128` intact, Ampere row `cpu`,`cuda124`–`cuda130`, driver chooser). No edit made — the Turing exception survives the re-point. |

Task 2 verify (grep-verified): installation cuda124-total I=6 (5 sanctioned survivors + the
label-stripped extras-table row; band 1..6), cuda130-total T=14 (≥13), cuda124 install-cmds
IC=0 (≤1), recommended-on-cuda124 R=0, gpu_support G=4, quick_start Q=1, ug-getting_started
U=0, tutorial W=0, index N=1, lane labels L=29, census `216/217 tests collected (1 deselected)`
→ CENSUS-STABLE.

**Plan-literal reconciliation (documented):** the plan's action (1) parenthetical marked
:457's `.[cuda124,mamba]` "sanctioned survivor, keep", but its own count arithmetic
overrides that: the 5 listed survivors + the kept table row already total 6 — the band's
exact ceiling — so keeping :457 would land I=7 outside the verify band. Flipped :457 to
`.[cuda130,mamba]` (the action clause's main list ":309, :457, :474, :497, :573 switch
their generic default" includes it; three "exactly 5 sanctioned survivors" statements +
the verification section agree). IC=0 lands under the ≤1 bound; the mamba-combination
example class survives at :255 (warning) and README :177.

## Task 3 — README + full gate battery

| File (or class) | Verdict | Fix summary |
|---|---|---|
| `README.md` install block (:166-172) | **FIXED** | Recommended default is now `uv pip install -e '.[all,cuda130]'` (CUDA 13.0 leads). Stale 5090 line "cuda128 & torch==2.7" DELETED, replaced by the 18-02 GPU-matrix guidance ("driver < 580: cuda128; driver >= 580: cuda130 works"). cuda124 appears only as the spaced "CUDA 12.4–12.8 lanes" prose older-GPU mention with the matrix pointer — no literal cuda124 install command (RC=0). "Other supported versions" enumeration stays verbatim (cpu, cuda126, cuda128, cuda130). |
| `README.md` extras table (:201/:204) | **FIXED (annotation moved)** | "(recommended)" MOVED from the cuda124 row to the cuda130 row; cuda124 row keeps "Most modern GPUs" minus the label. |
| `README.md` examples (:211) | **FIXED** | `.[all,cuda124]` → `.[all,cuda130]` ("Everything + CUDA 13.0" comment). |
| `README.md` test counts (:30/:558/:580) | **FIXED (honest)** | Three stale "1,900+" / "~1,900" claims replaced by the 2556-passed fast-lane baseline (18-02-SUMMARY D4): "2,500+ test cases" ×2 (:30, :558) and "fast lane (2,556 tests)" (:580). |
| `README.md` 3 sanctioned survivors | **KEPT** | :176 hardware-group Warning (enumeration + `.[cuda124,mamba]` combination example, one line), :180 GPU-matrix pointer prose ("Turing 20xx → cuda124–cuda128"), :201 extras-table cuda124 row (label-stripped). R=3 exactly. |

Task 3 gate battery (all green, 2026-10-15):
- README greps: stale-counts A=0, recommended-on-cuda124 B=0, stale-5090 S=0,
  cuda124-total R=3 (≤3), cuda130 T=6 (≥6), cuda124-install-cmds RC=0
- Extras residue gate: `pytest tests/test_extras_guard.py -q --no-cov` → **11 passed**
- Sync gates: `check_docs_sync.py` OK; `check_notebook_md_sync.py` 24 pairs in sync
- Fast lane: `pytest -m 'not slow' --no-cov -q` → **2556 passed, 1 skipped, 75 deselected**
  (exit 0 — exactly the 18-02 D4 baseline, unchanged)
- Census (verified Task 2, still binding — README/docs edits cannot move collection):
  `216/217 tests collected (1 deselected)` → CENSUS-STABLE
- Strict build: `PATH=/tmp/mkdocs-venv19/bin:$PATH mkdocs build --strict` → **exit 0,
  0 WARNING lines** (19-01's zero-warning state survived this plan's edits) → BATTERY-GREEN

**Task 3 execution note:** the first block rewrite dropped `cuda130` from the "Other
supported versions" enumeration, landing T=4 (< the ≥6 gate). Restored the enumeration
verbatim per the plan's "the 'Other supported versions' enumeration stays" and carried
the driver>=580/cuda130 fact on the 5090 line — T=6, gates green. Fixed before commit;
no plan deviation (the plan's own T≥6 arithmetic exposed it).
