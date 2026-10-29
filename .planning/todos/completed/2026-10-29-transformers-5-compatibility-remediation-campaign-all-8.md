---
created: 2026-10-29T01:27:13.274Z
title: "Transformers-5 compatibility remediation campaign: all 8 unsupported presets + support story in docs"
area: general
severity: major
resolves_phase: 27.2
files:
  - dnallm/utils/transformers_compat.py
  - dnallm/models/special/dnabert2.py
  - dnallm/models/special/gpn.py
  - dnallm/models/special/omnidna.py
  - dnallm/models/special/crossdna.py
  - dnallm/models/model_info.yaml
  - scripts/census_compat.py
  - docs/user_guide/model_compatibility.md
  - tests/utils/test_transformers_compat_census.py
  - tests/models/test_census_compat.py
  - tests/models/test_compat_docs_consistency.py
  - tests/models/test_model_info_registry.py
completed: 2026-10-29
status: completed
---

## Problem

The Phase-27 census (transformers 5.17.0, evidence pinned per entry in `dnallm/models/model_info.yaml`)
leaves 8 of 49 pretrained presets unusable or unchecked: 7 `remote-code-broken`
(DNABERT-2, GENA-LM, GENA-LM-BigBird, GENERanno, GPN, MutBERT, Omni-DNA) and 1 `census-blocked`
(CrossDNA). Root causes are measured, per-row, and stable:

1. **DNABERT-2** (`zhihan1996/DNABERT-2-117M`): transformers 5 constructs models on the meta device
   where 4.x used CPU; remote `bert_layers.py:399 rebuild_alibi_tensor` does `torch.Tensor(list)`
   (cpu) x `torch.arange` (meta) — crashes before weights load. No removed API involved.
2. **GENA-LM** (`AIRI-Institute/gena-lm-bert-base`): published checkpoint persists a corrupted
   `bert.embeddings.token_type_ids` buffer (value 3901475131392 vs `type_vocab_size=2`); forward
   crashes on every transformers version. Import was already restored by the
   `_patch_apply_chunking_to_forward` rung; only the data defect remains.
3. **GENA-LM-BigBird** (`AIRI-Institute/gena-lm-bigbird-base-sparse`): remote code imports external
   `deepspeed` for block-sparse attention compute. Not stubbable (stub = silently wrong math).
4. **GENERanno** (`lgq12697/GENERanno-eukaryote-0.5b-base`, retargeted 2026-10-29 quick-261029-c95):
   remote `GenerannoRotaryEmbedding` ROPE_INIT_FUNCTIONS lacks a `'default'` key under tf5.17
   construction (`KeyError: 'default'`); evidence pinned at ms commit `0ae1f40f`. This is the first
   measured appearance of the ROPE failure class research predicted in Phase 27.
5. **GPN** (`songlab/gpn-brassicales`): `ConvNetForMaskedLM` removed wholesale from transformers 5;
   vendoring an architecture is outside the per-name rung discipline. `special/gpn.py` handler
   already wires the external `gpn` package — it is an install-guidance gap, not a code gap.
6. **MutBERT** (`JadenLong/MutBERT-Human-Ref`): remote `modeling_mutbert.py:808`
   `input_shape = input_ids.size()[:-1]` (marked `# modified` by authors) drops the sequence dim for
   every 2-D input; fails identically on transformers 4.x. Model's own forward-contract bug.
7. **Omni-DNA** (`zehui127/Omni-DNA-116M`): remote `configuration_olmo.py:8` imports external
   `olmo` package. `special/omnidna.py` handler already exists — install-guidance gap.
8. **CrossDNA** (`chengCCC/CrossDNA_pretrain`): repo ships no root config.json (checkpoints under
   `8.1M/28.6M/71.6M/519M` subdirs). `special/crossdna.py` already navigates subdirs
   (`_PREFERRED_CHECKPOINT_DIRS`), so dnallM-side loading works; the census-blocked verdict is the
   generic-route census口径, not dnallm usability.

Owner rulings (2026-10-29): severity=major; scope=ALL fixes included, explicitly including the two
"fix-on-behalf-of-authors" items (GENA-LM buffer reset, MutBERT forward wrap) despite the tension
with the honest-census discipline — the resolution is that dispositions stay measured-first: a fix
lands only as a rung + a green re-probe, and the public page then states "works with DNALLM's
built-in compatibility fixes", never a silent "works as-is".

## Solution

Campaign plan (v1.5 candidate phase; v1.4 stays on its merge-close track). Every item ships code +
tests in the same change, and its disposition flips ONLY after a measured green probe re-run
(census discipline D-06). Suggested plan split: 3 plans.

**Plan A — compat rungs (the 4 code fixes)**, all in `dnallm/utils/transformers_compat.py`, each
keyed to its measured failure class, version-guarded, no-op when target absent:
- A1 GENERanno: register a `'default'` ROPE init alias into remote ROPE_INIT_FUNCTIONS dicts at the
  load boundary (the rung discipline's per-name pattern, like the apply_chunking rung).
- A2 DNABERT-2: force CPU instantiation for DNABERT-2-family loads (equivalent of 4.x construction
  path) OR wrap `rebuild_alibi_tensor` device alignment; extend `special/dnabert2.py`.
- A3 GENA-LM (owner-approved author-defect repair): post-load reset of
  `bert.embeddings.token_type_ids` buffer to `torch.arange(type_vocab_size)` — restores the exact
  constructor-intended semantics; document prominently that dnallm repairs this checkpoint defect.
- A4 MutBERT (owner-approved author-defect repair): wrap remote forward to correct `input_shape`
  to `input_ids.shape`; document as author-bug repair.

**Plan B — install-guidance + CrossDNA reporting**:
- B1 GPN / Omni-DNA: keep handlers, upgrade ImportError messages to exact `pip install` lines and
  point to the new docs section; no new deps (gpn/olmo stay external).
- B2 GENA-LM-BigBird: document-only (deepspeed deliberately outside dependency set; state clearly
  why it is not stubbable).
- B3 CrossDNA: align census reporting with the dnallm load path (either census tool learns the
  special-handler route or the page annotates dnallm-usable), so the public page stops implying the
  model is unusable when dnallm loads it fine.

**Plan C — docs/support surface (owner requirement: 设定和说明)**:
- C1 Regenerate `docs/user_guide/model_compatibility.md` after re-probes; wording for fixed models
  = "Works with DNALLM's built-in compatibility fixes (tested)"; bump DOCS_DATA_DATE with the
  re-census.
- C2 New plain-language docs section "可选依赖与支持说明" (optional dependencies & support):
  per-family table of what works out-of-the-box / needs `pip install gpn|olmo` / needs deepspeed /
  repaired-by-dnallm, in biologist-readable prose.
- C3 README supported-models statement + link to the compatibility page.
- C4 TUI/MCP model listings carry the support flag (registry tf_compat_status already provides the
  data; surface it in listing output).
- C5 Update test pins (broken counts shrink as rungs land) and the census lane expectation.

Related backlog (recorded in quick-261029-c95 SUMMARY): lgq12697 bulk ingestion (~130 unregistered
entries) is a separate v1.5 campaign; keep the two phases independent.
