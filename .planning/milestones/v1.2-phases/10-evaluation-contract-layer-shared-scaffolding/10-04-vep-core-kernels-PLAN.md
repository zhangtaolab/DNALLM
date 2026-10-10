---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 04
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/inference/vep.py
  - tests/inference/test_vep.py
autonomous: true
requirements: [VEP-01]   # cross-phase: the Phase-10 head start of the requirement that completes and verifies in Phase 11 (B5); ROADMAP Phase 10 success criterion 5
user_setup: []
estimate:
  tokens: 26000
  raw_tokens: 26000
  tasks: 3
  confidence: low   # 0 calibration samples (first phase of v1.2); factor 1 per estimate-calibration

must_haves:
  truths:
    # REV-08 core (goal-backward from ROADMAP SC 5; full VEP-01 verifies in Phase 11)
    - "dnallm/inference/vep.py exists with align_variant implementing the same-slot evaluability rule: ref/alt token sequences must be the same length and differ at exactly ONE token slot; a passing variant returns the slot index plus ref/alt token ids"
    - "A non-evaluatable variant returns evaluatable=False with a machine-readable skip_reason (length-changing allele, multi-slot difference, or no-change) and raises nothing — skips are data, not exceptions (the R1-3e-1 protocol answer: the skip fraction becomes a reportable finding)"
    - "align_variant raises a matchable ValueError when the given ref does not match the sequence bases at pos (input-contract violation) — distinct from the skip path"
    - "pos is 0-based into the sequence (documented); the 1-based VCF coordinate conversion belongs to Phase 11's evaluate_vcf, not this kernel"
    - "The CLM kernel (clm_log_likelihood) computes the full-sequence causal log-likelihood — shifted log_softmax gather sum, the same math as mutagenesis.py clm_evaluate (311-347) — with the scoring formula written into its docstring (protocol declaration)"
    - "The MLM kernel (mlm_slot_log_prob) masks one slot and returns the log-probability of a target token id at that slot — the same mask-and-predict math as mutagenesis.py mlm_evaluate (257-309) — with the formula written into its docstring"
    - "Both kernels run under torch.no_grad() with device placement via a get_model_device helper mirroring mutagenesis.py:241-255, and are deterministic on the same input (tiny-real-model test asserts identical repeated scores)"
    - "The module is self-contained (planner discretion resolved: kernels are adapted inside vep.py with docstring attribution to the mutagenesis kernels — no restructuring of mutagenesis.py under the no-refactors constraint)"
    - "vep.py reaches >=96% line coverage via fast-lane tests using tests/conftest.py fixtures (tiny_real_model, simple_dna_tokenizer) — no network, no real-model downloads, no new skips"
    - "dnallm/inference/__init__.py is untouched (no re-export; import via dnallm.inference.vep) and vep.py docstrings are born with the D-08 terminology ('DNA large language models')"
    - "No evaluate_vcf, no CLI entry point, no ClinVar tests in Phase 10 — that completion surface is Phase 11 B5 (milescope boundary per ROADMAP Phase 10 SC 5)"
  artifacts:
    - dnallm/inference/vep.py  # new: VariantAlignment, align_variant, clm_log_likelihood, mlm_slot_log_prob, get_model_device
    - tests/inference/test_vep.py  # new: TestAlignVariant, TestClmLogLikelihood, TestMlmSlotLogProb + coverage completion
  key_links:
    - "align_variant(...).slot_index/ref_token_id/alt_token_id -> mlm_slot_log_prob(model, tokenizer, sequence, slot_index, token_id) (the MLM scoring chain: alignment supplies the slot and the competing token ids)"
    - "clm_log_likelihood(alt-window) - clm_log_likelihood(ref-window) -> the CLM delta-log-likelihood paradigm (consumed by Phase 11 score_variant)"
    - "tokenizer call convention shared by align_variant and both kernels: tokenizer(sequence, return_tensors='pt', add_special_tokens=True) so slot indices align across the chain"
  prohibitions:
    - "align_variant must NOT silently accept a variant whose tokenizations differ at more than one slot — multi-slot differences are skips (near-random AUROC from protocol mismatch is treated as a protocol bug first, PITFALLS #5)"
    - "The kernels must NOT route through trainer/DNAInference — pure model+tokenizer functions"
    - "Phase 10 must NOT add evaluate_vcf, CLI wiring, or __init__ re-exports (Phase 11 B5 scope)"
    - "No refactors of mutagenesis.py (no-refactors constraint; kernel reuse is adaptation-with-attribution, not extraction)"
    - "No new skips and no network access in tests (fast-lane only; if a slow real-model test is ever added later it needs slow marker + models.lock row + typed skip from birth)"
    - "Docstrings must NOT use pre-sweep terminology (born correct per D-08)"
---

<objective>
Agent lane A4 (owner-fixed wave structure): start the REV-08 long pole (~2 days, the milestone's longest item) by landing the vep.py CORE — the align_variant same-slot evaluability rule (the protocol answer to reviewer R1-3e-1) and the CLM/MLM scoring kernels with unit tests — so Phase 11 B5 only has to build evaluate_vcf, the CLI, and the ClinVar acceptance on top of proven kernels.

Purpose: VEP-01 (REV-08, reviewers R2-3 / R1-3e-1) is the milestone's critical path; the same-slot rule is where zero-shot VEP fails silently (PITFALLS #5), so it starts in Wave 1 despite being P1 (research SUMMARY).
Output: new dnallm/inference/vep.py (core only) + tests/inference/test_vep.py at the >=96% per-module standard.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-CONTEXT.md
@.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-PATTERNS.md
@.planning/research/SUMMARY.md
</context>

<tasks>

<task type="tracer">
  <name>Task 1: align_variant — the same-slot evaluability rule with VariantAlignment result type</name>
  <files>dnallm/inference/vep.py, tests/inference/test_vep.py</files>
  <read_first>
  - dnallm/inference/mutagenesis.py (257-347 — tokenizer call convention at 278/331: tokenizer(seq, return_tensors="pt", add_special_tokens=True); get_model_device 241-255; module docstring + class facade style 1-60)
  - tests/conftest.py (SimpleDNATokenizer at 14-26 — char-level vocab [PAD],[UNK],[CLS],[SEP],[MASK],A,C,G,T with N->mask; fixtures simple_dna_tokenizer/tiny_real_model at 168-182)
  - tests/inference/test_mutagenesis.py (header docstring, _make_* helper pattern, class-per-function grouping)
  - 10-PATTERNS.md vep.py section (kernel excerpts, no-analog note for align_variant)
  - .planning/research/PITFALLS.md Pitfall 5 (silent slot misalignment)
  </read_first>
  <action>
  Create dnallm/inference/vep.py — Google-style module docstring ("DNA large language model" terminology from birth, D-08) declaring the zero-shot variant-effect scoring protocol: the same-slot evaluability rule as the answer to reviewer R1-3e-1, and the two scoring-kernel formulas (full formulas land with the kernels in Task 2; the module docstring states the protocol and that evaluate_vcf/CLI arrive with the next phase).

  1. VariantAlignment: frozen dataclass with fields evaluatable: bool, slot_index: int | None, ref_token_id: int | None, alt_token_id: int | None, skip_reason: str | None.

  2. align_variant(sequence: str, pos: int, ref: str, alt: str, tokenizer) -> VariantAlignment:
     - Input contract: pos is a 0-based index into sequence; verify sequence[pos:pos+len(ref)] == ref, else raise ValueError(f"Reference allele '{ref}' does not match sequence at position {pos} (found '{sequence[pos:pos+len(ref)]}')") — matchable substring "does not match".
     - Build alt_sequence = sequence[:pos] + alt + sequence[pos+len(ref):].
     - Tokenize both with the shared convention tokenizer(s, return_tensors="pt", add_special_tokens=True) and take input_ids[0].
     - Same-slot rule: if the two id lists differ in length -> skip_reason="length-changing allele"; elif they are identical -> skip_reason="no change"; elif the count of differing indices != 1 -> skip_reason="multi-slot token difference"; else evaluatable=True with slot_index/ref_token_id/alt_token_id from the single differing index.
     - Skips return evaluatable=False with the reason and None fields — never raise.

  3. tests/inference/test_vep.py mirroring test_mutagenesis.py conventions (docstring stating the mocked fast-lane strategy, absolute imports, class grouping). TestAlignVariant with the char-level simple_dna_tokenizer (single-char substitutions are same-slot by construction): (a) SNP "A"->"T" at a mid-sequence pos -> evaluatable, slot_index points at the differing id, ids match tokenizer vocabulary; (b) the exactly-one-differing-slot assertion — for a passing variant, compare the full id lists and assert exactly one index differs (PITFALLS #5 guard, explicit in the test); (c) length-changing allele (ref "A", alt "TG") -> evaluatable=False, skip_reason="length-changing allele", no exception; (d) identical ref/alt -> skip_reason="no change"; (e) ref mismatch -> pytest.raises(ValueError, match="does not match"); (f) multi-slot skip path via a minimal stub tokenizer defined in the test file (a small class returning crafted id lists that differ at two indices) -> skip_reason="multi-slot token difference".
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_vep.py -q -k TestAlignVariant</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - dnallm/inference/vep.py exists with VariantAlignment + align_variant; no evaluate_vcf/CLI in the module
  - A test asserts the passing case differs at exactly one token slot (list comparison, not spot-check)
  - Skip paths return data (evaluatable=False + reason) without raising; the ref-mismatch ValueError carries the "does not match" substring
  - grep -n "add_special_tokens=True" shows the shared tokenizer convention
  </acceptance_criteria>
  <done>The same-slot rule works end-to-end on a real tokenizer with deterministic evaluatable/skip/ValueError outcomes — the R1-3e-1 protocol piece is real and tested.</done>
</task>

<task type="auto">
  <name>Task 2: CLM/MLM scoring kernels with formula docstrings (protocol declaration)</name>
  <files>dnallm/inference/vep.py, tests/inference/test_vep.py</files>
  <read_first>
  - dnallm/inference/mutagenesis.py (clm_evaluate 311-347 — shifted log_softmax/gather math to mirror; mlm_evaluate 257-309 — mask step to mirror; get_model_device 241-255)
  - tests/conftest.py (tiny_real_model/TinyDNAModel — the real tiny torch module producing real logits)
  - .planning/research/ARCHITECTURE.md REV-08 data-flow section (CLM delta-log-lik and MLM log-odds paradigm definitions)
  </read_first>
  <action>
  1. get_model_device(model) -> torch.device: module-level helper mirroring mutagenesis.py:241-255 exactly (model.device attr -> next(model.parameters()).device -> cpu fallback).

  2. clm_log_likelihood(model, tokenizer, sequence: str) -> float: @torch.no_grad(); tokenize with the shared convention, move to device, forward; shift logits/labels (predict token t given tokens < t), log_softmax(dim=-1), gather at the label ids, return the float sum. Docstring carries the formula — log P(sequence) = sum_t log P(token_t | tokens_<t) — and states it is the same scoring math as Mutagenesis.clm_evaluate (mutagenesis.py:311-347), and that the CLM paradigm scores a variant as clm_log_likelihood(alt) - clm_log_likelihood(ref).

  3. mlm_slot_log_prob(model, tokenizer, sequence: str, slot_index: int, token_id: int) -> float: @torch.no_grad(); tokenize, clone input_ids, set position slot_index to tokenizer.mask_token_id, forward, log_softmax at that slot, return float logp[token_id]. Docstring carries the formula — log P(token_id | masked context) — the attribution to Mutagenesis.mlm_evaluate (mutagenesis.py:257-309), and the MLM paradigm: the variant log-odds is mlm_slot_log_prob(alt_id) - mlm_slot_log_prob(ref_id) at the alignment slot (the ids come from align_variant).

  4. Tests with tiny_real_model + simple_dna_tokenizer (real torch compute, deterministic): TestClmLogLikelihood — (a) returns a finite float <= 0 for a valid sequence; (b) determinism: two calls give bit-identical results; (c) a longer/different sequence yields a different value. TestMlmSlotLogProb — (a) returns finite float <= 0; (b) determinism; (c) competing token ids at the same slot give different log-probs (the ref/alt discrimination the paradigm depends on); (d) masking a slot never alters other positions' ids (clone check — the mutagenesis-style masked-inputs dict).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_vep.py -q -k "Clm or Mlm"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - Both kernel docstrings contain their scoring formulas (grep -c "log P" dnallm/inference/vep.py >= 2) and the mutagenesis attribution
  - Tests prove determinism (repeated identical calls) and ref/alt discrimination (different token ids -> different log-probs)
  - Both kernels run under torch.no_grad() with device placement via get_model_device
  </acceptance_criteria>
  <done>The two paradigm kernels produce real, deterministic, documented scores on the tiny real model — Phase 11's score_variant/evaluate_vcf assemble on top of these.</done>
</task>

<task type="auto">
  <name>Task 3: Coverage completion to >=96% + born-correct terminology + module boundary proof</name>
  <files>dnallm/inference/vep.py, tests/inference/test_vep.py</files>
  <read_first>
  - pyproject.toml ([tool.coverage.*] — the per-module 96% working standard context; [tool.ruff] line-length 100)
  - tests/inference/test_vep.py (own Tasks 1-2 state — close the residual branches)
  </read_first>
  <action>
  1. Run the module-scoped coverage and close every branch below the bar with fast-lane tests (typical residuals: get_model_device branches — an object exposing .device, a parameters()-only model, a plain fallback object; align_variant boundary indices — pos at sequence start/end; the no-change skip's None fields). All inputs synthetic/tiny-real — no network, no real-model downloads.

  2. Terminology + boundary audit of the finished module: docstrings use "DNA large language model(s)" spellings (zero hits for the old variants), the module imports nothing from dnallm at module level beyond torch/stdlib needs, and dnallm/inference/__init__.py is untouched (no re-export — verified by git status).

  3. Confirm the Phase-10/Phase-11 boundary holds: the module contains align_variant, the two kernels, VariantAlignment, get_model_device and nothing user-facing beyond them (no evaluate_vcf, no CLI, no VCF parsing — that is Phase 11 B5; no CHANGELOG entry for REV-08 in Phase 10 since the capability is not yet user-visible — its entry lands with the Phase 11 completion commit).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_vep.py --cov=dnallm.inference.vep --cov-report=term -q</automated>
    <fails_when>the term table shows dnallm/inference/vep.py below 96% coverage, or any test fails, or "0 passed" in the summary</fails_when>
    <automated>! grep -qiE "DNA[ -]language[ -]model" dnallm/inference/vep.py && test -z "$(git status --porcelain dnallm/inference/__init__.py)" && echo BOUNDARY-OK</automated>
    <fails_when>the command does not print BOUNDARY-OK — either an old-terminology hit exists in vep.py, or dnallm/inference/__init__.py was modified (re-export prohibition breached)</fails_when>
  </verify>
  <acceptance_criteria>
  - Coverage row for dnallm/inference/vep.py at >=96% from fast-lane tests only
  - Zero old-terminology hits in vep.py; dnallm/inference/__init__.py unmodified
  - The module's public surface is exactly VariantAlignment, align_variant, clm_log_likelihood, mlm_slot_log_prob (+ get_model_device helper)
  - uv run --no-sync pytest tests/inference/ -q green (no regressions in the inference test dir)
  </acceptance_criteria>
  <done>The vep.py core is complete for Phase 10: same-slot rule + both kernels, tested to the per-module standard, born with correct terminology, leaving Phase 11 B5 a pure fill-in task.</done>
</task>

</tasks>

## Artifacts this phase produces (plan 10-04)

- `dnallm/inference/vep.py` — new module (Phase-10 core scope only)
- `VariantAlignment` (frozen dataclass: evaluatable, slot_index, ref_token_id, alt_token_id, skip_reason)
- `align_variant(sequence, pos, ref, alt, tokenizer) -> VariantAlignment`
- `clm_log_likelihood(model, tokenizer, sequence) -> float`
- `mlm_slot_log_prob(model, tokenizer, sequence, slot_index, token_id) -> float`
- `get_model_device(model) -> torch.device` (module helper)
- `tests/inference/test_vep.py` — TestAlignVariant, TestClmLogLikelihood, TestMlmSlotLogProb + coverage-completion tests
- Deliberately NOT produced here (Phase 11 B5): evaluate_vcf (scikit-allel `read_vcf` per the owner decision of 2026-10-09 superseding the stdlib-reader plan), VCF coordinate conversion, score_variant dispatcher + paradigm guard, CLI entry, ClinVar tests, REV-08 CHANGELOG entry

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Caller-supplied sequences/alleles -> kernels | align_variant and the kernels operate on strings/ints passed by library callers; no file or network input in Phase 10 |

## STRIDE Threat Register

Threat IDs continue after plans 10-01..03 (T-10-01..07, T-10-SC reserved).

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-10-08 | Denial of Service | kernels on hostile sequence lengths | low | accept | Inputs are developer-supplied strings; no unbounded loops (single forward pass per kernel; align_variant is two tokenizations + list diff); Phase 11's evaluate_vcf owns any row-count limits |
| T-10-09 | Tampering | align_variant skip semantics silently flipping to accept | medium | mitigate | The exactly-one-slot rule is asserted by an explicit test comparing full id lists; skips are structured data (skip_reason) so downstream accounting cannot mistake them for scores (PITFALLS #5 countermeasure) |
| T-10-SC | Tampering | package installs | high | accept | Phase 10 adds no dependencies (stdlib + existing torch only); the milestone's single approved addition, scikit-allel for VCF reading, lands with Phase 11 B5 and is not needed by this core |
</threat_model>

<verification>
- `uv run --no-sync pytest tests/inference/ -q` green
- Module coverage row >=96% (proof command in Task 3)
- `uv run --no-sync python scripts/check_code.py` green (ruff line-length 100, relative-import convention inside dnallm/, mypy)
- Phase-level: no files outside dnallm/inference/vep.py and tests/inference/test_vep.py modified by this plan (git status)
</verification>

<success_criteria>
- The REV-08 core is real and proven: same-slot rule with skip-as-data semantics, both scoring kernels with formula docstrings, >=96% fast-lane coverage, correct boundary (no Phase-11 surface) — the long pole has started per ROADMAP SC 5
</success_criteria>

<output>
Create `.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-04-SUMMARY.md` when done
</output>
