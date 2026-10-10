---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/datahandling/data.py
  - dnallm/cli/cli.py
  - dnallm/cli/inference.py
  - dnallm/cli/train.py
  - dnallm/inference/benchmark.py
  - dnallm/inference/inference.py
  - dnallm/inference/interpret.py
  - dnallm/inference/plot.py
  - dnallm/mcp/server.py
  - dnallm/models/losses.py
  - dnallm/models/model.py
  - dnallm/models/modeling_auto.py
  - dnallm/tasks/task.py
  - dnallm/utils/sequence.py
  - docs/  # ~31 non-mirror pages + docs/example wrapper pages carrying old terminology
  - README.md
  - mkdocs.yml
  - example/notebooks/overview.md
  - docs/user_guide/fine_tuning/peft_adapters.md
  - tests/datahandling/test_dna_dataset.py
  - CHANGELOG.md
coupling_justified:
  - "10-01: CHANGELOG.md is the one sanctioned shared append surface (D-09) — both lanes idempotently ensure the ## [Unreleased] block exists (create-if-absent above ## [0.7.1]), then append their own distinct REV-ID bullet via unique-anchor insert, re-reading the file immediately before editing; either landing order yields the correct changelog and a git-level conflict is resolved by re-appending the missing bullet"
  - "10-02: CHANGELOG.md shared append surface (D-09) — same idempotent create-if-absent + unique-anchor append discipline as the 10-01 entry; order irrelevant"
autonomous: true
requirements: [DOCS-01]
user_setup: []
estimate:
  tokens: 30000
  raw_tokens: 30000
  tasks: 3
  confidence: low   # 0 calibration samples (first phase of v1.2); factor 1 per estimate-calibration
  # Scope note: the ~20+ files_modified entries expand to ~45-50 touched files via the docs/ glob —
  # this is the D-08 locked single-lane sweep (NOT split); per-file context cost is grep hit lines +
  # one scoped Edit, never a whole-file read. See "Context budget strategy" below the objective.

must_haves:
  truths:
    # DOCS-01 core (goal-backward from ROADMAP SC 4)
    - "The old terminology is gone from this lane's entire owned surface: zero case-insensitive hits for the 'DNA[ -]language[ -]model' pattern across docs/ (incl. docs/example wrappers), README.md, example/notebooks/overview.md, and the 13 A3-owned dnallm docstring files (D-08 full-surface sweep; trainer.py/configs.py/metrics.py are swept by their owners in plans 10-01/10-02, vep.py is born correct in 10-04)"
    - "example/notebooks/overview.md and docs/example/notebooks/overview.md remain byte-identical after the sweep — edited as a pair, scripts/check_docs_sync.py green (PITFALLS #9 mirror trap)"
    - "DNADataset.validate_sequences carries a Google-style docstring documenting the cross-model valid_chars comparability hazard (whole-sequence drop on any out-of-charset character; each model silently evaluates a different subset; comparisons require a common valid_chars subset) and that charset membership is case-sensitive literal matching (e.g. 'ACGTacgt' admits lowercase)"
    - "validate_sequences logs a dropped-row count line when rows are filtered — format includes the dropped count, the before-count, and the applied valid_chars — and stays silent when zero rows are dropped (boundary edge: no log-with-zero noise; documented choice)"
    - "A dataset where every row is dropped logs the full count and returns the empty dataset without error (empty-input edge)"
    - "docs/user_guide/fine_tuning/peft_adapters.md exists as the LoRA/QLoRA usage chapter (grounded, working config examples) with an honest IA³ section stating it lands with use_ia3 in the next release — no fabricated API (IA³ part completes in Phase 12 after PEFT-01, per the requirement's split-delivery note)"
    - "mkdocs.yml nav registers the new chapter page under Fine Tuning; the docs build surface stays green under every docs-validation gate step: check_docs_sync.py, validate_docs_snippets.py, validate_yaml.py, pytest tests/examples/test_examples.py, pytest tests/configuration/test_yaml_load.py"
    - "CHANGELOG.md carries the (REV-03, Ed-2/Ed-6/R1-3c) entry under ## [Unreleased] in the same commit as the docs sweep, continuing the one-entry-per-fix evidence chain (D-09; SHA backfill is Phase 12 C3)"
    - "(flagged assumption, precision edge probe: no numeric display/rounding/tie-breaking contract changes in Phase 10 — the phase introduces no metric-value formatting; if a future change formats metric numbers for docs, half-even vs half-up must be specified then)"
  artifacts:
    - dnallm/datahandling/data.py  # validate_sequences docstring + dropped-row log line
    - tests/datahandling/test_dna_dataset.py  # log-line behavior tests
    - docs/user_guide/fine_tuning/peft_adapters.md  # new LoRA/QLoRA/IA³ chapter
    - mkdocs.yml  # nav entry for the chapter
    - README.md, docs/**, 13 dnallm docstring files, example/notebooks/overview.md  # terminology sweep
    - CHANGELOG.md  # REV-03 Unreleased entry
  key_links:
    - "validate_sequences docstring -> mkdocstrings-rendered api/datahandling/data.md page (the API-docs surface where the comparability warning becomes visible — PITFALLS #9: docstring + docs warning in the same change)"
    - "example/notebooks/overview.md <-> docs/example/notebooks/overview.md byte-identity -> scripts/check_docs_sync.py gate"
    - "docs sweep -> docs-validation workflow (.github/workflows/docs-validation.yml) staying green"
  prohibitions:
    - "The sweep must NOT touch files owned by A1/A2/A4 in this wave: dnallm/finetune/trainer.py, dnallm/configuration/configs.py, dnallm/tasks/metrics.py, dnallm/tasks/metric_registry.py, dnallm/inference/vep.py (D-08)"
    - "The sweep must NEVER edit only one side of a mirrored example/ <-> docs/example/ pair (byte-identity gate)"
    - "The comparability work must NOT add a new filtering API — D3 adjudication: unified-subset filtering stays pipeline-side (dnallmmark); suite side is docstring + docs warning + count log line only"
    - "The IA³ section must NOT document parameters or behavior that do not exist yet (no fabricated API; TrainingConfig.use_ia3 exists as a field per D-07 but the trainer branch is Phase 11)"
    - "The sweep must not reflow/reformat unrelated content (scoped replacements only — validate_docs_snippets and the mirror gate must stay green)"
    - "No new skips; no changes to scripts/check_docs_sync.py itself"
---

<objective>
Agent lane A3 (owner-fixed wave structure): land DOCS-01 — the full-surface terminology unification to "DNA large language models" (docs/ non-mirror pages, docs/example wrappers, README, 13 A3-owned docstring files, the example/ mirror pair), the validate_sequences comparability warning with its dropped-row count log line (D3 adjudication: docs + one log line, no new API), the LoRA/QLoRA usage chapter with an honest IA³ pointer, and the CHANGELOG evidence-chain entry — all inside the docs-validation gate.

Purpose: DOCS-01 (REV-03, reviewers Ed-2/Ed-6/R1-3c) makes the revision docs surface honest and opens the rebuttal-letter evidence chain; the mkdocstrings-rendered API pages only show the new terminology if the docstrings are swept (D-08 rationale).
Output: swept surface, warned-and-logged validate_sequences, new chapter, REV-03 CHANGELOG entry.
</objective>

### Context budget strategy (why ~45-50 touched files fit a 30k-token plan)

The file count is high but the edit profile is mechanical, and D-08 locks this sweep into ONE
lane — no split. Three properties keep context cost commensurate with edit complexity:

1. **Grep-driven discovery, never whole-file reads.** Task 2 does not read the swept files:
   `grep -rlniE "DNA[ -]language[ -]model"` enumerates the hit files; per-file `grep -n` fetches
   only the matching lines (plus a few context lines to make a unique `Edit` anchor); the
   longest-first mapping is applied as scoped single-line replacements; re-running the same grep
   to zero is the completion proof. Per-file context cost ≈ hit lines, not file size. Authored
   (generated-from-scratch) content is confined to four small surfaces: the validate_sequences
   docstring + one log line, the peft_adapters.md page, one mkdocs.yml nav line, one CHANGELOG
   bullet.
2. **No reflow, no reformat — by gate, not by discipline.** The mirror byte-identity gate
   (check_docs_sync.py) and validate_docs_snippets both punish broad diffs, so every replacement
   is the smallest scoped Edit — which is also the cheapest possible diff in context.
3. **Per-task projection: Task 1 ≈ 9k (data.py method + tests), Task 2 ≈ 13k (sweep across
   ~31 docs pages + docs/example wrappers, README, the mirror pair, 13 docstring files),
   Task 3 ≈ 8k (chapter + nav + CHANGELOG) — total ≈ 30k, matching the estimate block.** The
   sweep is resumable by construction (one file's scoped replacement at a time, grep re-run
   between files), so context never spikes even if the enumerated hit list runs larger than
   expected.

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
  <name>Task 1: validate_sequences comparability warning + dropped-row count log line (the code-bearing vertical slice of DOCS-01)</name>
  <files>dnallm/datahandling/data.py, tests/datahandling/test_dna_dataset.py</files>
  <read_first>
  - dnallm/datahandling/data.py (validate_sequences at 842-860 — the exact method; nearby print conventions at 1124, 1421, 1697)
  - dnallm/utils/sequence.py (check_sequence at 89-124 — whole-sequence drop semantics the docstring must describe)
  - tests/datahandling/test_dna_dataset.py (TestSequenceProcessing at ~631 — the existing validate_sequences tests to extend)
  - .planning/research/PITFALLS.md Pitfall 9 (docs-only trap + the one-log-line runtime affordance ruling)
  </read_first>
  <action>
  1. Rewrite the validate_sequences docstring (Google style, untyped-param form): keep Args; add the comparability warning — sequences containing ANY character outside valid_chars are dropped WHOLE (via check_sequence); 13 strict-charset models (e.g. "ACGTacgt") silently evaluate different subsets of the same dataset, so cross-model comparisons are only valid over a common valid_chars subset (the D3 adjudication: unified-subset prefiltering itself is pipeline-side, not a dnallm API); charset membership is case-sensitive literal matching. Document the log behavior: when rows are dropped a single print("[Warning] ...") line reports the dropped count, the before-count, and the applied filters; zero dropped rows logs nothing.

  2. Add the count log line (a log line, NOT a new API — D3 ruling): capture len(self.dataset) before the existing .filter(...) call, compute n_dropped = before - len(self.dataset) after, and when n_dropped > 0 print one f-string line in house style, e.g. print(f"[Warning] validate_sequences dropped {n_dropped} of {before} rows (valid_chars={valid_chars!r}, minl={minl}, maxl={maxl}); cross-model comparisons require a common valid_chars subset."). No other behavior change — filtering semantics are untouched.

  3. tests/datahandling/test_dna_dataset.py — extend TestSequenceProcessing: (a) dataset with some droppable rows (reuse the existing ["ATCG","GCTA","TAGC","NNNN","AT"] shape) run with capsys or patch("builtins.print"): exactly one "[Warning] validate_sequences dropped" line, containing the correct dropped count; (b) zero-dropped dataset (all rows valid): no "[Warning] validate_sequences" output (boundary edge — silent at zero, by documented choice); (c) all-rows-dropped dataset: the line reports the full count and the resulting dataset length is 0 without raising (empty edge).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/datahandling/test_dna_dataset.py -q -k "TestSequenceProcessing"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
    <automated>uv run --no-sync python scripts/check_docs_sync.py</automated>
    <fails_when>non-zero exit — the mirror must already be green before the sweep touches it (baseline proof)</fails_when>
  </verify>
  <acceptance_criteria>
  - grep -n "cross-model" dnallm/datahandling/data.py finds the docstring warning text; grep -n "valid_chars" in the docstring shows the case-sensitivity statement
  - A test asserts the dropped-count line fires exactly once with the correct count; another asserts silence at zero drops; another asserts the all-dropped edge
  - The .filter call chain is unchanged (no new public API, no signature change on validate_sequences)
  </acceptance_criteria>
  <done>validate_sequences documents the comparability hazard and logs dropped-row counts (only when rows actually drop), with same-change tests — the D3-adjudicated suite-side piece of the warning requirement.</done>
</task>

<task type="auto">
  <name>Task 2: Full-surface terminology sweep — docs/, README, 13 docstring files, the example/ mirror pair</name>
  <files>docs/, README.md, example/notebooks/overview.md, dnallm/cli/cli.py, dnallm/cli/inference.py, dnallm/cli/train.py, dnallm/inference/benchmark.py, dnallm/inference/inference.py, dnallm/inference/interpret.py, dnallm/inference/plot.py, dnallm/mcp/server.py, dnallm/models/losses.py, dnallm/models/model.py, dnallm/models/modeling_auto.py, dnallm/tasks/task.py, dnallm/utils/sequence.py</files>
  <read_first>
  - scripts/check_docs_sync.py (whole file — mirror mechanics; .md wrappers docs-only on the docs side; both-sides .md like notebooks/overview.md must match byte-for-byte)
  - .github/workflows/docs-validation.yml (the five gate steps this sweep must keep green)
  - 10-CONTEXT.md D-08 (sweep scope + same-wave ownership exclusions)
  - mkdocs.yml (nav structure — where pages live)
  </read_first>
  <action>
  Scope: every file under docs/ (non-mirror pages AND docs/example generated .md wrappers), README.md, example/notebooks/overview.md (the single example/ file carrying the term — sweep it FIRST, then apply the identical edit to its mirror docs/example/notebooks/overview.md so byte-identity holds at every commit), and exactly these 13 A3-owned dnallm files: cli/cli.py, cli/inference.py, cli/train.py, inference/benchmark.py, inference/inference.py, inference/interpret.py, inference/plot.py, mcp/server.py, models/losses.py, models/model.py, models/modeling_auto.py, tasks/task.py, utils/sequence.py. Do NOT touch trainer.py, configs.py, metrics.py, metric_registry.py, vep.py (same-wave ownership, D-08).

  Mapping (apply as scoped replacements — no reflowing of surrounding content):
  - "DNA language models" -> "DNA large language models"
  - "DNA language model" -> "DNA large language model"
  - "DNA Language Models" -> "DNA Large Language Models"
  - "DNA Language Model" -> "DNA Large Language Model"
  Order the replacements longest-first so the plural forms are not corrupted by the singular rule. Docstring/code-comment/heading occurrences only — never rename identifiers, import paths, config keys, or URLs.

  Discovery is grep-driven, not list-driven: enumerate the exact hit list per file with grep -rlniE "DNA[ -]language[ -]model" before editing (docs pages may also carry hyphenated "DNA-language-model" forms — the grep pattern catches them; map them to the spaced target spelling). After editing, re-run the same grep over the A3-owned surface and require zero hits.
  </action>
  <verify>
    <automated>grep -rlniE "DNA[ -]language[ -]model" docs/ README.md example/ dnallm/cli/cli.py dnallm/cli/inference.py dnallm/cli/train.py dnallm/inference/benchmark.py dnallm/inference/inference.py dnallm/inference/interpret.py dnallm/inference/plot.py dnallm/mcp/server.py dnallm/models/losses.py dnallm/models/model.py dnallm/models/modeling_auto.py dnallm/tasks/task.py dnallm/utils/sequence.py | wc -l</automated>
    <fails_when>the printed count is not 0 (residual old-terminology files in this lane's owned surface)</fails_when>
    <automated>uv run --no-sync python scripts/check_docs_sync.py && uv run --no-sync python scripts/validate_docs_snippets.py && uv run --no-sync python scripts/validate_yaml.py</automated>
    <fails_when>non-zero exit from any of the three gate scripts (mirror drift, broken python snippets, broken YAML)</fails_when>
  </verify>
  <acceptance_criteria>
  - The grep-zero proof above returns 0 on the A3-owned surface
  - diff between example/notebooks/overview.md and docs/example/notebooks/overview.md is empty
  - All three docs gate scripts exit 0
  - git status shows no modifications to dnallm/finetune/trainer.py, dnallm/configuration/configs.py, dnallm/tasks/metrics.py, dnallm/tasks/metric_registry.py, or dnallm/inference/vep.py from this plan
  </acceptance_criteria>
  <done>Terminology unified across the full A3 surface with the mirror pair kept byte-identical and every docs gate green; remaining dnallm/ occurrences are confined to files their owners sweep in plans 10-01/10-02 (phase verification re-greps the whole tree).</done>
</task>

<task type="auto">
  <name>Task 3: LoRA/QLoRA/IA³ usage chapter + mkdocs nav + CHANGELOG REV-03 entry</name>
  <files>docs/user_guide/fine_tuning/peft_adapters.md, mkdocs.yml, CHANGELOG.md</files>
  <read_first>
  - mkdocs.yml (nav Fine Tuning block at 74-79)
  - configs/lora_config.yaml and configs/finetune_config.yaml (the grounded config examples the chapter documents)
  - dnallm/configuration/configs.py — LoraConfig (340-372) and TrainingConfig.use_qlora (310-313) ONLY for reference (do not modify the file)
  - docs/user_guide/fine_tuning/getting_started.md (house chapter style/tone)
  - CHANGELOG.md (the ## [Unreleased] anchor — lanes run in parallel, re-read immediately before editing)
  </read_first>
  <action>
  1. Create docs/user_guide/fine_tuning/peft_adapters.md — the parameter-efficient fine-tuning chapter: (a) LoRA section — YAML example grounded in the real LoraConfig fields (r, lora_alpha, target_modules, lora_dropout, bias, task_type) + DNATrainer(use_lora=True) usage; (b) QLoRA section — finetune.use_qlora: true with the quantization_config shape from the trainer docstring example (load_in_4bit, bnb_4bit_compute_dtype, bnb_4bit_use_double_quant, bnb_4bit_quant_type) and the bitsandbytes requirement; (c) IA³ section — honest forward pointer: IA³ adapters (finetune.use_ia3) land with the next release's trainer branch per the revision plan; the field exists in TrainingConfig but is not yet wired (no fabricated examples of IA³ training runs). All ```python blocks must be syntactically valid (validate_docs_snippets AST-checks them).

  2. mkdocs.yml: register the page in the Fine Tuning nav block (after Advanced Techniques) as `- PEFT Adapters (LoRA / QLoRA / IA³): user_guide/fine_tuning/peft_adapters.md`.

  3. CHANGELOG.md (deliberate shared append surface per D-09; unique-anchor insert, re-read first): ensure `## [Unreleased]` + `### Added`/`### Changed` subheadings exist (create above `## [0.7.1] - 2026-10-08` if absent). Under `### Changed`: "- Docs: terminology unified to \"DNA large language models\" across the documentation, README, and API docstrings; `validate_sequences` now documents the cross-model `valid_chars` comparability hazard and logs a dropped-row count; new LoRA/QLoRA/IA³ usage chapter (REV-03, Ed-2/Ed-6/R1-3c)". Same-commit as the chapter/sweep commit.
  </action>
  <verify>
    <automated>uv run --no-sync python scripts/validate_docs_snippets.py && uv run --no-sync python scripts/validate_yaml.py && uv run --no-sync pytest tests/examples/test_examples.py tests/configuration/test_yaml_load.py -q</automated>
    <fails_when>non-zero exit from either script, or the pytest summary line reports any failure (docs-validation workflow parity)</fails_when>
    <automated>grep -c "peft_adapters" mkdocs.yml && grep -c "(REV-03, Ed-2/Ed-6/R1-3c)" CHANGELOG.md</automated>
    <fails_when>either count is 0 — the nav entry or the CHANGELOG evidence entry is missing</fails_when>
  </verify>
  <acceptance_criteria>
  - docs/user_guide/fine_tuning/peft_adapters.md exists with LoRA and QLoRA grounded examples and an IA³ forward-pointer section that names no nonexistent API
  - mkdocs.yml Fine Tuning block contains the peft_adapters nav entry
  - CHANGELOG.md carries the (REV-03, Ed-2/Ed-6/R1-3c) entry under ## [Unreleased]
  - The docs-validation workflow's five steps all pass locally (sync, snippets, YAML, example tests, YAML-load tests)
  </acceptance_criteria>
  <done>The chapter is live in the nav, the evidence chain has its REV-03 entry, and the whole docs-validation gate is green end to end.</done>
</task>

</tasks>

## Artifacts this phase produces (plan 10-03)

- `docs/user_guide/fine_tuning/peft_adapters.md` (new page: LoRA/QLoRA sections + IA³ forward pointer)
- mkdocs.yml nav entry `- PEFT Adapters (LoRA / QLoRA / IA³): user_guide/fine_tuning/peft_adapters.md`
- `DNADataset.validate_sequences` — rewritten docstring (comparability warning, case-sensitivity, log-behavior documentation) + dropped-row count print (behavior: logs only when n_dropped > 0)
- CHANGELOG.md `## [Unreleased]` section (first creator) + REV-03 entry with inline `(REV-03, Ed-2/Ed-6/R1-3c)` tag
- Terminology-swept surface: docs/** (~31 non-mirror pages + docs/example wrappers), README.md, example/notebooks/overview.md (+ mirror), 13 dnallm docstring files
- Test additions in tests/datahandling/test_dna_dataset.py (dropped-count log: fires-with-count / silent-at-zero / all-dropped)

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Static docs content -> rendered public site | Docs/README text is published verbatim via MkDocs; no runtime input crosses into execution |

## STRIDE Threat Register

Threat IDs continue after plans 10-01/10-02 (T-10-01..04, T-10-SC reserved).

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-10-05 | Information Disclosure | validate_sequences log line | low | mitigate | Logs counts and filter parameters only — never sequence contents, sample ids, or labels; format fixed in the plan |
| T-10-06 | Tampering | docs code snippets published as usage guidance | low | accept | validate_docs_snippets AST-checks every python block; examples grounded in real config fields; no network/executable docs snippets introduced |
| T-10-07 | Repudiation | CHANGELOG evidence chain | low | mitigate | One entry per revision fix with inline REV-ID + reviewer comment id (D-09); commit SHA backfilled by Phase 12 C3 — greppable provenance by design |
| T-10-SC | Tampering | package installs | high | accept | Phase 10 adds no dependencies (the milestone's single approved addition, scikit-allel, lands in Phase 11 B5); no package-manager install tasks in this plan |
</threat_model>

<verification>
- All five docs-validation workflow steps pass locally (sync, snippets, YAML, tests/examples/test_examples.py, tests/configuration/test_yaml_load.py)
- `uv run --no-sync pytest tests/datahandling/test_dna_dataset.py -q` green
- Phase-level cross-check (after all four lanes land): `grep -rlniE "DNA[ -]language[ -]model" dnallm/ docs/ README.md example/` returns nothing anywhere in the tree
- `uv run --no-sync python scripts/check_code.py` green (ruff on the data.py change)
</verification>

<success_criteria>
- DOCS-01 holds on this lane's surface: terminology unified, validate_sequences warned + counted, chapter live, CHANGELOG entry traceable, docs build green under the gate — with the IA³ chapter section honestly deferred to Phase 12 after PEFT-01
</success_criteria>

<output>
Create `.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-03-SUMMARY.md` when done
</output>
