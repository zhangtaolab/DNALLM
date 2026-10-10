---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 05
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/inference/vep.py
  - dnallm/cli/vep.py
  - pyproject.toml
  - README.md
  - tests/inference/test_vep.py
  - tests/cli/test_vep_cli.py
  - tests/inference/data/synthetic_variants.vcf
  - tests/expected_skips.yaml
  - models.lock
  - CHANGELOG.md
autonomous: true
requirements: [VEP-01]
user_setup:
  - service: pypi
    why: "scikit-allel install (the phase's only sanctioned dependency addition, owner decision D-08 2026-10-09; legitimacy pre-verified: canonical cggh/scikit-allel repo, official readthedocs API docs, Windows cp310-313 wheels + numpy 1.26.4/2.2.0 compat empirically verified by the owner before approval)"
    env_vars: []
    dashboard_config: []
coupling_justified: >
  CHANGELOG.md: single sanctioned cross-lane append surface (one REV-08 unique-anchor
  bullet, D-09 discipline). README.md: two lanes append documentation this wave —
  11-02 writes one paragraph inside "## 🧬 Supported Models" (top, line ~32); this plan
  adds a NEW protocol-declaration section immediately before "## 🧪 Testing" (bottom,
  line ~517). Distinct, distant anchors; scoped Edit only (never full-file Write);
  re-read-before-edit; pathspec-limited commits. tests/expected_skips.yaml and
  models.lock are append-only registries this lane alone touches this wave (B1-B4 add
  no rows). pyproject.toml is solely this lane's (the only sanctioned dep change).
  dnallm/inference/vep.py is solely this lane's (Phase-10 core extended, kernels
  never re-implemented).

estimate:
  tokens: 36000
  raw_tokens: 36000
  tasks: 3
  confidence: low   # calibration sample_count=0, factor=1

must_haves:
  truths:
    - "A user can score variants zero-shot from a VCF: evaluate_vcf reads via allel.read_vcf, aligns through the landed align_variant, scores through the landed kernels, and reports per-variant scores + skip accounting + AUROC/AUPRC via the registry (VEP-01)"
    - "evaluate_vcf uppercases every window before alignment/scoring — lowercase soft-masked input mapping k-mers to <unk> and vanishing as 'no change' skips is test-proven absent (VEP-01 encoding edge, RESEARCH Pitfall 5, empirically verified trap)"
    - "VCF 1-based POS converts to the 0-based align_variant contract (POS=1 → pos=0); the committed fixture covers pos=1 and window-edge variants, proving the coordinate system (VEP-01 boundary edge)"
    - "read_vcf is called with a deliberately sized alt_number (>= 4, default 3 truncates 4+-allelic rows); multi-allelic rows are handled per-ALT or skip-as-data with the handling test-proven on the fixture's multi-allelic row (VEP-01 ordering edge)"
    - "A VCF with 0 scorable variants returns evaluated=0 with the full skip-count table and skip_fraction 1.0 reported as a finding — metrics None, no crash, no fabricated AUROC (VEP-01 empty edge)"
    - "ref/alt windows share identical left context (the driver substitutes within one window, preserving the landed kernel's contract); any driver code tokenizing ref and alt sequences built from different windows is a bug"
    - "score_variant(paradigm='clm'|'mlm') wraps the landed kernels with the paradigm↔architecture mismatch guard raising a matchable ValueError (CLM on a bidirectional MLM; MLM without a mask token) — never a silent skip, which would let a misconfiguration masquerade as a near-random finding (D-11)"
    - "Skip-as-data remains reserved for the same-slot alignment rule alone, with reason + count + skip fraction as the documented channel (D-11: the two channels never mix)"
    - "The ClinVar acceptance convention is D-17: >= 1 review star + SNVs only (CLNVC single_nucleotide_variant) + P/LP vs B/LB labels (VUS/conflicting/0-star excluded), reported alongside every result — never a bare AUROC without its convention (Pitfall 7)"
    - "The ClinVar 1k-sample × >= 5-model (CLM/MLM mix) slow-lane acceptance produces AUROCs recorded with the convention block and compared against the RESEARCH magnitude anchors within-paradigm/within-convention; fast-lane AUROC-shape proven with mocked scores on the committed fixture (D-09 two-tier)"
    - "The scoring formulas (CLM delta-log-likelihood; MLM log-odds) and the alignment protocol are written into the vep.py docstrings AND the README protocol section (VEP-01 protocol declaration)"
    - "The CLI entry dnallm-vep mirrors the mutagenesis precedent and yields per-variant scores + skip accounting + metrics + the convention block as JSON (D-10)"
    - "The fast lane stays network-free: the synthetic fixture is committed; the real ClinVar download lives only in slow-marked tests with a typed network skip allowlisted same-change in tests/expected_skips.yaml, and models.lock rows cover any newly fetched artifacts (D-09)"
  artifacts:
    - path: dnallm/inference/vep.py
      provides: "evaluate_vcf driver (allel.read_vcf, CLNSIG/CLNREVSTAT/CLNVC filtering, uppercase windows, skip accounting, registry metrics) + score_variant with the paradigm guard"
      contains: "evaluate_vcf"
    - path: dnallm/cli/vep.py
      provides: "dnallm-vep console command (--config/-c, --vcf, --reference, --model-name/-m, --output/-o)"
      contains: "click"
    - path: pyproject.toml
      provides: "scikit-allel>=1.3.13,<2 dependency (the milestone's only sanctioned dep change, D-08) + dnallm-vep script entry"
      contains: "scikit-allel"
    - path: tests/inference/data/synthetic_variants.vcf
      provides: "Committed network-free fixture: SNV + indel + boundary (POS=1, window edge) + multi-allelic + lowercase-reference + N-containing cases (D-09 tier 1)"
    - path: tests/cli/test_vep_cli.py
      provides: "CliRunner smoke tests with mocked evaluate_vcf"
    - path: README.md
      provides: "Zero-shot VEP protocol section (formulas, same-slot rule, skip accounting, ClinVar convention block) inserted before ## 🧪 Testing"
      contains: "variant effect"
  key_links:
    - from: dnallm/inference/vep.py
      to: allel.read_vcf
      via: "VCF parsing delegated wholesale (fields list incl. CLNSIG/CLNREVSTAT/CLNVC; alt_number deliberately sized) — never a hand-rolled reader"
      pattern: "read_vcf"
    - from: dnallm/inference/vep.py
      to: dnallm/inference/vep.py (Phase-10 core)
      via: "evaluate_vcf/score_variant consume align_variant + clm_log_likelihood + mlm_slot_log_prob — wrapped, never re-implemented"
      pattern: "align_variant"
    - from: dnallm/inference/vep.py
      to: dnallm/tasks/metric_registry.py
      via: "AUROC/AUPRC computed exclusively through metric_registry.resolve — read-only"
      pattern: "metric_registry"
    - from: dnallm/cli/vep.py
      to: dnallm/inference/vep.py
      via: "lazy import inside the click command body (mutagenesis precedent); errors via click.echo(err=True) + sys.exit(1)"
      pattern: "evaluate_vcf"
  prohibitions:
    - "No VCF parsing re-implementation — allel.read_vcf owns gzip/header/INFO/ALT dimensionality (Don't Hand-Roll)"
    - "No re-implementation of align_variant or the scoring kernels — the Phase-10 core is wrapped, not rewritten"
    - "No cyvcf2/pysam (Out of Scope: no Windows wheels); no biopython/statsmodels/torchmetrics (STACK rejected list)"
    - "No filesystem paths derived from VCF record fields — deterministic output names only (V5)"
    - "No new dnallm/__init__.py re-exports (facade stays byte-stable)"
    - "No pyproject changes beyond the scikit-allel dependency line and the dnallm-vep script entry (the sanctioned two-line delta)"
    - "No real ClinVar data committed to the repo (D-09: fast lane network-free, real data stays out)"
    - "No test matching scikit-allel/transformers foreign exception strings — dnallm's own ValueErrors only"
    - "No silent paradigm mismatch (D-11) and no mixing skip-as-data into the mismatch channel"
    - "No Co-Authored-By trailers; commits are pathspec-limited (git commit -- <paths>) in the shared tree"
---

<objective>
B5 (REV-08 completion): zero-shot variant scoring from VCF end-to-end —
`evaluate_vcf` over scikit-allel with ClinVar label filtering, `score_variant` with
the paradigm↔architecture guard, the `dnallm-vep` CLI, the README protocol
declaration, and the two-tier ClinVar test data (committed synthetic fixture for the
fast lane; the real 1k-sample × >= 5-model acceptance in the slow lane).

Purpose: VEP-01 — reviewers score variants zero-shot with a protocol whose alignment
rule, skip accounting, and conventions are explicit enough to answer R1-3e① and
survive the ascertainment-bias objection (Pitfall 7).
Output: vep.py completion, cli/vep.py + pyproject entries (the milestone's only
sanctioned dependency change, D-08), fixture + tests at the >= 96% standard, README
protocol, expected_skips/models.lock rows, CHANGELOG entry.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-CONTEXT.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-RESEARCH.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-PATTERNS.md
</context>

<tasks>

<task type="tracer">
  <name>Task 1: evaluate_vcf end-to-end — scikit-allel dep + driver + committed fixture + mocked-score AUROC shape (fast lane)</name>
  <reversibility rating="costly">The pyproject dependency addition (scikit-allel + its dask[array] transitive) is the milestone's single sanctioned dependency change; undo touches CI legs, lockfiles, and the owner-approved decision record.</reversibility>
  <files>pyproject.toml, dnallm/inference/vep.py, tests/inference/test_vep.py, tests/inference/data/synthetic_variants.vcf</files>
  <read_first>
  - dnallm/inference/vep.py (the ENTIRE landed Phase-10 core: VariantAlignment, align_variant's 0-based pos contract, clm_log_likelihood, mlm_slot_log_prob — wrap these, never re-implement)
  - 11-RESEARCH.md sections: Pattern 4 (evaluate_vcf driver shape with read_vcf fields/alt_number), Code Examples (variant loop), Pitfall 5 (uppercase), Pitfall 6 (tokenizer-class table), Standard Stack (scikit-allel row + install command)
  - 11-PATTERNS.md: vep.py evaluate_vcf analog (skip-record style, 0-based contract, docstring deferral note to extend)
  - pyproject.toml (dependencies list + [project.scripts] block)
  - tests/inference/test_vep.py (existing TestAlignVariant/TestClmLogLikelihood/TestMlmSlotLogProb — extend, do not disturb)
  - tests/conftest.py (simple_dna_tokenizer, tiny_model_factory)
  </read_first>
  <action>
  Install first, then one end-to-end path: the committed synthetic VCF flows through
  evaluate_vcf into per-variant scores + skip counts + an AUROC — with the model
  scores mocked so the fast lane is deterministic and network-free.

  Dependency (D-08):
  - `uv pip install "scikit-allel>=1.3.13,<2"` into the dev venv (shared tree note:
    installs are additive only — dask[array] is the sole new transitive dep per the
    owner's empirical check; no existing dep versions change).
  - pyproject.toml: add `"scikit-allel>=1.3.13,<2"` to the dependencies list. This
    and the Task 2 script line are the ONLY pyproject deltas this phase.

  Fixture (D-09 tier 1) — `tests/inference/data/synthetic_variants.vcf` (committed):
  - A small hand-written VCF (valid header + records) covering, at minimum: plain
    SNVs; a 1-bp indel (length-changing skip on every tokenizer class); POS=1 and
    window-edge variants; a multi-allelic row (ALT with multiple alleles); a variant
    whose reference context must be uppercased (lowercase soft-masked reference); an
    N-containing window case. Pair it with a committed matching synthetic reference
    (fasta or plain-sequence sidecar in the same data dir — executor's choice) so
    window building is self-contained.

  Driver — evaluate_vcf in dnallm/inference/vep.py:
  - Signature shape: `evaluate_vcf(model, tokenizer, vcf_path, reference, *,
    paradigm, context_window=VepConfig default, clnsig_filter=<D-17 defaults>,
    alt_number=<sized >= 4>, output_dir=None) -> VepResult` (dataclass: per-variant
    records with delta scores + labels, skip_counts per reason, evaluated/skipped
    totals, skip_fraction, metrics dict or None, convention block dict).
  - Parsing: `allel.read_vcf(vcf_path, fields=["variants/CHROM", "variants/POS",
    "variants/ID", "variants/REF", "variants/ALT", "variants/CLNSIG",
    "variants/CLNREVSTAT", "variants/CLNVC"], alt_number=...)` — alt_number sized
    deliberately (>= 4; the default 3 truncates 4+-allelic rows). Import allel
    lazily inside the function so the module stays importable without the dep
    (graceful ImportError message naming the extra).
  - Label/convention filtering (D-17): keep CLNVC single_nucleotide_variant only;
    labels P/LP → 1 vs B/LB → 0; exclude VUS/conflicting; CLNREVSTAR >= 1-star
    floor; the applied convention block is returned in the result AND reported
    alongside metrics (never a bare AUROC).
  - Per variant: pos0 = POS - 1 (1-based → 0-based); build the window from the
    reference with symmetric context, `.upper()` ALWAYS (docstring states why —
    Pitfall 5); identical left context for ref/alt (substitute within the one
    window); call the LANDED align_variant(window, local_pos, ref, alt, tokenizer);
    evaluatable → score via the landed kernels by paradigm (mlm: slot log-prob
    difference at the alignment slot; clm: full-sequence delta log-likelihood on the
    substituted window); skip → increment skip_counts[reason].
  - Metrics: AUROC/AUPRC through metric_registry.resolve over (labels, deltas);
    when evaluated == 0 → metrics None + skip_fraction 1.0 reported as a finding
    (empty edge; no crash).
  - Module docstring: extend the numbered Features list (evaluate_vcf + the next
    task's score_variant/CLI) and keep the deferral note removed/updated.

  Tests (fast lane, network-free — extend tests/inference/test_vep.py):
  - New TestEvaluateVcf class: mocked score kernels (patch clm_log_likelihood /
    mlm_slot_log_prob to controlled values) → deterministic AUROC (perfect
    separation → AUROC == 1.0 shape assertion); coordinate conversion (a POS=1
    record scores the first base — assert via the mocked kernel's received
    arguments); uppercase behavior (the lowercase-context record is scored, NOT
    skipped as no-change); skip accounting (the indel row counts under
    length-changing allele; the multi-allelic row handled per its policy); 0-scorable
    case (a filtered-to-empty VCF variant set → metrics None, skip_fraction 1.0);
    read_vcf call shape (alt_number >= 4 asserted via patched allel.read_vcf).
  </action>
  <verify>
    <automated>uv run --no-sync python -c "import allel; print(allel.__version__)"</automated>
    <fails_when>non-zero exit or a traceback (module not installed)</fails_when>
    <automated>uv run --no-sync pytest tests/inference/test_vep.py -q -k "TestEvaluateVcf" -m "not slow"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
    <automated>test "$(grep -c "scikit-allel" pyproject.toml)" -ge 1 && echo DEP-OK</automated>
    <fails_when>DEP-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - `uv run --no-sync python -c "import allel"` succeeds; pyproject.toml carries scikit-allel>=1.3.13,<2 and `git diff HEAD -- pyproject.toml` shows only dependency-list lines (script entry lands in Task 2)
  - `tests/inference/data/synthetic_variants.vcf` committed alongside its reference sidecar; fixture covers SNV + indel + POS=1/window-edge + multi-allelic + lowercase + N cases
  - evaluate_vcf exists in vep.py wrapping (not re-implementing) align_variant + both kernels; `grep -c "read_vcf" dnallm/inference/vep.py` >= 1; windows uppercased (`grep -c "upper()" dnallm/inference/vep.py` >= 1)
  - TestEvaluateVcf green with >= 8 tests incl. AUROC-shape, coordinate, uppercase, skip-accounting, empty, alt_number assertions
  - Existing Phase-10 tests still green: `uv run --no-sync pytest tests/inference/test_vep.py -q -k "TestAlignVariant or TestClmLogLikelihood or TestMlmSlotLogProb" -m "not slow"`
  </acceptance_criteria>
  <done>The VCF → score → AUROC slice runs end-to-end on committed data with mocked model scores: dependency landed (D-08), driver wraps the landed kernels, uppercase + coordinate + skip semantics proven, fast lane network-free.</done>
</task>

<task type="auto">
  <name>Task 2: score_variant + paradigm guard + dnallm-vep CLI + README protocol declaration</name>
  <files>dnallm/inference/vep.py, dnallm/cli/vep.py, pyproject.toml, README.md, tests/inference/test_vep.py, tests/cli/test_vep_cli.py</files>
  <read_first>
  - dnallm/cli/mutagenesis.py (the D-10 precedent: structure lines 1-15, click options 31-88, lazy imports inside the command body, error handling 131-141, output writing 247-254)
  - dnallm/inference/vep.py (Task 1's evaluate_vcf + the landed kernels — score_variant composes them)
  - pyproject.toml ([project.scripts] block, lines ~271-277)
  - README.md ("## 🧪 Testing" section at line ~517 — the anchor: the new section inserts immediately BEFORE it)
  - 11-RESEARCH.md: D-11 rationale (Pitfall 6 — mismatched-paradigm AUROCs are expected near-random), architecture diagram (paradigm guard position)
  - .planning/research/261009-paper-revision-suite-plan.md (R1-3e reviewer-thread context for the protocol wording)
  </read_first>
  <action>
  score_variant (dnallm/inference/vep.py):
  - `score_variant(model, tokenizer, sequence, pos, ref, alt, *, paradigm="mlm") ->
    float` (the delta score): runs align_variant; evaluatable → CLM:
    clm_log_likelihood(substituted window) - clm_log_likelihood(reference window);
    MLM: mlm_slot_log_prob(alt_id) - mlm_slot_log_prob(ref_id) at the alignment
    slot; not evaluatable → the skip record is returned/raised as data per the
    module's existing skip-as-data channel (reason preserved).
  - Paradigm↔architecture mismatch guard (D-11) BEFORE scoring, raising a matchable
    ValueError (dnallm's own message — never peft/transformers text): paradigm="clm"
    on a bidirectional model (config-declared decoder/architecture heuristics) and
    paradigm="mlm" when the tokenizer has no mask_token_id. Skip-as-data stays
    reserved exclusively for the same-slot rule.
  - Docstrings carry the scoring formulas verbatim (CLM delta-log-likelihood; MLM
    log-odds) — the in-code half of the protocol declaration.

  CLI (dnallm/cli/vep.py, NEW — mutagenesis precedent):
  - click command `main` with options: `--config/-c` (VepConfig YAML path),
    `--vcf` (path), `--reference` (path), `--model-name/-m` (required),
    `--source` (huggingface/modelscope), `--paradigm` (choice clm/mlm),
    `--output/-o` (JSON path, defaults to stdout).
  - Lazy imports inside the command body (`from ..inference.vep import
    evaluate_vcf`; `from ..models import load_model_and_tokenizer`);
    `logger = get_logger("dnallm.cli.vep")` at module top (relative imports inside
    dnallm/); load errors → `click.echo(f"Error ...", err=True)` + `sys.exit(1)`;
    output JSON via `Path(...).mkdir(parents=True, exist_ok=True)` + `json.dump(
    indent=2)` or stdout — mirroring mutagenesis lines 247-254.
  - The output JSON contains: per-variant scores, skip_counts + totals +
    skip_fraction, metrics (or None), and the convention block.
  - pyproject.toml [project.scripts]: append
    `dnallm-vep = "dnallm.cli.vep:main"` beside dnallm-mutagenesis.

  README protocol declaration (scoped Edit — re-read first, insert ONE new section
    immediately before the "## 🧪 Testing" header; touch nothing else — 11-02 edits
    the top-of-file Supported Models section concurrently):
  - Section content: the two scoring formulas; the same-slot evaluability rule
    (differ by exactly one token slot); skip-as-data accounting (reasons + fraction
    reported); the D-17 ClinVar convention block (>= 1 review star, SNVs only,
    P/LP vs B/LB, VUS/conflicting excluded — reported alongside every AUROC); a
    short `dnallm-vep` usage example. Magnitude anchors from RESEARCH's table
    cited as within-convention comparables (never as bare thresholds).

  Tests:
  - tests/inference/test_vep.py TestScoreVariant: guard cases (paradigm="clm" on a
    mocked bidirectional config → ValueError; paradigm="mlm" with mask-less
    tokenizer → ValueError); happy paths on tiny models for both paradigms;
    skip-record passthrough for a length-changing allele.
  - tests/cli/test_vep_cli.py (NEW): click CliRunner smoke tests with evaluate_vcf
    and load_model_and_tokenizer mocked — successful run writes/echoes the JSON
    shape (scores + skip accounting + metrics + convention); load failure exits 1
    with the stderr message; --output writes the file.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_vep.py tests/cli/test_vep_cli.py -q -m "not slow"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary</fails_when>
    <automated>test "$(grep -c "dnallm-vep" pyproject.toml)" -ge 1 && test "$(grep -c "same-slot" README.md)" -ge 1 && echo WIRING-OK</automated>
    <fails_when>WIRING-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - score_variant exists with formula docstrings; both guard branches raise matchable dnallm ValueErrors (tests prove each)
  - `dnallm/cli/vep.py` exists; `grep -c "dnallm-vep" pyproject.toml` >= 1; CLI smoke tests green (JSON shape, error exit)
  - README contains the protocol section before ## 🧪 Testing with the formulas, same-slot rule, skip accounting, and the D-17 convention block; the README diff is confined to that insertion
  - `git diff HEAD -- pyproject.toml` across the lane totals exactly the dependency line + the script line
  - `uv run --no-sync pytest tests/inference/test_vep.py tests/cli/test_vep_cli.py -q -m "not slow"` fully green (existing Phase-10 classes undisturbed)
  </acceptance_criteria>
  <done>score_variant + guard live, the dnallm-vep entry point works end-to-end (mocked), the README protocol declaration completes the VEP-01 documentation surface.</done>
</task>

<task type="auto">
  <name>Task 3: ClinVar 1k × >= 5-model slow acceptance + typed skips + models.lock + coverage + CHANGELOG</name>
  <precondition>Slow-lane network route available (ClinVar + single-chromosome GRCh38 reference downloads); the five pinned scoring models cached or fetchable (all have models.lock rows).</precondition>
  <files>tests/inference/test_vep.py, tests/expected_skips.yaml, models.lock, CHANGELOG.md</files>
  <read_first>
  - tests/expected_skips.yaml (typed-prefix matcher format — exact/prefix/reason_like; the network-unavailable: precedent)
  - models.lock (pinned rows: plant-dnabert-BPE, plant-dnagpt-6mer, nucleotide-transformer-v2-50m-multi-species, plant-dnamamba-BPE-open_chromatin, plant-dnagpt-BPE-promoter)
  - 11-RESEARCH.md: State of the Art ClinVar anchor table (NT 0.80 MLM 2.5B; evo2 0.98 CLM 40B; DNABERT-2 0.538 embedding-distance; Alfisi <0.6 strict labeling), Pitfall 7 (convention comparability), D-09/D-17, Validation Architecture B5 strategy
  - dnallm/utils/genomic_coords.py (fetch_sequence uppercase precedent — reference window building conventions)
  - .planning/research/261009-paper-revision-suite-plan.md (reviewer-comment id for REV-08 — the R1-3e thread)
  - CHANGELOG.md (## [Unreleased] anchor + entry format)
  </read_first>
  <action>
  Slow-lane acceptance (D-09 tier 2 — real data stays out of the repo):
  - Test class TestClinVarAcceptance (slow-marked, typed network skip when the
    ClinVar/reference hosts are unreachable — a new typed reason prefix, e.g.
    `clinvar-unavailable:`, added to tests/expected_skips.yaml same-change):
    download the ClinVar GRCh38 VCF and a single-chromosome GRCh38 reference from
    parameterized canonical hosts (executor picks the chromosome so the 1k-sample
    target is comfortably reachable; document both URLs in the test docstring).
  - Sample ~1k variants under D-17: CLNVC single_nucleotide_variant, >= 1 review
    star, P/LP vs B/LB labels (VUS/conflicting/0-star excluded); report the
    convention block + per-star counts with the results.
  - Score with >= 5 pinned models forming a CLM/MLM mix — recommended from
    models.lock: zhangtaolab/plant-dnabert-BPE (ms), InstaDeepAI/
    nucleotide-transformer-v2-50m-multi-species (hf, MLM),
    zhangtaolab/plant-dnagpt-6mer (ms, CLM),
    zhangtaolab/plant-dnamamba-BPE-open_chromatin (ms, CLM),
    zhangtaolab/plant-dnagpt-BPE-promoter (ms/hf, CLM) — paradigm matched to each
    architecture via the Task 2 guard.
  - Assertions (honest magnitudes per Pitfall 7): every model's AUROC is recorded
    with the convention block; each within a plausible sanity band (>= 0.5 — at or
    above the random floor) and the per-model results are compared in-test against
    the RESEARCH anchor table WITHIN-paradigm/within-convention (documented finding,
    not a bare threshold); skip fractions reported per reason per model (the
    tokenizer-class finding — Pitfall 6).
  - models.lock: add sha-pinned rows for any artifact actually fetched that lacks
    one (the five models are pinned; the ClinVar/reference downloads are documented
    in the test, not models.lock — they are datasets/reference files, not model
    registries; follow the file's existing precedent for dataset rows if one
    applies, else the test docstring).
  - Per-module coverage: drive vep.py + cli/vep.py to >= 96% via the fast-lane
    tests (coverage run/report workaround — never pytest --cov).
  - CHANGELOG (D-09 discipline): re-read, then append one bullet tagged
    `(REV-08, <reviewer-comment-id — the R1-3e thread>)` under the idempotent
    `## [Unreleased]` anchor (### Added). Unique anchor; never touch sibling
    lanes' entries.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_vep.py -q -m slow -k "clinvar"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary (a typed network skip is a pass-with-skip, not a failure)</fails_when>
    <automated>uv run --no-sync coverage run -m pytest tests/inference/test_vep.py tests/cli/test_vep_cli.py -q -m "not slow" && uv run --no-sync coverage report --include="dnallm/inference/vep.py,dnallm/cli/vep.py"</automated>
    <fails_when>non-zero exit, or either module's coverage row shows a total below 96</fails_when>
    <automated>test "$(grep -c "clinvar-unavailable" tests/expected_skips.yaml)" -ge 1 && test "$(grep -c "(REV-08," CHANGELOG.md)" -ge 1 && echo B5-CLOSE-OK</automated>
    <fails_when>B5-CLOSE-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - ClinVar 1k-sample × >= 5-model (CLM/MLM mix) acceptance runs under D-17 with the convention block reported beside every AUROC and per-model skip fractions recorded
  - Typed network skip allowlisted same-change (`grep -c "clinvar-unavailable" tests/expected_skips.yaml` >= 1); no real ClinVar data committed (`git status` clean of VCF/FASTA payloads after the lane's commits)
  - vep.py and cli/vep.py per-module coverage >= 96%
  - `grep -c "(REV-08," CHANGELOG.md` >= 1 under ## [Unreleased]
  - Fast lane still green: `uv run --no-sync pytest tests/inference/test_vep.py tests/cli/test_vep_cli.py -q -m "not slow"`
  </acceptance_criteria>
  <done>VEP-01 accepted: the zero-shot VCF protocol proven on real ClinVar data across >= 5 models with honest convention reporting, the fast lane network-free at the 96% standard, evidence chain complete.</done>
</task>

</tasks>

<artifacts_produced>
Symbols this plan creates (B5 lane):
- `evaluate_vcf(...)` + `VepResult` dataclass (dnallm/inference/vep.py) — allel.read_vcf parsing, D-17 CLNSIG/CLNREVSTAT/CLNVC filtering, uppercase windows, skip accounting, registry metrics
- `score_variant(model, tokenizer, sequence, pos, ref, alt, *, paradigm)` + the paradigm↔architecture mismatch ValueError guard (D-11)
- `dnallm/cli/vep.py` (new module): the `dnallm-vep` click command
- pyproject.toml: `scikit-allel>=1.3.13,<2` dependency + `dnallm-vep = "dnallm.cli.vep:main"` script (the milestone's only sanctioned pyproject delta)
- `tests/inference/data/synthetic_variants.vcf` + reference sidecar (committed fixture, D-09 tier 1)
- Test classes: `TestEvaluateVcf`, `TestScoreVariant`, `TestClinVarAcceptance` (tests/inference/test_vep.py) + the CLI suite (tests/cli/test_vep_cli.py)
- tests/expected_skips.yaml: the clinvar-unavailable typed prefix (or equivalent typed reason)
- models.lock: rows for any newly fetched artifacts
- README.md: zero-shot VEP protocol section (formulas, same-slot rule, skip accounting, D-17 convention block)
- CHANGELOG.md: REV-08 entry under ## [Unreleased]
</artifacts_produced>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| ClinVar/reference downloads → parsers | untrusted remote data enters allel.read_vcf and window building (V5) — zip-bomb/malformed-record/huge-ALT exposure |
| VCF record fields → logic | CHROM/POS/REF/ALT/CLNSIG values drive control flow and label assignment |
| package install → venv | scikit-allel + dask[array] land as the phase's only new dependency (D-08) |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-10 | Tampering / DoS | evaluate_vcf VCF ingestion (vep.py) | high | mitigate | Untrusted-input discipline per RESEARCH V5: strict read_vcf field list; alt_number bounded and deliberately sized; SNV/label filtering before any scoring loop; row-count sanity on the sampled slice; malformed records surface as allel errors wrapped in dnallm ValueErrors at the driver boundary (never bare foreign tracebacks into user output) |
| T-11-11 | Tampering | output path construction in evaluate_vcf/CLI | medium | mitigate | Never derive filesystem paths from VCF record fields (CHROM/ID are untrusted strings): deterministic output names under caller output_dir; CLI --output is an explicit user path with parents=True exist_ok mkdir |
| T-11-12 | Tampering | CLNSIG label parsing (vep.py) | medium | mitigate | Label whitelist exactly {P/LP → 1, B/LB → 0}; every other CLNSIG value (VUS, conflicting, novel strings, multi-value fields) is excluded, not guessed — exclusion counts reported; convention block emitted with results |
| T-11-SC | Tampering | package installs (scikit-allel) | high | mitigate | Package Legitimacy Audit verdict SUS (unknown-downloads signal only; canonical github.com/cggh/scikit-allel registry + official readthedocs verified in research). The blocking human checkpoint is ALREADY SATISFIED by the owner's explicit 2026-10-09 approval (D-08) recording empirical verification of Windows cp310-313 wheels and numpy 1.26.4/2.2.0 compat before approval — re-gating a decided door would be checkpoint fatigue. Residual control: bounded range >=1.3.13,<2 in pyproject (maintenance-mode successor sgkit justifies the upper guard); no postinstall scripts in the Python ecosystem |
</threat_model>

<verification>
- Fast lane: `uv run --no-sync pytest tests/inference/test_vep.py tests/cli/test_vep_cli.py -q -m "not slow"` — network-free, all Phase-10 classes undisturbed
- Slow lane: `uv run --no-sync pytest tests/inference/test_vep.py -q -m slow -k "clinvar"` — 1k-sample × >= 5-model acceptance (typed skip when offline)
- Per-module coverage (cov-crash workaround): `uv run --no-sync coverage run -m pytest tests/inference/test_vep.py tests/cli/test_vep_cli.py -q -m "not slow"` then `uv run --no-sync coverage report --include="dnallm/inference/vep.py,dnallm/cli/vep.py"` — both >= 96%
- Invariants: `git diff HEAD -- pyproject.toml` totals exactly two lines (dependency + script); `git diff HEAD -- dnallm/__init__.py` empty; `git diff HEAD -- dnallm/tasks/metric_registry.py` empty; README diff confined to the new protocol section; `grep -c "(REV-08," CHANGELOG.md` >= 1; skip allowlist entry present
- Owner directive 2026-10-09: run ONLY the targeted verifiers above and this lane's test files — no repo-wide lanes, no check_code.py full sweeps
</verification>

<success_criteria>
- VEP-01 fully test-proven: evaluate_vcf via scikit-allel with the D-17 convention, same-slot rule + skip-as-data accounting (fraction as finding), paradigm guard raising, coordinate/uppercase/multi-allelic/empty semantics proven on the committed fixture, CLI entry live, formulas in docstrings + README, ClinVar 1k × >= 5-model slow acceptance with honest magnitude reporting
- Every dnallm/ change shipped with its tests in the same commit; vep.py + cli/vep.py >= 96% per-module coverage
- The only dependency change in the phase is the sanctioned scikit-allel addition; fast lane network-free; no real data committed; CHANGELOG entry traceable
</success_criteria>

<output>
Create `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-05-SUMMARY.md` when done
</output>
