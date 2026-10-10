---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/models/model.py
  - tests/models/test_model.py
  - README.md
  - CHANGELOG.md
autonomous: true
requirements: [BASE-01]
coupling_justified: >
  CHANGELOG.md: single sanctioned cross-lane append surface (one REV-06 unique-anchor
  bullet, D-09 discipline). README.md: two lanes append documentation this wave (this
  plan and 11-05) — this plan's touch is ONE paragraph inside the existing
  "## 🧬 Supported Models" section (top of README, line ~32) while 11-05 adds a new
  section near the bottom (before "## 🧪 Testing", line ~517). Distinct, distant
  anchors; scoped Edit only (never full-file Write); re-read-before-edit;
  pathspec-limited commits. All other files are exclusively this lane's (model.py
  sole owner per the ROADMAP ownership map).

estimate:
  tokens: 22000
  raw_tokens: 22000
  tasks: 3
  confidence: low   # calibration sample_count=0, factor=1

must_haves:
  truths:
    - "load_model_and_tokenizer(..., random_init=True) produces a genuinely from-scratch model: AutoConfig.from_pretrained + AutoModel*.from_config — never from_pretrained weights (BASE-01)"
    - "A loud 'randomly initialized' banner plus a per-tensor parameter-hash table emits through get_logger INFO lines (D-05) — every float parameter tensor gets its own short hash, not one global hash"
    - "Per-tensor hashes of the random-init path differ from the pretrained path on every float parameter tensor; the explicitly documented exceptions (with counts) are non-float buffers (int/bool) and tied-weight aliases that share storage (BASE-01 precision/adjacency edges — RESEARCH A-class exception handling)"
    - "Hashes are computed over raw tensor bytes (detach → cpu → numpy → tobytes → short hex); tied weights report identical hashes across the tie because they share storage — expected, documented, not a failure (BASE-01 precision edge)"
    - "Same-seed reproducibility: two random_init loads with the identical seed on CPU produce identical per-tensor hash tables; seeding happens CPU-canonical BEFORE model construction (BASE-01)"
    - "The no-download proof asserts the weight-fetch path (_get_model_path_and_imports) is never invoked for weights under random_init — the config.json fetch via AutoConfig.from_pretrained is explicitly allowed and documented (BASE-01; the proof is also a security property: no uncontrolled egress)"
    - "A config.json carrying no weight-init semantics still random-initializes cleanly through from_config (BASE-01 empty-config edge)"
    - "The tokenizer loads normally on the random_init path (the post-chain tokenizer handling is preserved — the guarded merge chain's must-not-return-early contract)"
    - "Special-family handlers outside RANDOM_INIT_SUPPORTED_FAMILIES raise an explicit matchable ValueError naming the family and the allowlist (D-06/D-07); generic Auto* families are always allowed"
    - "Two proven architectures in the slow lane: one generic AutoModel family (BERT-style small model via the from_config path) + the mamba allowlist member (trust_remote_code from_config branch) (D-06)"
  artifacts:
    - path: dnallm/models/model.py
      provides: "load_model_and_tokenizer random_init kwarg, RANDOM_INIT_SUPPORTED_FAMILIES frozenset, from_config random path with banner + per-tensor hash logging"
      contains: "RANDOM_INIT_SUPPORTED_FAMILIES"
    - path: tests/models/test_model.py
      provides: "TestRandomInit class — fast-lane mocked proofs + slow-lane two-architecture acceptance"
      contains: "random_init"
    - path: README.md
      provides: "From-scratch baselines paragraph under ## 🧬 Supported Models documenting random_init and the family allowlist (D-07)"
      contains: "RANDOM_INIT_SUPPORTED_FAMILIES"
  key_links:
    - from: dnallm/models/model.py
      to: transformers AutoConfig/AutoModel
      via: "random_init=True → AutoConfig.from_pretrained(trust_remote_code=True) + Auto*.from_config(config) — the no-checkpoint instantiation path (verified transformers 5.17.0 auto_factory.py:206-233)"
      pattern: "from_config"
    - from: dnallm/models/model.py
      to: dnallm/utils/logger.py
      via: "banner + per-tensor hash table via get_logger INFO (D-05) — get_logger, NOT trainer's print style"
      pattern: "get_logger"
    - from: dnallm/models/model.py
      to: dnallm/models/model.py special-handler chain
      via: "RANDOM_INIT_SUPPORTED_FAMILIES frozenset gates the _handle_* chain: off-list special families raise ValueError before any handler claims the load (D-07)"
      pattern: "RANDOM_INIT_SUPPORTED_FAMILIES"
  prohibitions:
    - "No pyproject.toml changes (B5 solely owns pyproject this wave)"
    - "No new dnallm/__init__.py re-exports (facade stays byte-stable)"
    - "No test matching transformers/peft foreign exception strings — assert dnallm's own ValueError messages only (version-span rule)"
    - "No private transformers symbol imports in tests"
    - "No assertion that random_init performs zero network at all — config.json fetch is allowed; the proof targets the weight-fetch path only"
    - "No edits outside this lane's files (model.py, own tests, README Supported-Models paragraph, CHANGELOG append)"
    - "No Co-Authored-By trailers; commits are pathspec-limited (git commit -- <paths>) in the shared tree"
    - "No single-global-hash test — the hash proof is per tensor by construction (a global hash passes with leftover pretrained tensors)"
---

<objective>
B2 (REV-06): genuinely from-scratch baseline loading — `random_init=True` on
load_model_and_tokenizer with a loud banner, per-tensor hash proof, CPU-canonical
seeding, a no-download guarantee, and a special-family allowlist.

Purpose: BASE-01 — reviewers need honest from-scratch baselines whose random
initialization is provable (not "probably random"), reproducible, and download-free.
Output: random_init path in model.py, allowlist frozenset, fast-lane proofs, slow-lane
two-architecture acceptance, README documentation, CHANGELOG entry.
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
  <name>Task 1: random_init end-to-end — kwarg → allowlist gate → from_config random model + banner/hash logs (mocked boundary)</name>
  <files>dnallm/models/model.py, tests/models/test_model.py</files>
  <read_first>
  - dnallm/models/model.py (lines 740-918: signature at 753-761, num_labels ValueError idiom at 744-748, special-handler chain at 806-863, _get_model_path_and_imports call at ~868, guarded merge chain + tokenizer post-processing at 898-918 with the must-NOT-return-early comment)
  - 11-RESEARCH.md sections: Pattern 3 (verified transformers 5.17.0 from_config path + B2 shape), Pitfall 4 (meta-device/tied-weights traps, no-download proof target), Assumption A4 (head-shaping via config attributes), Assumption A6 (mamba is NOT a _handle_* family — loads generic with trust_remote_code)
  - 11-PATTERNS.md: model.py random_init analog (dispatch chain, logging via get_logger, ValueError idiom)
  - tests/models/test_model.py (existing test conventions in this file)
  </read_first>
  <action>
  One end-to-end path first: `load_model_and_tokenizer(model_name, task_config,
  random_init=True)` on a tiny local config returns a randomly initialized model with
  the banner + hash table logged and the tokenizer loaded — proven on a mocked/tiny
  boundary with zero weight fetch.

  model.py:
  - Add `random_init: bool = False` to the load_model_and_tokenizer signature (same
    style as the existing kwargs) and document it in the Google-style Args/Raises
    docstring ( Raises: ValueError for random_init on unsupported special families).
  - Add module-level `RANDOM_INIT_SUPPORTED_FAMILIES: frozenset[str]` (D-07) near the
    other module constants, with a docstring/comment listing what qualifies; seed it
    with the mamba-trust-remote-code member per D-06/A6 (the allowlist gates SPECIAL
    families only — the generic Auto* path is always allowed and needs no entry).
  - At the top of the special-handler chain: when random_init is on and a special
    handler would claim this model (the existing family-detection the handlers
    themselves use — e.g. name-prefix matching), check the family against the
    frozenset; off-list → matchable ValueError naming the family, the allowlist
    contents, and the fact that from-scratch init is unsupported for it (D-06).
  - Generic random path (before the pretrained _get_model_path_and_imports call):
    `config = AutoConfig.from_pretrained(model_name, trust_remote_code=True,
    revision=revision)`; set head-shaping fields on the config object exactly as the
    pretrained path does (safe_num_labels, id2label/label2id — A4: config attributes,
    not ctor kwargs); `torch.manual_seed(seed)` with CPU-canonical seeding BEFORE
    construction (seed from task_config/train seed or a documented default); select
    the same Auto* class the task-type path selects and call
    `AutoModelClass.from_config(config, trust_remote_code=True)` — never
    from_pretrained. Keep the model on CPU through init (Pitfall 4b: init on CPU then
    .to(device), avoiding meta-device tied-weight copies).
  - Log via get_logger (NOT print — this is model.py): one loud INFO banner line
    containing "randomly initialized", then one INFO line per parameter tensor:
    tensor name + short hash (sha over `.detach().cpu().numpy().tobytes()`, first 10
    hex chars) + shape (D-05 — greppable, no sidecar file, no output_dir dependency).
  - Preserve the tokenizer post-processing below the merge chain for the random path
    too (the chain's must-not-return-early contract — tokenizer loads normally).

  Tests (tests/models/test_model.py, new TestRandomInit class, fast lane, network-free
  via tiny local config fixtures / tmp config.json):
  - Tracer test: random_init=True returns (model, tokenizer); banner + >= 1 per-tensor
    hash line captured via caplog at INFO level.
  - Allowlist: a mocked/patched special-family detection (e.g. an evo-style name)
    raises ValueError matching the family name and "RANDOM_INIT_SUPPORTED_FAMILIES"
    semantics (matchable message).
  - Mocked generic path: patch AutoConfig.from_pretrained/Auto*.from_config with tiny
    fakes so no network and no hub is touched.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m "not slow"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - `grep -n "random_init" dnallm/models/model.py` shows the kwarg, the branch, and the docstring entries; `grep -c "RANDOM_INIT_SUPPORTED_FAMILIES" dnallm/models/model.py` >= 2 (definition + use)
  - `grep -n "randomly initialized" dnallm/models/model.py` hits the banner; caplog-based test asserts both the banner and per-tensor hash lines
  - The from_config call path appears and no `from_pretrained` model-construction call exists on the random branch
  - `uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m "not slow"` passes with >= 3 tests, all network-free (mocked boundary)
  </acceptance_criteria>
  <done>random_init kwarg live end-to-end on the mocked boundary: from_config random model, banner + per-tensor hashes via get_logger, allowlist ValueError for off-list special families, tokenizer intact.</done>
</task>

<task type="auto">
  <name>Task 2: Fast-lane proofs — no-download, same-seed reproducibility, seed-before-init ordering</name>
  <files>tests/models/test_model.py</files>
  <read_first>
  - dnallm/models/model.py (Task 1's random branch + `_get_model_path_and_imports` definition — the weight-fetch patch target)
  - 11-RESEARCH.md Pitfall 4 (no-download proof = patched weight-fetch path, NOT zero-network; seed-then-init ordering; per-tensor comparison rationale)
  - tests/conftest.py (tiny_model_factory — local tiny models for network-free comparisons)
  </read_first>
  <action>
  Extend TestRandomInit with the proof battery (still fast lane, network-free):
  - No-download proof: patch `_get_model_path_and_imports` (the weight-fetch seam) so
    any invocation under random_init fails the test ( MagicMock whose side_effect
    records / raises AssertionError). Document in the test docstring that the
    AutoConfig config.json fetch is legitimately excluded from this proof.
  - Same-seed reproducibility: two random_init loads with the identical seed →
    identical per-tensor hash tables (collect the hash lines from caplog or recompute
    via the same helper); different seeds → at least the head/first layers differ.
  - Seed-before-init ordering: patch torch.manual_seed with a recorder and assert it
    is called BEFORE the model-construction call (ordering edge — a seed applied
    after init silently breaks reproducibility).
  - Tied-weights/byte-level semantics: using a tiny model with tied weights (or a
    constructed one), assert the hash helper reports identical hashes for tensors
    sharing storage (documented expected behavior, precision edge) and that int/bool
    buffers are included in the table with their own hashes.
  - Config-without-init-semantics edge: a minimal tmp config.json (no init-relevant
    fields) still random-initializes (hashes differ across two differently-seeded
    loads).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m "not slow"</automated>
    <fails_when>non-zero exit, or a test named test_random_init_* fails in the output, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - The no-download test fails if the random branch ever calls _get_model_path_and_imports (side_effect guard) — proven by the test existing and passing against the real branch
  - Same-seed test asserts full hash-table equality; the ordering test asserts manual_seed precedes construction
  - Tied-weight alias and non-float buffer behaviors asserted and documented in docstrings (the RESEARCH A-class exceptions with counts available in the log table)
  - `uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m "not slow"` passes with >= 8 tests total in TestRandomInit
  </acceptance_criteria>
  <done>All BASE-01 proof clauses hold on the fast lane: no weight download, same-seed reproducibility, seed-before-init ordering, byte-level hash semantics documented for ties and buffers.</done>
</task>

<task type="auto">
  <name>Task 3: Slow-lane two-architecture acceptance + per-tensor difference proof + README + CHANGELOG</name>
  <precondition>models.lock-pinned small models zhangtaolab/plant-dnabert-BPE (ms) and zhangtaolab/plant-dnamamba-BPE-open_chromatin (ms) are cached or the slow-lane network route is available.</precondition>
  <files>tests/models/test_model.py, README.md, CHANGELOG.md</files>
  <read_first>
  - models.lock (pinned rows: plant-dnabert-BPE ms route, plant-dnamamba-BPE-open_chromatin ms route)
  - 11-RESEARCH.md: Pitfall 4 (leftover-pretrained detection via per-tensor diff), Environment Availability (cached models)
  - README.md (## 🧬 Supported Models section, line ~32 — the anchor for the new paragraph)
  - .planning/research/261009-paper-revision-suite-plan.md (reviewer-comment id to cite inline for REV-06)
  - CHANGELOG.md (## [Unreleased] anchor + entry format)
  </read_first>
  <action>
  Slow-lane acceptance (slow-marked, typed network skip when the models are not
  cached — follow this file's existing real-model test conventions):
  - Generic AutoModel family (D-06 first member): `zhangtaolab/plant-dnabert-BPE`
    (modelscope route, models.lock row exists) — random_init=True: assert banner +
    hash lines logged, same-seed reload reproduces the hash table, tokenizer loads,
    and one forward pass runs on CPU.
  - Mamba allowlist member (D-06 second member): 
    `zhangtaolab/plant-dnamamba-BPE-open_chromatin` (modelscope route) — exercises the
    trust_remote_code from_config branch; same assertions.
  - Per-tensor difference proof (Pitfall 4c): for the plant-dnabert-BPE architecture,
    load pretrained once and random_init once; assert every float parameter tensor's
    hash differs; explicitly enumerate and count the allowed exceptions (non-float
    buffers; tied-weight aliases sharing storage) in the test output/docstring.
  - models.lock: add rows only if a genuinely new artifact was fetched (both models
    already pinned — expected delta: none).

  README.md (D-07 — scoped Edit, re-read first):
  - Inside the existing `## 🧬 Supported Models` section (top of file), add one short
    paragraph documenting from-scratch baselines: `load_model_and_tokenizer(...,
    random_init=True)` semantics, the banner + per-tensor hash proof, and
    RANDOM_INIT_SUPPORTED_FAMILIES as the special-family allowlist. Touch nothing
    else in the file (11-05 edits a distant section concurrently).

  CHANGELOG (D-09 discipline):
  - Re-read, then append one bullet tagged `(REV-06, <reviewer-comment-id>)` under the
    idempotent `## [Unreleased]` anchor (### Added). Unique anchor; never touch
    sibling lanes' entries.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m slow</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary</fails_when>
    <automated>test "$(grep -c "RANDOM_INIT_SUPPORTED_FAMILIES" README.md)" -ge 1 && test "$(grep -c "(REV-06," CHANGELOG.md)" -ge 1 && echo B2-DOCS-OK</automated>
    <fails_when>B2-DOCS-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - Two architectures proven in the slow lane (generic AutoModel BERT-style + mamba trust_remote_code member), each with banner/hashes/reproducibility/tokenizer/forward assertions
  - Per-tensor pretrained-vs-random difference proven with explicitly counted exceptions (ties, non-float buffers)
  - `grep -c "RANDOM_INIT_SUPPORTED_FAMILIES" README.md` >= 1 and the README diff is confined to the Supported Models section
  - `grep -c "(REV-06," CHANGELOG.md` >= 1 under ## [Unreleased]
  - Per-module scoped coverage: `uv run --no-sync coverage run -m pytest tests/models/test_model.py -q` then `uv run --no-sync coverage report --include="dnallm/models/model.py"` shows the module >= 96%
  </acceptance_criteria>
  <done>BASE-01 fully proven: honest from-scratch loading on two architectures with per-tensor hash evidence, no-download guarantee, reproducibility, allowlist enforcement, README + CHANGELOG documentation.</done>
</task>

</tasks>

<artifacts_produced>
Symbols this plan creates (B2 lane):
- `load_model_and_tokenizer(..., random_init: bool = False)` kwarg (dnallm/models/model.py)
- `RANDOM_INIT_SUPPORTED_FAMILIES: frozenset[str]` module-level constant (dnallm/models/model.py, D-07)
- Private helpers on the random path (named_modules hash table builder, allowlist gate) inside model.py
- Test class `TestRandomInit` (tests/models/test_model.py): fast-lane proofs + slow-lane two-architecture acceptance + per-tensor difference proof
- README.md: from-scratch baselines paragraph under ## 🧬 Supported Models
- CHANGELOG.md: REV-06 entry under ## [Unreleased]
</artifacts_produced>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| remote config.json → AutoConfig | model config is fetched from HF/ModelScope hubs (untrusted-by-default remote content) |
| user kwargs → loader dispatch | random_init + family detection decide which code path claims the load |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-04 | Information Disclosure | random_init weight-fetch path in model.py | medium | mitigate | A from-scratch path that silently downloads pretrained weights is both wrong science and uncontrolled egress: no-download proof patches _get_model_path_and_imports and fails on any invocation; config.json fetch explicitly scoped and documented |
| T-11-05 | Elevation of Privilege | random_init × special-family handlers | medium | mitigate | random_init widens the trust_remote_code surface if every special family opts in: RANDOM_INIT_SUPPORTED_FAMILIES frozenset (D-07) gates the chain with a matchable ValueError off-list; generic Auto* only for everything else |
| T-11-SC | Tampering | package installs | high | mitigate | This lane installs nothing (transformers/torch are existing dependencies; zero pyproject changes). The phase's only sanctioned install is plan 11-05's scikit-allel under owner decision D-08 |
</threat_model>

<verification>
- Fast lane: `uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m "not slow"` — network-free, all proofs green
- Slow lane: `uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m slow` — two architectures + per-tensor difference
- Per-module coverage (cov-crash workaround): `uv run --no-sync coverage run -m pytest tests/models/test_model.py -q` then `uv run --no-sync coverage report --include="dnallm/models/model.py"` — >= 96%
- Invariants: `git diff HEAD -- pyproject.toml` and `git diff HEAD -- dnallm/__init__.py` empty over this lane's commits; README diff confined to the Supported Models section; `grep -c "(REV-06," CHANGELOG.md` >= 1
- Owner directive 2026-10-09: run ONLY the targeted verifiers above and this lane's test files — no repo-wide lanes, no check_code.py full sweeps
</verification>

<success_criteria>
- BASE-01 fully test-proven: from_config random init, loud banner + per-tensor hash proof differing from pretrained on every float tensor (documented exceptions counted), same-seed reproducibility, no weight download, tokenizer normal, allowlist ValueError, two architectures
- Every dnallm/ behavior change shipped with its tests in the same commit; model.py >= 96% per-module coverage
- No new dependencies, no facade re-exports, no out-of-lane edits, CHANGELOG entry traceable
</success_criteria>

<output>
Create `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-02-SUMMARY.md` when done
</output>
