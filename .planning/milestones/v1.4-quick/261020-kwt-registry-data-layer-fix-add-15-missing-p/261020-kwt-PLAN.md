---
phase: 261020-kwt
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/models/model_info.yaml
  - dnallm/models/modeling_auto.py
  - tests/models/test_model_info_registry.py
autonomous: true
requirements:
  - KWT-01
  - KWT-02
  - KWT-03
  - KWT-04
user_setup: []

estimate:
  tokens: 60000
  raw_tokens: 30000
  tasks: 2
  confidence: med

must_haves:
  truths:
    - "All 15 zhangtaolab plant tokenizer-variant checkpoints are pretrained registry entries with the correct task_type (mask for the dnabert / nucleotide-transformer / dnamodernbert variants; generation for the dnagpt / dnagemma / dnamamba variants)"
    - "PRETRAIN_MODEL_MAPS maps Plant DNAGPT to Causal Language Model and no family task-class string anywhere carries the misspelled form"
    - "Every finetuned registry entry has base_model, except any entry explicitly listed in the SUMMARY as underivable (expected: none — planning-time derivation resolved all 164)"
    - "Every base_model value resolves to a pretrained registry model id, or is one of the two documented pre-existing external PlantCAD2 pointers"
    - "The tRNADetector and tRNAPointer MODEL_INFO titles read 'a DNA sequence' (space restored)"
  artifacts:
    - "dnallm/models/model_info.yaml — 49 pretrained entries (was 34), 187 finetuned entries all carrying base_model lineage"
    - "dnallm/models/modeling_auto.py — 3 corrected task-class strings, 2 corrected titles"
    - "tests/models/test_model_info_registry.py — fast-leg registry-integrity tests, no network"
  key_links:
    - "finetuned base_model values → pretrained section model ids (registry-internal lineage resolution, verified by test)"
    - "PRETRAIN_MODEL_MAPS['Plant DNAGPT'] → ['Zhangtaolab', 'Causal Language Model']"
    - "tests/models/test_model_info_registry.py → packaged dnallm/models/model_info.yaml via Path(dnallm.__file__).parent (same pattern as tests/models/test_plant_helixseek_registry.py)"
---

<objective>
Registry data-layer fix across the model registry YAML and the MODEL_INFO / PRETRAIN_MODEL_MAPS tables: add the 15 missing PDLLMs tokenizer-variant pretrained entries (KWT-01), correct 3 task-class strings (KWT-02), backfill base_model lineage for all 164 finetuned entries missing it (KWT-03), and fix 2 title typos (KWT-04) — with new registry-integrity tests shipped in the same change (owner rule: dnallm/ changes ship with pytest coverage).

Purpose: the YAML registry is missing every non-BPE tokenizer variant that MODEL_INFO already advertises (6mer/singlebase/2-5mer), the lineage field exists for only 23 of 187 finetuned entries, and two task-class strings misroute family detection — Plant DNAGPT is tagged Masked while its checkpoint config is GPT2LMHeadModel, and both EVO entries carry a spelling error in the task class.
Output: corrected dnallm/models/model_info.yaml, corrected dnallm/models/modeling_auto.py, new tests/models/test_model_info_registry.py.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@/home/forrest/Github/DNALLM/.planning/STATE.md
@dnallm/models/model_info.yaml
@dnallm/models/modeling_auto.py
@tests/models/test_plant_helixseek_registry.py

Planning-time grounding (verified 2026-10-20 against the working tree — reuse, do not re-fetch anything from HF/ModelScope):

- Registry shape: `pretrained:` section lines 1-172 (34 entries), `finetuned:` section lines 175-1669 (187 entries). Pretrained entry convention: `name` / `model` / `task.describe` / `task.task_type`, double-quoted strings, 2-space indent. The Plant DNABERT entry at line 2 is the style template for the 15 additions.
- The 15 missing repo ids all already exist in MODEL_INFO `huggingface`/`modelscope` lists (dnallm/models/modeling_auto.py lines 48-164). Architectures cross-check the task_types: BertForMaskedLM / EsmForMaskedLM / ModernBertForMaskedLM → mask; GPT2LMHeadModel / GemmaForCausalLM / MambaForCausalLM → generation.
- base_model placement convention (see PlantCAD2 entries at model_info.yaml lines 1460-1462 and PlantHelixSeek entries at 1652-1663): a sibling of `model:`, on the line directly after `model:` and before `task:`, double-quoted.
- base_model coverage at HEAD: 23 of 187 finetuned entries have it (21 PlantCAD2 + 2 PlantHelixSeek). All 164 missing ones were derivable unambiguously at planning time: 136 by longest pretrained-repo-id prefix at a `-` boundary (zero multi-prefix collisions), 2 by explicit repo map (tRNADetector, tRNAPointer), 26 by finetuned-entry-name family prefix (see Task 2).
- Pre-existing external pointers: 15 PlantCAD2 finetuned entries point base_model at `kuleshov-group/PlantCAD2-Medium-l48-d1024` / `kuleshov-group/PlantCAD2-Large-l48-d1536`, which are real upstream repos but NOT registry pretrained entries (only PlantCAD2-Small is registered). Do NOT touch these; the test allowlists them.
- All 187 finetuned model ids are unique, all 34 pretrained ids are unique, and the two sections share zero repo ids — so keying text insertions on `model: "<repo>"` lines is safe.
- model_info.yaml lines ~1636-1651 carry a provenance comment block (freeze hashes for PlantHelixSeek). A YAML round-trip (yaml.dump) would destroy it — Task 2 must use text insertion, never round-trip.
- Grep-gate baselines at HEAD (HEAD e60303f-ish, branch v1.4): `grep -c "Casual" dnallm/models/modeling_auto.py` = 2 (lines 23, 24); `grep -c "aDNA" dnallm/models/modeling_auto.py` = 2 (lines 166, 179); pretrained count = 34; finetuned-with-base_model = 23. Post-fix targets: 0, 0, 49, 187 respectively.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add 15 pretrained variant entries, fix 3 task-class strings and 2 title typos, with registry-integrity tests</name>
  <files>dnallm/models/model_info.yaml, dnallm/models/modeling_auto.py, tests/models/test_model_info_registry.py</files>
  <behavior>
    - Test: the pretrained section parses to exactly 49 entries and contains each of the 15 new repo ids exactly once, each with the pinned task_type (mask: plant-dnabert-6mer, plant-dnabert-singlebase, plant-nucleotide-transformer-6mer, plant-nucleotide-transformer-singlebase, plant-dnamodernbert-singlebase; generation: plant-dnagpt-6mer, plant-dnagpt-singlebase, plant-dnagemma-6mer, plant-dnagemma-singlebase, plant-dnamamba-2mer, plant-dnamamba-3mer, plant-dnamamba-4mer, plant-dnamamba-5mer, plant-dnamamba-6mer, plant-dnamamba-singlebase) and a non-empty name whose wording mentions the tokenizer variant (6mer / singlebase / 2mer-5mer).
    - Test: PRETRAIN_MODEL_MAPS["Plant DNAGPT"] == ["Zhangtaolab", "Causal Language Model"] (import from dnallm.models.modeling_auto).
    - Test: no string in any PRETRAIN_MODEL_MAPS family entry contains the misspelled task class (the test defines the misspelling literal itself; read modeling_auto.py lines 23-24 at HEAD for the exact form).
    - Test: MODEL_INFO titles for tRNADetector and tRNAPointer both contain "a DNA sequence" and neither contains the joined form.
  </behavior>
  <action>
    KWT-01 — In dnallm/models/model_info.yaml, insert 15 new entries into the `pretrained:` section immediately after the existing "Plant DNAModernBert" entry (ends line 31), keeping the six BPE family entries grouped first. One entry per repo id, mirroring the "Plant DNABERT" entry style exactly (name / model / task.describe / task.task_type, double quotes, 2-space indent): "Plant DNABERT 6mer" (zhangtaolab/plant-dnabert-6mer, mask), "Plant DNABERT singlebase" (zhangtaolab/plant-dnabert-singlebase, mask), "Plant DNAGemma 6mer" (zhangtaolab/plant-dnagemma-6mer, generation), "Plant DNAGemma singlebase" (zhangtaolab/plant-dnagemma-singlebase, generation), "Plant DNAGPT 6mer" (zhangtaolab/plant-dnagpt-6mer, generation), "Plant DNAGPT singlebase" (zhangtaolab/plant-dnagpt-singlebase, generation), "Plant NT 6mer" (zhangtaolab/plant-nucleotide-transformer-6mer, mask), "Plant NT singlebase" (zhangtaolab/plant-nucleotide-transformer-singlebase, mask), "Plant DNAMamba 2mer" through "Plant DNAMamba 6mer" (zhangtaolab/plant-dnamamba-2mer, -3mer, -4mer, -5mer, -6mer, all generation), "Plant DNAMamba singlebase" (zhangtaolab/plant-dnamamba-singlebase, generation), "Plant DNAModernBert singlebase" (zhangtaolab/plant-dnamodernbert-singlebase, mask). Each task.describe must name the tokenizer variant, e.g. "Pretrained model for DNA sequence understanding with 6mer tokenizer." / "...with singlebase tokenizer." (pattern follows the family's model_tags in MODEL_INFO). Do not add base_model to pretrained entries and do not touch the finetuned section in this task.
    KWT-02 — In dnallm/models/modeling_auto.py, fix exactly 3 strings in PRETRAIN_MODEL_MAPS: line 7 Plant DNAGPT second element from "Masked Language Model" to "Causal Language Model" (verified this session: the remote config.json is GPT2LMHeadModel / model_type gpt2); lines 23 and 24, EVO-1 and EVO-2, replace the misspelled task class (read the two lines for the exact current spelling) with "Causal Language Model". Change nothing else in the file in this task beyond KWT-04.
    KWT-04 — In the same file, fix the two tRNA titles (lines 166 and 179): the joined two-word form becomes "Model for predicting whether a DNA sequence is a tRNA in plants" (space inserted after the leading article).
    Then create tests/models/test_model_info_registry.py per the behavior block, following tests/models/test_plant_helixseek_registry.py conventions: module docstring explaining these are fast-leg structure tests over the packaged YAML (no network), REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml", yaml.safe_load with a pytest.fail wrapper on YAMLError, absolute imports (import dnallm, from dnallm.models.modeling_auto import MODEL_INFO, PRETRAIN_MODEL_MAPS). Google-style docstrings, PEP 604 unions, descriptive test_* names. Keep the misspelled task class and the joined title form each in exactly one module-level test constant so the production sources stay the only other place they could regress from.
  </action>
  <verify>
    <automated>cd /home/forrest/Github/DNALLM &amp;&amp; .venv/bin/python -m pytest tests/models/test_model_info_registry.py -v &amp;&amp; .venv/bin/python -c "import yaml; d=yaml.safe_load(open('dnallm/models/model_info.yaml')); assert len(d['pretrained'])==49, len(d['pretrained'])" &amp;&amp; [ "$(grep -c Casual dnallm/models/modeling_auto.py)" = "0" ] &amp;&amp; [ "$(grep -c aDNA dnallm/models/modeling_auto.py)" = "0" ] &amp;&amp; .venv/bin/ruff format --check tests/models/test_model_info_registry.py &amp;&amp; .venv/bin/ruff check tests/models/test_model_info_registry.py</automated>
  </verify>
  <done>
    Pretrained section holds 49 entries including all 15 variants with correct task_type; Plant DNAGPT maps to Causal Language Model; zero task-class misspellings and zero joined-title typos remain in modeling_auto.py (baselines were 2 and 2); the new test file passes green and is ruff-clean; no existing test regresses (tests/models/ collection still green).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Backfill base_model lineage for the 164 finetuned entries, with lineage tests</name>
  <files>dnallm/models/model_info.yaml, tests/models/test_model_info_registry.py</files>
  <precondition>Task 1 landed — the 15 variant pretrained entries exist (several bases, e.g. the singlebase variants, only resolve after Task 1).</precondition>
  <behavior>
    - Test: every finetuned entry has a base_model key unless its model id is listed in a DERIVATION_EXEMPT allowlist constant; the test also asserts the allowlist equals the actual set of entries lacking base_model, so any future entry added without lineage fails loudly (expected exempt set after this task: empty).
    - Test: every base_model value is either a model id present in the pretrained section or a member of EXTERNAL_BASE_ALLOWLIST == {kuleshov-group/PlantCAD2-Medium-l48-d1024, kuleshov-group/PlantCAD2-Large-l48-d1536} (pre-existing accurate external pointers, out of scope for this fix).
    - Test (spot-pins of the load-bearing mappings): zhangtaolab/tRNADetector -> zhangtaolab/plant-dnamamba-singlebase; zhangtaolab/tRNAPointer -> zhangtaolab/plant-nucleotide-transformer-singlebase; zhangtaolab/plant-dnabert-BPE-promoter -> zhangtaolab/plant-dnabert-BPE; zhangtaolab/plant-dnamamba-6mer-promoter -> zhangtaolab/plant-dnamamba-6mer; zhangtaolab/agront-1b-promoter -> InstaDeepAI/agro-nucleotide-transformer-1b; zhangtaolab/dnabert2-promoter -> zhihan1996/DNABERT-2-117M; zhangtaolab/nucleotide-transformer-v2-100m-promoter -> InstaDeepAI/nucleotide-transformer-v2-100m-multi-species.
    - Test: no pretrained entry has a base_model key (lineage is a finetuned-only field).
  </behavior>
  <action>
    KWT-03 — Compute the base for each of the 164 finetuned entries lacking base_model using this three-tier derivation (planning-time simulation resolved all 164 with zero ambiguity; expected split 136 / 2 / 26): (1) longest pretrained repo id that is a prefix of the finetuned repo id at a `-` boundary, evaluated against the post-Task-1 49-entry pretrained set — e.g. zhangtaolab/plant-dnabert-BPE-promoter -> zhangtaolab/plant-dnabert-BPE, zhangtaolab/plant-dnamamba-singlebase-lncRNAs -> zhangtaolab/plant-dnamamba-singlebase; (2) explicit repo map for the two alias repos: zhangtaolab/tRNADetector -> zhangtaolab/plant-dnamamba-singlebase, zhangtaolab/tRNAPointer -> zhangtaolab/plant-nucleotide-transformer-singlebase; (3) finetuned-entry-name family prefix for the three families whose repos use short aliases — entry name starting "DNABERT-2 " -> zhihan1996/DNABERT-2-117M, "Nucleotide Transformer " -> InstaDeepAI/nucleotide-transformer-v2-100m-multi-species, "AgroNT " -> InstaDeepAI/agro-nucleotide-transformer-1b (26 entries; the "Plant Nucleotide Transformer ..." names do NOT match this key). If any entry still resolves ambiguously or not at all, leave it without base_model and record it in the SUMMARY under an explicit "underivable" list — never guess. Apply the edits by inserting one line `base_model: "<resolved id>"` directly after the `model:` line of each target finetuned entry (placement matches the PlantCAD2/PlantHelixSeek convention at lines 1460-1462/1652-1663). Use text insertion keyed on the unique `model: "<repo>"` lines inside the finetuned section — a scripted insertion or careful manual edit both work; do NOT round-trip the file through yaml.dump, it would destroy the provenance comment block at lines ~1636-1651 and reformat 1600+ untouched lines. Then extend tests/models/test_model_info_registry.py with the lineage tests from the behavior block (module constants for the two allowlists; assert base_model resolution against the parsed pretrained section).
  </action>
  <verify>
    <automated>cd /home/forrest/Github/DNALLM &amp;&amp; .venv/bin/python -m pytest tests/models/test_model_info_registry.py tests/models/test_plant_helixseek_registry.py -v &amp;&amp; .venv/bin/python -c "import yaml; d=yaml.safe_load(open('dnallm/models/model_info.yaml')); pre={e['model'] for e in d['pretrained']}; ext={'kuleshov-group/PlantCAD2-Medium-l48-d1024','kuleshov-group/PlantCAD2-Large-l48-d1536'}; missing=[e['model'] for e in d['finetuned'] if 'base_model' not in e]; bad=[(e['model'],e['base_model']) for e in d['finetuned'] if 'base_model' in e and e['base_model'] not in pre and e['base_model'] not in ext]; assert not missing, missing; assert not bad, bad; assert len(d['finetuned'])==187" &amp;&amp; .venv/bin/ruff format --check tests/models/test_model_info_registry.py &amp;&amp; .venv/bin/ruff check tests/models/test_model_info_registry.py</automated>
  </verify>
  <done>
    All 187 finetuned entries carry base_model except any explicitly listed as underivable in the SUMMARY (expected: none); every base_model resolves to a pretrained registry id or one of the two documented external PlantCAD2 pointers; the spot-pinned mappings hold; the 23 pre-existing base_model values (PlantCAD2, PlantHelixSeek) are byte-unchanged; both registry test files pass green together and the new tests are ruff-clean.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| registry YAML → downstream consumers | users and tooling (model discovery, fine-tuning lineage) read model_info.yaml / MODEL_INFO as authoritative metadata; wrong values mislead without any runtime error |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261020-01 | Tampering | dnallm/models/model_info.yaml lineage + variant data | medium | mitigate | registry-integrity tests pin all 15 variant entries with task_type, enforce base_model presence + resolution (registry-internal or explicit external allowlist), and fail loudly on any future finetuned entry added without lineage |
| T-261020-02 | Information Disclosure (misleading metadata) | PRETRAIN_MODEL_MAPS task-class strings | low | mitigate | test asserts Plant DNAGPT maps to Causal Language Model and that the misspelled class appears in no family entry — wrong class strings silently misroute family detection |
| T-261020-SC | Tampering | package installs | high | accept | no package-manager installs in this plan (pure data/test edits on an existing venv) — nothing for the package-legitimacy gate to gate |
</threat_model>

<verification>
- `.venv/bin/python -m pytest tests/models/test_model_info_registry.py tests/models/test_plant_helixseek_registry.py -v` — all green, no network.
- Registry invariants after both tasks: pretrained == 49 entries; finetuned == 187 entries; base_model present on every finetuned entry except SUMMARY-listed underivable ones (expected none); every base_model resolves to a pretrained id or the two external PlantCAD2 pointers.
- Grep gates (baselines at HEAD → post-fix): "Casual" in dnallm/models/modeling_auto.py 2 → 0; "aDNA" in dnallm/models/modeling_auto.py 2 → 0.
- `.venv/bin/ruff format --check` and `.venv/bin/ruff check` on the new test file.
- No repo-wide lanes (owner directive 2026-10-09: targeted verify commands + lane test files only). No network fetches anywhere in verification.
</verification>

<success_criteria>
- KWT-01: 15 variant pretrained entries present with correct task_type and variant-mentioning names; pretrained count exactly 49.
- KWT-02: Plant DNAGPT task class corrected to Causal Language Model; both EVO task-class misspellings corrected; typo-guard test prevents recurrence.
- KWT-03: base_model backfilled for all derivable finetuned entries (planning-time expectation: all 164; any residue explicitly listed in the SUMMARY, none guessed); values resolve to registry entries or documented external pointers.
- KWT-04: both tRNA titles read "a DNA sequence".
- tests/models/test_model_info_registry.py exists, passes, and ships in the same change as the dnallm/ edits (owner pytest rule).
</success_criteria>

<output>
Create `/home/forrest/Github/DNALLM/.planning/quick/261020-kwt-registry-data-layer-fix-add-15-missing-p/261020-kwt-SUMMARY.md` when done — include the derivation split (prefix / repo-map / name-map counts) and the underivable-entries list (expected empty).
</output>
