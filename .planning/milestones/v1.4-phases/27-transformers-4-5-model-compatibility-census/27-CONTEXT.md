# Phase 27: Transformers 4/5 Model Compatibility Census - Context

**Gathered:** 2026-10-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a per-model transformers compatibility matrix over the **`pretrained:` section of `dnallm/models/model_info.yaml` (49 entries)**, with empirical import-level evidence gathered under the installed transformers 5.17.0, disposition of every failure into `dnallm/utils/transformers_compat.py` shim fixes (shim + test) or the existing typed-skip + traceback path, the census results written back into `model_info.yaml` as new registry fields, a plain-language docs table generated from the registry, a CI consistency gate over the committed data, and a dispatch-only re-verification lane on the self-hosted runner.

**Count correction:** ROADMAP.md says "238 models"; the yaml actually carries **49 pretrained + 187 finetuned = 236**. Scope is the 49 pretrained entries. Research/planning should correct the ROADMAP count via the proper roadmap update path.

</domain>

<decisions>
## Implementation Decisions

### Census Scope
- **D-01: Pretrained-only census.** Only the 49 `pretrained:` entries are censused. The 187 `finetuned:` entries are excluded — their compatibility is inherited from their base models. (Owner clarification at discussion open.)

### Evidence Depth (layered)
- **D-02: Layered evidence — full import census + sampled loads.** All 49 entries receive import-level evidence (remote-code files fetched, import chains resolved under transformers 5.17.0). Real loads (config + tokenizer + weights) are performed for exactly two classes: (a) models that fail at import level — to verify shim fixes actually restore the load; (b) one representative per model family — to prove that passing imports also survives runtime.
- **D-03: Load-level pass bar = load + one short-sequence real forward.** Model construction and weight loading alone do not count; one short real forward must run. Rationale: historically real breakage surfaced inside forward (`get_extended_attention_mask`, `head_mask` class — v1.2 evidence), not at load.
- **D-04: Giants (evo-1, evo-2) excluded from load sampling.** Import-level evidence only; the matrix labels them honestly (import-verified only). They load through `dnallm/models/special/evo.py` (stripedhyena path), not the trust_remote_code surface.

### Version Anchors (single environment)
- **D-05: Empirical census only under transformers 5.17.0; no separate 4.x environment.** (Owner: "就用 5.17 检测即可，如果不兼容 4, 自然会报错" — just detect under 5.17; 4-incompatibility will error naturally.) Reasoning: all 49 checkpoints are 4.x-era code; breakage only runs in one direction (transformers 5 removing 4.x APIs), so pass-under-5.17 implies 4-side compatibility. The matrix/docs must label this honestly: the 5 side reads "tested on transformers 5.17.0"; the 4 side reads "inferred (model code written in the transformers 4 era)" — never claim separate 4.x testing. Roadmap taxonomy (`both` / `4-only` / `5-only` / `remote-code-broken`) is retained; the `5-only` bucket is expected to stay empty. — **Reversibility:** reversible — a 4.x verification environment can be added later without undoing anything.

### Fix Strategy (census-first)
- **D-06: Census evidence first, per-failure solutions after.** No pre-committed alias-vs-vendor priority. Run the census, capture each failure's exact shape (which import, what cause), then choose the fix per failure. (Owner: "先查吧，查出来问题再看解决方案".) Owner expectation: the dominant class is class/function renames (e.g. `PreTrained*`/`Pretrained*` casing) — cheap to fix. Fix takes priority over skip (2026-10-02 owner precedent: "fix all non-gated census failures now"); only structurally unfixable failures go to the typed-skip + traceback path (locked in ROADMAP).

### Matrix Home (registry fields + generated table + gate)
- **D-07: Census results become new fields on the `pretrained:` entries in `model_info.yaml`** — compatibility status + evidence + a transformers version stamp, following the REG-02 enrichment pattern (`tokenizer_type`/`max_length` precedents). `RegistryEntry` in `dnallm/models/registry.py` and the REG-03 integrity tests (`tests/models/test_registry.py`, `tests/models/test_model_info_registry.py`) extend in the same change. The docs table is generated from the registry in plain language for biologists, and a CI consistency check keeps the generated table and the registry from drifting (Phase 26 `dnallm/mcp/consistency.py` precedent), running on the standard lanes. — **Reversibility:** one-way — REG-03's field-level insertion-only check treats registry fields as add-only; removing fields after the census ships would violate that invariant and break any consumer that read them. Field naming and schema details are planner's to propose.

### Recurrence (dispatch lane, fail-closed)
- **D-08: Census tool is committed and re-runnable via a dispatch-only lane on the self-hosted runner** (`.github/workflows/feasibility.yml` precedent). Re-run on transformers bumps (operationalizing the dependabot "bump must re-verify" note) or whenever needed. A re-run whose results disagree with the committed `model_info.yaml` fields turns the lane red — fail-closed, same pattern as `scripts/audit_skips.py`. No nightly residency. — **Reversibility:** reversible — a workflow file.

### Claude's Discretion
- Census tool implementation shape (script vs pytest-collected lane, artifact format for the recorded evidence)
- Exact registry field names/schema and the version-stamp representation
- Docs page placement and table layout (per-model rows per ROADMAP deliverable)
- Which entry represents each model family in load sampling
- Alias-to-5.x vs vendor-4.49.0 choice per individual failure, decided on census evidence per D-06
- How the ROADMAP "238" count gets corrected (via the proper roadmap update path during planning)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Census target & registry
- `dnallm/models/model_info.yaml` — the census target; `pretrained:` section (49 entries) is scope; existing enriched fields (`tokenizer_type`, `max_length`) show the field pattern to extend
- `dnallm/models/registry.py` — sole parser of the yaml; frozen typed `RegistryEntry` (`extra="forbid"`) must gain the compat fields; `reload()` hook exists
- `tests/models/test_registry.py`, `tests/models/test_model_info_registry.py` — REG-03 integrity tests (section counts 49/187 pinned, insertion-only field check) that must extend with the new fields

### Compat shim & its tests
- `dnallm/utils/transformers_compat.py` — the shim layer (11 rungs); the rung pattern is: probe-gated, absence-gated per name, idempotent sentinels, no-op on transformers 4.x; vendored implementations cite upstream tag v4.49.0
- `tests/utils/test_transformers_compat.py`, `tests/utils/test_transformers_compat_tie_weights.py`, `tests/utils/test_transformers_compat_device.py`, `tests/utils/test_transformers_compat_np.py` — the TDD-per-rung pattern every new rung must follow

### Typed-skip & fail-closed audit contract
- `tests/expected_skips.yaml` + `scripts/audit_skips.py` — typed-skip allowlist; audit fails CI on any unmatched skip (the fail-closed pattern D-08 reuses)
- `tests/examples/_execution.py` — typed skip helpers (`network_unavailable_skip` etc.) with evidence embedded

### CI & runner
- `.github/dependabot.yml` — transformers KNOWN RISK comment (ModelScope mamba remote code imports `MambaCache`, removed in 5.x; a bump must re-verify the mamba load path)
- `.github/workflows/feasibility.yml` — dispatch-only lane precedent on the self-hosted runner
- `models.lock` — pinned remote artifacts (`<hf|ms> <repo-id>@<revision>`) for reproducible census evidence

### Consistency-gate precedent
- `dnallm/mcp/consistency.py` — Phase 26 registry-keyed consistency check with tiered findings and a CI gate over committed configs; the structural model for D-07's gate

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `transformers_compat.py` rung infrastructure — `apply_patches()` ordering, probe helpers, idempotent sentinels; new fixes extend this file, not a new mechanism
- `dnallm/models/registry.py` loader — typed frozen entries; extension is additive (REG-02 proved the path)
- `models.lock` revision pinning — census evidence should record repo revisions the same way for reproducibility
- `expected_skips.yaml` + `scripts/audit_skips.py` — the fail-closed committed-claim-vs-reality audit that D-08's drift check mirrors
- Self-hosted runner (`dnallm-nightly`): huggingface.co unreachable (hf-mirror only), ModelScope works, hub caches warm, giants cached at `~/models-giants`

### Established Patterns
- Probe-gated, absence-gated, idempotent compat rungs that no-op on transformers 4.x (D-07 ladder)
- TDD-first per rung: failing test reproducing the exact import error before the fix lands
- Fail-closed audits over committed claims (skip audit, Phase 26 consistency gate)
- Honest labeling culture: inference is labeled inference (D-05), import-only verification is labeled import-only (D-04); plain-language user-facing prose for biologists throughout docs deliverables

### Integration Points
- `RegistryEntry` + `_build_registry` in `dnallm/models/registry.py` — where the new fields land
- `transformers_compat.py::apply_patches()` — where every new fix rung registers
- Docs generation for the compatibility table joins the existing MkDocs site (`mkdocs.yml`)
- The consistency gate joins the standard CI lanes via existing testpaths (Phase 26 placement pattern)
- The census dispatch lane joins `.github/workflows/` next to `feasibility.yml`

</code_context>

<specifics>
## Specific Ideas

- Owner's framing of the version question: "其实就用 5.17 检测即可，如果不兼容 4, 自然会报错" — single-environment census, 4-side failure surfaces naturally to users on 4.
- Owner's framing of fixes: "我记得就是类/函数的名字改了，修起来不是很麻烦的" — expectation that renames (incl. `PreTrained*`/`Pretrained*` casing) dominate and are cheap; census first, then per-problem solutions.
- Owner's added area at selection time: incompatible models may be fixed by substituting the transformers 5 package surface for the removed 4.x one via if/else or try/except — i.e., alias mapping is an expected fix shape, not just vendoring.
- Docs table language must stay plain for biologists (owner directive 2026-10-21); suggested honest labels: 5 side "在 transformers 5.17.0 上实测" / 4 side "推断兼容（模型代码写于 transformers 4 时代）".

</specifics>

<deferred>
## Deferred Ideas

- **Load-time compat notice** (noted by Claude during discussion, not owner-raised): once the registry carries per-model compat fields, `load_model_and_tokenizer` could consult them and warn/raise with the exact failing import when a user requests a 4-only model under transformers 5. New runtime capability — belongs in its own future phase, not this census.

</deferred>

---

*Phase: 27-Transformers 4/5 Model Compatibility Census*
*Context gathered: 2026-10-27*
