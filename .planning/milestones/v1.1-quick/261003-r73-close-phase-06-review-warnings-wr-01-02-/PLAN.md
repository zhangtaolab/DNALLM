---
quick_id: 261003-r73
slug: close-phase-06-review-warnings-wr-01-02-03
description: >-
  Close the three routed-open Phase-06 code-review warnings: WR-01 (no CI leg installs
  the fla extra — nightly smokes false-green on the non-KDA fallback), WR-03 (slice_gff_rows
  strips only \n — CRLF silently corrupts column 9), WR-02 (smoke _load_with_fallback
  converts dnallm regressions into whitelisted green skips).
estimate: 50000
created: 2026-10-03
files_modified:
  - .github/workflows/ci.yml
  - tests/models/test_plant_helixseek_smoke.py
  - dnallm/utils/genomic_coords.py
  - tests/utils/test_genomic_coords.py
---

<objective>
Close warnings WR-01, WR-02, WR-03 from the Phase-06 code review
(.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW.md, findings verified
live 2026-10-03), all routed open in 06-REVIEW-DISPOSITION.md:

- **WR-01**: no CI leg installs the `fla` extra (every leg installs `.[base]` — ci.yml lines
  68, 157, 252-254, 319-320, 386, 470). The coverage-nightly leg runs the full suite
  including the two slow PlantHelixSeek checkpoint smokes WITHOUT fla, so those loads run the
  remote code's silent pure-PyTorch non-KDA fallback (probe-set p(CRE) 0.0073 in-DHS vs
  0.0087 non-DHS — positionally dead) while the shape-only asserts stay green — a false-green
  gate over ~1.9 GB-per-model downloads.
- **WR-03**: `slice_gff_rows` rstrips only `"\n"` (genomic_coords.py:227,234). CRLF-terminated
  GFF input leaves `\r` at the end of column 9 and anywhere lines are split, silently
  corrupting attribute values — exactly the silent-corruption failure mode the module
  docstring promises is structurally impossible.
- **WR-02**: `_load_with_fallback` in tests/models/test_plant_helixseek_smoke.py:65-75 catches
  bare `Exception` and `pytest.skip`s with the whitelisted `environment-unavailable:` prefix —
  a regression inside dnallm's own `load_model_and_tokenizer` would present as a green
  environment skip.

Purpose: make the nightly gate exercise the real KDA kernel path (or skip loudly and typed
when fla is absent), make CRLF GFF input either parse identically to LF or raise loudly, and
make dnallm-side load regressions FAIL instead of silently skipping.

Output: two nightly CI legs installing `.[base,fla]`, a typed importorskip guard on the two
slow smokes, a CRLF-robust `slice_gff_rows` with new unit tests, and an exception classifier
in the smoke module with regression tests — each task committed atomically.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW.md
@.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md
@.github/workflows/ci.yml
@tests/models/test_plant_helixseek_smoke.py
@tests/models/test_plant_helixseek_fla_kernels.py
@tests/expected_skips.yaml
@dnallm/utils/genomic_coords.py
@tests/utils/test_genomic_coords.py
@dnallm/models/model.py
</context>

<constraints>
- Branch `phs` only. Commit and push; never add attribution trailers (owner rule).
- Every dnallm/ code change ships with pytest coverage in the same change (Task B tests;
  Task C is itself test code and carries its own regression tests).
- Never add a conftest.py under tests/examples/.
- No refactors beyond the three fixes; no new dependencies; no changes to fast CI legs
  beyond what WR-01 names.
- All commands assume cwd at repo root with the project venv active
  (.venv/bin/pytest resolves first on PATH).
- Do NOT run tests/models/test_plant_helixseek_smoke.py without `-m "not slow"` during
  verification — the two slow tests download ~1.9 GB checkpoints each.
</constraints>

<tasks>

<task type="auto">
  <name>Task A (WR-01): install fla on the nightly CI legs + typed importorskip guard in the slow smokes</name>
  <files>.github/workflows/ci.yml, tests/models/test_plant_helixseek_smoke.py</files>
  <action>
Two changes, both verified against the live workflow file (step names are stable; line
numbers current as of 2026-10-03):

1. **ci.yml — coverage-nightly job** (job `coverage-nightly (py3.12, full suite incl. slow)`,
   step "Create virtual environment and install dependencies", currently line 470): change
   `uv pip install -e ".[base]"` to `uv pip install -e ".[base,fla]"`. This is the only leg
   whose pytest invocation (line 486, no `-m` filter) runs the two slow PlantHelixSeek smoke
   tests — with fla installed they finally load through the real KDA kernels instead of the
   silent fallback. Keep `UV_HTTP_TIMEOUT`/`UV_CONCURRENT_DOWNLOADS` env and the numpy pin
   step untouched.

2. **ci.yml — mamba nightly job** (job on `[self-hosted, dnallm-nightly]` with the
   "Create virtual environment and install mamba dependencies" step, currently line 319):
   change its `uv pip install -e ".[base]"` to `uv pip install -e ".[base,fla]"`. This is the
   review's "ideally the mamba nightly leg" and the disposition's "nightly legs" (plural):
   that leg runs `-m "not slow"`, so the fast guard test
   `test_chunk_kda_importable_when_fla_installed` (tests/models/test_plant_helixseek_fla_kernels.py:58)
   converts from a typed skip into an actually-exercised fla import check on that box. Also
   minimally update the step's explanatory comment (currently ends "Same extras set the other
   legs install; coverage-nightly proves .[base] resolves green on this exact box.") to note
   the fla addition and that per pyproject.toml:157 the fla spec rides the already-installed
   torch/triton (no backend extras). Leave the `uv pip install -e ".[mamba]" --no-cache-dir
   --no-build-isolation` line untouched.

   Do NOT touch the fast legs (lines 68, 157, 252-254, 386): they run `-m "not slow"`, never
   load the checkpoints, and keep lean PR-time installs; their typed fla skips are
   whitelisted (proven live — the fla_kernels guard already skips green on those legs today).

3. **Typed skip guard in the smokes** — at the top of BOTH slow tests
   (`test_planthelixseek_cre_smoke_load`, `test_planthelixseek_anno_smoke_load`), before
   `_emit_env()`, add a `pytest.importorskip` guard for the module `"fla"` whose `reason=`
   starts with the exact prefix `environment-unavailable: ` and states that without
   flash-linear-attention the PlantHelixSeek load would run the degraded non-KDA fallback.
   Match the established typed-skip shape in
   tests/models/test_plant_helixseek_fla_kernels.py:58-63 (importorskip with a
   multi-line reason literal). The prefix is what keeps the junit skip message whitelisted by
   the `prefix: "environment-unavailable:"` entry in tests/expected_skips.yaml, so the
   scripts/audit_skips.py gate stays green on legs without fla while the smokes stop
   validating the wrong kernel path there. Do not alter the test bodies otherwise.
  </action>
  <verify>
    <automated>python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci.yml YAML OK')" && test "$(grep -Fc '[base,fla]' .github/workflows/ci.yml)" = "2" && test "$(grep -Fc 'uv pip install -e ".[base]"' .github/workflows/ci.yml)" = "3" && test "$(grep -Fc 'pytest.importorskip(' tests/models/test_plant_helixseek_smoke.py)" = "2" && test "$(grep -Fc 'reason="environment-unavailable:' tests/models/test_plant_helixseek_smoke.py)" = "2" && pytest tests/models/test_plant_helixseek_smoke.py --collect-only -q && ruff format --check tests/models/test_plant_helixseek_smoke.py && ruff check tests/models/test_plant_helixseek_smoke.py</automated>
    <fails_when>ci.yml is not valid YAML; `[base,fla]` appears on fewer/more than exactly the two nightly legs (count != 2); any fast leg was switched (exact `.[base]` install count != 3 for lines 68/157/386); either slow test lacks the importorskip guard; the guard reason does not carry the `environment-unavailable:` prefix; the smoke module fails to collect; ruff fails.</fails_when>
  </verify>
  <acceptance_criteria>
    - coverage-nightly installs `.[base,fla]`; its full-suite run (slow included) exercises the real KDA kernels on both PlantHelixSeek loads.
    - mamba nightly installs `.[base,fla]` before `.[mamba]`; its not-slow census runs the fla chunk_kda import guard instead of skipping it.
    - Fast legs (ubuntu matrix, windows, cuda, coverage-gate) unchanged.
    - Both slow smokes skip typed (`environment-unavailable:` prefix) when fla is absent instead of running the fallback path; skip messages stay whitelisted by tests/expected_skips.yaml.
  </acceptance_criteria>
  <done>Both nightly legs install the fla extra, the two slow smokes carry the typed importorskip guard, all automated checks above pass, and the change is committed atomically.</done>
</task>

<task type="auto">
  <name>Task B (WR-03): CRLF-robust slice_gff_rows — strip line terminators, raise on embedded CR</name>
  <files>dnallm/utils/genomic_coords.py, tests/utils/test_genomic_coords.py</files>
  <action>
Fix `slice_gff_rows` (dnallm/utils/genomic_coords.py:224-237) per the review fix-hint plus
the loud-error arm the module contract demands. Design (decided at planning time, consistent
with the module's strict-parsing / loud-ValueError / order-preserving contract):

- **Trailing terminators are unambiguous — strip them.** Windows-authored GFFs are common
  and their line endings carry no information, so the function accepts `\n`, `\r\n`, and
  bare `\r` (old-Mac) terminators: compute the stripped row once per parsed row with
  `row.rstrip("\r\n")`, use it for BOTH the column split (replacing `row.rstrip("\n")` at
  line 227) and the `matched.append(...)` (replacing `row.rstrip("\n")` at line 234).
- **An embedded CR is ambiguous — raise.** After stripping terminators, if `"\r"` remains
  anywhere in the stripped row, raise `ValueError` with a matchable message (e.g.
  "Carriage return embedded in GFF3 row (not a line terminator): {row!r}") — a mid-row `\r`
  is not a terminator and would silently corrupt whichever column contains it. Raise for
  every parsed (non-comment, non-blank) row regardless of chromosome match, mirroring how
  the existing `len(columns) < 5` malformed-row check already raises for any parsed row.
- Everything else stays untouched: comment/blank/non-string skip logic, the fewer-than-5-
  columns check, chrom normalization + exact comparison, closed-interval overlap test,
  `require_nonempty` guard, input-order preservation. `parse_gff_attributes` needs no
  change (its `segment.strip()`/`item.strip()` already tolerate stray whitespace) — do not
  touch it; no refactor beyond the fix.
- Update the `slice_gff_rows` docstring: document in `Args`/`Raises` that line terminators
  (`\n`, `\r\n`, `\r`) are stripped from returned rows and that an embedded `\r` raises
  `ValueError` (silent CR contamination is now structurally impossible, per the module
  docstring's promise).

Add unit tests in the "GFF3 row slicing" section of tests/utils/test_genomic_coords.py
(following the file's existing comment style and GFF_ROWS fixture):

1. `test_slice_gff_rows_crlf_and_cr_terminators_match_lf_results` — build terminator twins
   of GFF_ROWS (append `"\n"`, `"\r\n"`, and `"\r"` respectively to each non-empty string
   row, leaving the None/blank entries as-is); assert `slice_gff_rows` over each twin equals
   the LF result and the expected `[GFF_ROWS[3], GFF_ROWS[4], GFF_ROWS[5], GFF_ROWS[6]]`;
   assert no returned row contains `"\r"`; assert `parse_gff_attributes` over the tab-split
   column 9 of a returned CRLF row yields clean, CR-free values.
2. `test_slice_gff_rows_rejects_embedded_carriage_return` — a row carrying a `\r` that is
   NOT a trailing terminator (e.g. a 9-column gene row with `"\rNote=..."` appended before
   the final newline) raises `pytest.raises(ValueError, match=...)` matching the new message.

The committed showcase data is LF and the existing LF tests must stay green UNCHANGED — do
not modify any existing test.
  </action>
  <verify>
    <automated>pytest tests/utils/test_genomic_coords.py -q && ! grep -qF 'rstrip("\n")' dnallm/utils/genomic_coords.py && grep -qF 'rstrip("\r\n")' dnallm/utils/genomic_coords.py && grep -q 'crlf_and_cr_terminators_match_lf_results\|crlf' tests/utils/test_genomic_coords.py && grep -q 'embedded_carriage_return' tests/utils/test_genomic_coords.py && ruff format --check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py && ruff check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py</automated>
    <fails_when>any test in tests/utils/test_genomic_coords.py fails (existing LF tests included — they must pass unchanged); the module still contains a bare `rstrip("\n")`; the `rstrip("\r\n")` terminator strip, the CRLF-twin test, or the embedded-CR pytest.raises test is missing; ruff fails.</fails_when>
  </verify>
  <acceptance_criteria>
    - A CRLF GFF3 fixture slices to results identical to its LF twin; no returned row carries `\r`; column-9 attribute values parse clean.
    - Bare-CR (old-Mac) terminators behave identically to LF.
    - A mid-row embedded `\r` raises ValueError with a matchable message (loud-error path has its pytest.raises test).
    - All existing tests in tests/utils/test_genomic_coords.py pass unchanged; module contract (order-preserving, ValueError on malformed input) intact.
  </acceptance_criteria>
  <done>slice_gff_rows strips \n/\r\n/\r terminators, raises loudly on embedded CR, ships with the two new tests in the same commit (owner rule), and all checks above pass.</done>
</task>

<task type="auto">
  <name>Task C (WR-02): classify smoke-test load failures — environment skips typed, dnallm regressions fail</name>
  <files>tests/models/test_plant_helixseek_smoke.py</files>
  <action>
Replace the catch-all in `_load_with_fallback` (tests/models/test_plant_helixseek_smoke.py:65-75)
with an exception classifier. The classifier design is derived from the actual exception
ladder in dnallm/models/model.py (verified live — read `download_model` lines 317-377 and
`load_model_and_tokenizer` lines 736-901 before editing):

- `download_model` swallows EVERY downloader exception inside its retry loop and terminates
  with a bare, UNCHAINED `ValueError(f"Model {model_name} download failed.")` (model.py:375)
  — one terminal signal covering network failures, hub outages, AND missing repos on both
  the ModelScope and HuggingFace sources. This ValueError is raised before the boundary
  wrap (the `_get_model_path_and_imports` call at model.py:834 sits outside the try at 844),
  so it arrives at the caller unchained: classify by message shape — a `ValueError` whose
  str starts with `"Model "` and ends with `" download failed."` is environmental.
- `load_model_and_tokenizer` wraps everything in its load block as
  `raise ValueError(f"Failed to load model: {e}") from e` (model.py:887-888), so
  hub/network causes are visible only in the `__cause__`/`__context__` chain: walk the chain
  (guard against cycles with a seen-id set) and classify as environmental if any node is a
  `ConnectionError`, `TimeoutError`, or `OSError` (requests' `RequestException` subclasses
  `IOError`/`OSError`; huggingface_hub HTTP errors and socket errors land there).
- `ImportError` (anywhere: the explicit modelscope/transformers guards at model.py:444-448
  and 476-480, or remote code importing an absent optional dep such as fla) is
  environmental.
- Everything else — `TypeError`/`AttributeError`/`KeyError` from dnallm's dispatch or config
  plumbing, a `ValueError("Failed to load model: ...")` chained from a non-network cause,
  CUDA-OOM `RuntimeError` — is NOT environmental and must propagate so the test fails with
  the real traceback.

Implementation shape: add a module-level `_is_environment_error(exc)` helper implementing
the classification above (Google-style docstring naming the rules and the model.py anchor
lines), then rewrite `_load_with_fallback`'s except arm to
`except Exception as exc:` → `if not _is_environment_error(exc): raise` → otherwise append
the existing `{source}: {type).__name__}: {exc}` evidence and continue to the next source.
The both-routes-exhausted `pytest.skip` message stays EXACTLY as-is (it carries the
whitelisted `environment-unavailable:` prefix, versions, and error evidence — the unchanged
skip contract). Update the module docstring's fallback-contract paragraph (and the helper's
docstring) to state the new rule: only environment-class failures skip; dnallm-side failures
propagate and fail the test.

Add fast regression tests (no network, no slow mark, no model downloads) as a new class
`TestLoadWithFallbackClassification` in the same file:

- Unit tests of `_is_environment_error`: `ConnectionError`/`TimeoutError`/`OSError`/
  `ImportError` are True; the exact terminal `ValueError(f"Model {CRE_REPO_ID} download
  failed.")` is True; a `ValueError("Failed to load model: ...")` raised `from` a
  `ConnectionError` is True (chain walk); a bare `TypeError` is False; a
  `ValueError("Failed to load model: ...")` raised `from` a `TypeError` is False.
- Behavior tests through `_load_with_fallback` with
  `monkeypatch.setattr("tests.models.test_plant_helixseek_smoke.load_model_and_tokenizer",
  fake)` and a minimal `TaskConfig` (construct it the way the smoke tests do, e.g.
  task_type="binary", num_labels=2 — add `threshold` only if Pydantic validation demands
  it): (1) a fake loader mimicking a dnallm regression (raises the boundary-shaped
  `ValueError("Failed to load model: ...")` from a `TypeError`) must PROPAGATE — assert
  `pytest.raises(ValueError, match="Failed to load model")` and that no skip occurs;
  (2) a fake loader raising environmental failures on both routes (boundary-shaped
  `ValueError` from `ConnectionError`) must skip — assert
  `pytest.raises(pytest.skip.Exception, match=r"environment-unavailable: PlantHelixSeek")`.

Do not touch the two slow test bodies (beyond Task A's guard), the typed skip message, or
any other file.
  </action>
  <verify>
    <automated>pytest tests/models/test_plant_helixseek_smoke.py -m "not slow" -q && grep -q '_is_environment_error' tests/models/test_plant_helixseek_smoke.py && grep -q 'TestLoadWithFallbackClassification' tests/models/test_plant_helixseek_smoke.py && grep -q 'pytest.skip.Exception' tests/models/test_plant_helixseek_smoke.py && ruff format --check tests/models/test_plant_helixseek_smoke.py && ruff check tests/models/test_plant_helixseek_smoke.py</automated>
    <fails_when>any not-slow test in the smoke module fails (the new classifier tests must pass); the classifier helper or its tests are absent; the file still catches-and-skips on bare Exception without classification; ruff fails. Running the file WITHOUT -m "not slow" is not part of verification (it would download ~1.9 GB checkpoints).</fails_when>
  </verify>
  <acceptance_criteria>
    - A dnallm-side load regression (non-network exception, wrapped or bare) FAILS the smoke test instead of producing a whitelisted green skip.
    - Genuine network/hub/missing-repo/environment failures on both routes still skip with the unchanged typed `environment-unavailable:` prefix carrying version + exception evidence.
    - The terminal `ValueError("Model ... download failed.")` from download_model is classified environmental (it is the sole signal for downloader failures on both sources).
    - New classifier tests are fast, not slow-marked, and run on every fast CI leg.
  </acceptance_criteria>
  <done>_load_with_fallback classifies exceptions via _is_environment_error, dnallm regressions propagate, environment failures keep the typed skip, the regression tests pass, and the change is committed atomically.</done>
</task>

</tasks>

<verification>
After all three tasks, from repo root with the project venv active:

1. `pytest tests/utils/test_genomic_coords.py tests/models/test_plant_helixseek_smoke.py tests/models/test_plant_helixseek_fla_kernels.py tests/models/test_plant_helixseek_registry.py -m "not slow" -q`
   — all pass; the existing 26 Phase-06 fast tests stay green unchanged (committed showcase
   data is LF; nothing in this task touches it).
2. `ruff format --check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py tests/models/test_plant_helixseek_smoke.py && ruff check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py tests/models/test_plant_helixseek_smoke.py`
3. `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('OK')"`
4. Skip-audit contract: the only new skip messages are the two importorskip reasons, which
   start with the already-whitelisted `environment-unavailable:` prefix
   (tests/expected_skips.yaml) — no whitelist edit needed; do not add entries.
5. `git log --oneline -4` shows three atomic fix commits scoped `quick-261003-r73` (plus the
   plan docs commit) and `git status` is clean; push branch `phs`.
</verification>

<success_criteria>
- WR-01 closed: both nightly legs install `.[base,fla]`; the coverage-nightly full-suite run
  exercises the real KDA kernels; smokes skip typed when fla is absent (whitelist intact).
- WR-03 closed: CRLF/CR-terminated GFF input slices identically to its LF twin; embedded CR
  raises ValueError; existing LF tests green unchanged; tests ship in the same commit as the
  dnallm/ change.
- WR-02 closed: environment-class failures skip typed; dnallm-side regressions fail the
  test; classifier decision covered by fast regression tests.
- 06-REVIEW-DISPOSITION.md rows WR-01/02/03 can be flipped to `fixed` referencing these
  commits.
</success_criteria>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| user-supplied GFF/GFF3 files -> slice_gff_rows | untrusted text input parsed by dnallm code |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-r73-01 | Tampering | CI dependency install (nightly legs now install fla) | low | accept | fla is not a new dependency — already declared in pyproject.toml with bounded range `flash-linear-attention>=0.5.2,<0.6` and audited in Phase 06; installs ride the existing uv resolution and cached torch/triton |
| T-r73-02 | Tampering | slice_gff_rows CR handling | low | mitigate | embedded `\r` now raises ValueError instead of silently corrupting column-9 values — strictly narrows the silent-parsing surface; no new execution/eval surface introduced |
</threat_model>

<output>
Create `261003-r73-SUMMARY.md` beside this PLAN.md when done. Commit sequence (branch `phs`,
no attribution trailers): `docs(quick-261003-r73): plan WR-01/02/03 closure` (this PLAN.md),
then one atomic commit per task in order A, B, C with scope `quick-261003-r73`, then push.
</output>
