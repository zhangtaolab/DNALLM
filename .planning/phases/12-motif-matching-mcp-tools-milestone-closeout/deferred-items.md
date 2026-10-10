# Deferred Items — Phase 12

## [Owner deferral — MOTIF-01, 2026-10-10] HBG1/BCL11A golden fixture awaits manuscript inputs

- **Found during:** Phase 12 verification — the single owner-input-gated truth
  (12-VERIFICATION.md Truth 2, `behavior_unverified_items[0]`).
- **Issue:** the paper-exact coordinate assertion needs four inputs that exist
  only in the owner's unrevised manuscript (not publicly indexed): (1) Fig 4a
  hotspot window coordinates verbatim, (2) motif ID — MA2324.1 vs MA2504.1 vs
  a CIS-BP PWM, (3) JASPAR release — 2024 vs 2026, (4) tolerance policy —
  0 bp exact vs ±2 bp figure-derived.
- **Owner decision (option B, 2026-10-10):** defer rather than block Phase 12;
  recorded as a frontmatter override in 12-VERIFICATION.md (same mechanism as
  the Phase 11 VEP-magnitude override, accepted_by owner Tao Zhang).
- **Why safe to defer:** the harness is committed, generic, and green on the
  synthetic stand-in (94/94 fast-lane incl. all 3 `TestGoldenHBG1BCL11A`
  tests). Activation is fixture-files-only: swap the two stand-in files in
  `tests/interpret/fixtures/hbg1_bcl11a/`, fill `expected_hits`/`tolerance_bp`,
  flip `pending: false` in `manifest.yaml`, run
  `uv run --no-sync pytest tests/interpret -q -k golden`. Zero code changes;
  no milestone reopen.
- **Tracking:** GitHub issue "Golden fixture: HBG1/BCL11A Fig 4a owner inputs
  (MOTIF-01 deferred activation)" filed 2026-10-10 against `revision`.

## [Out-of-scope discovery — 12-03 Task 1] mkdocs --strict fails on 15 pre-existing warnings

- **Found during:** 12-03 Task 1 verify (`uv run --no-sync mkdocs build --strict`)
- **Issue:** The strict build aborts with 15 warnings, all pre-existing and
  unrelated to the IA³ chapter: 14 broken relative links inside the
  `docs/example/marimo/*` mirrors (marimo-generated pages reference paths like
  `example/notebooks/benchmark.md` relative to their own directory, which do
  not resolve in the docs tree) plus one
  `user_guide/benchmark/advanced_techniques.md` link to `../../../CONTRIBUTING.md`
  (target outside `docs/`).
- **Proof of pre-existence:** rebuilt with the HEAD version of
  `peft_adapters.md` restored — identical 15 warnings, identical count. The
  12-03 change adds zero warnings; the page builds and renders.
- **Why deferred:** outside the 12-03 lane (marimo mirrors + benchmark page);
  the repo's docs-validation CI gate does not run `mkdocs build --strict`
  (it runs check_docs_sync / validate_docs_snippets / validate_yaml / example
  tests — all green), so this does not red CI today.
- **Suggested owner action:** either exclude `docs/example/marimo` from the
  mkdocs nav/strict validation, regenerate the marimo mirrors with absolute
  doc paths, or accept and switch the strict gate on after fixing the
  CONTRIBUTING link. Recorded here per the executor scope-boundary rule.

## [Code review — Phase 12 IN-05] numpy 2.5.x cannot be instrumented by coverage on Python 3.13

- **Found during:** Phase 12 code review (12-REVIEW.md, IN-05).
- **Issue:** the `numpy>=1.26.0` floor in `pyproject.toml` is unbounded, and
  a fresh non-matrix resolve pulls numpy 2.5.x; coverage 7.16.2 on Python
  3.13 then fails EVERY `pytest --cov` invocation at conftest import with
  `ImportError: cannot load module more than once per process` (numpy's
  double-init guard), under all `COVERAGE_CORE` tracer settings. Reproduced
  with coverage + numpy alone — no dnallm code involved. This also blocked
  independent re-measurement of the phase's per-module coverage claims
  during review (see the 12-REVIEW.md Summary honesty note).
- **Why deferred:** environment observation, not a correctness defect in
  this phase's diff. CI is protected by the explicit numpy matrix pins
  (1.26.4 / 2.2.0 in `.github/workflows/ci.yml`); only local/ungated
  resolves are exposed, and a dependency ceiling (`numpy<2.6` or similar) is
  a dependency-policy call for the owner, not a review fix.
- **Suggested owner action:** when the matrix retires the 1.26.4 leg, add an
  explicit numpy ceiling co-located with the pyarrow cap comment in
  `pyproject.toml`, or pin the dev venv; at minimum record the
  incompatibility in a comment next to the numpy floor.

