# Deferred Items — Phase 12

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
