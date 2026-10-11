# Project Research Summary

**Project:** DNALLM (`dnallm` PyPI package) — milestone v1.3 "Platform Reach & Packaging Honesty"
**Domain:** Maintenance/hardening milestone on a mature, heavily-gated Python bio-ML package (frozen truth chains, census-gated nightly, byte-identical docs mirrors, >90% coverage gate, typed-skip allowlist)
**Researched:** 2026-10-11
**Confidence:** HIGH for repo-grounded facts (all four researchers verified against the working tree at branch `v1.3`); MEDIUM for external-tool claims (cross-checked against official docs, live index fetches, and live venv probes)

## Executive Summary

v1.3 is not a feature milestone — it is an honesty milestone on an existing, mature toolkit. The work: make the example tree execute natively on Windows (where the pybedtools/bedtools ecosystem officially offers nothing — its own docs say "Windows is not supported"), make the published extras do what their names claim, rust-proof a publish chain that has already failed twice consecutively (v1.2.1: uv implicit-system-env removal, `check_code.py --with-tests` flag removal), disposition ~59 ollama test refs, and sweep docs accuracy. Experts build the Windows half the bioframe way: pure-numpy interval operations (documented bedtools-equivalence) where bedtools cannot install, while keeping the bedtools-computed frozen truth chain (jaccard=0.324727, exon_f1=0.7522) untouched on Linux — dnallm already exceeds the ecosystem bar here with two VERDICT-PARITY spikes (261010-wx2, 261011-0a3) that pinned the exact semantic contracts a port must carry.

The recommended approach adds essentially **one dependency at most**: `triton-windows>=3.7.1.post27,<3.8` behind a `platform_system == 'Windows'` marker in the `fla` extra (and even that is an owner declare-vs-document decision — the research leans DECLARE since the fork moved to the triton-lang org with attested builds, and the exact version is empirically proven on the owner's RTX 5080). Everything else is configuration, deletion, CI workflow authoring, and one ported library module (`dnallm/utils/genomic_intervals.py`, the spike cores productionized). The numpy evaluators need zero additions. The `cuda121` extra retires on verified fact (index frozen at torch 2.5.1); `rocm` should be removed, not restored (no rocm wheel exists for torch 2.10/2.11; the lane caps two minors below default and was never execution-tested). The publish chain moves to `pypa/gh-action-pypi-publish` OIDC Trusted Publishing plus a never-uploading smoke workflow (`uv build --no-sources` → `twine check` → clean-venv install-from-dist), with `PYPI_API_TOKEN` deleted only after the first observed green OIDC release. GAP-1 (NT promoter checkpoint on transformers 5.x) is smaller than feared: a live probe shows every removed API resolves via the existing `transformers_compat.py` rung ladder except exactly one method (`invert_attention_mask`).

The key risks are all silent-failure classes, and each has a concrete same-change mitigation: (1) dispatch leakage — the numpy path accidentally running on Linux and silently re-deriving the frozen truth chain (mitigate: fail-loud dispatch, a dispatch-direction structure test, nightly Linux re-execution in the same change); (2) spike-to-port drift — dropping an oracle-pinned contract like the 6-column null-B fill whose `-1` score slot makes the intergenic branch fire, a bug invisible in the metric (mitigate: port the spike corpora and edge cases as unit tests with baked oracle literals, one regression test per pin); (3) the smoke actually publishing — verified: `publish.yml` today triggers on `workflow_dispatch` with NO event guard on `twine upload`, so a naive dry-run dispatch is a real publish with immutable filenames (mitigate: guard or split the workflow before any scheduled trigger exists); (4) pip's warn-and-exit-0 on the `base` rename silently breaking downstream users and under-installing CI legs (mitigate: atomic pyproject+CI+guard-test+docs change, deprecated alias for one cycle); (5) the ollama sweep deleting non-ollama contracts that merely share the noun — kernel-plot compat guards, extras brackets, systemd pins (mitigate: per-assertion classification, post-sweep grep residue list).

## Key Findings

### Recommended Stack

From [STACK.md](STACK.md): the milestone needs almost nothing new. One dependency candidate (triton-windows, owner decision), one version cap (mypy), two deletions (cuda121, rocm), one rename (`base` → `dev-all`-class name), and workflow surgery. GPU matrix boundaries, numpy-vs-bedtools parity, and existing validated capabilities were deliberately NOT re-researched (owner directive).

**Core technologies:**
- **numpy (existing, `>=2.0` floor)** — jaccard + loj-intersect evaluators as pure library code; interval arithmetic is decade-stable core API; zero new deps. Do NOT add any interval library (pyranges/ruranges rejected with evidence: no win_amd64 wheels)
- **triton-windows `>=3.7.1.post27,<3.8; platform_system == 'Windows'`** — makes `dnallm[fla]` functional on Windows; fork now under triton-lang org, reproducible+attested builds since post27, exact version proven on sm_120; cap `<3.8` because 3.8 pairs with torch ≥2.14 (above the project's `<2.12` ceiling)
- **pypa/gh-action-pypi-publish `@release/v1`** — OIDC Trusted Publishing with native PEP 740 attestations (which `uv publish` can upload but not generate); job-level `id-token: write`, `environment: pypi`, NO username/password inputs; register on the EXISTING PyPI project
- **uv `build --no-sources` + twine `>=7,<8` (check only)** — uv docs recommend `--no-sources` to prove the pip-visible build; twine crossed a major unpinned in CI (6.x→7.0.0) — pin it, drop redundant `build`, never use twine as the upload path post-OIDC
- **mypy `>=1.15.0,<2.5`** — 2.4 is current latest and validated green; release cadence is weeks; 2.5 plausibly flips parser behavior
- **stdlib platform dispatch (`sys.platform == "win32"`)** — pick ONE idiom for example code; matches existing marker vocabulary

**Critical version facts (verified):** cu121 index frozen at torch 2.5.1 (dead lane under the `torch>=2.4,<2.12` floor); rocm6.3/6.4 both top at torch 2.9.1 Linux-only; plain-Linux `pip install torch` now serves cu130 (sm_75+); Windows PyPI torch wheels are CPU-only; `[tool.uv.sources]` is uv-only — pip never sees the index pins (the load-bearing uv-vs-pip lane asymmetry); transformers 5.17.0 probe against the actual NT remote code: all imports resolve natively or via existing rungs except `invert_attention_mask`.

### Expected Features

From [FEATURES.md](FEATURES.md): all four "users" (Windows-local researchers, pip/uv installers, PyPI consumers, maintainers) are served by the P1 set; P2 items ride capacity.

**Must have (table stakes, P1):**
- Platform dispatch + numpy evaluators + continuous parity tests on Linux CI + platform-aware gates + test-windows numpy executor + tolerance pins — acceptance is exon_f1=0.7522 in band on BOTH paths
- Extras hygiene: `base` rename/re-scope, cuda121 retirement, rocm restore-or-remove
- Install-lane docs + published GPU support matrix (pre-verified boundaries: Pascal cpu-only, 20xx→cu128, 30/40xx all extras, 50xx cu128+) + cuda130 default recommendation
- Publish smoke (workflow_dispatch dry-run chain, clean-venv install-from-wheel, never uploads)
- Ollama surface: per-item remove-vs-skip disposition (~59 refs), gates re-pinned
- GAP-1 (#11/#12) and #18 decided and executed — the only unacceptable state is leaving them owner-pending or silently broken

**Should have (differentiators, P2 if capacity):**
- OIDC Trusted Publishing migration (+ PEP 740 attestations, free with migration)
- Periodic (scheduled, not just dispatch) smoke cadence — rare in the ecosystem
- mkdocs `--strict` 15-warning cleanup; mypy cap; #20 cov-crash fix
- Cross-installer honesty matrix ("command × installer → what you actually get")

**Defer (v2+ / post-v1.3):**
- Runtime lane/architecture warning at import (torch 2.8-style)
- Staged publishing / TestPyPI rehearsal with promotion semantics
- pyranges/ruranges re-evaluation (only if win_amd64 wheels ever ship — trigger already ledgered)

**Anti-features (explicitly rejected):** re-freezing the truth chain on numpy; platform-split logic inside `dnallm/` library code; any third interval engine; exact float equality across paths; global `PretrainedConfig` monkeypatch for GAP-1; pinning transformers `<5`; blanket skip-everything on ollama; making cudaXXX extras pip-routable via dependency tricks; docs reorganization inside the accuracy pass.

### Architecture Approach

From [ARCHITECTURE.md](ARCHITECTURE.md): five integration shapes, none of which restructures a layer — (1) new import-pure library module, (2) runtime dispatch at example-code acquisition seams, (3) platform-aware test-gate inversion, (4) CI/publish workflow surgery, (5) compat-shim extension of the existing absence-gated attach pattern.

**Major components:**
1. **`dnallm/utils/genomic_intervals.py` (NEW)** — pure-numpy `parse_bed`/`jaccard`/`loj_intersect` with oracle-pinned contracts; one module (both ops share the parser); must sit OUTSIDE the 7-entry coverage omit list and count toward the shared 90% floor; NOT in `dnallm/tasks/` or `inference/`
2. **`tests/utils/test_genomic_intervals.py` (NEW)** — spike corpora + committed `.bed` fixtures with baked oracle literals; fast, non-slow-marked → runs on every leg including test-windows; no bedtools binary required
3. **Acquisition-fork adapter** — the numpy path must yield row objects with `.chrom/.start/.end/.name/.fields` (namedtuple) so the `tokens_to_nerdata` consumer stays byte-identical on both platforms; one forked line per file, never a forked consumer
4. **Three gate sites inverted** — `test_script_execution.py:128`, `test_notebook_execution.py:1397`, `test_plant_helixseek_showcase.py:523` flip from probe-then-skip to dispatch-then-execute; Windows branch contract tests via monkeypatched `platform.system`
5. **`publish-smoke` job/workflow** — build → check → install-from-dist → import probe; optionally TestPyPI under a SEPARATE trusted-publisher registration (never register the smoke as a production publisher)
6. **`transformers_compat.py` GAP-1 extension** — vendored `invert_attention_mask` rung (upstream v4.49.0 shape), absence-gated, then extend rung-by-rung on actual tracebacks

**Key patterns:** acquisition-fork/consumption-uniform; gate inversion on Windows; single-sourced frozen truth (selection.md parsed on both paths, tolerance in the comparison layer ONLY — never a second truth source); never sort loj inputs; join-not-merge (duplicates multiply, consumer dedups); `n_intersections` excluded from the production contract (raw-row vs merged counts diverge by design: 51 vs 45 on the frozen pair).

### Critical Pitfalls

From [PITFALLS.md](PITFALLS.md) (12 total; the five that shape the roadmap):

1. **Dispatch leakage breaks the frozen truth chain** — an inverted platform check, a try/except fallback, or an always-test-numpy refactor silently re-derives bedtools-computed values on Linux; if values hold there is NO alarm. Avoid: fail-loud dispatch (no except-fallback), a same-change dispatch-direction structure test, nightly Linux re-execution holding parsed selection.md bands.
2. **Spike-to-port semantic drift** — the null-B fill (`.`/`-1`/`-1`/`.`/`-1`/`.` at 6-column shape), A-file-order emission with unsorted input accepted, duplicate multiplication, and bookended-adjacency merge are parity PRECONDITIONS that look like implementation choices; the null-fill bug is invisible in exon_f1. Avoid: port the spike corpora (seeds 20261010/20261011 + handcrafted edges) as unit tests with baked literals; one regression test per pin; loud-ValueError parse guard.
3. **The smoke actually publishes** — verified in `publish.yml`: `workflow_dispatch` trigger + unguarded `twine upload` + a verify step that pip-installs from PyPI (not `dist/`), which validates the OLD release on any non-publish run. PyPI filenames are immutable. Avoid: event-gate or split workflows BEFORE any scheduled trigger exists; verify from `dist/`; never `--skip-existing`.
4. **`base` rename breaks silently** — pip ≥23.3 warns and exits 0 on a nonexistent extra (deps silently uninstalled downstream); in-repo, `base` has a two-level blast radius (`all` → `base` → 4 members) across 5 CI call sites + ~10 docs pages. Avoid: atomic same-commit sweep (pyproject + `all` + CI + guard tests + docs), deprecated alias extra for one cycle, grep gate for stale `.[base]`.
5. **Ollama sweep deletes non-ollama contracts** — `test_extras_guard.py` refs guard the ipython<9 kernel-plot pin and mcp membership; runner-infra refs pin systemd units and docs-mirror content. Avoid: classify per-ASSERTION not per-file; post-sweep `grep ollama tests/` must equal a sanctioned residue list recorded in the plan.

Also load-bearing: census-triple/skip-allowlist collisions (every test-set change re-pins the 208/217 D-03 triple, updates `expected_skips.yaml`, and dispositions ci.yml stage 3 — else pytest exit 5 Reds the nightly); Windows skip-green (the ever-green-skip dishonesty class — test-windows must EXECUTE the numpy path, with the unit layer platform-neutral so Linux coverage legs keep the ≥96% per-module standard); docs sweep before packaging = stale on arrival; OIDC claim-matching (owner+repo+workflow FILENAME+environment, exactly — and reusable workflows are unsupported as publishers).

## Implications for Roadmap

Phase numbering continues from 13 (PROJECT.md); suggested phases below are 14–20. ARCHITECTURE.md (build order A–G) and PITFALLS.md (mapping O/G/W/P/S/D) independently converged on the same shape; differences are noted inline.

### Phase 14: Ollama pytest-surface removal (Phase O)
**Rationale:** The only feature that REMOVES tests — doing it first shrinks every later edit surface and re-pins the census triple exactly once instead of repeatedly.
**Delivers:** Per-assertion remove-vs-skip disposition of ~59 refs across 5 test files + 2 notebooks, with a recorded ledger; deleted tests removed outright (no zombie skips); kept-surface typed skips with existing prefixes; D-03 census triple + `expected_skips.yaml` re-pinned; ci.yml stage 3 + selectors dispositioned; `langchain-ollama` kept in mcp extra (notebooks stay in-repo per WINDOWS #13 directive).
**Addresses:** FEATURES §5 (ollama disposition, P1).
**Avoids:** Pitfalls 5 (non-ollama guard deletion), 6 (census collisions), 7 (mirror/nav orphans).

### Phase 15: GAP-1 transformers 5.x disposition (Phase G)
**Rationale:** Must precede the Windows dispatch phase because both edit `test_script_execution.py` (the `_NT_STRUCTURAL_MARKER` ladder vs the `_gate_script_pybedtools` fork) — sequencing avoids reworking the same test class twice. The probe table makes the shim small: one known-missing method (`invert_attention_mask`) plus whatever the load run surfaces.
**Delivers:** Vendored absence-gated `invert_attention_mask` rung (Option A, research-recommended) or documented removal of the NT third model + pkl leg (Option B); #11/#12 un-skipped with probe machinery intact; #15 closure rides along on shim success with per-item probe evidence.
**Addresses:** FEATURES §1 GAP-1 row (P1); STACK §F.
**Avoids:** Pitfall 11 (shim blast radius across 35 model families — version-keyed minimal shim + non-NT-family regression test; premature un-skip).

### Phase 16: Windows parity core — numpy interval evaluators as library code (Phase W, part 1)
**Rationale:** Pure addition, depends on nothing; landing it as its own phase keeps the library module + its full test corpus atomic (owner rule: lib changes carry same-change coverage, per-module ≥96%).
**Delivers:** `dnallm/utils/genomic_intervals.py` (parse_bed, jaccard, loj_intersect with the three pinned contracts: oracle null-fill, A-file-order emission accepting unsorted input, join-multiplicity; `n_intersections` excluded); `tests/utils/test_genomic_intervals.py` with spike corpora + committed oracle fixtures + edge cases (boundary touching, chrom isolation via exact-string keys, duplicates, empty B); re-exports; coverage-omit proof.
**Addresses:** FEATURES §1 evaluator rows; STACK §A.
**Avoids:** Pitfall 2 (spike-to-port drift — the corpora tests ARE the prevention) and the coverage half of Pitfall 8.

### Phase 17: Windows parity rollout — platform dispatch + test-windows executor (Phase W, part 2)
**Rationale:** Depends on Phase 16; the fan-out phase (example code + mirrors + gates + CI) deserves its own verification loop.
**Delivers:** Acquisition-fork dispatch in `generate_bpe_dataset.py:7,249`, NER notebook cells 0/2/8, CRE cell 11 (+ all byte-identical docs mirrors); the three gates platform-aware with Windows-branch contract tests; NEW fast frozen-value tests (selection.md-parsed bands + pinned tolerance, ~1e-6 abs or 6-dp rounding) running on test-windows; exon_f1 band/tolerance pinned (currently evidence-only); ci.yml test-windows step; measured hosted-leg time budget in the first PR (D-12 precedent).
**Addresses:** FEATURES §1 dispatch/gates/executor rows (P1 headline).
**Avoids:** Pitfalls 1 (dispatch leakage — structure test + nightly re-execution same-change), 6 (Windows-leg skip messages in audit_skips; new markers must be registered under `--strict-markers`), 8 (skip-green + budget blowout — two-layer test design).

### Phase 18: Publish chain robustness (Phase S)
**Rationale:** Position-flexible, but landing it BEFORE packaging lets the smoke dispatch validate each extras change (ARCHITECTURE's E→F ordering); its FIRST act — the upload guard/split — must precede any scheduled trigger existing (Pitfall 3 is a time bomb otherwise).
**Delivers:** `publish-smoke.yml` (or guarded split) with build → `twine check` → clean-venv install-from-dist → import probe, never uploading; fixed or deleted `pip install dnallm` verify step; mypy cap in dev extra; #20 disposition (local-venv numpy 2.2.x pin or documented coverage-CLI workaround — no upstream fix exists); OIDC migration per the owner's deferred-decision row (job-level `id-token: write`, `environment: pypi`, publisher registered for the FINAL workflow filename, token deleted only after the first observed green OIDC release).
**Addresses:** FEATURES §3 (P1 smoke + P2 OIDC/mypy/#20).
**Avoids:** Pitfalls 3 (real publish from a dispatch) and 10 (OIDC claim mismatch; `--trusted-publishing always` not `automatic` so a lingering token can't silently win; no reusable-workflow factoring).

### Phase 19: Packaging & install honesty (Phase P)
**Rationale:** Validated by Phase 18's smoke dispatches; docs must describe the final extras shape, so this precedes the docs sweep.
**Delivers:** Atomic extras unit — `base` rename (target + alias-or-break is an owner call), cuda121 retirement (extra + uv index + sources + 5 conflict rows + test-cuda matrix leg + docs in one change), rocm removal + manual-AMD-lane docs (research-recommended over restore), #18 triton-windows declaration (research leans declare, owner call), install-lane docs (uv one-liner vs pip torch-index-first with explicit `--index-url`), cuda130 default recommendation, GPU support matrix page with the pre-verified boundary table, extras guard-test brackets, fresh-venv resolve verification per changed extra.
**Addresses:** FEATURES §2 (P1 packaging set).
**Avoids:** Pitfalls 4 (silent rename breakage) and 12 (half-changed cuda121/rocm/triton-windows surfaces).

### Phase 20: Docs accuracy sweep (Phase D)
**Rationale:** LAST, always — the sweep's deliverable is accuracy, a function of the final code state; running it before Phases 17/19 invalidates every reviewed page (installation.md alone has 8+ `.[base]` occurrences, a `.[cuda121]` example, and 3 cuda124-default mentions already retargeted by owner directive).
**Delivers:** Per-file review of docs/ + example/ with same-pass fixes and disposition records; lane-labeled install commands repo-wide; mkdocs `--strict` 15-warning delta accounted; mirror regeneration on every example edit (`check_docs_sync.py` reads bytes); showcase wording reconciled with the platform split.
**Addresses:** FEATURES §4 (P1 sweep + P2 strict cleanup).
**Avoids:** Pitfalls 7 (mirror gate) and 9 (stale-on-arrival).

### Phase Ordering Rationale

- **Subtraction before addition:** ollama removal first shrinks the files every later phase edits and re-pins collection counts once (Pitfall 6's designed bump-point).
- **Shared-file sequencing:** GAP-1 before Windows dispatch because both modify `test_script_execution.py`; evaluator library before dispatch because the fork imports it.
- **Smoke before extras:** the smoke job exists to catch exactly the install-lane rot class the extras re-scope risks; PITFALLS' alternative order (P then S) is viable provided the upload guard lands as S's isolated first act — either way the guard precedes any cron.
- **Docs last:** hard ordering dependency on nearly every other phase (Pitfall 9); it is the milestone's honesty backstop absorbing all downstream drift.
- **One collection-changing edit per phase:** the D-03 census triple and skip allowlist are deliberately pin-happy; never interleave two count-moving changes.

### Research Flags

Phases likely needing deeper research during planning (`/gsd-plan-phase --research-phase <N>`):
- **Phase 17 (Windows rollout):** test-windows execution subset scope + measured time budget; tolerance-pin mechanics (full-precision vs 6-dp anchor); marker registration for any new selector under `--strict-markers`; exon_f1 band pinning (currently evidence-only).
- **Phase 18 (Publish chain):** resolve the `uv publish --dry-run` CONFLICT (STACK verified it absent from the uv packaging guide; PITFALLS/FEATURES cite uv PRs #15638/#21439 adding it) against the repo's installed uv 0.12.21; read the pypa/gh-action-pypi-publish README for exact input syntax (not fetched); owner decision on subsuming OIDC into the smoke.
- **Phase 19 (Packaging):** owner decisions at discuss/plan time — base rename target + alias-or-break; rocm restore-or-remove; triton-windows declare-vs-document. (These are owner calls, not web research; the fresh-venv resolve verification is mechanical.)

Phases with standard patterns (skip research-phase):
- **Phase 14 (Ollama):** fully inventoried repo surface; per-assertion classification IS the plan; execution mechanical.
- **Phase 15 (GAP-1):** the probe table is done (live, 2026-10-11); the rung-ladder precedent has shipped 10 times.
- **Phase 16 (Evaluators):** spike cores + oracle pins exist in-repo; port + corpus tests is established practice (`genomic_coords.py` precedent).
- **Phase 20 (Docs):** per-file sweep against final surfaces; regression gates already exist (byte-sync, docs-validation, example-nightly).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Headline claims verified against official docs (PyPI, uv, pytorch indexes) + live probes of the dev venv and the NT remote code; exact tool patches (gh-action v1.14.x) LOW-MEDIUM; one internal conflict on `uv publish --dry-run` |
| Features | HIGH (repo) / MEDIUM (ecosystem) | Internal facts read directly (pyproject, publish.yml, WINDOWS ledger, spike verdicts); ecosystem claims cross-verified ≥2 sources; staged-publishing GA single-source LOW |
| Architecture | HIGH | Every integration point verified against the working tree; only the two web-sourced publish topics MEDIUM |
| Pitfalls | HIGH (repo) / MEDIUM (external) | All workflow/gate/test claims verified in source at branch v1.3; pip/twine/PyPI/OIDC behavior cross-checked |

**Overall confidence:** HIGH for everything repo-grounded (the majority of roadmap-relevant decisions); MEDIUM for external-tool specifics, each already annotated per-claim in the source files.

### Gaps to Address

- **`uv publish --dry-run` existence — direct conflict between researchers:** STACK says no (uv packaging guide, fetched); PITFALLS/FEATURES say yes (uv PRs #15638/#21439). Verify against installed uv 0.12.21 at Phase 18 plan time before the smoke design depends on it.
- **pypa/gh-action-pypi-publish exact input syntax:** README not fetched; read at Phase 18 plan time (permissions/environment/attestations inputs).
- **GAP-1 load-run unknowns:** the probe eliminates import-time risk, but the actual checkpoint load on 5.17 may surface removed APIs beyond `invert_attention_mask`; extend rung-by-rung on real tracebacks (established D-07 ladder), and hold Option B (removal) ready if the ladder grows disproportionate.
- **exon_f1 has no band in selection.md** (evidence-only in the Anno notebook): pinning its Windows-path band/tolerance is new v1.3 work discovered by architecture research — scope it inside Phase 17 explicitly.
- **test-windows time budget unmeasured:** the numpy-path fast tests add seconds-scale cost, but the hosted 60-min cap is mostly install today; measure in the first PR (D-12 precedent).
- **PyPI API token sunset:** no announcement found — treat OIDC as recommended, not forced (supports the P2 placement).
- **Trusted Publishing default-branch workflow requirement:** underdocumented (LOW) — do not let the migration plan depend on it.
- **mypy capping practice:** MEDIUM from training, not separately researched; only matters if the cap choice is contested at Phase 18.

## Sources

### Primary (HIGH confidence — repo, verified 2026-10-11 at branch v1.3)
- `.planning/PROJECT.md` (v1.3 scope), `.planning/WINDOWS.md` ledger #11–#20 (esp. #19 change-surface inventory), `.planning/STATE.md` deferred rows (extras hygiene, publish rust, mypy, OIDC, GPU matrix)
- `pyproject.toml` (extras incl. self-referential base/all, uv indexes/sources/conflicts, coverage omit list, pytest markers), `.github/workflows/ci.yml` (test-windows leg, D-03 census stage 0.5, stage 3), `.github/workflows/publish.yml` (unguarded dispatch upload verified)
- Spike artifacts `.planning/quick/261010-wx2` (jaccard PARITY; 15872/48878/0.324727; n_intersections non-portability) and `261011-0a3` (loj PARITY; null-fill/order/sortedness pins)
- `dnallm/utils/genomic_coords.py` + `transformers_compat.py` (module precedents), `tests/examples/*` (gate sites), `tests/expected_skips.yaml`, `scripts/audit_skips.py`, `scripts/check_docs_sync.py`
- Live probes: transformers 5.17.0 vs the NT promoter `modeling_esm.py` remote artifact (fetched); dev venv (mypy 2.4.0, coverage 7.16.2, torch 2.11.0+cu130, numpy 2.5.3); direct download.pytorch.org index inventories (cu121 frozen at 2.5.1; rocm tops at 2.9.1)

### Secondary (MEDIUM confidence — official docs / cross-checked)
- docs.pypi.org (Trusted Publishing using/adding a publisher, attestations), docs.astral.sh/uv (packaging guide, `--no-sources`, sources-are-uv-only), packaging.python.org GH-Actions publishing guide
- daler.github.io/pybedtools ("Windows is not supported"), bioframe paper + bedtools-users equivalence guide, torch Get Started selector + dev-discuss lane-EOL notices, spaCy cuda-extra removal discussion + jax PEP 685 notes, pip nonexistent-extra behavior (pypa/pip#12376 + multiple), PostHog test-maintenance skill + USGS dataretrieval bounded-skip PR, triton-lang/triton-windows releases + PyPI metadata

### Tertiary (LOW confidence — single/indirect sources)
- Staged-publishing GA date (single secondary source); gh-action exact patch v1.14.2 (newreleases tracking); Trusted Publishing default-branch workflow requirement (underdocumented — flagged, not relied upon)

---
*Research completed: 2026-10-11*
*Ready for roadmap: yes*
