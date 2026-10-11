# Pitfalls Research

**Domain:** Adding platform-split genomic evaluators, packaging re-scopes, publish smokes, ollama-surface removals, and docs-accuracy sweeps to DNALLM — a mature, heavily-gated Python package (frozen truth chains, census-hard-gated nightly lane, byte-identical docs mirrors, >90% coverage hard gate, typed-skip allowlist discipline, shared self-hosted GPU runner).
**Researched:** 2026-10-11
**Confidence:** HIGH for repo-grounded facts (verified against source at branch `v1.3`); MEDIUM for external-tool behavior (pip/twine/uv/PyPI/triton-windows — cross-checked web sources, official docs where reachable); each pitfall notes its grounding.

**Grounding:** `pyproject.toml` (extras incl. self-referential `base`/`all`, uv sources/conflicts with commented rocm block, coverage `fail_under=90`, pytest `--strict-markers`, markers list), `.github/workflows/publish.yml` (whole file), `.github/workflows/ci.yml:130-210,440-600,845-935` (test-windows leg, coverage-nightly, example-nightly stage 0.5 census pin + stage 3), `scripts/audit_skips.py`, `scripts/check_docs_sync.py`, `tests/expected_skips.yaml`, `tests/test_extras_guard.py`, `tests/test_runner_infra_contracts.py` (ollama refs), `tests/examples/test_notebook_execution.py` (ollama/gate refs), `tests/examples/test_plant_helixseek_showcase.py:515-560` (bedtools fail-loud + `pytest.mark.slow`), `.planning/WINDOWS.md` (#11, #12, #13, #18, #19, #20), `.planning/STATE.md` (deferred rows 2026-10-11: extras hygiene, publish rust, mypy drift, OIDC, GPU matrix), spike summaries `261010-wx2` / `261011-0a3` (parity pins).

**Phase labels:** roadmap not yet built; phase numbering continues from 13. Labels below are topic-keyed — **Phase O** (ollama surface removal), **Phase W** (Windows platform parity), **Phase G** (GAP-1 transformers 5.x shim), **Phase P** (packaging & install honesty), **Phase S** (publish smoke + OIDC + mypy cap), **Phase D** (docs accuracy sweep). Ordering recommendation in the Pitfall-to-Phase Mapping.

---

## Critical Pitfalls

### Pitfall 1 (W): The Windows numpy path silently becomes the Linux path too — breaking the bedtools-computed frozen truth chain the split was designed to protect

**What goes wrong:**
The owner directive is explicit (WINDOWS #19): Linux/macOS KEEP bedtools/pybedtools — "the frozen truth chain stays bedtools-computed, no re-freeze." The failure mode is dispatch leakage: the platform branch added to `generate_bpe_dataset.py` and `data_generation_and_inference.ipynb` (cell 2/cell 8, one shared code path) accidentally executes the numpy evaluator on Linux — via an inverted `sys.platform` check, a fallback "try pybedtools, except use numpy" that swallows a real pybedtools/bedtools breakage, or a test lane that exercises only the numpy path repo-wide. The committed jaccard=0.3247 / exon_f1=0.7522 values are then re-derived by a different engine, and either (a) they drift out of band (loud, nightly red) or (b) they hold (silent — the "no re-freeze" invariant is violated with no alarm). Worse: the Linux re-execution of the edited notebook can shift values for non-algorithmic reasons — dtype changes (int32 vs int64 accumulators), pre-sorting that plain `-loj` never did (spike 0a3 pin 3: plain `-loj` accepts unsorted input; sorting changes nothing semantically but any *filtering* or *dedup* bolted on "while we're in here" does), or a DataFrame reconstruction that reorders rows before the BILOU decode.

**Why it happens:**
The two paths live in the same cells/script; the natural test refactor ("just always test the numpy path — it's parity-proven!") removes the only execution evidence that Linux still runs bedtools. Parity spikes make the numpy path *feel* safe enough to let leak.

**How to avoid:**
- Dispatch must be fail-loud, not try/except: `if sys.platform == "win32": numpy_path() else: pybedtools_path()` with no except-fallback. A missing bedtools on Linux must keep hitting the existing fail-loud gate (`test_plant_helixseek_showcase.py:523` asserts `shutil.which("bedtools")`), never fall through to numpy.
- Add a same-change structure test asserting the dispatch direction: on non-Windows, the example code's pybedtools import/call is reachable and the numpy branch is not (and vice versa) — static asserts on cell/script source, precedent: the genome-wide denylist structure test (Phase 7) and the qwen-swap contract tests that assert notebook *source* content.
- The Linux nightly lane must re-execute the edited notebook/script and hold the parsed `selection.md` bands (existing D-05/D-06 machinery) — the re-execution after the edit IS the proof the Linux chain survived the surgery.
- Keep `.scratch/select_loci.py` and `selection.md:35` (historical methodology prose) untouched — the change-surface inventory in WINDOWS #19 already scopes them out.

**Warning signs:**
- A diff where the pybedtools branch lost lines (not just gained a sibling); any `except ImportError` around the platform dispatch.
- Nightly example census green but the committed notebook outputs show a regenerated jaccard/exon_f1 from a numpy-formatted print (different formatting = different engine).
- `git diff` touching `example/notebooks/plant_helixseek_shared/.scratch/{pred,truth}.bed` — the frozen BEDs are read-only, never written or moved (wx2 `requires` block).

**Phase to address:** Phase W — the dispatch structure test and the Linux re-execution must land in the same change as the platform split, not a follow-up.

---

### Pitfall 2 (W): Spike-to-port semantic drift — the production evaluator drops one of the oracle-pinned contracts the PARITY verdict was conditioned on

**What goes wrong:**
Both spikes are "task artifacts — never library code" (their own frontmatter). The v1.3 port re-implements their algorithms in `dnallm/` (precedent: `dnallm/utils/genomic_coords.py`), and the re-implementation "cleans up" away a load-bearing detail. The pins that must survive verbatim into the port:
1. **Null-B fill (loj):** a no-hit A row against the 6-column NER annotation B carries `.` `-1` `-1` `.` `-1` `.` — the B-column-5 `-1` is what makes the consumer's `name == "-1"` intergenic branch fire (0a3 Pin 1). A port that guesses all-`.` still labels intergenic — via the accidental `zero_map` fallback — so the exon_f1 band can hold while the semantics are wrong. The null vector is content/shape-dependent at other column counts (count-5 score slot is `-1` iff int-parseable; ≥7 always `.`).
2. **Emission order (loj):** plain `-loj` emits in A-file order with per-A B hits in B-file order, accepting unsorted A and B (0a3 Pin 2/3). The consumer's ascending traversal works because `tokens_bed` is *near-sorted in practice*, not because anything sorts. A port that sorts "for safety" changes row order downstream; a port that assumes hits arrive coordinate-sorted breaks when B's file order differs from start order.
3. **Join, not merge (loj):** one output line per A×B overlap pair; duplicates multiply per copy (0a3 multiplicity pin). The consumer's `token_seen[gene]` dedup absorbs A-side duplication — the port must keep emitting duplicates, never dedup early.
4. **Merge semantics (jaccard):** coalesce while `next start <= current end` — overlaps AND bookended adjacency (wx2). Chromosomes group by exact string key (`Chr1` ≠ `chr1`).
5. **n_intersections is NOT portable:** bedtools counts raw A rows (51) while the numpy analogue counts merged-A intervals (45) on the frozen pair (wx2 port note 2). Any production contract must either port raw-row semantics or exclude the field — gating on it would false-fail.

**Why it happens:**
The pins live in spike SUMMARY prose in `.planning/quick/`, which the porting agent may never re-read; each detail looks like an implementation choice, not a parity precondition. The null-fill bug is *invisible in the metric* (exon_f1 unchanged) — precisely the class of drift the frozen chain cannot catch.

**How to avoid:**
- Port the spike corpora, not just the algorithm: re-run the seeded randomized property cases (seeds 20261010 / 20261011) plus the handcrafted edge cases as pytest unit tests against the production evaluator, with the oracle expectations *baked as literals* where bedtools is unavailable (Windows) and against a live bedtools where available (Linux nightly, via the existing rootless micromamba cache). The spikes already proved determinism (byte-identical re-runs).
- One explicit regression test per pin: null-vector exact-string assertion at the 6-column shape (and the count-5 content dependence), order-preservation on shuffled A/B, duplicate-multiplication arithmetic, bookended-adjacency merge, `Chr1`≠`chr1` no-join.
- Assert-or-port `_null_b_fields` whole: the port must reject (loud `ValueError`) any annotation B that is not 6-column, rather than silently switching null encoding.
- Port the loud parse guard philosophy (`parse_bed3`'s file+line `ValueError`) — the `genomic_coords.py` silent-empty guard precedent.

**Warning signs:**
- Production evaluator code with `sorted(` near the loj path; a `set()` or dict-dedup between join and emission; `'.'` in every null slot of a B6 row; `n_intersections` appearing in any gated assertion.
- Unit tests that only cover the frozen pair (one case) and none of the 8/11 handcrafted edge shapes.

**Phase to address:** Phase W — pins and corpora tests ship same-change with the evaluator (owner rule: lib changes carry same-change pytest coverage; per-module ≥96% standard).

---

### Pitfall 3 (S): The publish smoke actually publishes — `workflow_dispatch` on `publish.yml` today runs the real `twine upload`

**What goes wrong:**
Verified in `.github/workflows/publish.yml`: the workflow triggers on BOTH `release: [published]` AND `workflow_dispatch`, and the `Publish to PyPI` step (`twine upload dist/*` with `PYPI_API_TOKEN`) has **no event guard**. The naive "add a weekly workflow_dispatch dry-run" is a real publish of the current version: PyPI filenames are immutable (re-upload → 400 "File already exists"), so a smoke that uploads either consumes the version number permanently or fails confusingly on the duplicate. Two aggravating details in the current file:
- The `Verify installation` step runs `pip install dnallm` — from PyPI, unpinned, NOT from `dist/`. On any run that didn't (or failed to) publish, this verifies the OLD release: a false-green that would make a smoke look meaningful while proving nothing about the built artifact.
- The smoke's job is to catch exactly the rust classes that already happened (STATE 2026-10-11: uv implicit-system-env removal, `check_code.py --with-tests` flag removal — two consecutive v1.2.1 release failures). A smoke that skips the install-from-dist step catches neither.

**Why it happens:**
`workflow_dispatch` looks like a "safe manual trigger"; people forget the release trigger shares the same job steps. Twine has no true `--dry-run` upload mode, so "dry-run" must be constructed by gating steps, and each gate is a place to be wrong. `--skip-existing` is a trap door: it substring-matches generic 400s and can mask genuine failures (invalid metadata, quota) — twine 6.2.0 removed it for non-PyPI targets for exactly this reason.

**How to avoid:**
- Gate every mutating step: `if: github.event_name == 'release'` on the upload step (or split into `publish.yml` (release-only) + `publish-smoke.yml` (schedule/workflow_dispatch, never uploads)). A separate workflow is easier to prove safe by inspection.
- Smoke body (never-publish, catches the real rust classes): `uv build` → `twine check --strict dist/*` → create fresh venv → `uv pip install --find-links dist dnallm` (or `pip install dist/*.whl`) → `python -c "import dnallm"` + `dnallm --help`-class CLI probe → optionally `uv publish --dry-run` (validates against PyPI's `/validate` endpoint without uploading contents). Fix or delete the current `pip install dnallm` verify step — it must install the built wheel.
- Never use `--skip-existing` to make smoke re-runs idempotent; a re-run failing on "file exists" is the correct loud signal.
- If the smoke also covers the OIDC migration: `uv publish --dry-run` exercises credential/audience paths but the trusted-publishing handshake is only fully proven by a real publish — plan the migration so the next real release is explicitly observed (see Pitfall 10).

**Warning signs:**
- Any step in a dispatch-reachable workflow containing `twine upload`, `uv publish` (without `--dry-run`), or `--skip-existing`.
- A smoke whose verify step installs from the index rather than `dist/`.
- Smoke logs showing an upload HTTP request at all.

**Phase to address:** Phase S, and it is the FIRST change of that phase — the guard precedes any scheduled trigger existing (a cron on an unguarded workflow is a time bomb).

---

### Pitfall 4 (P): The `base` extras rename breaks downstream users silently and the in-repo install surface non-atomically

**What goes wrong:**
Two halves:
1. **Downstream (verified external behavior):** when an extra no longer exists, `pip install dnallm[base]` prints `WARNING: dnallm does not provide the extra 'base'` and **exits 0 with the extra's dependencies silently uninstalled** (pip ≥23.3, confirmed through pip 25.3; PEP 685 normalization applies). Every existing user of the kitchen-sink `base` extra gets a quiet, broken environment on the next install — no error to report, nothing to search for. uv behaves equivalently for unknown extras.
2. **In-repo lockstep:** `base` is referenced by the self-referential `all` extra (`dnallm[base,...]`), four CI install sites (`ci.yml:84,169,410`, `publish.yml:39`), `feasibility.yml:67`, README (2 commented mentions), and ~10 docs pages (`installation.md` alone has 8+ `.[base]` occurrences including the extras table at line 201 and the `--resolution=lowest` lane at 591). If `base` is re-scoped to be *minimal* (matching its name) rather than renamed, every CI leg installing `.[base]` for testing silently loses the transitive closure (dev/test/notebook/mcp) — the legs go red across the board, or worse, run with less and pass for the wrong reason.

**Why it happens:**
Extras renames look like metadata-only edits; pip's warn-and-continue makes the downstream half invisible to the maintainer's own testing (the maintainer installs from source with the new names). The self-referential aliasing (`all` → `base` → `dev,test,notebook,mcp`) means one rename has a two-level blast radius.

**How to avoid:**
- Decide rename-vs-rename+alias: keep a deprecated `base` alias extra that still resolves (e.g. `base = ["dnallm[<new-name>]"]`) for at least one version cycle, with a README/docs migration note — the only way `pip install dnallm[base]` keeps working for existing users. If the owner chooses a hard break, the version must be a minor bump with the change flagged at the top of the changelog.
- Same-change sweep, mechanical: `grep -rn "\[base\]" README.md docs/ .github/ scripts/` must return zero stale hits (or only the sanctioned alias definition); update `all`'s member list; add an `EXPECTED_*_MEMBERS` bracket test mirroring `tests/test_extras_guard.py` for the renamed extra so a future edit can't silently drop members.
- Update all four CI install sites + publish.yml + feasibility.yml in the same commit as the pyproject edit — CI red on a missing extra is actually the *loud* failure; the danger is legs that still resolve but under-install.
- Ordering: this must land BEFORE the docs accuracy sweep (Pitfall 9) and ideally in the same phase as the cuda130-default/GPU-matrix docs decisions, so docs are written once against the final extras state.

**Warning signs:**
- A pyproject diff renaming an extra with no docs/CI diff in the same commit.
- CI logs containing `does not provide the extra` (pip) or uv's equivalent warning — treat any occurrence as a broken leg even when green.
- `pip install dnallm[base]` in a fresh venv resolving to fewer packages than before (diff the pip freeze).

**Phase to address:** Phase P, first plan of the phase; alias decision is an owner call at planning time (like the triton #18 decision).

---

### Pitfall 5 (O): Ollama-surface removal deletes tests that guard NON-ollama contracts

**What goes wrong:**
The directive says "~59 refs across 5 test files + 2 notebooks; remove-vs-skip per item." The refs are not equivalent. Verified by reading the files:
- `tests/test_extras_guard.py` (5 refs): these are the **extras-bracket guards** — `langchain-ollama>=1.1.0` declared in the `mcp` extra (REPAIR-04), the five pre-existing mcp members, the `ipython<9` kernel-plot pin, AND `TestNotebookKernelPlotCompat` (the installed IPython/matplotlib pair that kills `plt.subplots` in every notebook kernel). Removing these as "ollama-related" deletes the guard against the exact matplotlib/IPython regression that broke the example lane once already, and un-guards the pydantic-ai cap rationale (PRs #47/#48 pending on exactly these guards).
- `tests/test_runner_infra_contracts.py` (15 refs): pins the in-repo `scripts/runner/ollama.service` systemd unit's Environment lines (D-06/D-12), the README re-apply op, and **docs-mirror content contracts** (`"ollama:qwen3.5:4b"` must appear in the langchain notebook AND its `docs/example` mirror .md). These guard the runner box and mirror honesty, not ollama execution.
- `tests/test_models_lock_contracts.py` (1 ref): a docstring mention ("ollama models only") inside a models.lock contract test — zero ollama behavior.
- `tests/examples/test_notebook_execution.py` (33 refs): a mix of true ollama-execution surface (`_gate_ollama_stack`, stage-3 batch inventory) and the shared gating machinery (`_probe_http_with_retry` semantics tests) that also serves MCP-server probing.

**Why it happens:**
Grep-by-keyword classification. "5 files contain ollama" reads like "5 files are ollama tests." The kernel-plot compat test shares a file with an ollama-named declaration; the runner contracts share the noun.

**How to avoid:**
- Classify per-assertion, not per-file, at planning time (the directive already says "per item" — hold to it): (a) tests that EXECUTE ollama or gate on the ollama endpoint → remove or skip per owner; (b) tests that mention ollama while guarding extras/mirrors/runner-infra/kernel-plot → KEEP, possibly with renamed fixtures; (c) the two notebooks → if removed, the docs-mirror and mkdocs-nav consequences (Pitfall 7) ride along; if kept-unexecuted, say so in prose (honesty convention).
- After the sweep, `grep -rn ollama tests/` should return exactly the sanctioned residue list, recorded in the phase plan — a second grep gate so later phases can't accidentally re-widen.
- If `langchain-ollama` leaves the `mcp` extra when the notebooks' status changes, update `EXPECTED_MCP_MEMBERS` same-change — the bracket test will fail loudly otherwise (that is it working).

**Warning signs:**
- A removal diff touching `TestNotebookKernelPlotCompat`, `EXPECTED_MCP_MEMBERS`, or the systemd Environment-line pins.
- Coverage or CI green with the ipython<9 pin no longer asserted anywhere.
- The remove-vs-skip ledger in the plan listing file names without assertion names.

**Phase to address:** Phase O, at plan time (the classification IS the plan); execution is then mechanical.

---

### Pitfall 6 (O/W): Census-triple and skip-allowlist collisions — the nightly hard gates trip on count drift the change never anticipated

**What goes wrong:**
Three pinned gates interact with any test-set change:
1. **D-03 census triple** (example-nightly stage 0.5, ci.yml ~line 853): hard-asserts `208/217 tests collected (9 deselected)` with the exact stage-1 selector flags. REMOVING tests (ollama) or the mcp_example pair changes collected/deselected counts → nightly goes red at stage 0.5 before anything runs. SKIP-marking does NOT change collection counts — a skip still collects.
2. **`expected_skips.yaml` + audit_skips in 4 CI jobs:** every NEW skip message must match an allowlist entry or the job fails. Skip-marking the ollama surface with a fresh reason string ("removed-surface:" etc.) without a same-change allowlist entry → all four legs red. Conversely, removals orphan allowlist entries (harmless to the gate, but review debt — clean them same-change per the file's own "reviewable suppression record" charter).
3. **Stage-3 OLLAMA BATCH orphan:** example-nightly stage 3 re-runs "just the two gated mcp tests" when ollama is up. If those tests are REMOVED but the stage remains, pytest selects nothing → exit code 5 (no tests collected) → recorded as a failure in the stage-4 summary → nightly red even though nothing is wrong. The stage must be deleted/rewritten in the same change as the removal.

Platform-split adds a fourth: new platform-keyed skips fire on the **test-windows** junit (which also runs audit_skips, ci.yml line 187) with messages that may differ from Linux (precedent: `fd accounting needs /proc` entry exists precisely for the Windows leg).

**Why it happens:**
The gates are deliberately pin-happy (the bump-point comments say so); every count change is intentional friction. A planner thinking "this is just a test removal" misses that three separate pinned integers move.

**How to avoid:**
- Any Phase O / Phase W plan lists an explicit "pinned-gate re-pin" task: new census triple measured from a real collection run (exact selector flags), `expected_skips.yaml` delta, and ci.yml stage-3 disposition — each in the same commit as the test change.
- Prefer the typed-skip-prefix convention for anything kept; reuse existing prefixes (`environment-unavailable:`, `network-unavailable:`) where truthful rather than inventing near-duplicates.
- For the Windows numpy path: new markers (if used to select the numpy-path execution lane) must be registered in `pyproject.toml [tool.pytest.ini_options] markers` — `--strict-markers` fails collection otherwise, which itself would trip the census gate confusingly.

**Warning signs:**
- Nightly failing at stage 0.5 with "census collection mismatch" after a test-count change (this is the guard working — re-pin deliberately, never widen the grep).
- `UNEXPECTED SKIPS` in any audit_skips job output.
- pytest exit 5 in any nightly stage log.

**Phase to address:** Phases O and W — each phase's verification includes a dispatched (or owner-watched) nightly run before close, given the nightly lane is the only place these gates fire.

---

### Pitfall 7 (O/D): Notebook removal collides with the byte-identical mirror gate and mkdocs nav

**What goes wrong:**
If the two `example/mcp_example/*.ipynb` notebooks are REMOVED (rather than kept-unexecuted): `check_docs_sync.py` requires the mirror copies under `docs/example/mcp_example/` to be removed too — an `.ipynb` exists on BOTH sides, so leaving the mirror copy produces `ONLY in docs/example/` (the `.md` wrapper exemption covers only docs-side `.md` files). But the `.md` wrappers (docs-only by design) then reference notebooks that no longer exist, and `mkdocs.yml` nav entries pointing at removed/emptied pages break `mkdocs build --strict` (which already carries 15 pre-existing warnings — new broken-link warnings there are easy to misattribute to the pre-existing pile). The nightly docs-validation and mirror checks then fail on a change that was conceptually "just delete two notebooks."

**Why it happens:**
The mirror is enforced byte-identical in both directions, but the docs AROUND the mirror (wrappers, nav, showcase pages) are a separate consistency surface with no automated cross-check to notebook existence.

**How to avoid:**
- If removal: same-change — delete both sides' `.ipynb`, decide wrapper disposition (delete wrappers + nav entries together; a wrapper without its notebook is dead prose), run `mkdocs build --strict` locally and account for the delta vs the 15 known warnings.
- If keep-unexecuted (recommended: consistent with the owner's "window stays open in the ledger" stance on #13): the notebooks stay, their execution tests skip/remove, and a short prose note in the wrapper states they are not executed by CI (the project's honesty convention — same class as the IA³ forward-pointer handled in v1.2).
- Any Phase D sweep touching example notebooks at all (even a markdown-cell typo) must regenerate the mirror same-change via `scripts/generate_md_from_notebook.py` / `generate_md_from_marimo.py` — the byte-identical gate (`cmpfiles(..., shallow=False)`) reads bytes; "looks the same" is not a defense.

**Warning signs:**
- `SYNC ERRORS:` from check_docs_sync in docs-validation.
- mkdocs --strict warning count > 15 after a notebook/docs edit.
- A diff editing `example/**` with no corresponding `docs/example/**` diff (and vice versa).

**Phase to address:** Phase O (removal decision) and every phase touching example/ — the mirror-regen rule is a standing verification step for W and D too.

---

### Pitfall 8 (W): The Windows leg goes skip-green instead of becoming the numpy-path executor — and the coverage gate quietly reprices platform-split code

**What goes wrong:**
Two entangled failure modes:
1. **The dishonesty class the project explicitly rejects** (v1 Phase-5 decision: "an ever-green skip is the same dishonesty class as the false-green CI gates"): the easy way to make test-windows pass the bedtools-gated tests is `skipif(sys.platform == "win32")` on the EXISTING tests — Windows then never executes the numpy evaluator, and "Windows platform parity" ships as a skip. The owner directive is the opposite: test-windows becomes the numpy-path EXECUTOR.
2. **Coverage-gate math:** the numpy evaluators will live under `dnallm/` (in the coverage denominator on EVERY leg; the example code that calls them is NOT in the denominator). test-windows runs `pytest -m "not slow" --cov` (ci.yml line 182) — bare `--cov` activates `fail_under=90` there too. If the evaluator's unit tests are written skipif-guarded or only meaningful on Windows, the Linux coverage legs (fast gate + nightly census) lose their only coverage of the new modules → the 90 floor can breach (the floor is 90; the suite sits at 96.72 — headroom is real but not infinite for two new modules).

**Why it happens:**
Un-slowing the showcase/NER execution tests for Windows floods the hosted leg with heavy notebook execution (models, HF downloads, the 60-minute timeout, cold pulls — the owner already dropped models-cache as quota-hostile), so the tempting shortcut is skips on one side and execution on neither.

**How to avoid:**
- Two-layer test design, both layers platform-neutral where possible: (a) **unit layer** — evaluator correctness tests on synthetic BEDs (spike corpora, baked oracle literals) run on ALL legs including Linux: this is what keeps coverage and parity proven everywhere; (b) **execution layer** — the notebook/script execution tests get platform-AWARE dispatch (Linux: bedtools/pybedtools path on the nightly box; Windows: numpy path), selected by the existing gates (`_gate_script_pybedtools`, `_gate_pybedtools`, the `which('bedtools')` assert → numpy-path pass) rather than by a blanket marker flip. The three gate sites are inventoried in WINDOWS #19: `test_script_execution.py:128`, `test_notebook_execution.py:1397`, `test_plant_helixseek_showcase.py:523`.
- Scope what test-windows actually executes: the parity-relevant execution subset, not the whole slow lane; a hosted-leg time budget must be measured in the first Windows-path PR (precedent: D-12 measured budgets citing run ids).
- Keep the Windows numpy-path execution failure LOUD: if the numpy path is what Windows runs, a missing-model/env failure on Windows must surface as the typed skip classes already exist (probe-then-execute), never as a silent pass.

**Warning signs:**
- `skipif(sys.platform == "win32")` on any bedtools/pybedtools-gated test that has no Windows-executing sibling.
- test-windows junit where the numpy evaluator never appears in any stack trace.
- Linux coverage report showing the new evaluator modules below the per-module ≥96% standard.

**Phase to address:** Phase W — test architecture is decided at plan time, before the first evaluator commit.

---

### Pitfall 9 (D/P): The docs accuracy sweep lands before the packaging decisions and is stale on arrival

**What goes wrong:**
The sweep reviews `docs/` + `example/` per-file for accuracy vs code. If it runs BEFORE the extras rename, cuda130-default change, cuda121 retirement, GPU-matrix documentation, or the Windows platform-split example edits, every reviewed page is invalidated by the very next phase — the milestone ends with a "docs accuracy review" whose pages describe a package that no longer exists. The specific high-churn surfaces: `installation.md` (extras table line 201, `.[base]` ×8, `.[cuda121]` line 517, `.[base,cuda124]` line 442), `example/notebooks/overview.md` (3 mentions of `.[base,notebook,cuda124]` as the DEFAULT install — owner directive 2026-10-11 already retargets this to cuda130), README install block, and every page whose snippets embed example code (checked by `check_docs_code.py` / `validate_docs_snippets.py`).

**Why it happens:**
Docs feel like a safe parallel workstream, so they get scheduled first or in parallel. But this sweep's deliverable is *accuracy*, which is a function of the final code state — it is the one phase with a hard ordering dependency on nearly every other phase.

**How to avoid:**
- Order: O/G → W/P → S → D, with Phase D last as the honesty backstop absorbing all downstream doc consequences (the GPU support matrix row in STATE has the full pre-verified boundary table ready to paste: Pascal cpu-only, Turing = cuda121–128 ceiling, Ampere+ = cuda130 default, 50xx cuda128+; verify-against-local-torch evidence 2026-10-11).
- The sweep must be uv-vs-pip lane-split (STATE extras-hygiene item d): plain `pip install dnallm[cuda130]` silently ignores the uv-only `[tool.uv.sources]` index pin leaving users on default torch — every install command in docs needs its lane labeled, and the pip lane needs the explicit `--index-url https://download.pytorch.org/whl/cu130` guidance or an honest "uv-only convenience" note.
- Same-pass fix discipline (owner directive): findings are fixed in the same sweep, not parked — but fixes to MIRRORED content must follow Pitfall 7's regen rule, and fixes to notebook code cells require re-execution evidence (nightly) before the committed outputs change.

**Warning signs:**
- A roadmap where Phase D precedes P or W.
- Docs review notes that say "accurate as of commit X" where X precedes the pyproject rename commit.
- `grep -rn "cuda121\|\[base\]" docs/` returning hits after both packaging phases merged.

**Phase to address:** Phase D ordering decision — roadmap-creation time, before any phase executes.

---

### Pitfall 10 (S): The OIDC Trusted Publishing migration fails on claim-matching details a dry-run cannot prove — and the old token's removal races the new chain's first use

**What goes wrong:**
Verified external behavior (MEDIUM confidence, cross-checked): PyPI matches the OIDC token against the registered publisher on **owner + repository + workflow FILENAME (not the workflow's `name:`) + environment name**, exactly. The failure modes that recur in the wild: `environment:` on the job missing or differing from the PyPI-side registration (`invalid-publisher` / `invalid-pending-publisher`); a workflow rename (e.g. splitting `publish.yml` into `publish.yml` + `publish-smoke.yml` per Pitfall 3) that makes the SMOKE workflow an unregistered publisher while the release workflow stays registered — or vice versa; `permissions: id-token: write` missing at the job/workflow level; forks receiving `id-token: write` but failing at upload (expected; PR-authored code must never reach publishing anyway — matches the repo's self-hosted policy stance); reusable workflows NOT supported as publisher (warehouse#11096) — relevant if the smoke is factored into a reusable workflow to share steps with the release job, which is precisely the refactor a tidy implementer would do. Tooling: plain `twine upload` does NOT perform the OIDC exchange itself — the migration means either `pypa/gh-action-pypi-publish` (which exchanges the token then uploads via twine internally) or `uv publish --trusted-publishing` (the repo already uses uv everywhere else; `--check-url` gives the skip-existing-equivalent semantics safely). Finally: deleting `PYPI_API_TOKEN` before the first successful OIDC publish leaves the chain unable to fall back; keeping it forever defeats the migration's purpose.

**Why it happens:**
Every requirement is invisible until the first real release; a dry-run (`uv publish --dry-run`) validates metadata and the `/validate` endpoint but does not complete a trusted-publishing mint-and-upload, so the migration can be "done" and still fail only when a release is cut. The STATE row already flags the decision as riding the publish-smoke phase — the pitfall is treating the smoke as sufficient proof.

**How to avoid:**
- Register the publisher with the FINAL workflow filename(s) after the Pitfall-3 split, with a named GitHub `environment:` (e.g. `pypi`) used identically on both sides; add `permissions: id-token: write` (and `contents: read`) to the publishing job only, never workflow-wide.
- Migrate to `uv publish --trusted-publishing always` (explicit; `automatic` can silently fall back to token auth if the secret still exists — a false sense of migration) or the SHA-pinned `pypa/gh-action-pypi-publish`.
- Keep the token through the first observed real release, then delete the secret and re-run the smoke to prove it no longer depends on it. Record the observed release run id in the phase verification (the project's evidence convention).
- Do not factor the publish steps into a reusable workflow (unsupported as publisher).

**Warning signs:**
- `invalid-publisher` / `invalid-pending-publisher` in a publish attempt (claim mismatch — check environment + filename first).
- A migrated workflow still carrying `TWINE_PASSWORD`/`PYPI_API_TOKEN` refs after the first successful OIDC release.
- Smoke green while no real release has yet exercised the mint.

**Phase to address:** Phase S, sequenced after the Pitfall-3 workflow split; closure criterion includes one observed real release.

---

### Pitfall 11 (W/G): GAP-1 un-skips surface before the environment truly supports them — and the compat shim's blast radius crosses all 35 model families

**What goes wrong:**
WINDOWS #11/#12 are both the same class: the NT promoter checkpoint's remote code needs removed transformers-4.x `PretrainedConfig` defaults (`is_decoder`/`add_cross_attention`) under 5.x. The v1.3 disposition is "compat-shim vs removal at planning time." Two ways it goes wrong:
1. **Shim blast radius:** the natural home (`dnallm/utils/transformers_compat.py`, already spanning 4.49–5.x) patches config defaults GLOBALLY — every model family loads through the patched path, so a too-broad patch (re-adding defaults unconditionally, mutating `PretrainedConfig.__init__` for all models) can change behavior for models that never needed it. The shim must also fire BEFORE the remote module instantiates its config (import-order sensitivity), and remote-code execution is `trust_remote_code` territory — keep the shim declarative and version-keyed, and prove no-diff on a non-NT family.
2. **Premature un-skip:** #12's typed skip is self-healing by design (fires only while the env can't). If the shim works locally but the checkpoint still fails on CI (HF endpoint/mirror variance, runner cache state), un-skipping turns the leg red — and if someone then "fixes" the redness by loosening the probe, the probe-then-execute discipline (live evidence carried in skip messages, both directions proven — a Key Decision) erodes. Closure of #15 riding along has the same shape: those optional-dep gates prove *environment* availability, not code correctness.

**Why it happens:**
A local green is a powerful but partial proof; the gap between "loads on the dev box" and "loads in every CI lane" is exactly the network/cache variance the typed-skip system was built to absorb.

**How to avoid:**
- Decide shim-vs-removal in the plan with a blast-radius estimate: shim = version-keyed, minimal-attribute, guarded (matchable loud error if the remote code's expectations drift again), plus a regression test pinning a non-NT family's config behavior unchanged; removal = benchmark notebook loses its third model — an honest surface narrowing that must be documented, not slipped in.
- Un-skip only with the skip's own probe proven green in the target lane; keep the probe machinery intact so a future regression re-skips loudly rather than failing opaquely.
- If #15 closes on shim success, close it with per-item probe evidence (the ledger entry lists the specific gates: evo/megaDNA optional-dep, megaDNA demo cell, PlantCAD lora pair) — not a blanket un-skip.

**Warning signs:**
- A compat shim without a version guard or without a non-target-family regression test.
- Un-skipped tests whose probe helpers were deleted in the same diff.
- Benchmark docs still listing three models after a removal disposition.

**Phase to address:** Phase G (decision at plan time); interacts with Phase W — the Windows NER execution path needs the NT BPE checkpoint loadable, so G precedes or rides the front of W.

---

### Pitfall 12 (P/S): cuda121 retirement and the rocm pin leave stale references in CI, uv config, and docs — and the triton-windows declaration imports a community fork into the published surface

**What goes wrong:**
Three packaging half-changes:
1. **cuda121 retirement:** `pyproject.toml` keeps a `cuda121` extra, a `torch-cuda121` uv index, sources entries, and conflicts pairs; `ci.yml`'s `test-cuda` matrix still lists `'12.1'` (line 218 area); `docs/getting_started/installation.md:517` shows `.[cuda121]` as the chooser example. Retiring the extra without all three (extra+index+sources+conflicts, matrix leg, docs) leaves a CI leg installing a dead extra (pip warn-and-continue per Pitfall 4 — silent under-install on a GPU lane) and docs pointing users at a retired lane.
2. **rocm restore-or-remove:** the `[tool.uv.sources]`/conflicts rocm block is commented out, so `.[rocm]` installs default-index torch — README's ROCm claim is inert TODAY. Restoring means uncommenting an index pinned to `rocm6.2` — stale against the torch 2.11/<2.12 the package now rides (torch-ROCm wheel lanes move per release; a stale index either resolves old torch or fails loudly). Removing means the README claim and docs must stop promising ROCm (a docs-honesty item, not just pyproject hygiene).
3. **triton-windows (#18, MEDIUM confidence external):** the fork has moved under the `triton-lang` org with multi-vendor contributors and reproducible+attested builds since 3.7.1.post27, but remains a community package separate from upstream, version-coupled to torch (3.6↔2.10, 3.8↔2.14/2.15) with DLL/vcredist failure modes on user machines. Declaring it behind a Windows marker in the `fla` extra makes every Windows fla user ride community release cadence and the torch-pairing being right; NOT declaring it keeps the current manual-step documentation (the local remedy: triton-windows==3.7.1.post27) — the honest middle is documenting the exact version pairing and keeping `test_chunk_kda_importable_when_fla_installed`'s skip behavior truthful on Windows.

**Why it happens:**
Each is "just pyproject hygiene" — but every extra is a public API (Pitfall 4), every uv index a resolution path, and every README claim a contract.

**How to avoid:**
- Treat extras changes as atomic units: extra + uv index + sources + conflicts + CI matrix + docs + README in one change, verified by a fresh-venv resolve per affected extra (the `--resolution=lowest` lane in installation.md:591 is the existing precedent for resolve-verification).
- For rocm: verify the restored index actually carries torch matching the `<2.12` cap before committing; if not, the honest move is removal + docs, not a stale restore.
- For triton-windows: default to documentation-of-manual-step (current state, already proven on RTX 5080) unless the owner accepts the third-party dependency; if declared, pin `<` the next major and add a bracket guard so dependabot-style bumps can't silently decouple it from torch.

**Warning signs:**
- `grep -n "cuda121\|rocm" pyproject.toml .github/workflows/ci.yml README.md docs/ -r` returning inconsistent subsets after any packaging PR.
- A GPU CI leg green in less time than usual (under-installed via a warn-and-continue extra).
- README promising an install path that resolves a different torch than documented.

**Phase to address:** Phase P (all three), with the triton-windows decision recorded owner-side at plan time like the other owner calls.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `skipif(win32)` on bedtools-gated tests instead of a real Windows numpy-path lane | test-windows green fast | "Parity" ships unexecuted on Windows — the exact ever-green-skip dishonesty class the project bans | Never for the parity claim itself; acceptable only for genuinely Linux-only tools (fd accounting precedent) |
| Re-running spike scripts as the production evaluator (import from `.planning/quick/`) | Zero porting work | Planning artifacts become runtime dependencies; pins untested against library context | Never — port + unit-test (Pitfall 2) |
| Leaving `PYPI_API_TOKEN` after OIDC migration | Fallback safety | `--trusted-publishing automatic` silently uses it; migration is theater | Only until first observed real release, then delete (Pitfall 10) |
| Keeping a deprecated `base` alias extra forever | No downstream break | Misleading name persists (the original complaint) | One version cycle, with migration note |
| Orphaned `expected_skips.yaml` entries after removals | Zero gate friction | The allowlist drifts from reality, eroding its value as a reviewable record | One phase — clean same-change when the sweep lands |
| `|| true` on mypy in CI legs (current state) | No gate on a known-noisy check | Version drift accumulates invisibly (136 errors already seen on fresh runners) | Until the cap lands in Phase S; not after |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PyPI upload | Smoke/dispatch run reaching `twine upload`; trusting `--skip-existing` | Event-gate upload steps; build + `twine check --strict` + install-from-dist; filenames are immutable (Pitfall 3) |
| PyPI Trusted Publishing | Registering with workflow `name:` instead of filename; no `environment:`; reusable workflow; plain twine expected to do OIDC | Match owner/repo/filename/environment exactly; `uv publish --trusted-publishing always` or gh-action; prove on first real release (Pitfall 10) |
| pip extras | Assuming a bad extra errors | It warns and exits 0 — grep CI logs for `does not provide the extra` as a health check (Pitfall 4) |
| uv indexes | Documenting `pip install dnallm[cuda130]` as equivalent | `[tool.uv.sources]` is uv-only; pip lane needs explicit `--index-url` or honest labeling (Pitfall 9) |
| bedtools oracle | Re-freezing truth against a different bedtools version than v2.31.1 | The spikes' oracle is PATH-resolved v2.31.1; Linux keeps that binary (rootless micromamba cache keyed on it) — `--bedtools`-style override + version echo in any re-verification |
| GitHub self-hosted runner | Letting PR-authored CI reach the nightly box during new Windows-lane work | Keep push/PR guards off self-hosted labels; new legs hosted-only until proven (existing GATE-02 policy) |
| HF endpoints | Un-skipping model-download tests assuming reachability | Keep probe-then-execute with live evidence in messages (existing D-08 contract) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Un-slowing the showcase/NER execution tests for the Windows numpy lane | test-windows (60-min timeout) blows budget on hosted runners; fast PR legs everywhere slow down | Platform-aware dispatch on the execution layer only; scoped parity subset; measured budget in the first PR (D-12 precedent) | First push after the marker flip |
| Cold model pulls on windows-latest | Multi-minute HF downloads per run, no cache quota | Scope the Windows execution subset to small checkpoints; accept and budget, or pin a targeted cache | Immediately, every run |
| example-nightly schedule collisions on the shared GB10 box | OOM/VRAM exhaustion; memory floor trips | Keep the staged-serial D-07 layout; any new Windows-only work stays OFF the nightly box | When a new stage is inserted without the ≥35Gi floor assert between server-binding stages |
| Property-corpus tests at spike scale in the fast lane | 3,000 subprocess oracle invocations (5s on Linux with bedtools; minutes without) | Bake oracle literals for the Windows/no-bedtools legs; run the live-oracle variant only where bedtools exists (nightly) | First Windows run of the corpus test |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| workflow-wide `permissions: id-token: write` (instead of job-scoped) during OIDC migration | Every job in the workflow can mint PyPI tokens | Job-level permissions on the publish job only; `contents: read` elsewhere |
| Keeping `PYPI_API_TOKEN` in secrets post-migration "just in case" | Long-lived credential outlives its purpose; automatic trusted-publishing silently prefers it | Delete after first observed OIDC release; re-run smoke to prove independence (Pitfall 10) |
| Declaring triton-windows (community fork) without version ceiling | Supply-chain + silent torch decoupling for every Windows fla user | Prefer documented manual step; if declared: pin, bracket-guard, attested-builds series only (≥3.7.1.post27) |
| GAP-1 shim broadening remote-code config defaults globally | Behavior change across 35 model families; unpredictable interaction with future remote code | Version-keyed minimal shim + non-target-family regression test (Pitfall 11) |
| `continue-on-error` anywhere in the publish chain | Releases "succeed" unpublished (documented real-world TP failure mode) | Hard-fail every publish step; the smoke exists precisely to make failures loud |

## UX Pitfalls (installer/docs UX — this milestone's user-facing surface)

| Pitfall | User Impact | Better Approach |
|---------|-------------|------------------|
| Extras rename without an alias or migration note | `pip install dnallm[base]` quietly installs a fraction of the intended environment | Deprecated alias extra for a cycle + changelog/docs migration note (Pitfall 4) |
| GPU support matrix implying Pascal/Turing work on cuda130 | Users burn hours on `sm_75`/`sm_61` missing-kernel errors | Paste the pre-verified boundary table verbatim (Pascal cpu-only; Turing ceiling cu128; Ampere+ cu130; 50xx cu128+) with the minimum-GPU-line stated |
| Install docs silent on uv-vs-pip lane difference | pip users get default torch and blame the package | Lane-split every install command; explicit `--index-url` for pip (Pitfall 9) |
| Removed ollama notebooks leaving dead wrappers/nav | Docs readers hit broken pages | Wrapper + nav disposition decided with the removal (Pitfall 7) |
| Windows numpy-path failure modes phrased as internal errors | Windows users can't self-diagnose missing prerequisites | Loud, named-cause errors (the D-08 convention) in the platform dispatch |

## "Looks Done But Isn't" Checklist

- [ ] **Windows parity:** numpy evaluator has NO unit test on Linux asserting the oracle pins (null-fill, order, multiplicity) — parity is only "proven" by the spike prose, not the suite.
- [ ] **Windows parity:** Linux still executes bedtools after the split — verify a structure test asserts dispatch direction AND the nightly re-executed the edited notebook (Pitfall 1).
- [ ] **Windows parity:** frozen-value comparisons use the pinned tolerance, not `==` (trailing deltas: numpy full-precision 0.32472687098490116 vs the 6-dp anchor 0.324727, delta ~1.3e-7); `selection.md` bands themselves untouched.
- [ ] **Ollama removal:** `test_extras_guard.py` / `test_runner_infra_contracts.py` non-ollama contracts survive — grep for `EXPECTED_MCP_MEMBERS`, the systemd Environment pins, `TestNotebookKernelPlotCompat` after the sweep.
- [ ] **Ollama removal:** census triple re-pinned with the exact stage-1 selector flags; `expected_skips.yaml` delta present; ci.yml stage 3 dispositioned.
- [ ] **Packaging:** `grep -rn "\[base\]"` across README/docs/workflows/scripts returns only sanctioned occurrences; `all` extra updated; a fresh-venv `pip install dnallm[<new-extra>]` resolve-verified.
- [ ] **Packaging:** cuda121 gone from extra+index+sources+conflicts+matrix+docs consistently; rocm either verified-restored or honestly removed.
- [ ] **Publish smoke:** no dispatch-reachable step can upload; the verify step installs from `dist/`, not the index; smoke catches the two historical rust classes (uv system-env, check_code.py flags) by construction.
- [ ] **OIDC:** first real release observed and recorded; token secret deleted; smoke re-run green post-deletion.
- [ ] **Docs sweep:** runs LAST; every example/ edit carries its regenerated mirror; mkdocs --strict delta accounted vs the 15 known warnings; install commands lane-labeled.
- [ ] **GAP-1:** non-NT-family regression test present if shimmed; un-skipped tests retain their probe machinery.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Dispatch leakage — Linux truth re-derived by numpy, values drifted out of band | MEDIUM | Revert dispatch to fail-loud; re-execute nightly; re-verify bands from committed outputs; add the missing structure test |
| Dispatch leakage — values held silently (invariant violated, no alarm) | HIGH (trust repair) | Re-freeze decision back to owner: prove bedtools path re-executed, re-pin committed outputs, add the dispatch-direction test that was missing |
| Smoke published a real version | HIGH (irreversible artifact) | Version is consumed permanently; yank from PyPI if wrong, cut a patch release, then install the event guards that should have existed |
| Extras rename broke downstream | MEDIUM | Ship a compat alias extra in a patch release; changelog migration note; no way to reach already-broken envs except docs |
| Ollama sweep deleted non-ollama guards | LOW | Restore from git history (tests are self-contained); the census/gates surface the loss within one nightly cycle |
| Census gate red after count change | LOW | Measure the new triple with exact selector flags, re-pin deliberately (the designed bump-point — never widen the grep) |
| OIDC first-release failure (claim mismatch) | LOW | Fix environment/filename registration; release re-runs; token still present as fallback if not yet deleted |
| GAP-1 shim regressed another family | MEDIUM | Narrow the shim's version/attribute guard; restore pinned behavior via the regression test; fallback disposition = removal + docs |

## Pitfall-to-Phase Mapping

Suggested ordering (phases numbered from 13 by the roadmap author):

| Phase (topic) | Order | Pitfalls Prevented | Verification That Prevention Worked |
|---------------|-------|--------------------|-------------------------------------|
| Phase O — ollama surface removal | 1st | 5, 6, 7 | Per-assertion remove-vs-skip ledger; post-sweep `grep ollama tests/` = sanctioned residue; census triple re-pinned; nightly dispatched green; non-ollama guards still present |
| Phase G — GAP-1 disposition | 2nd (may fold into W's front) | 11 | Shim blast-radius test or documented removal; probes intact on any un-skipped test |
| Phase W — Windows platform parity | 3rd | 1, 2, 6, 8 (+ tolerance pins) | Spike-corpus unit tests green on Linux AND Windows; dispatch structure test; nightly Linux re-execution in band; test-windows executes the numpy path (evidence in junit); coverage ≥96% per-module on the new evaluators |
| Phase P — packaging & install honesty | 4th | 4, 9 (precondition), 12 | Fresh-venv resolves per changed extra; zero stale `.[base]`/cuda121 refs; GPU matrix page matches the pre-verified boundary table; extras bracket tests added |
| Phase S — publish smoke + OIDC + mypy cap | 5th (smoke guard can land earlier if split out) | 3, 10 | Smoke run green without any upload HTTP traffic; first real release observed via OIDC; token deleted; mypy cap matches CI-validated version |
| Phase D — docs accuracy sweep | LAST | 7, 9 (+ absorbs all downstream doc drift) | Per-file review record; mirrors byte-identical after every fix; mkdocs --strict delta accounted; install commands lane-labeled; cuda130 default consistent repo-wide |

**Ordering rationale:** O first because it *shrinks* the surface every later phase touches (one census re-pin, one allowlist delta, instead of repeated collisions). G before W because the Windows NER execution path needs the NT checkpoint loadable. P before D because docs must describe the final extras state. S is position-flexible except that its first act (the upload guard) should precede any scheduled trigger existing. W is the long pole and benefits from the reduced surface.

**Research flags for phases (for the roadmap author):**
- Phase W: needs plan-time spikes/decisions — test-windows execution subset scope + time budget; tolerance-pin values (full-precision vs 6-dp anchor); marker registration for any new selector.
- Phase S: needs plan-time verification of current uv publish `--dry-run` behavior in the repo's uv version, and the owner decision on subsuming OIDC.
- Phase P: owner decisions at plan time — base rename target + alias-or-break; rocm restore-or-remove; triton-windows declare-or-document.

## Sources

- Repository (HIGH confidence, primary): `pyproject.toml`; `.github/workflows/publish.yml`; `.github/workflows/ci.yml` (test-windows leg, census stage 0.5, stage 3, bedtools provisioning); `scripts/audit_skips.py`; `scripts/check_docs_sync.py`; `tests/expected_skips.yaml`; `tests/test_extras_guard.py`; `tests/test_runner_infra_contracts.py`; `tests/examples/test_plant_helixseek_showcase.py`; `.planning/WINDOWS.md` (#11/#12/#13/#18/#19/#20 with owner directives and the 2026-10-10/11 change-surface inventory); `.planning/STATE.md` deferred rows 2026-10-11; spike summaries `261010-wx2` / `261011-0a3` (pins, portability notes, determinism evidence).
- PyPI Trusted Publishing troubleshooting + internals (official docs; MEDIUM): https://docs.pypi.org/trusted-publishers/troubleshooting/ — claim matching, rate limits, rename breakage, reusable-workflow limitation (warehouse#11096).
- Trusted Publishing adoption/pitfalls guides (MEDIUM, cross-checked): https://safeguard.sh/resources/blog/pypi-trusted-publishing-adoption-guide and .../pypi-trusted-publishing-common-pitfalls — environment mismatch, continue-on-error masking, fork behavior.
- `uv publish --dry-run` (MEDIUM): https://github.com/astral-sh/uv/pull/15638 and https://github.com/astral-sh/uv/pull/21439 — `/validate` endpoint behavior, metadata/attestation preparation.
- pip nonexistent-extra behavior (MEDIUM, multi-source): https://github.com/peteromallet/desloppify/issues/149 (pip 25.3 warn + exit 0), https://github.com/pypa/pip/issues/12376, https://stackoverflow.com/questions/78082481/ (PEP 685 normalization, pip ≥23.3).
- PyPI filename immutability + `--skip-existing` fragility (MEDIUM): https://twine.readthedocs.io/en/latest/_modules/twine/commands/upload.html (400+"already exist" substring logic), https://discuss.python.org/t/announcement-twine-6-2-0/103548/1 (removed for non-PyPI targets), https://github.com/pypa/twine/issues/1265, https://github.com/sonatype/nexus-public/issues/696, https://github.com/anthropics/claude-agent-sdk-python/commit/117cffacae4a1a06f911b3b2d695e33e57c86ec3 (partial-upload/quota masking example).
- triton-windows status (MEDIUM): https://github.com/pytorch/pytorch/issues/178368 (fork → triton-lang org, separate-package stance), https://github.com/triton-lang/triton-windows/releases (version cadence, attested builds), compatibility table and vcredist caveats via project README/discussions.
- LOW confidence, deliberately not relied upon: any default-branch workflow-file requirement for Trusted Publishing (underdocumented — flagged as unverified in the digest; the migration plan should not depend on it).

---
*Pitfalls research for: DNALLM v1.3 — Platform Reach & Packaging Honesty*
*Researched: 2026-10-11*
