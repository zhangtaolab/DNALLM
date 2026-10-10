---
quick_id: 261003-ryz
slug: close-phase-06-review-info-findings-in-01-06
description: >-
  Close the six routed-open Phase-06 code-review info findings: IN-01 (fetch_sequence
  pyfastx handle leak + .fxi sidecar beside the user's FASTA), IN-02 (normalize_chrom
  accepts non-ASCII digits), IN-03 (import-purity test leaves the package attribute
  rebound to a re-executed module), IN-04 (Anno label_names single quotes), IN-05
  (fla extra matched by substring), IN-06 (local .scratch/ ignore redundancy —
  verify-then-decide by live git check-ignore evidence).
estimate: 60000
created: 2026-10-03
files_modified:
  - dnallm/utils/genomic_coords.py
  - tests/utils/test_genomic_coords.py
  - dnallm/models/model_info.yaml
  - tests/models/test_plant_helixseek_fla_kernels.py
  - example/notebooks/plant_helixseek_shared/.gitignore
  - .planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md
---

<objective>
Close info findings IN-01 through IN-06 from the Phase-06 code review
(.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW.md, findings
re-verified live 2026-10-03), all routed `open` in 06-REVIEW-DISPOSITION.md:

- **IN-01**: `fetch_sequence(path)` (dnallm/utils/genomic_coords.py:152-155) constructs
  `pyfastx.Fasta` and never releases it, and pyfastx builds a `.fxi` index sidecar beside
  the user's FASTA — an undocumented filesystem side effect. Live-verified pyfastx 2.3.1
  facts that shape the fix: the class has NO `close()`/`__enter__`/`__exit__`; dropping the
  last reference releases all 3 held fds immediately under CPython refcounting (verified
  via /proc/self/fd: +3 on construct, back to baseline after `del`; psutil confirms no
  lingering open files); `build_index=False` avoids the sidecar but `fetch()` then raises
  `NameError` (random access requires the index) so it is NOT viable; default construction
  creates `<fasta>.fxi` at construction time and reuses an existing one.
- **IN-02**: `normalize_chrom` (genomic_coords.py:72-73) — `name.isdigit()` is True for
  full-width/superscript/Arabic-Indic digits, so `normalize_chrom("２")` silently returns
  `"Chr２"` against the module's own "never renamed heuristically" contract.
- **IN-03**: `test_module_import_is_pyfastx_free` (tests/utils/test_genomic_coords.py:288-304)
  restores `sys.modules` but not the `dnallm.utils.genomic_coords` package attribute that
  `importlib.import_module` rebinds to the re-executed module object — a latent
  two-live-module-objects trap.
- **IN-04**: the Anno entry `label_names` (dnallm/models/model_info.yaml:1657) is
  single-quoted while the rest of the registry uses double quotes (verified live: the line
  reads `label_names: ['O', 'B-CDS', ...]`).
- **IN-05**: `test_fla_reachable_from_all`
  (tests/models/test_plant_helixseek_fla_kernels.py:50-52) matches the extra by substring
  (`"fla" in spec`), which a future `"flash-attn"`-style spec would satisfy vacuously.
- **IN-06**: the reviewer claims the local `.scratch/` entry
  (example/notebooks/plant_helixseek_shared/.gitignore:1) is redundant with root
  `.gitignore:60`; the 06-01 plan claimed the opposite. VERIFY-THEN-DECIDE by live
  `git check-ignore` evidence, not by either claim.

Purpose: retire the review's info-severity debt with surgical fixes, each shipped with
same-change tests where dnallm/ code changes (owner rule), and settle the IN-06
contradiction with committed command-line evidence.

Output: deterministic pyfastx handle/sidecar handling in `fetch_sequence`, ASCII-strict
`normalize_chrom`, contamination-free import-purity test, quote-normalized Anno registry
entry, exact-member fla-extra assertion, and a decided (not argued) IN-06 outcome — plus
06-REVIEW-DISPOSITION.md rows IN-01..IN-06 flipped to `fixed`.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW.md
@.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md
@dnallm/utils/genomic_coords.py
@tests/utils/test_genomic_coords.py
@tests/models/test_plant_helixseek_fla_kernels.py
@tests/models/test_plant_helixseek_registry.py
@dnallm/models/model_info.yaml
@example/notebooks/plant_helixseek_shared/.gitignore
@.gitignore
@pyproject.toml
</context>

<constraints>
- Branch `phs` only. Commit and push; never add attribution trailers (owner rule).
- Every dnallm/ code change ships with pytest coverage in the same change (Tasks 1-2 carry
  new tests; Task 4's data change ships with the committed registry structure tests re-run
  green in the same change — they pin the label content this edit must not alter).
- Never add a conftest.py under tests/examples/.
- No refactors beyond the six fixes; no new dependencies; no new test frameworks.
- model_info.yaml edit discipline: byte-identical prefix for everything ABOVE the Anno
  entry block — the edit is confined to the single `label_names` line inside the final
  appended entry (the file's last entry, ending `threshold: 0.5` + one trailing newline,
  both preserved). Verify the scoping with the one-line-diff gate in Task 4 before
  committing.
- All commands assume cwd at repo root; use the project venv binaries explicitly
  (`.venv/bin/python -m pytest ...`, `.venv/bin/python -m ruff ...`).
- Do NOT run tests/models/test_plant_helixseek_smoke.py at all during this task — its slow
  tests download ~1.9 GB checkpoints each and none of the six findings touch it.
- Tasks 1-3 all edit tests/utils/test_genomic_coords.py (Tasks 1-2 also edit
  dnallm/utils/genomic_coords.py): execute strictly in order, one commit per task, re-run
  the file's suite after each.
</constraints>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1 (IN-01): fetch_sequence path branch — deterministic handle release + .fxi sidecar cleanup</name>
  <files>dnallm/utils/genomic_coords.py, tests/utils/test_genomic_coords.py</files>
  <behavior>
    - Test: after `fetch_sequence(str(fa_path), "Chr1", 10, 20)` on a fresh FASTA, the
      correct slice is returned AND no `mini.fas.fxi` exists in the tmp dir afterward AND
      the FASTA itself is untouched.
    - Test: a PRE-EXISTING `.fxi` sidecar (built in the test by constructing pyfastx.Fasta
      directly, then dropping the reference) SURVIVES a path-branch call — caller/user-owned
      indexes are never deleted.
    - Test (skipif `/proc/self/fd` absent, i.e. non-Linux): the process fd count is
      identical before and after a path-branch call — no leaked handles.
    - Existing tests (including `test_fetch_sequence_one_based_inclusive`, which already
      exercises the path branch at line 187) pass unchanged in behavior.
  </behavior>
  <action>
    Confine the module change to the path branch of `fetch_sequence`
    (genomic_coords.py:150-162) plus its docstring. Live-verified pyfastx 2.3.1 API facts
    (do NOT re-litigate; re-confirm cheaply via the verify command if desired): no
    `close()`/context-manager protocol exists on `pyfastx.Fasta`; reference-drop releases
    its fds deterministically under CPython refcounting; `build_index=False` breaks
    `fetch()` with NameError (random access needs the index) and is therefore NOT an
    option; default construction creates/reuses `<fasta>.fxi`.

    Implementation shape: before constructing, record (a) the pyfastx index path — the
    string form of the caller's fasta path plus the `.fxi` suffix, as a `Path` — and (b)
    whether that sidecar already exists (a pre-existing index belongs to the caller;
    never delete it). Construct `pyfastx.Fasta` from the caller's path (str or Path both
    accepted, as today). Keep the existing `try`/`except (KeyError, NameError)` fetch
    contract byte-for-byte in behavior (unknown-chrom and empty-result ValueErrors
    unchanged), and add a `finally` block that runs only for the path branch: rebind the
    local `fa` name to None so the reference drop releases pyfastx's file handles before
    the function returns, then unlink the sidecar ONLY when this call created it, wrapped
    in `contextlib.suppress(OSError)` so a cleanup failure can never mask the fetch result
    or the documented ValueError (import `contextlib` alongside the existing stdlib
    imports at the top of the module). Caller-supplied open index objects are never closed
    and their sidecars never removed — ownership stays with the caller. Update the
    `fetch_sequence` docstring `fa` arg text to state this contract: the path branch is
    read-only — it releases the index and removes a `.fxi` it created, preserves a
    pre-existing one, and never closes a caller-owned index.

    Same-change tests (owner rule) in the FASTA section of
    tests/utils/test_genomic_coords.py, per the behavior block above; the fd test needs
    `import os` added to the test module and a skipif guard for platforms without
    `/proc/self/fd`. Commit scoped `quick-261003-ryz`, message naming IN-01.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_genomic_coords.py -q && grep -q 'fxi' dnallm/utils/genomic_coords.py && grep -q 'suppress' dnallm/utils/genomic_coords.py && grep -c 'fxi' tests/utils/test_genomic_coords.py | grep -qv '^0$' && .venv/bin/python -m ruff format --check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py && .venv/bin/python -m ruff check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py</automated>
    <fails_when>any test in tests/utils/test_genomic_coords.py fails; the module has no `.fxi` sidecar handling or no suppression guard around cleanup; the new sidecar/preexisting-index/fd tests are missing; ruff fails. Also fails if the unknown-chromosome or empty-result ValueError contract changed (existing guard tests would fail).</fails_when>
  </verify>
  <acceptance_criteria>
    - Path-branch fetch leaves NO `.fxi` beside a fresh FASTA; a pre-existing `.fxi` is
      preserved; pyfastx handles are released before the function returns.
    - Caller-owned open indices are never closed and their sidecars never deleted.
    - Cleanup failure cannot mask fetch results or the documented ValueErrors.
    - The behavior is documented in the `fetch_sequence` docstring.
  </acceptance_criteria>
  <done>fetch_sequence's path branch is side-effect-clean and handle-safe, the three new tests ship in the same commit, and all automated checks above pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2 (IN-02): normalize_chrom accepts only ASCII digits in the bare-numeric branch</name>
  <files>dnallm/utils/genomic_coords.py, tests/utils/test_genomic_coords.py</files>
  <behavior>
    - Test: `normalize_chrom("１")`, `normalize_chrom("２")`, `normalize_chrom("²")`,
      `normalize_chrom("٣")` (full-width, superscript, Arabic-Indic — all `isdigit()` True)
      each raise `ValueError` matching "Unrecognized chromosome" in BOTH styles
      (`tair` default and `ensembl`).
    - Test: positive control — `normalize_chrom("1") == "Chr1"` and
      `normalize_chrom("1", style="ensembl") == "1"` still hold.
    - All existing normalize_chrom tests pass unchanged.
  </behavior>
  <action>
    At genomic_coords.py:72, the bare-numeric branch guard `if name.isdigit():` accepts
    non-ASCII digit strings and silently returns `f"Chr{name}"` (e.g. `"Chr２"` — a
    different chromosome). Change the guard to require BOTH `name.isascii()` AND
    `name.isdigit()` so non-ASCII digits fall through to the existing
    `ValueError(f"Unrecognized chromosome {name!r}.")`. Make NO change to the
    chr-prefixed branch (line 69's `token.isdigit()`): `_CHROM_RE` (line 27) already
    restricts that branch's token to `[0-9]+|[CM]`, i.e. ASCII-only — the review
    confirms only the bare branch needs the fix. Add one clarifying phrase to the
    `normalize_chrom` docstring Args: bare Ensembl-style numerics are ASCII digits.
    Add the dedicated same-change test described in the behavior block (a new
    `test_normalize_chrom_rejects_non_ascii_digits`) rather than only extending the
    existing bad-forms loop, so the regression is self-describing. Commit scoped
    `quick-261003-ryz`, message naming IN-02.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_genomic_coords.py -q && grep -q 'isascii' dnallm/utils/genomic_coords.py && grep -q 'non_ascii_digits' tests/utils/test_genomic_coords.py && .venv/bin/python -m ruff format --check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py && .venv/bin/python -m ruff check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py</automated>
    <fails_when>any test in tests/utils/test_genomic_coords.py fails; the module still lacks the ASCII guard (no `isascii` in the bare-numeric branch); the non-ASCII-digit test is missing; ruff fails.</fails_when>
  </verify>
  <acceptance_criteria>
    - Non-ASCII digit names raise ValueError in both styles instead of being renamed.
    - ASCII behavior (bare numerics, chr-prefixed, organelle C/M pass-through) is
      byte-identical to before.
  </acceptance_criteria>
  <done>The ASCII-digit guard and its test are committed together; the whole test file passes.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3 (IN-03): import-purity test restores the package attribute, not just sys.modules</name>
  <files>tests/utils/test_genomic_coords.py</files>
  <behavior>
    - Test: after `test_module_import_is_pyfastx_free()` runs (called directly),
      `dnallm.utils.genomic_coords is sys.modules["dnallm.utils.genomic_coords"]` —
      identity holds; no second live module object survives the purity test.
    - The purity test itself still proves the same property as before (module importable
      with `sys.modules["pyfastx"] = None` poisoning).
  </behavior>
  <action>
    In `test_module_import_is_pyfastx_free` (tests/utils/test_genomic_coords.py:288-304):
    add `import dnallm.utils` to the module-level imports. Alongside the existing
    `sys.modules` saves, save the package attribute —
    `saved_pkg_attr = getattr(dnallm.utils, "genomic_coords", None)` (the attribute exists
    at import time: dnallm/utils/__init__.py:9 does `from .genomic_coords import (...)`).
    In the `finally` block, after restoring the `sys.modules` entries, restore the package
    attribute: rebind `dnallm.utils.genomic_coords` back to `saved_pkg_attr` when it
    existed; when it did not (defensive branch), delete the rebound attribute instead.
    Add the guard test from the behavior block (new
    `test_import_purity_leaves_single_live_module`) that invokes the purity test function
    directly and then asserts the identity — this fails against the current code (the
    attribute stays rebound to the re-executed module) and passes after the restore.
    Commit scoped `quick-261003-ryz`, message naming IN-03.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_genomic_coords.py -q && grep -q 'saved_pkg_attr' tests/utils/test_genomic_coords.py && grep -q 'single_live_module' tests/utils/test_genomic_coords.py && .venv/bin/python -m ruff format --check tests/utils/test_genomic_coords.py && .venv/bin/python -m ruff check tests/utils/test_genomic_coords.py</automated>
    <fails_when>any test fails; the package-attribute save/restore or the identity guard test is missing; the purity test no longer proves pyfastx-free import; ruff fails.</fails_when>
  </verify>
  <acceptance_criteria>
    - After the purity test runs, exactly one live genomic_coords module object exists and
      the package attribute and the sys.modules entry point at it.
    - No other test in the file is affected.
  </acceptance_criteria>
  <done>The contamination-free purity test and its identity guard test are committed; the whole file passes.</done>
</task>

<task type="auto">
  <name>Task 4 (IN-04): normalize the Anno label_names quotes to double quotes</name>
  <files>dnallm/models/model_info.yaml</files>
  <action>
    First verify live what the current quotes are (the reviewer said single — confirmed by
    the planner, but re-confirm at execution time): the Anno entry's `label_names`
    (model_info.yaml:1657, inside the file's LAST entry) reads
    `label_names: ['O', 'B-CDS', 'I-CDS', 'L-CDS', 'U-CDS', 'B-INTRON', 'I-INTRON',
    'L-INTRON', 'U-INTRON', 'B-UTR5', 'I-UTR5', 'L-UTR5', 'U-UTR5', 'B-UTR3', 'I-UTR3',
    'L-UTR3', 'U-UTR3']` while the CRE entry above uses double quotes. Replace the single
    quotes with double quotes on EXACTLY that one line, preserving member order and
    content verbatim. HARD CONSTRAINTS: everything ABOVE the Anno entry block stays
    byte-identical (this file has no-trailing-newline append semantics and the Anno block
    is the appended tail — the edit must be a quote-only change inside that block); the
    file keeps its single trailing newline after `threshold: 0.5`. SCOPE WARNING
    (planner-verified live): there is a SECOND pre-existing single-quoted label_names at
    model_info.yaml:1447 (the "Plant NT singlebase tRNAPointer" entry,
    `label_names: ['O','B-Intron', ...]`) — it predates Phase 06, sits ABOVE the Anno
    block, and MUST NOT be touched (the review's "rest of the registry uses double
    quotes" is inaccurate for that one legacy line; IN-04 is scoped to the Anno entry
    only). Before committing, gate the scoping: `git diff -U0` on the file must show
    exactly one `-` line and one `+` line (excluding the `---`/`+++` headers), and the
    `+` line must be the Anno label_names line. Then re-run the committed fast-leg
    registry structure tests — they assert label CONTENT (`ANNO_LABELS` equality,
    `test_anno_entry_matches_frozen_order`) not quote style, and must pass unchanged;
    that re-run is this dnallm/ change's same-change pytest coverage (owner rule). Commit
    scoped `quick-261003-ryz`, message naming IN-04.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/models/test_plant_helixseek_registry.py -q && .venv/bin/python -c "import yaml; d = yaml.safe_load(open('dnallm/models/model_info.yaml')); print('yaml parses,', sum(len(m.get('task', {}).get('label_names', [])) for m in d['models'] if isinstance(m, dict)), 'label entries')" && ! grep -qE "label_names: \['.*U-UTR3" dnallm/models/model_info.yaml && grep -qF "label_names: ['O','B-Intron'" dnallm/models/model_info.yaml && test "$(grep -cF 'label_names: ["' dnallm/models/model_info.yaml)" -eq 147 && test "$(git diff -U0 -- dnallm/models/model_info.yaml | grep -cE '^[+-][^+-]')" -le 2</automated>
    <fails_when>any registry structure test fails; the YAML no longer parses; the Anno label_names line is still single-quoted; the legacy tRNAPointer line 1447 was modified (prefix property broken — its single-quoted form must survive verbatim); the double-quoted count is not exactly 147; the pre-commit diff gate (run before committing, per the action) shows more than the one label_names line changed.</fails_when>
  </verify>
  <acceptance_criteria>
    - The Anno label_names line uses double quotes, content and order identical.
    - Zero other bytes in model_info.yaml changed (one-line diff gate).
    - Registry structure tests pass and the YAML parses.
  </acceptance_criteria>
  <done>The quote-normalized registry entry is committed with the one-line diff gate and green registry tests.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 5 (IN-05): match the fla extra by exact bracket member, not substring</name>
  <files>tests/models/test_plant_helixseek_fla_kernels.py</files>
  <behavior>
    - Parser unit tests (pure string parsing, no tomllib, no skipif):
      `_meta_extra_names("dnallm[base,dev,test,notebook,docs,ui,mcp,fla]")` contains
      `"fla"` as a member; `"dnallm[base,flash-attn]"` and
      `"dnallm[base,fla-core,mamba-fla]"` do NOT; a whitespace variant
      `"dnallm[base, fla ]"` does; non-meta specs (`"fla"`, `"torch>=2.4.0,<2.12"`)
      return an empty list.
    - `test_fla_reachable_from_all` passes against the real pyproject (`all` is exactly
      `["dnallm[base,dev,test,notebook,docs,ui,mcp,fla]"]`, pyproject.toml:127-129) and
      would now FAIL if the fla member were replaced by a substring-colliding name.
  </behavior>
  <action>
    In tests/models/test_plant_helixseek_fla_kernels.py, `test_fla_reachable_from_all`
    (lines 50-52) currently asserts `any("fla" in spec for spec in extras["all"])` — a
    substring match a future `"flash-attn"`-style spec would satisfy vacuously. Add a
    module-level helper `_meta_extra_names(spec)` that fullmatches the self-referential
    spec shape `dnallm[...]` with a regex (e.g. anchored `dnallx`-style — precisely:
    literal `dnallm[`, capture everything up to the closing `]`, literal `]` at end of the
    stripped spec), splits the captured interior on commas, strips whitespace from each
    member, drops empties, and returns the member list; any spec that does not match the
    shape returns an empty list. Rewrite the assertion to require `"fla"` to be a member
    of the parsed list of at least one spec in `extras["all"]`. Add the parser regression
    tests from the behavior block in a new test class (fast, no tomllib needed — the
    parser tests carry no skipif; the two pyproject-reading tests KEEP their existing
    WR-05 tomllib guard untouched). Commit scoped `quick-261003-ryz`, message naming IN-05.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/models/test_plant_helixseek_fla_kernels.py -q && grep -q '_meta_extra_names' tests/models/test_plant_helixseek_fla_kernels.py && ! grep -qF '"fla" in spec' tests/models/test_plant_helixseek_fla_kernels.py && .venv/bin/python -m ruff format --check tests/models/test_plant_helixseek_fla_kernels.py && .venv/bin/python -m ruff check tests/models/test_plant_helixseek_fla_kernels.py</automated>
    <fails_when>any test in the file fails; the substring assertion survives anywhere in the file; the parser helper or its collision regression tests are missing; ruff fails.</fails_when>
  </verify>
  <acceptance_criteria>
    - The fla-reachability guard does exact member matching on the parsed bracket list.
    - Substring-colliding extra names provably do not satisfy the guard (regression tests).
    - The tomllib 3.10 guard on pyproject-reading tests is untouched.
  </acceptance_criteria>
  <done>The exact-member assertion and parser regression tests are committed; the file passes.</done>
</task>

<task type="auto">
  <name>Task 6 (IN-06, verify-then-decide): settle the .scratch/ ignore redundancy by live check-ignore evidence, then close the dispositions</name>
  <files>example/notebooks/plant_helixseek_shared/.gitignore, .planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md</files>
  <action>
    IN-06 is VERIFY-THEN-DECIDE: resolve by live evidence, not by the reviewer's claim
    (root covers it) or the 06-01 plan's claim (root does not cover .py files there).
    Planner pre-verified the chain on 2026-10-03 — the executor MUST reproduce it and
    commit the command + output verbatim in the SUMMARY (required for either outcome):

    1. Unmasked control: `git check-ignore -v example/notebooks/plant_helixseek_shared/.scratch/foo.py`
       — resolves to `example/notebooks/plant_helixseek_shared/.gitignore:1:.scratch/`
       (the local file wins by depth precedence; exit 0).
    2. Masked probe: copy the local .gitignore to a byte backup outside the repo, `mv` the
       file aside, re-run the same `git check-ignore -v` for BOTH
       `example/notebooks/plant_helixseek_shared/.scratch/foo.py` and the bare directory
       path — both resolve to `.gitignore:60:.scratch/` (root, unanchored, exit 0).
       Restore the file (`mv` back) and `cmp` against the backup to prove the probe was
       non-destructive. This is the decisive fact: an unanchored directory pattern ignores
       ALL contained files regardless of extension, so the 06-01 claim that root patterns
       do not cover `.py` files inside `.scratch/` is disproven.
    3. Decision rule (per the task contract): the root patterns already cover it, so
       REMOVE the local entry. The entry is the file's only line, so remove the file:
       `git rm example/notebooks/plant_helixseek_shared/.gitignore`. Do NOT touch root
       `.gitignore:60` (it carries the owner-rule comment "One-off census/probe scripts
       (owner rule: never committed)"). Post-removal, re-run the check-ignore — it must
       still resolve via root `.gitignore:60` with exit 0, and `git status --porcelain`
       under that directory must show only the .gitignore deletion (no scratch files
       becoming visible/untracked under .scratch/).
    4. Do NOT rewrite 06-01-PLAN.md / 06-01-SUMMARY.md (phase history is immutable); the
       quick SUMMARY records the correction with the committed evidence.

    Closure bookkeeping (same final commit): update
    .planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md —
    flip rows IN-01 through IN-06 from `open` to `fixed`, each citing its per-finding
    commit hash and a one-line description in the Source cell (same format as the WR
    rows), and update the frontmatter `disposition:` values and the `open:` counter to 0.
    Commit scoped `quick-261003-ryz`, message naming IN-06 + closure; push branch `phs`
    (all six commits) with no attribution trailers.
  </action>
  <verify>
    <automated>git check-ignore -v example/notebooks/plant_helixseek_shared/.scratch/foo.py && ! test -f example/notebooks/plant_helixseek_shared/.gitignore && git status --porcelain example/notebooks/plant_helixseek_shared/ | grep -qE '^.?D .*\.gitignore$' && git diff HEAD -- .gitignore | grep -cE '^[+-][^+-]' | grep -qx 0 && test "$(grep -cE '^\| IN-0[1-6] \| info \| fixed \|' .planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md)" -eq 6 && test "$(grep -c '^open: 0$' .planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md)" -eq 1</automated>
    <fails_when>check-ignore exits nonzero or no longer resolves after the removal (the scratch home would become committable — the removal must be reverted and the KEEP outcome recorded instead, with the same evidence); root .gitignore lost its `.scratch/` line; the local file still exists; any file under .scratch/ shows up as trackable; the disposition rows are not all flipped with open: 0.</fails_when>
  </verify>
  <acceptance_criteria>
    - The masked/unmasked `git check-ignore -v` command+output pairs are committed
      verbatim in the SUMMARY (mandatory for either outcome).
    - The decided outcome is implemented (local entry removed BECAUSE root provably
      covers it) — or, if the executor's live evidence contradicts the planner's, the
      KEEP outcome with the evidence and a `skipped`-with-reason disposition instead.
    - Root .gitignore untouched; scratch home stays ignored; disposition rows IN-01..IN-06
      all `fixed` with commit references; `open: 0`.
  </acceptance_criteria>
  <done>IN-06 is decided by committed evidence, the redundant local ignore is removed (or kept, per the executor's reproduced evidence), all six disposition rows reference their fix commits, and branch phs is pushed without attribution trailers.</done>
</task>

</tasks>

<verification>
1. Per-task suites green (run after each task, and once more at the end):
   `.venv/bin/python -m pytest tests/utils/test_genomic_coords.py tests/models/test_plant_helixseek_registry.py tests/models/test_plant_helixseek_fla_kernels.py -q`
2. Lint/format on every changed Python file:
   `.venv/bin/python -m ruff format --check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py tests/models/test_plant_helixseek_fla_kernels.py && .venv/bin/python -m ruff check dnallm/utils/genomic_coords.py tests/utils/test_genomic_coords.py tests/models/test_plant_helixseek_fla_kernels.py`
3. model_info.yaml scoping gate (Task 4): one-line diff, registry tests green, YAML parses.
4. IN-06 evidence chain (Task 6): both check-ignore outputs committed in the SUMMARY;
   post-removal ignore resolution still via root `.gitignore:60`.
5. `git log --oneline -8` shows six atomic fix/test commits scoped `quick-261003-ryz`
   (plus the plan docs commit), `git status` clean, branch `phs` pushed, no attribution
   trailers anywhere (owner rule).
</verification>

<success_criteria>
- IN-01 closed: path-branch fetch leaves no `.fxi` it created, preserves pre-existing
  indexes, releases handles before returning; three same-change tests.
- IN-02 closed: non-ASCII digit names raise ValueError in both styles; ASCII behavior
  unchanged; same-change test.
- IN-03 closed: the import-purity test restores the package attribute; identity guard
  test proves a single live module object.
- IN-04 closed: Anno label_names double-quoted; one-line diff; registry structure tests
  green in the same change.
- IN-05 closed: fla reachability asserted by exact bracket-member parse with
  substring-collision regression tests.
- IN-06 closed: decided by committed check-ignore evidence (planner evidence says root
  covers it, so the local entry is removed); either outcome carries the evidence.
- 06-REVIEW-DISPOSITION.md: all six IN rows `fixed`, `open: 0`.
</success_criteria>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| user-supplied FASTA paths -> fetch_sequence | dnallm code deletes a filesystem object derived from untrusted input |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-ryz-01 | Tampering / DoS | fetch_sequence sidecar cleanup (IN-01) | medium | mitigate | unlink targets ONLY the exact `<fasta-path>.fxi` sidecar string, only when this call observed it absent before constructing, wrapped in `contextlib.suppress(OSError)`; the user's FASTA itself is never touched and pre-existing indexes are never deleted |
| T-ryz-02 | Information disclosure / repudiation | local .gitignore removal (IN-06) | low | mitigate | removal gated on live `git check-ignore` proof that root `.gitignore:60` still ignores the scratch home (post-removal re-check required); evidence committed in the SUMMARY so the decision is auditable |
</threat_model>

<output>
Create `SUMMARY.md` beside this PLAN.md when done — it MUST embed the IN-06 check-ignore
command+output pairs verbatim (mandatory committed evidence for either outcome), list the
six per-finding commits, and note the 06-01 plan-claim correction. Commit sequence
(branch `phs`, no attribution trailers): `docs(quick-261003-ryz): plan IN-01..06 closure`
(this PLAN.md), then one atomic commit per task in order 1-6 with scope
`quick-261003-ryz`, then push.
</output>
