---
phase: 261003-jpr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/utils/transformers_compat.py
  - tests/utils/test_transformers_compat.py
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md
autonomous: true
requirements:
  - IN-01
estimate:
  tokens: 12000
  raw_tokens: 12000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - With transformers unimportable (simulated by a None sys.modules entry), every one of the nine `_patch_*` installers in dnallm/utils/transformers_compat.py returns None without raising — including `_patch_pretrained_config_legacy_defaults`, `_patch_mamba_cache`, and `_patch_legacy_init_weights_bookkeeping` (IN-01 core).
    - Calling `apply_patches()` completes without raising under the same condition — the module-docstring contract "importing DNALLM never breaks an otherwise working environment" (transformers_compat.py:7-11) holds for the whole installer chain, which runs eagerly at package import.
    - Behavior on the installed transformers 5.17.0 is unchanged — all 76 pre-existing contract tests pass unmodified (a guard only adds a failure path that imports never take when transformers is importable).
    - The pattern is pinned forward: the new absence test collects installers dynamically from `vars(transformers_compat)`, and a roster test fails loudly the moment a `_patch_*` installer is added or renamed without consciously extending the roster.
  artifacts:
    - dnallm/utils/transformers_compat.py — the three bare transformers imports (lines 553, 764, 990 at planning-time HEAD) wrapped in the module's standard `except Exception` → `return` guard.
    - tests/utils/test_transformers_compat.py — new class `TestTransformersAbsenceContract` with 11 test items (9 parametrized installer cases + roster pin + apply_patches survival), red on current code.
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md — IN-01 entry flipped from open to fixed (front-matter findings list, markdown table row with the fix-commit hash, open count 3 → 2).
  key_links:
    - Eager import at dnallm/utils/__init__.py:8 (`from . import transformers_compat`) → module-level `apply_patches()` call (transformers_compat.py:1030) → per-installer guard — the chain that must never raise at `import dnallm` time.
    - New test `monkeypatch.setitem(sys.modules, "transformers", None)` → each installer's try-import raises ModuleNotFoundError (parent-first resolution; live-probed at planning time for all import forms the nine installers use) → the guard's `except Exception` → `return`.
---

<objective>
Close IN-01 from the Phase 05 incremental review (05-REVIEW.md:234-247, incremental commit a943411): three
patch installers in dnallm/utils/transformers_compat.py import transformers submodules bare, breaking the
module-docstring contract that every patch no-ops when transformers is absent or lacks the target
("importing DNALLM never breaks an otherwise working environment"). The module runs `apply_patches()` at
import time and dnallm/utils/__init__.py:8 imports it eagerly, so a bare import that fails (transformers
not installed, or a future transformers renaming `configuration_utils` / `cache_utils` / `modeling_utils`)
crashes `import dnallm` outright.

Live-verified at planning time on branch phs HEAD (note: the dispatch description's parenthetical naming —
"get_extended_attention_mask from 261002-se3, get_head_mask" — was garbled; those two installers ALREADY
have the guard at lines 472-475 and 919-922, and must NOT be touched). The three actually-unguarded
installers, matching the review's line numbers exactly, all landed via quick task 261002-sl7:

- `_patch_pretrained_config_legacy_defaults` — line 553: `import transformers.configuration_utils`
- `_patch_mamba_cache` — line 764: `import transformers.cache_utils`
- `_patch_legacy_init_weights_bookkeeping` — line 990: `import transformers.modeling_utils`

Fix per the reviewer's instruction: wrap each in the same guard `_patch_remote_code_pruning_helpers` uses
(lines 363-366), and extend the contract tests to pin the pattern. The review classifies this as
info-severity consistency/robustness hardening — transformers is a hard dependency today, so nothing
changes on the present path; the fix hardens the absent/renamed-transformers path only.

Purpose: restore the module's own documented contract for every installer, and make its violation
impossible to reintroduce silently.

Output: guards on the three installers + a dynamically-collected absence contract test (red first, then
green), committed atomically per the owner rule that every dnallm/ change ships with pytest coverage in
the same change; IN-01 marked fixed in the 05 disposition ledger (the established CR-01/WR-01 close-out
pattern).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/utils/transformers_compat.py
@tests/utils/test_transformers_compat.py
@.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md

Key code points (all verified live at planning time, branch phs; line numbers exact at HEAD):

- dnallm/utils/transformers_compat.py — 1030 lines; module docstring lines 7-11 state the absence
  contract; `apply_patches()` at 1015-1025 calls the nine installers in order; module body calls it at
  line 1030. dnallm/utils/__init__.py:8 imports the module eagerly.
- The exemplar guard (`_patch_remote_code_pruning_helpers`, 363-366): try / `import
  transformers.modeling_utils` / `except Exception:  # pragma: no cover - transformers not installed` /
  `return`. Second guard at 372-375 shows the variant comment `# pragma: no cover - module absent from
  this transformers`. DeBERTa guard at 806-811 shows `# pragma: no cover - transformers not installed /
  module renamed`. Match these comment styles verbatim (see Task 2).
- The three unguarded installers (ONLY these get guards): 553, 764, 990 as listed in the objective. Each
  begins with the bare submodule import as its FIRST transformers touch, so wrapping it changes nothing
  when the import succeeds. `_patch_legacy_init_weights_bookkeeping` also calls
  `_post_init_computes_tied_weights_keys()` (956-976), but that helper has its own internal try/except →
  False and is only reached after the now-guarded import succeeds — leave it untouched.
- Already-compliant installers (do not modify): `_patch_get_parameter_or_buffer` (178-181),
  `_patch_initialize_weights_for_quantized_missing` (247-250), `_patch_remote_code_pruning_helpers`
  (363-375), `_patch_get_extended_attention_mask` (472-475), `_patch_deberta_vocab_dict` (806-811),
  `_patch_get_head_mask` (919-922).
- tests/utils/test_transformers_compat.py — 1291 lines, 76 tests, runs in ~3.5s. Everything the new
  class needs is already imported: `sys` (line 13), `pytest` (19), `transformers_compat` module object
  (line 30). No new imports required.
- The sys.modules-None mechanism is the file's established pattern: line 372-373 sets
  `monkeypatch.setitem(sys.modules, "bitsandbytes", None)` with the comment "None in sys.modules makes
  `import bitsandbytes` raise ImportError." Live-probed at planning time: setting
  `sys.modules["transformers"] = None` makes ALL import forms the nine installers use raise
  ModuleNotFoundError (a subclass of ImportError) via parent-first resolution — `import
  transformers.modeling_utils` / `.configuration_utils` / `.cache_utils` ("'transformers' is not a
  package"), `from transformers import ...` ("import of transformers halted; None in sys.modules"), and
  `from transformers.models.deberta_v2... import ...`. monkeypatch restores the entry on teardown, and
  the installers are all idempotent-sentinel-gated, so the test never disturbs the live patched classes
  (the test module's own docstring contract, lines 3-7).
- Baseline: `.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q` → 76 passed in ~3.5s;
  `ruff check` and `ruff format --check` clean on both files. Run tests ONLY via `.venv/bin/python -m
  pytest` from the repo root — NEVER `uv run pytest` (known resolver failure, 05-02 deferred item).
- Disposition ledger: .planning/phases/05-execution-harness-honest-gates-runner-feasibility/
  05-REVIEW-DISPOSITION.md — IN-01 appears in the front-matter findings list (`disposition: open`,
  severity info), in the markdown table (`| IN-01 | info | open | - |`), and in the front-matter count
  `open: 3` (total: 15). The established close-out pattern (CR-01 → 032b308, WR-01 → 3fe80bf) is a fix
  commit followed by a separate docs commit updating all three places with the fix commit hash.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED — absence-contract tests proving the three bare imports crash apply_patches</name>
  <files>tests/utils/test_transformers_compat.py</files>
  <behavior>
    - RED against current code: append one new class `TestTransformersAbsenceContract` (Google-style
      class docstring citing IN-01 / 05-REVIEW.md:234 and explaining the mechanism: a None entry for
      "transformers" in sys.modules makes every transformers import form raise ModuleNotFoundError via
      parent-first resolution — live-probed; the guards' broad `except Exception` is what the module
      contract at transformers_compat.py:7-11 promises for this case).
    - Module-level helper `_collect_patch_installers()`: return sorted names from
      `vars(transformers_compat)` where the name starts with `_patch_` and the value is callable —
      dynamic collection is deliberate: a future installer that forgets the guard is caught by this
      test automatically, not by the next reviewer.
    - Module-level frozenset `EXPECTED_PATCH_INSTALLERS` holding exactly the nine current names:
      `_patch_get_parameter_or_buffer`, `_patch_initialize_weights_for_quantized_missing`,
      `_patch_remote_code_pruning_helpers`, `_patch_get_extended_attention_mask`,
      `_patch_pretrained_config_legacy_defaults`, `_patch_mamba_cache`, `_patch_deberta_vocab_dict`,
      `_patch_get_head_mask`, `_patch_legacy_init_weights_bookkeeping`.
    - Test 1 `test_installer_noops_when_transformers_unimportable`, parametrized over
      `_collect_patch_installers()`: `monkeypatch.setitem(sys.modules, "transformers", None)` then
      `assert getattr(transformers_compat, installer_name)() is None`. Nine items: the six guarded
      installers pass today; the three IN-01 installers FAIL with ModuleNotFoundError — that is the
      red proof.
    - Test 2 `test_installer_roster_is_pinned`: `assert set(_collect_patch_installers()) ==
      EXPECTED_PATCH_INSTALLERS` — passes today; exists so adding/renaming an installer fails loudly
      here until the roster (and the absence contract) is consciously extended.
    - Test 3 `test_apply_patches_survives_unimportable_transformers`: under the same monkeypatch,
      `assert apply_patches() is None` — FAILS today (raises ModuleNotFoundError at the fourth
      installer, `_patch_pretrained_config_legacy_defaults`) because apply_patches runs eagerly at
      `import dnallm` time via dnallm/utils/__init__.py:8 and transformers_compat.py:1030.
    - Expected RED tally: `.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q`
      reports 4 failed (3 parametrized items + Test 3), 83 passed (76 pre-existing + 6 guarded
      parametrized items + Test 2), 87 items total.
  </behavior>
  <action>
Append the helper, the roster frozenset, and the class per the behavior block, following the file's
conventions: Google-style docstrings on the class and each test explaining WHY the contract exists
(eager apply_patches at import time; a future transformers renaming a submodule must degrade to a
no-op patch, never crash `import dnallm`), English comments only, ruff line-length 100, no new module
imports (sys, pytest, transformers_compat are already imported). Do not modify any existing test. Do
NOT commit yet — the red run is evidence; tests + fix land as one atomic commit in Task 2.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q</automated>
  </verify>
  <done>
New class with 11 test items exists; the run reports exactly 4 failed / 83 passed — the 3 IN-01
installers fail with ModuleNotFoundError, `test_apply_patches_survives_unimportable_transformers`
fails at the fourth-installer raise, and all 76 pre-existing tests plus the roster test and the 6
guarded-installer items pass. Nothing committed yet.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: GREEN — guard the three bare imports; ledger IN-01 fixed; atomic commits</name>
  <files>dnallm/utils/transformers_compat.py, tests/utils/test_transformers_compat.py, .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md</files>
  <behavior>
    - GREEN: all 87 items in tests/utils/test_transformers_compat.py pass — the three installers return
      None under the None-transformers monkeypatch and apply_patches survives it.
    - All 76 pre-existing tests pass byte-identical (guards only add a path imports never take when
      transformers is importable — the live attach/sentinel/idempotence state on transformers 5.17.0 is
      untouched).
    - The roster test, the six guarded-installer items, and every other existing behavior are unchanged.
  </behavior>
  <action>
In dnallm/utils/transformers_compat.py, wrap each of the three bare imports in the module's standard
guard — try, the existing import statement unchanged, `except Exception` with a `# pragma: no cover`
comment, `return` — exactly the shape of the exemplar at lines 363-366:

1. `_patch_pretrained_config_legacy_defaults` (line 553, `import transformers.configuration_utils`):
   comment `# pragma: no cover - transformers not installed / module renamed` (the DeBERTa-guard
   variant at 806-811 — a future transformers renaming this submodule is the exact failure mode IN-01
   describes).
2. `_patch_mamba_cache` (line 764, `import transformers.cache_utils`): same comment variant.
3. `_patch_legacy_init_weights_bookkeeping` (line 990, `import transformers.modeling_utils`): comment
   `# pragma: no cover - transformers not installed` — verbatim match with the two existing
   modeling_utils guards (363-366, 472-475).

Change nothing else: the already-guarded installers (including `_patch_get_extended_attention_mask`
and `_patch_get_head_mask`, which the dispatch description misnamed — they already comply), the
`_post_init_computes_tied_weights_keys` helper, every gate/sentinel/setattr body, `apply_patches`,
and the module docstring all stay byte-for-byte. Keep the `# pragma: no cover` markers even though the
new tests exercise these except branches — matching the file's established style (the bitsandbytes
guard at line 274 is likewise pragma-marked while covered by
`test_passthrough_when_bitsandbytes_unavailable`).

Then verify and commit:

4. Run `.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q` — 87 passed; then
   `.venv/bin/python -m ruff check dnallm/utils/transformers_compat.py
   tests/utils/test_transformers_compat.py` and the same with `ruff format --check` — both clean.

5. Commit BOTH code files as one atomic commit (owner rule: dnallm/ change ships with its pytest
   coverage in the same change), message:
   `fix(quick-261003-jpr): guard the three unguarded transformers_compat patch installers (IN-01)`
   No attribution trailers of any kind.

6. Update the disposition ledger .planning/phases/05-execution-harness-honest-gates-runner-feasibility/
   05-REVIEW-DISPOSITION.md in all three places: the front-matter findings entry for IN-01
   (`disposition: open` → `fixed`), the markdown table row (`| IN-01 | info | fixed | <fix-commit-hash>
   fix(quick-261003-jpr) |`), and the front-matter count (`open: 3` → `open: 2`). Commit separately as
   `docs(quick-261003-jpr): IN-01 marked fixed in 05 disposition ledger` — the established CR-01/WR-01
   two-commit close-out pattern.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q && .venv/bin/python -m ruff check dnallm/utils/transformers_compat.py tests/utils/test_transformers_compat.py && .venv/bin/python -m ruff format --check dnallm/utils/transformers_compat.py tests/utils/test_transformers_compat.py</automated>
  </verify>
  <done>
87 passed in the contract file (76 pre-existing unmodified + 11 new); each of the three installers
shows the try/except-Exception guard wrapping its previously bare import with the specified pragma
comment, and no other code in the module changed; ruff check and format clean; ledger shows IN-01
fixed with `open: 2`; two commits landed (fix commit with both code files, docs commit with the
ledger), no attribution trailers.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| dependency environment → package import surface | `import dnallm` → dnallm/utils/__init__.py:8 → transformers_compat module body → apply_patches() at line 1030; an unimportable/renamed transformers submodule crosses this boundary at import time |
| test process → sys.modules | the new tests null the "transformers" sys.modules entry via monkeypatch (restored on teardown) to simulate the absent-transformers environment |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-quick-in01-01 | Denial of Service | apply_patches() bare transformers imports (transformers_compat.py:553/764/990) | low | mitigate | The three guards make an absent/renamed transformers submodule degrade to a no-op patch instead of crashing `import dnallm` wholesale; pinned by the parametrized absence test (dynamically collected over all `_patch_*` installers) and the apply_patches survival test. Severity low per the review's own classification: transformers is a hard dependency today, so the failure mode requires a stripped environment or a future rename (05-REVIEW.md:237-243). |
| T-quick-in01-02 | Repudiation | broad `except Exception` swallowing non-ImportError failures during patch install | low | accept | Matches the module-wide established guard pattern (six existing guards do exactly this); a silently skipped patch only degrades to stock-transformers behavior, which the affected trust_remote_code workflow surfaces immediately and loudly — the exact trade the module contract chose. |
| T-quick-in01-SC | Tampering | package installs | high | accept | No pip/npm/cargo install tasks in this plan — stdlib-only guard edits plus test/ledger edits, zero new dependencies, so no supply-chain surface is introduced. |
</threat_model>

<verification>
- `.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q` — 87 passed (was 76 at baseline; Task 1 observed 4 failed / 83 passed as the red proof).
- `.venv/bin/python -m ruff check dnallm/utils/transformers_compat.py tests/utils/test_transformers_compat.py` and `ruff format --check` on the same files — clean (both clean at baseline too).
- `git log --oneline -3` shows the fix(quick-261003-jpr) commit touching exactly dnallm/utils/transformers_compat.py + tests/utils/test_transformers_compat.py, followed by the docs(quick-261003-jpr) ledger commit.
- The absence contract is structurally self-pinning: the parametrized test iterates `vars(transformers_compat)` at collection time, so any future `_patch_*` installer is inside the absence test automatically, and the roster test forces conscious extension when the installer set changes.
</verification>

<success_criteria>
- IN-01 closed: every patch installer in dnallm/utils/transformers_compat.py no-ops when transformers is unimportable — proven by tests that are red on the old code (three installers raise ModuleNotFoundError; apply_patches dies at the fourth installer) and green after (all return None, apply_patches survives).
- The present path is byte-identical on transformers 5.17.0: all 76 pre-existing contract tests pass unmodified.
- The module-docstring contract ("importing DNALLM never breaks an otherwise working environment") is now enforced by the test suite for every installer, current and future, via dynamic collection plus the pinned roster.
- Ledger updated (IN-01 fixed, open: 2) and the whole change committed per the owner rule: dnallm/ code modification ships with pytest coverage in the same change, no attribution trailers.
</success_criteria>

<output>
Create `.planning/quick/261003-jpr-fix-in-01-the-three-new-patch-installers/261003-jpr-SUMMARY.md` when done
</output>
