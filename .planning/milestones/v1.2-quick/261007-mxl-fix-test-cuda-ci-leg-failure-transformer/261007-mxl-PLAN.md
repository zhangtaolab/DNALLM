---
phase: quick-261007-mxl
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/utils/transformers_compat.py
  - dnallm/__init__.py
  - tests/utils/test_transformers_compat_device.py
autonomous: true
requirements:
  - QUICK-261007-MXL-01
user_setup: []

estimate:
  tokens: 17000
  raw_tokens: 11000
  tasks: 1
  confidence: high

must_haves:
  truths:
    - "import dnallm succeeds on a CUDA-built torch with no visible GPU under transformers 5.19+: the transformers.utils.import_utils device-type query answers 'cpu' instead of raising RuntimeError('Cannot access accelerator device when none is available'), so the 37 collection errors on the test-cuda CI legs disappear"
    - "On transformers without get_device_type (5.17 local, verified absent) the new rung is a pure no-op — transformers.utils.import_utils is left byte-identically untouched"
    - "On environments where the native device query succeeds (GPU machines like the dev box and nightly runner, CPU-only torch wheels like the passing CPU legs) the probe keeps the rung from installing anything, so behavior is identical"
    - "dnallm.utils — and therefore transformers_compat.apply_patches() — is imported before dnallm.models triggers the first transformers modeling_utils resolution, for every import dnallm / import dnallm.* entry point (Python always executes the root package __init__ first)"
    - "apply_patches() registers the new rung BEFORE any rung that imports transformers.modeling_utils (the first today is _patch_get_parameter_or_buffer)"
    - "No transformers version pin or upper bound is introduced anywhere (the open >=4.49,<6 span is preserved; compat lives in transformers_compat.py)"
  artifacts:
    - "dnallm/utils/transformers_compat.py gains _patch_device_type_query (absence-gated on get_device_type, probe-gated via _device_type_query_broken, idempotent via the _dnallm_device_type_patch module sentinel, installing a RuntimeError-to-'cpu' wrapper with *args/**kwargs passthrough and an identity-guarded transformers.utils re-export mirror), registered as the FIRST call in apply_patches(), preceded by the file-convention forensic comment block documenting the 5.19 import chain and the remote CI evidence"
    - "dnallm/__init__.py imports from .utils BEFORE from .models, with a short comment recording the pre-modeling patch ordering contract"
    - "tests/utils/test_transformers_compat_device.py holds the contract tests covering installer registration and ordering, absence no-op, native-success no-op, install-on-RuntimeError, delegation and argument passthrough, idempotency, loud propagation of non-RuntimeError probe failures, the identity-guarded re-export mirror, and the root-init source ordering"
  key_links:
    - "dnallm/__init__.py (.utils import moved above .models) -> dnallm/utils/__init__.py line 8 transformers_compat -> module-level apply_patches() (transformers_compat.py:1128) -> _patch_device_type_query as first rung -> transformers.utils.import_utils.get_device_type wrapper -> the later 'from transformers import PreTrainedModel, ...' at dnallm/models/model.py:15 (the first modeling_utils resolution in the import graph) no longer crashes on GPU-less CUDA torch"
    - "tests/utils/test_transformers_compat_device.py <-> installer internals: fake sys.modules['transformers.utils.import_utils'] plus a monkeypatched parent-package attribute keep every install/no-op branch deterministic on any host (GPU or CPU, transformers 5.17 or 5.19+), so local proof does not depend on reproducing the GPU-less CUDA environment"
---

<objective>
Fix the test-cuda CI leg collection failure: transformers 5.19.0 (freshly resolved by the open >=4.49,<6 range on remote CI) imports its flex_attention integration eagerly inside modeling_utils (chain: transformers/modeling_utils.py:82 -> integrations/flex_attention.py:46 -> is_torch_flex_attn_available() -> get_device_type() -> torch.accelerator.current_accelerator()), which raises RuntimeError("Cannot access accelerator device when none is available") when a CUDA-built torch runs on a GPU-less machine. The test-cuda legs (ubuntu-latest, torch 2.6.0 cu121/cu124 wheels, no GPU) hit 37 collection errors on every module importing dnallm. CPU-torch legs pass with the same transformers 5.19; the dev box and nightly runner (GPUs, transformers 5.17) never see it.

The fix is the established NT-shim precedent: a new absence-gated, probe-gated rung in dnallm/utils/transformers_compat.py that makes the transformers device-type query import-safe when torch has a compiled CUDA backend but no visible device (honest "cpu" answer instead of raise), PLUS a root-init import reorder — planning verified that the currently claimed ordering (utils before models) does NOT hold in the live tree, so without the reorder the shim loads too late to help.

Purpose: the test-cuda legs have been red at collection since transformers 5.19.0 entered the open range; the project constraint forbids pinning or upper-bounding transformers (span 4.49-5.x), so the repair must land at the compat seam.

Output: one new compat rung (~60 lines with its forensic comment block), a one-line import reorder with comment in dnallm/__init__.py, and a new contract test file (~9 tests) — all in one commit per the owner rule (dnallm/ changes ship with same-change pytest).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@dnallm/utils/transformers_compat.py
@dnallm/__init__.py
@tests/utils/test_transformers_compat_np.py

Live-tree facts observed at planning time (2026-10-07, branch phs, .venv transformers 5.17.0 / torch 2.11.0+cu130 / CUDA available):

- CRITICAL ORDERING CORRECTION — the assumption "transformers_compat is imported eagerly via dnallm/utils/__init__ BEFORE dnallm/__init__.py line 23 imports dnallm/models" is FALSE in the current tree. Verified chain: dnallm/__init__.py imports .configuration (line 22), .models (line 23), .utils (line 27); dnallm/configuration/configs.py imports only os/yaml/typing/pydantic (never dnallm.utils); dnallm/models/__init__.py line 1 loads model.py whose transformers import at line 15 ("from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, BitsAndBytesConfig") precedes "from ..utils import get_logger" at line 19. On a GPU-less CUDA-torch machine the modeling_utils lazy-module resolution at model.py:15 raises BEFORE dnallm.utils could ever load. The fix must therefore BOTH add the shim AND move the .utils import above .models in dnallm/__init__.py.
- Circular-import safety of the reorder (verified): dnallm/utils/*.py outside the two compat files contain zero imports from anywhere inside dnallm (grep for internal relative imports returns nothing) — utils is a pure leaf package (external deps only), so importing it earlier cannot create a cycle. Python always executes the root package __init__ before any dnallm.* submodule, so the reorder covers every entry point including import dnallm.models.modeling_auto and dnallm-mcp-server.
- Second ordering requirement (verified): apply_patches() (transformers_compat.py:1112-1123) itself imports transformers.modeling_utils inside its first rung _patch_get_parameter_or_buffer (line 179). The new rung must be the FIRST call in apply_patches() and must import ONLY transformers.utils.import_utils (a leaf utility module whose import does not resolve modeling_utils; importing it executes the lazy transformers/__init__ plus utils/__init__, both safe).
- transformers 5.17 local shape (the versions without the bug): get_device_type exists NOWHERE in the installed 5.17 (site-packages grep: zero matches) — the absence gate is live-observable. modeling_utils.py:83 already has "from .integrations.flex_attention import flex_attention_forward"; integrations/flex_attention.py:46 calls is_torch_flex_attn_available() at module level; 5.17's is_torch_flex_attn_available (utils/import_utils.py:747) is a pure torch-version check (>= 2.5) — the get_device_type() device query is the 5.19 addition (remote evidence). flex_attention.py reaches it via "from ..utils import is_torch_flex_attn_available", whose body resolves get_device_type through import_utils module globals — so patching the defining module covers the crash chain.
- The availability functions are decorated @lru_cache + @_make_compile_constant (import_utils.py:744-747); replacing the module attribute with a plain wrapper that delegates to the captured original keeps the original's decorators and cache intact (the wrapper only adds a try/except frame).
- Direct precedent to mirror: the numpy-fromstring rung at transformers_compat.py:1024-1109 — probe predicate (_numpy_fromstring_works calls the real function once instead of hasattr), sentinel (_dnallm_fromstring_patch), direct module-attribute assignment (numpy.fromstring = _np_fromstring, line 1108), and the dedicated contract test file tests/utils/test_transformers_compat_np.py (fake-module monkeypatch tests). The new tests follow that file's shape: sys.modules['transformers.utils.import_utils'] gets a types.ModuleType fake AND the parent package attribute is monkeypatched to the same fake so the installer's post-import attribute reference resolves it deterministically.
- Existing suite safety (verified): tests/utils/test_transformers_compat.py and test_transformers_compat_np.py assert apply_patches idempotency and co_names membership but never a full rung-list equality, so adding a first rung breaks no existing test. tests/utils/test_transformers_compat_np.py:36 asserts "_patch_numpy_fromstring" in apply_patches.__code__.co_names — still true.
- CI shape: .github/workflows/ci.yml test-cuda job at line 208 — runs-on ubuntu-latest (no GPU), python 3.11, matrix cuda-version 12.1/12.4, installs -e ".[base,cuda121]" / cuda124 (torch 2.6.0 cu wheels); transformers 5.19.0 was freshly resolved by the open range. The legs run on push/PR, so the next push to phs is the remote proof.
- No local reproduction (per task constraint): the dev box has a GPU and transformers 5.17, both of which mask the bug. Local proof = the contract tests (fake modules simulate the raising path) plus a fresh-import smoke proving the reorder introduced no circular import. Remote evidence only for the real GPU-less CUDA environment.
- Repo conventions: ruff line-length 100, double quotes, English comments; plain submodule imports (e.g. "import transformers.cache_utils" at line 768) need no ty ignore — if scripts/check_code.py flags the new import, add the "# ty: ignore[unresolved-import]" comment form already used at lines 22-23/813/976.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add the probe-gated device-type-query rung, reorder the root init, and hold both with contract tests (one commit)</name>
  <files>dnallm/utils/transformers_compat.py, dnallm/__init__.py, tests/utils/test_transformers_compat_device.py</files>
  <behavior>
    - Test: installer _patch_device_type_query is callable and apply_patches.__code__.co_names lists it, at an index BEFORE _patch_get_parameter_or_buffer (the first modeling_utils-importing rung).
    - Test: against a fake transformers.utils.import_utils lacking get_device_type (the 5.17-and-older shape), the installer sets nothing and leaves no _dnallm_device_type_patch sentinel.
    - Test: against a fake whose get_device_type returns a device string, the attribute is left the identical object and no sentinel is set (GPU machines, CPU-only wheels).
    - Test: against a fake whose get_device_type raises RuntimeError("Cannot access accelerator device when none is available") on every call, the installer sets the sentinel and the installed attribute is a new callable that returns "cpu" while the underlying stub still raises (the wrapper catches per call, mirroring a permanently device-less machine).
    - Test: delegation and passthrough — after install over a stub that raises only on the probe call and then succeeds, calling the installed wrapper with positional and keyword arguments returns the stub's successful value and the stub records the forwarded arguments.
    - Test: a second installer call after install leaves the installed attribute the identical object (sentinel idempotency).
    - Test: a stub raising ValueError (not RuntimeError) propagates out of the installer call — unknown failure states stay loud, never silently answered "cpu".
    - Test: when the real transformers.utils package exposes a get_device_type attribute that IS the fake module's original object, it is mirrored to the wrapper too; when it holds a different object, it is left untouched (identity-guarded re-export mirror).
    - Test: the source of dnallm/__init__.py places the "from .utils import" line before the "from .models import" line (root-init ordering contract, with an explanatory failure message).
  </behavior>
  <action>
    In dnallm/utils/transformers_compat.py, insert a new section after the _patch_numpy_fromstring block (after line 1109, before def apply_patches() at line 1112), following the file's per-rung convention of a forensic comment block plus gate predicate plus installer. The comment block records: transformers 5.19.0 (freshly resolved by the open >=4.49,<6 range on remote CI) eagerly imports its flex_attention integration inside modeling_utils (transformers/modeling_utils.py:82 -> integrations/flex_attention.py:46 -> is_torch_flex_attn_available() -> get_device_type() -> torch.accelerator.current_accelerator()), which raises RuntimeError "Cannot access accelerator device when none is available" when a CUDA-built torch runs on a GPU-less machine — the test-cuda legs (ubuntu-latest, torch 2.6.0 cu121/cu124, no GPU) fail all 37 collecting modules; CPU-torch legs and GPU machines never see it; get_device_type is absent from 5.17 (verified live locally), hence the absence gate; the probe design (one real call, per the 08-03 np.fromstring precedent) so working environments are never patched; and the decision to catch RuntimeError only (the observed torch signature — anything else fails loudly rather than lying "cpu"). Write _device_type_query_broken(module) returning True only when the module's get_device_type() raises RuntimeError, mirroring _numpy_fromstring_works. Write _patch_device_type_query(): import transformers.utils.import_utils inside a try/except returning silently when transformers is absent; resolve the module; no-op when get_device_type is absent (getattr None check — transformers <= 5.17); no-op when the module already carries the _dnallm_device_type_patch sentinel; no-op when the probe succeeds; otherwise capture the original, define a closure get_device_type(*args, **kwargs) that returns the original's result and, on RuntimeError, returns the string "cpu" (the honest no-accelerator answer), assign it to the import_utils module by direct attribute assignment with the sentinel set True (numpy-rung style, lines 1108-1109), and finally mirror: if the transformers.utils package object (already in scope from the same import statement) exposes a get_device_type attribute that is the identical captured original object, overwrite that reference with the wrapper too — never stamp a non-identical attribute. Add _patch_device_type_query() as the FIRST call inside apply_patches(), leaving the ten existing calls untouched and in order, and extend apply_patches' docstring no further than one clause noting the first rung must precede any modeling_utils import. Add ty ignore comments only if scripts/check_code.py reports unresolved-import on the new import, using the file's existing form.

    In dnallm/__init__.py, move the existing line "from .utils import get_logger, setup_logging" (currently line 27) to immediately before "from .models import load_model_and_tokenizer" (currently line 23), keeping the line itself byte-identical, and add a two-to-three-line English comment above it stating the contract: dnallm.utils must load before dnallm.models because transformers_compat installs its pre-modeling device-query patch ahead of the first "from transformers import" modeling-symbol resolution (transformers >= 5.19 queries the accelerator during that import and raises on CUDA-built torch without a visible GPU — the test-cuda CI legs). Change nothing else in the file; the __all__ list and every other import stay untouched.

    Create tests/utils/test_transformers_compat_device.py mirroring tests/utils/test_transformers_compat_np.py: module docstring summarizing the rung and the three-part discipline, imports of sys/types/pytest plus "from dnallm.utils import transformers_compat" and "import transformers.utils", a lazy _patch_fn() resolver, and a TestDeviceTypeQueryShim class implementing every behavior above. Every branch test builds a types.ModuleType fake for transformers.utils.import_utils and installs it BOTH via monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake) AND monkeypatch.setattr on the real transformers.utils package's import_utils attribute, so the installer's import-then-attribute reference resolves the fake deterministically regardless of CPython parent-attribute behavior; the re-export test additionally monkeypatches the real parent package's get_device_type attribute and relies on monkeypatch teardown for full restoration. The RuntimeError stub's message must be the literal remote text "Cannot access accelerator device when none is available". Add a separate test (same file) for the root-init ordering: read dnallm/__init__.py via pathlib, assert both import lines are found and the .utils line's index precedes the .models line's, with an assertion message naming the pre-modeling contract. Follow repo style: ruff line-length 100, double quotes, Google-style docstrings, English comments.

    All three file edits land in ONE commit — the dnallm/ change ships with its same-change pytest per the owner rule.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_transformers_compat_device.py tests/utils/test_transformers_compat_np.py tests/utils/test_transformers_compat.py -q && .venv/bin/python -c "import dnallm; import dnallm.models.modeling_auto; import dnallm.inference; print('fresh-import smoke OK')" && .venv/bin/python scripts/check_code.py</automated>
  </verify>
  <done>
    The new contract file passes all tests and the two existing transformers_compat suites stay green (no co_names or idempotency regressions from the added first rung); the fresh-import smoke proves the root-init reorder introduced no circular import on this GPU box (transformers 5.17: the rung no-ops via the absence gate); scripts/check_code.py passes (ruff format/check across repo including the new test file, plus the ty gate); git diff shows exactly three files changed in one commit — dnallm/utils/transformers_compat.py (new rung + first-position registration), dnallm/__init__.py (import move + comment), tests/utils/test_transformers_compat_device.py (new); no version pin or upper bound on transformers anywhere in the diff; the test-cuda legs themselves are proven on the next push to phs (remote-only environment, evidence per the no-local-reproduction constraint).
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| third-party library surface | The shim monkeypatches transformers.utils.import_utils (import-time availability query) inside the dnallm process |
| CI environment diversity | The patched behavior must stay inert on GPU machines, CPU-only torch wheels, and transformers 4.49-5.18 |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261007-mxl-01 | Tampering | transformers.utils.import_utils.get_device_type monkeypatch surface | medium | mitigate | Absence gate (get_device_type must exist), probe gate (one real call must raise RuntimeError before anything is installed), module sentinel idempotency, identity-guarded re-export mirror, and RuntimeError-only catching keep the wrapper confined to the exact observed crash path — unknown exception types propagate loudly instead of being masked by a "cpu" answer; contract tests pin every install/no-op boundary on fakes |
| T-261007-mxl-02 | Denial of Service | import dnallm on GPU-less CUDA CI legs (37 collection errors) | high | mitigate | Root-init reorder guarantees transformers_compat.apply_patches() runs before the first modeling_utils resolution (dnallm/models/model.py:15) for every dnallm.* entry point; the rung is registered before any modeling_utils-importing rung inside apply_patches; the source-ordering contract test keeps future edits from silently reverting the reorder |

No new packages are installed by this plan (no package-manager install tasks exist), so the package-legitimacy gate and the reserved T-{phase}-SC row do not apply.
</threat_model>

<verification>
- `.venv/bin/python -m pytest tests/utils/test_transformers_compat_device.py -q` passes — every install/no-op branch of the rung is contract-pinned on fake modules (local proof without reproducing the GPU-less CUDA environment, per the remote-evidence-only constraint).
- `.venv/bin/python -m pytest tests/utils/test_transformers_compat_np.py tests/utils/test_transformers_compat.py -q` stays green — the added first rung breaks no existing apply_patches/idempotency contract.
- The fresh-import smoke (`import dnallm; import dnallm.models.modeling_auto; import dnallm.inference`) exits 0 — the root-init reorder introduces no circular import.
- `.venv/bin/python scripts/check_code.py` passes — ruff format/check (line-length 100) and the ty gate accept the new code.
- `git diff --stat` shows exactly the three planned files in one commit; grep of the diff for transformers version constraints confirms no pin or upper bound was added.
- The real GPU-less CUDA environment is proven on the next push to phs: both test-cuda matrix legs (cu121, cu124) collect the suite with zero collection errors — remote runner evidence closes the loop, exactly as the Windows-leg quick task (261007-mhz) was closed.
</verification>

<success_criteria>
- dnallm/utils/transformers_compat.py carries the new _patch_device_type_query rung (absence-gated, probe-gated, sentinel-idempotent, RuntimeError-to-"cpu" with args passthrough, identity-guarded re-export mirror), registered as the FIRST apply_patches() call, with the forensic comment block documenting the 5.19 chain and remote evidence.
- dnallm/__init__.py imports .utils before .models with the ordering-contract comment; the source-ordering test enforces it.
- tests/utils/test_transformers_compat_device.py passes with the full behavior set; existing transformers_compat suites stay green; check_code.py passes.
- No transformers version pin or upper bound introduced; the 4.49-5.x span constraint is preserved.
- Single atomic commit containing all three files per the owner same-change-test rule; test-cuda legs green on the next phs push.
</success_criteria>

<output>
Create `.planning/quick/261007-mxl-fix-test-cuda-ci-leg-failure-transformer/261007-mxl-SUMMARY.md` when done
</output>
