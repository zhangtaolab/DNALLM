# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra - Pattern Map

**Mapped:** 2026-10-21
**Files analyzed:** 27 (18 new, 9 modified)
**Analogs found:** 21 / 27 (6 have no in-repo analog — all Textual-side)

Every path below is git-tracked source except the explicitly marked new files.
Verified with `git ls-files -- <path>`: `dnallm/tui/`, `tests/tui/`,
`dnallm/utils/env_info.py` do not exist yet (new). No gitignored mirrors are
referenced anywhere in this document.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/__init__.py` (mod) | package facade | request-response (lazy attribute resolution) | itself (eager, `:1-34`) | exact (refactor target) |
| `dnallm/models/__init__.py` (mod) | package facade | lazy attribute resolution + ordering guarantee | `dnallm/__init__.py:24-30` (ordering comment only) | partial |
| `dnallm/utils/__init__.py` (mod) | package facade | lazy attribute resolution | `dnallm/__init__.py:20-34` | partial |
| `dnallm/utils/env_info.py` (new) | utility (core collector) | transform (facts → report dict) | `dnallm/utils/support.py:1-25` + `dnallm/preflight/availability.py:173-193` | role-match |
| `dnallm/tui/__init__.py` (new) | package facade / entry | request-response | `dnallm/preflight/__init__.py:1-41` | role-match |
| `dnallm/tui/app.py` (new) | app (composition root) | event-driven | none (no Textual in repo) | none |
| `dnallm/tui/messages.py` (new) | message types | pub-sub (worker → UI) | `dnallm/preflight/report.py:146-163` (frozen typed payloads) | partial |
| `dnallm/tui/workers.py` (new) | service seam | event-driven (thread → message) | `dnallm/models/model.py:411-485` (wraps) + RESEARCH workers example | none (seam), role-match for the wrap |
| `dnallm/tui/screens/env.py` (new) | component (screen) | request-response | `dnallm/preflight/report.py` consumers | none |
| `dnallm/tui/screens/data.py` (new) | component (screen) | request-response | none | none |
| `dnallm/tui/screens/suitability.py` (new) | component (screen) | request-response + event-driven probe | none | none |
| `dnallm/tui/screens/config.py` (new) | component (screen) | request-response (form + validation) | `dnallm/preflight/config_gen.py:382-438` (contract it drives) | none |
| `dnallm/tui/screens/download.py` (new) | component (screen) | streaming (progress) | none | none |
| `dnallm/tui/screens/summary.py` (new) | component (screen) | request-response | none | none |
| `dnallm/tui/styles.tcss` (new) | config (stylesheet) | n/a | none (no `.tcss` in repo) | none |
| `dnallm/cli/cli.py` (mod) | controller (CLI) | request-response | itself `:248-317` (mcp command) | exact |
| `pyproject.toml` (mod) | config (packaging) | n/a | `:146-159` (full/base/all) | exact |
| `.github/workflows/ci.yml` (no edit expected) | config (CI) | batch | `:92,172,261,341,408,543,758` | exact |
| `README.md` (mod) | docs | n/a | `:178` | exact |
| `docs/getting_started/installation.md` (mod) | docs | n/a | `:257` | exact |
| `docs/getting_started/install_honesty.md` (mod) | docs | n/a | `:14-15` | exact |
| `tests/test_extras_guard.py` (mod) | test (contract guard) | batch | itself `:138-172`, `:219-261` | exact |
| `tests/test_package_facade.py` (new) | test (import-purity contract) | batch (subprocess) | `tests/preflight/test_structure.py:1-158` + `tests/preflight/test_validation.py:706-733` | role-match |
| `tests/tui/test_structure.py` (new) | test (AST guard) | batch | `tests/preflight/test_structure.py` | exact |
| `tests/tui/test_workers.py` (new) | test | batch (fake probers) | `tests/preflight/test_compatibility.py:446-470` | role-match |
| `tests/tui/test_screens.py` (new) | test (table test) | batch | `tests/preflight/test_integration.py:1-70` | role-match |
| `tests/tui/test_app.py` (new) | test (Pilot) | event-driven | `tests/mcp/test_timeout.py:1-55` (async idiom only) | partial |
| `tests/utils/test_env_info.py` (new) | test | batch (monkeypatch) | `tests/utils/test_support.py:1-35` | role-match |

Path note: RESEARCH proposes `tests/tui/test_env_info.py`; this repo's test tree
mirrors the package layout (`tests/utils/` already holds every `dnallm/utils/*`
test), and `env_info` is `dnallm/utils/env_info.py` (D-02). Recommend
`tests/utils/test_env_info.py`; the `tests/tui/test_structure.py` guard should
then assert `dnallm/utils/env_info.py` is the ONLY torch-touching module the TUI
may call indirectly (see Shared Patterns → AST guard triad).

---

## Pattern Assignments

### `dnallm/__init__.py` (package facade, lazy resolution)

**Analog:** itself — current eager implementation (`dnallm/__init__.py:1-34`).
No module-level `__getattr__` exists anywhere in `dnallm/` today (grep: only
instance-level `__getattr__` in `dnallm/models/special/mutbert.py:32` and
`dnallm/utils/transformers_compat.py:307`), so the PEP 562 shape comes from
RESEARCH.md, not from in-repo precedent.

**Current shape to refactor** (`dnallm/__init__.py:5-34`):

```python
__all__ = [
    "Benchmark", "DNADataset", "DNAInference", "DNAInterpret", "DNATrainer",
    "Mutagenesis", "__version__", "cli", "get_logger", "load_config",
    "load_model_and_tokenizer", "setup_logging",
]

from .version import __version__

from .configuration import load_config

# dnallm.utils must load before dnallm.models: transformers_compat installs
# its pre-modeling device-query patch at .utils import time, ahead of the
# first "from transformers import" modeling-symbol resolution below
# (transformers >= 5.19 queries the accelerator during that import and
# raises on CUDA-built torch without a visible GPU -- the test-cuda CI legs).
from .utils import get_logger, setup_logging
from .models import load_model_and_tokenizer
from .datahandling import DNADataset
from .finetune import DNATrainer
from .inference import DNAInference, DNAInterpret, Benchmark, Mutagenesis
from .cli import cli
```

**Hard constraint from an existing guard** (`tests/preflight/test_structure.py:102-109`):

```python
def test_root_facade_does_not_mention_preflight() -> None:
    """dnallm/__init__.py stays torch-heavy by design; preflight is not a member."""
    source = Path(dnallm.__file__).read_text(encoding="utf-8")
    assert "preflight" not in source, (...)
```

That assertion is a **substring** check over the whole file. The lazy refactor
must keep the literal `preflight` (and, for symmetry, `tui`) out of the root
facade entirely — including comments and docstrings. The `preflight` CLI
subcommand therefore lands in `dnallm/cli/cli.py`, never as a root member.

**Keep-eager set (discretion, recommended):** `__version__` (`dnallm/version.py:1`)
and `load_config` are cheap; keeping them eager preserves
`tests/test_version_contract.py:17-22`'s `dnallm.__version__` access and any
`from dnallm import load_config` at import time. The ordering guarantee itself
moves to `dnallm/models/__init__.py`, so the root facade carries no ordering
logic.

See RESEARCH.md "PEP 562 lazy facade (root, shape)" for the `_LAZY` /
`__getattr__` / `__dir__` body.

---

### `dnallm/models/__init__.py` (package facade, ordering guarantee)

**Analog:** `dnallm/__init__.py:24-30` — the comment block above is the *entire*
existing statement of the utils-before-models guarantee. Today it is positional;
after D-01 it must be executable.

**Why it matters** (`dnallm/models/__init__.py:1-17`):

```python
from .model import load_model_and_tokenizer, DNALLMforSequenceClassification
from .modeling_auto import PRETRAIN_MODEL_MAPS, MODEL_INFO
from .registry import (
    RegistryEntry, all_entries, effective_max_length, effective_tokenizer_type,
    get_by_base_model, get_by_task_type, is_lab_model, reload, resolve,
    sort_key_lab_first, try_resolve,
)
from .tokenizer import DNAOneHotTokenizer
from .losses import FocalLoss
```

Line 1 is the torch path: `dnallm/models/model.py:12-14` imports torch /
transformers at module scope. Any `import dnallm.models.*` — including
`dnallm/preflight/availability.py:53` and `dnallm/preflight/compatibility.py:34`
which do `from ..models.registry import ...` — executes this package init.

**`__all__` is the contract** (`dnallm/models/__init__.py:19-37`): 16 names, all
of which must keep resolving identically after the refactor (import-parity test,
below). `registry` symbols (lines 3-15) are torch-free in principle but the
package init is not, so the lazy `__getattr__` must resolve *every* name through
the shims-first path:

```python
# order inside __getattr__ (D-01 trap a):
import_module("dnallm.utils._transformers_compat")   # apply_patches() runs at
import_module("dnallm.utils._cuda_compat")           # import (transformers_compat.py:1399, cuda_compat.py:86)
module = import_module(".model" if name in _HEAVY else ".registry")
```

Caution: importing `dnallm.utils._transformers_compat` executes
`dnallm/utils/__init__.py` first (parent package init), so the utils facade must
already be lazy or the shim import re-introduces torch at the parent level.

---

### `dnallm/utils/__init__.py` (package facade, three torch paths)

**Analog:** `dnallm/__init__.py:20-34` (same eager-facade shape), plus the
research-verified torch surface inside utils itself.

**Three torch paths, not one** — all must move behind `__getattr__`:

```python
# dnallm/utils/__init__.py:7-8
from . import cuda_compat as _cuda_compat
from . import transformers_compat as _transformers_compat
...
# dnallm/utils/__init__.py:42
from .support import is_fp8_capable, is_flash_attention_capable
```

- `dnallm/utils/transformers_compat.py:14-19` — `import torch` at module scope;
  `apply_patches()` runs at import (`:1399`).
- `dnallm/utils/cuda_compat.py` — `preload_cuda13_libs()` called at import (`:86`);
  ctypes/glob only, but it is a shim and belongs on the same lazy path.
- `dnallm/utils/support.py:1-2` — `from torch.cuda import get_device_capability`
  at module scope. Re-export names `is_fp8_capable` / `is_flash_attention_capable`
  (`__all__` lines 55-56) are also in `_cuda_compat` / `_transformers_compat`
  in `__all__` lines 47-48 (leading underscore names, still exported).

**Safe to keep eager:** `genomic_coords`, `genomic_intervals`, `sequence`,
`logger`, `training_plots` (lines 9-16, 22-43 minus 42). `get_logger` must stay
reachable without torch because every preflight module does
`from ..utils import get_logger` (`report.py:31`, `availability.py:54`,
`validation.py:40`, `compatibility.py:44`, `config_gen.py:59`).

**Consumer check** (grep, whole repo): nothing outside `dnallm/utils/__init__.py`
references `_cuda_compat` or `_transformers_compat` as attributes — only
`dnallm/models/special/evo.py:6` consumes `is_flash_attention_capable` /
`is_fp8_capable` via `from ...utils import ...`. Lazy resolution therefore breaks
no importer.

**Existing shim tests** that must stay green: `tests/utils/test_cuda_compat.py:1-20`
(`from dnallm.utils import cuda_compat` — submodule import, unaffected) and
`tests/utils/test_transformers_compat.py:1-20` (`import bitsandbytes.functional
as bnb_functional` + `import torch` in the test, depends on patches being applied
by the time the test module imports `dnallm`; still true because the test imports
`dnallm.utils.transformers_compat` explicitly).

---

### `dnallm/utils/env_info.py` (utility, core-owned collector — D-02)

**Analog:** `dnallm/utils/support.py:1-25` (utils-level hardware probe with a
module logger, plain functions, no class) **combined with**
`dnallm/preflight/availability.py:173-193` (the "import the heavy library inside
the function body" idiom):

```python
def effective_endpoints() -> dict[str, str]:
    ...
    import huggingface_hub.constants as hf_constants
    from modelscope.utils.file_utils import get_modelscope_cache_dir

    return {
        "huggingface": hf_constants.ENDPOINT,
        "modelscope_cache": get_modelscope_cache_dir(),
    }
```

This is the exact precedent for D-02's "torch imported inside the function
body only": module-level imports stay stdlib (`importlib.metadata`,
`platformdirs`), torch/`datasets`/`transformers` appear only inside the collect
function.

**Error-handling pattern to copy** — degrade, never raise:
`dnallm/utils/support.py:8-25` returns `False` plus `logger.warning` on an
unusable device; `dnallm/preflight/availability.py` degrades every probe failure
to `AvailabilityState.UNKNOWN` with a reason (PFC-06). The env collector returns
rows with a plain-language "unknown/checking failed" value instead of raising —
a TUI screen must never have to catch.

**Test analog:** `tests/utils/test_support.py:1-35` — monkeypatch the torch
symbol at module scope (`monkeypatch.setattr(support, "get_device_capability",
lambda: (9, 0))`) rather than constructing fake devices:

```python
class TestIsFp8Capable:
    def test_hopper_or_newer_reports_capable(self, monkeypatch):
        monkeypatch.setattr(support, "get_device_capability", lambda: (9, 0))
        assert support.is_fp8_capable() is True
```

Mirror this: patch the collector's torch seam (e.g. a module-level
`_torch()` helper or the imported names) so CUDA-less CI legs exercise the CPU
branch deterministically. `tests/utils/test_support.py` shows the repo already
assumes torch importable at test time; env_info tests should not assume a GPU.

**Rows to render (D-03):** dnallm version, Python, torch (`cuda.is_available`,
device name, VRAM total/free), transformers, numpy, datasets, free disk of
model-cache roots. Version strings come from `importlib.metadata.version(...)`
(RESEARCH "Don't Hand-Roll"), never from importing the library.

---

### `dnallm/tui/__init__.py` (package facade / entry point)

**Analog:** `dnallm/preflight/__init__.py:1-41` — a docstring that documents the
package contract, then a flat re-export list with `__all__`:

```python
"""dnallm.preflight — pure, read-only pre-flight validation API.
...
4. Import-light at the SOURCE level: modules under ``dnallm/preflight/``
   never import torch/transformers/datasets ...
"""

from .availability import clear_probe_memo, effective_endpoints, probe_availability
from .compatibility import check_compatibility
...
__all__ = [ ... ]
```

**Difference the TUI must honor:** `dnallm/tui/__init__.py` is imported only by
the CLI command body and by tests, so it may import textual eagerly *relative to
its own package* — but TUI-01's AST guard forbids torch/transformers anywhere
inside `dnallm/tui/`, and the structure test for preflight (`:102-109`) means the
root facade still never mentions it. Keep `__all__` to the entry symbol(s)
(`run_preflight`) — every name in it gets an importability assertion in the
structure test, so do not export screens.

---

### `dnallm/tui/messages.py` (message types, worker → UI payloads)

**Analog:** `dnallm/preflight/report.py:146-163` — frozen, typed, `__all__`-listed
payload models, with the rationale spelled out at `report.py:11-14`:

```python
3. Reason codes are machine-readable UPPER_SNAKE strings (``str`` enums) —
   agent-friendly by design: every finding and every dimension verdict
   carries one, so shells (Phase 25 TUI, Phase 26 MCP) can route on codes
   without owning verdict logic.
```

```python
class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)
    severity: Severity
    reason_code: str
    message: str
    count: int | None = None
```

**Adaptation:** Textual messages must subclass `textual.message.Message`, so
these are not Pydantic models. Copy the *discipline*, not the base class: one
class per payload, an `__all__`, a docstring naming the producer and consumer,
and payload fields that are already-core report objects (`DataReport`,
`CompatibilityReport`, `GeneratedConfig`, `AvailabilityReport`) plus primitives
for progress. Never re-shape a report into screen-local tuples — that is the
first step toward verdict logic in a screen.

---

### `dnallm/tui/workers.py` (worker seam)

**Analog (the thing being wrapped):** `dnallm/models/model.py:411-417` signature
and `:439-485` retry loop.

```python
def download_model(
    model_name: str,
    downloader: Any,
    revision: str | None = None,
    max_try: int = 10,
    allow_patterns: list[str] | None = None,
) -> str:
```

```python
        try:
            download_kwargs: dict[str, Any] = {"revision": revision}
            if allow_patterns is not None:
                download_kwargs["allow_patterns"] = allow_patterns
            status = downloader(model_name, **download_kwargs)
            if status != "incomplete":
                ...
                break
        # track the error
        except Exception as e:
            ...
            time.sleep(1)

    if status == "incomplete":
        raise ValueError(f"Model {model_name} download failed.")
```

Two load-bearing facts for the seam:
1. `downloader` is injectable — that is the cancel hook AND the test hook.
2. `except Exception` (line 461) swallows any `Exception`-derived cancel
   sentinel and retries 10× with `time.sleep(1)`. The sentinel must derive from
   `BaseException` (RESEARCH Flag 2 / Pitfall 2). `download_model` is NOT modified
   this phase.

**No in-repo analog for the worker itself** (no `textual`, no `@work`, no
thread-pool message plumbing in `dnallm/`). Closest structural approximations:
`dnallm/preflight/availability.py:89-92` — a shared `ThreadPoolExecutor` with a
bounded-wait pattern and an explicit comment about not doing cancellation games;
and the function-local heavy-import idiom (`dnallm/cli/cli.py:57-58`). Use
RESEARCH.md "Thread worker + typed message" and "Cancellable download wrapper"
verbatim as the shape.

**Logging inside workers:** `from ..utils import get_logger` and
`logger = get_logger("dnallm.tui.workers")` (module-level logger idiom:
`dnallm/cli/cli.py:8-10`, `dnallm/preflight/availability.py:62`). No `print()`
(ruff `T20`).

---

### `dnallm/tui/app.py` (composition root)

**Analog:** none. No Textual code exists in the repository. Closest
architectural cousin is the CLI group as a wiring layer: `dnallm/cli/cli.py:13-27`
(declares the surface, delegates all work) and `dnallm/mcp/server.py` (an app
object that owns lifecycle and hands work to a manager). The app's job here is
narrower still: hold the injected seams (`prober`, `downloader`) so Pilot tests
can run network-free, and route screens.

**Injection contract to design for (from the core):**
- `check_compatibility(..., prober=..., probe_shortlist=...)` —
  `dnallm/preflight/compatibility.py:132-140`
- `probe_availability(model, *, prober=..., timeout=...)` —
  `dnallm/preflight/availability.py:97-102`
- `download_model(model_name, downloader, ...)` — `dnallm/models/model.py:411-417`

All three already accept fakes; the app must thread them down rather than
monkeypatching in tests (the Phase-23 unit-suite discipline,
`tests/preflight/test_compatibility.py:446-456`).

---

### `dnallm/tui/screens/*.py` (six screens)

**Analog:** none for the widget layer. The screens' *rendering contract* is the
report inventory in `dnallm/preflight/report.py` — these are the exact objects
each screen receives and must render without recomputation:

| Screen | Core function(s) it may call (the TUI-08 table) | Report objects rendered |
|--------|--------------------------------------------------|-------------------------|
| `env.py` | `dnallm.utils.env_info.collect` (not preflight) | collector dict (TUI-03 rows) |
| `data.py` | `validate_data` (`validation.py:82-87`) | `DataReport` `:234-258`, `SplitReport` `:202-231`, `LengthStats` `:166-185`, `CharsetCensusEntry` `:188-199`, `Finding` `:146-163`, `Severity` `:59-64` |
| `suitability.py` | `check_compatibility` (`compatibility.py:132-140`), `probe_availability` (`availability.py:97-102`) | `CompatibilityReport` `:347-360`, `ModelVerdict` `:280-298`, `DimensionVerdict` `:261-277`, `ListingEntry` `:322-344`, `AvailabilityReport` `:301-319`, `VerdictStatus` `:67-77`, `AvailabilityState` `:80-85`, `Hub` `:88-94`, reason-code enums `:97-143` |
| `config.py` | `generate_config` (`config_gen.py:206-236`), `edit_config` (`config_gen.py:382-438`) | `GeneratedConfig` `:379-391`, `SectionError` `:363-376` |
| `download.py` | `download_model` (`model.py:411`) via the worker | returned path / `BaseException` sentinel |
| `summary.py` | none | already-held reports |

**Field-level rules the screens must encode (not invent):**
- D-12 read-only facts: `ModelVerdict.entry_name`/`is_lab`, `ListingEntry.num_labels`/`label_names`
  (`report.py:286-298`, `:338-343`) — read-only; `max_length` prefilled from
  `registry.effective_max_length` (emitted by `generate_config` at
  `config_gen.py:281-284`) but editable.
- D-13 error rendering: `SectionError(section, field, message)` maps 1:1 onto the
  `section.field → message` line the split pane shows (`report.py:363-376`).
- D-11 preservation: sections outside the five core ones (`ia3`/`vep`/`sweep`)
  survive because the TUI passes the raw mapping to `edit_config`, which preserves
  unpatched keys (`config_gen.py:374-378`: `base.update(value)` per section only).
- D-08 no-data mode: `check_compatibility(data=None, ...)` degrades to
  `CompatReasonCode.NO_DATA` (`report.py:123`); the screen renders a
  "no data provided yet" state rather than a FAIL.
- D-05/D-06 honesty: `AvailabilityState.UNKNOWN` renders "hosting not verified"
  and stays selectable; never relabel.

**Forbidden in every screen** (RESEARCH Anti-Patterns, enumerable by the TUI-08
table test): any comparison over `Severity`/`VerdictStatus` beyond selecting a
style, any `sorted()` over `report.listing`, any direct `yaml.dump`, any
`import torch`, any direct import of a preflight submodule (facade only).

**Markup safety (Pitfall 5):** registry names and data findings are untrusted
text; set `markup=False` on `Static`/`Label` or apply `rich.markup.escape`.

---

### `dnallm/cli/cli.py` (CLI controller — the `preflight` subcommand)

**Analog:** the `mcp-server` command in the same file, `:248-317` — click group
member, options declared on the decorator, heavy import inside the body, and a
two-level `try/except` that turns ImportError into friendly output:

```python
@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), ...)
def mcp_server(config, host, port, log_level, transport):
    """Start MCP (Model Context Protocol) server"""
    try:
        # Import and run the MCP server
        from ..mcp.server import main as mcp_server_main
        ...
    except ImportError as e:
        click.echo(f"Error importing MCP server: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"MCP server startup failed: {e}")
        sys.exit(1)
```

**Group / logger conventions** (`cli.py:6-27`): `import click`, `import sys`,
`from ..utils import get_logger`; `logger = get_logger("dnallm.cli")`;
`@click.group()` + `@click.version_option()`.

**The `preflight` command (TUI-02) copies three things:** lazy
`from ..tui import run_preflight` inside the body; `except ImportError` →
`click.echo("... pip install dnallm[tui]")` + `sys.exit(1)`; and
`--config/-c` with `type=click.Path(exists=True)` for D-07's direct-jump entry.
Do **not** reuse the argv-swapping mechanism the mcp/config-generator commands
use (`:288-310`, `:231-238`) — the TUI takes Python-level arguments, not argv.

---

### `pyproject.toml` (packaging/config)

**Analog:** the extras block itself, `pyproject.toml:81-159`.

```toml
[project.optional-dependencies]
dev = [
    "dnallm[test,notebook]",
    ...
]
test = [
    "pytest>=8.4",
    "pytest-asyncio>=1.0",
    ...
]
full = [
    "dnallm[dev,test,notebook,mcp]",
    "isort>=6.0.1",
]
# Deprecated one-cycle alias (owner 2026-10-14): base was renamed to full in
# v1.3; this alias still installs the full set (no silent under-install).
# Remove in v1.4 -- pip/uv then only warn that the extra does not exist and
# exit 0, leaving a core-only under-install (which this cycle prevents).
base = [
    "dnallm[full]",
]
all = [
    "dnallm[full,dev,test,notebook,docs,ui,mcp,fla]",
]
```

**First act, in this order (RESEARCH Flag 3):**
1. Add `tui = ["textual>=8.2,<9", "platformdirs>=3.6,<5"]`.
2. Add `"dnallm[tui]"` to `test` (`:91-99`).
3. Add `tui` to `all` (`:157-159`) → `dnallm[full,dev,test,notebook,docs,ui,mcp,fla,tui]`.
4. Delete `base` and its 4-line deprecation comment (`:150-156`).
5. Docs: `README.md:178`, `docs/getting_started/installation.md:257`,
   `docs/getting_started/install_honesty.md:14-15` (the "during the v1.3 alias
   cycle" row is deleted; the "after the v1.4 alias removal" row stays and
   becomes the live state).

Comment style in this file is long-form rationale above each non-obvious entry
(`:104-117`, `:131-144`, `:180-190`) — the `tui` entries should carry a one-line
rationale (why textual is optional, why `platformdirs` matches textual's own
range), not just the pins.

**Coverage config is untouchable for this phase** (`pyproject.toml:442-452` omit
list; `fail_under = 90` at `:456`): no `tui` entry may be added, and the
structure test asserts it. Note `[tool.pytest.ini_options].testpaths` is
`["tests", "dnallm/mcp/tests"]` (`:406`) — `tests/tui/` is collected
automatically; `--asyncio-mode=auto` and `--timeout=300` (`:415-416`) are what
make Pilot tests decorator-free and cap a hung test.

**Scripts** (`:261-268`): no new console script this phase — `dnallm preflight`
is a subcommand of the existing `dnallm` group.

---

### `.github/workflows/ci.yml` (no edit expected)

Every pytest leg installs `-e ".[full]"`; once `test` carries `dnallm[tui]`,
textual rides all legs including Windows:

- `:92`, `:172`, `:408` — `uv pip install -e ".[full]"`
- `:261` — `".[full,cuda130]"`; `:341` — `".[full,fla]"`; `:543` — `".[full,fla]"`;
  `:758` — `".[full,fla,dev,mcp]"`

Analog value is as a **verification checklist**, not an edit target: confirm no
leg installs a hand-listed subset that would skip `test` (none does). The
first-act plan should still grep `ci.yml` for `.[test]`/`.[base]` before
declaring "no install-line edits".

---

### `tests/test_extras_guard.py` (modified contract guard — PKG-01)

**Analog:** itself. Three things in this file go red the moment `base` is
removed, and all three are the specification for the edit:

1. `:159-172` — `test_base_alias_is_pure_self_reference` asserts
   `_load_extras().get("base") is not None`. Must become an absence assertion
   (`"base" not in extras`) with a docstring naming PKG-01 and the v1.4 cycle.
2. `:62-72` — `EXPECTED_FULL_MEMBERS` / `EXPECTED_BASE_ALIAS` constants; the
   alias constant goes away.
3. `:219-261` `SANCTIONED_BASE_BRACKET_RESIDUE` — four sanctioned
   `(path, line)` pairs: the README note, the installation.md note, and TWO
   `install_honesty.md` rows. After PKG-01 the README/installation notes are
   deleted and the "during the v1.3 alias cycle" honesty row is deleted, so the
   sanctioned set shrinks; `test_base_bracket_residue_equals_sanctioned_list`
   (`:334-349`) is bidirectional (extra **and** missing both fail), which is the
   deletion-creep guard. Update both sides in one change.

**Pattern to reuse for the new `tui` extra:** `:100-115` — parse
`pyproject.toml` with `tomllib`, assert exact member strings
(`EXPECTED_MCP_MEMBERS` style), assert nothing pre-existing was lost. A
`TestTuiExtraMembers` class asserting `textual>=8.2,<9` and
`platformdirs>=3.6,<5` as exact strings plus `"dnallm[tui]" in extras["test"]`
is the natural extension.

**Repo-scan style** (`:300-332`): `SCAN_ROOTS`, `SKIP_DIR_NAMES`,
`path.relative_to(REPO_ROOT).as_posix()` (Windows-safe), sanctioned frozensets
stored with forward slashes. Any new residue gate must use `.as_posix()` — the
comment at `:327-330` records the Windows-leg failure that proved it.

---

### `tests/tui/test_structure.py` (new AST guard suite)

**Analog:** `tests/preflight/test_structure.py` — copy the helpers verbatim and
swap the target package. Extract these blocks:

**AST helper** (`:55-68`):

```python
def _imported_top_level_modules(tree: ast.AST) -> set[str]:
    """Top-level module names referenced by any Import/ImportFrom node, any scope."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.add(module.split(".")[0])
            for alias in node.names:
                names.add(alias.name.split(".")[0])
    return names
```

`ast.walk` (not `tree.body`) is what makes the guard scope-blind — that is the
property that matters for a "no torch even inside a function" rule.

**Guard shape** (`:71-82`): collect offenders into a dict keyed by repo-relative
posix path, assert `offenders == {}` with a message naming the invariant and
referencing the decision id.

**Guards this phase needs (all mirroring the above):**
1. Zero `textual` / `rich` imports anywhere under `dnallm/` **except**
   `dnallm/tui/**` (TUI-01).
2. Zero `torch` / `transformers` / `datasets` imports anywhere under
   `dnallm/tui/**` at any scope (TUI-01) — including inside function bodies.
3. `pyproject.toml [tool.coverage.run] omit` carries no `tui` entry (`:112-121`
   pattern, `tomllib` parse).
4. `dnallm/tui/__init__.py` declares a non-empty `__all__` and every name
   resolves (`:124-130`).
5. Panel→core allowlist as a *table test* (TUI-08): for each
   `dnallm/tui/screens/*.py`, parse the AST for `dnallm.preflight` /
   `dnallm.models.model` / `dnallm.utils.env_info` import names and assert the
   set equals the screen's allowlisted set. `test_screens.py` can own this or
   the structure file can — keep one owner.
6. Root-facade non-membership extension: assert `"tui" not in` the root
   `dnallm/__init__.py` source (mirrors `:102-109`) and that the preflight
   guard still passes.

**Discipline (mandatory, from `:27-29`):** each guard is probe-verified RED under
a deliberate local violation before being committed green.

---

### `tests/test_package_facade.py` (new import-purity + parity contract)

**Analogs:** `tests/test_version_contract.py` (cross-cutting root-level contract
test that reads `pyproject.toml` with `tomllib` and imports `dnallm`) and
`tests/preflight/test_structure.py`.

**Runtime purity MUST be a subprocess** — in-process `sys.modules` assertions are
vacuous because `tests/conftest.py:9` imports torch at session start. Copy the
subprocess idiom from `tests/preflight/test_validation.py:706-733`:

```python
    import subprocess  # ruff: ignore[suspicious-subprocess-import]
    import sys
    ...
    code = (
        "from dnallm.preflight import validate_data;"
        ...
    )
    env = {key: value for key, value in os.environ.items() ...}
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv only
        [sys.executable, "-X", "utf8=0", "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
```

For purity use `[sys.executable, "-I", "-c", code]` (isolated: no cwd on
`sys.path`, no user site, ignores `PYTHONPATH`) and emit a machine-readable
marker line (`print("HEAVY=" + ",".join(sorted(...)))`) — the repo's existing
test matches a marker line because the package logger also writes to stdout
(`:731-732`). Assert `torch`, `transformers`, `datasets` absent after
`import dnallm.preflight`.

**Import-parity test:** for each of `dnallm`, `dnallm.models`, `dnallm.utils`,
import the module, then for every name in `__all__` assert
`getattr(module, name) is getattr(importlib.import_module(target_submodule), name)`
and that `dir(module)` includes the name (`__dir__` contract). Also exercise
`from dnallm import *` inside the subprocess for `__all__` parity.

**Ordering-guarantee test (D-01 trap a):** a subprocess that does
`import dnallm.models; dnallm.models.load_model_and_tokenizer` directly (no root
`import dnallm` first) and asserts success. On the CUDA-less CI legs this is the
test that catches a lost `_transformers_compat` import.

---

### `tests/tui/test_workers.py` (seam unit tests)

**Analog:** `tests/preflight/test_compatibility.py:446-470` — fake prober via
`Mock(return_value=AvailabilityReport(...))`, then assert call count and call
args:

```python
def _mock_prober():
    """A fake prober returning one shared remote-confirmed report."""
    return Mock(
        return_value=AvailabilityReport(
            state=AvailabilityState.REMOTE_CONFIRMED,
            size_bytes=1,
            revision="abc",
            source_hub=Hub.MODELSCOPE,
        )
    )


def test_prober_shortlist_probes_first_n_in_lab_order() -> None:
    prober = _mock_prober()
    result = check_compatibility(
        task_type="binary", intent="predict", prober=prober, probe_shortlist=3
    )
    assert prober.call_count == 3
    ...
```

Same discipline: fake `downloader` callable that counts calls, so the cancel
assertion is "no *further* downloader calls after `cancelled_event` is set"
(RESEARCH Pitfall 1 warning sign), not "the thread stopped".

**Wave-0 probe (RESEARCH Assumption A3):** a tiny test that pushes a
`BaseException`-derived sentinel through the REAL `download_model` with a fake
downloader and asserts it propagates in one attempt (no 10× retry, no ~10 s
sleep). Land it before any screen work — it validates the whole cancel design.

---

### `tests/tui/test_app.py` (Pilot tests)

**Analog:** partial only. The repo's async test idiom is
`tests/mcp/test_timeout.py:1-55` — plain `async def` (no decorator, since
`--asyncio-mode=auto`), a fixture that builds the object with heavy collaborators
patched, and `pytest.mark.asyncio` used in the older style
(`:52`). Textual Pilot usage (`run_test(size=(100, 40))`,
`await app.workers.wait_for_complete()`, `await pilot.pause()`) has **no in-repo
precedent** — it comes from RESEARCH.md "Pilot test (headless, deterministic)".

**Rules to keep wall-time bounded (RESEARCH Flag 4):** fakes for `prober` and
`downloader` always; `pause()` after every interaction; `wait_for_complete()`
instead of polling; no network, no torch. Budget ≤ ~30 s for `tests/tui/`.

---

### `tests/tui/test_screens.py` (panel→core table test, TUI-08)

**Analog:** `tests/preflight/test_integration.py:1-70` — a facade-only,
story-shaped test that documents "the one-core-many-shells contract" in its
module docstring and asserts a `REQUIRED_FACADE_NAMES` list. The structural idea
(an explicit, declared name list that the test enforces) transfers directly:
declare `SCREEN_CORE_ALLOWLIST = {screen_module: {function, ...}}` and assert
both directions (nothing missing, nothing extra).

---

### `tests/utils/test_env_info.py` (collector unit tests)

**Analog:** `tests/utils/test_support.py:1-35` (see `dnallm/utils/env_info.py`
above). Add a "no GPU / no CUDA" case and a "torch import fails" case, mirroring
`test_missing_flash_attn_reports_incapable` (`:17-22`: `monkeypatch.setitem(sys.modules, "flash_attn", None)`).

---

## Shared Patterns

### Lazy / function-local heavy imports
**Source:** `dnallm/cli/cli.py:57-58` and `:92-93`; `dnallm/preflight/availability.py:187-188`.
**Apply to:** `dnallm/tui/workers.py` (textual lives only inside `dnallm/tui/`),
`dnallm/utils/env_info.py` (torch inside the collector body), the new
`preflight` CLI command body (import `dnallm.tui` there), and every lazy
`__getattr__`.

```python
    from ..finetune import DNATrainer
    from ..configuration import load_config
```
```python
    import huggingface_hub.constants as hf_constants
    from modelscope.utils.file_utils import get_modelscope_cache_dir
```

### Logging
**Source:** `dnallm/cli/cli.py:8-10`, `dnallm/preflight/availability.py:62`.
**Apply to:** every new `dnallm/` module.

```python
from ..utils import get_logger

logger = get_logger("dnallm.tui.workers")
```

No `print()` in library code (ruff `T20` selected). `click.echo` is the sanctioned
user-facing output channel, and only in `dnallm/cli/`.

### Error handling / degradation
**Source:** `dnallm/preflight/availability.py` (never raises; degrades to UNKNOWN
with a reason), `dnallm/utils/support.py:8-25` (returns False + warning),
`dnallm/cli/cli.py:312-317` (ImportError → friendly echo → exit 1).
**Apply to:** env collector (unreadable rows, not exceptions), workers (worker
error → plain-language message via `Worker.StateChanged`), CLI command
(missing `tui` extra).

```python
    except ImportError as e:
        click.echo(f"Error importing MCP server: {e}")
        sys.exit(1)
```

### Typed frozen payloads as the UI contract
**Source:** `dnallm/preflight/report.py:146-391` (`ConfigDict(frozen=True)` on
every model, `__all__` at `:35-56`, docstring invariants at `:1-25`).
**Apply to:** `messages.py` payload fields; screens must consume these objects
verbatim.

### AST structure guards
**Source:** `tests/preflight/test_structure.py` (helpers `:50-68`, guards
`:71-158`, probe-red discipline `:27-29`).
**Apply to:** `tests/tui/test_structure.py`, plus an extension so the preflight
guard list grows only additively (its `:102-109` root-facade assertion must be
preserved unchanged).

### Subprocess isolation for import tests
**Source:** `tests/preflight/test_validation.py:706-733`.
**Apply to:** all `sys.modules` purity and import-parity assertions.

### Coverage discipline
**Source:** `pyproject.toml:442-456`; structure guard `test_structure.py:112-121`.
**Apply to:** every new module — new code enters the `fail_under=90` denominator
immediately and needs ≥96% per-module; never add an omit entry.

### Docstrings
**Source:** `dnallm/preflight/availability.py:97-124` (`Args:` with
`name: description`, `Returns:`, `Raises:`), `dnallm/utils/env_info.py`'s
neighbors in `dnallm/utils/support.py:8-25`. Google style, `str | None` PEP 604,
relative imports inside `dnallm/`, absolute imports in tests.

---

## No Analog Found

| File | Role | Data Flow | Reason / closest approximation |
|------|------|-----------|-------------------------------|
| `dnallm/tui/app.py` | app | event-driven | No Textual anywhere in the repo. Closest: `dnallm/cli/cli.py:13-27` (wiring-only surface). Use RESEARCH "Pattern 1/2". |
| `dnallm/tui/workers.py` | service seam | event-driven | No `@work`, no thread→message plumbing. Closest: `dnallm/preflight/availability.py:89-92` (bounded `ThreadPoolExecutor`). Use RESEARCH "Thread worker + typed message" + "Cancellable download wrapper". |
| `dnallm/tui/screens/*.py` | component | request-response / streaming | No widget code exists. Render contract comes from `dnallm/preflight/report.py`; call contract from the screen→core table (TUI-08). |
| `dnallm/tui/styles.tcss` | config | n/a | No `.tcss` or stylesheet of any kind in the repo. |
| `dnallm/__init__.py` / `models/__init__.py` / `utils/__init__.py` lazy facades | facade | lazy resolution | No module-level `__getattr__` in `dnallm/` today (verified by grep). Use RESEARCH "PEP 562 lazy facade" + "Models lazy facade with the ordering guarantee". |
| `tests/tui/test_app.py` (Pilot body) | test | event-driven | Async idiom exists (`tests/mcp/test_timeout.py`), Pilot does not. Use RESEARCH "Pilot test". |

---

## Ordering Constraints for the Planner

1. **First act is atomic** (TUI-01 + PKG-01 + structure tests + worker seam):
   `pyproject.toml` extras surgery → `tests/test_extras_guard.py` update → docs
   notes → `tests/tui/test_structure.py` (probe-red then green) →
   `dnallm/tui/{__init__,messages,workers}.py`. No panels in this act.
2. **D-01 lazy facades land with `tests/test_package_facade.py` in the same
   change** — the 400+-test suite is the regression net (D-01 reversibility note),
   and the ordering guarantee has no other executable proof.
3. **Screens after the seam**, one screen per plan, each with its own tests
   (same-change pytest rule).
4. `dnallm/utils/env_info.py` + `tests/utils/test_env_info.py` can land with the
   env screen, but before it (the screen only renders its dict).

## Metadata

**Analog search scope:** `dnallm/` (all subpackages via grep/glob), `tests/`,
`pyproject.toml`, `.github/workflows/ci.yml`, `README.md`, `docs/getting_started/`.
**Files read (with line ranges):** `dnallm/__init__.py`, `dnallm/models/__init__.py`,
`dnallm/utils/__init__.py`, `dnallm/utils/support.py`, `dnallm/utils/transformers_compat.py` (head),
`dnallm/utils/cuda_compat.py` (head), `dnallm/cli/cli.py:1-120,230-321`,
`dnallm/preflight/{__init__,report,compatibility,availability,validation,config_gen}.py`,
`dnallm/models/model.py:400-499`, `tests/preflight/test_structure.py`,
`tests/preflight/test_integration.py:1-70`, `tests/preflight/test_compatibility.py:440-475`,
`tests/preflight/test_validation.py:700-733`, `tests/cli/test_cli.py:1-60,300-400`,
`tests/utils/test_support.py`, `tests/utils/test_cuda_compat.py:1-20`,
`tests/test_extras_guard.py`, `tests/test_version_contract.py`,
`tests/mcp/test_timeout.py:1-55`, `tests/conftest.py:1-60`, `pyproject.toml`, `.github/workflows/ci.yml` (install lines).
**Not used:** `.planning/graphs/graph.json` (stale per RESEARCH metadata note).
**Pattern extraction date:** 2026-10-21
