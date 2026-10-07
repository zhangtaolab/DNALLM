# Phase 8: Full Execution Rollout & Repair Loop - Pattern Map

**Mapped:** 2026-10-04
**Files analyzed:** 17 (new/modified file groups; notebook families counted as groups)
**Analogs found:** 16 / 17 (1 with no in-repo analog — ollama systemd unit; RESEARCH.md supplies a verbatim template)

All analog paths below verified git-tracked (`git ls-files` non-empty). No gitignored
mirror paths are referenced.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `.github/workflows/ci.yml` (new `example-nightly` job) | config (CI) | batch | `coverage-nightly` job, ci.yml:406-497 + `test-mamba` job, ci.yml:262-344 | exact |
| `models.lock` (new revision-pinned entries) | config | file-I/O | `models.lock:4-13` | exact |
| `dnallm/models/model.py` (`allow_patterns` passthrough) | service (model loader) | file-I/O | `download_model` + `_get_model_path_and_imports`, model.py:317-492 | exact (self-extension) |
| `dnallm/models/special/evo.py` (giants allow_patterns + evo-1 route) | handler | file-I/O | `_handle_evo1_models` load chain, evo.py:310-398 | exact (self-extension) |
| `dnallm/utils/transformers_compat.py` (np.fromstring shim + new rungs) | utility (compat shim) | event-driven (import-time patch) | 9 existing shims; closed map at transformers_compat.py:489-510 | exact |
| `dnallm/models/special/megadna.py` (DNATokenizer generate repair) | handler | transform | DNATokenizer class, megadna.py:38-105 | exact (self-extension) |
| `pyproject.toml` (langchain-ollama in `mcp` extra) | config | config | `mcp` extra, pyproject.toml:115-121 | exact |
| `tests/examples/_execution.py` (per-notebook env overrides) | test harness utility | batch | `_ENV_OVERRIDES` + env sandwich, _execution.py:248-256, 374-390 | exact (self-extension) |
| `tests/examples/test_notebook_execution.py` (lane/gate growth, stage deselection) | test | batch | itself: ACTIVE_NOTEBOOKS:66-80, GATED_NOTEBOOKS:570-585, gates:505-563 | exact |
| `tests/examples/test_marimo_execution.py` (D-18 deepening) | test | batch | itself: test_app_exports_html, lines 45-73 | exact |
| `tests/examples/test_script_execution.py` (heal marker retirement) | test | batch | itself: marker conversion, lines 92-105 | exact |
| `tests/expected_skips.yaml` (new typed skips, if any) | config | config | typed prefixes block, lines 29-41 | exact |
| `example/notebooks/*` family repairs (evo ref 131k→8k, finetune_generation wget+pin, finetune_custom_head demo cell, source= alignment) + 2 `example/mcp_example/*` | content (notebook) | transform | showcase notebooks' provenance/guard pattern (`plant_helixseek_cre.ipynb`); sibling notebooks | role-match |
| `docs/example/**/*.md` mirrors (byte-sync regeneration per D-20) | docs | transform | existing mirrors (frontmatter `notebook:`/`sync_check: true`) + `scripts/generate_md_from_notebook.py` | exact |
| `tests/models/test_model.py` (allow_patterns regression) | test | unit/mock | `TestDownloadModel`, test_model.py:43-140 | exact |
| `tests/models/test_special/*` (megaDNA tokenizer regression) | test | unit/mock | `tests/models/test_special/test_evo.py` (class-grouped wrapper tests) | exact |
| `infra/` or `scripts/runner/` ollama systemd unit + README (NEW dir, planner names it) | config (infra) | request-response (service) | none in repo — RESEARCH.md "ollama systemd unit" template (verbatim stock unit + `OLLAMA_HOST=127.0.0.1:11434`) | none |

## Pattern Assignments

### `.github/workflows/ci.yml` — new `example-nightly` job (config/CI, batch)

**Analog:** `coverage-nightly` job (ci.yml:406-497) for skeleton/cache/audit; `test-mamba` job (ci.yml:262-344) for self-hosted build-step precedent.

**Event gating — copy VERBATIM (security requirement, STRIDE fork-PR mitigation):**
```yaml
# Source: .github/workflows/ci.yml:412-413
    runs-on: [self-hosted, dnallm-nightly]
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
```

**Shared models.lock-keyed cache (D-09 — reuse this exact key; read-only sharing):**
```yaml
# Source: .github/workflows/ci.yml:457-465
      - name: Restore model caches (keyed on models.lock)
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/huggingface/hub
            ~/.cache/modelscope/hub
          key: ${{ runner.os }}-models-${{ hashFiles('models.lock') }}
          restore-keys: |
            ${{ runner.os }}-models-
```

**Install line (D-10 extends the coverage-nightly precedent `uv pip install -e ".[base,fla]"` at ci.yml:467-476):**
```yaml
# Source: .github/workflows/ci.yml:467-476 (extend extras per D-10)
        run: |
          uv venv
          # fla included (WR-01): this is the only leg running the full suite
          uv pip install -e ".[base,fla]"
```

**Gated-family source-build step precedent (test-mamba, ci.yml:308-323) — flash-attn/mamba wheel builds follow this shape but ADD wheel caching (research "Pattern: wheel caching"):**
```yaml
# Source: .github/workflows/ci.yml:320-323
        run: |
          uv venv
          uv pip install -e ".[base,fla]"
          uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation
```

**Fail-soft summary semantics (D-08): the per-stage pytest invocations mirror the coverage-nightly census step (ci.yml:489-492); NO continue-on-error anywhere (test-mamba comment at ci.yml:325-328 is the honesty precedent); final step computes non-zero exit from accumulated stage results.**
```yaml
# Source: .github/workflows/ci.yml:489-492
      - name: Run gated full census (slow included)
        run: |
          source .venv/bin/activate
          .venv/bin/python -m pytest -ra --durations=0 --junitxml=pytest-junit-nightly.xml --cov -p no:cacheprovider -p no:progress
```

**Skip audit gate (every stage's junit must flow through this):**
```yaml
# Source: .github/workflows/ci.yml:494-497
      - name: Skip audit (nightly junit)
        run: |
          source .venv/bin/activate
          python scripts/audit_skips.py pytest-junit-nightly.xml tests/expected_skips.yaml
```

Staged-serial layout (D-07) maps to four separate pytest invocations in one job:
stage 1 `tests/examples` with the two mcp gated ids deselected; stage 2 `dnallm/mcp/tests`
with server up; stage 3 the two mcp gated tests re-run; stage 4 summary. Between stages:
`pkill ipykernel_launcher` + VRAM settle (research architecture diagram).

---

### `models.lock` — new revision-pinned entries (config, file-I/O)

**Analog:** `models.lock:4-13` — copy the two-spaces-after-prefix, aligned-column, trailing `#` purpose-comment format exactly; D-15 adds a pinned form (`@<sha>` or `rev=` — planner picks one greppable form, CI-08 in Phase 9 will parse it):
```
# Source: models.lock:4-13 (verbatim)
hf  microsoft/DialoGPT-small                         # tests/models/test_model.py::test_download_real_huggingface_connection
hf  zhangtaolab/plant-dnagpt-BPE-promoter            # tests/inference/test_inference_real_model.py (from_pretrained, HF route)
ms  zhangtaolab/plant-dnabert-BPE                    # tests/finetune/test_trainer_real_model.py (12 call sites, source=modelscope)
ms  zhangtaolab/PlantHelixSeek-CRE                    # tests/examples/test_plant_helixseek_showcase.py (nightly showcase execution, source=modelscope)
```
Header comment (models.lock:1-3) already documents the cache-key rotation contract — do not
change it. Candidate ids/prefixes/shas are in 08-RESEARCH.md "models.lock candidate additions"
(9 ms rows + 5 hf rows, all registry-verified). evo-1-8k-base goes `hf` with the giants-tier
comment (NOT in cache, D-14).

---

### `dnallm/models/model.py` — `allow_patterns` passthrough (service, file-I/O)

**Analog:** the functions being extended (must hold at LOAD time, not just prefetch — Pitfall 4).

**Current call site with NO allow_patterns (model.py:348) — the exact line the extension threads through:**
```python
# Source: dnallm/models/model.py:317-322, 343-351
def download_model(
    model_name: str,
    downloader: Any,
    revision: str | None = None,
    max_try: int = 10,
) -> str:
    ...
    while True:
        ...
        try:
            status = downloader(model_name, revision=revision)
```

**Hub branches that pass the downloader (model.py:421-431):**
```python
# Source: dnallm/models/model.py:421-431
    elif source_lower == "huggingface":
        from huggingface_hub import snapshot_download as hf_snapshot_download

        model_path = download_model(model_name, downloader=hf_snapshot_download, revision=revision)

    elif source_lower == "modelscope":
        from modelscope.hub.snapshot_download import (
            snapshot_download as ms_snapshot_download,
        )

        model_path = download_model(model_name, downloader=ms_snapshot_download, revision=revision)
```
Extension shape (research Pattern: evo-1 giants tier): `allow_patterns: list[str] | None = None`
on both `download_model` and `_get_model_path_and_imports`, forwarded as a downloader kwarg
(only when not None, so the ModelScope downloader is untouched); evo-1 handler passes
`["*.safetensors", "*.json", "*.txt", "README.md"]`. Error-handling/retry loop stays as-is.

**Regression test rides in the existing class (see `tests/models/test_model.py` below).**

---

### `dnallm/models/special/evo.py` — giants route (handler, file-I/O)

**Analog:** its own load chain. The revision rule and the passthrough call site:
```python
# Source: dnallm/models/special/evo.py:368-378
            from ..model import _get_model_path_and_imports

            evo_model = CustomEvo1()
            revision = "1.1_fix" if "." in model_name and source == "huggingface" else "main"
            _, modules = _get_model_path_and_imports(model_name, source, revision=revision)
            evo_model.model = evo_model.load_checkpoint(
                model_name=model_name,
                revision=revision,
                config_path=config_path,
                modules=modules,
            )
```
Note: `evo-1-8k-base` contains no dot ⇒ fetches `main` ⇒ lock pin uses the observed main sha
`a9be7b6...` (research "Existing infrastructure facts"). The ImportError guard pattern for
optional deps (evo.py:340-346) is the template for any new prerequisite handling.

---

### `dnallm/utils/transformers_compat.py` — np.fromstring shim / new rungs (utility, event-driven)

**Analog:** the absence-gated idempotent shim pattern. Every new rung (e.g. the stripedhyena
`np.fromstring` fix for CharLevelTokenizer) copies this three-part shape.

**Part 1 — vendored constant / closed map with provenance comment (transformers_compat.py:501-510):**
```python
# Source: dnallm/utils/transformers_compat.py:501-510 (verbatim)
# ... This rung SUPERSEDES the 05-04 D-07 rung termination ... per
# the owner instruction of 2026-10-02: fix all non-gated census failures now.
# If execution surfaces another removed 4.x config default that remote code
# reads, extend the closed map -- never a catch-all.

_LEGACY_PRETRAINED_CONFIG_DEFAULTS: dict[str, object] = {
    "is_decoder": False,
    "add_cross_attention": False,
}
```

**Part 2 — patch function: try-import guard + absence gate + idempotency sentinel (transformers_compat.py:472-486):**
```python
# Source: dnallm/utils/transformers_compat.py:472-486
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers not installed
        return

    if hasattr(PreTrainedModel, "get_extended_attention_mask"):
        return

    if getattr(PreTrainedModel, "_dnallm_extended_mask_patch", False):
        return

    PreTrainedModel.get_extended_attention_mask = (  # type: ignore[method-assign]
        _get_extended_attention_mask
    )
    PreTrainedModel._dnallm_extended_mask_patch = True  # type: ignore[attr-defined]
```

**Part 3 — register in apply_patches (transformers_compat.py:1024-1034):**
```python
def apply_patches():
    """Apply all compatibility patches. Safe to call multiple times."""
    _patch_get_parameter_or_buffer()
    ...
    _patch_legacy_init_weights_bookkeeping()
```
A rung patching numpy (not transformers) targets `numpy.fromstring` the same way: gate on
`hasattr(np, "fromstring")` absence, sentinel-attribute, register in `apply_patches()`. Any
new library-change tests belong beside the existing shim tests (same file's test module per
D-19 — check `tests/utils/` for the existing transformers_compat tests when implementing).

---

### `dnallm/models/special/megadna.py` — DNATokenizer generate repair (handler, transform)

**Analog:** the class being repaired (megadna.py:38-105). Known-bug neighborhood — note the
`m in "megaDNA_updated"` substring checks (megadna.py:111-118) when writing the regression
test; the ImportError wrap (136-142) is the optional-dep pattern.
```python
# Source: dnallm/models/special/megadna.py:38-49, 72-83
            class DNATokenizer(PreTrainedTokenizer):
                """
                This tokenizer treats each nucleotide (A, T, C, G)
                as a separate token, along with special tokens for padding,
                end-of-sequence, and unknown tokens.
                """
                vocab_files_names = {  # ruff: ignore[mutable-class-default]
                    "vocab_file": "vocab.txt"
                }
                DEFAULT_TOKENS = ("**", "#")
                ...
                @property
                def vocab_size(self) -> int:
                    return len(self.vocab)

                def get_vocab(self) -> dict[str, int]:
                    return self.token_to_id.copy()
```
Regression test goes in `tests/models/test_special/` (see below), NOT `tests/examples/`
(D-19 layer ownership: this is a library bug).

---

### `pyproject.toml` — langchain-ollama in `mcp` extra (config)

**Analog:** the mcp extra block (pyproject.toml:115-121). One line added alphabetically-ish
in the existing style; version from research: `langchain-ollama>=1.1.0` verified on PyPI.
```toml
# Source: pyproject.toml:115-121
mcp = [
    "mcp>=1.0.0,<2",
    "langchain>=1.3.6",
    "langchain_mcp_adapters>=0.2.1",
    "nest-asyncio>=1.5.9",
    "pydantic-ai<3",
]
```
Constraint (research): gated-family packages (evo-model, stripedhyena, evo2, vtx,
flash-attn, MEGABYTE_pytorch, megaDNA clone) NEVER land in extras — job-step only.

---

### `tests/examples/_execution.py` — per-notebook env overrides (test harness, batch)

**Analog:** the existing `_ENV_OVERRIDES` save/restore sandwich. The giants mechanism
(D-14) extends this to per-notebook env (evo notebook kernel gets
`HF_HUB_CACHE=~/models-giants/hub`), keyed off NOTEBOOK_EXEC_SPECS the same way
`kernel_name` already is (spec key at _execution.py:189).
```python
# Source: tests/examples/_execution.py:252-256
_ENV_OVERRIDES: dict[str, str] = {
    "MPLBACKEND": "Agg",
    "TOKENIZERS_PARALLELISM": "true",
    "WANDB_MODE": "disabled",
}
```
```python
# Source: tests/examples/_execution.py:374-390 (the sandwich inside run_notebook)
    saved_env = {key: os.environ.get(key) for key in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    try:
        client.execute()  # internally: setup_kernel -> cells -> finally _cleanup_kernel()
        return nb
    except (CellExecutionError, CellTimeoutError) as exc:
        ...
        raise
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
```
Spec entry precedent carrying a non-budget key (kernel_name) — an `env` spec key follows
this shape (_execution.py:179-190):
```python
    str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_langchain_agents.ipynb"): {
        "cell_timeout": 1800,
        "extra_inputs": [],
        "kernel_name": LANGCHAIN_KERNEL_NAME,
    },
```
The harness contract lines that must NOT change: `allow_errors=False` (_execution.py:368,
fail-at-first-error is EXEC-02's semantics) and the typed-skip helpers (_execution.py:752-801).

---

### `tests/examples/test_notebook_execution.py` — lane growth / stage deselection (test, batch)

**Analog:** itself. Moving a family from gated to ACTIVE = append to this list, with a
comment citing the passing campaign evidence (the file's established comment discipline):
```python
# Source: tests/examples/test_notebook_execution.py:66-80 (excerpt)
ACTIVE_NOTEBOOKS = [
    EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb",
    EXAMPLE_DIR / "notebooks" / "generation" / "inference.ipynb",
    ...
    EXAMPLE_DIR / "notebooks" / "inference_for_tRNA" / "inference.ipynb",
]
```
Gated registry + timeout overrides (test_notebook_execution.py:570-597):
```python
GATED_NOTEBOOKS: list[tuple[str, object]] = [
    ("mcp_example/mcp_client_ollama_langchain_agents.ipynb", _gate_ollama_stack),
    ("mcp_example/mcp_client_ollama_pydantic_ai.ipynb", _gate_ollama_stack),
    ("notebooks/generation_evo_models/inference.ipynb", _gate_evo),
    ...
    ("notebooks/lora_finetune_inference/lora_inference.ipynb", _gate_mamba),
]
```
Gate helpers to copy for any new gate (test_notebook_execution.py:537-548, 477-496):
```python
def _gate_optional_deps(nb_name: str, modules: tuple[str, ...]) -> None:
    results = {name: _probe_module(name) for name in modules}
    missing = [name for name, (ok, _ev) in results.items() if not ok]
    if missing:
        evidence = "; ".join(f"{results[name][1]}" for name in missing)
        optional_dep_skip(f"execute {nb_name} (prerequisites install-gated)", evidence=evidence)
```
```python
def _probe_http(url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            return True, f"HTTP {exc.code} ({exc.reason})"
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # probe reports any transport failure verbatim
        return False, f"{type(exc).__name__}: {exc}"
```
The D-13 ollama readiness probe (retry ~60s x 2s over `curl http://127.0.0.1:11434/api/tags`,
evidence into skip message) composes `_probe_http` with the retry window; the 4xx-honesty
contract above (any HTTP < 500 = reachable) is tested in `TestProbeHonesty` (lines 325-368) —
extend that class if the probe gains retry behavior. Cross-dir sandbox inputs table
(`_NOTEBOOK_EXTRA_INPUTS`, lines 92-96) is where any newly-discovered sibling-input fix lands
(the benchmark `../inference/test.csv` precedent — never a cwd change).

---

### `tests/examples/test_marimo_execution.py` — D-18 deepening (test, batch)

**Analog:** itself; current assertions are export + size only (the gap D-18 closes):
```python
# Source: tests/examples/test_marimo_execution.py:62-73
        spec = MARIMO_EXEC_SPECS[str(app_path)]
        html_out = run_marimo_app(
            app_path,
            marimo_sandbox,
            timeout=spec["timeout_s"],
            artifact_dir=tmp_path / "artifacts",
        )
        assert html_out.is_file(), f"export artifact missing: {html_out}"
        assert html_out.stat().st_size > 1000, (
            f"export artifact undersized: {html_out.stat().st_size} bytes"
        )
        assert_tree_clean()
```
D-18 adds: UI-element default-value assertions + exit-code assertion (exit code is already
implicit in run_marimo_app raising on non-zero — _execution.py:607) + export-html key-content
check (read `html_out.read_text()` and assert app-level default strings; artifact discarded,
never committed). Keep the same parametrize-over-MARIMO_EXEC_SPECS shape so new apps join
automatically. Anti-pattern (research "Don't Hand-Roll"): no DOM scraping — assert app-level
default values via HTML content checks.

---

### `tests/examples/test_script_execution.py` — heal verification (test, batch)

**Analog:** itself. Once the NT snapshot is pristine-restored and dev-extra deps present,
the script should heal past the marker to real green — the typed-skip conversion below
either keeps working (marker never fires) or gets retired per planner:
```python
# Source: tests/examples/test_script_execution.py:92-105
        try:
            run_example_script(SCRIPT, sandbox, artifact_dir=tmp_path / "artifacts")
        except AssertionError as exc:
            # Ladder terminal (D-06): only the documented structural marker
            # converts to the typed skip; every other failure stays loud.
            if _NT_STRUCTURAL_MARKER in str(exc):
                environment_unavailable_skip(
                    "execute generate_bpe_dataset.py (zhangtaolab/"
                    "plant-nucleotide-transformer-BPE remote modeling_esm.py needs "
                    "removed transformers-4.x PretrainedConfig defaults)",
                    evidence=str(exc).strip().splitlines()[-1],
                )
            raise
```
The 4xx/5xx honesty split (lines 74-90) is the pattern for any new network probe: 5xx joins
`network-unavailable:` skip; 4xx re-raises loudly (WR-04 semantics).

---

### `example/notebooks/*` family repairs + `example/mcp_example/*` (content, transform)

**Analog:** sibling notebooks; the Phase-7 showcase notebooks are the newest precedent for
provenance stamps and hard environment guards.

**Provenance stamp (D-21) — exact current form, from `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb`:**
- Markdown cell: "**Environment:** the exact transformers / torch / flash-linear-attention
  versions this notebook ran with are printed by the first code cell as
  `transformers_version=`, `torch_version=` and `fla_version=` key=value lines."
- First code cell prints those key=value lines (via `importlib.metadata`).

Per D-21 each repaired notebook gets its OWN real stamp on re-execution — no cross-notebook
consistency requirement.

**Hard environment guard precedent (showcase first code cell):** spec-probe (never a bare
module import — the fast-lane `test_examples.py` execs every import statement) + hard error
on absence. Copy this for any new prerequisite guard cells (e.g. evo family prerequisites).

**Known repair targets (research Bucket 2):**
- `generation_evo_models/inference.ipynb`: model reference 131k→8k + evo2 noFP8 condition.
- `finetune_generation/finetune_generation.ipynb`: `!wget` ordering repair (cell 2 `Fasta(...)`
  died) + the unpinned `!git clone https://github.com/lingxusb/megaDNA.git` cell → pinned
  `cb2f5ab4...` + `MEGABYTE_pytorch==0.2.1` (Pitfall 8).
- `finetune_custom_head/finetune.ipynb`: demo cell repair.
- `source=` alignment edits (D-15) ride each family's repair commit.

**D-02/D-20 discipline:** structural refactor allowed (cell merge/split/reorder) but every
repair commit is atomic: notebook + byte-synced mirror + re-exported wrapper excerpts +
narrative; `check_notebook_md_sync` green is the gate. A cwd change is NEVER a repair
(benchmark precedent: fixed via `_NOTEBOOK_EXTRA_INPUTS` seeding).

---

### `docs/example/**/*.md` mirrors (docs, transform)

**Analog:** existing mirrors; regenerate with the committed generator (never hand-edit):
```
# Source: scripts/generate_md_from_notebook.py:4-6 (usage)
Usage:
    python scripts/generate_md_from_notebook.py <notebook_path> <output_md_path>
```
Frontmatter the generator emits (generate_md_from_notebook.py:97-101) — this is what opts a
mirror into the sync check:
```python
    lines = [
        "---",
        f"notebook: {notebook_rel}",
        "sync_check: true",
        "---",
```
Commit gate (D-20): `python scripts/check_notebook_md_sync.py` — AST-based, exit 1 on drift
(scripts/check_notebook_md_sync.py:11-16). Mirror layout: `docs/example/` mirrors
`example/` (e.g. `docs/example/notebooks/embedding_attention.md`).

---

### `tests/models/test_model.py` — allow_patterns regression (test, unit/mock)

**Analog:** `TestDownloadModel` (test_model.py:43-140) — mock downloader, assert kwargs:
```python
# Source: tests/models/test_model.py:46-53
    def test_download_success_first_attempt(self):
        """Test successful download on first attempt."""
        mock_downloader = Mock(return_value="/path/to/model")

        result = download_model("test-model", mock_downloader, max_try=3)

        assert result == "/path/to/model"
        assert mock_downloader.call_count == 1
```
```python
# Source: tests/models/test_model.py:131-140 (kwargs assertion precedent)
    def test_download_no_revision_error_resets_revision(self):
        mock_downloader = Mock(side_effect=Exception("no revision found: deadbeef"))

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError, match=r"Model test-model download failed."):
                download_model("test-model", mock_downloader, revision="deadbeef", max_try=2)

        assert mock_downloader.call_count == 2
        assert mock_downloader.call_args_list[0].kwargs["revision"] == "deadbeef"
```
New tests assert `mock_downloader.call_args.kwargs["allow_patterns"]` is passed when set and
ABSENT when None (the ModelScope branch must stay untouched). Import style: absolute imports
from `dnallm.models.model` (tests convention).

---

### `tests/models/test_special/*` — megaDNA tokenizer regression (test, unit/mock)

**Analog:** `tests/models/test_special/test_evo.py` — class-per-behavior grouping
(`TestEvoTokenizerWrapperInit` line 47, `TestEvoTokenizerWrapperCall` line 88,
`TestEvoTokenizerWrapperPersistence` line 183), plain `nn.Module` fakes
(`FakeLoadedEvo2Model` line 240), descriptive behavior-named tests
(`test_batch_padding_longest`). A new `test_megadna.py` (or extension of
`tests/models/test_special/test_family_handlers.py`) follows this layout for the DNATokenizer
generate-bug regression. Fast-lane friendly: no model downloads, pure unit shapes.

---

## Shared Patterns

### Typed skip + allowlist gate (applies to ALL new skip paths — D-04, D-13)
**Source:** `tests/examples/_execution.py:752-801` + `tests/expected_skips.yaml:29-41` + `scripts/audit_skips.py`
**Apply to:** every gate, probe fallback, and script-lane skip added this phase.
```python
# Source: tests/examples/_execution.py:785-801 (representative helper)
def network_unavailable_skip(action: str, evidence: str) -> None:
    pytest.skip(f"network-unavailable: {action} ({evidence})")
```
```yaml
# Source: tests/expected_skips.yaml:29-41
  - prefix: "network-unavailable:"
    category: network
  - prefix: "environment-unavailable:"
    category: environment
  - prefix: "optional-dep:"
    category: optional-dep
```
Evidence ALWAYS goes into the skip message (probe output, curl output, retry log). A typed
skip is never success — junit is audited against expected_skips.yaml; fallback (D-13) is for
missing infrastructure ONLY, never wrong model output. Never add empty/wildcard entries.

### Self-hosted nightly job skeleton (D-05/D-06/D-08)
**Source:** `coverage-nightly` + `test-mamba` jobs in `.github/workflows/ci.yml`
**Apply to:** the new `example-nightly` job only — coverage-nightly and test-mamba stay untouched.
Verbatim-copied elements: `runs-on: [self-hosted, dnallm-nightly]`, event gating
(`schedule || workflow_dispatch`), uv install steps, `timeout-minutes` with the
sum-of-ceilings comment discipline (ci.yml:414-423), junit + skip-audit steps,
`if: always()` artifact upload (ci.yml:337-344 precedent).

### Absence-gated idempotent shim (D-17 extension route)
**Source:** `dnallm/utils/transformers_compat.py` (9 precedent patches; excerpt above)
**Apply to:** any new transformers-5/numpy rung surfaced by evo under 5.17 (Pitfall 2 —
budgeted as discovery). Closed map / vendored-verbatim / sentinel / register in
`apply_patches()`. Never a catch-all, never fork transformers.

### Atomic repair commit (D-02/D-19/D-20/D-21)
**Source:** established Phase 5-7 discipline; toolchain: `scripts/generate_md_from_notebook.py`,
`scripts/check_notebook_md_sync.py`, `scripts/generate_md_from_marimo.py`
**Apply to:** every repair this phase. One commit = notebook edit + byte-synced mirror +
re-exported wrapper excerpts + narrative, OR library fix + same-change pytest in module
tests/ (owner memory rule: any `dnallm/` change ships with pytest coverage in the same
change). Triage FIRST: harness-bug vs content-bug vs library-bug; cwd change is never a
repair.

### Env-override sandwich (deterministic headless execution)
**Source:** `tests/examples/_execution.py:248-256, 374-390` (excerpt above)
**Apply to:** the per-notebook giants env extension and any new determinism envs. Save →
update → try/finally restore, around the subprocess/kernel spawn only.

### Regression-test layer ownership (D-19)
**Source:** `tests/examples/` vs `tests/<module>/` layout
**Apply to:** every regression test. Example-content fixes → `tests/examples/`; library fixes
→ module tests (`tests/models/`, `tests/utils/`, ...). Fast-lane unit tests use mocks
(TestDownloadModel pattern); execution tests stay slow-marked.

## No Analog Found

| File | Role | Data Flow | Reason / Use Instead |
|------|------|-----------|----------------------|
| `infra/` or `scripts/runner/` ollama systemd unit + README (new dir; planner names it per Claude's Discretion) | config (infra) | request-response | No in-repo service definition exists. Use 08-RESEARCH.md "ollama systemd unit" section verbatim: stock unit + `Environment="OLLAMA_HOST=127.0.0.1:11434"` (loopback-only, D-12), optional `OLLAMA_MODELS=` pin, genericized PATH, README documenting the one-time `systemctl enable --now ollama` + `ollama pull qwen3.8:latest` owner step. Do NOT copy the ollama FAQ's `0.0.0.0` binding. |
| evo-1 giants prefetch step (inline python in ci.yml) | config (CI step) | file-I/O | No in-repo inline-python prefetch step exists. Use 08-RESEARCH.md "evo-1 giants prefetch (CI step shape)" verbatim (`snapshot_download(..., allow_patterns=[...], cache_dir=~/models-giants/hub)`). |

## Metadata

**Analog search scope:** `tests/examples/`, `tests/models/` (+`test_special/`), `tests/configuration/`, `dnallm/models/`, `dnallm/models/special/`, `dnallm/utils/`, `dnallm/inference/`, `scripts/`, `.github/workflows/`, `docs/example/`, `models.lock`, `pyproject.toml`, `example/notebooks/` (showcase provenance)
**Files scanned:** 18 primary analog files read (line-targeted for >500-line files); directory listings for `tests/models/test_special/`, `docs/example/`, `scripts/`, `infra/` (absent), `scripts/runner/` (absent)
**Tracked-source gate:** all named analogs verified via `git ls-files` (non-empty)
**Pattern extraction date:** 2026-10-04
