# Architecture Research

**Domain:** Example-execution testing + PlantHelixSeek showcase integration onto the existing DNALLM test/CI architecture
**Researched:** 2026-10-01 (milestone v1.1)
**Confidence:** HIGH for everything verified against the repo and by local empirical probes (marimo 0.25.0 script mode, nbclient 0.11.0 semantics, registry/loader code paths, CI wiring); MEDIUM for upstream model-repo facts (HF cards, remote-code inspection — single-org sources); LOW where noted (evo/evo2 package runtime on the nightly box).

## Standard Architecture

### System Overview

The v1.1 additions slot in as **one new test layer** (real execution, nightly-only) and **one new example vertical** (PlantHelixSeek showcase). Nothing in the existing fast-leg/gate path changes semantics; the nightly census simply collects more `slow`-marked tests, and the model cache contract (`models.lock`) grows entries.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ EXAMPLE LAYERS (example/)                                                  │
│                                                                            │
│  NEW  plant_helixseek_cre/   notebook + config + committed ≤200kb loci     │
│       plant_helixseek_anno/  fragments + per-dir .gitignore for DLs        │
│  EXISTING  notebooks/ (19 ipynb) · marimo/ (3 apps) · mcp_example/ (2)      │
├────────────────────────────────────────────────────────────────────────────┤
│ TEST LAYERS (tests/examples/)                                              │
│                                                                            │
│  L1  STRUCTURAL (existing test_examples.py + test_yaml_load.py)            │
│      → fast leg (push/PR): JSON/syntax/import checks, real load_config()   │
│      → auto-discovers new notebooks+YAMLs via rglob — zero wiring needed   │
│                                                                            │
│  L2  EXECUTION (NEW — this milestone)                                      │
│      → @slow only → nightly census leg (bare pytest, GPU runner)           │
│      → nbclient NotebookClient per notebook (cwd-sandboxed tmp copy)       │
│      → marimo script mode `python app.py` per app (subprocess, cwd=tmp)    │
│      → generate_bpe_dataset.py real run (subprocess)                       │
│      → showcase truth assertions (CRE↔PlantDHS gff, Anno↔TAIR10 gff3)      │
│      → typed skips: network-unavailable: (ollama) / optional-dep: (evo*)   │
├────────────────────────────────────────────────────────────────────────────┤
│ SUPPORT CONTRACTS                                                          │
│  models.lock (hf/ms/dataset: prefixes → nightly cache key)   [EXTEND]      │
│  dnallm/models/model_info.yaml finetuned: section            [EXTEND]      │
│  load_model_and_tokenizer generic task-type route            [NO CHANGE]   │
│  expected_skips.yaml + scripts/audit_skips.py                [EXTEND]      │
│  docs/example mirror + check_docs_sync.py + mkdocs nav       [REPAIR]      │
├────────────────────────────────────────────────────────────────────────────┤
│ CI                                                                          │
│  coverage-gate (push/PR, -m "not slow")  ← unaffected                       │
│  coverage-nightly (self-hosted GPU, full census)  ← collects L2             │
│  docs-validation (WR-08: drop continue-on-error, add mcp extra) [REPAIR]   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | New / Modified / Existing | Notes |
|-----------|----------------|---------------------------|-------|
| `tests/examples/_execution.py` | Harness mechanics: nbclient wrapper, marimo subprocess runner, tmp-sandbox seeding, tree-cleanliness guard | **NEW** | Private (`_`-prefixed) module; mirrors `dnallm/mcp/tests/_network_skip.py` precedent |
| `tests/examples/conftest.py` | pytest fixtures exposing the harness (`notebook_sandbox`, `marimo_sandbox`, network/dep probes) | **NEW** | Example-scoped; do NOT add to root `tests/conftest.py` (shared by 1,656 tests) |
| `tests/examples/test_notebook_execution.py` | Parametrized real execution of all `example/notebooks/**/*.ipynb` + 2 mcp_example notebooks | **NEW** | `@slow`, per-test timeout marks |
| `tests/examples/test_marimo_execution.py` | Headless script-mode execution of the 3 marimo apps | **NEW** | `@slow` |
| `tests/examples/test_example_script_execution.py` | Real run of `generate_bpe_dataset.py` | **NEW** | `@slow` |
| `tests/examples/test_plant_helixseek_examples.py` | CRE/Anno notebook execution + truth-agreement assertions | **NEW** | Asserts id2label match + prediction↔experimental-truth overlap |
| `tests/examples/test_examples.py` | Structural layer | Existing | Unchanged; rglob auto-covers new files |
| `tests/configuration/test_yaml_load.py` | Real `load_config()` on every example YAML | Existing | Auto-covers new YAMLs on the **fast** leg — new configs must be valid from commit #1 |
| `dnallm/models/model_info.yaml` | Registry entries for PlantHelixSeek-CRE/-Anno | **MODIFIED** | `finetuned:` section; patterns already exist (binary promoter entries; `token` entry tRNAPointer with BILOU list) |
| `models.lock` | Nightly model-cache key | **MODIFIED** | Add ~8+ ids with `hf`/`ms` prefix matching the route each execution actually uses |
| `tests/expected_skips.yaml` | Typed-skip allowlist | **MODIFIED** | `network-unavailable:` prefix entry already matches any new ollama skip; add one `optional-dep:`-style prefix entry for evo/evo2 skips |
| `.github/workflows/ci.yml` | Nightly cache path for evo2 (only if evo2 executes) | **MODIFIED (conditional)** | evo2 package caches outside `~/.cache/huggingface` |
| `.github/workflows/docs-validation.yml` | WR-08/09 repair | **MODIFIED** | Remove `continue-on-error`, install `.[test,dev,mcp]` |
| `example/notebooks/plant_helixseek_{cre,anno}/` | New showcase vertical | **NEW** | Notebook + YAML + `data/` fragments + `.gitignore` |
| `docs/example/...` + `mkdocs.yml` nav | Mirror + docs nav | **MODIFIED** | Mirror sync currently broken (see Anti-Pattern 5) |
| `dnallm/models/model.py` loader | Generic task-type dispatch | **NO CHANGE** | Binary/token routes + `trust_remote_code=True` already forwarded to model *and* tokenizer |

## Recommended Project Structure

### Test side

```
tests/examples/
├── test_examples.py                      # existing structural layer (untouched)
├── conftest.py                           # NEW — execution fixtures, scoped here only
├── _execution.py                         # NEW — harness mechanics (import-only, no tests)
│     ├── run_notebook(nb_path, sandbox, cell_timeout) → executed nb / raises
│     ├── run_marimo(app_path, sandbox, timeout) → (exit_code, stdout)
│     ├── seed_sandbox(src_dir, sandbox, extra_files) → Path
│     └── assert_tree_clean() — `git status --porcelain example/ docs/example/`
├── test_notebook_execution.py            # NEW — parametrized over NOTEBOOK_FILES
├── test_marimo_execution.py              # NEW — 3 apps
├── test_example_script_execution.py      # NEW — generate_bpe_dataset.py
└── test_plant_helixseek_examples.py      # NEW — CRE + Anno showcase + truth asserts
```

**Structure rationale:**
- **Harness lives in `tests/examples/`, split into a private mechanics module + a conftest fixture layer** — not in `dnallm/` (test-only code must not ship in the wheel; `[tool.coverage.run] source_pkgs=["dnallm"]` wouldn't measure it anyway), and not in the root `tests/conftest.py` (fixtures would load for all 1,656 tests). This is the established seam: `dnallm/mcp/tests/_network_skip.py` + per-package conftest.
- Parametrized ids use `str(p.relative_to(EXAMPLE_DIR))` exactly like the structural layer, so junit names stay greppable for `audit_skips.py`.
- A `NOTEBOOK_EXEC_SPECS` dict in `_execution.py` (path → {cell_timeout, test_timeout, extra_inputs}) is the single per-notebook tuning table — keeps marks declarative and reviewable.

### Example side

```
example/notebooks/
├── plant_helixseek_cre/
│   ├── plant_helixseek_cre.ipynb         # dnallm API load → sliding-window scan → CRE track + peak calls
│   ├── inference_cre_config.yaml         # task: binary, num_labels: 2, label_names (checkpoint order!)
│   ├── data/                             # committed, ≤200kb total per region
│   │   ├── <locus>_chr*_*.fa             # genome fragment(s) of selected loci
│   │   └── TAIR10_DHSs_<locus>.gff       # experimental truth fragment (PlantDHS)
│   └── .gitignore                        # downloads/ *.fa.gz full-chromosome, results/, *.bigWig, *.gff3 (generated)
└── plant_helixseek_anno/
    ├── plant_helixseek_anno.ipynb        # dnallm API load → sliding-window token preds → BILOU decode → GFF3
    ├── inference_anno_config.yaml        # task: token, num_labels: 17, 17 label_names in checkpoint order
    ├── data/
    │   ├── <locus>_chr*_*.fa
    │   └── TAIR10_GFF3_<locus>.gff3      # truth fragment (gene annotation)
    └── .gitignore                        # same pattern
```

**Structure rationale:**
- Per-task subdirectory matches the existing convention (`inference_for_tRNA/`, `generation_megaDNA/`, …), which the structural tests' `rglob` and the docs mirror already generalize over.
- Per-dir `.gitignore` for heavyweight downloads mirrors `example/notebooks/finetune_NER_task/.gitignore` (ignores `*.gz`, generated beds) and `example/marimo/benchmark/.gitignore` (`results/`). Root `.gitignore` already covers `example/notebooks/*/output*/`, `*/results*/`, `*.pdf`, `.ipynb_checkpoints/` — add bigWig/GFF3-generated and arabidopsis-download patterns locally.
- Fragments are committed **inputs**: they get mirrored to `docs/example/` like every other input file (200kb is fine for the docs site) and are the deterministic fixtures the nightly execution tests run on. Full-genome downloads from arabidopsis.org remain gitignored intermediates — they are exercised either behind an opt-in notebook flag or not at all in CI (they live outside the HF/ModelScope caches that `models.lock` keys, so CI could not cache them anyway).

## Architectural Patterns

### Pattern 1: Two-layer example testing — structural fast, execution slow

**What:** Keep `test_examples.py` (JSON/syntax/import) + `test_yaml_load.py` (real `load_config()`) on the fast push/PR leg; put every real-execution test under `@pytest.mark.slow` so it lands only in the nightly census.

**When to use:** Always for this milestone. Marker selection *is* leg selection in this repo — the fast leg runs `-m "not slow"`, the nightly leg runs bare pytest.

**Trade-offs:** A notebook whose config is structurally valid but semantically broken (wrong `num_labels`) fails only nightly. Mitigated because `load_config()` Pydantic validation already gates YAML on the fast leg.

**Marker plan (explicit):**

| Concern | Decision | Evidence / mechanism |
|---|---|---|
| New markers | **None.** Reuse `slow` only | `--strict-markers` is on; leg split is already `-m "not slow"` vs bare; a new `network`/`example` marker would add a third selection axis CI never uses |
| Per-test timeouts > 300s | `@pytest.mark.timeout(N)` per test/class; marker overrides the `--timeout=300` ini default | Exactly the existing ladder: 900 (downloads), 1800 (real-model inference class), 3600/7200 (real finetune). Suggested for executions: inference notebooks **1800**, showcase CRE/Anno **3600**, marimo finetune/benchmark **7200** |
| Two-level timeout interplay | nbclient `timeout=` is **per cell** (verified empirically: `CellTimeoutError` fired at 2.5s with `timeout=2` on a 3s cell) — it is the *inner* guard; the pytest-timeout mark is the *outer* backstop | Set per-cell ~600–900s so a hung kernel self-reports before the outer mark kills the test process (outer kill can orphan the jupyter kernel child) |
| Network semantics | No network marker. Model downloads **never skip** (nightly runner has network + warm cache). External daemons/toolchains produce **typed skips**: `network-unavailable:` (ollama) and a new `optional-dep:` prefix (evo/evo2/stripedhyena) | `network-unavailable:` prefix entry already in `expected_skips.yaml` matches any ollama skip message; add one `optional-dep:` prefix entry. Untyped `pytest.skip` fails the run via `audit_skips.py` |
| Ollama notebooks | `@pytest.mark.slow`, nightly leg, probe `localhost:11434` (socket connect) in setup → skip `network-unavailable: ollama server not reachable on localhost:11434` | Same pattern as the 6 MCP live-server probes (`dnallm/mcp/tests/_network_skip.py::skip_if_unreachable`). Their *imports* (langchain-mcp-adapters, pydantic-ai, nest-asyncio) are already satisfied by the `mcp` extra inside `.[base]`; only the daemon is missing on the runner |

### Pattern 2: Registry-first model onboarding — generic loader, no special handler

**What:** PlantHelixSeek-CRE/-Anno are added to `model_info.yaml` `finetuned:` and loaded through the existing generic `_load_model_by_task_type` route. **No `special/` handler is needed.**

**Evidence (verified against repo code + HF repo inspection):**

1. The repos are `custom_code` models (`model_type: "HelixSeek"`, auto_map `model.HelixSeekForTokenClassification` etc.) — but the generic route already forwards `trust_remote_code=True` to `AutoModelForSequenceClassification` / `AutoModelForTokenClassification` (`dnallm/models/model.py:556-617`) and to the tokenizer (`dnallm/models/tokenizer.py:260` default, `:286`).
2. The model ids don't substring-collide with any special-handler trigger (`evo`, `megadna`, `gpn`, `enformer`, `space`, `borzoi`, `crossdna`, `dnabert2`, `mutbert`, `basenji2`, `omnidna`) — checked against the dispatch chain at `model.py:772-883` — so dispatch falls through cleanly to the generic loader.
3. The remote code's only **hard** third-party import is `einops` (`from einops import rearrange, repeat` at top of `attention.py`) — already a core dnallm dependency (`einops>=0.7.0`). `fla` (flash-linear-attention) and `flash_attn` are guarded (`FLA_AVAILABLE` try/except; `is_flash_attn_2_available()`) with pure-PyTorch fallbacks — verified by reading the raw remote files. `flash-linear-attention` (PyPI 0.5.2) can be added later purely for speed; it is **not** a correctness dependency.
4. Task-type coverage: `binary` → `AutoModelForSequenceClassification` with `num_labels/id2label/label2id/problem_type`; `token` → `AutoModelForTokenClassification` with `add_prefix_space=True` on the tokenizer — both exist today and are exercised by the tRNAPointer (`token`, 7 BILOU labels) precedent.

**Registry entries to add (following in-file patterns):**

```yaml
finetuned:
  - name: "PlantHelixSeek CRE"
    model: "zhangtaolab/PlantHelixSeek-CRE"
    base_model: "zhangtaolab/PlantHelixSeek"        # pattern: PlantCAD2 entries
    task:
      describe: "Predict cis-regulatory elements (open chromatin) in plant genomes by using PlantHelixSeek model."
      task_type: "binary"
      num_labels: 2
      label_names: ["Not CRE", "CRE"]               # confirm order vs checkpoint config.id2label during the phase
      threshold: 0.5
  - name: "PlantHelixSeek Anno"
    model: "zhangtaolab/PlantHelixSeek-Anno"
    base_model: "zhangtaolab/PlantHelixSeek"
    task:
      describe: "Predict gene structure (CDS/intron/5'UTR/3'UTR) per nucleotide by using PlantHelixSeek model."
      task_type: "token"
      num_labels: 17
      label_names: ['O','B-CDS','I-CDS','L-CDS','U-CDS',
                    'B-INTRON','I-INTRON','L-INTRON','U-INTRON',
                    'B-UTR5','I-UTR5','L-UTR5','U-UTR5',
                    'B-UTR3','I-UTR3','L-UTR3','U-UTR3']   # exact 17 BILOU tags from the HF card
      threshold: 0.5
```

**Trade-offs / caveats:**
- dnallm builds `id2label` from the config's `label_names` (`_create_label_mappings`) and passes it into `from_pretrained`, **overriding** the checkpoint's mapping. The list order must equal the checkpoint's id order, or predictions get silently permuted. The execution test must `assert model.config.id2label == expected` after load — and the truth-agreement assertion backstops it behaviorally.
- `_safe_num_labels` raises for `token` when `num_labels is None` (`model.py:671-678`) — YAML + registry must always carry it (they do).
- `attn_implementation: "eager"` is forced in `model_load_kwargs` — safe for custom code (eager is the only universally implemented path) and matches how every other custom-code model loads here.
- Upstream pins transformers 4.49; our span is 4.49–5.x. The remote code imports `transformers.cache_utils.Cache` / custom cache classes — a transformers-5 incompatibility here would surface as a load error on the first nightly run (dev env is transformers 5.17 — smoke-load locally first). This is the one genuine compat risk; the fix locus would be upstream remote code, so the fallback is a typed skip + upstream issue, not a dnallm shim.

### Pattern 3: Sandboxed execution — tmp-copy + cwd redirect

**What:** Each execution test copies the notebook (or marimo app) *plus its sibling input files* into a `tmp_path` sandbox and executes with `cwd=sandbox`: nbclient via `NotebookClient(...).execute(cwd=sandbox)` (cwd is a documented `start_kernel` kwarg), marimo via `subprocess.run([sys.executable, app], cwd=sandbox)`.

**Why:** Notebooks read inputs by relative path (`./inference_config.yaml`, `test.csv`) and write outputs the same way (`./results/`, PDFs, bigWig/GFF3). Redirecting cwd keeps inputs resolvable *and* keeps every generated artifact inside a pytest-managed temp dir that vanishes after the test — no in-tree strays, no gitignore whack-a-mole. This is the Phase-2 discipline ("PDF tests leave the tree clean") generalized.

**Belt-and-braces:** an `assert_tree_clean()` helper (`git status --porcelain example/ docs/example/`) run after the execution suite catches notebooks that write to repo-root gitignored dirs (`outputs/`, `results/`) or anywhere unexpected.

**Trade-offs:** Notebooks referencing files *outside* their own directory (e.g. `../`, repo-root paths) break in the sandbox. Handle via the per-artifact `extra_inputs` list in `_execution.py`'s spec table (copy/symlink those too). This is also a *repair signal*: an example that only runs from an undisclosed cwd is a docs bug.

### Pattern 4: models.lock as the nightly cache contract

**What:** `models.lock` (root) lists every remote artifact the slow suite fetches; `hashFiles('models.lock')` keys the nightly `~/.cache/huggingface/hub` + `~/.cache/modelscope/hub` restore. Extending it is the whole CI-wiring story for new models.

**Entries to add** (prefix MUST match the route each execution actually uses — `hf` for `source="huggingface"`, `ms` for `source="modelscope"`; zhangtaolab models have ModelScope twins, e.g. the CRE page exists at `modelscope.cn/models/zhangtaolab/PlantHelixSeek-CRE`):

```
hf  zhangtaolab/PlantHelixSeek-CRE            # new showcase (or ms: if notebook uses source=modelscope)
hf  zhangtaolab/PlantHelixSeek-Anno           # same route decision
hf  lingxusb/megaDNA                          # generation_megaDA notebook (+ megaDNA_updated variant)
hf  kuleshov-group/PlantCAD2-Small-l24-d0768  # finetune_custom_head
hf  InstaDeepAI/nucleotide-transformer-v2-50m-multi-species   # NT-50m reference in notebooks
hf  zhangtaolab/plant-dnagpt-BPE              # base model referenced 15x across notebooks — NOT yet locked
hf  togethercomputer/evo-1-131k-base          # generation_evo_models (typed optional-dep skip if pkg absent)
hf  arcinstitute/evo2_1b_base                 # same, see caveat below
# plus audit-driven: every other zhangtaolab task model the notebooks load that isn't already in the lock
```

**Caveats:**
- **evo2 caches outside the HF hub cache** (the `evo2` pip package manages its own model dir), so an `hf arcinstitute/evo2_1b_base` entry rotates the key but warms nothing. If evo2 execution is enabled, add its cache dir (e.g. `~/.cache/evo2`) to the nightly `actions/cache` path list — verify the actual path in-phase. evo2 0.6.0 additionally wants flash-attn 2.8 + CUDA 12.1+ (python 3.11/3.12) — on the aarch64 nightly box that is a source build; **recommendation: typed `optional-dep:` skip for the evo2/evo-1 execution tests in this milestone unless the box proves able** (confidence LOW that it runs cleanly there; treat enabling them as stretch).
- Any edit to `models.lock` rotates the cache key by design — batch all additions in one commit to avoid repeated cold nights.
- Update the file's header comment to say it covers artifacts fetched by the slow suite **and the example executions**.

### Pattern 5: Showcase truth-in-the-loop

**What:** The committed Arabidopsis fragments are chosen (during the phase) so that CRE predictions overlap PlantDHS `TAIR10_DHSs.gff` sites and Anno BILOU decodes match `TAIR10_GFF3` gene structures *substantially* on those loci. The execution test then asserts a floor on that agreement every night.

**Flow (per the upstream `scripts/cis_regulatory` / `scripts/gene_annotation` pipelines, re-implemented in-notebook on the dnallm API):**

```
committed locus FASTA fragment (≤200kb)
  → sliding-window scan (window/stride params from upstream script READMEs; in-notebook loop)
  → DNAInference.predict / token logits per window            [dnallm API]
  → stitch window scores back to genomic coordinates
  → CRE: score track (+ optional bigWig via pyBigWig) + peak calling → compare vs PlantDHS gff  (overlap floor assert)
  → Anno: argmax BILOU per base → B/I/L/U span decode → GFF3 records → compare vs TAIR10 gff3 (gene-level match floor assert)
```

**Trade-offs:** Choosing "prediction-matching-truth" loci is curation, not cheating, *provided* the assertion is a floor on substantial agreement (not identity) and the loci are documented in the notebook. Thresholds are a phase decision (start permissive, tighten once measured).

## Data Flow

### Notebook execution request flow

```
nightly census (bare pytest, GPU runner)
  → collect tests/examples/test_*execution*.py (slow-marked)
  → fixture: seed_sandbox() copies notebook + sibling inputs (+ extra_inputs) → tmp_path
  → NotebookClient(nb, timeout=cell_timeout, allow_errors=False, kernel_name="python3").execute(cwd=sandbox)
      → kernel: real load_config → load_model_and_tokenizer (HF/ModelScope, cache warm from models.lock)
      → real training / inference / plots → writes only inside sandbox
  → assertion: executed nb has no error outputs; (showcase) truth floors; assert_tree_clean()
  → junit xml → audit_skips.py gate
```

### Registry → loader → engine flow for the new models

```
model_info.yaml finetuned entry (metadata/UI/docs)
notebook YAML: task{task_type: binary|token, num_labels, label_names, threshold}
  → load_config() (Pydantic; fast-leg gated by test_yaml_load.py)
  → load_model_and_tokenizer(repo_id, task_config, source=…)
      → no special-handler substring match → generic _load_model_by_task_type
      → AutoModelFor{Sequence,Token}Classification.from_pretrained(trust_remote_code=True, num_labels, id2label, label2id)
      → tokenizer via load_tokenizer_with_fallback (trust_remote_code=True; add_prefix_space for token)
  → DNAInference(model, tokenizer, config) → predict/embeddings → notebook post-processing
```

## Scaling / CI Considerations

| Concern | Now (v1 shipped) | After v1.1 | Headroom check |
|---|---|---|---|
| Nightly census runtime | ~15 min warm | + 21 notebook/kernel launches + model loads + light training; realistically +1–3 h | Job `timeout-minutes: 900`; per-test marks are the primary guard — keep the *sum* of new marks well under it (≈24 new slow tests at 1800–7200s worst-case ≈ 14–48 h *serial worst case* — in practice most finish in minutes; if the sum becomes binding, split example execution into its own nightly job or shard) |
| Disk on runner | warm HF/MS caches | + ~2 GB per 0.5B F32 model (CRE+Anno ≈ 4 GB) + evo models if enabled | Cache is keyed/rotated; prune old restore keys when rotating |
| Fast leg | unchanged | +1 YAML × 2 (new configs, `load_config` only) — seconds | None |
| Coverage gate | 96.3% ≥ 90 | Unaffected (execution tests run in the nightly `--cov` run; dnallm code they execute only *adds* covered lines — but note the floor is enforced there too, so a catastrophic execution-skip day must not drop measured coverage below 90; skips don't reduce measured lines, only tests not run — safe) | None |
| Windows / matrix legs | `not slow` only | Never see execution tests | None |

### Scaling priorities

1. **First bottleneck: serial nightly wall-clock.** Mitigate by keeping per-cell nbclient timeouts tight (fail fast), warm caches (correct `models.lock` prefixes), and — if needed — a second nightly job.
2. **Second bottleneck: cache misses from wrong-prefix lock entries.** Each wrong `hf`/`ms` prefix silently costs a full re-download every night. Verify each entry against the notebook's `source=` the first green run.

## Anti-Patterns

### Anti-Pattern 1: Adding a `special/` handler for PlantHelixSeek

**What people do:** Write `_handle_planthelixseek_models` "because it's a new model family."
**Why it's wrong:** The generic task-type route already covers binary/token with `trust_remote_code` (verified). The guarded handler chain is where the Phase-2 CrossDNA result-overwrite bug lived — every handler added is dispatch risk for zero capability.
**Do this instead:** Registry entry + generic route; a smoke-load test (`load_model_and_tokenizer` + `id2label` assert) proves it on the nightly box. Only escalate to a handler if execution surfaces a real quirk.

### Anti-Pattern 2: Executing notebooks in-tree

**What people do:** Run nbclient with `cwd=example/notebooks/<dir>/` and rely on `.gitignore`.
**Why it's wrong:** The committed notebooks already drift (see Anti-Pattern 5); in-tree runs tempt "save outputs back into the committed ipynb", bloat the repo, and any un-ignored artifact strays. Phase 2 spent effort deleting exactly such strays.
**Do this instead:** tmp-sandbox + cwd redirect (Pattern 3) + `assert_tree_clean()`.

### Anti-Pattern 3: New markers or untyped skips

**What people do:** Add `@pytest.mark.network` / `@pytest.mark.notebook`; call `pytest.skip("no ollama")`.
**Why it's wrong:** `--strict-markers` + the two-leg `-m "not slow"` design means new markers are dead config; untyped skips fail `audit_skips.py` and thus the nightly job.
**Do this instead:** `slow` + typed prefix skips allowlisted in `expected_skips.yaml` (`network-unavailable:` entry already exists and prefix-matches; add one `optional-dep:` prefix entry).

### Anti-Pattern 4: Guessing the label order for Anno

**What people do:** Copy the 17 BILOU names from the paper in a different order than the checkpoint's `id2label`.
**Why it's wrong:** dnallm overrides `id2label` from config — a permuted list silently permutes predictions while shapes still validate (17=17). Truth-agreement would mysteriously collapse.
**Do this instead:** Read `config.id2label` from the loaded checkpoint during the phase, freeze that exact order into registry + notebook YAML, and assert equality in the execution test.

### Anti-Pattern 5: Trusting the docs mirror contract as-is

**What people do:** Assume `docs/example/` is synced and `check_docs_sync.py` is green.
**Why it's wrong (verified today):** `check_docs_sync.py` currently **exits 1** — wrapper `.md` files exist only in `docs/example/notebooks/`, several notebooks DIFFER (drifted outputs), and `generate_bpe_dataset.py` is missing from the mirror. docs-validation masks all of this with `continue-on-error` (WR-08).
**Do this instead:** As part of the mirror repair, teach `check_docs_sync.py` to ignore docs-only wrapper `*.md` files (or mirror them deliberately), re-copy drifted files byte-identically, then add the new showcase dirs to both trees + `mkdocs.yml` nav. New nav entries follow the existing wrapper pattern (`docs/example/notebooks/plant_helixseek_cre.md` → nav under Inference or a new Showcase group).

### Anti-Pattern 6: Putting the harness in `dnallm/` or the root conftest

**What people do:** Ship a `dnallm/utils/nb_runner.py` or load execution fixtures globally.
**Why it's wrong:** Test-only code enters the wheel and the coverage denominator; global fixtures slow every unrelated test session.
**Do this instead:** `tests/examples/_execution.py` + `tests/examples/conftest.py` (Pattern: `_network_skip.py`).

## Integration Points

### External services / artifacts

| Integration | Pattern | Notes / gotchas |
|---|---|---|
| HF `zhangtaolab/PlantHelixSeek-CRE` | `load_model_and_tokenizer(..., source="huggingface"\|"modelscope")`, `trust_remote_code` auto | 0.5B F32 (~2 GB) — budget runner disk; CC-BY-NC-4.0 license (fine for examples, note in notebook) |
| HF `zhangtaolab/PlantHelixSeek-Anno` | same | 17 BILOU labels; `token` task forces `add_prefix_space` tokenizer kwarg |
| ModelScope twins | `ms` prefix possible | Prefix must match notebook route or nightly cache never warms |
| arabidopsis.org downloads (TAIR10 FASTA/GFF3, PlantDHS gff) | gitignored per-dir intermediates; committed ≤200kb fragments are the CI inputs | Not cacheable via models.lock (outside HF/MS caches) |
| ollama daemon (`localhost:11434`) | socket probe → typed `network-unavailable:` skip | Also verify at runtime whether the langchain notebook needs `langchain-ollama` (imports today don't show it; execution will tell — if needed, either add to `mcp` extra or typed-skip) |
| `evo2` / `evo-1` (PyPI: evo2 0.6.0, evo-1 1.1.2, stripedhyena 0.2.2) | typed `optional-dep:` skip by default | evo2 needs flash-attn + its own cache dir (extend nightly cache path if enabled); confidence LOW these run on the aarch64 box this milestone |
| marimo 0.25 script mode | `subprocess python app.py` | Verified: UI elements yield default values headlessly; exit 1 on cell error; apps must set `value=` defaults on interactive elements (repair items otherwise) |

### Internal boundaries

| Boundary | Communication | Notes |
|---|---|---|
| `tests/examples/` ↔ `tests/conftest.py` | fixture inheritance only | Keep execution fixtures out of root conftest |
| execution tests ↔ `dnallm` public API | `load_config` / `load_model_and_tokenizer` / `DNAInference` exactly as documented | The tests are also API conformance tests — resist monkeypatching inside execution tests |
| execution tests ↔ `expected_skips.yaml` | junit messages → `audit_skips.py` | Every new skip path must land in the allowlist in the same PR |
| `model_info.yaml` ↔ loader | metadata only (UI/docs); runtime takes repo id + TaskConfig | Registry correctness is asserted by the smoke-load test, not by the loader |
| docs mirror ↔ `example/` | byte-identical copy enforced by `check_docs_sync.py` (post-repair) | New dirs must be copied + nav-wrapped; wrapper `.md` handling is part of the repair |

## Suggested Build Order (dependency-respecting)

1. **Registry + smoke-load** — add `model_info.yaml` entries; nightly-box/local smoke test loading CRE/Anno via the generic route, assert `id2label` order (freeze label lists). *No dependencies; unblocks everything.*
2. **Mirror repair + WR-08/09** — fix `check_docs_sync.py` semantics (wrapper `.md`), re-sync drifted mirror, drop `continue-on-error` in docs-validation, add `mcp` extra there. *Independent of 1; do early so new files land in an honest contract.*
3. **Execution harness** — `tests/examples/_execution.py` + `conftest.py`, piloted on 1–2 already-passing notebooks (e.g. `inference/`), tmp-sandbox + tree-clean guard. *Parallel with 1–2.*
4. **Showcase data selection** — pick loci, generate ≤200kb fragments + truth fragments, commit under `example/notebooks/plant_helixseek_{cre,anno}/data/` + per-dir `.gitignore`s. *Needs 1 (to measure prediction-truth agreement during selection).*
5. **New notebooks** — CRE/Anno notebooks on the dnallm API (sliding window, post-processing, truth comparison cells), YAML configs valid for the fast-leg `load_config` gate from the first commit. *Needs 1 + 4.*
6. **Per-artifact execution tests + models.lock extension** — parametrized execution over all notebooks/marimo/script; typed skips (ollama, evo*); extend `models.lock` (+ cache path for evo2 only if enabled); `expected_skips.yaml` entries. *Needs 3 + repaired examples; repair loop lives here.*
7. **Showcase truth assertions** — fold agreement-floor asserts into the CRE/Anno execution tests. *Needs 4 + 5 + 6.*
8. **CI wiring + census verification** — confirm nightly collects the new slow tests, junit/skip-audit green, cache warm (first-run watch), runtime within budget. *Last; validates everything.*

## Sources

- Repo-verified (HIGH): `tests/examples/test_examples.py`, `tests/configuration/test_yaml_load.py`, `tests/expected_skips.yaml`, `pyproject.toml` (pytest/coverage/extras), `.github/workflows/ci.yml` + `docs-validation.yml`, `models.lock`, `dnallm/models/model_info.yaml`, `dnallm/models/model.py` (dispatch, `_load_model_by_task_type`, `_safe_num_labels`, `_create_label_mappings`), `dnallm/models/tokenizer.py` (`load_tokenizer_with_fallback`), `dnallm/models/special/{evo,megadna}.py`, `scripts/check_docs_sync.py` (run: exit 1), `scripts/validate_yaml.py`, `.gitignore`, per-dir example `.gitignore`s.
- Empirical probes in the project venv (HIGH): marimo 0.25.0 script mode (default UI values, exit codes 0/1); nbclient 0.11.0 (`CellExecutionError`, per-cell `CellTimeoutError`, `execute(cwd=…)`).
- [zhangtaolab/PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek), [PlantHelixSeek-CRE](https://huggingface.co/zhangtaolab/PlantHelixSeek-CRE), [PlantHelixSeek-Anno](https://huggingface.co/zhangtaolab/PlantHelixSeek-Anno) model cards + HF API file listing + raw remote-code inspection (MEDIUM; single-org sources cross-checked against each other).
- [github.com/zhangtaolab/PlantHelixSeek](https://github.com/zhangtaolab/PlantHelixSeek) — `scripts/cis_regulatory/`, `scripts/gene_annotation/`, upstream pins (MEDIUM).
- [ModelScope zhangtaolab/PlantHelixSeek-CRE](https://modelscope.cn/models/zhangtaolab/PlantHelixSeek-CRE) page-title existence check (MEDIUM-LOW; body not rendered).
- [evo2 on PyPI](https://pypi.org/project/evo2/) + `pip index versions` for evo-1/stripedhyena/flash-linear-availability (MEDIUM).

---
*Architecture research for: DNALLM v1.1 — example execution testing + PlantHelixSeek showcase*
*Researched: 2026-10-01*
