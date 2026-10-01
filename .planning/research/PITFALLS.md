# Pitfalls Research

**Domain:** Adding real-model example-execution testing (nbclient notebooks, headless marimo apps, helper scripts, YAML configs) and PlantHelixSeek Arabidopsis showcase notebooks to an existing mature pytest/CI system (96.30% coverage gate, typed-skip allowlist, self-hosted GPU nightly runner)
**Project:** DNALLM (`dnallm`) — milestone v1.1 Example Execution Testing & Repair
**Researched:** 2026-10-01
**Confidence:** HIGH overall — top findings are grounded in direct inspection of this repo's notebooks/CI/skip-allowlist, plus verification against official docs (nbclient, pyBigWig, marimo, GitHub Actions limits, ollama, W&B). Item-level confidence marked inline.

## How this was verified

- **[REPO]** — verified by direct inspection of this repository on 2026-10-01 (file paths quoted).
- **[DOCS]** — verified against official documentation of the tool in question.
- **[WEB]** — community/secondary sources only; lower confidence, flagged.
- The nightly runner is an **aarch64 NVIDIA GB10 (Grace Blackwell)** box — verified locally (`nvidia-smi`: `NVIDIA GB10`); this fact drives several toolchain pitfalls below.

---

## Critical Pitfalls

### Pitfall 1: Leaked jupyter kernels after a failed/timed-out notebook test poison the persistent GPU runner

**What goes wrong:** A notebook cell hangs (model download stalls, OOM-thrash, infinite loop). The test times out, the nightly "moves on" — but the `ipykernel_launcher` process survives, still holding VRAM and the ZMQ port range. On an ephemeral hosted runner this is invisible; on the persistent `dnallm-nightly` box the next night's run starts with gigabytes of VRAM already consumed, and failures look like random CUDA OOM in *unrelated* tests. Three leak paths, all real: **[DOCS]** (a) pytest-timeout kills *the pytest process*, never the process group, and skips fixture teardown entirely (pytest-timeout issues [#134](https://github.com/pytest-dev/pytest-timeout/issues/134), [#159](https://github.com/pytest-dev/pytest-timeout/issues/159)); (b) nbclient's graceful kernel shutdown can hang on a busy kernel and must fall back to a kill; (c) a custom kernel-manager path makes *you* responsible for `shutdown_kernel()` (nbclient [#213](https://github.com/jupyter/nbclient/issues/213)).

**Why it happens:** Killing a process does not kill its grandchildren. Kernel lifetime spans the test that launched it, but every timeout mechanism in this stack (pytest-timeout signal/thread, job-level `timeout-minutes`) targets the wrong process.

**How to avoid:**
- Execute through `NotebookClient` **as a context manager** (`with NotebookClient(nb, ...) as client: client.execute()`), which installs SIGINT/SIGTERM cleanup and shuts the kernel down on cell errors — nbclient's error path *does* run kernel cleanup before re-raising (`on_notebook_error` fires "before kernel cleanup"). **[DOCS]**
- Set `shutdown_kernel="immediate"` for hard-realism cases where graceful shutdown hangs.
- Layer timeouts so **nbclient's own cell timeout fires first** (it raises `CellTimeoutError` in-process, cleanup runs, fixtures tear down normally). Order: nbclient per-cell timeout < `@pytest.mark.timeout(N)` per-test mark < remaining job budget. Never leave nbclient timeout at a wrapper-style 30–60s default with a 300s pytest mark — invert that and pytest-timeout kills the process with the kernel alive.
- Add a **post-run runner hygiene step** in the nightly job: `pkill -f ipykernel_launcher || true` + `nvidia-smi` VRAM assertion before cache save. Cheap, catches any leak path that evolves.
- Do not bypass jupyter_client with custom kernel managers.

**Warning signs:** `nvidia-smi` shows VRAM used before any test runs; nightly failures cluster on the *second* night after a first-night hang; orphan `ipykernel_launcher` processes in `ps aux`.

**Phase to address:** Phase 1 (execution harness bring-up) — the harness must ship with the context-manager pattern, timeout layering, and hygiene step on day one, before any real notebook runs.

---

### Pitfall 2: Timeout layering arithmetic breaks the nightly census (300s global vs per-test marks vs nbclient cell timeouts vs 900-min job)

**What goes wrong:** **[REPO]** `pyproject.toml` addopts impose `--timeout=300` globally; the nightly job already documents a delicate arithmetic: per-test ceilings sum to 840 min against a 900-min job kill, tuned so "a hung test fails via its own mark (junit + skip audit still run) instead of a job kill (no junit, model-cache forfeit)". Adding ~20 unbounded real-model executions (finetune notebooks run full training; benchmark runs multi-model × multi-dataset; evo-1 download alone is ~30 GB) without recomputing that arithmetic produces either (a) random job kills that forfeit the junit artifact and the model cache save, or (b) per-cell iopub watchdogs firing during long *silent* model loads (a cell that prints nothing for minutes trips output-inactivity timeouts, not runtime timeouts — different knob). **[DOCS]**

**Why it happens:** Three independent timeout systems (nbclient cell/iopub, pytest-timeout, GitHub job timeout) each default to values tuned for other workloads, and their interaction is only documented in a CI comment nobody re-reads.

**How to avoid:**
- Give every execution test an explicit `@pytest.mark.timeout(N)` sized above its nbclient cell timeout plus margin; register any new marker under `--strict-markers` (a `notebook`/`example-exec` marker must be declared in `pyproject.toml` or the run fails).
- Set nbclient `timeout` and `iopub_timeout` **explicitly per artifact class** (inference vs finetune vs evo) — never rely on defaults; a finetune cell can legitimately be silent for 10+ minutes.
- Update the ci.yml arithmetic comment and keep `sum(per-test ceilings) < 900 min` verified as a review checklist item whenever a mark changes.
- Measure actual runtimes in Phase 1 and record per-artifact budgets (mirrors the v1 "audit first" pattern).

**Warning signs:** Nightly killed at exactly `timeout-minutes` with no junit; `CellTimeoutError`/`Timeout waiting for IOPub output` on cells that are merely slow; census duration creep week over week.

**Phase to address:** Phase 1 (harness + measured baseline), revisited in the CI-wiring phase when the tests join the census.

---

### Pitfall 3: Notebooks execute with side effects against the repo — dirty git tree, cross-test contamination, and "repaired" notebooks that were never broken

**What goes wrong:** **[REPO]** The examples are not hermetic, by design (they are user-facing demos): `finetune_generation.ipynb` writes `ath_cds.csv` to cwd and sets `output_dir="./outputs_dnagpt"` (HF Trainer writes checkpoints + **tensorboard event files** — every example finetune config sets `report_to: "tensorboard"`); the NER `data_generation_and_inference.ipynb` runs `!wget -c https://rice.uga.edu/...` shell magics and writes BED files to cwd; `generate_bpe_dataset.py` writes the BPE pkl; kernels create `.ipynb_checkpoints/`. Executing in-place (a) dirties the git tree (Phase 2 of v1 had to fix exactly this class of bug for PDF tests — "autouse tmp_path rebind, gitignore fixed, 9 strays deleted"), (b) lets one notebook's outputs feed the next test, and (c) invites **wrong repairs**: the \#1 false positive is `FileNotFoundError: ./inference_evo_config.yaml` because the harness ran the kernel with cwd = repo root, misread as a notebook bug and "fixed" by editing the notebook.

**Why it happens:** Every notebook assumes it runs from its own directory (`load_config("./xxx.yaml")` in all of them). A pytest process naturally has cwd = repo root. nbclient only sets the kernel cwd if you pass `resources={"metadata": {"path": ...}}`.

**How to avoid:**
- **Copy, don't execute in place:** per test, `shutil.copytree` the artifact's example dir (notebook + YAMLs + the marimo app's `plant_DNA_LLMs_finetune_list.xlsx`, verified present at `example/marimo/inference/`) into `tmp_path`, execute with kernel cwd = the copy, and pass `resources={"metadata": {"path": str(tmp_copy)}}`. This is the established repo pattern (v1 Phase 2 PDF fix) generalized.
- **Never write executed notebooks back** (`--inplace` / overwriting the source `.ipynb`): persist the executed copy as a CI **artifact on failure only** for debugging. 19 of 21 notebooks carry committed outputs **[REPO]** — an in-place rewrite produces a 100k-line diff (the evo notebook is 164 KB) and destroys the curated outputs.
- Add a **tree-cleanliness guard**: session-scoped check on the nightly that `git status --porcelain` is empty after the census (excluding known cache dirs), so any new side-effect path fails loudly once instead of silently straying.
- Extend `.gitignore` *before* first execution for the known intermediates: genome archives (`osa1_r7.*`), `*.pkl` BPE datasets, `*.bw`/`*.bigwig`, DHS/GFF downloads, `ath_cds.csv`, `outputs_*`. **[REPO]** Current patterns cover `example/notebooks/*/output*/`, `results*/`, `*.pdf` — but nothing for the showcase-data class. Note `output*/` does match `outputs_dnagpt/` but only under `example/notebooks/*/`.
- Triage rule for the repair workflow: a failure that reproduces only under the harness (cwd, env, ports) is a harness bug; log it in a harness-vs-content triage list so "fix the example" doesn't mask "fix the harness".

**Warning signs:** `git status` noise after a local execution run; diffs containing tensorboard `events.out.tfevents.*` or `ath_cds.csv`; a repaired notebook whose fix is a path change rather than a code change.

**Phase to address:** Phase 1 (harness hermeticity) — the copy-to-tmp isolation and the guard test must exist before the first real execution; gitignore additions land in the same phase.

---

### Pitfall 4: Huge model downloads vs actions/cache quota — one 30 GB model evicts the entire existing warm cache

**What goes wrong:** **[DOCS]** GitHub's cache service enforces a **10 GB per-repository** default (LRU eviction + 7-day idle eviction), *regardless of runner type* — the self-hosted box does not exempt you. **[REPO]** The nightly job caches `~/.cache/huggingface/hub` + `~/.cache/modelscope/hub` keyed on `models.lock` (8 models today, comfortably under quota). **[DOCS/WebFetch]** `togethercomputer/evo-1-131k-base` is a **29.7 GB repo** — 3 safetensors shards (~12.9 GB) *plus a redundant* `pytorch_model.pt` (16.8 GB); a plain `snapshot_download` grabs all of it. Adding evo-1 + evo2 + megaDNA to the cached paths blows the quota: GitHub saves the new cache and immediately LRU-evicts the old one — the *existing* slow-suite warm cache disappears, nightly cold-starts (or worse, partially restores), and runtimes explode for reasons nobody connects to the new notebooks. The box itself has 2.4 TB free **[REPO]** — raw disk is not the problem; the cache *service* is.

**Why it happens:** The cache path glob is shared by all models, and eviction is silent and cross-entry. Nobody re-reads the quota rule when adding "just one more model".

**How to avoid:**
- **Filter the download, not just the cache:** use `allow_patterns` (safetensors + configs, exclude the redundant `.pt`) when fetching evo-1 — 12.9 GB instead of 29.7 GB. **[DOCS/WebFetch]**
- **Split cache tiers:** keep the small/medium model set in `actions/cache` (bounded, quota-safe) and hold giant artifacts (evo-1, evo2) in a **persistent on-disk directory outside the cached paths** (e.g. `~/models-big`), warm-once and never evicted — a pinned-revision model file is immutable so cache-keying adds nothing.
- Pin **revisions** in `models.lock` (see Pitfall 8) so a warm on-disk artifact is provably the reviewed one.
- Add a disk-headroom + cache-size report step to the nightly so growth is visible.

**Warning signs:** Nightly log shows `Cache not found` for a key that existed last week; restore step suddenly takes tens of minutes; `gh cache list` shows entries vanishing.

**Phase to address:** Phase 1/2 boundary — cache strategy must be decided before the first evo-class execution test runs on the runner (that first run is the one that evicts everything).

---

### Pitfall 5: evo/megaDNA toolchain infeasibility on the aarch64 GB10 runner — tests written for models that cannot ever run there

**What goes wrong:** **[REPO + DOCS]** The runner is aarch64 GB10 (Blackwell). The evo notebook's own install cell pins `flash_attn<=2.7.4.post1`, `transformer-engine[pytorch]==2.3.0`, `evo2==0.3.0`: flash-attn ships x86_64 CUDA wheels and needs a long source compile on aarch64; **[DOCS]** transformer-engine 2.3.0 predates Blackwell support, and the evo2 package docs state the **1B/20B/40B models require FP8 via Transformer Engine on an NVIDIA *Hopper* GPU** — GB10 is not Hopper (`is_fp8_capable()` in `dnallm/utils/support.py` checks compute capability ≥ 9.0; the evo YAML config selection will downgrade behavior). Separately, the megaDNA notebook instructs `git clone https://github.com/lingxusb/megaDNA && pip install -e .` — an **unpinned, unmaintained third-party repo** compiled against whatever torch shows up (2.11 today). Writing execution tests for these as "will pass once fixed" burns weeks: they are *environment*-gated, not *content*-gated, and no amount of example repair fixes them.

**Why it happens:** The notebooks were authored on x86_64 GPU workstations. "The CI box has a GPU" hides that GPU *class* and CPU *arch* differ, and that the evo2 package's hardware requirements are stricter than "has CUDA".

**How to avoid:**
- **Capability-spike first:** in the earliest phase, empirically determine per-artifact feasibility on the actual box (evo-1 stripedhyena vs the HF-format variant, evo2-1b on Blackwell, megaDNA editable-install against torch 2.11) and record the verdict matrix before writing tests.
- Introduce a typed **`environment-unavailable:`** skip category (mirroring the existing `network-unavailable:` helper) with a narrow allowlist entry — an honest, audited skip for "this artifact's toolchain cannot exist on this runner", not a silent one.
- Prefer the smallest viable variant per family for execution (e.g. an evo2 tier that runs without TE, or the hf-format evo-1) and document the deviation from the notebook's as-shipped model name.
- Never let the megaDNA `git clone` path execute on the runner unpinned; if it must run, vendor a pinned, hash-verified copy or a small shim.

**Warning signs:** Multi-hour flash-attn/TE compile steps recurring every run (no wheel cache); `Failed to build transformer_engine_torch` (evo2 [#149](https://github.com/arcinstitute/evo2/issues/149), [#201](https://github.com/arcinstitute/evo2/issues/201)); tests "passing locally" on x86_64 but skipping nightly.

**Phase to address:** Phase 1 (feasibility spike + skip taxonomy) — must precede the execution-test rollout phase for these families.

---

### Pitfall 6: Network flakiness tempts silent skips — reintroducing exactly what the typed-skip discipline was built to kill

**What goes wrong:** Execution tests depend on many more network endpoints than the current slow suite: HF, ModelScope, `rice.uga.edu` (NER genome + GFF), `arabidopsis.org`/`plantdhs.org` (showcase truth data), the ollama model registry. Under flakiness, the path of least resistance is a broad `try/except Exception: pytest.skip()` — which **[REPO]** `scripts/audit_skips.py` will fail CI for (good), pushing developers to the *second* trap: adding an over-broad allowlist entry (`reason_like: "download failed"`) that silences every future network skip, converting the allowlist back into a silencer. A third variant: catching the failure and asserting nothing, so the test "passes" without executing the notebook.

**Why it happens:** The skip audit catches *unlisted* skips, but listing is self-service. Discipline erodes at 2 AM against a flaky mirror.

**How to avoid:**
- Reuse the existing typed helper pattern (`skip_if_unreachable` with a **`network-unavailable:`**-prefixed reason) for every new network gate; add endpoint-specific entries to `tests/expected_skips.yaml` with **narrow matchers** and a category — never wildcards, never empty.
- Distinguish *deterministic* environment skips (`environment-unavailable:` for toolchain, Pitfall 5) from *transient* network skips; they need different categories so a nightly report can say "3 real skips" instead of muddling them.
- Retry-with-backoff *inside* the fetch (the repo already does this for model downloads: `download_model(..., max_try=3)`) so transient blips never become skips at all.
- Keep the audit wired to the **nightly junit** (it already runs there) and treat any growth in skip count as a review-triggering event.

**Warning signs:** Census skip count drifting upward across nights; an allowlist entry whose matcher grows vaguer over time; execution tests with large `try/except` bodies.

**Phase to address:** Phase 1 (skip-taxonomy extension ships with the harness), enforced continuously by the existing audit gate.

---

### Pitfall 7: marimo `App.run()` executes **in-process** — UI defaults silently decide what runs, and app state leaks into pytest

**What goes wrong:** **[DOCS]** marimo's `App.run(defs)` returns `(outputs, defs)` and is the sanctioned programmatic entry — but unlike a jupyter kernel it runs **inside the hosting Python process**. Consequences: (a) the app's globals, CUDA allocations, and any `nest_asyncio`-style patches land in the *pytest* process and persist across tests; (b) `mo.ui.dropdown` values come from their `value=` defaults **[REPO]** (`'open chromatin'`, `'Plant DNABERT'` in `example/marimo/inference/inference_demo.py`) — headless runs silently execute whichever model/task the *default* names, so a default pointing at a heavy finetune turns an "execution smoke test" into a full training run; (c) `defs` overrides are all-or-nothing — marimo **skips execution of the cells that would define overridden variables and requires you to supply every definition those cells produced** **[DOCS]**, so partial overrides produce `NameError`s that look like app bugs; (d) the app reads `./plant_DNA_LLMs_finetune_list.xlsx` by relative cwd **[REPO]**.

**Why it happens:** marimo apps look like scripts but are reactive graphs; the testing instinct "import and call run()" ignores that the runtime model differs fundamentally from nbclient's kernel isolation.

**How to avoid:**
- Execute marimo apps in a **subprocess** (small wrapper: `python -c "import app_mod; outputs, defs = app_mod.app.run()"` with cwd = tmp copy, or `marimo export html --headless`), enforcing isolation and a clean VRAM/GC story per app — consistent with how jupyter notebooks get kernel isolation.
- Before running, **assert the default-driven execution plan is the cheap one**: parse the app source for default dropdown values and check them against a allowlist of "small/fast" models, or override via `defs` *completely* (supplying the full set of names the skipped cells define).
- Copy the app dir (including the xlsx) to tmp and run with cwd there.
- Assert on returned `defs` keys/shape (e.g. a metrics or engine variable exists and is finite), not merely "no exception".

**Warning signs:** An execution "smoke test" for a marimo app takes 30+ minutes; pytest process RSS/VRAM grows monotonically across app tests; `NameError` on variables the app defines in UI-driven cells.

**Phase to address:** Phase 2 (marimo execution rollout), with the subprocess harness pattern fixed in Phase 1.

---

### Pitfall 8: `trust_remote_code` + unpinned model refs = unreviewed arbitrary code execution on the self-hosted box

**What goes wrong:** **[DOCS/WebFetch]** `togethercomputer/evo-1-131k-base` is a stripedhyena/custom-code repo that executes repo-hosted Python (`model.py`, `engine.py`, `modeling_hyena.py`, …) via `trust_remote_code=True` — verified in its file listing. **[REPO]** DNALLM's loader passes `trust_remote_code=True` across 35 families, and models arrive from both HF and ModelScope. A notebook's `model_name` string is therefore an **indirect code-execution vector**: if the upstream repo re-pushes code (or a tag moves), the *same passing test* silently executes different code next run. The repo's security posture — PR-authored code never reaches the runner (dispatch/cron-only) — already handles repo content; the remaining hole is *remote* content pulled at runtime by that repo content. megaDNA's `git clone … && pip install -e .` instruction is the same hole one notch worse (arbitrary setup.py execution, unpinned).

**Why it happens:** Model repos are treated as data, but custom-code repos are data *plus code*. Nothing in a green test tells you the code you executed is the code you reviewed.

**How to avoid:**
- **Pin revisions (commit hashes) for every model the execution tests fetch** — extend `models.lock` with a revision column and have the harness/notebook resolve through it; rotation of a pin is then a reviewable diff. **[DOCS]** (huggingface_hub `revision=`)
- Record the provenance chain per artifact in the test (repo id + resolved commit) and print it in the junit log.
- Never execute the megaDNA git-clone path unpinned; vendor or shim (Pitfall 5).
- Treat `models.lock` edits as security-relevant review surface, exactly as the dispatch/cron posture is.

**Warning signs:** A nightly run downloading "the same" model again with no lock change; `revision` absent from fetch calls; new `model_name` literals in notebooks without a lock entry.

**Phase to address:** Phase 1 (harness resolves through pinned lock), Phase 2 onward (every new family extends the lock with pins).

---

### Pitfall 9: ollama on the self-hosted runner — unauthenticated service, port collision with the MCP live-server probes, and nondeterministic LLM assertions

**What goes wrong:** **[REPO]** The two `example/mcp_example` ollama notebooks (`mcp_client_ollama_pydantic_ai.ipynb`, `mcp_client_ollama_langchain_agents.ipynb`) need three live services: ollama REST at `localhost:11434` with model `qwen3.6:latest` pulled, the dnallm MCP server at `localhost:8000/mcp`, and a working agent loop. **[DOCS]** ollama has **zero authentication** on its REST API (default bind `127.0.0.1:11434`; unauthenticated model pull/delete; localhost binding remains reachable from a browser page via DNS rebinding — ollama [#16236](https://github.com/ollama/ollama/issues/16236)). **[REPO]** Port 8000 is *also* the target of the 6 existing MCP live-server probes that currently skip as typed `network-unavailable:` — start a real server on 8000 for the notebook and those probes may un-skip mid-census (ordering-dependent), or collide if anything else binds the port. Finally, asserting on the *agent's prose* is hopeless: the LLM may or may not call tools, and `time.sleep(3)` waits are not synchronization.

**Why it happens:** The notebooks were demoed on a developer workstation where ollama, the model, and the server were already up. CI needs all three as *managed, idempotent setup* plus assertions matched to what is actually deterministic.

**How to avoid:**
- Run ollama as a runner-user systemd unit with explicit `OLLAMA_HOST=127.0.0.1:11434`; **pre-pull the pinned model tag in a nightly setup step** (idempotent `ollama pull`), not inside tests; never set `OLLAMA_ORIGINS` to a wildcard.
- Assign the example's MCP server its **own port** (server supports `--port`); if the notebook hardcodes 8000 (it does — 3 references **[REPO]**), either serialize and use 8000 deliberately or patch the URL cell in the tmp copy (copy-to-tmp from Pitfall 3 makes this clean).
- Assert the deterministic layers: server started and reachable; `_list_loaded_models` tool returns the configured models; the agent run completes and `result.usage` exists. Do not assert specific prose or that tools were called N times.
- Keep these tests on the dispatch/nightly-only path (they already will be, being `slow`) — preserving the "PR code never touches the runner" invariant, which now also covers *services the repo depends on*.

**Warning signs:** MCP probes flipping between skip/run across nights; tests failing only when ollama's model list changed; assertions matching English sentences.

**Phase to address:** Phase 2/3 (ollama-family execution), with runner service setup landing in the CI-wiring phase.

---

### Pitfall 10: Executed-notebook nondeterminism turns the suite flaky — sampling, seeds, GPU float drift, tqdm, plotting backends, stale committed outputs

**What goes wrong:** **[REPO]** megaDNA generation runs `temperature=0.95, top_p=0.1`; only some notebooks set seeds (`finetune_generation`, `data_generation_and_inference` seed; most others don't); GPU reductions are float-nondeterministic across runs/devices; tqdm progress and matplotlib rendering differ headlessly; and the committed notebook outputs **cannot be trusted as ground truth** — e.g. `generation_megaDNA/inference.ipynb` says `source="huggingface"` in source but its committed output shows a ModelScope download path, i.e. the notebook was edited after last execution **[REPO]**. Assertions on exact values, exact output text, or "outputs match the committed ones" will flake or encode stale behavior.

**Why it happens:** Demos optimize for *looking* deterministic (nice printed numbers), and committed outputs create an illusion of a golden master.

**How to avoid:**
- Assert **structure and invariants**: generated sequence is valid DNA over the model's alphabet and non-empty; scores are finite and in expected ranges; shapes/dtypes of embeddings; metric keys exist and are within `[0, 1]`. Use **tolerance bands** for any numeric agreement (also required for cross-device drift — the showcase assertion must tolerate GB10-vs-x86 float differences).
- Seed at the harness layer where the notebook API allows (`set_seed` before kernel cells is not possible without editing the notebook — prefer tolerance over edits; if a notebook exposes seed config, set it in the copied YAML).
- Kernel env hygiene: `MPLBACKEND=Agg`, `WANDB_MODE=disabled` **[DOCS]** (belt-and-braces even though all example configs say `report_to: "tensorboard"` — one config drift to `wandb`/`all` would otherwise hang the nightly on a login prompt; tensorboard event files are a tree-cleanliness issue covered by Pitfall 3), pass via the kernel `env` dict.
- Treat committed outputs as historical context only; the executed-copy artifact (Pitfall 3) is the debugging record.
- For the **showcase truth-agreement assertion**, re-verify the threshold at loci-selection time and after any model-revision rotation (Pitfall 8) — the assertion guards regressions of the *pipeline*, not the physics.

**Warning signs:** The same test failing on a cadence unrelated to commits; assertion diffs that are all float-precision; failures that disappear on re-run.

**Phase to address:** Phase 1/2 (assertion conventions ship with the harness); tolerance policy for the showcase lands with the PlantHelixSeek phase.

---

### Pitfall 11: BigWig/GFF3 coordinate and chrom-naming traps — systematic off-by-one and *silently empty* results

**What goes wrong:** **[DOCS]** pyBigWig (and the bigWig/bigBed formats) use **0-based half-open** coordinates — "the first base of chr1 is start=0, end=1" — while **GFF3 is 1-based fully-closed**. Converting DHS/gene intervals from GFF3 to BigWig query space without `-1` on start shifts every window by one base. Worse: bigWig chrom names are **case-sensitive and exact-match** — TAIR-family files use `Chr1`…`Chr5` while Ensembl Plants uses `1`…`5` **[WEB, corroborated]** — and a wrong chrom name does not error: `bw.values("1", …)` on a `Chr1`-named file returns an **empty array**. A showcase notebook can "run green" while every signal extraction is vacuous, and the CRE/Anno agreement numbers become garbage that still passes a loose threshold. The repo already contains the correct idiom (`generate_bpe_dataset.py`: `start = int(info[3]) - 1`) — but only in one hand-rolled place. **[REPO]**

**Why it happens:** Three conventions (BED-style 0-based half-open, GFF3 1-based closed, and per-source chrom naming) meet in one notebook, and the failure mode is *silence*, not exceptions.

**How to avoid:**
- One shared, unit-tested normalization helper (GFF3→0-based, chrom-name mapping table) used by both showcase notebooks; unit-test it against tiny committed fixtures (a 200 bp region, one exon, one DHS).
- **Assert non-emptiness everywhere signal is extracted** (`assert len(vals) == expected_window_len`, `assert entries` non-empty) so a naming mismatch fails loudly.
- Pin the naming convention at data-selection time: inspect `bw.chroms()` and the GFF3 first column *once*, record the mapping in the notebook markdown, and assert it in the test.
- For the Anno 17-BILOU token task: the label set in the notebook, config `num_labels`/label maps, and dataset tags must match exactly — a mismatch crashes or silently trains garbage; add a fixture test asserting the 17-label vocabulary. **[REPO-derived requirement]**

**Warning signs:** Agreement metrics suspiciously flat/zero; `bw.values()` returning `[]`; every window scoring identically; per-base arrays shorter than window length.

**Phase to address:** PlantHelixSeek showcase phase (helper + fixtures first, notebook second).

---

### Pitfall 12: arabidopsis.org programmatic download blockers — HTML saved as `.gz`, login walls, and the showcase data that silently isn't there

**What goes wrong:** **[REPO-context, MEDIUM]** arabidopsis.org serves an SPA shell to non-browser clients (milestone-context knowledge); **[WEB]** TAIR's download infrastructure has shown "Cannot load directory content" states and login/ORCID requirements that break programmatic access; classic symptoms are `wget`/`requests` receiving 200 with an HTML body saved as `TAIR10_*.gz`, which then either crashes the parser or — with a lenient reader — produces nothing. The full-genome intermediates are planned to stay gitignored; the ≤200 kb committed showcase regions are the *only* guaranteed-present truth data. If the selection pipeline depends on a live arabidopsis.org fetch succeeding, loci selection is itself flaky and unauditable.

**Why it happens:** Data portals optimized for browsers; naive HTTP clients don't check content types; `.gz` magic bytes differ from HTML's `<`.

**How to avoid:**
- **Validate magic bytes** on every download (`gzip` header `1f 8b`, bigWig header `0x888FFC26`, GFF3 = text starting with `##gff-version 3`) before parsing; retry with a browser `User-Agent`; prefer `plantdhs.org/Download` direct file links (verified hosting TAIR10 DHS gff + bigwig) over arabidopsis.org pages. **[WEB]**
- Make loci selection a **one-time, committed-artifact-producing step**: the selection script may use live downloads, but its outputs (the ≤200 kb region slices + the documented selection rationale) are committed and reviewed — tests then depend only on committed data plus model inference, never on arabidopsis.org being up.
- Keep the gitignore for full-genome intermediates (Pitfall 3) aligned with the selection script's output paths so the repo can't accidentally absorb GB-scale files.

**Warning signs:** Downloaded "`.gz`" files that `gunzip` rejects; selection scripts that only work on the author's machine; showcase tests skipping whenever the portal is down.

**Phase to address:** PlantHelixSeek showcase phase, first task (data acquisition + validation), before any notebook is written against the data.

---

### Pitfall 13: Showcase overfitting — a cherry-picked locus presented as a benchmark claim

**What goes wrong:** The milestone *requires* selecting loci where predictions are "substantially consistent with experimental truth", then asserting that agreement in tests. Presented carelessly ("our model achieves X% agreement on Arabidopsis"), this is circular: the loci were chosen because they agree. Readers (and downstream docs/marketing) will cite the number as performance. The test itself is fine as a **regression guard**; the framing is the pitfall. Related: the assertion threshold chosen *on the same data it asserts* guarantees it passes at selection time and tells you nothing about generalization — and one model-revision rotation can silently invalidate it.

**Why it happens:** The guarantee ("predictions match truth on selected loci") is a legitimate *demonstration* device that looks structurally like a *benchmark*.

**How to avoid:**
- Fix the framing in the notebook and docs: "illustrative loci, selected because model output is consistent with experimental data (selection criteria: N regions screened, criterion C)" — never "accuracy/performance".
- Have the test assert the **documented threshold with a tolerance band**, and treat threshold changes as review-worthy diffs tied to a re-run of the selection rationale.
- Include at least one *negative-control* region (expected mismatch) so the pipeline demonstrably distinguishes signal from noise — cheap insurance against the everything-is-empty failure of Pitfall 11 masquerading as agreement.

**Warning signs:** Docs/README language drifting toward performance claims; a threshold nobody can re-derive; no record of how many loci were screened.

**Phase to address:** PlantHelixSeek showcase phase (selection-rationale doc is a deliverable), enforced at review.

---

### Pitfall 14: Repair-scope explosion through the `docs/example/` mirror and the WR-08 gate flip

**What goes wrong:** **[REPO]** Every notebook fix must propagate to the `docs/example/` mirror (`scripts/check_docs_sync.py`, `check_notebook_md_sync.py`, `generate_md_from_notebook.py`). Fixing WR-08 (removing `continue-on-error` from docs-validation) **turns the gate honest immediately** — at which point any accumulated mirror drift goes red on the next push, blocking unrelated work until reconciled. Doing mirror-sync as an end-of-milestone batch multiplies merge pain; doing gate-first without a drift inventory strandings the branch.

**Why it happens:** The false-green hid drift for months (that's what WR-08 *is*); an honest gate converts hidden debt into immediate failures.

**How to avoid:**
- Sequence explicitly: inventory mirror drift **first** (the audit pattern), fix WR-08 **with** the drift-closing changes in one reviewable unit, then keep sync tooling in the loop per notebook repair (regenerate MD as part of each fix) so drift can't re-accumulate.
- Budget repair work per wave with a visible ledger (the v1 audit-report pattern), so "fix every error execution surfaces" has a bounded, ranked queue rather than an amorphous backlog.

**Warning signs:** Docs-validation red on PRs that never touched docs; the same mirror diff reappearing in multiple PRs.

**Phase to address:** The CI-gate-repair phase (WR-08/09), with the per-fix mirror-sync habit starting in the first repair wave.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `try/except: pytest.skip()` around whole notebook executions | Stops nightly flakiness now | Silent skips destroy the census's meaning; audit red-flags force hasty allowlist widening | Never — use the typed helper + narrow allowlist entry |
| Executing notebooks in-place (no tmp copy) | Harness is 10 lines shorter | Dirty tree, cross-test contamination, wrong "repairs" for cwd bugs, 100k-line diffs | Never |
| `timeout=None` everywhere "because models are slow" | No false timeouts | One hang kills the job with no junit, cache forfeit, leaked kernels | Only with a per-test pytest mark ceiling above the worst cell and job-budget arithmetic updated |
| Asserting only "notebook executed without error" | Fast test authoring | Cell-level logic regressions (empty results, silent chrom mismatch) pass | Acceptable as wave-1 smoke; must be upgraded with output/defs assertions |
| Broad `reason_like` allowlist entries | Quiets the skip audit | Allowlist becomes a silencer — the exact anti-pattern its header warns about | Never |
| Committing showcase intermediates to make tests pass | Deterministic tests | GB-scale repo bloat, review noise | Only the curated ≤200 kb region slices, by design |
| Running marimo apps in-process via `App.run()` | No subprocess plumbing | State/CUDA leakage across tests; default-driven surprise training | Only for apps verified light, with explicit default-value assertions |
| Letting notebooks fetch showcase data live each run | No data-pipeline code | Flaky, unauditable tests dependent on a fragile portal | Never for tests; fine inside the one-time selection script |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| actions/cache (model caches) | Adding 30 GB models to the same cached paths as the 8-model warm set | Split tiers: quota-bounded cache for small/medium; persistent on-disk dir for giants; `allow_patterns` to skip redundant `.pt` |
| arabidopsis.org / TAIR | Trusting a 200 response means you got the file | Magic-byte validation, browser UA retry, prefer plantdhs.org direct links; tests depend only on committed slices |
| plantdhs.org BigWig + TAIR GFF3 | Mixing `Chr1` vs `1`, or 1-based GFF3 coords into 0-based pyBigWig queries | Normalize once in a tested helper; assert `bw.chroms()` mapping and non-empty extracts |
| rice.uga.edu (NER example) | `!wget -c` partial files reused across runs; genome fetch inside the test | Validate archives; fetch in setup with retry+timeout; consider a cached fixture path keyed on content hash |
| HuggingFace / ModelScope | Fetching by mutable ref with `trust_remote_code=True` | Pin revisions in `models.lock`; log resolved commit; rotate pins via review |
| ollama REST (11434) | Assuming auth, or binding beyond loopback; pulling models inside tests | Loopback-only systemd unit; idempotent pre-pull step; assert deterministic layers only |
| dnallm MCP server (8000) | Hardcoded 8000 colliding with the 6 live-server probes / other tests | Dedicated port or serialized exclusive use; patch the URL cell in the tmp copy |
| flash-attn / transformer-engine / evo2 / megaDNA GitHub installs | Assuming "GPU runner" ⇒ these install and run | Capability spike on the actual aarch64 GB10 box; typed `environment-unavailable:` skips for infeasible ones |
| HF Trainer tensorboard logging | Forgetting event files are side effects | tmp-copy cwd + tree-cleanliness guard (output_dir is cwd-relative in examples) |
| wandb (latent) | Assuming configs never drift to `report_to: wandb/all` | `WANDB_MODE=disabled` in kernel env as belt-and-braces |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| evo-class downloads in the cached path | Nightly cold-starts; cache evictions | Tiered cache + download filtering | First run that pushes repo cache >10 GB |
| Unbounded notebook runtimes in census | Job killed at 900 min, no junit | Per-artifact measured budgets; per-test marks; shrink epochs/max_steps for finetune examples (documented deviation) | ~3–4 full finetune notebooks in one census |
| VRAM accumulation across sequential execution tests | CUDA OOM in later, smaller tests | Per-test model teardown (`del model; torch.cuda.empty_cache()`), subprocess isolation for marimo, kernel-per-notebook | Second heavy model in one pytest process |
| Per-run flash-attn/TE source builds on aarch64 | Hours of compile per nightly | Prebuilt/warm env or typed environment skip | Any evo-family test on GB10 |
| Silent-cell iopub watchdog misfires | `Timeout waiting for IOPub output` on legitimate loads | Explicit generous `iopub_timeout` per artifact class | First >4-min silent model-load cell |
| `!wget -c` resume on corrupt HTML | Parse errors or empty datasets downstream | Magic-byte validation + clean re-fetch | First portal hiccup |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Unpinned `trust_remote_code` model refs executed on the self-hosted runner | Upstream repo compromise ⇒ arbitrary code on the box, invisible to review | Revision pins in `models.lock`; resolved-commit logging; lock edits are security review surface |
| Executing the megaDNA `git clone && pip install -e .` path unpinned | Arbitrary setup.py execution from an unmaintained repo | Vendor a pinned, hash-verified copy or shim; or typed environment skip |
| Exposing ollama beyond loopback / wildcard `OLLAMA_ORIGINS` | Unauthenticated model pull/delete/inference; DNS-rebinding from browser pages | Keep `127.0.0.1:11434` bind; no origins wildcard; runner user owns the unit |
| Weakening the dispatch/cron-only runner posture "just this once" for a PR | PR-authored code (incl. forks) executing on the box — the exact invariant v1 established | Never; execution tests are `slow` and live only in the nightly/dispatch census |
| Notebooks/agents binding servers to all interfaces | Services reachable from LAN on a persistent box | Explicit `--host 127.0.0.1` for any server a test starts; port discipline |
| Secrets in notebook outputs / committed executed copies | Token leakage into artifacts | Never write executed copies in-place; scrub env before kernel; artifacts on failure only |

## Showcase & Documentation Pitfalls (the "UX" of example notebooks)

| Pitfall | Reader Impact | Better Approach |
|---------|---------------|------------------|
| Cherry-picked locus framed as benchmark | Readers cite a circular number as performance | "Illustrative loci + selection criteria" framing; regression-guard test, not accuracy claim |
| No reproducibility info (model revision, seeds, tolerance) | "Works differently for me" disputes | Pin revision + state tolerances in notebook markdown; link the selection-rationale doc |
| Silent-empty signal extraction presented as agreement | Misleads expert readers; hides bugs | Non-emptiness assertions + negative-control region |
| Repairing notebooks to suit the harness (path edits, hardcoded ports) | Examples stop reflecting real user workflows | Fix the harness (cwd/env), keep examples user-shaped; deviations documented, not smuggled |
| Committed outputs drifting from source (already true for megaDNA) | Readers trust stale results | Regenerate curated outputs at milestone close; treat executed copies as debug artifacts |

## "Looks Done But Isn't" Checklist

- [ ] **Execution harness:** kernel cleanup verified on a deliberately-hanging notebook (kill test), not just happy path — verify orphan-check `ps`/`nvidia-smi` after
- [ ] **Timeout layering:** nbclient cell timeout < per-test mark < job budget, arithmetic updated in the ci.yml comment — verify sum still <900 min
- [ ] **Hermeticity:** full local run leaves `git status --porcelain` empty — verify the session-end guard test exists and would fail
- [ ] **Honest skips:** every new skip category has a narrow allowlist entry and a typed prefix — verify `audit_skips.py` exit 0 on the nightly junit *with* the new categories present
- [ ] **Execution ≠ correctness:** executed notebooks assert on outputs/defs (non-empty, in-range), not just "no cell raised" — verify at least one assertion would catch a silent-empty result
- [ ] **marimo apps:** run via subprocess with cheap verified defaults; `defs` asserted — verify an app test cannot trigger the default heavy finetune
- [ ] **ollama family:** model pre-pull idempotent; port plan vs the 6 MCP probes resolved — verify probes' skip/run behavior is deterministic under the new setup
- [ ] **Showcase data:** committed slices validated (magic bytes, chrom names, coordinate convention); downloads blocked ⇒ tests still green — verify by unplugging network once
- [ ] **Truth agreement:** threshold + tolerance documented and re-derivable; negative control included — verify assertion fails when fed shuffled truth
- [ ] **Cache strategy:** giants outside the quota-bounded cache; evo-1 fetched without the redundant `.pt` — verify `gh cache list` total after first nightly
- [ ] **Mirror sync:** WR-08 flipped together with drift closure; each notebook repair regenerates its docs MD — verify docs-validation green on the repair branch
- [ ] **Coverage expectations:** nobody expects example executions to raise the 96.30% (kernel subprocesses are unmeasured by design, AUDIT-04) — verify the phase plan says so explicitly

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Leaked kernels / VRAM exhaustion | LOW | Hygiene step (pkill + VRAM assert), runner reboot if wedged; add the missing cleanup path as a regression kill-test |
| Cache evicted by oversized model | MEDIUM | `gh cache delete` the offender; move giant to persistent dir; re-warm next nightly; add size report step |
| Dirty tree from missed side effect | LOW | `git checkout -- .` + add gitignore pattern + extend the tmp-copy list for that artifact; guard test now catches recurrences |
| Wrong "repair" applied to a healthy notebook | MEDIUM | Revert the content fix, fix the harness (cwd/env/port), re-run; add the case to the harness-vs-content triage list |
| Flaky assertion on sampled output | LOW | Convert to invariant/tolerance assertion; keep a debug flag to dump the executed copy |
| Showcase truth-agreement broke after model rotation | MEDIUM | Re-run selection rationale with pinned new revision; adjust documented threshold via review; never loosen silently |
| Skip audit red from a new network skip | LOW | Add endpoint-specific typed entry (narrow matcher + category); add fetch retry so it rarely fires |
| Silent-empty chrom mismatch discovered late | MEDIUM | Introduce the shared normalization helper + fixtures; re-run loci selection; republish showcase numbers with the correction noted |
| ollama/port collision mid-census | LOW | Serialize or re-port the example server; make the collision loud (port-in-use check in setup) |

## Pitfall-to-Phase Mapping

Suggested v1.1 phase structure (roadmap not yet written; names are recommendations):

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Kernel leaks / GPU poisoning | Phase 1 — Harness & Hermeticity | Deliberate-hang kill test; post-run `ps`/VRAM hygiene step in nightly |
| 2. Timeout arithmetic | Phase 1; recheck in CI phase | ci.yml comment arithmetic; sum(ceilings) < 900 min reviewed per mark change |
| 3. Side effects / dirty tree / cwd false repairs | Phase 1 | Tree-clean guard test green after full census |
| 4. Cache quota / 30 GB downloads | Phase 1–2 boundary | `gh cache list` bounded; evo fetch size ≈12.9 GB not 29.7 GB |
| 5. evo/megaDNA toolchain infeasibility | Phase 1 (feasibility spike) | Verdict matrix per artifact; typed `environment-unavailable:` entries audited |
| 6. Silent-skip regression | Phase 1 (taxonomy) | `audit_skips.py` exit 0 nightly; skip count stable |
| 7. marimo in-process traps | Phase 1 (pattern) / Phase 2 (rollout) | Apps run in subprocess; default-plan assertion |
| 8. trust_remote_code provenance | Phase 1 (pins) + each family | `models.lock` revisions; resolved-commit lines in junit |
| 9. ollama service/port/assertions | Phase 2–3 + CI phase | Idempotent pre-pull; deterministic port plan; probe behavior stable |
| 10. Nondeterminism / stale outputs | Phase 1–2 (conventions) | Invariant-style assertions; tolerance bands; no golden-output comparisons |
| 11. Coordinates / chrom naming | Showcase phase (first task) | Normalization-helper unit tests; non-empty assertions; negative control |
| 12. arabidopsis.org blockers | Showcase phase (data acquisition) | Magic-byte validation; tests green with network blocked |
| 13. Showcase overfitting framing | Showcase phase (rationale doc) | Review of notebook/docs language; threshold re-derivable |
| 14. Mirror drift / WR-08 flip | CI-gate phase | Docs-validation green on the branch that removes `continue-on-error` |

## Sources

- **Repo inspection (2026-10-01):** `example/` notebooks + configs (side-effect greps, committed-output audit, evo/megaDNA/ollama cell sources), `tests/examples/test_examples.py`, `tests/expected_skips.yaml`, `scripts/audit_skips.py`, `.github/workflows/ci.yml` (coverage-nightly/test-mamba jobs), `models.lock`, `pyproject.toml` (extras, pytest config), root `conftest.py`, `.gitignore`, `dnallm/models/special/evo.py`, `dnallm/utils/support.py`, `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` — **HIGH**
- **nbclient docs** (client/reference pages): timeout/iopub semantics, `shutdown_kernel` graceful/immediate, context-manager cleanup — [nbclient client docs](https://nbclient.readthedocs.io/en/latest/client.html) — **MEDIUM-HIGH**
- **pytest-timeout** issues [#134](https://github.com/pytest-dev/pytest-timeout/issues/134) (fixtures not torn down), [#159](https://github.com/pytest-dev/pytest-timeout/issues/159) (subprocess survives) — **HIGH**
- **GitHub Actions limits / cache**: [docs.github.com actions limits](https://docs.github.com/en/actions/reference/limits), [actions/cache](https://github.com/actions/cache), [Nov 2025 changelog (>10 GB opt-in)](https://github.blog/changelog/2025-11-20-github-actions-cache-size-can-now-exceed-10-gb-per-repository) — **HIGH**
- **pyBigWig README** "A note on coordinates": 0-based half-open; case-sensitive, non-mixable chrom names; empty-on-unknown — [github.com/dpryan79/pyBigWig](https://github.com/dpryan79/pyBigWig) — **HIGH**
- **TAIR/Ensembl chrom naming**: Biostars/Google-groups evidence of `Chr1` vs `1` harmonization — **MEDIUM** (community; verify against the actual committed files at selection time)
- **ollama security**: default `127.0.0.1:11434`, no auth, DNS rebinding — [ollama #16236](https://github.com/ollama/ollama/issues/16236), Elastic/CVE-2024-39719 write-ups — **HIGH** (behavior), bind default cross-checked
- **evo-1-131k-base file listing**: 29.7 GB total, safetensors ~12.9 GB + redundant `pytorch_model.pt` 16.8 GB, trust_remote_code stripedhyena variant — [HF repo tree](https://huggingface.co/togethercomputer/evo-1-131k-base/tree/main) — **HIGH** (fetched)
- **evo2 package requirements**: 1B/20B/40B need FP8 via Transformer Engine + Hopper; vtx/vortex + flash-attn==2.8.0.post2; build-failure issues — [pypi.org/project/evo2](https://pypi.org/project/evo2), [arcinstitute/evo2](https://github.com/arcinstitute/evo2) (#149, #201) — **HIGH** (requirements), GB10 impact is repo-derived inference — **MEDIUM**
- **marimo App API**: `run(defs)` semantics, all-or-nothing definition override, headless UI-value behavior via defaults / `set_ui_element_value` — [docs.marimo.io/api/app](https://docs.marimo.io/api/app), marimo discussions #3698 — **MEDIUM** (in-process execution is architectural, verified from the API design)
- **W&B headless**: `WANDB_MODE=disabled/offline`, set pre-init; Trainer prompting unless `report_to none` — [docs.wandb.ai](https://docs.wandb.ai/support/models/articles/how-do-i-disable-wandb-when-testing-my-code), HF forums — **MEDIUM-HIGH**
- **arabidopsis.org SPA/login-wall**: repo-internal milestone knowledge (no authoritative public doc found) + TAIR portal state reports — **LOW-MEDIUM**; mitigations valid regardless
- Prior-milestone artifacts consulted: `.planning/codebase/TESTING.md`, v1 `PITFALLS.md` (os._exit lesson — since fixed in root conftest), PROJECT.md Phase 1 AUDIT-04 note (pytest-cov 7 subprocess measurement removal)

---
*Pitfalls research for: DNALLM v1.1 — Example Execution Testing & Repair*
*Researched: 2026-10-01*
