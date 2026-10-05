---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
# Codebase Concerns

**Analysis Date:** 2026-10-05

*Verbatim sources: `.planning/STATE.md` (Blockers/Concerns + Pending Todos), `.planning/milestones/v1-MILESTONE-AUDIT.md` (tech-debt ledger), `.planning/WINDOWS.md` (broken-windows register), `.planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md`, live code at HEAD `9ec6bf5` (branch `phs`).*

## Tech Debt

**Torch ceiling is an ABI claim, not a version preference:**
- Issue: torch `<2.12` (mirrored across every cuda/rocm/mamba extra) exists solely to protect the exact-pinned native kernel stack — `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.7.0`, evo-venv `flash-attn==2.8.3.post1` — which compiles against the installed torch. Widening the ceiling re-resolves torch on any reinstall and breaks those kernels plus the ~50-min sm_120 flash-attn wheel build. Dependabot ignores torch minor/major bumps at `.github/dependabot.yml:30-40`; PR #42 is held open because its fast-leg green never exercised the dispatch-only legs that carry the kernels.
- Files: `pyproject.toml` (`[tool.uv.sources]`/extras), `.github/dependabot.yml:30-40`
- Impact: any automatic or casual torch bump silently breaks mamba/evo dispatch legs; CI green on the fast leg does not catch it.
- Fix approach: bump torch only as a coordinated kernel-rebuild (via `scripts/install_mamba.sh` + wheelhouse rebuild) + nightly-validation cycle, never via a merged dependabot PR alone.

**Transformers 5.x breaks ModelScope mamba remote code:**
- Issue: the open `transformers>=4.49.0,<6` range rides on remote code that has not adapted: ModelScope mamba `modeling_mamba.py` imports `MambaCache`, which transformers 5.17 no longer exports. Mamba-family model loading breaks on upgrades until upstream remote code adapts.
- Files: `pyproject.toml` (transformers range), `.github/dependabot.yml:22-27` (recorded KNOWN RISK), `.planning/WINDOWS.md`
- Impact: a transformers bump that passes the fast leg can still break `load_model_and_tokenizer` mamba paths at nightly time.
- Fix approach: every transformers bump must re-verify the mamba load path before merge (recorded dependabot rationale; see also `dnallm/utils/transformers_compat.py` as the landing zone for compat shims).

**Latent library bugs pinned by tests (waived to backlog at v1 close, still present at HEAD):**
- Issue: five documented bugs are intentionally pinned as-is by tests rather than fixed (owner-dispositioned, semantics ambiguous or out of scope):
  - `dnallm/inference/inference.py:1645` — generate-from-DataLoader never appends to `prompt_seqs` (`seqs.extend` on itself); causallm generate over a DataLoader returns an empty list.
  - `dnallm/inference/mutagenesis.py:429` — evaluate `strategy="max"` calls `raw_score.index(...)` on an ndarray → `AttributeError`.
  - `dnallm/models/model.py:264-265` (construction) + `:276-284` (call sites) — `cosine_similarity` builds `nn.CosineEmbeddingLoss` but every call is 2-arg `(logits, labels)`; missing `target` raises `TypeError` for every user selecting it (pinned by `test_forward_cosine_similarity_loss_crashes`).
  - `dnallm/datahandling/data.py:983` — `raw_reverse_complement` is a no-op: the `ds.map(concat_fn, ...)` result is discarded (latent bug pinned by test).
  - `dnallm/tasks/metrics.py:612` — `metrics_for_dnabert2("regression")` returns nested `{"r2": {"r2": float}}` (evaluate-metric dict re-wrapped); pinned as contract by a phase-03 test.
- Files: listed above; register: `.planning/WINDOWS.md` IDs 2/3/4 (waived), `v1-MILESTONE-AUDIT.md` Phase 03 ledger.
- Impact: silent wrong results (empty generation list, no-op augmentation, double-nested r2) rather than loud failures; downstream consumers must not rely on these paths.
- Fix approach: pick up in a next-milestone repair phase; each fix must un-pin the corresponding contract test in the same change.

**Verbose task_type aliases validate but are never normalized onto the model:**
- Issue: `dnallm/configuration/configs.py:102` pattern accepts `binary_classification`, `multi_class_classification`, `multi_label_classification`, `token_classification`; `model_post_init` (`configs.py:118-127`) normalizes them only into a local variable — `self.task_type` keeps the verbose spelling, which downstream dispatchers (metrics, heads, plotting) match against the short forms.
- Files: `dnallm/configuration/configs.py:102,118-127`
- Impact: configs using documented verbose spellings validate at load and then mis-dispatch downstream.
- Fix approach: write the normalized spelling back to `self.task_type` in `model_post_init` and update the pinning tests.

**Benchmark.run hardcodes the labels column:**
- Issue: `dnallm/inference/benchmark.py:296` reads `self.datasets[di]["labels"]` while `example/benchmark_config.yaml`-style configs declare `label_column: label` → `KeyError` before any model loads; blocks the benchmark notebook census row (WINDOWS.md #14, open, Phase-8 repair queue).
- Files: `dnallm/inference/benchmark.py:296`, `example/notebooks/benchmark/benchmark_config.yaml`
- Impact: benchmark example lane cannot execute against custom label columns.
- Fix approach: resolve the configured `label_column` through the dataset wrapper instead of the literal `"labels"` key.

**Import-time `logs/` sink recreated under pytest cwd:**
- Issue: `DNALLMLogger._setup_handlers` (`dnallm/utils/logger.py:56-60`) does `Path("logs").mkdir(exist_ok=True)` + file handler at import time — every run (including pytest from any cwd) creates `logs/dnallm.log`.
- Files: `dnallm/utils/logger.py:56-60`
- Impact: cwd pollution; plan-level "no logs dir" gates are unsatisfiable (had to be scoped to `mcp_server.log` instead).
- Fix approach: make the file sink opt-in (env var or explicit `setup_logging` call).

**Typing tooling mid-transition (ty up, mypy out):**
- Issue: owner decision 2026-10-05 — `ty` becomes the hard gate (fast leg + `scripts/check_code.py` + pre-commit) after the E-family diagnostic triage reaches 0 (post-`261003-0p0` baseline: 165 diagnostics); `mypy` retires from pre-commit + CI + `check_code.py`, `[tool.mypy]` config removed last. None of this is wired yet: CI still runs advisory mypy (`|| true`) at `.github/workflows/ci.yml:124` and `:203`; `scripts/check_code.py:128` still hard-requires `mypy` in `required_tools`; ty excludes exist at `pyproject.toml:399-404` (`[tool.ty.src]`) but no gate consumes them.
- Files: `.github/workflows/ci.yml:124,203`, `scripts/check_code.py:128`, `pyproject.toml:399-404`, `.pre-commit-config.yaml`
- Impact: two type checkers coexist; 165 untriaged ty diagnostics block the planned hard gate.
- Fix approach: Phase 9 scope — triage E-family to 0, wire ty into the three enforcement points, then retire mypy config/steps in reverse order.

**TypedDict consumer pass incomplete (cli remaining):**
- Issue: `load_config` (`dnallm/configuration/configs.py:495`) returns a per-key TypedDict; the five engine facades (`DNAInference`, `DNATrainer`, `Mutagenesis`, `Benchmark`, `DNAInterpret`) now accept `Mapping[str, Any]` (commits `86022f7`, `16a9ffb`, `07af770`), but the CLI consumer pass is still pending.
- Files: `dnallm/cli/cli.py`, `dnallm/configuration/configs.py:495`
- Impact: IDE/ty `invalid-argument-type` noise remains on the CLI path.
- Fix approach: same coordinated-change pattern as the five facades (types + tests in one change).

**Stale documentation (multiple ledger entries):**
- Issue: `README.md:511` still prescribes the retired `.[test,dev,mcp]` install (extras no longer resolve); `README.md:198` uses `.[test,cpu]`; `.github/workflows/README.md` describes only the 03:00 UTC nightly and omits the 05:30 example-nightly schedule and its staged contract; `CONTRIBUTING.md` still says line length 79 (enforced ruff config is 100; 79 only applies to the MCP module via `.flake8`).
- Files: `README.md:198,511`, `.github/workflows/README.md`, `CONTRIBUTING.md`
- Impact: documented commands fail; CI behavior misdocumented.
- Fix approach: one docs-sync pass; keep `scripts/check_docs_sync.py` lanes in mind.

**models.lock stale provenance comment:**
- Issue: `models.lock:15` — the `plant-dnamamba-BPE-open_chromatin` provenance comment names a config that was swapped to `plant-dnagpt-BPE-promoter` at commit `95c9ba0`.
- Files: `models.lock:15`
- Impact: provenance misleads cache-key debugging.
- Fix approach: one-line comment fix.

**WINDOWS.md ledger needs triage:**
- Issue: 5 open + 3 waived entries; at least one open entry is stale — ID 13 (mcp_example never executed) was de-facto closed by quick task `261003-csd` (both notebooks green in the gated lane) without the ledger being updated; v1 ship-triage WR-01 (nightly test-mamba continue-on-error) is contradicted by current code, which carries an explicit "No continue-on-error" contract at `.github/workflows/ci.yml:335`.
- Files: `.planning/WINDOWS.md`, `.github/workflows/ci.yml:335`
- Impact: `/gsd-ship` blocks while `open_count > 0` under `workflow.windows_enforce`; stale entries waste triage time.
- Fix approach: run the planned ledger triage (~10 entries + IN-01..07 info findings); mark fixed/waived with evidence.

**Fixed ~60s cost in test_timeout.py:**
- Issue: two full 30-second timeout waits burn a fixed ~60s on every census.
- Files: `tests/mcp/test_timeout.py` (see `deferred-items.md`)
- Impact: ~1 min added to every full-suite run.
- Fix approach: shorten `_tool_timeout_seconds` locally, as `test_timeout_configurable` already does.

## Known Bugs

**example-nightly flash-attn build-isolation failure (Phase 9 first item):**
- Symptoms: first full example-nightly dispatch (run `37278002681`) failed in the stage-0 flash-attn wheelhouse step — torch is missing at `get_requires_for_build_wheel` despite `--no-build-isolation` in `pip wheel`.
- Files: `.github/workflows/ci.yml:705-716` ("Stage 0: build/install flash-attn (cached wheelhouse, evo venv)")
- Trigger: cold cache (no `wheelhouse-flashattn/` wheel present) on the evo venv's `pip wheel "flash-attn==2.8.3.post1"`.
- Workaround: none wired; the run outcome lands per the Phase-5 D-04 post-merge boundary and is the first Phase 9 work item.

**05:30 cron double-triggers the other nightly jobs:**
- Symptoms: `ci.yml` schedules both `0 3 * * *` and `30 5 * * *` (`.github/workflows/ci.yml:12-18`); `coverage-nightly` and `test-mamba` gate only on `github.event_name == 'schedule'` (`ci.yml:275,423`), so the 05:30 entry re-fires BOTH jobs alongside `example-nightly`, contending for the single `dnallm-nightly` runner.
- Files: `.github/workflows/ci.yml:12-18,275,423`
- Trigger: every 05:30 UTC scheduled run.
- Workaround: queue serialization absorbs it (runner is single-slot); fix is a one-line job gate on `github.event.schedule` pending owner call.

**Transformers-5.17 remote-code gaps (benchmark third model / GAP-1 class):**
- Symptoms: `zhangtaolab/nucleotide-transformer-v2-100m-promoter` not loadable on transformers 5.17 — remote code needs removed 4.x `PretrainedConfig` defaults (`is_decoder`/`add_cross_attention`); the native-ESM route (`trust_remote_code=False`) was probed and refuted (FFN weight-shape mismatch, ckpt 4096x512 vs config 2048x512), so no drop-in fix exists for these checkpoints.
- Files: `example/notebooks/benchmark/benchmark.ipynb` (WINDOWS.md #11), `tests/examples/test_script_execution.py` (WINDOWS.md #12, self-healing typed skip)
- Trigger: loading the affected remote-code checkpoints on transformers 5.x.
- Workaround: shim layer in `dnallm/utils/transformers_compat.py` covers the import rungs (9-shim layer, quick tasks `261002-se3`/`261002-sl7`); the benchmark third-model row awaits owner disposition (D-09).

## Security Considerations

**Remote-code execution by design (trust_remote_code):**
- Risk: model loading runs arbitrary Python from HF/ModelScope repos (`trust_remote_code=True` across `dnallm/models/model.py`, ModelScope `Auto*` classes, special handlers).
- Files: `dnallm/models/model.py` (`load_model_and_tokenizer`), `dnallm/models/modeling_auto.py` (registry)
- Current mitigation: registry curated by owner (`dnallm/models/model_info.yaml`, 238 entries); nightly models pinned to revisions in `models.lock`; self-hosted legs are schedule/dispatch-only so fork PR code never executes on the runner (`ci.yml:275,423`); CI defaults `permissions: contents: read` (`ci.yml:23-24`).
- Recommendations: keep the models.lock revision pinning discipline for any new registry entry used in CI; never widen self-hosted job triggers to `pull_request`.

**MCP server has no authentication:**
- Risk: `dnallm/mcp/server.py` binds per config host/port (CI uses localhost:8000) with no auth layer; tools execute model inference.
- Files: `dnallm/mcp/server.py`, `dnallm/mcp/configs/mcp_server_config.yaml`
- Current mitigation: local-tool posture (localhost binding in shipped config); documented transports stdio/SSE/streamable-http.
- Recommendations: do not expose the server on public interfaces without adding an auth layer; keep shipped configs localhost-bound.

**Install-time curl|sh and sdist builds in CI:**
- Risk: `curl -LsSf https://astral.sh/uv/install.sh | sh` in every job; micromamba tarball fetch for bedtools; aarch64 sdist builds (pyBigWig) compile third-party C on the box.
- Files: `.github/workflows/ci.yml:57,653`, stage-0 steps
- Current mitigation: runs on ephemeral hosted runners or the owner-controlled self-hosted box; least-privilege GITHUB_TOKEN.
- Recommendations: acceptable for current posture; revisit if the threat model changes.

## Performance Bottlenecks

**Nightly runtime budgets sit near their caps:**
- Problem: coverage-nightly job kill is 900 min with per-test timeout ceilings summing ~840 min (+ stage marks); the Phase-7 timeout-arithmetic hand-off notes the sum-of-ceilings comment passes 900 on paper (~970 min) even though measured actuals are ~15 min for the slow showcase pair. example-nightly backstop is 2700 min against ~2620 min of per-test ceilings.
- Files: `.github/workflows/ci.yml:426-433,541-564`
- Cause: per-test pytest-timeout marks are the primary hang protection; job kills are backstops only.
- Improvement path: Phase 9 A/A-decided runtime cuts — (1) `finetune_custom_head` epochs 3→1 via a TEST-SANDBOX-ONLY YAML patch at the harness sandbox-patch step (~31→~11 min; committed notebook content unchanged); (2) mcp_example pair: per-request ollama `options.num_ctx` ~8k at the D-13/probe layer (the 256k kv-cache held 36GB VRAM and dominated both latency and the 14:07 VRAM trough). Both land with same-change tests in Phase 9.

**Recurring mamba kernel compile in test-mamba:**
- Problem: `--no-cache-dir` plus a fresh venv per run deliberately bypass all wheel caching, so the mamba-ssm/causal_conv1d source compile repeats every nightly (~40-90 min of the 180-min job cap).
- Files: `.github/workflows/ci.yml:277-284,313-333`
- Cause: chosen posture (proves the from-source path nightly).
- Improvement path: the example-nightly wheelhouse pattern (`ci.yml:718-737`) already caches these kernels for that job; porting it to test-mamba is possible but changes what that leg proves — owner call.

**MCP server eager model loading:**
- Problem: the server eagerly loads its 3 configured models at startup; readiness polls allow ~15 min even from warm cache.
- Files: `dnallm/mcp/server.py`, `dnallm/mcp/model_manager.py:22`, `.github/workflows/ci.yml:822-823`
- Cause: startup-load design in `ModelManager`.
- Improvement path: lazy load-on-first-tool-call if nightly wall time matters.

**Runner shares $HOME with the dev box:**
- Problem: the `dnallm-nightly` self-hosted runner lives in the owner's `$HOME` (warm hf/modelscope/giants/ollama caches are the point), so dev-box GPU work and CI coexist by discipline, not isolation.
- Files: `.github/workflows/ci.yml` (self-hosted jobs), `scripts/runner/` (runner ops)
- Cause: single-box economics; owner rule: local caches (hf/modelscope/giants/ollama) are the persistence layer keeping nightlies download-free and are NEVER cleaned by default — cleanup requires explicit owner approval plus dead-data evidence (the 2026-10-05 84→19GB hub cleanup was the one approved exception).
- Improvement path: stage-boundary discipline (D-07): kill orphaned stage processes, verify port free, ≥35 Gi VRAM available before any heavy stage, recorded before/after in the census log. Runner restarts need the sanitized-env procedure (`env -i`) or uv installs into the wrong venv.

## Fragile Areas

**Import-time compat shims (transformers_compat / cuda_compat):**
- Files: `dnallm/utils/transformers_compat.py` (1,117 lines), `dnallm/utils/cuda_compat.py`, eagerly imported by `dnallm/utils/__init__.py`
- Why fragile: monkey patches mutate process-wide library behavior at import; each new transformers version can shift the patch surface (absence guards added by quick task `261003-jpr` — `TestTransformersAbsenceContract` dynamically collects installers so future ones are auto-covered).
- Safe modification: any new installer must follow the guarded no-op pattern; keep the dynamic absence-contract test roster in sync.
- Test coverage: contract tests exist (`tests/utils/`); the shim-vs-upstream drift risk on transformers bumps remains the exposure.

**Single self-hosted runner is a serialization point:**
- Files: `.github/workflows/ci.yml` (`runs-on: [self-hosted, dnallm-nightly]` on test-mamba, coverage-nightly, example-nightly)
- Why fragile: one box serves all nightly legs; a hung job monopolizes it (job timeouts are the only whole-job bound on self-hosted); queue order decides whether example-nightly actually starts 2.5 h after coverage-nightly.
- Safe modification: keep the stagger (03:00 / 05:30) and the timeout backstops; never merge nightly jobs onto one runner slot without re-planning (D-05/D-06 owner-rated costly to reverse).
- Test coverage: n/a (infra).

**Stage-ordering contract in example-nightly (D-07):**
- Files: `.github/workflows/ci.yml:526-540,792-906`
- Why fragile: server-binding stages (MCP on :8000, ollama batch) must never start while the previous torch-heavy stage still holds GPU memory or kernel processes; exactly one :8000 server at a time; pkill/settle steps between stages are load-bearing.
- Safe modification: any new stage must slot into the 0→1→1.5→2→2.5→3→4 ordering and append to `stage-results.txt` (D-08 fail-soft — no step may be forever-green).
- Test coverage: junit per stage + stage-4 skip audits.

**evo/giants lane is mid-restructure (owner directive):**
- Files: `.github/workflows/ci.yml:660-779` (evo venv, flash-attn wheelhouse, giants prefetch steps still present at HEAD), `tests/examples/test_notebook_execution.py` (evo lane)
- Why fragile: owner directive (16:27 CST 2026-10-05) excludes giants-class models from the pytest census — Phase 9 must wire example-nightly WITHOUT the evo-lane pytest steps (deselect precedent: stage-1 mcp), retire giants-prefetch + evo-venv CI steps, keep evo tests in-repo as a dispatch/manual lane with committed executed-notebook outputs as evidence, and keep `models.lock` evo rows as provenance. Until wired, the CI file and the directive disagree.
- Safe modification: scope-fence the exclusion in 09-CONTEXT; do not delete the evo tests or the models.lock rows.
- Test coverage: manual/dispatch lane after Phase 9.

**Network reality on the runner:**
- Files: `.github/workflows/ci.yml:565-571` (`HF_ENDPOINT: https://hf-mirror.com` pinned job-wide), `dnallm/models/model.py:380-391` (mirror toggle)
- Why fragile: huggingface.co is unreachable from the box; only the mirror serves artifacts — cold fetches dead-wait on the origin if the env var is missing.
- Safe modification: always inherit the job-wide HF_ENDPOINT in new stages; record mirror assumptions in stage comments.

**aarch64 sdist builds need the LDFLAGS workaround:**
- Files: `.github/workflows/ci.yml:331,487,621` (pyBigWig `-L$(sys.prefix)/lib` fix, run `37184990854` fallout)
- Why fragile: sysconfig bakes a nonexistent `/opt/hostedtoolcache` LIBDIR; any new sdist-building dependency on the box may hit the same class of link failure.
- Safe modification: keep the export in every install step that builds from sdist.

**MCP concurrency layer (recently burned, now guarded):**
- Files: `dnallm/mcp/model_manager.py:121` (`_load_model_sync` executor bridge), `dnallm/mcp/server.py:282` (`_with_timeout_wrapper`)
- Why fragile: two live serving bugs were found and fixed here in October (single-flight inference — concurrent DataLoader forks + filelock fork-unsafe deadlock; `dna_interpret` mamba guard — captum backward on DNAMamba SIGKILLs the server, exit 137), followed by review fixes CR-01 (threading.Lock spanning the executor-submitted callable) and WR-01 (interpret behind `_interpret_thread_lock` so the 30 s tool timeout can fire).
- Safe modification: follow the CR-01 lock pattern for any new blocking tool; never release locks on asyncio cancellation paths.
- Test coverage: regression tests exist for both; new blocking tools need the same red-then-green pattern.

**CI leg inconsistencies (ungated lanes):**
- Files: `.github/workflows/ci.yml:205-265` (test-cuda runs `pytest tests/` single root, no `--cov` — neither dual-root nor floor-enforced, informational); `dnallm/mcp/tests/` live-server probes (6, localhost:8000-targeted) deterministically skip in CI as allowlisted `network-unavailable:` typed skips.
- Why fragile: green on these legs says less than it appears to.
- Safe modification: treat test-cuda as smoke only; do not cite it as coverage evidence.
- Test coverage: by definition not covered by the gate.

## Scaling Limits

**GitHub Actions cache quota vs model-cache size:**
- Current capacity: actions/cache quota 10 GB per repo.
- Limit: a lock-only models cache measured 15.2 GiB and never actually saved; the models-cache layer was therefore DROPPED from CI entirely (owner decision 2026-10-05) — cold pulls accepted (65-min cold stage 1 proven vs the 2700-min example-nightly budget).
- Scaling path: box-side caches (never cleaned by default per owner rule; hub at ~19 GB after the one approved cleanup) + the exact-key read-only reuse pattern between nightly jobs (D-09) if caching is ever reintroduced.

**Coverage-nightly on-paper ceiling vs job cap:**
- Current capacity: 900-min job kill; per-test marks sum ~840 min (+2400 s CRE + 5400 s Anno showcase marks pushing the documented sum past the cap on paper, ~970 min).
- Limit: a week where actuals approach ceilings would breach the job kill before the per-test marks fire.
- Scaling path: CI-06 pre-authorizes a separate example-execution nightly job (already exercised — example-nightly exists); further splits follow the same pattern; measured actuals (~15 min for the slow showcase pair, ~2 h 35 m full examples lane) leave large headroom today.

**Single-runner queue depth:**
- Current capacity: one `dnallm-nightly` slot serving test-mamba (180 min), coverage-nightly (900 min), example-nightly (2700 min) nightly, queue-serialized.
- Limit: total nightly demand is bounded only by the stagger discipline; a re-run or the 05:30 double-trigger stacks multi-hour jobs back-to-back.
- Scaling path: fix the cron gate; a second runner label would break the single-`:8000`/single-cache assumptions — re-plan before adding.

## Dependencies at Risk

**mcp (pinned `<2`):**
- Risk: 2.x breaks the server API (commit `3490b71`); pin is deliberate.
- Impact: no path to mcp 2.x features; upstream security fixes in 2.x are unavailable.
- Migration plan: planned upgrade as its own phase (dependabot ignores semver-major at `.github/dependabot.yml:12-15`), re-verifying `dnallm/mcp/server.py` tool registration and the client pair.

**transformers (open `<6`) vs remote-code rot:**
- Risk: each 5.x minor can remove more 4.x defaults that remote model code depends on (`MambaCache` today; `is_decoder`/`add_cross_attention` already shimmed).
- Impact: model-family loading breaks at runtime, not install time.
- Migration plan: compat shims accumulate in `dnallm/utils/transformers_compat.py`; every bump re-verifies the mamba load path (dependabot rationale); long-term upstream remote-code adaptation.

**datasets (pinned `<=3.2.0`):**
- Risk: tokenization-pipeline API coupling; the pin is the constraint.
- Impact: cannot take datasets 3.3+ fixes.
- Migration plan: unpin only alongside a `dnallm/datahandling/data.py` tokenization-path re-verification.

**Native kernel exact pins (mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, flash-attn 2.8.3.post1):**
- Risk: coupled to the installed torch ABI (see Tech Debt above); dependabot-ignored entirely.
- Impact: kernel upgrades require local CUDA toolchain rebuilds (`scripts/install_mamba.sh`, wheelhouse steps).
- Migration plan: coordinated kernel-rebuild + nightly-validation cycle only.

**fla / flash-linear-attention (`>=0.5.2,<0.6`):**
- Risk: PlantHelixSeek KDA kernels depend on it; the bare spec (no backend extra) rides the installed torch/triton — a fla bump that pulls its own torch would break the ABI coupling.
- Impact: silent fallback to the remote code's non-KDA path (measured dead: 0.0073 vs 0.7673) if fla is absent.
- Migration plan: bounded range is deliberate; guard tests + docs FAQ pin the expectation.

**ipython (`>=8.31,<9`) in the notebook extra:**
- Risk: pygenometracks hard-caps `matplotlib<3.9`, making IPython 8 the only kernel-plot lever.
- Impact: blocks IPython 9 features; upstream pgt is the real constraint.
- Migration plan: revisit when pgt lifts the matplotlib cap.

## Missing Critical Features

**ty hard gate (Phase 9 scope, not yet wired):**
- Problem: `ty` is configured (`pyproject.toml:399-404`) but enforced nowhere; 165-diagnostics E-family baseline awaits triage to 0 before the gate can land in the fast leg, `scripts/check_code.py`, and pre-commit.
- Blocks: the "coverage cannot regress + types enforced" quality posture the milestone claims.

**Cron disambiguation for the 05:30 schedule:**
- Problem: no job gate distinguishes which cron entry fired, so the stagger double-triggers two other nightly jobs.
- Blocks: clean nightly queueing; trustworthy example-nightly start times.

**evo-lane census exclusion wiring:**
- Problem: owner directive excludes evo/giants from the pytest census, but the deselect, the retiring of giants-prefetch/evo-venv CI steps, and the dispatch/manual-lane story are all Phase 9 work — CI at HEAD still contains the full evo provisioning.
- Blocks: the Phase 9 "CI Wiring & Census Verification" definition of done.

## Test Coverage Gaps

**Unguarded-guard tests (v1 ledger, open):**
- What's not tested: the WR-01 multilabel-curve guard fix (phase 01) has no regression test — the guarded path is unreachable from the suite; the phase-03 multilabel AUROC/AUPRC absent-summary guard likewise has no test.
- Files: `dnallm/inference/plot.py` (curve guard), `dnallm/tasks/metrics.py` (absent-summary guard); ledger: `v1-MILESTONE-AUDIT.md` phase 01/03 WR-08/WR-05.
- Risk: the guards can be deleted or broken silently.
- Priority: Medium.

**Local-only MCP live-server probes:**
- What's not tested in CI: the 6 live-server probes (`dnallm/mcp/tests/test_sse_client.py`, `test_streamable_http_client.py`) target localhost:8000 and no PR-gate job starts a server; they skip as allowlisted `network-unavailable:` typed skips and run only inside example-nightly stages 2/3.
- Files: `.github/workflows/ci.yml:498-503`, `tests/expected_skips.yaml`
- Risk: PR-gate green does not cover the SSE/streamable-http client paths.
- Priority: accepted by design (nightly covers them); keep the allowlist exact.

**test-cuda leg is coverage-blind:**
- What's not tested: single-root `pytest tests/` with no `--cov` (`.github/workflows/ci.yml:262-265`); GPU-path code (device placement, CUDA compat) has no coverage instrumentation anywhere in CI.
- Files: `.github/workflows/ci.yml:205-265`
- Risk: GPU-path regressions surface only as runtime failures on nightly legs.
- Priority: Low (informational by audit disposition).

**evo family exits the pytest census (owner directive):**
- What's not tested after Phase 9: evo-lane pytest steps in example-nightly; evidence shifts to committed executed-notebook outputs plus a dispatch/manual lane.
- Files: `.github/workflows/ci.yml:660-779`, `tests/examples/test_notebook_execution.py`
- Risk: evo-family regressions are caught only on manual dispatch; scope-fence in 09-CONTEXT must record this trade.
- Priority: decided (owner, 2026-10-05) — not a gap to close, a constraint to document.

**Windows leg ledger triage:**
- What's not tested/verified: `test-windows` (py3.12 fast leg) exists, but the WINDOWS.md ledger triage (~10 entries + IN-01..07 info findings) and several Windows-specific dispositions are pending.
- Files: `.github/workflows/ci.yml:126-203`, `.planning/WINDOWS.md`
- Risk: Windows regressions of info severity linger un dispositioned.
- Priority: Low.

---

*Concerns audit: 2026-10-05*
