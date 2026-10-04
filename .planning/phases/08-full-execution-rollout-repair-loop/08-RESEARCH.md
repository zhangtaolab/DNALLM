# Phase 8: Full Execution Rollout & Repair Loop - Research

**Researched:** 2026-10-04
**Domain:** example-tree real-execution rollout (nbclient/marimo/script lanes), transformers-5 remote-code shims, GPU-runner CI job assembly, models.lock/cache-tiering, ollama/MCP coexistence infrastructure
**Confidence:** HIGH (in-repo state and registry facts verified this session; runner-box state unverifiable from dev box — flagged per-item)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### 修复节奏与验收线
- **D-01:** 推进分组按**模型家族**（evo 系 / NT 系 / 小模型群 / mcp+ollama 批等），家族内 notebook+marimo+script 一起跑一起修 — 不按 census 错误类别波次、不逐件串行。家族清单由 planner 从 census+ACTIVE/GATED 清单推导。
- **D-02:** 修复时对 notebook 内容的修改边界 = **允许结构重构**（单元格合并/拆分/重排），以可执行性优先；不受"仅修错误"约束。镜像随之字节同步（D-11 沿袭）。— **Reversibility:** costly — 结构重构后 wrapper 摘录与镜像需成套重导出，回退需恢复整本 notebook。
- **D-03:** dev-box 侧验收 = **逐项全量对账**：每个修复项落地后跑一次全量 census（21+3+1+YAML）对账 + 全快道，每步都有全量基线（成本接受）。runner 官方确认仍按 Phase-5 D-04 合并后进行。
- **D-04:** 上游不可修类（UPSTREAM 根因，如 BPE tokenizer 工件本身损坏且 dnallm 侧无修复点）= 保留 cell 原样 + **证据化 typed skip**（复用 05-04 模式，证据进 skip 消息）；不重写数据路径偏离上游教程。

#### Nightly 作业布局与共存
- **D-05:** example 执行落**独立 example-execution nightly job**（启用 CI-06 预授权），与 coverage-nightly 并行调度、时长预算互不挤占 — **Reversibility:** costly — job 拆分后合并需重排全部 nightly 预算与顺序约定。
- **D-06:** 两 nightly **错峰串行**（如 coverage 03:00 UTC、example 05:30 UTC 起步；具体时刻 planner 定），避免同时占 GPU/磁盘带宽；job 内部串行。
- **D-07:** VRAM/端口共存（MCP-02）= **阶段化串行**：job 内先重 torch 执行（含 example notebooks）→ 后 MCP live-server 探针批（:8000）→ 最后 ollama 批；阶段间显式清理 VRAM/进程；排序写入 ci.yml 注释与计划文档。
- **D-08:** example job 失败语义 = **fail-soft + 汇总非零退出**：单件失败不阻后续项，job 末尾按失败数非零退出（honest-gates 原则；不允许永绿）。
- **D-09:** 模型缓存 = **共享 coverage-nightly 的 models.lock-keyed hub cache**（同 key 只读复用）；巨型模型单独 tier 不进此 cache（见 D-13）。
- **D-10:** 环境安装 = **全 extras** `.[base,fla,dev,mcp]`（系统依赖如 bedtools 在 job 步骤装），对齐 runner 现有 nightly 腿安装线。

#### ollama 基础设施（MCP-01）
- **D-11:** 预拉模型 = **notebook 原引用模型**（保持教程内容不变；前提体积/VRAM 可行——研究阶段核实具体 id 与大小）。
- **D-12:** systemd 服务**定义文件进仓**（infra/ 或 scripts/runner/，含 README），runner 上一次手工 enable；可审计可重建。loopback-only（127.0.0.1）。
- **D-13（探针/回退）:** 就绪探针 = **测试内探针**：ollama 批测试前置 `curl http://127.0.0.1:11434/api/tags` 重试窗口（~60s×2s）；失败 → typed `network-unavailable:` skip，curl 输出+重试日志作证据写进 skip 消息。回退线**仅限基础设施缺失**（服务未起/模型未拉/端口不可达）；模型在但输出内容异常属测试断言问题，必须修、不许 skip。

#### 巨型模型、锁与结构家族
- **D-14:** evo-1（CI-05）= snapshot_download **allow_patterns=safetensors-only（~12.9GB）** + 巨型模型放 runner **本地巨仓目录**（如 ~/models-giants/，经 env/symlink 指向；具体机制研究阶段定），不进 10GB-quota cache、永不驱逐热缓存。— **Reversibility:** costly — 巨仓布局与 cache 策略进入 job 定义后重排成本高。
- **D-15:** models.lock（CI-04）新条目 = **逐 id + ms 前缀 + revision pin（commit sha）+ 用途注释**，对齐现有两行格式；notebook 的 `source=` 与前缀对齐（不一致处改 notebook）。
- **D-16:** lock 覆盖范围 = **census 全部真实执行过的模型 id**（ACTIVE×13 + GATED×8 已覆盖的 + 本阶段新跑的）；~8+ 只是下限。
- **D-17:** NT-REMOTE-STRUCTURAL 家族（v2-100m-promoter 等 3 项、05-04 梯子遗留 owner 处置）= **研究阶段先评估 shim 深度**（离 HEAD 多远、vendor 面多大、小变体是否同坎）再定：可修则小变体→加深 shim 真实执行，不可修才证据化 typed skip。处置结论必须逐项记账。

#### 追加决策（marimo / wrapper / 回归 / 版本戳）
- **D-18:** marimo 断言深度 = 标准三件套（无头跑通 + UI 元素默认值断言 + 退出码断言）**+ export-html 产物生成与关键内容校验**（工件不入仓，仅临时断言后丢弃）。
- **D-19:** 回归测试归属 = **按层归属**：example 内容修复的回归测试进 `tests/examples/`；dnallm 库修复的进对应模块 `tests/`（对齐 owner 的库改动同船 pytest 规则）。
- **D-20:** wrapper 同步 = 修复提交为**一个原子提交**（notebook + 字节镜像 + 重导出的 wrapper 摘录 + 必要的叙述更新），`check_notebook_md_sync` AST 校验绿是提交门槛。
- **D-21:** 版本戳 = **逐本真实戳**：每本 notebook 修复重执行时更新自己的 provenance 版本行（transformers/torch/fla key=value），不追求跨本一致。

### Claude's Discretion
- 家族划分的具体清单（从 census + ACTIVE/GATED 推导）与修复顺序。
- 错峰调度的具体时刻（错开即可）。
- evo-1 本地巨仓的具体路径与指向机制（env var vs symlink）。
- ollama systemd 单元的进仓目录命名（infra/ vs scripts/runner/）。
- 结构重构的具体单元格编排（可执行性优先原则下）。

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-02 | All 21 example notebooks execute all code cells end-to-end with real models on the nightly GPU runner (slow-marked; allow_errors=False fail-at-first-error per notebook, fail-soft across notebooks) | Current lane state verified (ACTIVE 13 / GATED 8); harness contract already enforces fail-at-first-error; the gap is gated-family prerequisites in the example job (see Rollout State + Patterns) |
| EXEC-03 (re-verify) | All 3 marimo apps execute headlessly via subprocess with UI elements yielding defaults and exit codes asserted | Current test asserts export + size only; D-18 deepening gaps documented (defaults + HTML key content) |
| EXEC-04 | `generate_bpe_dataset.py` executes against its committed inputs producing its dataset artifact in-sandbox | Script lane test exists with self-healing typed skip; NT shim rung already landed — heal path documented + dev-box cache-pollution pitfall |
| EXEC-05 | Every example YAML config passes real `load_config()` Pydantic validation on the fast leg | Verified this session: 21/21 YAMLs pass `scripts/validate_yaml.py`; `tests/configuration/test_yaml_load.py` auto-parametrizes via rglob (new YAMLs auto-covered) |
| REPAIR-01 | Every error surfaced by real execution is fixed — each with a regression test; harness-bug vs content-bug triaged explicitly (no cwd false-repairs) | Remaining repair queue enumerated with exact locations (megaDNA generate, finetune_generation ordering, evo notebook reference, benchmark latent hardcode) |
| REPAIR-03 | dnallm library bugs exposed by execution are fixed with regression tests | Already marked Complete in traceability (quick tasks se3/sl7/csd/hhj/ij4); verify no regression during full rollout |
| REPAIR-04 | The langchain notebook's `!uv pip install langchain-ollama` cell is repaired — dependency declared in the `mcp` extra | mcp extra verified to lack langchain-ollama; langchain-ollama 1.1.0 verified on PyPI (matches the notebook's installed output) |
| CI-04 | models.lock extended with all newly-executed model ids (~8+) with ModelScope-first prefixes, source=-aligned, revision-pinned | Complete candidate-id table verified against both registries (existence, sizes, MS availability); revision-pin mechanisms documented |
| CI-05 | Cache strategy survives the giants — evo-1 via allow_patterns safetensors-only (~12.9GB), tiered outside the 10GB-quota cache | Sizes verified via HF tree API (12.913GB safetensors vs 16.81GB .pt); mechanism options + recommendation; cache-quota arithmetic documented |
| MCP-01 | ollama on the nightly runner as loopback-only systemd infra with pre-pulled small model + readiness probe; both mcp_example notebooks execute end-to-end; typed skip only as documented fallback | qwen3.8:latest verified live (27.3B Q4_K_M, 17.74GB, tools+thinking); stock systemd unit captured verbatim; ollama loopback default verified from official docs |
| MCP-02 | Port and VRAM coexistence planned against the 6 MCP live-server probes (:8000) and heavy torch tests | Server config/port verified; staged-serial layout (D-07) mapped to concrete pytest stage invocations; both-up execution already proven on dev box (261003-csd) |
</phase_requirements>

## Summary

Phase 8 starts from a much stronger position than the phase description implies. The Phase-5 census (25 items: 11 PASS / 12 FAIL / 2 deferred) has been almost fully consumed by quick tasks 261002-sl7 and 261003-csd: **13 notebooks are in the always-execute ACTIVE lane, 8 in probe-then-execute GATED lanes (the mcp pair already at owner-approved EXECUTE state), all 3 marimo apps and the YAML leg are green, and the entire NT-REMOTE-STRUCTURAL family — the D-17 research question — turns out to be already fixed at the deepest rung needed**: `dnallm/utils/transformers_compat.py` now carries 9 absence-gated shims (read this session), including the `PretrainedConfig` legacy-defaults `__getattr__` that superseded the 05-04 "not vendored-pure-helper territory" termination. D-17's honest disposition is therefore: (a) the three census items are shim-covered, (b) one dev-box evidence gap remains — the local ModelScope snapshot of `zhangtaolab/nucleotide-transformer-v2-100m-promoter` is still hand-patched (`config.json.dnallm-bak`, `modeling_esm.py.dnallm-bak` verified present in the cache), so the shim-only path for the benchmark's NT leg was never proven on a pristine snapshot, and (c) the script lane should self-heal to real green once dev-extra deps (pyfastx/pybedtools + bedtools binary) are present.

The real remaining work splits into three buckets. **Bucket 1 — make the GATED families actually execute** (EXEC-02's "all 21"): the gated notebooks are gated on prerequisites that D-10's install line `.[base,fla,dev,mcp]` does NOT include — evo-1 needs `evo-model`+`stripedhyena`+flash-attn sm_120 source build (~50 min) + an `np.fromstring` shim; evo2 needs `evo2==0.3.0`+`vtx` with Transformer Engine deliberately absent; megaDNA needs the pinned clone `cb2f5ab4` + `MEGABYTE_pytorch==0.2.1`; the two lora notebooks need the `[mamba]` extra (source build). These must land as example-job steps (test-mamba precedent), with wheel caching so flash-attn/mamba kernels don't rebuild nightly. **Bucket 2 — the known repair items**: evo notebook model-reference update (131k→8k + evo2 noFP8), the megaDNA `DNATokenizer` generate bug in `dnallm/models/special/megadna.py`, finetune_generation's genome-input ordering (the `!wget` cell exists but execution died at cell 2 — content repair with the in-notebook Ensembl URL), REPAIR-04 (add `langchain-ollama` to the mcp extra), D-18 marimo deepening. **Bucket 3 — infrastructure**: the separate example-execution nightly job (staggered cron, staged serial per D-07, fail-soft per D-08, shared models.lock-keyed cache per D-09), the evo-1 giants tier outside the 10GB cache (sizes verified: 12.913GB safetensors vs 16.81GB .pt vs 29.7GB full), models.lock additions (9 ModelScope-verified zhangtaolab ids + 5 HF-only ids, all registry-checked with shas), and the loopback-only ollama systemd infra (qwen3.8:latest = 27.3B Q4_K_M, 17.74GB — feasible on GB10 unified memory, and both-up execution was already proven on the dev box).

Two planning risks deserve early attention. First, **cache quota arithmetic**: GitHub Actions caches are capped at 10GB per repository with LRU eviction (now with an opt-in pay-as-you-go tier above that); adding ~7-9GB of new models to the existing warm cache likely exceeds the cap — the plan needs a measure-then-decide step (exclusions vs pay-as-you-go), and this is exactly why CI-05's giants tier is load-bearing. Second, **the evo-1 8k variant was proven under transformers 4.57.6 in a throwaway venv** (05-FEASIBILITY verbatim: "plausibly also runs under 5.x, untested") — its first execution in the project venv (transformers 5.17) is a genuine unknown and may surface new remote-code rungs; budget the evo family for discovery.

**Primary recommendation:** Land the example-job skeleton early (existing ACTIVE set only, staggered cron, fail-soft) so the runner produces nightly signal while the repair families proceed in the D-01 order NT-heal → small-model/marimo/langchain → evo-giants → megaDNA → mamba-lora → ollama/MCP final integration, then the D-03 full reconciliation census; treat models.lock growth + cache-quota measurement as one atomic decision point, and use the harness per-notebook env-override extension + a small `allow_patterns` passthrough in `download_model` as the evo-1 giants mechanism.

## Project Constraints (from CLAUDE.md)

- **No new test frameworks** — pytest + pytest-cov only; the nbclient harness is the locked lane (CONTEXT canonical refs).
- **Compatibility**: suite must keep passing on Python 3.11/3.12/3.13 and numpy 1.26.4 & 2.2.0; tests must not pin to a single transformers minor version. Job-step installs of evo/evo2/flash-attn/megaDNA live in CI only, never in pyproject extras (except the already-sanctioned `mcp` extra addition for langchain-ollama).
- **Coverage gate**: example execution runs in kernel subprocesses and by design does not move the 96.30% gate (AUDIT-04 precedent; CI-09 documents this in Phase 9).
- **Scope**: bug fixes limited to what correctness/coverage requires; no refactors beyond that (D-02's notebook structural-refactor allowance is the owner-sanctioned exception, inside `example/` only).
- **Owner memory rule**: any `dnallm/` code modification ships with pytest coverage in the same change (= D-19).
- **Style**: ruff (line 100), Google docstrings, relative imports inside `dnallm/`, absolute in tests/scripts; comments in English.
- **Runner ops memory**: restart the self-hosted runner with sanitized env (`env -i`) or uv installs into the wrong venv.
- **Git**: work lands on `phs`; commit/push without attribution trailers; manual-push-only during campaigns.

## Current Rollout State (verified this session)

The planner's most important input — what already exists vs what Phase 8 must build:

| Lane | State | Evidence |
|------|-------|----------|
| ACTIVE notebooks (always execute) | **13** | `tests/examples/test_notebook_execution.py:66-80` — `ACTIVE_NOTEBOOKS` lists inference, generation, in_silico_mutagenesis, interpretation, data_prepare/predict, finetune_binary, finetune_multi_labels, finetune_NER_task/data_generation_and_inference, benchmark, data_prepare/finetune/finetune_data, embedding_attention, finetune_NER_task/finetune_NER_task, inference_for_tRNA [VERIFIED: tests/examples/test_notebook_execution.py:66-80] |
| GATED notebooks (probe-then-execute) | **8** | `GATED_NOTEBOOKS` at tests/examples/test_notebook_execution.py:570-585: mcp pair (`_gate_ollama_stack`), generation_evo_models (`_gate_evo`), generation_megaDNA + finetune_custom_head + finetune_generation (`_gate_megadna`), lora pair (`_gate_mamba`) [VERIFIED: tests/examples/test_notebook_execution.py:570-585] |
| Fail-at-first-error | already enforced | `allow_errors=False,  # fail at first error -- the repair signal this milestone exists for` in `run_notebook` [VERIFIED: tests/examples/_execution.py:368] |
| Typed-skip prefixes registered | yes | `network-unavailable:`, `environment-unavailable:`, `optional-dep:` in tests/expected_skips.yaml:29-41 [VERIFIED: tests/expected_skips.yaml:29-41] |
| transformers-5 shim set | **9 patches** | `apply_patches()` at dnallm/utils/transformers_compat.py:1024-1034: `_patch_get_parameter_or_buffer`, `_patch_initialize_weights_for_quantized_missing`, `_patch_remote_code_pruning_helpers`, `_patch_get_extended_attention_mask`, `_patch_pretrained_config_legacy_defaults`, `_patch_mamba_cache`, `_patch_deberta_vocab_dict`, `_patch_get_head_mask`, `_patch_legacy_init_weights_bookkeeping` [VERIFIED: dnallm/utils/transformers_compat.py:1024-1034] |
| marimo lane | green, shallow | 3 apps in `MARIMO_EXEC_SPECS`; test asserts export + `>1000` bytes + tree-clean only [VERIFIED: tests/examples/test_marimo_execution.py:63-73] |
| Script lane | self-healing typed skip | `test_generate_bpe_dataset_produces_artifact` converts only the marker `Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'` into `environment_unavailable_skip`; everything else re-raises; artifact assertions (file exists, >0, fresh mtime) already written [VERIFIED: tests/examples/test_script_execution.py:43,91-108] |
| YAML leg | green | Ran `scripts/validate_yaml.py` this session: `Total: 21 files / All YAML files passed validation.` [VERIFIED: executed 2026-10-04]; `test_yaml_load.py` parametrizes via `EXAMPLE_DIR.rglob("*.yaml")` so new YAMLs auto-join [VERIFIED: tests/configuration/test_yaml_load.py:13-14] |
| mcp pair at EXECUTE state | yes (dev box) | Both-up executes, any-down typed-skips with both probe results; langchain sibling under isolated kernelspec `dnallm-mcp-langchain` (throwaway venv `.scratch/mcp-example-venvs/langchain`, `VIRTUAL_ENV` pinned in kernel.json) [VERIFIED: tests/examples/_execution.py:80-81 and test_notebook_execution.py:505-534] |
| mcp extra | lacks langchain-ollama | `mcp = ["mcp>=1.0.0,<2", "langchain>=1.3.6", "langchain_mcp_adapters>=0.2.1", "nest-asyncio>=1.5.9", "pydantic-ai<3"]` [VERIFIED: pyproject.toml:115-121] |
| docs mirror + wrapper sync tooling | exists | `scripts/check_docs_sync.py`, `scripts/check_notebook_md_sync.py`, `scripts/generate_md_from_notebook.py`, `scripts/generate_md_from_marimo.py`, `scripts/validate_docs_snippets.py` [VERIFIED: scripts/ directory listing] |

### D-17 verdict: NT-REMOTE-STRUCTURAL shim depth

The question CONTEXT D-17 poses ("离 HEAD 多远、vendor 面多大、小变体是否同坎") resolves as follows:

1. **Shim depth already sufficient.** The terminal rung that killed all three census items (`'EsmConfig' object has no attribute 'is_decoder'` — remote `modeling_esm.py` reading removed 4.x `PretrainedConfig` defaults) is closed by `_patch_pretrained_config_legacy_defaults`, which installs a `PretrainedConfig.__getattr__` over the closed default map `{"is_decoder": False, "add_cross_attention": False}` [VERIFIED: dnallm/utils/transformers_compat.py:507-510, quoted verbatim below in Code Examples]. The module comment records that this rung "SUPERSEDES the 05-04 D-07 rung termination ... per the owner instruction of 2026-10-02: fix all non-gated census failures now" [VERIFIED: dnallm/utils/transformers_compat.py:501-504].
2. **Vendor surface is small and bounded** — per shim it is one function/class vendored verbatim from upstream tag v4.49.0 with documented deviations (pruning helpers, `get_extended_attention_mask`, `get_head_mask`, `MambaCache` ~150 lines, config defaults map, DebertaV2 vocab normalization, init-weights bookkeeping wrapper). Nothing forks transformers; patches attach missing names onto live transformers modules/classes, absence-gated and idempotent — they no-op entirely on transformers 4.x.
3. **Small variants share the rung and are covered.** All three known variants are green in the ACTIVE lane: `zhangtaolab/nucleotide-transformer-v2-100m-promoter` (benchmark third model — notebook PASSED 49.9s per the sl7 summary [VERIFIED: .planning/quick/261002-sl7-.../261002-sl7-SUMMARY.md:82]), `zhangtaolab/plant-nucleotide-transformer-BPE` (NER notebook, PASSED 924.9s real training, same summary), and `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (embedding_attention — its `find_pruneable_heads_and_indices` from `transformers.pytorch_utils` rung is covered by `_attach_remote_code_pruning_helpers` attaching to BOTH `modeling_utils` and `pytorch_utils` [VERIFIED: dnallm/utils/transformers_compat.py:344-376]).
4. **Two residuals to disposition per D-17's "逐项记账"**: (a) the dev-box MS snapshot of v2-100m-promoter is hand-patched (see Pitfall 1) so the benchmark NT leg's shim-only path is unproven locally — restore pristine and re-execute; (b) `generate_bpe_dataset.py` (EXEC-04) should now heal past its marker into real green — its typed skip is a live probe that self-heals by design [VERIFIED: tests/examples/test_script_execution.py:38-43].

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Notebook real execution (21 + 2 showcase) | Test layer (`tests/examples/`, nbclient kernel subprocess) | CI nightly runner | Execution truth lives in pytest parametrizations; the job only schedules/installs |
| Gated-family prerequisites (evo/evo2/megaDNA/mamba) | CI job steps (example-execution nightly) | — | Deliberately NOT pyproject extras (05-FEASibility isolation rule; project venv untouched) |
| ollama service | Runner OS (systemd unit, in-repo definition) | Test-layer readiness probe (D-13) | Long-lived infrastructure, not per-job state; unit file is auditable/rebuildable in-repo |
| dnallm MCP server :8000 | CI job step (background process, stage-scoped) | — | Started/stopped per D-07 stage; config from `dnallm/mcp/configs/mcp_server_config.yaml` |
| Model cache tiering | CI cache config (actions/cache paths+key) | Runner filesystem (giants dir) | 10GB quota governs cached paths; giants live outside by construction |
| evo-1 giant fetch policy | dnallm library (`download_model` allow_patterns passthrough) + harness env override | CI prefetch step | safetensors-only must hold at load time, not just prefetch time (see Patterns) |
| models.lock content | Repo root file (source of truth, cache key) | Notebooks' `source=` fields | D-15 alignment: lock prefix ↔ notebook route |
| Repair truth (regression tests) | `tests/examples/` (content) / module `tests/` (library) | — | D-19 layer ownership |
| Docs mirror regeneration | Repair commit (atomic, D-20) | sync scripts as commit gate | Mirror follows the notebook byte-identically in the same commit |

## Standard Stack

No new test frameworks (CLAUDE.md constraint). Everything below is either already in the repo or a CI-only/job-step/runner-host install.

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| pytest + nbclient harness | pytest>=8.4, nbclient>=0.10 (installed 0.11.0) | execution lanes | Locked by Phase 5; `NOTEBOOK_EXEC_SPECS` budgets for all 21+2 notebooks already exist [VERIFIED: tests/examples/_execution.py:102-221] |
| marimo CLI (export html) | >=0.16.3 (census ran 0.25.0) | headless app execution | 05-FEASIBILITY flavor decision (deterministic exit + artifact) |
| GitHub Actions (actions/cache@v4, schedule+dispatch) | — | example-execution nightly job | Existing coverage-nightly pattern to clone [VERIFIED: .github/workflows/ci.yml:406-497] |
| systemd unit (runner host) | — | ollama loopback service | Official ollama Linux packaging; in-repo per D-12 |
| ollama + qwen3.8:latest | qwen3.8:27b-q4_K_M, 17.74GB on disk | MCP client notebooks' LLM | The notebooks' literal reference; verified feasible [VERIFIED: local API probe this session] |

### Supporting (job-step installs, gated families — versions from spike/PyPI)
| Package | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| langchain-ollama | 1.1.0 (PyPI latest; matches notebook output) | REPAIR-04: declare in `mcp` extra | pyproject edit + regression test |
| evo-model + stripedhyena | 0.5 / 0.2.2 (`--no-deps`) | evo-1 handler prerequisites | evo family job step |
| flash-attn | 2.8.3.post1, source build `FLASH_ATTN_CUDA_ARCHS="120" MAX_JOBS=8 --no-build-isolation --no-cache-dir` (~50 min) | evo-1/evo2 attention kernels | evo family; cache the built wheel |
| evo2 + vtx | **pin 0.3.0** + 1.1.0 (PyPI latest evo2 is 0.6.0 — unproven) | evo2 handler; TE must remain absent | evo family job step |
| megaDNA clone + MEGABYTE_pytorch | `github.com/lingxusb/megaDNA` @ `cb2f5ab4cc88dc0effe05c5f23358862c837014a` + **MEGABYTE_pytorch==0.2.1** (latest 0.3.6 unproven; repo's own pin is 0.2.1) | megaDNA handler | megaDNA family job step |
| `[mamba]` extra | `causal_conv1d==1.7.0`, `mamba-ssm==2.3.2.post1` [VERIFIED: pyproject.toml mamba extra] | PlantCAD2 remote code | lora pair job step (test-mamba precedent: `--no-cache-dir --no-build-isolation`) |
| bedtools | v2.31.1 on dev box (linuxbrew) | pybedtools runtime binary | job step / runner probe — runner user lacks sudo, use micromamba or static binary |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Wheel-cached flash-attn/mamba builds | rebuild every run (test-mamba status quo) | rebuild burns ~50+ min nightly inside the example budget; wheel cache keyed on version+arch is one job step |
| Separate example nightly job (D-05) | widen coverage-nightly | locked by D-05/CI-06 pre-authorization; budgets don't fit (07 hand-off: sum-of-ceilings already ~970 min on paper) |
| `HF_HUB_CACHE` env override for evo-1 giants | symlink into hub cache / `source="local"` route | symlink breaks under actions/cache (follow→quota bust, preserve→dangling); local-source changes tutorial content and passes `revision="main"` to a local dir (unverified tolerance) — see Patterns |

**Version verification (executed this session):** PyPI JSON API — langchain-ollama 1.1.0, evo-model 0.5, stripedhyena 0.2.2, evo2 0.6.0 (pin 0.3.0), vtx 1.1.0, flash-attn 2.8.3.post1, MEGABYTE_pytorch 0.3.6 (pin 0.2.1) [VERIFIED: PyPI registry].

## Package Legitimacy Audit

Ran `gsd-tools query package-legitimacy check --ecosystem pypi` on the seven pip-installable packages. All returned `SUS` with the single reason `unknown-downloads` (the checker has no PyPI download telemetry for these); every one resolves to an authoritative source repo and was already executed successfully in the Phase-5 spike campaigns.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| langchain-ollama | PyPI | pub 2026-04-07 | unknown | docs.langchain.com/oss/python/integrations/providers/ollama (official LangChain partner pkg) | SUS (telemetry only) | Approved — official LangChain integration package; notebook output shows it installing as `langchain-ollama==1.1.0` [VERIFIED: notebook output + PyPI] |
| evo-model | PyPI | pub 2026-02-16 | unknown | github.com/evo-design/evo | SUS (telemetry only) | Approved — canonical evo-1 package (05-FEASIBILITY warns PyPI name `evo-1` is an unrelated SLAM package — never install that name) |
| stripedhyena | PyPI | pub 2024-02-23 | unknown | github.com/togethercomputer/stripedhyena | SUS (telemetry only) | Approved — togethercomputer canonical |
| evo2 | PyPI | pub 2026-06-19 (0.6.0) | unknown | github.com/arcinstitute/evo2 | SUS (telemetry only) | Approved — arcinstitute canonical; PIN 0.3.0 (spike-proven) |
| vtx | PyPI | 1.1.0 | unknown | pypi.org/project/vtx (arcinstitute) | SUS (telemetry only) | Approved — pulled by evo2 per spike |
| flash-attn | PyPI | 2.8.3.post1 | unknown | github.com/Dao-AILab/flash-attention (maintainer tri@tridao.me) | SUS (telemetry only) | Approved — Dao-AILab canonical |
| MEGABYTE_pytorch | PyPI | 0.3.6 | unknown | github.com/lucidrains/MEGABYTE-pytorch (lucidrains@gmail.com) | SUS (telemetry only) | Approved — lucidrains canonical; PIN 0.2.1 (megaDNA repo's own requirement) |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none beyond the telemetry-only rows above (no `checkpoint:human-verify` gate warranted; all are spike-proven with recorded evidence). One hard warning inherited from 05-FEASIBILITY: **never `pip install evo-1`** — that PyPI name is an unrelated package; the evo-1 prerequisite is `evo-model` [CITED: 05-FEASIBILITY.md prerequisites section].

## Architecture Patterns

### System Architecture Diagram — example-execution nightly job (D-05..D-08, D-13, D-14)

```
                    ┌──────────────────────── GitHub schedule (staggered) ────────────────────────┐
                    │  coverage-nightly  cron "0 3 * * *"   (existing, untouched)                 │
                    │  test-mamba        schedule-gated     (existing, untouched)                 │
                    │  example-nightly   cron e.g. "30 5 * * *"  (NEW; single [self-hosted,       │
                    │                     dnallm-nightly] runner ⇒ jobs queue serially)           │
                    └──────────────────────────────────┬──────────────────────────────────────────┘
                                                       ▼
  Stage 0 INSTALL   uv venv → uv pip install -e ".[base,fla,dev,mcp]"
                    + job steps: bedtools probe/fallback · gated-family prereqs
                      (evo-model 0.5, stripedhyena 0.2.2 --no-deps, flash-attn wheel-cache,
                       evo2==0.3.0 + vtx 1.1.0 [TE absent], megaDNA clone cb2f5ab4 +
                       MEGABYTE_pytorch==0.2.1, .[mamba] build)
                     └─ restore models.lock-keyed hub cache (shared key, read-only reuse, D-09)
                        + giants prefetch → ~/models-giants (allow_patterns safetensors-only)
                                                       ▼
  Stage 1 TORCH-HEAVY (D-07 first): pytest tests/examples
      ACTIVE notebooks ×13 ─┐
      showcase ×2 (nightly lane, 2400/5400 marks)
      marimo ×3 export-html │  fail-at-first-error per item (allow_errors=False)
      script: generate_bpe_dataset.py (rice downloads in-sandbox)
      GATED families (gates now GREEN because prereqs installed):
        evo ×1 (kernel env → giants HF_HUB_CACHE) · megaDNA ×3 · mamba-lora ×2
        [mcp pair NOT here — stage 3]
      YAML fast leg (EXEC-05, no kernel)
                                                       ▼
  Stage 1.5 CLEANUP: pkill ipykernel_launcher · assert no stray kernels · VRAM settle
                                                       ▼
  Stage 2 MCP :8000:  start dnallm-mcp-server (streamable-http, mcp_server_config.yaml)
      → 6 MCP live-server probes (dnallm/mcp/tests) — execute for real, no typed skips
                                                       ▼
  Stage 3 OLLAMA: readiness probe curl 127.0.0.1:11434/api/tags (retry ~60s × 2s, D-13)
      → both mcp_example notebooks (langchain via dnallm-mcp-langchain isolated kernel)
      → stop MCP server
                                                       ▼
  Stage 4 SUMMARY: junit + skip audit + `if: always()` artifact upload
      fail-soft: every item ran; pytest exit code = summarized non-zero (D-08)
```

Reader trace: a notebook error in stage 1 never blocks later notebooks (pytest parametrization) and never aborts stages 2-3 (separate pytest invocations); the job exits non-zero at the end if anything failed — no continue-on-error, no forever-green.

### Recommended family grouping (D-01 — proposed, planner finalizes)

| # | Family | Contents | Prerequisites/infra | Risk |
|---|--------|----------|---------------------|------|
| G0 | Job skeleton | example-nightly job with CURRENT ACTIVE set + stagger + fail-soft + shared cache | none | low — early nightly signal |
| G1 | NT heal + script lane (EXEC-04, D-17 closure) | restore pristine v2-100m-promoter snapshot; re-run benchmark + NER notebook; script self-heal to real green; bedtools on box | dev extra + bedtools | low |
| G2 | small-model + marimo + langchain (EXEC-03/05, REPAIR-04) | marimo D-18 deepening; YAML re-verify; langchain-ollama in mcp extra | none | low |
| G3 | evo giants (CI-05) | evo notebook ref update 131k→8k + evo2 noFP8 condition; np.fromstring shim in transformers_compat; giants tier; flash-attn wheel cache | evo-model/stripedhyena/flash-attn | **medium — transformers 5.17 untested for 8k remote code** |
| G4 | megaDNA | prereq install; DNATokenizer generate repair (library bug); finetune_custom_head demo cell; finetune_generation wget-ordering + pinned clone | pinned clone + MEGABYTE_pytorch | medium |
| G5 | mamba lora pair | `[mamba]` build in job; 2 notebooks execute | kernel builds | medium (build time) |
| G6 | ollama/MCP integration (MCP-01/02) | systemd unit in-repo; runner enable + pull; stage 2/3 wiring; probes | owner manual step | low (dev-box-proven) |
| G7 | models.lock + quota + census (CI-04, D-03) | lock entries + revision pins; cache-quota measure/decide; final full reconciliation | — | medium (quota arithmetic) |

### Pattern: repair-loop unit (REPAIR-01, D-02, D-19, D-20, D-21)

One repair = one atomic commit: (notebook edit — structural refactor allowed — + byte-synced docs mirror + re-exported wrapper excerpts + narrative) OR (library fix + same-change pytest in module tests/) OR (harness fix + test); `check_notebook_md_sync` AST green is the commit gate; the repaired notebook's provenance line gets its own real version stamp (`transformers=5.17.0 torch=2.11.0+cu130 fla=0.5.2` key=value form, D-21). Triage every failure FIRST as harness-bug vs content-bug vs library-bug; a cwd change is never a repair (the benchmark precedent: root cause was the missing `../inference/test.csv` sibling input, fixed by `_NOTEBOOK_EXTRA_INPUTS` seeding, not by chdir).

### Pattern: evo-1 giants tier (D-14/CI-05 — recommended mechanism)

Verified constraint chain: dnallm's evo-1 load path calls `_get_model_path_and_imports(model_name, source, revision)` whose hub branch does `status = downloader(model_name, revision=revision)` [VERIFIED: dnallm/models/model.py:348] — a FULL-repo `snapshot_download` with NO `allow_patterns` passthrough. Verified repo shape: `togethercomputer/evo-1-8k-base` @ `a9be7b66485080893399ade87c7d34f81ad3e249` contains BOTH 3 safetensors shards (4.980+4.930+3.003 = **12.913GB**) and `pytorch_model.pt` (**16.814GB**) [VERIFIED: HF tree API this session]. Therefore safetensors-only must hold at LOAD time, not just at prefetch time — an external prefetch alone would be followed by dnallm's own full-repo pull dragging the 16.8GB .pt into the giants dir.

Recommended composite:
1. **Library (small, tested)**: extend `download_model`/`_get_model_path_and_imports` with an `allow_patterns: list[str] | None = None` passthrough; the evo-1 handler passes `["*.safetensors", "*.json", "*.txt", "README.md"]` for the evo-1 family only. Same-change pytest in `tests/models/`.
2. **Harness (small, tested)**: per-notebook env overrides — extend `_ENV_OVERRIDES`/`run_notebook` so the evo notebook's kernel gets `HF_HUB_CACHE=~/models-giants/hub` (keeping the notebook content natural, `source="huggingface"`).
3. **CI**: prefetch step warms `~/models-giants` once (12.9GB, survives nightly, outside cached paths — never evicts the 10GB warm cache).

Rejected alternatives: symlink from `~/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base` into the giants dir (actions/cache either follows symlinks → 12.9GB enters the quota cache, or preserves them → dangling after restore — both fatal); `source="local"` route (works — `_get_model_path_and_imports` local branch is `model_path = model_name` [VERIFIED: dnallm/models/model.py:416-419] — but rewrites tutorial content and passes `revision="main"` into `from_pretrained(local_dir)` whose tolerance is unverified [ASSUMED]).

### Pattern: staged pytest invocations for D-07 coexistence

The mcp pair and the 6 MCP live probes must run ONLY in their stages. Concretely: stage 1 runs `tests/examples` with the mcp pair deselected (e.g. `--deselect` entries or a stage-2-only marker), stage 2 runs `dnallm/mcp/tests` with the server up, stage 3 re-runs just the two gated mcp tests (server up + ollama probed). Existing gates make this safe: `_gate_ollama_stack` skips typed whenever either endpoint is down [VERIFIED: tests/examples/test_notebook_execution.py:505-534], so a stage-order mistake degrades to an honest typed skip, never a false green.

### Pattern: wheel caching for source builds

flash-attn (~50 min sm_120 build) and mamba-ssm/causal_conv1d rebuild on every fresh-venv run under `--no-cache-dir --no-build-isolation` (documented in the test-mamba job comment [VERIFIED: .github/workflows/ci.yml:270-279]). Example job should instead build once into a wheel dir and `actions/cache` it keyed on `package version + sw version + runner arch`, then `uv pip install <wheel>` — turns a ~50-90 min nightly tax into seconds.

### Anti-Patterns to Avoid
- **Installing gated-family prerequisites into pyproject extras** — 05-FEASibility locked them as job-step-only (project venv/pyproject porcelain-empty proven during the census).
- **`pip install evo-1`** — unrelated PyPI package; the prerequisite is `evo-model` (05-FEASIBILITY).
- **Installing evo2's `transformer-engine`** — the empty TE meta package raises RuntimeError that escapes vortex's ImportError guard; TE must remain absent [CITED: 05-FEASIBILITY.md].
- **Treating a typed skip as success** — every skip is junit-audited against expected_skips.yaml; fallback (D-13) is for missing infrastructure ONLY, never for wrong model output.
- **Editing the patched NT cache in place as a "fix"** — cache edits are invisible, unportable, and mask shim gaps (Pitfall 1).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Notebook execution semantics | custom executor / nbmake | existing nbclient harness (`run_notebook`, budgets, artifacts) | EXEC-01/06 proven; per-cell timeout + kernel-kill + partial artifacts already contracted |
| ollama service management | nohup/supervisor scripts in job | systemd unit (in-repo, owner-enabled once) | D-12; survives job teardown; auditable |
| Readiness probing | ad-hoc sleep loops | D-13 retry window over `curl /api/tags` with evidence into the skip message | honest-gates contract |
| Cache eviction logic | manual cache pruning scripts | actions/cache key rotation via models.lock edits (existing mechanism: "Edit an entry to rotate the cache key" [VERIFIED: models.lock:2-3]) | the lock IS the key; rotation is designed |
| transformers-4 API restoration | per-notebook try/except patches | the established absence-gated shim pattern in `transformers_compat.py` (closed map, verbatim vendoring, idempotent sentinels) | 9 precedent patches; extending the closed map is the documented route for any new rung [VERIFIED: dnallm/utils/transformers_compat.py:501-505] |
| marimo UI assertions | DOM scraping of exported HTML | assert app-level default values via the app's module/HTML content checks per D-18 | export already proves the reactive graph ran; defaults assertions target app state |

**Key insight:** every mechanism this phase needs already has an in-repo precedent — the plan's job is composition (job stages, cache tiers, infra files), not invention.

## Common Pitfalls

### Pitfall 1: The dev-box NT snapshot is hand-patched (evidence poison)
**What goes wrong:** `~/.cache/modelscope/hub/models/zhangtaolab/nucleotide-transformer-v2-100m-promoter/` currently contains `config.json.dnallm-bak` and `modeling_esm.py.dnallm-bak` — the snapshot was patched by the 05-06 orchestrator probe [VERIFIED: directory listing this session; census note "snapshot PATCHED by the orchestrator probe"]. The ACTIVE benchmark notebook therefore passes locally partly on the hand patch, not purely on the shims.
**Why it happens:** the probe predates the sl7 shim landing and was never reverted.
**How to avoid:** restore pristine (`mv` the `.dnallm-bak` files back or delete + re-download the snapshot), then re-execute the benchmark notebook and NER paths on the dev box BEFORE the D-03 reconciliation claims shim-only green. The runner cache is clean, so the runner would otherwise surface this as a first-nightly surprise.
**Warning signs:** any local NT-family green that cannot be reproduced after `rm -rf` of the cached snapshot.

### Pitfall 2: evo-1-8k remote code under transformers 5.17 is unproven
**What goes wrong:** the FEASIBILITY verdict for evo-1-8k was taken under transformers 4.57.6 in the throwaway venv — "the 8k remote code constructs without `pos_idx_in_fp32` so it plausibly also runs under 5.x, untested" [CITED: 05-FEASIBILITY.md]. First execution in the project venv (5.17) may surface new remote-code rungs (plus the known `np.fromstring` shim requirement for stripedhyena's CharLevelTokenizer).
**How to avoid:** budget the evo family for discovery; run the 8k variant on the dev box first (D-03); extend the closed map/shim set per the established pattern as rungs appear.
**Warning signs:** any `AttributeError`/`ImportError` naming `stripedhyena`/remote `positional_embeddings.py`.

### Pitfall 3: models.lock growth busts the 10GB cache quota
**What goes wrong:** GitHub caches are capped at 10GB per repository, LRU-evicted when exceeded (a pay-as-you-go tier above 10GB exists since Nov 2025) [CITED: docs.github.com caching docs + github.blog]. New entries ≈ 9×0.34-0.41GB (zhangtaolab ms set) + evo2 2.7GB + megaDNA 0.58GB + InstaDeepAI ~0.5GB + PlantCAD ≈ 7-9GB on top of the existing warm cache — likely over quota → nightly LRU churn re-downloads evicted models.
**How to avoid:** measure the cache total on the runner when adding entries; keep the evo-1 giant OUT (12.9GB, CI-05); decide exclusions vs pay-as-you-go with the owner (Open Question 2).
**Warning signs:** nightly logs showing model re-downloads that were warm the previous night.

### Pitfall 4: prefetch-only safetensors is defeated by the loader's full-repo pull
**What goes wrong:** a CI step that fetches evo-1 safetensors-only into a giants dir does not help if the notebook's load path then runs dnallm's own `snapshot_download` without `allow_patterns` — it downloads the missing 16.8GB `pytorch_model.pt` into the giants dir [VERIFIED constraint: dnallm/models/model.py:348].
**How to avoid:** the allow_patterns passthrough must live in the library load path (Pattern above), not only in the prefetch step.

### Pitfall 5: runner has no sudo — system deps need rootless fallbacks
**What goes wrong:** `apt-get install bedtools` in a job step fails — the self-hosted runner user lacks sudo (the ci.yml free-disk step comment documents exactly this) [VERIFIED: .github/workflows/ci.yml:34-37 comment; dev-box sudo requires a password].
**How to avoid:** probe `command -v bedtools`; fallback to micromamba/bioconda into a job-local prefix or a cached static bedtools binary on PATH.
**Warning signs:** job step failing with permission denied on apt.

### Pitfall 6: three nightly jobs, one runner — staggering is queue ordering, not parallelism
**What goes wrong:** a single self-hosted runner executes one job at a time; coverage-nightly (900-min cap), test-mamba (180-min) and the new example job will queue. A bad cron choice makes the example job start arbitrarily late.
**How to avoid:** pick the D-06 stagger with the measured durations in mind (coverage 03:00 UTC; example at e.g. 05:30 UTC still overlaps a long coverage run — GitHub queues, never parallel-runs on one runner box, which is exactly the D-06 intent: no GPU/disk contention).
**Warning signs:** example job start timestamps far past their cron time.

### Pitfall 7: mcp stage ordering vs the ollama gate's fail-loud history
**What goes wrong:** running the full `tests/examples` in stage 1 includes the mcp pair, which (server down) honestly typed-skips — then stage 3 re-runs them for real; if the plan forgets the deselect, the pair executes once (skip) and once (real), which is harmless but confusing in junit, and the skip counts land in the audit.
**How to avoid:** explicit stage-1 deselection of the two mcp gated ids; the audit stays meaningful.

### Pitfall 8: the megaDNA unpinned clone cell inside finetune_generation
**What goes wrong:** the notebook contains `!git clone https://github.com/lingxusb/megaDNA.git` + unpinned install; FEASIBILITY locked the pinned clone (`cb2f5ab4...` + `MEGABYTE_pytorch==0.2.1`) as the only sanctioned route [VERIFIED: notebook URL grep; CITED: 05-FEASIBILITY.md].
**How to avoid:** content-repair the clone cell to the pinned commit (D-02 allows the edit; the finetune_generation wget-ordering fix rides the same commit).

## Code Examples

### Current models.lock entry format (copy exactly — D-15)
```
# Source: models.lock:4-14 (verbatim)
hf  microsoft/DialoGPT-small                         # tests/models/test_model.py::test_download_real_huggingface_connection
ms  zhangtaolab/plant-dnabert-BPE                    # tests/finetune/test_trainer_real_model.py (12 call sites, source=modelscope)
```
Two spaces after the prefix tag, aligned columns, trailing `#` purpose comment. New entries add a revision pin (commit sha) per D-15 — the current file has none yet, so the format grows a pinned form; keep it greppable (e.g. `@<sha>` or a `rev=` field — planner picks one; CI-08's notebook-vs-lock drift test arrives in Phase 9 and will parse it).

### models.lock candidate additions (registry-verified this session)

All executed ids from the census lanes. `ms` = ModelScope-first (owner rule; all zhangtaolab ids verified HTTP 200 on ModelScope with sizes); `hf` = MS 404 fallback.

| id | prefix | MS size (verified) | HF sha (verified, main) | Used by |
|----|--------|--------------------|--------------------------|---------|
| zhangtaolab/plant-dnagpt-BPE | ms | 0.37GB | — | generation, finetune_multi_labels, finetune_custom_head |
| zhangtaolab/plant-dnagpt-6mer | ms | 0.36GB | — | finetune_NER_task/data_generation_and_inference |
| zhangtaolab/plant-dnagpt-singlebase | ms | 0.34GB | — | finetune_generation |
| zhangtaolab/plant-nucleotide-transformer-BPE | ms | 0.41GB | — | finetune_NER_task notebook + generate_bpe_dataset.py |
| zhangtaolab/nucleotide-transformer-v2-100m-promoter | ms | 0.38GB | — | benchmark third model |
| zhangtaolab/tRNADetector | ms | 0.36GB | — | inference_for_tRNA |
| zhangtaolab/tRNAPointer | ms | 0.37GB | — | inference_for_tRNA |
| zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf | ms | 0.37GB | — | interpretation |
| zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast | ms | 0.37GB | — | in_silico_mutagenesis |
| InstaDeepAI/nucleotide-transformer-v2-50m-multi-species | **hf** (MS 404) | — | `81b29e5786726d891dbf929404ef20adca5b36f1` | embedding_attention (raw AutoModel route) |
| togethercomputer/evo-1-8k-base | hf recommended (MS 200 exists but MS route unproven for evo) | — | `a9be7b66485080893399ade87c7d34f81ad3e249` | generation_evo_models (after D-06 ref update); **giants tier — NOT in cache** |
| arcinstitute/evo2_1b_base | hf (MS 200, unproven route) | — | `2279e1df422c991037470302360edd40d0d2ea1e` | generation_evo_models (2.7GB via dnallm HF snapshot — fits cache) |
| lingxusb/megaDNA_updated | **hf** (MS 404) | — | `ed298be539e1667b52a1181a6472528a34dd2ef9` | generation_megaDNA, finetune_custom_head, finetune_generation (0.58GB) |
| kuleshov-group/PlantCAD2-Small-l24-d0768 | **hf** (MS 404) | — | `f756c255cb76e9f538c3acec04acf4214ed03fb3` | lora pair |
| dataset zhangtaolab/plant-multi-species-core-promoters | already in lock | — | — | — |

[VERIFIED: HF API + ModelScope API probes this session, 2026-10-04.] MS revision shas for the `ms` rows: query `https://modelscope.cn/api/v1/models/<id>/repo/revisions` at execution time [ASSUMED endpoint — verify]. Note D-15's alignment rule: several notebooks currently carry BOTH a `source="huggingface"` and a `source="modelscope"` call variant (e.g. finetune_binary shows both in one grep) — alignment edits ride each family's repair commit.

### evo-1 giants prefetch (CI step shape)
```python
# allow_patterns per CI-05 — files verified on HF main this session:
# model-0000{1,2}-of-00003.safetensors 4.98+4.93GB, model-00003-of-00003.safetensors 3.00GB,
# model.safetensors.index.json, config.json, generation_config.json,
# special_tokens_map.json, tokenizer_config.json, README.md   (exclude pytorch_model.pt 16.81GB)
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "togethercomputer/evo-1-8k-base",
    revision="a9be7b66485080893399ade87c7d34f81ad3e249",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "README.md"],
    cache_dir=os.path.expanduser("~/models-giants/hub"),
)
```

### ollama systemd unit (in-repo template; D-12)
Stock unit verified live on the dev box at `/etc/systemd/system/ollama.service`:
```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=..."   # genericize: the dev-box unit carries box-specific miniconda/linuxbrew paths

[Install]
WantedBy=default.target
```
[VERIFIED: file read this session.] Add for D-12's loopback-only auditability: `Environment="OLLAMA_HOST=127.0.0.1:11434"` — ollama "binds 127.0.0.1 port 11434 by default. Change the bind address with the OLLAMA_HOST environment variable" [CITED: docs.ollama.com/faq]; systemd env vars are set via `systemctl edit ollama.service` `Environment` lines + daemon-reload + restart [CITED: docs.ollama.com/faq]. Optionally `OLLAMA_MODELS=` to pin model storage. One-time runner setup: install ollama, `systemctl enable --now ollama`, `ollama pull qwen3.8:latest` (17.74GB).

### The closed-map shim (the D-17 pattern to extend if new rungs appear)
```python
# Source: dnallm/utils/transformers_compat.py:507-510 (verbatim)
_LEGACY_PRETRAINED_CONFIG_DEFAULTS: dict[str, object] = {
    "is_decoder": False,
    "add_cross_attention": False,
}
```
Module guidance, verbatim [dnallm/utils/transformers_compat.py:504-505]: "If execution surfaces another removed 4.x config default that remote code reads, extend the closed map -- never a catch-all."

### Existing infrastructure facts the job steps will quote
- coverage-nightly cache config [VERIFIED: .github/workflows/ci.yml:458-465]: paths `~/.cache/huggingface/hub` + `~/.cache/modelscope/hub`, key `${{ runner.os }}-models-${{ hashFiles('models.lock') }}`, restore-keys `${{ runner.os }}-models-`. D-09 = example job reuses this exact key.
- Event gating for runner security [VERIFIED: .github/workflows/ci.yml:413]: `if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'`.
- MCP server endpoint config [VERIFIED: dnallm/mcp/configs/mcp_server_config.yaml]: `host: "0.0.0.0"`, `port: 8000`, three lazy models (promoter/conservation/open_chromatin) matching the notebooks' tool calls. NOTE: CLI `--host/--port` are dead flags — yaml always wins (STATE.md deferred item); the stage script must not rely on CLI overrides.
- The evo-1 handler's revision logic [VERIFIED: dnallm/models/special/evo.py:371]: `revision = "1.1_fix" if "." in model_name and source == "huggingface" else "main"` — `evo-1-8k-base` has no dot ⇒ fetches `main`; lock pin should use the observed main sha `a9be7b6...`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 05-04 ruling: NT structural rung "not vendored-pure-helper territory" → typed skip | 9-shim absence-gated compat layer; NT family executes green | 2026-10-02 (sl7, owner instruction) | D-17 starts from "already fixed"; only evidence gaps remain |
| mcp pair under never-auto-execute sentinel (T-05-16) | EXECUTE state: both-up executes, any-down typed-skips with both probes | 2026-10-03 (261003-csd) | Stage 3 is execution, not skip bookkeeping |
| GitHub cache hard 10GB/repo cap | pay-as-you-go beyond 10GB available | 2025-11-20 | Quota decision has an owner-priced escape valve |
| evo-1 unrunnable on any in-span transformers (131k variant) | 8k variant FEASIBLE under 4.57.6 | 2026-10-02 (05-FEASIBILITY) | Phase 8 executes 8k + updates notebook ref (D-06) |

**Deprecated/outdated to watch:** evo2 PyPI 0.6.0 supersedes the spike-proven 0.3.0 (pin 0.3.0); MEGABYTE_pytorch 0.3.6 supersedes the repo-pinned 0.2.1 (pin 0.2.1); ollama FAQ example uses `0.0.0.0` — do NOT copy that binding (loopback-only per D-12).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Runner box is the same GB10 class as the dev box (disk ≥ ~50GB free for giants + new cache entries, CUDA toolchain for source builds) | Environment Availability | First-nightly failures; mitigate with an early probe step in G0 |
| A2 | ollama is NOT yet installed/enabled on the runner and qwen3.8 is not pulled there (dev box has it; runner state unobservable from here) | MCP-01 | Stage 3 typed-skips forever-green; the D-13 fallback would fire legitimately — the owner manual step is on the critical path |
| A3 | `from_pretrained(local_dir, revision="main")` tolerates the revision kwarg (matters only for the rejected local-source giants option) | Patterns | None if recommended option (env override) is chosen |
| A4 | transformers `from_pretrained` loads a snapshot containing only safetensors shards + index + configs (no .pt) without re-fetching | Patterns (giants) | Load-time re-download of .pt; executor verifies in the first evo run |
| A5 | MS revision-pin API `.../repo/revisions` yields a commit sha for ms lock entries | models.lock table | ms entries ship unpinned or need a different endpoint — cosmetic, not blocking |
| A6 | Existing warm cache ≈ 3.5-4.5GB (estimated from lock entries, not measured) | Pitfall 3 | Quota arithmetic shifts; the plan measures before deciding |
| A7 | evo-1-8k remote code constructs under transformers 5.17 (spike verdict was 4.57.6; FEASIBILITY itself says "untested") | Pitfall 2 | New shim rungs in family G3 — budgeted as discovery |
| A8 | Pay-as-you-go cache billing is not enabled for this repo (owner billing choice unknown) | Pitfall 3 | If enabled, quota pressure disappears; owner decision point |
| A9 | mamba-ssm/causal_conv1d source builds succeed on the runner (test-mamba job exists but its post-merge runner leg is still pending dispatch confirmation per 05 D-04) | Family G5 | lora pair stays optional-dep typed-skip — conflicts with EXEC-02; escalate to owner |
| A10 | bedtools is absent on the runner (unknown; dev box has it via linuxbrew) | Pitfall 5 | Job-step fallback (micromamba/static) covers it |
| A11 | `rice.uga.edu` + Ensembl FTP are reachable from the runner (proven from the dev box only) | Script lane / G4 | 5xx falls into the script test's sanctioned `network-unavailable:` path; 4xx re-raises loudly (WR-04 semantics) |

## Open Questions

1. **Runner environment ground truth** (A1/A2/A9/A10/A11) — ollama? bedtools? disk? build toolchain? rice/Ensembl reachability?
   - What we know: dev-box parity for everything; runner unobservable from here.
   - Recommendation: G0's first dispatch doubles as the probe — add explicit `command -v`/`curl`/`df -h` echo steps to the job skeleton so the first nightly run reports the runner inventory as CI logs.
2. **Cache quota strategy** (Pitfall 3, A6/A8) — measure the post-growth cache; if >10GB: exclude which entries (evo2 2.7GB is the biggest non-giant), or owner opts into pay-as-you-go?
   - Recommendation: measure in G7; present numbers to owner; keep giants out regardless.
3. **evo-1 lock prefix** — `ms` exists (HTTP 200) but the MS snapshot route for the evo handler is unproven; `hf` route is spike-proven.
   - Recommendation: `hf togethercomputer/evo-1-8k-base` + notebook `source="huggingface"`; note the MS mirror in the lock comment.
4. **Does the example job install the `[mamba]` extra?** D-10's locked line omits it, but EXEC-02's "all 21" requires it for the lora pair.
   - Recommendation: treat the mamba build as a job step (test-mamba precedent, wheel-cached) — flag to owner if the nightly build cost proves unacceptable.
5. **finetune_generation input strategy** — the `!wget` cell exists (Ensembl release-62 URL verified in-notebook) yet the census died at cell 2 `Fasta(...)` — ordering repair vs seeded-extra input.
   - Recommendation: ordering/content repair per D-02 (the notebook must be self-sufficient as a tutorial), mirroring the rice-notebook precedent.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| GB10 GPU (dev box) | all execution | ✓ | compute cap 12.1, driver 580.178.04, CUDA 13.0 [VERIFIED: census environment record] | — |
| GB10 runner `[self-hosted, dnallm-nightly]` | nightly jobs | ✓ (job-proven for coverage-nightly class) | same class [ASSUMED A1] | dev-box dev-leg per D-03/D-04 |
| ollama + qwen3.8:latest (dev box) | mcp pair | ✓ | ollama at /usr/local/bin, qwen3.8:latest 17.74GB (qwen3.8:27b-q4_K_M, 27.3B, Q4_K_M, tools+vision+thinking, ctx 262144) [VERIFIED: live API probe] | — |
| ollama on RUNNER | MCP-01 | ✗ unknown (assumed absent) | — | one-time owner setup (unit + pull); then D-13 probe |
| systemd ollama unit (dev box) | D-12 template | ✓ (running, `ollama serve` pid live; loopback default) | — | in-repo unit + `systemctl enable` |
| bedtools | script lane (pybedtools runtime) | ✓ dev box (linuxbrew v2.31.1) / ✗ unknown runner | 2.31.1 | micromamba/bioconda job-local prefix or cached static binary (no sudo) |
| pyfastx + pybedtools (python) | script + finetune_generation | ✓ dev venv [VERIFIED: import check] | dev extra | installed by D-10 line |
| rice.uga.edu + Ensembl FTP | NER script + finetune_generation | ✓ from dev box (census 779.8s cold run) | — | 5xx→`network-unavailable:` typed skip (script test); 4xx re-raises |
| HF + ModelScope APIs | all model fetches | ✓ [VERIFIED: probes this session] | — | retry-with-backoff in `download_model` |
| flash-attn build (sm_120) | evo family | ✓ proven on dev box (2.8.3.post1, ~50 min) | — | wheel cache; script-mode evo2 fallback exists for no-flash-attn boxes (dnallm ships `-noFA.yml` variants [VERIFIED: dnallm/models/special/evo.py config selection]) |
| Disk for giants (≥13GB) + cache growth (~8GB) | CI-05/CI-04 | ✓ dev box 2.3T free [VERIFIED: df] / runner unknown | — | giants prefetch is one-time; measure runner disk in G0 |

**Missing dependencies with no fallback:** none blocking the dev-box leg (D-03). Runner unknowns (ollama, bedtools, disk, toolchain) each have a documented fallback or owner step.

**Missing dependencies with fallback:** ollama-on-runner (owner setup; D-13 typed skip is the documented interim); bedtools-on-runner (rootless install); mamba build on runner (A9 — escalate if broken).

## Security Domain

ASVS level 1 (config: `security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | nothing new authenticates; ollama/MCP are loopback services with no auth added (existing posture) |
| V3 Session Management | no | no sessions introduced |
| V4 Access Control | marginal | loopback-only binding for ollama (D-12) is the access control; MCP server config binds `0.0.0.0` [VERIFIED: mcp_server_config.yaml] on an isolated single-tenant runner — note in plan; changing it is out of scope (CLI host flags are dead, deferred-items) |
| V5 Input Validation | yes | every example YAML re-validated through pydantic `load_config()` (EXEC-05) — the existing validation lane IS the control |
| V6 Cryptography | no | no new crypto |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Fork/PR code executing on the self-hosted GPU runner | Elevation of Privilege | event gating `schedule \|\| workflow_dispatch` only (existing pattern [VERIFIED: ci.yml:270,413]) — the new example job MUST copy it verbatim |
| trust_remote_code model code running on the runner | Tampering / Elevation | accepted existing risk (scoped to pinned repo ids; models.lock revision pins (D-15) tighten provenance — every executed id gets a commit sha) |
| Notebook `!uv pip install` cells mutating the project venv | Tampering | isolated kernelspec `dnallm-mcp-langchain` with `VIRTUAL_ENV` pinned to a throwaway venv [VERIFIED: tests/examples/_execution.py:76-81] — already proven; REPAIR-04's pyproject declaration does NOT remove the isolation |
| ollama exposed on the network | Information Disclosure | loopback-only default + explicit `OLLAMA_HOST=127.0.0.1:11434` in the in-repo unit (D-12) |
| In-repo service definition drift from the live runner | Tampering (config drift) | unit file lives in-repo with README; one-time manual enable documented (D-12 auditability requirement) |
| Supply-chain: gated-family packages | Tampering | legitimacy audit above (all authoritative-source, spike-proven); version pins (`evo2==0.3.0`, `MEGABYTE_pytorch==0.2.1`, clone @ `cb2f5ab4...`) prevent floating installs; never install the `evo-1` PyPI name |

## Sources

### Primary (HIGH confidence — verified in-session by direct tool call)
- Repo reads: `tests/examples/_execution.py`, `tests/examples/test_notebook_execution.py`, `tests/examples/test_script_execution.py`, `tests/examples/test_marimo_execution.py`, `tests/configuration/test_yaml_load.py`, `dnallm/utils/transformers_compat.py` (all 9 shims), `dnallm/models/model.py:317-492`, `dnallm/models/special/evo.py`, `dnallm/models/special/megadna.py:28-144`, `dnallm/inference/benchmark.py:285-305`, `.github/workflows/ci.yml`, `models.lock`, `tests/expected_skips.yaml`, `pyproject.toml` extras, `dnallm/mcp/configs/mcp_server_config.yaml`, `/etc/systemd/system/ollama.service`
- Live executions this session: `scripts/validate_yaml.py` (21/21 OK); local ollama `/api/tags` probe (qwen3.8:latest full details); dev-box cache inspection (NT `.dnallm-bak` pollution); `pyfastx`/`pybedtools` import check; `df`/`nvidia-smi`/`sudo -n` probes
- Registry APIs: Hugging Face model API + tree API (shas, file lists, evo-1-8k sizes); ModelScope API (existence + repo file sizes for 14 ids); PyPI JSON API (7 packages)
- `.planning/phases/05-.../05-CENSUS.md`, `05-FEASIBILITY.md`, `05-CONTEXT.md` (D-04/05/06/08/09), `.planning/phases/07-.../07-CONTEXT.md` (D-13/14/16), `.planning/quick/261002-sl7-.../261002-sl7-{PLAN,SUMMARY}.md`

### Secondary (MEDIUM confidence)
- [CITED: docs.ollama.com/faq] — loopback default bind, OLLAMA_HOST, systemd Environment editing
- [CITED: docs.github.com caching docs + github.blog (2025-11-20)] — 10GB/repo cache cap, LRU eviction, pay-as-you-go tier

### Tertiary (LOW confidence / ASSUMED)
- Runner-box state (A1/A2/A9/A10/A11); MS revisions endpoint shape (A5); warm-cache size estimate (A6); `from_pretrained(local_dir, revision=...)` tolerance (A3); safetensors-only snapshot load behavior (A4)

### Knowledge graph
`.planning/graphs/graph.json` exists but is stale (105h old, 371 commits behind, built at 3df056c) — it predates the Phase 5-7 artifacts that form this phase's core context, so semantic relationships were taken from the primary documents instead.

## Metadata

**Confidence breakdown:**
- Rollout current-state: HIGH — every lane/gate/shim read from source this session
- Registry/size/quota facts: HIGH — HF/MS/PyPI APIs probed live; quota policy cited from official docs
- ollama infra: HIGH on dev box + docs; LOW on runner state (A2)
- evo-under-5.17 risk: LOW-to-MEDIUM by design (A7 — the phase's discovery budget)
- Runner environment: LOW (unobservable from dev box; G0 probe step recommended)

**Research date:** 2026-10-04
**Valid until:** 2026-10-18 (repo facts drift as repairs land; registry facts stable ~30 days)
