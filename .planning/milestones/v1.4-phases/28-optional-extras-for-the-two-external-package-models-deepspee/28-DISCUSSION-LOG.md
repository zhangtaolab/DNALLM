# Phase 28: Optional extras: deepspeed & flash-attn model support - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-29
**Phase:** 28-Optional extras: deepspeed & flash-attn model support
**Areas discussed:** Census & disposition semantics, CI lane architecture, Extra composition & install mechanics, Support words & docs surface
**Interaction language:** Chinese (owner request mid-session; technical terms in English)

---

## Census & Disposition Semantics

### Q1: Disposition value for extras-green entries

| Option | Description | Selected |
|--------|-------------|----------|
| 新增 extra-required-tf5 (Recommended) | New status value mirroring shim-required-tf5 (27.2 D-03): verdict records the green load AND what it required; registry/docs/support.py learn one new enum | ✓ |
| 直接翻绿 | Flip to verified/import-ok with evidence naming the extra; page would claim "works" for default-install users who cannot load it | |
| 保持 broken,只改措辞 | Keep remote-code-broken; words/docs only; contradicts roadmap flip language and the CI lane's green probe | |

**User's choice:** 新增 extra-required-tf5
**Notes:** Owner first asked "tf5 是什么?" — explained: tf5 = transformers 5; registry fields `tf_compat_*`; the new value means "measured green under transformers 5 once the named extra is installed", same honesty structure as shim-required-tf5.

### Q2: Census measurement convention going forward

| Option | Description | Selected |
|--------|-------------|----------|
| census 保持默认安装 (Recommended) | Census lane keeps measuring the plain-install experience; the two rows seed as expected failures (CR-01 discipline); extras evidence from separate probes | ✓ |
| census lane 加装 extras | Census installs extras before running; convention becomes best-case env; 49-matched gate semantics change; lane runtime grows by flash-attn build time | |

**User's choice:** census 保持默认安装

### Q3: Evidence grade for the flip

| Option | Description | Selected |
|--------|-------------|----------|
| 复现级证据,阶段内翻转 (Recommended) | Local runner-box replica env runs the green probe (extras installed, load+forward); committed flip lands in-phase; workflow lands on v1.4 simultaneously (27.2 SHIM_RESTORED_LOADS precedent) | ✓ |
| 等 live lane 变绿再翻 | Phase ships extras/lane/words with "coming" wording; flip deferred to post-merge live run; incomplete deliverable, inconsistent with 27.2 precedent | |

**User's choice:** 复现级证据,阶段内翻转

### Q4: BigBird's dual dependency (rung + extra)

| Option | Description | Selected |
|--------|-------------|----------|
| 单值 + evidence 写明 (Recommended) | Single extra-required-tf5 value; BigBird's evidence names both the rung and the extra (27-03 precedent: rung rides in evidence, not a new enum) | ✓ |
| 组合枚举值 | New shim+extra-required-tf5 combined enum; enum bloat for near-zero information gain | |

**User's choice:** 单值 + evidence 写明

**User then chose "继续问本区域" once before moving on.**

---

## CI Lane Architecture

### Q1: Lane home

| Option | Description | Selected |
|--------|-------------|----------|
| 新建 extras_compat.yml (Recommended) | New dedicated workflow, self-hosted GPU box, schedule/dispatch only; census_compat.yml byte-untouched | ✓ |
| census 工作流内加 job | Second job inside census_compat.yml; fewer registrations but coupled scheduling and longer census trigger cycle | |

**User's choice:** 新建 extras_compat.yml

### Q2: Trigger cadence

| Option | Description | Selected |
|--------|-------------|----------|
| nightly + dispatch + 开关 (Recommended) | Own cron + workflow_dispatch + fourth boolean `run_extras_compat` in ci.yml nightly selection (261026-b34 pattern) | ✓ |
| 仅手动 dispatch | Zero periodic box load but no drift guard (upstream model-repo / package changes never self-expose) | |

**User's choice:** nightly + dispatch + 开关

### Q3: Probe tooling

| Option | Description | Selected |
|--------|-------------|----------|
| 复用 census 工具 (Recommended) | census_compat.py single-entry mode in the extras env; identical evidence format; writer/drift-gate/docs zero adaptation | ✓ |
| 独立轻量脚本 | Separate load+forward script; second evidence pipeline with long-term drift risk | |

**User's choice:** 复用 census 工具

### Q4: Pass-gate strictness

| Option | Description | Selected |
|--------|-------------|----------|
| 探针绿 + 注册表匹配 (Recommended) | Two gates: probes green AND fresh records match committed extra-required-tf5 rows (census drift-gate mirror, fail-closed); support-word consistency stays in pytest offline gate | ✓ |
| 只门探针绿 | Simplest; red can't distinguish "model broke" from "evidence stale" | |
| 全套门都进 lane | Adds docs-regen byte-identity to the lane; redundant with existing gates, more maintenance | |

**User's choice:** 探针绿 + 注册表匹配

---

## Extra Composition & Install Mechanics

### Q0 (freeform pivot): CrossDNA extra naming/composition

**Options first presented:** flash-attn 组合 fla / crossdna 模型名 / 扩建现有 fla extra
**User's free-text response:** "PlantHelixSeek 必须默认支持，不能通过外挂支持，必须直接进 dnallm" — an owner directive that reframed the area: fla must leave the extras system entirely.

**Follow-up confirmed:** fla (with the WPAR-07 Windows triton-windows marker line moved verbatim) is promoted into `[project]` base dependencies; the fla extra is deleted; `all` drops its fla reference; `optional_dependencies.md` PlantHelixSeek section becomes "built-in". Breaking change (dnallm[fla] syntax disappears) accepted.

| Option | Description | Selected |
|--------|-------------|----------|
| 彻底晋升+删除 extra (Recommended) | fla into base deps; extra definition deleted; no zombie alias | ✓ |
| 晋升+保留兼容 extra | Empty compat alias for dnallm[fla]; zombie extra to document | |

**User's choice:** 彻底晋升+删除 extra

### Q1: CrossDNA extra name (after fla became base)

| Option | Description | Selected |
|--------|-------------|----------|
| flash-attn (Recommended) | Contains only flash-attn itself; named for the missing package, consistent with deepspeed extra naming | ✓ |
| crossdna 模型名 | Model-named extra opens a naming convention with poor discoverability | |

**User's choice:** flash-attn

### Q2: Platform markers

| Option | Description | Selected |
|--------|-------------|----------|
| 不加标记,大声失败 (Recommended) | Unsupported platforms fail loudly at install (compile error); docs state Linux + CUDA | ✓ |
| 加标记,静默跳过 | `platform_system != 'Windows'` markers; resolve-green but silently empty extras | |

**User's choice:** 不加标记,大声失败

### Q3: Flash-attn CI install mechanics (GB10 aarch64 + cu130 + torch 2.11)

| Option | Description | Selected |
|--------|-------------|----------|
| sha256 固定脚本 (Recommended) | New install_flash_attn.sh mirroring install_mamba.sh: pinned version, hash gate before execution, mirror support; wheel if available else pinned sdist; lane never bare-pips; extra = end-user route | ✓ |
| 裸 pip + band | Simplest but inconsistent with the mamba supply-chain discipline; unbounded compile time | |
| uv index 钉住 | Requires new wheel-hosting infrastructure; out of scope | |

**User's choice:** sha256 固定脚本
**Notes:** Wheel availability for the GB10 combo is a research-phase fact-finding item; the mechanism stance is locked now.

### Q4: Meta-extra membership

| Option | Description | Selected |
|--------|-------------|----------|
| 都不进 (Recommended) | Neither extra joins all/full; both are heavy compile packages; `all` slims naturally with fla's promotion | ✓ |
| 都进 all | "Everything" semantics literal but makes .[all] Linux-CUDA-only | |
| 只 deepspeed 进 | Asymmetric rule, hard to document | |

**User's choice:** 都不进

---

## Support Words & Docs Surface

### Q1: Support-word phrasing

| Option | Description | Selected |
|--------|-------------|----------|
| works with X extra (Recommended) | `works with the deepspeed extra` / `works with the flash-attn extra`, parallel to `works with built-in fixes`; single vocabulary across docs/TUI/MCP; pip command in docs detail only | ✓ |
| 词直接写命令 | Word embeds the pip command; breaks TUI/MCP column layout (27.2-05 tail-append) | |

**User's choice:** works with X extra

### Q2: DOCS_DATA_DATE movement

| Option | Description | Selected |
|--------|-------------|----------|
| 随翻转再生成 (Recommended) | Page regenerates with the flip; date follows the evidence change; single-entry probes suffice (quick-261029-c95 precedent); byte-identity gate holds | ✓ |
| 只全量 census 才动日期 | Date would lag page content | |

**User's choice:** 随翻转再生成

### Q3: optional_dependencies.md rewrite depth

| Option | Description | Selected |
|--------|-------------|----------|
| 受影响面全改 (Recommended) | New "what is an optional extra" intro; four-row summary-table update; BigBird/CrossDNA sections become practical guides (command + platform + compile-time expectations); GPN/Omni-DNA verbatim | ✓ |
| 最小补丁 | Two sections + table rows only; no concept on-ramp for new users | |

**User's choice:** 受影响面全改

### Q4: Missing-extra error guidance

| Option | Description | Selected |
|--------|-------------|----------|
| 加载链拦截增强 (Recommended) | Intercept the relevant ImportError, append one plain-language line naming the extra + exact pip command (D-10 precedent shape); interception layer is implementer discretion | ✓ |
| 仅文档引导 | Docs-only; bare ModuleNotFoundError at the failure site; roadmap explicitly requires error messages naming extras | |

**User's choice:** 加载链拦截增强

---

## Claude's Discretion

- Version bands for deepspeed / flash-attn (research-measured)
- deepspeed lane install route (default: plain extra via pip; escalate to pinned script only on measured need)
- Exact evidence-string format (existing `hub repo@revision; cause` pattern)
- Drift-gate seed mechanics for the two expected-failure rows
- Offline-gate pytest file placement
- BigBird generic-chain interception layer choice
- README statement extras mention (only if it currently names the two models as broken)
- Plan split / wave structure

## Deferred Ideas

- GPN / Omni-DNA as extras — permanently out (version-locked packages; roadmap-locked guidance cases)
- lgq12697 bulk ingestion (~130 entries) — v1.5 campaign (carried from 27.2)
- Live dispatch of census + extras lanes — post-merge operator steps
- uv index pinning for flash-attn — rejected mechanism, revisit only on future need
