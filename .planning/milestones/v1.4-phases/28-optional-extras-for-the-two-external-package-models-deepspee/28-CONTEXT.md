# Phase 28: Optional extras: deepspeed & flash-attn model support - Context

**Gathered:** 2026-10-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the two fixable external-package models installable via curated optional extras, following the mamba-extra precedent: a `deepspeed` extra for GENA-LM-BigBird (`AIRI-Institute/gena-lm-bigbird-base-sparse`) and a `flash-attn` extra for CrossDNA (`chengCCC/CrossDNA_pretrain`), each backed by CI lane coverage and a measured-green-only disposition flip to a new `extra-required-tf5` registry status. The user-facing surface (support words, `optional_dependencies.md`, `model_compatibility.md`, load-chain error messages) updates to name the extras.

**Owner-directed scope addition (2026-10-29, mid-discussion):** PlantHelixSeek (`zhangtaolab/PlantHelixSeek`, a lab model) must be supported out-of-the-box, not via an extra — `fla` (flash-linear-attention) is promoted from optional extra to base dependency. This rides this phase because it restructures the same optional-dependency architecture.

Inherited unchanged: census discipline (dispositions flip ONLY on measured green probes; replica-grade evidence accepted until lanes are live post-merge); GPN and Omni-DNA stay version-locked guidance cases, untouched; every `dnallm/` change ships with pytest in the same change.

</domain>

<decisions>
## Implementation Decisions

### Census & Disposition Semantics

- **D-01: New registry status value `extra-required-tf5`**, mirroring the `shim-required-tf5` precedent (27.2 D-03): the verdict records BOTH the green load AND what it required (the extra) — never a "works as-is" claim. One new enum value; registry schema (`extra="forbid"`, insertion-only), docs status cells, and `support.py` all learn it.
- **D-02: The census keeps measuring the default install, forever.** `census_compat.yml` is byte-untouched; its default-install convention and 49-matched drift-gate semantics are unchanged. The two `extra-required-tf5` records keep measuring as failures in the default environment and are seeded into the drift gate as expected failures (the CR-01 unreachable-repo discipline). Extras-environment green evidence comes from separate probes that flip only these two entries.
- **D-03: Replica-grade evidence, in-phase flip** (27.2 `SHIM_RESTORED_LOADS` precedent): the local runner-box replica env (extras installed) runs the green probes (load + forward), and the committed `extra-required-tf5` flips land inside this phase. The workflow file lands on v1.4 simultaneously; the live lane becomes the durable guard after the v1.4→main merge. — **Reversibility:** reversible — registry rows flip back if a later probe goes red (green-drop rule symmetry).
- **D-04: Single status value; evidence names ALL dependencies.** BigBird's green load needs BOTH the `_patch_apply_chunking_to_forward` rung AND the deepspeed extra — its evidence string names both (27-03 precedent: rung name rides in evidence, never a new enum). CrossDNA's evidence names the flash-attn extra (fla is base after D-09). No combined enums.

### CI Lane Architecture

- **D-05: New dedicated workflow `extras_compat.yml`** on the self-hosted GPU box, schedule/dispatch only, never PR-reachable (fork-PR safety rule). Single responsibility: install extras → run the two models' green probes → emit extras evidence. `census_compat.yml` gets zero edits.
- **D-06: Trigger cadence = nightly cron + `workflow_dispatch` + a fourth per-lane boolean** (`run_extras_compat`, default true) in the `ci.yml` nightly lane-selection block (261026-b34 pattern). Nightly guards upstream drift (model-repo changes, deepspeed/flash-attn upgrades); dispatch serves reproduction and the post-merge first live run.
- **D-07: Probes reuse `census_compat.py` single-entry mode** in the extras-equipped environment. Evidence format is identical to census records; the update-capable registry writer (27.2-01), drift gate, and docs generator need zero adaptation.
- **D-08: Two pass gates in the lane:** (1) both model probes green; (2) fresh extras-probe records match the committed registry `extra-required-tf5` records (mirror of the census drift gate, fail-closed) — upstream model-repo changes surface as red. Support-word/disposition consistency re-derivation stays in the pytest offline gate (27.1-01 pattern), not in the lane.

### Extra Composition & Install Mechanics

- **D-09: `fla` is promoted to a base dependency** (owner directive: PlantHelixSeek must work out-of-the-box). `flash-linear-attention>=0.5.2,<0.6` plus the WPAR-07 Windows `triton-windows` marker line move verbatim into `[project]` dependencies; the `fla` extra definition is deleted; the `all` meta-extra drops its `fla` reference; `optional_dependencies.md`'s PlantHelixSeek section becomes "built-in". Breaking change accepted: `dnallm[fla]` syntax disappears (upgrading users get fla naturally on reinstall). — **Reversibility:** costly — undoing means re-publishing the install contract (base dep set shrink + re-introducing an extra) in a future release.
- **D-10: The CrossDNA extra is named `flash-attn`** and contains only flash-attn itself (fla is base after D-09). Named for the missing package, consistent with the `deepspeed` extra naming rule; no model-named extras.
- **D-11: No platform markers on either extra.** pip fails loudly on Windows/macOS at install time (compile failure) rather than silently installing an empty extra. Docs state Linux + CUDA as the supported platform for both extras.
- **D-12: CI installs flash-attn via a new sha256-pinned installer script** (`scripts/install_flash_attn.sh`) mirroring `install_mamba.sh` discipline exactly: pinned version + hash gate before any bytes are extracted or executed + mirror-source support; artifact is a pinned wheel URL if one exists for GB10 (aarch64 + cu130 + torch 2.11), else a pinned sdist with documented build flags. The lane calls only the script, never bare pip. The pyproject extra remains the end-user route. Wheel availability is a research-phase fact-finding item; the mechanism stance is locked.
- **D-13: Neither extra joins any meta-extra** (`all`/`full` stay without deepspeed/flash-attn). Both are heavy compile packages; `optional_dependencies.md` documents them as separately installed. `all` naturally slims when its `fla` reference drops (D-09).

### Support Words & Docs Surface

- **D-14: Support words are `works with the deepspeed extra` / `works with the flash-attn extra`** — parallel in shape to the shim-required `works with built-in fixes` sentence. One vocabulary source shared across docs/TUI/MCP (27.2 D-16 discipline). The literal `pip install 'dnallm[deepspeed]'` command lives in docs detail text, never inside the short word (TUI/MCP column-width constraint, 27.2-05 tail-append layout).
- **D-15: `model_compatibility.md` regenerates with the flip; `DOCS_DATA_DATE` follows the evidence change.** Single-entry re-probe evidence suffices — no full 49-entry census re-run is required for a two-row change. Byte-identity regeneration gate keeps holding (quick-261029-c95 registry-data-change-drives-regen precedent).
- **D-16: `optional_dependencies.md` gets a full affected-surface rewrite:** a plain-language "what is an optional extra" intro paragraph; the summary table updated for four rows (PlantHelixSeek → built-in, BigBird/CrossDNA → extra install, GPN/Omni-DNA unchanged); BigBird and CrossDNA sections become practical install guides with the pip command + platform requirements (Linux + CUDA) + compile-time expectations. GPN/Omni-DNA sections stay verbatim (version-locked guidance cases).
- **D-17: Missing-extra errors are intercepted on the load chain** with one appended plain-language line naming the extra and the exact pip command (CrossDNA in its special handler; BigBird at the generic chain where the deepspeed `ImportError` is detected — exact interception layer is implementer discretion). Same shape as the GPN/Omni-DNA D-10 precedent from 27.2.

### Claude's Discretion

- Version bands for `deepspeed` and `flash-attn` in pyproject (research-measured against torch 2.11 / transformers 5.17)
- deepspeed lane install route — default: the plain `.[deepspeed]` extra via pip (regular PyPI wheel, band-pinned); escalate to a pinned script only if measurement shows a need
- Exact evidence-string format for the two flipped rows (follow the existing `hub repo@revision; cause` pattern)
- Exact seed mechanics for the two expected-failure rows in the drift gate
- Which pytest files host the new offline gates (existing census/compat test files preferred)
- The interception layer for BigBird's generic-chain deepspeed ImportError
- Whether the README supported-models statement needs an extras mention (only if it currently names the two models as broken)
- Final plan split and wave structure

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Campaign scope & inherited discipline

- `.planning/ROADMAP.md` §Phase 28 — the phase goal text and the owner option-1 ruling note
- `.planning/phases/27.2-transformers-5-compatibility-remediation-all-8-unsupported-p/27.2-CONTEXT.md` — D-02/D-03 (measured-first flips, honest wording), D-11/D-12 (BigBird deepspeed + CrossDNA flash-attn findings this phase upgrades), D-16 (support-word single-source discipline)
- `.planning/phases/27-transformers-4-5-model-compatibility-census/27-CONTEXT.md` — census-first discipline D-01..D-08 this phase inherits
- `.planning/PROJECT.md` §Key Decisions — the option-1 ruling row ("extras follow the mamba-extra precedent with measured-green-only flips")

### Extras & installer precedent

- `pyproject.toml` — `[project.optional-dependencies]` (the `fla` extra to promote/delete, the `mamba` extra + its comment block as the precedent shape), `[project] dependencies` (base target for fla), `[tool.uv.sources]`/`[tool.uv.index]`
- `scripts/install_mamba.sh` — the sha256-pinned installer discipline to mirror for flash-attn: hash gate before extraction/execution, `--verify-file`/`--patch-only` style flags, mirror-index support, pinned-constants header
- `.github/workflows/census_compat.yml` — the census lane that stays untouched; CR-01 seeding discipline reference for D-02
- `.github/workflows/ci.yml` — nightly lane-selection booleans (`run_coverage_nightly`/`run_example_nightly`/`run_test_mamba`) where `run_extras_compat` lands (D-06)

### Registry & census tooling

- `dnallm/models/model_info.yaml` — the three touched entries: GENA-LM-BigBird (~line 291), CrossDNA (~line 480), PlantHelixSeek (~line 449); `tf_compat_status`/`tf_compat_evidence`/`tf_compat_tested` fields flip per D-01..D-04
- `dnallm/models/registry.py` — `RegistryEntry` typed validation where the new enum value registers (extra="forbid", insertion-only invariant)
- `scripts/census_compat.py` — single-entry re-probe mode (D-07); the update-capable registry writer with green-drop rule (27.2-01)
- `tests/models/compat_census.json` — the committed 49-record census artifact; two rows become expected-failure seeded (D-02)
- `tests/models/test_census_compat.py`, `tests/models/test_compat_docs_consistency.py`, `tests/models/test_model_info_registry.py` — evidence pins / docs drift gate / registry integrity pins that update with the flips
- `models.lock` — `<hf|ms> <repo-id>@<revision>` pinning pattern for re-probe evidence

### Support surface

- `dnallm/models/support.py` — `SUPPORT_WORD_BY_REPO` overrides (BigBird's `needs deepspeed` → D-14 word; CrossDNA override added) and `DOCS_STATUS_CELLS` shared source (census generator reads it lazily — WR-02 loader guard)
- `docs/user_guide/optional_dependencies.md` — the rewrite target (D-16); currently documents all four external-package models in 27.2 wording
- `docs/user_guide/model_compatibility.md` — the generated page; `DOCS_DATA_DATE` bump per D-15
- `dnallm/models/special/crossdna.py` — the CrossDNA handler interception point (D-17); `_PREFERRED_CHECKPOINT_DIRS` already consulted by the census tool
- `dnallm/models/model.py` — the generic load chain; BigBird's deepspeed `ImportError` interception candidate (D-17)
- `dnallm/preflight/report.py` (`ListingEntry.support`) + `dnallm/tui/screens/suitability.py` + `dnallm/mcp/server.py` — the three surfaces that consume support words (27.2-05 additive tail-append pattern)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `scripts/install_mamba.sh` — the full pinned-installer skeleton (constants header, hash gate, mirror support, verification subcommands) to fork for `install_flash_attn.sh`
- `census_compat.py` single-entry mode + update-capable registry writer — the exact machinery 27.2 used for disposition flips; extras probes are the same flow in a different env
- `support.py` word-derivation + `DOCS_STATUS_CELLS` — the single-source vocabulary table gains two words (or word variants); all three display surfaces consume it already
- The nightly lane-selection boolean block in `ci.yml` — the D-06 pattern copies an existing row

### Established Patterns

- Measured-first flips with replica-grade interim evidence (27.2 D-02; G-27-1's 1:1 runner-replica repro discipline)
- sha256 hash gate BEFORE any downloaded byte is extracted or executed (install_mamba.sh supply-chain stance)
- Honest verdict wording: green claims always name what made them green (shim-required → "works with built-in fixes"; extra-required → "works with the X extra")
- Fail-closed gates re-derived independently from committed claims (docs-consistency gate; 27.1-01)
- Plain-language prose for biologists (owner directive 2026-10-21) governs every new user-visible word

### Integration Points

- `pyproject.toml` base deps + extras + `all` (D-09/D-10/D-11/D-13)
- `dnallm/models/registry.py` enum validation (D-01)
- `.github/workflows/extras_compat.yml` (new) + `ci.yml` boolean (D-05/D-06)
- `census_compat.py` probe invocations in the extras env; `compat_census.json` seeding (D-02/D-07/D-08)
- `support.py` word table; `optional_dependencies.md`; `model_compatibility.md` generator (D-14/D-15/D-16)
- `special/crossdna.py` + `models/model.py` load chain (D-17)

</code_context>

<specifics>
## Specific Ideas

- Owner's own words on the scope addition: "PlantHelixSeek 必须默认支持，不能通过外挂支持，必须直接进 dnallm" — lab models are first-class; no extra-install step may stand between a default install and `zhangtaolab/*` models
- The support-word phrasing was chosen deliberately to parallel the existing honest-green sentence family: `works with the deepspeed extra` sits next to `works with built-in fixes` — same grammar, same honesty structure, different enabler
- BigBird is the "double dependency" case (rung + extra); CrossDNA is the "pure extra" case (fla now base + flash-attn) — the two rows exercise both shapes of the new status value

</specifics>

<deferred>
## Deferred Ideas

- **GPN / Omni-DNA as extras** — permanently out (version-locked packages: gpn pins `transformers==5.15.0`, ai2-olmo pins `numpy<2`; they can never enter the main env). Roadmap-locked guidance cases; `optional_dependencies.md` keeps their verbatim guidance sections.
- **lgq12697 bulk ingestion (~130 unregistered ModelScope entries)** — separate v1.5 campaign (carried from 27.2).
- **Live dispatch of census + extras lanes** — both stay post-v1.4→main-merge operator steps (GitHub registers dispatchable workflows from the default branch only); replica evidence is the interim proof for both.
- **uv index pinning for flash-attn wheels** — rejected mechanism (would need new hosting infrastructure); revisit only if a future need emerges.

</deferred>

---

*Phase: 28-Optional extras: deepspeed & flash-attn model support*
*Context gathered: 2026-10-29*
