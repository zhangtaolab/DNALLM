---
status: complete
task: fla-dependency
created: 2026-10-03T17:30:00+08:00
completed: 2026-10-03T17:35:00+08:00
commits: [2259573]
---

# quick 261003-fla — flash-linear-attention as a declared, documented dependency

**Trigger:** Owner decision upgrade during /gsd-execute-phase 06 (17:03 CST), on top of
checkpoint decision B+ (fla 0.5.2, 16:40 CST). The original B+ instruction recorded a
*deferred* pyproject follow-up; the owner upgraded it: the dependency must land in the docs
and in the dependency list now.

## What landed (2259573)

- `pyproject.toml`: new `fla` extra `flash-linear-attention>=0.5.2,<0.6` (bare-install
  contract documented in a comment: fla's own `[cuda]`/`[rocm]` backend extras pin their own
  torch since v0.5 and would downgrade environments); `fla` wired into `all`.
- `README.md`: `fla` row in Dependency Groups (combinable, not a hardware group);
  PlantHelixSeek-CRE/-Anno row in the special-models table; dedicated section carrying the
  silent-fallback warning with measured evidence (p(CRE) 0.0073 in-DHS vs 0.0087 non-DHS on
  the fallback; 0.7673/0.2230 with kernels) and the bare-install + 0.5.x-bound guidance.
- `docs/faq/models_troubleshooting.md`: "PlantHelixSeek predictions are uniform /
  positionally meaningless" entry — the exact symptom a user without fla sees.
- `tests/models/test_plant_helixseek_fla_kernels.py` (pytest 同船 per owner rule):
  3 guard tests — extra declared with bounded range; reachable from `all`;
  `fla.ops.kda.chunk.chunk_kda` import path stable when installed (typed
  `environment-unavailable:` importorskip skip when absent). 3 passed; registry+smoke
  5 passed; ruff check+format clean.

## Why the bound `>=0.5.2,<0.6`

fla 0.4→0.5 changed both install semantics (backend extras) and `chunk_kda`-adjacent
behavior; the 16-window probe validated exactly 0.5.2. Cross-minor stability is not
guaranteed by upstream, so the spec pins inside 0.5.x and the import guard fails loudly if a
future minor moves the module.

## Provenance

Root-cause evidence: 06-03 checkpoint falsification (`example/notebooks/plant_helixseek_shared/.scratch/fla-fallback-diagnosis.md`,
scratch by design) + 0.5.2 probe (`fla-probe-0.5.2.md`, same dir). Decision trail: STATE.md
06-03 decisions (B+ at 16:40, upgrade at 17:03 relayed to the executor at 17:05).
