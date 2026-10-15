# Cross-Installer Install Honesty Matrix

What each dnallm install command actually delivers under each installer —
command × installer → the honest outcome. This page exists because the two
installers read different parts of `pyproject.toml`: the CUDA wheel-index
pins live in `[tool.uv.sources]`, which only uv respects (see the
[lane asymmetry note](installation.md#two-install-lanes-uv-and-pip-the-tooluvsources-asymmetry)).

## The matrix

| Command | uv (`uv pip install`) | plain `pip` | Provenance |
|---|---|---|---|
| `dnallm[full]` | the full dev aggregate: `dev` + `test` + `notebook` + `mcp` + isort + types-transformers | identical set — no index machinery involved | live probe 2026-10-14 |
| `dnallm[base]` — during the v1.3 alias cycle | resolves the alias → the **full member set**, exit 0, **no warning** | identical — full set, exit 0, **no warning** | live probe 2026-10-14 |
| `dnallm[base]` — after the v1.4 alias removal | `warning: The package ... does not have an extra named 'base'`; installs **core only**; exit 0 | `WARNING: dnallm X.Y.Z does not provide the extra 'base'`; installs **core only**; exit 0 | live probe 2026-10-14 (missing-extra case) |
| `dnallm[cuda130]` | torch from the pinned `torch-cuda130` index (`[tool.uv.sources]` respected) | torch from default PyPI — Linux: the cu130-bundled default build; Windows: **CPU-only** | uv docs (sources are uv-only); v1.3 stack research |
| `dnallm[fla]` on Windows | pulls `triton-windows` via the PEP 508 `platform_system == 'Windows'` marker (declared v1.3) | identical — the marker is standard wheel metadata honored by both | PEP 508 marker semantics; Windows execution proof at 3.7.1.post27 (WINDOWS ledger #18, RTX 5080) |

## Reading the rows honestly

- **No installer warns on an extra that exists.** During the v1.3 alias
  cycle the deprecated `base` spelling installs the full set silently and
  correctly under both installers — that is exactly why the deprecation's
  loud surface is this page and the v1.4 removal notes in the README and
  [Installation](installation.md) pages, not an installer warning (there is
  no PEP/PyPA extra-deprecation metadata channel; PEP 685 only normalizes
  names).
- **The warn-and-exit-0 rows are the POST-removal state** — pre-explained
  here so the v1.4 removal surprises nobody: once the alias is dropped, both
  installers print a one-line warning, exit 0, and leave a **core-only
  under-install** (the silent hazard the alias cycle exists to prevent).
- **The pip lane composes with the torch-LAST rule**: for GPU torch under
  plain pip, install dnallm first and the GPU torch build **last**, then
  verify the `torch.version.cuda` suffix — see
  [GPU Support with plain pip](installation.md#gpu-support-with-plain-pip).
- **triton-windows is a community fork**, published under the triton-lang
  org (github.com/triton-lang/triton-windows) with reproducible, attested
  builds since 3.7.1.post27; upstream triton ships no win_amd64 wheels. On
  Linux the marker evaluates false and nothing is pulled.

## Probe provenance

Rows 1–3: live-probed 2026-10-14 with pip 26.2.1 and uv 0.13.1 in clean
seeded venvs, installing a synthetic wheel carrying the exact alias shape
(`base = ["pkg[full]"]`) from a local wheel path; the full behavior matrix
(alias, missing-extra, empty-extra) is recorded in the phase-18 research.
Row 4: the uv documentation statement "Sources are only respected by uv"
(v1.3 stack research, fetched from docs.astral.sh). Row 5: the v1.3
pyproject declaration plus PEP 508 marker semantics; the Windows pull was
empirically executed at triton-windows 3.7.1.post27 on an RTX 5080
(WINDOWS ledger #18).
