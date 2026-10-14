# GPU Support Matrix

Which NVIDIA GPU generation is served by which CUDA extra — published from
the pre-verified boundary record (v1.2 close, 2026-10-11), adjusted for the
v1.3 cu121 lane retirement. **No row is re-derived from vendor compatibility
charts**: every boundary below traces to the verified record and its local
verification basis (torch 2.11.0+cu130 on the verification box; arch list
sm_80 / sm_90 / sm_100 / sm_110 / sm_120 — no sm_75, no PTX fallback).

## The matrix

| GPU generation | Example chips | Served by | Notes |
|---|---|---|---|
| Pascal (GTX 10xx) | GTX 1080 Ti, 1070 | **cpu-only — no GPU extra** | torch wheels dropped sm_61 after cu118: GPU acceleration never existed in dnallm's extras (owner decision 2026-10-11: no 1080 support, follow the upstream envelope). **The minimum GPU line is Turing (20xx).** |
| Turing (RTX/GTX 20xx) | RTX 2080 Ti, RTX 2060 | `cuda124` – `cuda128` | sm_75 IS in cu128 — torch 2.11 dropped only Volta sm_70 (source: pytorch#172352) — but NOT cu130, whose arch list starts at sm_80 with no PTX fallback. The retired cu121 lane (v1.3; index frozen at torch 2.5.1) once served this generation too; the recommended lane now is `cuda128`. |
| Ampere (RTX 30xx) | RTX 3090, 3060 | every hardware extra (`cpu`, `cuda124`–`cuda130`) | cu130 works via sm_80 same-major binary compatibility (sm_80 SASS serves sm_86/89). |
| Ada (RTX 40xx) | RTX 4090, 4070 | every hardware extra | same sm_80-major binary compatibility as Ampere. |
| Blackwell (RTX 50xx) | RTX 5090, 5080 | `cuda128` + `cuda130` | 50xx needs cuda128 or newer. |

**The default-PyPI torch build on Linux bundles CUDA 13.0** and therefore
serves Ampere (30xx) and newer — that is the "cuda130 default = Ampere+"
boundary. On Windows the default PyPI torch wheel is CPU-only; the CUDA
extras' index pins are uv-only (see the
[lane asymmetry note](installation.md#two-install-lanes-uv-and-pip--the-tooluvsources-asymmetry)).

## Forward notes

- cu128 wheels are deprecated from torch 2.12, so the 20xx (Turing) ceiling
  will freeze gradually as dnallm's torch floor rises.
- Driver too old for the newest lanes? A driver-based chooser exists across
  `cuda124`–`cuda130`: pick the newest lane your NVIDIA driver supports
  (cu130 needs driver >= 580).

## Verification provenance

- Boundaries verified 2026-10-11 (v1.2 close; the pre-verified support-matrix
  record recovered verbatim from commit 0542a6b) — local basis: torch
  2.11.0+cu130, arch list [sm_80, sm_90, sm_100, sm_110, sm_120], no sm_75,
  no PTX fallback.
- v1.3 adjustment (2026-10-14): the 20xx row narrows from the original
  cu121–cu128 span of the record to **cuda124–cuda128**, because the cu121
  lane retired in v1.3 (its wheel index is frozen at torch 2.5.1,
  re-verified 2026-10-14).

## Related pages

- [Installation](installation.md) — the uv/pip lane split, the extras
  tables, and the deprecated `base` alias note.
- [Cross-Installer Install Honesty Matrix](install_honesty.md) — what each
  command × installer combination actually delivers.
