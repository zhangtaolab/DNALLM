# Optional Dependencies and Support

DNALLM loads DNA language models through a widely used AI library called
*transformers* — the software that downloads a model's files and runs it.
Every preset model that ships with DNALLM was tested against transformers 5,
and the per-model results live in the generated
[Model Compatibility](model_compatibility.md) table. This page explains,
in plain language, what each support word means, which models need an extra
software package before they can run, and what DNALLM does for the models it
repairs itself.

## What is an optional extra?

DNALLM's base install — what a plain `pip install dnallm` gives you —
already includes everything the large majority of models need, including
every package behind the lab's own models such as PlantHelixSeek. An
optional extra is something different: an add-on you explicitly ask for by
name at install time, because it is a heavy package that only a small
number of models use. Extras are never installed automatically, and DNALLM
never silently substitutes a stand-in for missing software — a fake
replacement would compute wrong numbers without telling you.

DNALLM currently offers two extras, and the sections below give their
measured install guides:

- `dnallm[deepspeed]` — the `deepspeed` package used by GENA-LM-BigBird.
- `dnallm[flash-attn]` — the compiled `flash-attn` package used by
  CrossDNA.

Both extras build compiled code on your machine and are supported on
Linux + CUDA computers (a Linux system with an NVIDIA GPU). GPN and
Omni-DNA are deliberately not extras: their packages pin other software to
exact versions and must live in separate environments (see their sections).

## What the support words mean

Listings and guides use one shared set of words:

| Support word | What it means |
| --- | --- |
| works | Loaded and ran a real prediction under transformers 5 during testing. |
| family-tested | The model's own program code loaded cleanly; a tested close relative in the same model family ran the full prediction. |
| works with built-in fixes | The model runs under transformers 5 only because DNALLM applies a repair described below — it does not work as-is. |
| works with an optional extra | The model runs only after you install one named optional package — the listing names the exact extra to install. |
| not working with transformers 5 | The model cannot be loaded as published; the reason is either its own code, a missing external package, or a change in the library. |
| untested | No verdict was recorded — the model could not be checked. |

Four models show an exact package phrase instead of the generic word —
`needs pip install gpn`, `needs pip install ai2-olmo`, `needs deepspeed`,
and `works with the flash-attn extra` — because for those models the
actionable answer is a specific package, and the sections below give the
measured details.

## GENERanno

**Support word: works with built-in fixes.**

The model's code looks up a standard position-encoding recipe named
`default` inside transformers' recipe table. Transformers 5 removed that
entry from the table, so the model's code crashes with a missing-key error
before any weights load. DNALLM restores the exact recipe (copied verbatim
from transformers 4.49, the version the model was written for) into the
table before the model loads. With that one entry restored, the model loads
and runs a real forward pass. The repair is applied automatically at load —
this model does not work as-is, only through the fix.

## DNABERT-2

**Support word: works with built-in fixes.**

For speed, transformers 5 assembles new models on a placeholder "meta"
device and moves them onto a real one afterwards. DNABERT-2's code builds
one small internal table (used for its ALiBi position weighting) without
naming a device, so under transformers 5 that table lands on the
placeholder while its other operand sits on the CPU — the two can never be
multiplied and loading crashes. DNALLM wraps exactly that one table-building
step so both pieces are created on the same real device, keeping the fast
low-memory construction path intact. The model then loads and runs; the
repair is automatic.

## GENA-LM

**Support word: works with built-in fixes.**

GENA-LM's code creates a small bookkeeping buffer (`token_type_ids`) filled
with zeros when the model class is built, and marks it *non-persistent* —
meaning it is never stored in the published files, which is normal and
correct. Under transformers 5, that buffer is incorrectly materialized from
the placeholder device holding garbage numbers instead of the zeros its own
code specifies, and the first prediction crashes on the impossible values.
DNALLM detects the garbage values after loading and resets the buffer to
zeros — exactly what the constructor intended. The published model files
are fine; nothing about the checkpoint is broken, and the repair runs
automatically on every transformers version where the defect is detected.

## MutBERT

**Support word: works with built-in fixes.**

The model authors' code contains a line (marked `# modified` by the authors
themselves) that computes the input's shape while dropping one dimension,
so every normal two-dimensional input — a batch of sequences — crashes with
a shape mismatch before the model can run. The same code contains a second
branch that computes the shape correctly. DNALLM detects the defective line
in the loaded code and routes two-dimensional inputs through that correct
branch. The repair is automatic and applies wherever the defect is present,
on any transformers version.

## GPN

**Support word: needs pip install gpn.**

GPN's published code expects an external Python package (`gpn`) that DNALLM
does not install. The basic command is:

```bash
pip install gpn
```

Measured caveats for the `gpn` 0.9.0 wheel (checked 2026-10-29, none of this
is installed by DNALLM):

- It requires **Python 3.13 only** (`>=3.13,<3.14`), so it will not install
  on the Python 3.11/3.12 environments most DNALLM setups use.
- It **pins `transformers==5.15.0`**, which would override your own
  transformers version — install it in a **separate environment**, not the
  one running DNALLM.
- Its wheel layout does **not provide the `gpn.model` module** the model's
  loader imports (only subpackage modules like `gpn/msa/model.py`). If you
  need that exact import, install from the source repository instead:
  `uv pip install git+https://github.com/songlab-cal/gpn.git`.

## Omni-DNA

**Support word: needs pip install ai2-olmo.**

Omni-DNA's configuration code imports from an external package (`ai2-olmo`)
that DNALLM does not install. The basic command is:

```bash
pip install ai2-olmo
```

Measured caveat for `ai2-olmo` 0.6.0 (checked 2026-10-29): it **pins
`numpy<2`**, so installing it into an existing environment would downgrade
your NumPy — use a **separate environment** for it.

## GENA-LM-BigBird

**Support word: needs deepspeed.**

This model's block-sparse attention — the memory-saving trick that lets
BigBird look at long sequences — loads its implementation classes from the
external `deepspeed` package by name, at runtime. DNALLM deliberately does
not install `deepspeed` automatically and deliberately does not stub it:
the classes carry out the attention math, so a stand-in
would silently compute the wrong block-sparse attention and hand you
incorrect predictions that look perfectly plausible.

The curated way to install it is the `deepspeed` extra:

```bash
pip install 'dnallm[deepspeed]'
```

The extra is supported on Linux + CUDA computers, and two measured facts
shape the install:

- **The version band is deliberate.** The extra pins deepspeed to the
  0.19.6 release only: the sparse-attention module this model needs was
  deleted from deepspeed itself in 0.19.7, so any newer version installs a
  package that no longer contains it.
- **The package itself needs no GPU-compiler build.** The sparse operation
  runs on Triton — the GPU compiler that ships alongside torch — so
  installing the extra is the whole step; there is no separate nvcc build
  for this operation.

What the extra buys, measured on a GPU replica of the census runner with
deepspeed 0.19.6 installed: **loading** works with the deepspeed extra. In
the default environment the model cannot even be built (the deepspeed
import is missing); with the extra, the checkpoint constructs — this also
relies on DNALLM's built-in `_patch_apply_chunking_to_forward` fix from
the GENA-LM family above, applied automatically at load. The model's
**forward pass** — actually running a prediction — remains blocked by an
upstream library death: deepspeed 0.19.6's sparse kernel needs a Triton
API (`libtriton.superblock`) that no Triton 3.x release ships, which is
the same rot that made deepspeed delete the module in 0.19.7. The
compatibility table records this split exactly: load works with the
deepspeed extra but forward is blocked by an upstream-dead Triton API.
If you need predictions rather than the checkpoint itself, pick a model
that works.

## CrossDNA

**Support word: works with the flash-attn extra.**

CrossDNA's attention layer is built on `flash-attn`, a separately compiled
attention package. Half of its software stack — flash-linear-attention
(`fla`) — already ships with every DNALLM install as a base dependency,
so nothing extra is needed for that half. The compiled `flash-attn`
package itself is the curated extra, installed with:

```bash
pip install --no-build-isolation 'dnallm[flash-attn]'
```

The `--no-build-isolation` flag is required, not optional: flash-attn's
build script imports torch while compiling, and an isolated build
environment contains no torch, so a plain install fails before any code is
compiled. Everything the build needs (torch, ninja, and friends) is
already part of a DNALLM install.

What to expect, measured:

- **Platform:** Linux + CUDA. The package compiles GPU code on your
  machine — no ready-made build exists for current CUDA and torch
  versions.
- **Time:** the install compiles flash-attn 2.8.3.post2 from source, and a
  full CUDA source build can take hours on a typical machine.
- **CI:** for its own GPU runner, DNALLM builds flash-attn through the
  pinned installer script `scripts/install_flash_attn.sh` (a
  verified-source build tailored to that machine); end users do not need
  the script — the pip command above is the supported route.

With the extra installed, the model was measured green end to end on the
GPU replica of the census runner: it loads and runs a real forward pass
(flash-attn 2.8.3.post2, evidence date 2026-10-29).

## PlantHelixSeek

**Support word: works.**

PlantHelixSeek — the lab's own model behind the CRE and gene-structure
annotation tutorials — needs the `flash-linear-attention` package for its
kernels. That package is now part of DNALLM's base install: every plain
`pip install dnallm` includes it, along with its Windows companion
package (a Triton build for Windows), so no extra-install step of any kind
remains. Earlier versions exposed it as a separate `fla` extra; that extra
has been retired, and upgrading to the current version installs the
package automatically.

The tutorial notebooks keep one safety check: if flash-linear-attention is
somehow missing, the notebook stops with a clear error instead of letting
the model silently fall back to a degraded path that produces positionally
dead outputs.

## External packages at a glance

| Model | Situation | What to do next |
| --- | --- | --- |
| PlantHelixSeek | built-in | Nothing — the package it needs ships with every DNALLM install. |
| GENA-LM-BigBird | needs deepspeed | `pip install 'dnallm[deepspeed]'` lets the checkpoint load; the forward pass stays blocked upstream (see its section). |
| CrossDNA | works with the flash-attn extra | `pip install --no-build-isolation 'dnallm[flash-attn]'` on Linux + CUDA (see its section). |
| GPN and Omni-DNA | unchanged | Install `gpn` / `ai2-olmo` in separate environments — both pin other packages to exact versions (see their sections). |

## The eight at a glance

The table below is the repair-and-history overview of the models
from the transformers-5 compatibility campaign; the external-package
summary above gives the current install story in four rows.

| Model | Support word | What to do next |
| --- | --- | --- |
| GENERanno | works with built-in fixes | Nothing — the repair runs automatically at load. |
| DNABERT-2 | works with built-in fixes | Nothing — the repair runs automatically at load. |
| GENA-LM | works with built-in fixes | Nothing — the repair runs automatically at load. |
| MutBERT | works with built-in fixes | Nothing — the repair runs automatically at load. |
| GPN | needs pip install gpn | Install `gpn` in a separate environment (Python 3.13, pins `transformers==5.15.0`); see the GPN section. |
| Omni-DNA | needs pip install ai2-olmo | Install `ai2-olmo` in a separate environment (pins `numpy<2`); see the Omni-DNA section. |
| GENA-LM-BigBird | needs deepspeed | Install the deepspeed extra (`pip install 'dnallm[deepspeed]'`) to load the checkpoint — the forward pass stays blocked by an upstream library change; see the section below. |
| CrossDNA | works with the flash-attn extra | Install with `pip install --no-build-isolation 'dnallm[flash-attn]'` (Linux + CUDA, compiled from source); see the section below. |

## Provenance

Every verdict and caveat above was measured, not inferred: the model
repairs were verified by loading each model and running a real forward pass
under **transformers 5.17.0** (torch 2.11.0), and the package facts come
from inspecting the published `gpn` 0.9.0 and `ai2-olmo` 0.6.0 wheels. The
BigBird and CrossDNA extras facts were measured in a separate **extras
replica** — an environment matching the census runner plus the two curated
extras (deepspeed 0.19.6; flash-attn 2.8.3.post2 compiled from its source
package, torch 2.14.1) — on **2026-10-29**. All probes are
**replica-grade**; the automated census and extras lanes re-confirm them
after the next release merge (neither lane has run live yet). Per-model
statuses live in the generated
[Model Compatibility](model_compatibility.md) table, which is rebuilt from
the model registry — the single source of truth.
