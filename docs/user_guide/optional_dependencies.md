# Optional Dependencies and Support

DNALLM loads DNA language models through a widely used AI library called
*transformers* — the software that downloads a model's files and runs it.
Every preset model that ships with DNALLM was tested against transformers 5,
and the per-model results live in the generated
[Model Compatibility](model_compatibility.md) table. This page explains,
in plain language, what each support word means, which models need an extra
software package before they can run, and what DNALLM does for the models it
repairs itself.

Nothing on this page is installed for you automatically: DNALLM deliberately
does not ship the `gpn`, `ai2-olmo`, or `deepspeed` packages, and it never
silently substitutes a stand-in for missing software — a fake replacement
would compute wrong numbers without telling you.

## What the support words mean

Listings and guides use one shared set of words:

| Support word | What it means |
| --- | --- |
| works | Loaded and ran a real prediction under transformers 5 during testing. |
| family-tested | The model's own program code loaded cleanly; a tested close relative in the same model family ran the full prediction. |
| works with built-in fixes | The model runs under transformers 5 only because DNALLM applies a repair described below — it does not work as-is. |
| not working with transformers 5 | The model cannot be loaded as published; the reason is either its own code, a missing external package, or a change in the library. |
| untested | No verdict was recorded — the model could not be checked. |

Three models show an exact package phrase instead of the generic broken
word — `needs pip install gpn`, `needs pip install ai2-olmo`, and
`needs deepspeed` — because for those models the actionable answer is a
specific package, and the sections below give the measured details.

## GENERanno

**Support word: works with built-in fixes.**

The model's code looks up a standard position-encoding recipe named
`default` inside transformers' recipe table. Transformers 5 removed that
entry from the table, so the model's code crashes with a missing-key error
before any weights load. DNALLM restores the exact recipe (copied verbatim
from transformers 4.49, the version the model was written for) into the
table before the model loads. With that one entry restored, the model loads
and runs a real forward pass — the repair is applied automatically, and you
never claim the model works as-is.

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
not install `deepspeed` and deliberately does not stub it: the classes carry
out the attention math, so a stand-in would silently compute the wrong
block-sparse attention and hand you incorrect predictions that look
perfectly plausible. If you need this model, install `deepspeed` yourself
(separately compiled; heavy) — otherwise pick a model that works.

## CrossDNA

**Support word: not working with transformers 5.**

The CrossDNA repository stores its checkpoints in size-named subdirectories
(`8.1M/`, `28.6M/`, …) with no configuration file at the repository root.
DNALLM's own loader, and now the compatibility census, resolve the same
checkpoint subdirectory (`8.1M/`, the handler's preferred checkpoint), so
the two measure through one route. The re-probe through that route measured
an honest verdict: the selected checkpoint's code constructs its attention
through the separately compiled `flash-attn` package, which DNALLM does not
install, so the model is recorded as not working with transformers 5 — a
measured result, not a guess. If you need it, `flash-attn` must be compiled
and installed separately; otherwise pick a model that works.

## The eight at a glance

| Model | Support word | What to do next |
| --- | --- | --- |
| GENERanno | works with built-in fixes | Nothing — the repair runs automatically at load. |
| DNABERT-2 | works with built-in fixes | Nothing — the repair runs automatically at load. |
| GENA-LM | works with built-in fixes | Nothing — the repair runs automatically at load. |
| MutBERT | works with built-in fixes | Nothing — the repair runs automatically at load. |
| GPN | needs pip install gpn | Install `gpn` in a separate environment (Python 3.13, pins `transformers==5.15.0`); see the GPN section. |
| Omni-DNA | needs pip install ai2-olmo | Install `ai2-olmo` in a separate environment (pins `numpy<2`); see the Omni-DNA section. |
| GENA-LM-BigBird | needs deepspeed | Install `deepspeed` yourself or pick another model; it cannot be safely stubbed. |
| CrossDNA | not working with transformers 5 | Needs the separately compiled `flash-attn` package; otherwise pick another model. |

## Provenance

Every verdict and caveat above was measured, not inferred: the model
repairs were verified by loading each model and running a real forward pass
under **transformers 5.17.0** (torch 2.11.0), and the package facts come
from inspecting the published `gpn` 0.9.0 and `ai2-olmo` 0.6.0 wheels. The
probes are **replica-grade** (re-run on a machine matching the census
runner on **2026-10-29**); the automated census lane re-confirms them after
the next release merge. Per-model statuses live in the generated
[Model Compatibility](model_compatibility.md) table, which is rebuilt from
the model registry — the single source of truth.
