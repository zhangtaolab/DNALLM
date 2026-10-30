# Models Troubleshooting

#### PlantHelixSeek predictions are uniform / positionally meaningless

PlantHelixSeek-CRE/-Anno remote code imports `fla.ops.kda.chunk.chunk_kda` from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for its HelixSeekDelta (KDA) layers. **Without the package the models still load and run — there is no error —** but 9/39 layers silently fall back to a pure-PyTorch path that is not KDA math, and outputs become positionally uninformative (measured: Anno predicts all-O at 0.82–0.98 confidence everywhere; CRE gives p(CRE) ≈ 0.007 both inside and outside real DHS sites).

**Fix:** `flash-linear-attention` ships as a base dependency of DNALLM, so install or reinstall the standard package — there is no extra to install (the former `fla` extra has been retired, and upgrading installs the package automatically):

```bash
uv pip install -e .   # in a repository clone; wheel users: pip install --upgrade dnallm
```

With the kernels installed the same checkpoints separate cleanly (p(CRE) 0.77 in-DHS vs 0.22 non-DHS on the probe set). DNALLM installs the package **bare** — the `[cuda]`/`[rocm]` extras it shipped since v0.5 are legacy names no longer on the current install surface (they pin their own torch and would downgrade your environment) — and keeps the version within `0.5.x` (`chunk_kda` semantics are not stable across minor versions). For current GPU installation lanes see the [GPU Support Matrix](../getting_started/gpu_support.md).

#### Mamba models on macOS (Apple Silicon)

`mamba-ssm` relies on [Triton](https://github.com/triton-lang/triton), which only provides pre-built wheels for Linux. As a result, the `[mamba]` extra cannot be installed on macOS.

**Workaround:** HuggingFace `transformers` includes a pure-PyTorch Mamba implementation (`MambaForCausalLM`) that is the accepted official path on macOS without `mamba-ssm` (owner ruling 2026-10-24) — including its mambapy pscan gate, which the base `mambapy` dependency satisfies on every platform, so no checkpoint re-export is needed. To use it:

```bash
# Install without the [mamba] extra
uv pip install -e '.[full]'
```

The path runs on CPU and on MPS (Apple Silicon): the Predictor keeps device `mps` automatically and always forces fp32 (fp16 on MPS is unverified and stays conservatively off). Measured 2026-10-24 on torch 2.14.1, a 512-token forward runs 349 ms on MPS vs 2119 ms on CPU (~6.1x) with bitwise-identical outputs. MPS for this path needs torch >= 2.12, which is inside dnallm's declared torch range (`torch>=2.4.0,<2.15`). On older torch (<2.12), explicitly set `inference.device: cpu` in your config; automatic device detection does not check this version requirement. Note that this path is still significantly slower than the optimized `mamba-ssm` kernels (which require Linux + CUDA).
