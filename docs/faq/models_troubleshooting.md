# Models Troubleshooting

#### PlantHelixSeek predictions are uniform / positionally meaningless

PlantHelixSeek-CRE/-Anno remote code imports `fla.ops.kda.chunk.chunk_kda` from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for its HelixSeekDelta (KDA) layers. **Without the package the models still load and run — there is no error —** but 9/39 layers silently fall back to a pure-PyTorch path that is not KDA math, and outputs become positionally uninformative (measured: Anno predicts all-O at 0.82–0.98 confidence everywhere; CRE gives p(CRE) ≈ 0.007 both inside and outside real DHS sites).

**Fix:** install the `fla` extra (included in `all`):

```bash
uv pip install -e '.[fla]'
```

With the kernels installed the same checkpoints separate cleanly (p(CRE) 0.77 in-DHS vs 0.22 non-DHS on the probe set). Install the package **bare** — since v0.5 its `[cuda]`/`[rocm]` extras pin their own torch and would downgrade your environment — and keep the version within `0.5.x` (`chunk_kda` semantics are not stable across minor versions).

#### Mamba models on macOS (Apple Silicon)

`mamba-ssm` relies on [Triton](https://github.com/triton-lang/triton), which only provides pre-built wheels for Linux. As a result, the `[mamba]` extra cannot be installed on macOS.

**Workaround:** HuggingFace `transformers` includes a pure-PyTorch Mamba implementation (`MambaModel`) that works on macOS without `mamba-ssm`. To use it:

```bash
# Install without the [mamba] extra
uv pip install -e '.[base]'
```

Then load a Mamba model through `transformers` (cpu only) as usual. Note that this fallback path is significantly slower than the optimized `mamba-ssm` kernels (which require Linux + CUDA).
