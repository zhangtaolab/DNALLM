# Models Troubleshooting

#### Mamba models on macOS (Apple Silicon)

`mamba-ssm` relies on [Triton](https://github.com/triton-lang/triton), which only provides pre-built wheels for Linux. As a result, the `[mamba]` extra cannot be installed on macOS.

**Workaround:** HuggingFace `transformers` includes a pure-PyTorch Mamba implementation (`MambaModel`) that works on macOS without `mamba-ssm`. To use it:

```bash
# Install without the [mamba] extra
uv pip install -e '.[base]'
```

Then load a Mamba model through `transformers` (cpu only) as usual. Note that this fallback path is significantly slower than the optimized `mamba-ssm` kernels (which require Linux + CUDA).
