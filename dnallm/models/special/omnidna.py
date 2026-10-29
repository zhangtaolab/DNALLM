omnidna_models = [
    "Omni-DNA-20M",
    "Omni-DNA-60M",
    "Omni-DNA-116M",
    "Omni-DNA-300M",
    "Omni-DNA-700M",
    "Omni-DNA-1B",
]


def _handle_omnidna_models(model_name: str, extra: str | None = None) -> str | None:
    """Handle special case for Omni-DNA models."""

    if extra:
        omnidna_models.append(extra)

    for m in omnidna_models:
        if m in model_name:
            try:
                from olmo import version  # ty: ignore[unresolved-import]  # optional dep, guarded

                _ = version.VERSION

            except ImportError as e:
                raise ImportError(
                    f"ai2-olmo package is required for "
                    f"{model_name} but not installed. "
                    "Install it with: pip install ai2-olmo "
                    "(measured caveat, ai2-olmo 0.6.0: pins numpy<2 and can "
                    "downgrade a numpy 2 environment - use a separate "
                    "environment). See "
                    "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/"
                ) from e
            return m

    return None
