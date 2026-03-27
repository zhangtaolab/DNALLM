omnidna_models = [
    "Omni-DNA-20M",
    "Omni-DNA-60M",
    "Omni-DNA-116M",
    "Omni-DNA-300M",
    "Omni-DNA-700M",
    "Omni-DNA-1B",
]


def _handle_omnidna_models(
    model_name: str, extra: str | None = None
) -> str | None:
    """Handle special case for Omni-DNA models."""

    if extra:
        omnidna_models.append(extra)

    for m in omnidna_models:
        if m in model_name:
            try:
                from olmo import version

                _ = version.VERSION

            except ImportError as e:
                raise ImportError(
                    f"ai2-olmo package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://huggingface.co/zehui127/Omni-DNA-20M"
                ) from e
            return m

    return None
