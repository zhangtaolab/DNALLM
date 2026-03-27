gpn_models = [
    "gpn-brassicales",
    "gpn-animal-promoter",
    "gpn-msa-sapiens",
    "PhyloGPN",
    "gpn-brassicales-gxa-sorghum-v1",
]


def _handle_gpn_models(
    model_name: str, extra: str | None = None
) -> str | None:
    """Handle special case for GPN models."""

    if extra:
        gpn_models.append(extra)

    for m in gpn_models:
        if m in model_name:
            try:
                import gpn.model

                _ = gpn.model

            except ImportError as e:
                raise ImportError(
                    f"gpn package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/songlab-cal/gpn"
                ) from e
            return m

    return None
