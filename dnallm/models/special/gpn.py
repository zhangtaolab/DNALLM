gpn_models = [
    "gpn-brassicales",
    "gpn-animal-promoter",
    "gpn-msa-sapiens",
    "PhyloGPN",
    "gpn-brassicales-gxa-sorghum-v1",
]


def _handle_gpn_models(model_name: str, extra: str | None = None) -> str | None:
    """Handle special case for GPN models."""

    if extra:
        gpn_models.append(extra)

    for m in gpn_models:
        if m in model_name:
            try:
                import gpn.model  # ty: ignore[unresolved-import]  # optional dep, guarded

                _ = gpn.model

            except ImportError as e:
                raise ImportError(
                    f"gpn package is required for "
                    f"{model_name} but not installed. "
                    "Install it with: pip install gpn "
                    "(measured caveats, gpn 0.9.0 wheel: requires Python 3.13 "
                    "through 3.13.x only, pins transformers==5.15.0, and does not "
                    "provide the gpn.model module this handler imports - use a "
                    "separate environment). See "
                    "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/"
                ) from e
            return m

    return None
