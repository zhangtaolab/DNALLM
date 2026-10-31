"""Shared plain-language support vocabulary for registry compatibility verdicts.

One registry-derived word per entry: the docs generator, the TUI listing,
and the MCP listing must show the same support vocabulary (D-16), so users
can tell whether a model works, works through DNALLM's repairs, needs an
external package, or remains untested — without reading raw registry enums
or technical tracebacks. Finetuned entries inherit their base model's
status: the Phase 27 census measures pretrained repos only, and a
fine-tuned checkpoint runs on its base's code.

The module is synchronous and deterministic. It performs no network access
and never parses the packaged registry itself — base-model resolution goes
through :mod:`dnallm.models.registry`, the sole parser — and the derived
word is presentation-only: it is never persisted back to the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import registry

if TYPE_CHECKING:  # import-only type (never imported at runtime)
    from .registry import RegistryEntry

__all__ = [
    "DOCS_STATUS_CELLS",
    "SUPPORT_WORDS",
    "SUPPORT_WORD_BY_REPO",
    "support_word",
]

#: The plain word for an entry the support vocabulary cannot classify: a
#: missing or unknown status, an unresolvable base model, or an inheritance
#: cycle. Degradation, never an error (listings must render every entry).
UNTESTED_WORD = "untested"

#: Plain-language word for each census ``tf_compat_status`` value (D-16).
SUPPORT_WORDS: dict[str, str] = {
    "verified-tf5": "works",
    "import-verified-tf5": "family-tested",
    "shim-required-tf5": "works with built-in fixes",
    # 28-02 D-01/D-14: measured green only after installing a curated
    # optional extra (the repo override names the exact extra).
    "extra-required-tf5": "works with an optional extra",
    "remote-code-broken": "not working with transformers 5",
    # The census could not reach or resolve the repo, so the honest word is
    # the same as an unmeasured entry: no verdict was recorded either way.
    "census-blocked": "untested",
}

#: Exact external-package phrases for the four repos whose situation a
#: generic status word cannot express (D-10, D-11, D-14): the words a listing
#: must show instead of the generic label, never claiming DNALLM ships the
#: package. BigBird keeps its need-word (28-02 owner ruling A: the deepspeed
#: extra greens the load, but the forward stays structurally dead - a
#: "works with the deepspeed extra" word would overclaim); CrossDNA's
#: measured green run names its exact extra.
SUPPORT_WORD_BY_REPO: dict[str, str] = {
    "songlab/gpn-brassicales": "needs pip install gpn",
    "zehui127/Omni-DNA-116M": "needs pip install ai2-olmo",
    "AIRI-Institute/gena-lm-bigbird-base-sparse": "needs deepspeed",
    "chengCCC/CrossDNA_pretrain": "works with the flash-attn extra",
}

#: The four generated-page status strings (D-13): the canonical wording of
#: ``docs/user_guide/model_compatibility.md``, shared with the census
#: generator so the byte-identical regeneration gate and the independent
#: docs gate can both hold it to a single source.
DOCS_STATUS_CELLS: dict[str, str] = {
    "verified-tf5": "Tested working",
    "import-verified-tf5": "Tested loading its code; full run inferred from the family model",
    "shim-required-tf5": "Works with DNALLM's built-in compatibility fixes (tested)",
    "extra-required-tf5": "Works after installing an optional extra (tested)",
}


def _resolve_base_entry(entry: RegistryEntry) -> RegistryEntry | None:
    """Follow ``base_model`` to the first ancestor carrying a census verdict.

    Cycle-safe via a repo-id seen-set and never raises: an unresolvable
    base or an inheritance cycle returns ``None``, which callers map to the
    untested word.

    Args:
        entry: The starting registry entry (pretrained or finetuned).

    Returns:
        The entry whose status applies (``entry`` itself when it carries a
        verdict or has no base to follow), or ``None`` when the chain is
        broken or cyclic.
    """
    seen = {entry.model}
    current = entry
    while current.tf_compat_status is None and current.base_model is not None:
        base_repo = current.base_model
        if base_repo in seen:
            return None
        seen.add(base_repo)
        try:
            current = registry.resolve(base_repo)
        except ValueError:
            return None
    return current


def support_word(entry: RegistryEntry) -> str:
    """Return the plain-language support word for one registry entry.

    A finetuned entry inherits its base model's support word: the base
    chain is followed until an ancestor carries a census verdict. The exact
    external-package override for the resolved repo wins over the general
    status mapping. A missing verdict, unknown status, unresolvable base,
    or inheritance cycle all map to ``"untested"`` — never an error, so
    every listing can render every entry.

    Args:
        entry: The registry entry (pretrained or finetuned).

    Returns:
        One of the shared support words (D-16), or one of the exact
        external-package phrases for the four override repos.
    """
    resolved = _resolve_base_entry(entry)
    if resolved is None:
        return UNTESTED_WORD
    override = SUPPORT_WORD_BY_REPO.get(resolved.model)
    if override is not None:
        return override
    return SUPPORT_WORDS.get(resolved.tf_compat_status, UNTESTED_WORD)
