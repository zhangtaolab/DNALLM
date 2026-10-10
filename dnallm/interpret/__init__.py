"""Interpretation utilities extending model inference with motif analysis.

This subpackage currently exposes the FIMO-convention motif scanner core
(:mod:`dnallm.interpret.motifs`). It is intentionally NOT re-exported from
``dnallm/__init__.py`` (the root facade stays byte-stable); import it as
``from dnallm.interpret import motifs`` or pull the public names from here.
"""

from .motifs import (
    Motif,
    parse_meme,
    scan_single_strand,
)

__all__ = [
    "Motif",
    "parse_meme",
    "scan_single_strand",
]
