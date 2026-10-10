"""Interpretation utilities extending model inference with motif analysis.

This subpackage exposes the FIMO-convention motif scanner and the JASPAR REST
client (:mod:`dnallm.interpret.motifs`). It is intentionally NOT re-exported
from ``dnallm/__init__.py`` (the root facade stays byte-stable); import it as
``from dnallm.interpret import motifs`` or pull the public names from here.
"""

from .motifs import (
    Motif,
    ScanResult,
    gc_background,
    parse_cisbp,
    parse_meme,
    scan,
    scan_single_strand,
)

__all__ = [
    "Motif",
    "ScanResult",
    "gc_background",
    "parse_cisbp",
    "parse_meme",
    "scan",
    "scan_single_strand",
]
