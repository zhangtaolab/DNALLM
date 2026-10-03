"""RED-phase stub — implemented in the GREEN step of plan 06-02."""


def normalize_chrom(name: str, *, style: str = "tair") -> str:
    """Convert a chromosome name to the requested naming style."""
    raise NotImplementedError


def gff1_to_half_open(start: int, end: int) -> tuple[int, int]:
    """Convert GFF3 1-based closed coordinates to BED 0-based half-open."""
    raise NotImplementedError


def half_open_to_gff1(start0: int, end0: int) -> tuple[int, int]:
    """Convert BED 0-based half-open coordinates to GFF3 1-based closed."""
    raise NotImplementedError


def fetch_sequence(fa, chrom: str, start: int, end: int, *, uppercase: bool = True) -> str:
    """Fetch a 1-based inclusive sequence slice from an indexed FASTA."""
    raise NotImplementedError


def parse_gff_attributes(attrs: str) -> dict[str, list[str]]:
    """Parse a GFF3 column-9 attribute string into an ordered mapping."""
    raise NotImplementedError


def slice_gff_rows(
    rows, chrom: str, start: int, end: int, *, require_nonempty: bool = False
) -> list[str]:
    """Filter GFF3 rows to a 1-based closed locus, preserving input order."""
    raise NotImplementedError
