"""
Genomic coordinate and chromosome-name normalization helpers.

The single place where the three conventions meeting in every genomics data
path are converted: 0-based half-open (BED) <-> 1-based closed (GFF3/FASTA)
coordinates, and TAIR ("Chr1") <-> Ensembl ("1") chromosome names. This
module provides functions for:

- Normalizing chromosome names between TAIR and Ensembl styles
- Converting between GFF3 1-based closed and BED 0-based half-open coordinates
- Fetching 1-based inclusive sequence slices from an indexed FASTA (pyfastx)
- Parsing GFF3 column-9 attributes (trailing ';' and comma-joined values)
- Filtering GFF3 rows down to a genomic locus

Every function validates its input and raises ValueError on anything outside
the documented forms: the characteristic failure mode of hand-scattered
conversion code is silence (empty results, off-by-one bases), never a loud
error, so silent-empty results are structurally impossible here.
"""

import contextlib
import re
from pathlib import Path

# Documented chromosome-name forms: a 'chr' prefix (any case) followed by a
# numeric token or a TAIR organelle letter (C/M), or a bare numeric Ensembl
# name. Anything else — e.g. 'chromosome1' — raises rather than being renamed.
_CHROM_RE = re.compile(r"^[cC][hH][rR]([0-9]+|[CM])$")


def _validate_gff1(start: int, end: int) -> tuple[int, int]:
    """Validate a pair of GFF3 1-based closed coordinates."""
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"GFF3 coordinates must be integers, got {start!r}, {end!r}.")
    if start < 1:
        raise ValueError(f"GFF3 coordinates are 1-based: start {start} must be >= 1.")
    if end < start:
        raise ValueError(f"Invalid GFF3 coordinates: start {start} > end {end}.")
    return start, end


def normalize_chrom(name: str, *, style: str = "tair") -> str:
    """
    Convert a chromosome name to the requested naming style.

    Args:
        name: Chromosome name. Accepted forms: TAIR-style ``Chr1`` (the
            ``chr`` prefix is case-insensitive; the token is a number or
            the organelle letter ``C``/``M``, preserved verbatim) and bare
            Ensembl-style numerics (``1``; ASCII digits only — lookalike
            Unicode digits raise). The organelle names
            ``ChrC``/``ChrM`` pass through untouched in both styles.
        style: Target style: ``"tair"`` (default, e.g. ``Chr1``) or
            ``"ensembl"`` (e.g. ``1``).

    Returns:
        str: The chromosome name in the requested style.

    Raises:
        ValueError: If the name is empty or outside the documented forms
            (never renamed heuristically), or the style is unknown.
    """
    if style not in ("tair", "ensembl"):
        raise ValueError(f"Unknown chromosome style {style!r}: expected 'tair' or 'ensembl'.")
    if not isinstance(name, str) or not name:
        raise ValueError("Chromosome name must be a non-empty string.")
    match = _CHROM_RE.match(name)
    if match is not None:
        token = match.group(1)
        if style == "ensembl" and token.isdigit():
            return token
        return f"Chr{token}"  # TAIR target, or an organelle pass-through
    if name.isascii() and name.isdigit():
        return name if style == "ensembl" else f"Chr{name}"
    raise ValueError(f"Unrecognized chromosome name {name!r}.")


def gff1_to_half_open(start: int, end: int) -> tuple[int, int]:
    """
    Convert GFF3 1-based closed coordinates to BED 0-based half-open.

    Args:
        start: 1-based inclusive start (>= 1). A length-1 feature has
            ``start == end`` and maps to a half-open interval of width 1.
        end: 1-based inclusive end (>= start).

    Returns:
        tuple[int, int]: ``(start - 1, end)`` — exactly-touching features
        map to adjacent, non-overlapping intervals.

    Raises:
        ValueError: If the coordinates are not integers, ``start < 1``, or
            ``start > end``.
    """
    start, end = _validate_gff1(start, end)
    return start - 1, end


def half_open_to_gff1(start0: int, end0: int) -> tuple[int, int]:
    """
    Convert BED 0-based half-open coordinates to GFF3 1-based closed.

    Args:
        start0: 0-based start (>= 0).
        end0: half-open end (> start0).

    Returns:
        tuple[int, int]: ``(start0 + 1, end0)`` — exact inverse of
        :func:`gff1_to_half_open`.

    Raises:
        ValueError: If the coordinates are not integers, ``start0 < 0``, or
            the interval is empty (``end0 <= start0``): a zero-width
            half-open interval has no 1-based closed representation.
    """
    for value in (start0, end0):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"BED coordinates must be integers, got {start0!r}, {end0!r}.")
    if start0 < 0:
        raise ValueError(f"BED coordinates are 0-based: start {start0} must be >= 0.")
    if end0 <= start0:
        raise ValueError(
            f"Half-open interval ({start0}, {end0}) is empty; 1-based closed "
            "intervals cannot represent zero width."
        )
    return start0 + 1, end0


def fetch_sequence(fa, chrom: str, start: int, end: int, *, uppercase: bool = True) -> str:
    """
    Fetch a 1-based inclusive sequence slice from an indexed FASTA.

    Args:
        fa: An open ``pyfastx.Fasta`` index, or a path to a FASTA file
            (opened lazily — pyfastx is a dev extra, imported only inside
            this function so importing the module never requires it).
            The path branch is read-only: it releases the index and removes
            a ``.fxi`` sidecar it created, preserves a pre-existing one,
            and never closes a caller-owned index.
        chrom: Chromosome name in any documented style; normalized (TAIR)
            before the lookup.
        start: 1-based inclusive start (GFF3 convention).
        end: 1-based inclusive end (>= start).
        uppercase: When True (default), upper-case the returned sequence.

    Returns:
        str: The sequence bases spanning ``[start, end]``.

    Raises:
        ValueError: If the coordinates are invalid, the chromosome name is
            unknown, or the fetch returns an empty result — a silent empty
            return is structurally impossible.
    """
    start, end = _validate_gff1(start, end)
    chrom = normalize_chrom(chrom)
    from_path = isinstance(fa, (str, Path))
    created_index: Path | None = None
    if from_path:
        import pyfastx

        # pyfastx builds (or reuses) a <fasta>.fxi index beside the file; one
        # that already exists belongs to the caller and is never removed.
        sidecar = Path(f"{fa}.fxi")
        if not sidecar.exists():
            created_index = sidecar
        fa = pyfastx.Fasta(fa)
    try:
        seq = fa.fetch(chrom, (start, end))
    except (KeyError, NameError) as e:  # pyfastx signals unknown names with NameError
        raise ValueError(f"Unknown chromosome {chrom!r} in FASTA index.") from e
    finally:
        if from_path:
            # pyfastx.Fasta has no close()/context manager: dropping the last
            # reference releases its file descriptors, deterministically and
            # also on the error path (where a traceback would keep the frame
            # alive). Cleanup must never mask the fetch result or the
            # documented ValueError, hence the suppressed unlink.
            fa = None
            if created_index is not None:
                with contextlib.suppress(OSError):
                    created_index.unlink()
    if not seq:
        raise ValueError(f"Empty sequence fetched for {chrom!r} [{start}, {end}].")
    return seq.upper() if uppercase else seq


def parse_gff_attributes(attrs: str) -> dict[str, list[str]]:
    """
    Parse a GFF3 column-9 attribute string into an insertion-ordered mapping.

    Tolerates the trailing ``;`` carried by 197,160 TAIR10 CDS rows and
    splits comma-joined multi-values (e.g. double Parents) into lists.

    Args:
        attrs: Raw column-9 text, e.g. ``"ID=x;Parent=a,b;"``.

    Returns:
        dict[str, list[str]]: Attribute name -> list of values, in source
        order. An empty or ``.`` column yields an empty dict.

    Raises:
        ValueError: If a non-empty segment carries no ``=``.
    """
    parsed: dict[str, list[str]] = {}
    if not attrs or attrs.strip() == ".":
        return parsed
    for segment in attrs.split(";"):
        segment = segment.strip()
        if not segment:
            continue  # the trailing ';' produces an empty tail segment
        if "=" not in segment:
            raise ValueError(f"Malformed GFF3 attribute (missing '='): {segment!r}")
        key, _, value = segment.partition("=")
        parsed[key.strip()] = [item.strip() for item in value.split(",")]
    return parsed


def slice_gff_rows(
    rows, chrom: str, start: int, end: int, *, require_nonempty: bool = False
) -> list[str]:
    """
    Filter GFF3 rows to a 1-based closed locus, preserving input order.

    Args:
        rows: Iterable of raw GFF3/GFF line strings. Comment (``#``), blank,
            and non-string entries are skipped; remaining rows need at
            least 5 tab-separated columns. Line terminators (``\\n``,
            ``\\r\\n`` CRLF, and old-Mac ``\\r``) are stripped from each row
            before parsing and from the returned strings.
        chrom: Chromosome name in any documented style; both it and each
            row's name are normalized before exact comparison.
        start: 1-based inclusive locus start.
        end: 1-based inclusive locus end (>= start).
        require_nonempty: When True, raise instead of returning an empty
            result (the silent-empty guard for locus slicing).

    Returns:
        list[str]: The matching raw row strings, terminator-stripped, input
            order preserved, never re-sorted.

    Raises:
        ValueError: If the coordinates are invalid, a row is malformed, a
            row carries a carriage return that is not a line terminator
            (an embedded ``\\r`` would silently corrupt whichever column
            contains it), or ``require_nonempty`` is set and no row matches.
    """
    start, end = _validate_gff1(start, end)
    chrom = normalize_chrom(chrom)
    matched: list[str] = []
    for row in rows:
        if not isinstance(row, str) or not row.strip() or row.startswith("#"):
            continue
        stripped = row.rstrip("\r\n")
        if "\r" in stripped:
            raise ValueError(
                f"Carriage return embedded in GFF3 row (not a line terminator): {row!r}"
            )
        columns = stripped.split("\t")
        if len(columns) < 5:
            raise ValueError(f"Malformed GFF3 row (fewer than 5 columns): {row!r}")
        if normalize_chrom(columns[0]) != chrom:
            continue
        row_start, row_end = int(columns[3]), int(columns[4])
        if row_start <= end and row_end >= start:  # closed-interval overlap
            matched.append(stripped)
    if require_nonempty and not matched:
        raise ValueError(f"No GFF3 rows found for {chrom} [{start}, {end}].")
    return matched
