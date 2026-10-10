"""FIMO-convention motif scanning against JASPAR and CIS-BP PWMs.

This module provides:
- Strict MEME-format motif parsing (JASPAR ``?format=meme`` responses and local files)
- CIS-BP local PWM-table parsing (CIS-BP exposes no REST API; tables are local input)
- Exact dynamic programming p-value calibration following the FIMO/MEME convention
- A zero-order, GC-matched background computed from the target windows
- Both-strand scanning via reverse complement (palindromes reported on both strands)
- p < 1e-4 threshold with Benjamini-Hochberg FDR over the FULL window x motif test set
- A stdlib JASPAR REST client (retry-with-backoff, size-capped reads, pinned release)

Calibration honesty (decision D-01): every p-value comes from an exact dynamic
programming computation of the log-odds null distribution under the documented
MEME/FIMO convention (Staden 1989 lineage) -- NOT from empirical-null sampling.
"Exact" refers to the null distribution being computed exactly by DP; FIMO's own
documented convention discretizes per-column scores to integers in ``[0..100]``
(``PSSM_RANGE = 100``, MEME 4.8.1 ``src/pssm.h:13``), and this module deliberately
adopts that same integer scaling (decision D-02: no ad-hoc float/numpy score
quantization -- the standard convention itself is the scale we follow). Reported
scores carry this quantization, inverted back to bits as
``score_bits = scaled / scale + w * offset`` (MEME 4.8.1 ``src/logodds.c:50``).

Calibration details:

- Pseudocount: FIMO default 0.1, scaled by the background frequency of each letter
  (``p_ij = (freq_ij * nsites + 0.1 * bg_j) / (nsites + 0.1)``; probability form
  ``(freq_ij + 0.1 * bg_j) / 1.1`` when the header reports ``nsites == 0``), then
  ``score_ij = log2(p_ij / bg_j)`` in bits.
- Background: zero-order (single-nucleotide) Markov, GC-matched by counting
  letters over ALL target windows -- one background per scan set, shared by every
  motif's DP so p-values are cross-motif comparable. The stock uniform
  ``Background letter frequencies`` line embedded in JASPAR MEME responses is a
  placeholder and is parsed-and-ignored.
- Threshold: FIMO default p < 1e-4, resolved as the minimal integer scaled score
  ``x`` whose DP tail probability ``Pr(score >= x)`` drops below 1e-4.
- FDR: a single Benjamini-Hochberg call (``scipy.stats.false_discovery_control``)
  over the concatenated p-vector of the FULL window x motif x strand test set --
  never a per-sequence or per-window correction (decision D-03); ``q < 0.05`` rows
  are reported.
- E-value: ``p * total tested positions`` (expected false positives at that p).

Provenance: MEME Suite 4.8.1 source (``src/pssm.h:13``, ``src/pssm.c``
``get_pdf_table`` / ``get_pv_lookup`` -- ``pssm->pv[x] = Pr(score >= x)``, and
``src/logodds.c`` ``scale_lo`` plus the inversion comment at line 50) and the FIMO
documentation (meme-suite.org/meme/doc/fimo.html), read 2026-10-10.
"""

import math
import re
from dataclasses import dataclass

# --- FIMO constants (provenance: MEME Suite 4.8.1 source, read 2026-10-10) ---
# FIMO's internal integer-score granularity for scaled log-odds PSSMs
# (MEME 4.8.1 src/pssm.h:13 -- `#define PSSM_RANGE 100`).
PSSM_RANGE = 100
# fimo --motif-pseudo default: 0.1, applied per letter scaled by background
# frequency (FIMO documentation, "Background" section).
FIMO_PSEUDOCOUNT = 0.1
# fimo --thresh default: report matches with p-value < 1e-4.
FIMO_P_THRESHOLD = 1e-4
# Motif width bound enforced at parse time (matches the DP array bound the
# FIMO recipe assumes; w * PSSM_RANGE stays a small pure-Python table).
MAX_MOTIF_WIDTH = 100
# Background floor: letters absent from the target windows get a small
# strictly-positive frequency (renormalized) so log-odds stay finite --
# the MEME tools likewise require strictly positive background frequencies.
BG_FLOOR = 1e-3

_LETTERS = "ACGT"
_LETTER_INDEX = {letter: index for index, letter in enumerate(_LETTERS)}

_MATRIX_HEADER_RE = re.compile(
    r"alength=\s*(?P<alength>\d+)\s+w=\s*(?P<w>\d+)\s+nsites=\s*(?P<nsites>-?\d+)"
)


@dataclass
class Motif:
    """A position-frequency matrix motif in A/C/G/T row order.

    Attributes:
        motif_id: Stable identifier (e.g. ``MA2324.1`` from JASPAR, or the
            CIS-BP ``Motif ID`` metadata value).
        name: Human-readable motif name; falls back to ``motif_id``.
        freq_rows: One row per position, ``[A, C, G, T]`` probabilities in
            ``[0, 1]``.
        nsites: Site count the frequencies were estimated from; ``0`` switches
            log-odds scoring to the probability-form pseudocount branch.
    """

    motif_id: str
    name: str
    freq_rows: list[list[float]]
    nsites: int

    @property
    def width(self) -> int:
        """Number of positions in the motif."""
        return len(self.freq_rows)


def _parse_background_line(line: str, lineno: int) -> None:
    """Validate (and deliberately ignore) a MEME background-frequencies line.

    JASPAR embeds a stock uniform ``Background letter frequencies`` block in
    every MEME response; the scan background is GC-matched from the target
    windows instead, so the values are parsed for shape only and discarded.

    Args:
        line: The letter/frequency pairs line (e.g. ``A 0.25 C 0.25 ...``).
        lineno: 1-based line number for error messages.

    Raises:
        ValueError: If the line is not alternating letter/frequency pairs.
    """
    tokens = line.split()
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"MEME line {lineno}: background frequencies must be letter/frequency "
            f"pairs, got '{line}'."
        )
    for i in range(0, len(tokens), 2):
        letter, value = tokens[i], tokens[i + 1]
        if letter not in _LETTERS:
            raise ValueError(f"MEME line {lineno}: background letter '{letter}' is not in ACGT.")
        try:
            float(value)
        except ValueError as e:
            raise ValueError(
                f"MEME line {lineno}: background frequency '{value}' is not numeric."
            ) from e


def parse_meme(text: str) -> list[Motif]:
    """Parse MEME-format motif text with a strict parse-or-reject grammar.

    Recognized line types only: ``MEME version`` headers, inline
    ``ALPHABET= ACGT``, ``strands: + -``, ``Background letter frequencies``
    (with its pairs line, validated and ignored), ``MOTIF <id> [name]``,
    ``letter-probability matrix: alength= ... w= ... nsites= ... [E= ...]``
    headers, the ``w`` numeric probability rows that follow, and ``URL``
    trailers. Any other content raises -- fetched or hand-edited text is
    untrusted input (T-12-01/T-12-03).

    Args:
        text: MEME-format document (e.g. a JASPAR ``?format=meme`` response).

    Returns:
        One :class:`Motif` per ``MOTIF`` block, in document order.

    Raises:
        ValueError: If the text is empty, contains unrecognized lines, a
            matrix header is malformed (``alength`` != 4, ``w`` outside
            ``[1, MAX_MOTIF_WIDTH]``, missing fields, negative ``nsites``),
            a row is non-numeric or outside ``[0, 1]``, or a block is
            incomplete.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("MEME motif text is empty.")
    motifs: list[Motif] = []
    pending_id: str | None = None
    pending_name: str | None = None
    header: dict[str, int] | None = None
    rows: list[list[float]] = []
    rows_needed = 0
    expect_background = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if expect_background:
            _parse_background_line(line, lineno)
            expect_background = False
            continue
        if line.startswith("MEME version"):
            continue
        if line.startswith("Background letter frequencies"):
            expect_background = True
            continue
        if line.startswith("ALPHABET="):
            alphabet = line[len("ALPHABET=") :].strip()
            if set(alphabet) - set(_LETTERS):
                raise ValueError(f"MEME line {lineno}: alphabet must be ACGT, got '{alphabet}'.")
            continue
        if line.startswith("strands:"):
            strands = line[len("strands:") :].split()
            if not strands or any(strand not in {"+", "-"} for strand in strands):
                raise ValueError(f"MEME line {lineno}: malformed strands line '{line}'.")
            continue
        if line.startswith("MOTIF"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"MEME line {lineno}: MOTIF line lacks an id.")
            if pending_id is not None or header is not None:
                raise ValueError(
                    f"MEME line {lineno}: MOTIF block for '{pending_id}' has no "
                    "letter-probability matrix."
                )
            pending_id = parts[1]
            pending_name = " ".join(parts[2:]) or pending_id
            continue
        if line.startswith("letter-probability matrix:"):
            if pending_id is None:
                raise ValueError(
                    f"MEME line {lineno}: letter-probability matrix without a preceding MOTIF line."
                )
            if header is not None:
                raise ValueError(f"MEME line {lineno}: duplicate letter-probability matrix header.")
            match = _MATRIX_HEADER_RE.search(line)
            if match is None:
                raise ValueError(
                    f"MEME line {lineno}: matrix header must carry alength/w/nsites, got '{line}'."
                )
            alength = int(match.group("alength"))
            width = int(match.group("w"))
            nsites = int(match.group("nsites"))
            if alength != 4:
                raise ValueError(f"MEME line {lineno}: alength must be 4 (A/C/G/T), got {alength}.")
            if not 1 <= width <= MAX_MOTIF_WIDTH:
                raise ValueError(
                    f"MEME line {lineno}: motif width w={width} outside [1, {MAX_MOTIF_WIDTH}]."
                )
            if nsites < 0:
                raise ValueError(f"MEME line {lineno}: nsites must be >= 0, got {nsites}.")
            header = {"alength": alength, "w": width, "nsites": nsites}
            rows = []
            rows_needed = width
            continue
        if line.startswith("URL"):
            continue
        if rows_needed > 0:
            tokens = line.split()
            if len(tokens) != 4:
                raise ValueError(
                    f"MEME line {lineno}: probability row must have 4 values "
                    f"(A/C/G/T), got {len(tokens)}."
                )
            try:
                values = [float(token) for token in tokens]
            except ValueError as e:
                raise ValueError(
                    f"MEME line {lineno}: non-numeric probability row '{line}'."
                ) from e
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(
                    f"MEME line {lineno}: probability row values must be in [0, 1], got '{line}'."
                )
            rows.append(values)
            rows_needed -= 1
            if rows_needed == 0:
                motifs.append(
                    Motif(
                        motif_id=pending_id or "",
                        name=pending_name or pending_id or "",
                        freq_rows=rows,
                        nsites=header["nsites"] if header else 0,
                    )
                )
                pending_id = None
                pending_name = None
                header = None
            continue
        raise ValueError(f"MEME line {lineno}: unrecognized content '{line[:60]}'.")
    if expect_background:
        raise ValueError("MEME text ends after 'Background letter frequencies' with no data.")
    if pending_id is not None or header is not None or rows_needed > 0:
        raise ValueError(
            f"MEME text ends with an incomplete MOTIF block for '{pending_id}' "
            f"({rows_needed} rows missing)."
        )
    if not motifs:
        raise ValueError("MEME text contains no MOTIF blocks.")
    return motifs


def log_odds_matrix(
    freq_rows: list[list[float]], nsites: int, bg: dict[str, float]
) -> list[list[float]]:
    """Convert PFM rows to log-odds bits with FIMO's background-scaled pseudocount.

    Implements the FIMO default ``--motif-pseudo 0.1`` convention: the
    pseudocount is applied per letter after multiplying by the corresponding
    background frequency. With ``nsites > 0`` counts are reconstructed from
    the frequencies; ``nsites == 0`` uses the probability form.

    Args:
        freq_rows: Per-position ``[A, C, G, T]`` probabilities.
        nsites: Site count from the motif header (0 selects the probability
            form).
        bg: Background frequencies with keys ``A``/``C``/``G``/``T``.

    Returns:
        Log-odds score matrix in bits, same shape as ``freq_rows``.
    """
    n = float(nsites) if nsites > 0 else 1.0
    matrix: list[list[float]] = []
    for row in freq_rows:
        scores_row: list[float] = []
        for letter, freq in zip(_LETTERS, row, strict=True):
            if nsites > 0:
                p = (freq * n + FIMO_PSEUDOCOUNT * bg[letter]) / (n + FIMO_PSEUDOCOUNT)
            else:
                p = (freq + FIMO_PSEUDOCOUNT * bg[letter]) / (1.0 + FIMO_PSEUDOCOUNT)
            scores_row.append(math.log2(p / bg[letter]))
        matrix.append(scores_row)
    return matrix


def pvalue_table(
    scores: list[list[float]], bg: dict[str, float]
) -> tuple[list[float], float, float, int]:
    """Exact-DP null distribution of the scaled log-odds score (FIMO recipe).

    Scales the score matrix to integers in ``[0..PSSM_RANGE]`` per column
    (``scale = PSSM_RANGE / (max - min)``, ``offset = min``), then runs the
    column-wise convolution ``pdf[k + s] += pdf[k] * bg[letter]`` in pure
    Python (decision D-02) and folds the pdf into the reverse cumulative
    table ``pv[x] = Pr(score >= x)`` (MEME 4.8.1 ``src/pssm.c``).

    Args:
        scores: Log-odds score matrix (bits) from :func:`log_odds_matrix`.
        bg: Background frequencies with keys ``A``/``C``/``G``/``T``.

    Returns:
        Tuple ``(pv, scale, offset, max_scaled)`` where ``pv[x]`` is
        ``Pr(scaled score >= x)`` for ``0 <= x <= w * PSSM_RANGE``.

    Raises:
        ValueError: If the matrix is empty or has no score variation (a
            uniform PWM has no informative null distribution).
    """
    flat = [value for row in scores for value in row]
    if not flat:
        raise ValueError("Motif score matrix is empty.")
    small, large = min(flat), max(flat)
    if large == small:
        raise ValueError("Motif has no score variation (uniform PWM); p-values are undefined.")
    scale = PSSM_RANGE / (large - small)
    offset = small
    scaled = [[round((value - offset) * scale) for value in row] for row in scores]
    pdf = [1.0] + [0.0] * (len(scores) * PSSM_RANGE)
    for scaled_row in scaled:
        nxt = [0.0] * len(pdf)
        for letter_index, s in enumerate(scaled_row):
            prob = bg[_LETTERS[letter_index]]
            for k, mass in enumerate(pdf):
                if mass:
                    nxt[k + s] += mass * prob
        pdf = nxt
    total = sum(pdf)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"P-value DP failed to normalize (total probability {total}).")
    pv = pdf
    for x in range(len(pv) - 2, -1, -1):
        pv[x] += pv[x + 1]
    return pv, scale, offset, len(scores) * PSSM_RANGE


def _threshold_scaled(pv: list[float], p_threshold: float) -> int:
    """Minimal integer scaled score whose tail probability is under threshold.

    ``pv`` is non-increasing with ``pv[0] == 1.0``, so descending until the
    first score still at/above the threshold and stepping past it yields the
    minimal passing score; when no achievable score passes (even the maximum
    is at/above the threshold) the returned value is ``len(pv)``, one past
    the maximum scaled score.
    """
    for x in range(len(pv) - 1, -1, -1):
        if pv[x] < p_threshold:
            continue
        return x + 1
    return len(pv)


def threshold_bits(
    pv: list[float],
    scale: float,
    offset: float,
    w: int,
    *,
    p_threshold: float = FIMO_P_THRESHOLD,
) -> float:
    """Report threshold in bits via the FIMO inversion ``x/scale + w*offset``.

    Args:
        pv: Reverse-cumulative p-value table from :func:`pvalue_table`.
        scale: Scale factor from :func:`pvalue_table`.
        offset: Offset (minimum matrix entry) from :func:`pvalue_table`.
        w: Motif width.
        p_threshold: P-value threshold (FIMO default 1e-4).

    Returns:
        The reporting threshold in bits; scores at or above it have
        ``p < p_threshold`` under the DP null.
    """
    return _threshold_scaled(pv, p_threshold) / scale + w * offset


@dataclass
class _Pssm:
    """Internal scaled scoring model for one motif under one background."""

    motif_id: str
    w: int
    scaled: list[list[int]]
    scale: float
    offset: float
    pv: list[float]
    x_threshold: int


def _build_pssm(motif: Motif, bg: dict[str, float], p_threshold: float) -> _Pssm:
    """Build the scaled PSSM + DP p-value table for one motif/background pair."""
    scores = log_odds_matrix(motif.freq_rows, motif.nsites, bg)
    pv, scale, offset, _ = pvalue_table(scores, bg)
    scaled = [[round((value - offset) * scale) for value in row] for row in scores]
    return _Pssm(
        motif_id=motif.motif_id,
        w=motif.width,
        scaled=scaled,
        scale=scale,
        offset=offset,
        pv=pv,
        x_threshold=_threshold_scaled(pv, p_threshold),
    )


def _iter_position_scores(seq: str, pssm: _Pssm):
    """Yield ``(position, scaled_score)`` for every scorable window position.

    Positions whose k-mer contains characters outside ACGT (e.g. ``N``)
    cannot be scored and are skipped entirely -- they are not tested.
    """
    index = _LETTER_INDEX
    w = pssm.w
    for pos in range(len(seq) - w + 1):
        total = 0
        scorable = True
        for j in range(w):
            letter_index = index.get(seq[pos + j])
            if letter_index is None:
                scorable = False
                break
            total += pssm.scaled[j][letter_index]
        if scorable:
            yield pos, total


def gc_background(windows: list[str]) -> dict[str, float]:
    """Zero-order GC-matched background from ALL target-window bases.

    Counts A/C/G/T over every window (case-insensitive; other characters are
    ignored) so that one background serves the whole scan set -- sharing it
    across every motif's DP is what makes p-values cross-motif comparable.
    Letters absent from the windows are floored at ``BG_FLOOR`` and the
    vector is renormalized, keeping log-odds finite.

    Args:
        windows: Target DNA windows.

    Returns:
        Background frequencies with keys ``A``/``C``/``G``/``T`` summing to 1.

    Raises:
        ValueError: If the windows contain no A/C/G/T bases at all.
    """
    counts = dict.fromkeys(_LETTERS, 0)
    total = 0
    for window in windows:
        for base in window.upper():
            if base in counts:
                counts[base] += 1
                total += 1
    if total == 0:
        raise ValueError("gc_background: windows contain no A/C/G/T bases.")
    freqs = {letter: max(counts[letter] / total, BG_FLOOR) for letter in _LETTERS}
    norm = sum(freqs.values())
    return {letter: freq / norm for letter, freq in freqs.items()}


def scan_single_strand(
    window: str,
    motif: Motif,
    *,
    background: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Score one window against one motif on the ``+`` strand (p threshold only).

    This is the core scoring slice; the full scanner (:func:`scan`, added by
    the same module) layers the reverse-complement strand, the full-set BH
    correction, and E-values on top of exactly this scoring path.

    Args:
        window: DNA window (case-insensitive; non-ACGT positions are not
            scored).
        motif: Motif to scan with.
        background: Background frequencies; when ``None`` a zero-order
            GC-matched background is computed from this window (there is
            deliberately no uniform default).

    Returns:
        Hit dicts ``{motif_id, start, end, strand, score_bits, p}`` for every
        position with ``p < FIMO_P_THRESHOLD``; ``start``/``end`` are
        forward-window, 0-based, end-exclusive.
    """
    window = window.upper()
    if background is None:
        background = gc_background([window])
    pssm = _build_pssm(motif, background, FIMO_P_THRESHOLD)
    hits: list[dict[str, object]] = []
    for pos, s in _iter_position_scores(window, pssm):
        p = pssm.pv[s]
        if p < FIMO_P_THRESHOLD:
            hits.append({
                "motif_id": motif.motif_id,
                "start": pos,
                "end": pos + pssm.w,
                "strand": "+",
                "score_bits": s / pssm.scale + pssm.w * pssm.offset,
                "p": p,
            })
    return hits
