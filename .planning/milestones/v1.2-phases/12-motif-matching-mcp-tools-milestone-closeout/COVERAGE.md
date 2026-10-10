# API Coverage — JASPAR REST (jaspar.elixir.no/api/v1)

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Scope: the stdlib JASPAR REST client in `dnallm/interpret/motifs.py` (MOTIF-01 / REV-10).
> CIS-BP has no REST API (bulk ZIPs only — re3data registry, RESEARCH Pattern 3) and is
> intentionally a LOCAL-parse input, not an API integration.

| capability | decision | reason |
|---|---|---|
| matrix search (`GET /api/v1/matrix/?name=...&collection=...&release=...`) | INTEGRATE | Resolves matrix IDs by TF name (e.g. BCL11A → MA2324.1/MA2504.1); needed by the fixture workflow and library callers |
| matrix download, MEME format (`GET /api/v1/matrix/{id}/?format=meme`) | INTEGRATE | Core PWM acquisition; MOTIF-01 locks "prefer `format=meme`" |
| collection filter (`collection=CORE` on matrix search) | INTEGRATE | Scopes searches to the curated CORE collection; the fixture workflow depends on it |
| matrix download, other formats (jaspar / pfm / transfas) | OPT-OUT | MEME format is the locked preferred wire format per MOTIF-01; no consumer exists for the other formats this phase |
| tax_id filter on matrix search | OPT-OUT | The fixture workflow resolves TFs by name within CORE; no taxonomy-driven discovery need in v1.2 |
| releases list (`GET /api/v1/releases/`) | OPT-OUT | Release is a frozen module-constant query parameter with provenance (fixture reproducibility, RESEARCH Pitfall 7); runtime release enumeration has no consumer |
| bulk download (matrix-set ZIPs) | OPT-OUT | Hotspot-window scanning needs per-motif fetch only; genome-wide/bulk scanning is the deferred GENOMEWIDE-SCAN v1.3+ scope |
