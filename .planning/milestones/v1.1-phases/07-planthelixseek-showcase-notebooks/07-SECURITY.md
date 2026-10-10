---
phase: "07"
slug: "planthelixseek-showcase-notebooks"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-04"
---

# Phase 07 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| notebook kernel -> repo tree | Local (D-09) notebook execution runs with the developer cwd inside the repo; derived artifacts could dirty the tree | derived BED/GFF3/intermediates (local only) |
| fast-lane test exec -> notebook source | tests/examples/test_examples.py execs every import statement node found in notebook source on legs without optional deps | import statements / executed code |
| model hub -> runtime | trust_remote_code checkpoint load through the dnallm route (owner-org ModelScope repos) | model weights + remote code |
| environment -> result integrity | absence of the fla KDA kernels silently changes the math instead of failing | environment -> computed metrics |
| upstream source -> notebook constants | the B<->L permutation is transcribed from an external repo file not reproduced here (A4) | transcription of upstream code |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-07-01 | Tampering | local CRE notebook execution | medium | mitigate | Derived artifacts under gitignored `outputs/` only (.gitignore:63 + check_docs_sync IGNORE); Task git-status gates; nightly tmp-sandbox + assert_tree_clean | closed |
| T-07-02 | Tampering | results integrity (fla fallback) | high | mitigate | D-16 first-code-cell hard guard (find_spec + RuntimeError) verified in both notebooks; structure test pins guard shape; nightly legs install `.[base,fla]` | closed |
| T-07-03 | Elevation | test_notebook_imports exec surface | medium | mitigate | No `import fla` statement anywhere in notebook source (version via importlib.metadata); AST-level structure test enforces it | closed |
| T-07-04 | Elevation | checkpoint supply chain (trust_remote_code) | medium | accept | Registry-pinned owner-org repos with recorded commit shas (Phase-6 provenance in model_info.yaml + models.lock); no new model ids this phase; ASVS L1 accepts with existing pinning controls | closed |
| T-07-05 | Tampering | committed-notebook drift | medium | mitigate | Structure tests pin guard/provenance cells for BOTH notebooks; nightly re-execution re-verifies floors against current source | closed |
| T-07-06 | Tampering | B<->L permutation transcription (A4) | medium | mitigate | Permutation transcribed from upstream predict_genome_multigpu.py:97-101 with source citation; executed notebook reproduces exon_f1=0.7522 / genes=59 exactly; nightly genes_above_floor band is the backstop | closed |
| T-07-07 | Tampering | decode-rule substitution | medium | mitigate | Frozen argmax BILOU decode pinned by must_haves prohibition (verified: argmax is the only decode path in code); any decode change invalidates the bands | closed |
| T-07-08 | Tampering | local Anno execution side effects | medium | mitigate | Derived GFF3/intermediates under gitignored `outputs/` only; Task git-status gates; nightly sandbox + assert_tree_clean (slow test re-verified live by the verifier) | closed |
| T-07-SC | Tampering | package installs (reserved) | high | mitigate | No package-manager installs this phase (pyproject.toml untouched in the phase diff); no [ASSUMED]/[SUS] packages; reserved row kept | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-07-01 | T-07-04 | trust_remote_code checkpoint load from owner-org ModelScope repos, pinned by the packaged registry with recorded commit shas; no new model ids introduced this phase; residual risk accepted at ASVS L1 | owner (via plan-time threat model, disposition: accept) | 2026-10-03 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-04 | 9 | 9 | 0 | gsd orchestrator (L1 grep-depth, ASVS L1 short-circuit per secure-phase.md Step 3) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-04
