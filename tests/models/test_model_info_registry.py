"""Fast-leg registry-integrity tests for the model registry data layer (KWT-01..04).

No network: these tests parse the packaged ``model_info.yaml`` directly (same
pattern as ``tests/models/test_plant_helixseek_registry.py``) and pin the
pretrained tokenizer-variant checkpoints, the ``PRETRAIN_MODEL_MAPS`` task-class
strings, the tRNA titles in ``MODEL_INFO``, and the finetuned-entry base_model
lineage. The registry YAML has no runtime consumer inside the package, so wrong
values mislead without any runtime error — these structure tests are the
enforcement layer (quick task 261020-kwt).
"""

from pathlib import Path

import dnallm
import pytest
import yaml
from dnallm.models.modeling_auto import MODEL_INFO, PRETRAIN_MODEL_MAPS

REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml"

# Misspelled task class ("Casual" for "Causal") and the joined title form
# ("aDNA sequence" for "a DNA sequence") each live in exactly this module and
# the production sources they guard against — a recurrence of either typo in
# modeling_auto.py fails the tests below.
MISSPELLED_TASK_CLASS = "Casual Language Model"
JOINED_TITLE_FORM = "aDNA sequence"

EXPECTED_PRETRAINED_COUNT = 49

# The 15 PDLLMs tokenizer-variant pretrained checkpoints (KWT-01), pinned to
# their task types. Architectures cross-checked at planning time: Bert/Esm/
# ModernBert masked-LM heads -> mask; GPT2/Gemma/Mamba causal-LM heads ->
# generation.
VARIANT_TASK_TYPES = {
    "zhangtaolab/plant-dnabert-6mer": "mask",
    "zhangtaolab/plant-dnabert-singlebase": "mask",
    "zhangtaolab/plant-nucleotide-transformer-6mer": "mask",
    "zhangtaolab/plant-nucleotide-transformer-singlebase": "mask",
    "zhangtaolab/plant-dnamodernbert-singlebase": "mask",
    "zhangtaolab/plant-dnagpt-6mer": "generation",
    "zhangtaolab/plant-dnagpt-singlebase": "generation",
    "zhangtaolab/plant-dnagemma-6mer": "generation",
    "zhangtaolab/plant-dnagemma-singlebase": "generation",
    "zhangtaolab/plant-dnamamba-2mer": "generation",
    "zhangtaolab/plant-dnamamba-3mer": "generation",
    "zhangtaolab/plant-dnamamba-4mer": "generation",
    "zhangtaolab/plant-dnamamba-5mer": "generation",
    "zhangtaolab/plant-dnamamba-6mer": "generation",
    "zhangtaolab/plant-dnamamba-singlebase": "generation",
}

# Pre-existing external base_model pointers (PlantCAD2 Medium/Large upstream
# repos, real but not registry pretrained entries — only PlantCAD2-Small is
# registered). Documented out of scope for the lineage backfill.
EXTERNAL_BASE_ALLOWLIST = {
    "kuleshov-group/PlantCAD2-Medium-l48-d1024",
    "kuleshov-group/PlantCAD2-Large-l48-d1536",
}

# Finetuned entries intentionally lacking base_model. Empty after the KWT-03
# backfill: the test asserts this set equals the actual entries without
# lineage, so any future finetuned entry added without a base_model fails
# loudly instead of silently regressing coverage.
DERIVATION_EXEMPT: set[str] = set()

# Load-bearing lineage mappings pinned explicitly (KWT-03): the two repo-alias
# tRNA entries, two prefix-derived variant bases, and the three name-derived
# short-alias family bases.
SPOT_PINNED_BASE_MODELS = {
    "zhangtaolab/tRNADetector": "zhangtaolab/plant-dnamamba-singlebase",
    "zhangtaolab/tRNAPointer": "zhangtaolab/plant-nucleotide-transformer-singlebase",
    "zhangtaolab/plant-dnabert-BPE-promoter": "zhangtaolab/plant-dnabert-BPE",
    "zhangtaolab/plant-dnamamba-6mer-promoter": "zhangtaolab/plant-dnamamba-6mer",
    "zhangtaolab/agront-1b-promoter": "InstaDeepAI/agro-nucleotide-transformer-1b",
    "zhangtaolab/dnabert2-promoter": "zhihan1996/DNABERT-2-117M",
    "zhangtaolab/nucleotide-transformer-v2-100m-promoter": (
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    ),
}


def _load_registry() -> dict:
    """Parse the packaged registry with yaml.safe_load, failing with file context.

    Returns:
        The full registry mapping with ``pretrained`` and ``finetuned`` sections.

    Raises:
        pytest.fail.Failed: If the file does not parse as a YAML mapping.
    """
    content = REGISTRY_PATH.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML error in {REGISTRY_PATH.name}: {e}")
    assert isinstance(data, dict), "registry must parse to a mapping"
    return data


def _pretrained() -> list[dict]:
    """Return the pretrained section as a list of entry mappings."""
    section = _load_registry().get("pretrained")
    assert isinstance(section, list), "pretrained section must be a list"
    return section


def _finetuned() -> list[dict]:
    """Return the finetuned section as a list of entry mappings."""
    section = _load_registry().get("finetuned")
    assert isinstance(section, list), "finetuned section must be a list"
    return section


def test_pretrained_section_holds_49_entries() -> None:
    """Pretrained section carries exactly 49 entries (34 original + 15 variants)."""
    assert len(_pretrained()) == EXPECTED_PRETRAINED_COUNT


def test_variant_checkpoints_pinned_with_task_type_and_variant_names() -> None:
    """Each of the 15 variant checkpoints appears exactly once (KWT-01).

    Pins the pretrained task_type per variant and requires both the entry name
    and the task description to mention the tokenizer variant.
    """
    pretrained = _pretrained()
    for repo, task_type in VARIANT_TASK_TYPES.items():
        matches = [e for e in pretrained if e.get("model") == repo]
        assert len(matches) == 1, f"expected exactly one pretrained entry for {repo}"
        entry = matches[0]
        assert entry["task"]["task_type"] == task_type, repo
        variant = repo.rsplit("/", 1)[1].rsplit("-", 1)[1]
        assert isinstance(entry.get("name"), str), repo
        assert entry["name"], repo
        assert variant in entry["name"], f"{repo}: name {entry['name']!r} lacks {variant!r}"
        assert variant in entry["task"]["describe"], (
            f"{repo}: describe {entry['task']['describe']!r} lacks {variant!r}"
        )


def test_plant_dnagpt_family_maps_to_causal_language_model() -> None:
    """Plant DNAGPT checkpoints are GPT2LMHeadModel, so the family maps to Causal (KWT-02)."""
    assert PRETRAIN_MODEL_MAPS["Plant DNAGPT"] == ["Zhangtaolab", "Causal Language Model"]


def test_no_pretrain_family_entry_contains_misspelled_task_class() -> None:
    """No PRETRAIN_MODEL_MAPS family entry carries the misspelled task class (KWT-02)."""
    for family, spec in PRETRAIN_MODEL_MAPS.items():
        for element in spec:
            assert MISSPELLED_TASK_CLASS not in element, family


def test_trna_titles_read_a_dna_sequence() -> None:
    """tRNADetector/tRNAPointer titles restore the space in "a DNA sequence" (KWT-04)."""
    for family in ("tRNADetector", "tRNAPointer"):
        title = MODEL_INFO[family]["title"]
        assert "a DNA sequence" in title, family
        assert JOINED_TITLE_FORM not in title, family


def test_every_finetuned_entry_carries_base_model() -> None:
    """Every finetuned entry has base_model unless explicitly exempt (KWT-03).

    The exempt set must equal the actual set of entries lacking lineage, so a
    future finetuned entry added without base_model fails here even if someone
    widens DERIVATION_EXEMPT without cleaning it up afterwards.
    """
    missing = {e["model"] for e in _finetuned() if "base_model" not in e}
    assert missing == DERIVATION_EXEMPT


def test_every_base_model_resolves_to_pretrained_or_external() -> None:
    """Each base_model is a pretrained registry id or a documented external pointer."""
    pretrained_ids = {e["model"] for e in _pretrained()}
    for entry in _finetuned():
        base = entry.get("base_model")
        if base is None:
            continue
        assert base in pretrained_ids or base in EXTERNAL_BASE_ALLOWLIST, (
            f"{entry['model']}: base_model {base!r} resolves to neither"
        )


def test_spot_pinned_lineage_mappings_hold() -> None:
    """The load-bearing base_model mappings keep their pinned values (KWT-03)."""
    by_repo = {e["model"]: e for e in _finetuned()}
    for repo, expected_base in SPOT_PINNED_BASE_MODELS.items():
        assert repo in by_repo, repo
        assert by_repo[repo]["base_model"] == expected_base, repo


def test_pretrained_entries_carry_no_base_model() -> None:
    """base_model is a finetuned-only field — no pretrained entry carries it."""
    offenders = [e["model"] for e in _pretrained() if "base_model" in e]
    assert offenders == []


# ─────────────────────────────────────────────────────────────────────────────
# 22-02 enrichment integrity (REG-02/REG-03) — extension only; nothing above
# this section is modified. The enrichment adds tokenizer_type / max_length
# fields via scripted text insertion and an end-of-file provenance block.
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_FINETUNED_COUNT = 187

# Provenance sentinels: the kwt freeze-provenance block (2026-10-03) plus the
# 22-02 enrichment provenance block appended at end of file. Each sentinel
# must appear exactly once, verbatim — the blocks are unrecoverable once lost
# (Pitfall 3), so their presence is pinned here (REG-03).
PROVENANCE_SENTINELS = (
    "# PlantHelixSeek finetuned checkpoints (Phase 06-01 registry freeze, 2026-10-03).",
    "# Checkpoint commit shas (trust_remote_code provenance):",
    "#   PlantHelixSeek-CRE  7093de3baf64bf59ac147be3971482a13238aabd",
    "#   PlantHelixSeek-Anno 6d39386ab562b2a9a3d3581ebe28e235383c8d3c",
    "# Freeze tooling is intentionally uncommitted (owner constraint 2026-10-02): it ran",
    "# Registry enrichment: tokenizer_type / max_length (Phase 22-02, 2026-10-20).",
)

# Spot pins (T-22-05 mitigation): load-bearing enrichment values measured by
# the one-time remote probes (D-01; recorded in the gitignored
# .scratch/enrichment_results.json). hyenadna is pinned to the MEASURED
# config.json max_seq_len=32770 (32k context + 2 special tokens), not the
# repo-name 32k figure; the nucleotide-transformer pin (2050) likewise holds
# the measured 2048+2 value.
SPOT_PINNED_MAX_LENGTHS = {
    "LongSafari/hyenadna-small-32k-seqlen-hf": 32770,
    "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16": 131072,
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16": 131072,
    "zhangtaolab/plant-dnabert-BPE": 512,
    "zhangtaolab/plant-nucleotide-transformer-BPE": 2050,
}

SPOT_PINNED_TOKENIZER_TYPES = {
    "zhangtaolab/plant-dnabert-BPE": "DebertaV2Tokenizer",
    "zhangtaolab/plant-dnabert-6mer": "EsmTokenizer",
    "zhangtaolab/plant-dnamamba-singlebase": "EsmTokenizer",
}

# Finetuned entries whose base_model is not a registry pretrained id — the 14
# external-based plantcad adapters (bases: kuleshov-group/PlantCAD2-Medium /
# -Large, per EXTERNAL_BASE_ALLOWLIST above). They cannot inherit, so D-02
# requires an explicit committed max_length on every one of them.
EXPECTED_EXTERNAL_BASED_COUNT = 14


def _positive_int(value) -> bool:
    """True for a strict positive int (bool excluded, None excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def test_all_pretrained_entries_carry_positive_int_max_length() -> None:
    """Every pretrained entry carries max_length as a positive int (D-02, 49/49)."""
    offenders = [e["model"] for e in _pretrained() if not _positive_int(e.get("max_length"))]
    assert offenders == []


def test_lab_pretrained_entries_carry_tokenizer_type() -> None:
    """All 23 zhangtaolab pretrained entries carry a non-empty str tokenizer_type (D-03)."""
    lab_pre = [e for e in _pretrained() if e["model"].startswith("zhangtaolab/")]
    assert len(lab_pre) == 23
    offenders = [
        e["model"]
        for e in lab_pre
        if not isinstance(e.get("tokenizer_type"), str) or not e["tokenizer_type"]
    ]
    assert offenders == []


def test_third_party_enrichment_values_present_or_absent_never_degenerate() -> None:
    """A present enrichment field is a non-empty str / positive int — never null, empty, or 0.

    Third-party gaps surface as ABSENT keys (D-03): the ``in`` checks below skip
    absent keys and catch exactly the degenerate forms a hand-edit could
    introduce (``tokenizer_type: null``, ``max_length: 0``).
    """
    for entry in _pretrained() + _finetuned():
        if "tokenizer_type" in entry:
            value = entry["tokenizer_type"]
            assert isinstance(value, str), entry["model"]
            assert value, entry["model"]
        if "max_length" in entry:
            assert _positive_int(entry["max_length"]), entry["model"]


def test_finetuned_max_length_only_as_explicit_override() -> None:
    """Committed finetuned max_length differs from the base's, or the base is external (D-02).

    Identical values are inherited at query time and must never be duplicated
    into the YAML; explicit values exist only for measured differences and for
    the 14 external-based plantcad repos, whose bases sit outside the registry.
    """
    pre_max_length = {e["model"]: e.get("max_length") for e in _pretrained()}
    for entry in _finetuned():
        own = entry.get("max_length")
        if own is None:
            continue
        base = entry.get("base_model")
        if base in pre_max_length:
            assert own != pre_max_length[base], (
                f"{entry['model']}: duplicates its base's committed max_length "
                f"{own} (D-02 forbids inheritance duplication)"
            )
    external = [e for e in _finetuned() if e.get("base_model") not in pre_max_length]
    assert len(external) == EXPECTED_EXTERNAL_BASED_COUNT
    assert all(e["model"].startswith("plantcad/") for e in external)
    offenders = [e["model"] for e in external if not _positive_int(e.get("max_length"))]
    assert offenders == []


def test_provenance_blocks_present_verbatim() -> None:
    """Each provenance sentinel appears exactly once in the file text (REG-03)."""
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    for sentinel in PROVENANCE_SENTINELS:
        assert text.count(sentinel) == 1, sentinel


def test_registry_section_counts_unchanged() -> None:
    """The enrichment is field-level insertion: counts stay 49 pretrained / 187 finetuned."""
    assert len(_pretrained()) == EXPECTED_PRETRAINED_COUNT
    assert len(_finetuned()) == EXPECTED_FINETUNED_COUNT


def test_spot_pinned_enrichment_values() -> None:
    """Load-bearing enrichment values keep their probe-measured pins (T-22-05)."""
    by_repo = {e["model"]: e for e in _pretrained()}
    for repo, expected in SPOT_PINNED_MAX_LENGTHS.items():
        assert repo in by_repo, repo
        assert by_repo[repo].get("max_length") == expected, repo
    for repo, expected in SPOT_PINNED_TOKENIZER_TYPES.items():
        assert repo in by_repo, repo
        assert by_repo[repo].get("tokenizer_type") == expected, repo
