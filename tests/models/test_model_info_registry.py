"""Fast-leg registry-integrity tests for the model registry data layer (KWT-01/02/04).

No network: these tests parse the packaged ``model_info.yaml`` directly (same
pattern as ``tests/models/test_plant_helixseek_registry.py``) and pin the
pretrained tokenizer-variant checkpoints, the ``PRETRAIN_MODEL_MAPS`` task-class
strings, and the tRNA titles in ``MODEL_INFO``. The registry YAML has no runtime
consumer inside the package, so wrong values mislead without any runtime error —
these structure tests are the enforcement layer (quick task 261020-kwt).
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
