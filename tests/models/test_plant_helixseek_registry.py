"""Fast-leg structure tests for the PlantHelixSeek registry entries (REG-01/REG-02).

No network: these tests parse the packaged ``model_info.yaml`` directly (it has no
runtime consumer — Phase 06 research Pattern 1) and pin both finetuned entries to the
frozen upstream label order. The slow-leg smoke test
(``tests/models/test_plant_helixseek_smoke.py``) imports the frozen constants below
and asserts them post-load against the real checkpoints, so a permuted or
semantically void registry list fails there against this committed record.
"""

from pathlib import Path

import dnallm
import pytest
import yaml

REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml"

CRE_REPO_ID = "zhangtaolab/PlantHelixSeek-CRE"
ANNO_REPO_ID = "zhangtaolab/PlantHelixSeek-Anno"
BASE_MODEL_ID = "zhangtaolab/PlantHelixSeek"

# Frozen 2026-10-03 (Phase 06-01). Semantic source: the upstream training script
# zhangtaolab/PlantHelixSeek, scripts/gene_annotation/train_token_cls.py:78-96
# (LABEL_NAMES), fetched 2026-10-02. The checkpoint configs carry no ordering
# semantics — Anno config.id2label is the transformers LABEL_i placeholder pattern and
# CRE config has none (proven by the one-shot freeze run) — so the upstream order is
# the contract. CRE class order: class-1 == CRE per upstream scripts/cis_regulatory/
# README.md (bin score = arithmetic mean of class-1 probabilities).
CRE_LABELS = ["Not CRE", "CRE"]
ANNO_LABELS = [
    "O",
    "B-CDS",
    "I-CDS",
    "L-CDS",
    "U-CDS",
    "B-INTRON",
    "I-INTRON",
    "L-INTRON",
    "U-INTRON",
    "B-UTR5",
    "I-UTR5",
    "L-UTR5",
    "U-UTR5",
    "B-UTR3",
    "I-UTR3",
    "L-UTR3",
    "U-UTR3",
]


def _load_finetuned() -> list[dict]:
    """Parse the packaged registry with yaml.safe_load, failing with file context."""
    content = REGISTRY_PATH.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML error in {REGISTRY_PATH.name}: {e}")
    assert isinstance(data, dict), "registry must parse to a mapping"
    finetuned = data.get("finetuned")
    assert isinstance(finetuned, list), "finetuned section must be a list"
    return finetuned


def _entry(repo_id: str) -> dict:
    """Resolve exactly one finetuned entry by repo id (duplicate-append guard)."""
    matches = [e for e in _load_finetuned() if e.get("model") == repo_id]
    assert len(matches) == 1, (
        f"expected exactly one finetuned entry for {repo_id}, found {len(matches)}"
    )
    return matches[0]


def test_registry_yaml_parses():
    """The packaged registry parses cleanly with yaml.safe_load."""
    assert len(_load_finetuned()) > 0


def test_cre_entry_matches_frozen_order():
    """CRE entry: binary task, 2 labels, frozen ["Not CRE", "CRE"] order (REG-02)."""
    entry = _entry(CRE_REPO_ID)
    task = entry["task"]
    assert task["task_type"] == "binary"
    assert task["num_labels"] == 2
    assert task["label_names"] == CRE_LABELS
    assert task["threshold"] == 0.5
    assert entry["base_model"] == BASE_MODEL_ID
    # Pitfall-1 guard: never the transformers placeholder pattern.
    assert all("LABEL_" not in name for name in task["label_names"])


def test_anno_entry_matches_frozen_order():
    """Anno entry: token task, 17 unique BILOU labels in the frozen upstream order."""
    entry = _entry(ANNO_REPO_ID)
    task = entry["task"]
    assert task["task_type"] == "token"
    assert task["num_labels"] == 17
    assert task["label_names"] == ANNO_LABELS
    assert len(set(ANNO_LABELS)) == 17
    assert task["threshold"] == 0.5
    assert entry["base_model"] == BASE_MODEL_ID
    assert task["label_names"][1] == "B-CDS"
    assert all("LABEL_" not in name for name in task["label_names"])
