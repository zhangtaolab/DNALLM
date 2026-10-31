"""Cross-module user-story tests over the ``dnallm.preflight`` facade (23-05).

This suite proves the phase's user story end-to-end through the PUBLIC
facade only — every import comes from ``dnallm.preflight`` directly, never
from a submodule path or private helper (the one-core-many-shells
contract: if a shell cannot drive the story through the facade, the
facade is incomplete).

Stories covered:

1. Facade completeness: every documented public name resolves via the
   facade and appears in ``__all__``.
2. The happy-path user story: tmp csv dataset -> validate_data ->
   check_compatibility(predict intent, fake remote-confirmed prober,
   shortlist 3) -> pick the top lab listing entry -> generate_config ->
   edit_config (num_labels relabel) — every step returns a report, zero
   network (the prober is a fake).
3. The severed-network story (SC-5): the same flow with a prober raising
   ``ConnectionError`` — availability degrades to unknown with a reason,
   the flow never hard-errors, and every step still returns a report.
4. Intent routing inside the story: the finetune branch lists pretrained
   entries only, lab-first.

Note on ``edit_config`` input: ``GeneratedConfig.config`` holds the
VALIDATED section objects; the round-trip editor's documented input is
the raw mapping, so the story derives the raw dict from the generated
report (``model_dump`` per section) before patching — exactly what a
shell serializing the generated config would hand back.
"""

from __future__ import annotations

import csv

import dnallm.preflight as preflight
from dnallm.preflight import (
    AvailabilityReport,
    AvailabilityState,
    CompatibilityReport,
    DataReport,
    GeneratedConfig,
    Hub,
    check_compatibility,
    validate_data,
)

# The shell-facing contract: the four user-story functions plus the
# endpoint/reporting helpers and the report types shells render.
REQUIRED_FACADE_NAMES = [
    "AvailabilityReport",
    "CompatibilityReport",
    "DataReport",
    "GeneratedConfig",
    "ModelVerdict",
    "SectionError",
    "check_compatibility",
    "clear_probe_memo",
    "edit_config",
    "effective_endpoints",
    "generate_config",
    "probe_availability",
    "validate_data",
]


def _write_story_csv(directory) -> str:
    """Write the story's tmp csv: sequence/label columns, mixed lengths,
    one row over 512 chars, two N-containing rows."""
    rows = [
        ("sequence", "label"),
        ("ACGTACGTAC", "1"),
        ("ACGT" * 200, "0"),  # 800 chars — over every pinned char budget
        ("ACGTNACGTN", "1"),  # N-containing
        ("NNNN", "0"),  # N-containing (all-N) and below min length
        ("acgtacgtacgt", "1"),  # lowercase soft-mask
    ]
    path = directory / "story.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return str(path)


def _remote_confirmed_prober(model_id: str) -> AvailabilityReport:
    """Fake prober standing in for a healthy network: every repo is
    confirmed on ModelScope with plausible metadata (zero real I/O)."""
    return AvailabilityReport(
        state=AvailabilityState.REMOTE_CONFIRMED,
        size_bytes=1_048_576,
        revision="deadbeef",
        source_hub=Hub.MODELSCOPE,
    )


def _severed_prober(model_id: str) -> AvailabilityReport:
    """Fake prober standing in for a severed network (SC-5)."""
    raise ConnectionError("network severed for test")


def _raw_config(generated: GeneratedConfig) -> dict:
    """Derive the raw config mapping from a generated report.

    ``GeneratedConfig.config`` carries the validated section objects;
    ``edit_config`` takes the raw mapping, so a shell serializes the
    generated sections (``model_dump`` per section, plain dicts kept
    as-is) before handing them back for editing.
    """
    return {
        name: (section.model_dump() if hasattr(section, "model_dump") else section)
        for name, section in generated.config.items()
    }


def test_facade_exports_complete_public_api() -> None:
    """Every documented public name resolves on the facade and is in __all__."""
    missing = [name for name in REQUIRED_FACADE_NAMES if not hasattr(preflight, name)]
    assert missing == [], f"facade is missing public names: {missing}"
    not_exported = [name for name in REQUIRED_FACADE_NAMES if name not in preflight.__all__]
    assert not_exported == [], f"facade __all__ does not declare: {not_exported}"
    for name in preflight.__all__:
        assert hasattr(preflight, name), f"__all__ name does not resolve: {name}"


def test_user_story_happy_path_end_to_end(tmp_path) -> None:
    """The full story through the facade: csv -> validate -> suitability
    listing (probed) -> generate -> edit; every step returns a report."""
    report = validate_data(_write_story_csv(tmp_path))
    assert isinstance(report, DataReport)
    assert report.seq_col == "sequence"
    assert report.splits[0].row_count == 5  # counted, never dropped

    result = check_compatibility(
        report,
        task_type="binary",
        intent="predict",
        prober=_remote_confirmed_prober,
        probe_shortlist=3,
    )
    assert isinstance(result, CompatibilityReport)
    assert result.listing, "predict listing for binary must be non-empty"
    assert result.listing[0].is_lab, "listing is lab-first: entry 0 must be a lab model"
    for entry in result.listing[:3]:
        assert entry.availability is not None, "shortlisted entry must carry availability"
        assert entry.availability.state == AvailabilityState.REMOTE_CONFIRMED

    generated = preflight.generate_config(result.listing[0].model)
    assert isinstance(generated, GeneratedConfig)
    assert generated.errors == []
    task = generated.config["task"]
    assert task.task_type == "binary"
    assert task.label_names, "task section prefilled from the registry entry"

    edited = preflight.edit_config(_raw_config(generated), {"task": {"num_labels": 3}})
    assert isinstance(edited, GeneratedConfig)
    assert edited.errors == []
    assert edited.config["task"].num_labels == 3


def test_user_story_severed_network_still_completes(tmp_path) -> None:
    """SC-5: with the network severed the same story completes —
    availability degrades to unknown with a reason, nothing raises, and
    the network-free steps (generate/edit) are unaffected."""
    report = validate_data(_write_story_csv(tmp_path))
    assert isinstance(report, DataReport)

    result = check_compatibility(
        report,
        task_type="binary",
        intent="predict",
        prober=_severed_prober,
        probe_shortlist=3,
    )
    assert isinstance(result, CompatibilityReport)
    assert result.listing
    for entry in result.listing[:3]:
        assert entry.availability is not None
        assert entry.availability.state == AvailabilityState.UNKNOWN
        assert entry.availability.reason == "PROBE_ERROR"

    generated = preflight.generate_config(result.listing[0].model)
    assert isinstance(generated, GeneratedConfig)
    assert generated.errors == [], "generate_config never touches the network"

    edited = preflight.edit_config(_raw_config(generated), {"task": {"num_labels": 3}})
    assert isinstance(edited, GeneratedConfig)
    assert edited.errors == [], "edit_config never touches the network"


def test_finetune_intent_lists_pretrained_entries_lab_first(tmp_path) -> None:
    """The finetune branch of the story lists pretrained entries only
    (disjoint from the predict branch's head-shipping finetuned entries),
    lab-first."""
    report = validate_data(_write_story_csv(tmp_path))
    predict = check_compatibility(report, task_type="binary", intent="predict")
    finetune = check_compatibility(report, task_type="binary", intent="finetune")

    assert finetune.listing, "finetune listing for binary must be non-empty"
    assert finetune.listing[0].is_lab, "finetune listing is lab-first at entry 0"

    # Lab-first is a prefix property of every listing: all lab entries
    # rank ahead of every non-lab entry (PFC-07).
    for result in (predict, finetune):
        flags = [entry.is_lab for entry in result.listing]
        assert flags == sorted(flags, reverse=True)

    # Base<->finetuned routing: the two intents never list the same model.
    predict_models = {entry.model for entry in predict.listing}
    finetune_models = {entry.model for entry in finetune.listing}
    assert predict_models
    assert predict_models.isdisjoint(finetune_models)

    # Pretrained entries ship no task heads.
    assert all(entry.num_labels is None for entry in finetune.listing)
