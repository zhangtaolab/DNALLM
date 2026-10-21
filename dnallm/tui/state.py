"""The wizard's single shared state object (25-05).

Screens read and write one :class:`WizardState` instance owned by the app, so a
panel never re-derives a report another panel already computed and back
navigation keeps accumulated work. The report types are imported for typing
only: importing this module never pulls the core's heavy stack in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..preflight import (
        AvailabilityReport,
        CompatibilityReport,
        DataReport,
        GeneratedConfig,
    )

__all__ = ["WizardState"]


@dataclass
class WizardState:
    """Everything the wizard has learned so far.

    Attributes:
        config_path: The config file named with ``--config``. It is stored but
            never opened here; the config panel is the only reader, and it
            loads the file through the pre-flight safe-loader path.
        data_report: The frozen report ``validate_data`` returned, once the
            data panel has run it.
        compatibility_report: The frozen report ``check_compatibility``
            returned, once the suitability panel has run it.
        availability_by_entry: Per-model availability already known, keyed by
            registry entry name. Rendering only — a row that is present is
            never re-probed.
        selected_entry: The model the user chose in the suitability panel.
        generated_config: The validated config the config panel produced.
        export_path: Where the generated config was written, when it was saved.
    """

    config_path: str | None = None
    data_report: DataReport | None = None
    compatibility_report: CompatibilityReport | None = None
    availability_by_entry: dict[str, AvailabilityReport] = field(default_factory=dict)
    selected_entry: str | None = None
    generated_config: GeneratedConfig | None = None
    export_path: str | None = None
