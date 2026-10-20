"""Contract tests for the legacy tie_weights arity compatibility rung.

transformers 5.x calls ``tie_weights`` with kwargs (``recompute_mapping``
from init_weights; ``missing_keys`` + ``recompute_mapping`` after
from_pretrained weight loading). A remote-code class overriding
``tie_weights`` with the older zero-kwarg signature — first seen live on
lgq12697/PlantBiMoE's ``PlantbimoeForMaskedLM.tie_weights(self)`` — dies
with a TypeError inside ``post_init`` before any weights load. The rung
under test (``_patch_legacy_tie_weights_arity`` + ``_bridge_legacy_tie_weights``)
installs a per-instance forwarder that strips the 5.x-only kwargs and
delegates to the legacy override bare. The tests hold the rung to the same
discipline as the other compat shims: bridged legacy overrides construct
and tie under post_init, native 5.x signatures are never shadowed, the
named-kwargs a legacy signature does declare still pass through, and the
installer is idempotent.
"""

import inspect
from typing import ClassVar

import pytest
import torch
from transformers import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel

import dnallm  # applies transformers_compat at import (side effect)
from dnallm.utils import transformers_compat


class _LegacyTieModel(PreTrainedModel):
    """Remote-code stand-in: the 4.x zero-kwarg tie_weights override."""

    config_class = PreTrainedConfig

    calls: ClassVar[list[dict]] = []

    def __init__(self, config=None):
        super().__init__(config or PreTrainedConfig())
        # Reset on the class (not self): tie_weights logs through
        # type(self).calls, and an instance attr would shadow it.
        type(self).calls = []

    def _init_weights(self, module):
        pass

    def tie_weights(self):
        type(self).calls.append({"args": (), "kwargs": {}})


class _NamedKwargLegacyModel(_LegacyTieModel):
    """Legacy override that still declares one pre-5.x named parameter."""

    def tie_weights(self, caller=None):
        type(self).calls.append({"args": (), "kwargs": {"caller": caller}})


class _NativeWideModel(_LegacyTieModel):
    """Remote-code stand-in that already absorbed the 5.x protocol."""

    def tie_weights(self, **kwargs):
        type(self).calls.append({"args": (), "kwargs": dict(kwargs)})


def test_patch_is_installed_with_base_taking_the_kwarg():
    """On this transformers the rung must be installed (base names recompute_mapping)."""
    if not transformers_compat._base_tie_weights_takes_recompute_mapping():
        pytest.skip("installed transformers predates the 5.x tie protocol")
    assert getattr(PreTrainedModel, "_dnallm_tie_weights_arity_patch", False)


def test_legacy_override_constructs_and_ties_bare():
    """post_init on a zero-kwarg override constructs instead of TypeError-ing."""
    model = _LegacyTieModel()
    model.post_init()
    assert model.calls, "legacy tie_weights must have been reached"
    assert model.calls[-1] == {"args": (), "kwargs": {}}


def test_bridge_drops_only_the_5x_kwargs():
    """The bridged instance accepts the 5.x call sites' kwargs, passing named ones through."""
    model = _NamedKwargLegacyModel()
    model.post_init()
    model.tie_weights(caller="from_pretrained", recompute_mapping=False, missing_keys=None)
    assert model.calls[-1] == {"args": (), "kwargs": {"caller": "from_pretrained"}}


def test_native_wide_signature_is_not_shadowed():
    """A **kwargs override is left on the class — no instance forwarder installed."""
    model = _NativeWideModel()
    model.post_init()
    assert "tie_weights" not in model.__dict__, "native wide override must stay untouched"
    model.tie_weights(recompute_mapping=False)
    assert model.calls[-1]["kwargs"] == {"recompute_mapping": False}


def test_bridge_is_self_guarding():
    """Re-running the bridge over an already-bridged instance does not stack."""
    model = _LegacyTieModel()
    transformers_compat._bridge_legacy_tie_weights(model)
    first = model.tie_weights
    transformers_compat._bridge_legacy_tie_weights(model)
    assert model.tie_weights is first


def test_installer_is_idempotent():
    """A second installer call does not wrap init_weights again."""
    before = vars(PreTrainedModel).get("init_weights")
    transformers_compat._patch_legacy_tie_weights_arity()
    assert vars(PreTrainedModel).get("init_weights") is before


def test_real_transformers_base_signature_is_untouched():
    """The patch never rewrites the base class's own tie_weights."""
    params = inspect.signature(PreTrainedModel.tie_weights).parameters
    assert "self" in params
    if transformers_compat._base_tie_weights_takes_recompute_mapping():
        assert "recompute_mapping" in params
