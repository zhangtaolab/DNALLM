"""Contract tests for the transformers device-type-query compatibility rung.

transformers >= 5.19 (freshly resolved by the open ``>=4.49,<6`` range on
remote CI) eagerly imports its flex_attention integration inside
modeling_utils, whose availability check queries the accelerator device and
raises ``RuntimeError("Cannot access accelerator device when none is
available")`` when a CUDA-built torch runs on a GPU-less machine -- the
test-cuda CI legs fail every collecting module that imports dnallm. The
rung under test wraps ``transformers.utils.import_utils.get_device_type``
so the query answers the honest ``"cpu"`` instead of raising, gated to stay
inert everywhere else. The tests below hold the rung to the same three-part
discipline as the other compat shims: absence gate, idempotency sentinel,
no-op when the native query already works. Every install/no-op branch runs
against a fake ``transformers.utils.import_utils`` module so local proof
never depends on reproducing the GPU-less CUDA environment.
"""

import pathlib
import sys
import types

import pytest

import dnallm
import transformers.utils
from dnallm.utils import transformers_compat

_ERROR_MESSAGE = "Cannot access accelerator device when none is available"


def _patch_fn():
    """Resolve the installer lazily so a missing rung fails the test, not collection."""
    return transformers_compat._patch_device_type_query


def _utils_package():
    """The ``transformers.utils`` package the installer resolves at call time.

    ``sys.modules`` is the canonical entry: the top-level ``transformers``
    parent is a ``_LazyModule`` that re-imports can swap, but its ``utils``
    attribute always resolves to this one package object.
    """
    return sys.modules["transformers.utils"]


def _install_fake_import_utils(monkeypatch, original):
    """Install *original* as a fake import_utils module in both resolution spots.

    The installer imports ``transformers.utils.import_utils`` and then reads
    the module back through the parent-package attribute, so the fake must
    sit in ``sys.modules`` AND on the ``transformers.utils`` package to be
    resolved deterministically on every host (GPU or CPU, transformers
    5.17 or 5.19+).
    """
    fake = types.ModuleType("transformers.utils.import_utils")
    fake.get_device_type = original
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake)
    monkeypatch.setattr(_utils_package(), "import_utils", fake)
    return fake


class TestDeviceTypeQueryShim:
    """Probe-gated get_device_type RuntimeError-to-"cpu" wrapper (quick 261007-mxl)."""

    def test_installer_exists_and_registered_before_modeling_rungs(self):
        """The rung exists and apply_patches() registers it before any modeling_utils rung."""
        assert callable(_patch_fn())
        apply_src = transformers_compat.apply_patches.__code__.co_names
        assert "_patch_device_type_query" in apply_src
        assert apply_src.index("_patch_device_type_query") < apply_src.index(
            "_patch_get_parameter_or_buffer"
        ), "the device-query rung must run before the first modeling_utils-importing rung"

    def test_patch_noops_when_get_device_type_absent(self, monkeypatch):
        """transformers <= 5.17 (get_device_type exists nowhere) is left untouched."""
        fake = types.ModuleType("transformers.utils.import_utils")
        monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake)
        monkeypatch.setattr(_utils_package(), "import_utils", fake)

        _patch_fn()()

        assert not hasattr(fake, "get_device_type")
        assert getattr(fake, "_dnallm_device_type_patch", False) is False

    def test_patch_noops_when_native_query_succeeds(self, monkeypatch):
        """A working native query (GPU machine, CPU-only torch wheel) is never wrapped."""

        def working_query():
            return "cuda"

        fake = _install_fake_import_utils(monkeypatch, working_query)

        _patch_fn()()

        assert fake.get_device_type is working_query
        assert getattr(fake, "_dnallm_device_type_patch", False) is False

    def test_patch_installs_when_probe_raises_runtime_error(self, monkeypatch):
        """A CUDA-built torch without a visible GPU gets the honest "cpu" wrapper."""

        def deviceless_query():
            raise RuntimeError(_ERROR_MESSAGE)

        fake = _install_fake_import_utils(monkeypatch, deviceless_query)

        _patch_fn()()

        assert fake._dnallm_device_type_patch is True
        installed = fake.get_device_type
        assert installed is not deviceless_query
        assert installed() == "cpu"
        # The wrapper catches per call (a permanently device-less machine);
        # the underlying stub it delegates to still raises untouched.
        with pytest.raises(RuntimeError, match=_ERROR_MESSAGE):
            deviceless_query()

    def test_installed_wrapper_delegates_and_forwards_arguments(self, monkeypatch):
        """After the one-time probe failure the wrapper delegates with full passthrough."""
        calls = []

        def flaky_query(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise RuntimeError(_ERROR_MESSAGE)  # the probe call only
            return "cuda"

        fake = _install_fake_import_utils(monkeypatch, flaky_query)

        _patch_fn()()

        result = fake.get_device_type("positional", keyword=1)
        assert result == "cuda"
        assert calls == [((), {}), (("positional",), {"keyword": 1})]

    def test_patch_is_idempotent_via_sentinel(self, monkeypatch):
        """A second install call must not rebind the wrapper."""

        def deviceless_query():
            raise RuntimeError(_ERROR_MESSAGE)

        fake = _install_fake_import_utils(monkeypatch, deviceless_query)

        _patch_fn()()
        first = fake.get_device_type
        _patch_fn()()

        assert fake.get_device_type is first

    def test_unknown_probe_failures_stay_loud(self, monkeypatch):
        """Only the observed RuntimeError signature is answered; anything else raises."""

        def broken_query():
            raise ValueError("unknown device-query failure")

        fake = _install_fake_import_utils(monkeypatch, broken_query)

        with pytest.raises(ValueError, match="unknown device-query failure"):
            _patch_fn()()

        assert fake.get_device_type is broken_query
        assert getattr(fake, "_dnallm_device_type_patch", False) is False

    def test_reexport_mirror_updates_identical_reference(self, monkeypatch):
        """A transformers.utils re-export holding the exact original is mirrored to the wrapper."""

        def deviceless_query():
            raise RuntimeError(_ERROR_MESSAGE)

        fake = _install_fake_import_utils(monkeypatch, deviceless_query)
        monkeypatch.setattr(_utils_package(), "get_device_type", deviceless_query, raising=False)

        _patch_fn()()

        assert _utils_package().get_device_type is fake.get_device_type

    def test_reexport_mirror_leaves_divergent_reference_untouched(self, monkeypatch):
        """A re-export already holding a different object is never stamped over."""

        def deviceless_query():
            raise RuntimeError(_ERROR_MESSAGE)

        divergent = object()
        fake = _install_fake_import_utils(monkeypatch, deviceless_query)
        monkeypatch.setattr(_utils_package(), "get_device_type", divergent, raising=False)

        _patch_fn()()

        assert _utils_package().get_device_type is divergent
        assert fake.get_device_type is not divergent


class TestRootInitOrdering:
    """The root package must load dnallm.utils before dnallm.models."""

    def test_utils_imports_before_models_in_root_init(self):
        """The root __init__ orders the .utils import before the .models import.

        transformers_compat.apply_patches() must run ahead of the first
        "from transformers import" modeling-symbol resolution (dnallm.models
        model.py), because transformers >= 5.19 queries the accelerator
        during that import and raises on CUDA-built torch without a visible
        GPU -- the test-cuda CI legs.
        """
        source = pathlib.Path(dnallm.__file__).read_text(encoding="utf-8")
        utils_idx = source.find("from .utils import")
        models_idx = source.find("from .models import")
        assert utils_idx != -1, "dnallm/__init__.py must keep its 'from .utils import' line"
        assert models_idx != -1, "dnallm/__init__.py must keep its 'from .models import' line"
        assert utils_idx < models_idx, (
            "dnallm/__init__.py must import .utils BEFORE .models: transformers_compat "
            "installs the pre-modeling device-query patch at .utils import time, while "
            "dnallm.models resolves transformers modeling_utils first (the >= 5.19 "
            "accelerator query crashes GPU-less CUDA torch during collection)"
        )
