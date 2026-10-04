"""Tests for the numpy ``fromstring`` compatibility rung.

``np.fromstring`` was removed in numpy 2.0 (binary mode deprecated since
1.14 with "use frombuffer instead"). stripedhyena's ``CharLevelTokenizer``
(the evo-1 family tokenizer) still calls it, so ``dnallm`` installs an
absence-gated shim that restores the historical binary-mode behavior. The
tests below hold the rung to the same three-part discipline as the nine
transformers shims: absence gate, idempotency sentinel, no-op when the
library already provides the API.
"""

import sys
import types

import numpy as np
import pytest

from dnallm.utils import transformers_compat


def _patch_fn():
    """Resolve the installer lazily so a missing rung fails the test, not collection."""
    return transformers_compat._patch_numpy_fromstring


class TestNumpyFromstringShim:
    """Absence-gated np.fromstring restore (the numpy rung, 08-03)."""

    def test_installer_exists_and_registered(self):
        """The rung exists and apply_patches() lists it."""
        assert callable(_patch_fn())
        apply_src = transformers_compat.apply_patches.__code__.co_names
        assert "_patch_numpy_fromstring" in apply_src

    def test_patch_noops_when_numpy_provides_fromstring(self, monkeypatch):
        """numpy 1.x (has fromstring) must be left byte-identically untouched."""

        def original(*args, **kwargs):
            return None

        fake = types.ModuleType("numpy")
        fake.fromstring = original
        monkeypatch.setitem(sys.modules, "numpy", fake)

        _patch_fn()()

        assert fake.fromstring is original
        assert getattr(fake, "_dnallm_fromstring_patch", False) is False

    def test_patch_installs_fallback_when_absent(self, monkeypatch):
        """numpy 2.x (no fromstring) gets the vendored fallback attached."""
        fake = types.ModuleType("numpy")
        fake.frombuffer = np.frombuffer  # the real implementation
        monkeypatch.setitem(sys.modules, "numpy", fake)
        assert not hasattr(fake, "fromstring")

        _patch_fn()()

        assert callable(fake.fromstring)
        assert fake._dnallm_fromstring_patch is True

    def test_patch_is_idempotent_via_sentinel(self, monkeypatch):
        """A second install call must not rebind the fallback."""
        fake = types.ModuleType("numpy")
        fake.frombuffer = np.frombuffer
        monkeypatch.setitem(sys.modules, "numpy", fake)

        _patch_fn()()
        first = fake.fromstring
        _patch_fn()()

        assert fake.fromstring is first

    def test_patch_noops_when_numpy_missing(self, monkeypatch):
        """No numpy at all -> plain None return, never a raise out of import."""
        monkeypatch.setitem(sys.modules, "numpy", None)
        assert _patch_fn()() is None


class TestNumpyFromstringFallbackBehavior:
    """The vendored fallback parses byte strings like historical numpy."""

    def _fallback(self):
        """The vendored fallback, resolved directly from the compat module."""
        return transformers_compat._np_fromstring

    def test_binary_mode_uint8_matches_historical_behavior(self):
        """np.fromstring(b'ACGT', dtype=np.uint8) == [65, 67, 71, 84]."""
        fromstring = self._fallback()
        result = fromstring(b"ACGT", dtype=np.uint8)
        assert result.tolist() == [65, 67, 71, 84]
        assert result.dtype == np.uint8

    def test_binary_mode_count_is_honored(self):
        """The count argument truncates exactly like the historical API."""
        fromstring = self._fallback()
        result = fromstring(b"ACGTACGT", dtype=np.uint8, count=4)
        assert result.tolist() == [65, 67, 71, 84]

    def test_binary_mode_result_is_writable(self):
        """Historical fromstring returned a writable copy, not a buffer view."""
        fromstring = self._fallback()
        result = fromstring(b"ACGT", dtype=np.uint8)
        result[0] = 84
        assert result[0] == 84

    def test_live_numpy_has_fromstring_after_import(self):
        """After apply_patches() the live numpy answers fromstring (shim or native)."""
        transformers_compat.apply_patches()
        assert hasattr(np, "fromstring")
        assert np.fromstring(b"ACGT", dtype=np.uint8).tolist() == [65, 67, 71, 84]

    def test_text_mode_raises_instructive_error(self):
        """sep != '' (text mode) is refused with a pointer to loadtxt."""
        fromstring = self._fallback()
        with pytest.raises(ValueError, match="loadtxt"):
            fromstring("1,2,3", sep=",")
