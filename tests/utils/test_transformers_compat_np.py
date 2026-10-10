"""Tests for the numpy ``fromstring`` compatibility rung.

``np.fromstring`` was removed in numpy 2.0 (binary mode deprecated since
1.14 with "use frombuffer instead"). stripedhyena's ``CharLevelTokenizer``
(the evo-1 family tokenizer) still calls it, so ``dnallm`` installs an
absence-gated shim that restores the historical binary-mode behavior. The
tests below cover the surviving rungs: absence gate (fallback install),
raising-stub replacement, idempotency sentinel, and missing-module no-op;
the numpy 1.x no-op rung (native ``fromstring`` left untouched) retired with
the ``>=2.0.0`` floor (2026-10-10).
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

    def test_patch_installs_fallback_when_absent(self, monkeypatch):
        """numpy 2.x (no fromstring) gets the vendored fallback attached."""
        fake = types.ModuleType("numpy")
        fake.frombuffer = np.frombuffer  # the real implementation
        monkeypatch.setitem(sys.modules, "numpy", fake)
        assert not hasattr(fake, "fromstring")

        _patch_fn()()

        assert callable(fake.fromstring)
        assert fake._dnallm_fromstring_patch is True

    def test_patch_replaces_raising_numpy2_stub(self, monkeypatch):
        """The numpy 2.x raising stub is treated as absent and replaced.

        numpy 2.x keeps the NAME ``fromstring`` as a stub that raises on
        every call, so a mere hasattr gate would no-op on exactly the
        versions that need the shim -- the probe must call it.
        """

        def raising_stub(string, dtype=float, count=-1, sep=""):
            raise ValueError("The binary mode of fromstring is removed, use frombuffer instead")

        fake = types.ModuleType("numpy")
        fake.fromstring = raising_stub
        fake.uint8 = np.uint8
        fake.frombuffer = np.frombuffer
        fake.array = np.array
        monkeypatch.setitem(sys.modules, "numpy", fake)

        _patch_fn()()

        assert fake.fromstring is transformers_compat._np_fromstring
        assert fake.fromstring(b"ACGT", dtype=fake.uint8).tolist() == [65, 67, 71, 84]

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
    """The vendored fallback parses bytes and str like historical numpy."""

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

    def test_binary_mode_str_count_is_honored(self):
        """The count argument truncates str input exactly like bytes input."""
        fromstring = self._fallback()
        result = fromstring("ACGTACGT", dtype=np.uint8, count=4)
        assert result.tolist() == [65, 67, 71, 84]

    def test_binary_mode_str_result_is_writable(self):
        """str input also yields the historical writable copy, not a view."""
        fromstring = self._fallback()
        result = fromstring("ACGT", dtype=np.uint8)
        result[0] = 84
        assert result[0] == 84

    def test_binary_mode_str_non_ascii_encodes_utf8(self):
        """Non-ASCII sanity pin: the documented encode choice is utf-8.

        Outside the stripedhyena ASCII vocab the shim's contract is its own
        utf-8 encode (two bytes for U+00E9), asserted here so a future
        switch to latin-1 must be a deliberate comment+test change.
        """
        fromstring = self._fallback()
        result = fromstring("é", dtype=np.uint8)
        assert result.tolist() == [195, 169]

    def test_text_mode_raises_instructive_error(self):
        """sep != '' (text mode) is refused with a pointer to loadtxt."""
        fromstring = self._fallback()
        with pytest.raises(ValueError, match="loadtxt"):
            fromstring("1,2,3", sep=",")
