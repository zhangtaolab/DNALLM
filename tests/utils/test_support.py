"""Tests for the hardware capability helpers (dnallm.utils.support)."""

import sys
import types

from dnallm.utils import support


class TestIsFlashAttentionCapable:
    """The flash_attn availability probe."""

    def test_installed_flash_attn_reports_capable(self, monkeypatch):
        """An importable flash_attn module reports True."""
        fake_flash = types.ModuleType("flash_attn")
        monkeypatch.setitem(sys.modules, "flash_attn", fake_flash)

        assert support.is_flash_attention_capable() is True

    def test_missing_flash_attn_reports_incapable(self, monkeypatch, caplog):
        """An unimportable flash_attn reports False with a warning."""
        monkeypatch.setitem(sys.modules, "flash_attn", None)

        assert support.is_flash_attention_capable() is False
        assert "Cannot find supported Flash Attention" in caplog.text


class TestIsFp8Capable:
    """The CUDA compute-capability FP8 probe (device capability is patched)."""

    def test_hopper_or_newer_reports_capable(self, monkeypatch):
        """Compute capability 9.0+ reports True."""
        monkeypatch.setattr(support, "get_device_capability", lambda: (9, 0))
        assert support.is_fp8_capable() is True

    def test_older_device_reports_incapable(self, monkeypatch, caplog):
        """Compute capability below 9.0 reports False with a warning."""
        monkeypatch.setattr(support, "get_device_capability", lambda: (8, 6))

        assert support.is_fp8_capable() is False
        assert "does not support FP8" in caplog.text
