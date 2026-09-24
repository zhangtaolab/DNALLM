"""Tests for the CUDA 13 library preloader (dnallm.utils.cuda_compat)."""

import ctypes
import sys

import pytest

from dnallm.utils import cuda_compat


def test_preload_is_idempotent():
    """Repeated calls must be safe no-ops."""
    cuda_compat.preload_cuda13_libs()
    cuda_compat.preload_cuda13_libs()
    assert cuda_compat._preloaded is True


def test_preload_noops_without_cuda13_wheels(monkeypatch):
    """On cpu/cu12/rocm environments nothing must load or raise."""
    monkeypatch.setattr(cuda_compat, "_preloaded", False)
    monkeypatch.setattr(cuda_compat, "_cuda13_wheels_available", lambda: False)
    cuda_compat.preload_cuda13_libs()
    assert cuda_compat._preloaded is True


@pytest.mark.skipif(sys.platform != "linux", reason="SONAME check is Linux-specific")
def test_nvjitlink_soname_registered_on_cuda13(monkeypatch):
    """On a CUDA 13 environment the preload must register the SONAME."""
    monkeypatch.setattr(cuda_compat, "_preloaded", False)
    if not cuda_compat._cuda13_wheels_available():
        pytest.skip("CUDA 13 wheels not installed in this environment")
    cuda_compat.preload_cuda13_libs()
    # Would raise OSError before the fix on CUDA 13 builds.
    ctypes.CDLL("libnvJitLink.so.13")
