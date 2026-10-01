"""Tests for the CUDA 13 library preloader (dnallm.utils.cuda_compat)."""

import ctypes
import sys
from unittest.mock import Mock

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


def test_wheels_unavailable_without_bitsandbytes(monkeypatch):
    """A missing bitsandbytes package must report unavailable before globbing."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name, package=None: None)
    assert cuda_compat._cuda13_wheels_available() is False


def test_wheels_unavailable_when_no_library_matches(monkeypatch):
    """Installed bitsandbytes plus zero matching wheel paths is unavailable."""
    monkeypatch.setattr("glob.glob", lambda *args, **kwargs: [])
    assert cuda_compat._cuda13_wheels_available() is False


def test_preload_tolerates_library_load_failure(monkeypatch):
    """A failing CDLL load must be swallowed and mark the preload as done."""
    monkeypatch.setattr(cuda_compat, "_preloaded", False)
    monkeypatch.setattr("glob.glob", lambda *args, **kwargs: ["/fake/libnvJitLink.so.13"])
    failing = Mock(side_effect=OSError("cannot open shared object file"))
    monkeypatch.setattr("ctypes.CDLL", failing)

    cuda_compat.preload_cuda13_libs()  # must not raise

    assert cuda_compat._preloaded is True
    # One CDLL attempt per wheel pattern for this platform (linux has a single
    # SONAME pattern; win32 preloads four DLL families).
    assert failing.call_count == len(cuda_compat._LIB_PATTERNS[sys.platform])
    assert failing.call_args.args[0] == "/fake/libnvJitLink.so.13"
