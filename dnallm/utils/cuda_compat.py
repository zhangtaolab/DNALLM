"""Preload CUDA 13 shared libraries shipped as pip wheels.

bitsandbytes loads its native binary via ctypes, which resolves CUDA
libraries by SONAME through the OS dynamic linker. The CUDA 13 wheels
(new unified ``nvidia/cu13`` layout) are not on any default linker
search path: they have no ldconfig entry, and torch does not load
``libnvJitLink`` at import time (it is only pulled in lazily through
cusolver). On CUDA 13 builds bitsandbytes therefore fails with::

    OSError: libnvJitLink.so.13: cannot open shared object file

Preloading the wheel-provided libraries with ``RTLD_GLOBAL`` registers
their SONAMEs process-wide before bitsandbytes loads, fixing QLoRA
training and 4-bit inference for CLI, library, and test usage alike.

Like transformers_compat, this module is defensive: it silently no-ops
when bitsandbytes or the CUDA 13 wheels are absent (cpu / cu12 / rocm /
conda-CUDA environments), on platforms without a known layout, or when
an individual library fails to load — importing DNALLM must never break
an otherwise working environment.
"""

import ctypes
import glob
import importlib.util
import os
import sys
import sysconfig

# Patterns are relative to site-packages. ``**`` also matches the legacy
# per-library layout (``nvidia/nvjitlink/lib``) should a distribution
# ship it for CUDA 13.
#
# Linux preloads only libnvJitLink: torch's own import already registers
# cudart/cublas/cublasLt SONAMEs, and nvJitLink is the one library reached
# lazily (via cusolver) and thus missing when bitsandbytes loads.
# Windows DLL search does not offer the same guarantee, so all four are
# preloaded from the wheel ``bin`` directories.
_LIB_PATTERNS = {
    "linux": ("nvidia/**/lib/libnvJitLink.so.13",),
    "win32": (
        "nvidia/**/bin/nvJitLink*.dll",
        "nvidia/**/bin/cudart*.dll",
        "nvidia/**/bin/cublas*.dll",
        "nvidia/**/bin/cublasLt*.dll",
    ),
}

_preloaded = False


def _cuda13_wheels_available():
    """Whether bitsandbytes and at least one CUDA 13 wheel library exist."""
    if importlib.util.find_spec("bitsandbytes") is None:
        return False
    purelib = sysconfig.get_paths()["purelib"]
    for pattern in _LIB_PATTERNS.get(sys.platform, ()):
        if glob.glob(os.path.join(purelib, pattern), recursive=True):
            return True
    return False


def preload_cuda13_libs():
    """RTLD_GLOBAL-load CUDA 13 wheel libraries; no-op when unavailable."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True

    purelib = sysconfig.get_paths()["purelib"]
    # RTLD_GLOBAL is POSIX-only; the mode argument is ignored on Windows.
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for pattern in _LIB_PATTERNS.get(sys.platform, ()):
        for path in sorted(glob.glob(os.path.join(purelib, pattern), recursive=True)):
            try:
                ctypes.CDLL(path, mode=mode)
            except OSError:
                # Wrong arch or corrupt wheel — let bitsandbytes raise its
                # own, more informative error instead.
                pass


# Preload on module import so the SONAMEs are registered before any
# bitsandbytes binary is loaded through DNALLM (mirrors transformers_compat).
if _cuda13_wheels_available():
    preload_cuda13_libs()
