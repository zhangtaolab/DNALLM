#!/bin/bash
#
# install_mamba.sh — install the mamba-ssm/causal_conv1d CUDA kernels by
# building mamba-ssm from a sha256-pinned sdist with the four
# -std=c++17 -> -std=c++20 setup.py edits (upstream PR #1000 equivalent).
#
# WHY: torch 2.14 raises ATen's C++ standard floor to C++20, while
# mamba-ssm 2.3.2.post1's setup.py hardcodes -std=c++17 in its HIP and CUDA
# branches — the kernel compile dies on an #error under modern torch
# (evidence: run 40140203118 on the GB10 runner). Upstream has shipped no
# fixed release (latest = 2.3.2.post1, May 2026) and no wheels for
# torch >= 2.11 on any architecture (release wheel matrix stops at torch
# 2.10), so the old guessed-GitHub-release-wheel path can never serve this
# box on modern torch. PR #1000 flips the same four flags but remains
# unmerged.
#
# WHAT: fetch the mamba-ssm sdist file directly from a package index's
# simple page (default PyPI; override with --index-url, e.g. the TUNA
# mirror https://pypi.tuna.tsinghua.edu.cn/simple), verify its sha256
# against the pinned constant below BEFORE anything is extracted or
# executed, flip the four -std flags in the extracted tree, build the
# wheel from the patched tree, build causal_conv1d 1.7.0 unpatched from
# the same index (its CUDA branch omits -std entirely, so BuildExtension
# picks the standard and it compiles green as-is), then install both
# wheels. The fetch deliberately avoids pip's resolution machinery: pip
# builds sdist metadata by EXECUTING setup.py from the downloaded archive,
# which would run unverified bytes before the hash gate.
#
# SUPPLY CHAIN: the sha256 pin is the byte-integrity control, which makes
# ANY index route safe — PyPI itself or a mirror cannot serve altered bytes
# without the gate failing closed. There is deliberately no flag or
# environment variable that overrides the hash constant. GitHub access is
# NOT required at any point in this pipeline.
#
# Usage:
#   scripts/install_mamba.sh [--default] [--patch-only] [--verify-file PATH]
#       [--index-url URL] [--python PATH] [--wheelhouse DIR] [--no-install]
#
#   --default          accepted for backward compatibility; no effect (this
#                      pipeline has no interactive prompts)
#   --patch-only       fetch + verify + patch + count assertions, then stop
#                      before any build; needs only the python stdlib plus
#                      tar/sha256sum — works on boxes without torch, CUDA,
#                      or pip
#   --verify-file PATH sha256-check a local file against the pinned constant
#                      and exit 0/1 (no download, no build, no python)
#   --index-url URL    package index (default https://pypi.org/simple)
#   --python PATH      interpreter to build/install with (default: python)
#   --wheelhouse DIR   directory for the built wheels (default: fresh temp
#                      directory)
#   --no-install       build the wheels but skip installing them

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Pinned constants ------------------------------------------------------
# A version bump MUST re-verify the sha256 constant and the 4-flag patch
# shape together; --patch-only re-runs both checks cheaply.
MAMBA_SSM_VERSION="2.3.2.post1"
CAUSAL_CONV1D_VERSION="1.7.0"
MAMBA_SDIST_SHA256="104cc47e9101e5401a675fa2b784f2952b9b037f3b1dd83b5ac544394e95d028"

# --- Helper functions for colored output ---
info() {
    echo -e "\033[0;32m[INFO] $1\033[0m"
}

warn() {
    echo -e "\033[0;33m[WARN] $1\033[0m"
}

error() {
    echo -e "\033[0;31m[ERROR] $1\033[0m"
    exit 1
}

# --- Argument parsing ---
PYTHON_BIN="python"
INDEX_URL="https://pypi.org/simple"
WHEELHOUSE_DIR=""
PATCH_ONLY=false
NO_INSTALL=false
VERIFY_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --default)
            info "--default accepted for backward compatibility; this pipeline is fully non-interactive."
            ;;
        --patch-only)
            PATCH_ONLY=true
            ;;
        --verify-file)
            [ $# -ge 2 ] || error "--verify-file requires a path argument."
            VERIFY_FILE="$2"
            shift
            ;;
        --index-url)
            [ $# -ge 2 ] || error "--index-url requires a URL argument."
            INDEX_URL="$2"
            shift
            ;;
        --python)
            [ $# -ge 2 ] || error "--python requires an interpreter path."
            PYTHON_BIN="$2"
            shift
            ;;
        --wheelhouse)
            [ $# -ge 2 ] || error "--wheelhouse requires a directory argument."
            WHEELHOUSE_DIR="$2"
            shift
            ;;
        --no-install)
            NO_INSTALL=true
            ;;
        *)
            error "Unknown option: $1 (see the usage comment at the top of this script)."
            ;;
    esac
    shift
done

# --- --verify-file mode: pure hash check against the pinned constant ------
# Needs no python, pip, or network; exits 0 on match, 1 on mismatch.
if [ -n "$VERIFY_FILE" ]; then
    [ -f "$VERIFY_FILE" ] || error "No such file: $VERIFY_FILE"
    ACTUAL_SHA=$(sha256sum "$VERIFY_FILE" | awk '{print $1}')
    if [ "$ACTUAL_SHA" = "$MAMBA_SDIST_SHA256" ]; then
        info "verify-file: $VERIFY_FILE matches the pinned mamba-ssm $MAMBA_SSM_VERSION sdist sha256."
        exit 0
    fi
    error "verify-file: sha256 MISMATCH for $VERIFY_FILE
  expected: $MAMBA_SDIST_SHA256
  actual:   $ACTUAL_SHA"
fi

# --- Interpreter availability ----------------------------------------------
# The fetch below needs only the interpreter itself (stdlib); pip is not
# required until the build steps and is checked there with its own
# bootstrap hint.
"$PYTHON_BIN" --version >/dev/null 2>&1 \
    || error "Python interpreter '$PYTHON_BIN' not found. Pass --python /path/to/python (a venv interpreter is fine)."

# --- Environment check: build path only ------------------------------------
# Informational logging plus the torch-presence guard. --patch-only skips
# this so the patch verification runs on boxes without torch/CUDA.
if [ "$PATCH_ONLY" = false ]; then
    info "Starting environment check..."
    TORCH_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
    if [ -z "$TORCH_VERSION" ]; then
        error "PyTorch is not importable by '$PYTHON_BIN'. The kernel build requires a CUDA-enabled PyTorch to be installed first."
    fi
    CUDA_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || true)
    if [ -z "$CUDA_VERSION" ]; then
        error "No CUDA runtime detected from PyTorch. The mamba-ssm/causal_conv1d kernels require a CUDA-enabled PyTorch build."
    fi
    info "Detected PyTorch: $TORCH_VERSION (CUDA $CUDA_VERSION)"
    # TORCH_CUDA_ARCH_LIST is deliberately NOT set here: the pinned sdist's
    # setup.py embeds its own gencode list (compute_121/sm_121 included) and
    # the build box auto-detects; a caller-exported value passes through
    # untouched.
fi

# --- 1. Fetch the sdist directly from the index -----------------------------
# Resolves the pinned filename on the index's PEP 503 simple page, then
# downloads the file bytes. As an early cross-check, the sha256 fragment
# the index declares in the file link is compared against the pinned
# constant BEFORE a single byte is downloaded (mismatch fails closed with
# the route named); the authoritative gate is the local sha256sum below.
# Nothing from the archive executes here — that is why this is NOT a pip
# resolution command (pip executes setup.py for sdist metadata).
fetch_sdist() { # fetch_sdist INDEX_URL SDIST_NAME PINNED_SHA DEST
    "$PYTHON_BIN" - "$1" "$2" "$3" "$4" <<'PY'
import re
import sys
from urllib.parse import urljoin
from urllib.request import urlopen

index_url, sdist_name, pinned_sha, dest = sys.argv[1:5]
page_url = index_url.rstrip("/") + "/mamba-ssm/"
with urlopen(page_url, timeout=120) as resp:
    html = resp.read().decode("utf-8", "replace")
target = None
for href, text in re.findall(r'<a\s[^>]*href="([^"]+)"[^>]*>([^<]*)</a>', html):
    if text.strip() == sdist_name:
        target = href
        break
if target is None:
    sys.exit(f"error: {sdist_name} is not listed on {page_url}")
if "#sha256=" in target:
    target, declared = target.split("#sha256=", 1)
    declared = declared.split("&", 1)[0].strip()
    if declared and declared != pinned_sha:
        sys.exit(f"error: index-declared sha256 {declared} differs from the pinned constant for {sdist_name}")
url = urljoin(page_url, target)
with urlopen(url, timeout=600) as resp, open(dest, "wb") as out:
    while True:
        chunk = resp.read(1 << 20)
        if not chunk:
            break
        out.write(chunk)
print(f"fetched {sdist_name} from {url}")
PY
}

SDIST_NAME="mamba_ssm-${MAMBA_SSM_VERSION}.tar.gz"
DOWNLOAD_DIR="$(mktemp -d)"
info "Fetching $SDIST_NAME from $INDEX_URL ..."
fetch_sdist "$INDEX_URL" "$SDIST_NAME" "$MAMBA_SDIST_SHA256" "${DOWNLOAD_DIR}/${SDIST_NAME}" \
    || error "Fetch of $SDIST_NAME from $INDEX_URL failed (route unreachable, file not listed, or the index-declared sha256 differs from the pinned constant)."
SDIST_PATH="${DOWNLOAD_DIR}/${SDIST_NAME}"
[ -f "$SDIST_PATH" ] || error "Fetch did not produce $SDIST_NAME in $DOWNLOAD_DIR."

# --- 2. Hash gate: fail closed BEFORE any extraction, patch, or build ------
info "Verifying sha256 of $SDIST_NAME ..."
ACTUAL_SHA=$(sha256sum "$SDIST_PATH" | awk '{print $1}')
if [ "$ACTUAL_SHA" != "$MAMBA_SDIST_SHA256" ]; then
    error "sha256 MISMATCH for $SDIST_NAME — refusing to extract, patch, or build.
  expected: $MAMBA_SDIST_SHA256
  actual:   $ACTUAL_SHA
The route ($INDEX_URL) served bytes that differ from the pinned sdist. This gate fails closed; there is no override."
fi
info "sha256 verified: $ACTUAL_SHA"

# --- 3. Extract into a fresh empty directory --------------------------------
# Later steps receive the extracted directory path as an argument; nothing
# ever runs from inside the extracted tree.
EXTRACT_DIR="$(mktemp -d)"
info "Extracting $SDIST_NAME into $EXTRACT_DIR ..."
tar -xzf "$SDIST_PATH" -C "$EXTRACT_DIR"
SRC_DIR="${EXTRACT_DIR}/mamba_ssm-${MAMBA_SSM_VERSION}"
SETUP_PY="${SRC_DIR}/setup.py"
[ -f "$SETUP_PY" ] || error "Extracted tree does not contain mamba_ssm-${MAMBA_SSM_VERSION}/setup.py."

# --- 4. Patch: -std=c++17 -> -std=c++20 with occurrence-exact assertions ---
# Exactly 4 before, exactly 0 c++17 after, exactly 4 c++20 after (counted
# per occurrence, not per line). Any other count is a hard error: a future
# version bump that re-pins the hash but changes the patch shape must fail
# loudly, never silently build unpatched.
info "Patching -std=c++17 -> -std=c++20 in $SETUP_PY ..."
BEFORE_COUNT=$(( $(grep -o -- '-std=c++17' "$SETUP_PY" | wc -l) ))
if [ "$BEFORE_COUNT" -ne 4 ]; then
    error "Expected exactly 4 '-std=c++17' occurrences in setup.py, found $BEFORE_COUNT — patch shape changed; refusing to build."
fi
sed -i 's/-std=c++17/-std=c++20/g' "$SETUP_PY"
REMAIN_COUNT=$(( $(grep -o -- '-std=c++17' "$SETUP_PY" | wc -l) ))
CXX20_COUNT=$(( $(grep -o -- '-std=c++20' "$SETUP_PY" | wc -l) ))
if [ "$REMAIN_COUNT" -ne 0 ] || [ "$CXX20_COUNT" -ne 4 ]; then
    error "Patch assertions failed: -std=c++17 remaining: $REMAIN_COUNT (want 0), -std=c++20 total: $CXX20_COUNT (want 4)."
fi
info "Patch applied: 4/4 flags flipped, 0 c++17 remain."

if [ "$PATCH_ONLY" = true ]; then
    echo "PATCH VERIFIED: sha256 ok, 4/4 flags flipped, 0 remain"
    exit 0
fi

# --- 5. Build the patched mamba_ssm wheel -----------------------------------
# pip enters the pipeline here (the fetch never used it). uv-created venvs
# ship without pip, hence the bootstrap hint.
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    error "pip is not importable by '$PYTHON_BIN'. Bootstrap it first, e.g.:
  uv pip install --python <venv> pip      (uv lane)
  $PYTHON_BIN -m ensurepip                (plain venv lane)"
fi
# Built from the PATCHED extracted directory, never from the tarball path
# (pip wheel on the tarball rebuilds the unpatched archive). MAMBA_FORCE_BUILD
# is the literal string TRUE — the canonical upstream gate that bypasses
# setup.py's own guessed-wheel pip attempt.
if [ -z "$WHEELHOUSE_DIR" ]; then
    WHEELHOUSE_DIR="$(mktemp -d)"
else
    mkdir -p "$WHEELHOUSE_DIR"
fi
info "Building patched mamba_ssm wheel into $WHEELHOUSE_DIR ..."
MAMBA_FORCE_BUILD=TRUE "$PYTHON_BIN" -m pip wheel "$SRC_DIR" \
    --no-build-isolation --no-deps --no-cache-dir -w "$WHEELHOUSE_DIR"

# --- 6. Build causal_conv1d unpatched from the index ------------------------
info "Building causal_conv1d==${CAUSAL_CONV1D_VERSION} (unpatched) ..."
"$PYTHON_BIN" -m pip wheel "causal_conv1d==${CAUSAL_CONV1D_VERSION}" \
    --no-build-isolation --no-cache-dir \
    --index-url "$INDEX_URL" -w "$WHEELHOUSE_DIR"

# --- 7. Install both wheels -------------------------------------------------
# Deps resolve from the index, so the dependency posture stays identical to
# the old mamba-extra install.
if [ "$NO_INSTALL" = true ]; then
    info "--no-install: wheels are in $WHEELHOUSE_DIR; skipping install."
    exit 0
fi
MAMBA_WHEEL=$(find "$WHEELHOUSE_DIR" -maxdepth 1 -name "mamba_ssm-${MAMBA_SSM_VERSION}-*.whl" | head -n 1)
CONV_WHEEL=$(find "$WHEELHOUSE_DIR" -maxdepth 1 -name "causal_conv1d-${CAUSAL_CONV1D_VERSION}-*.whl" | head -n 1)
[ -n "$MAMBA_WHEEL" ] || error "No mamba_ssm wheel found in $WHEELHOUSE_DIR."
[ -n "$CONV_WHEEL" ] || error "No causal_conv1d wheel found in $WHEELHOUSE_DIR."
info "Installing $MAMBA_WHEEL and $CONV_WHEEL ..."
"$PYTHON_BIN" -m pip install --no-cache-dir --index-url "$INDEX_URL" \
    "$MAMBA_WHEEL" "$CONV_WHEEL"
info "mamba kernels installed successfully."
