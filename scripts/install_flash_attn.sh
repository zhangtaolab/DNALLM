#!/bin/bash
#
# install_flash_attn.sh — install flash-attn by building the pinned
# 2.8.3.post2 sdist from source with an explicit sm_121 gencode patch.
#
# WHY: no prebuilt flash-attn artifact exists for GB10-class boxes. The
# v2.8.3.post2 GitHub release ships 48 wheels, all linux_x86_64 + cu12 +
# torch 2.4-2.8; PyPI and the TUNA mirror carry the sdist only. The sdist's
# setup.py has no "121" architecture branch (its FLASH_ATTN_CUDA_ARCHS
# default is "80;90;100;120" with gencode branches for exactly those four),
# so a default build embeds four architecture families and runs on sm_121
# only through the same-major sm_120 binary-compatibility rule. This script
# patches setup.py to add the explicit compute_121/sm_121 branch so the
# build is deterministic for this device class.
#
# WHAT: fetch the flash_attn sdist file directly from a package index's
# simple page (default PyPI; override with --index-url, e.g. the TUNA
# mirror https://pypi.tuna.tsinghua.edu.cn/simple), verify its sha256
# against the pinned constant below BEFORE anything is extracted or
# executed, add the architecture-121 gencode branch to the extracted
# setup.py (occurrence-counted assertions on both sides of the edit),
# then build the wheel from the patched tree with --no-build-isolation
# (the sdist ships no pyproject.toml and setup.py imports torch
# unconditionally — an isolated PEP 517 build dies with
# ModuleNotFoundError: torch) and install exactly that wheel. The fetch
# deliberately avoids pip's resolution machinery: pip builds sdist
# metadata by EXECUTING setup.py from the downloaded archive, which
# would run unverified bytes before the hash gate.
#
# SUPPLY CHAIN: the sha256 pin is the byte-integrity control, which makes
# ANY index route safe — PyPI itself or a mirror cannot serve altered
# bytes without the gate failing closed. There is deliberately no flag
# or environment variable that overrides the hash constant. GitHub
# access is NOT required at any point in this pipeline.
#
# Usage:
#   scripts/install_flash_attn.sh [--default] [--patch-only] [--verify-file PATH]
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
#   --wheelhouse DIR   directory for the built wheel (default: fresh temp
#                      directory)
#   --no-install       build the wheel but skip installing it

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Pinned constants ------------------------------------------------------
# A version bump MUST re-verify the sha256 constant and the sm_121 patch
# shape together; --patch-only re-runs both checks cheaply.
FLASH_ATTN_VERSION="2.8.3.post2"
FLASH_ATTN_SDIST_SHA256="d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b"

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
    if [ "$ACTUAL_SHA" = "$FLASH_ATTN_SDIST_SHA256" ]; then
        info "verify-file: $VERIFY_FILE matches the pinned flash-attn $FLASH_ATTN_VERSION sdist sha256."
        exit 0
    fi
    error "verify-file: sha256 MISMATCH for $VERIFY_FILE
  expected: $FLASH_ATTN_SDIST_SHA256
  actual:   $ACTUAL_SHA"
fi

# --- Interpreter availability ----------------------------------------------
# The fetch below needs only the interpreter itself (stdlib); pip is not
# required until the build step and is checked there with its own bootstrap
# hint.
"$PYTHON_BIN" --version >/dev/null 2>&1 \
    || error "Python interpreter '$PYTHON_BIN' not found. Pass --python /path/to/python (a venv interpreter is fine)."

# --- Environment check: build path only ------------------------------------
# Informational logging plus the torch-presence guard. --patch-only skips
# this so the patch verification runs on boxes without torch/CUDA.
if [ "$PATCH_ONLY" = false ]; then
    info "Starting environment check..."
    TORCH_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
    if [ -z "$TORCH_VERSION" ]; then
        error "PyTorch is not importable by '$PYTHON_BIN'. The flash-attn source build requires a CUDA-enabled PyTorch to be installed first."
    fi
    CUDA_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || true)
    if [ -z "$CUDA_VERSION" ]; then
        error "No CUDA runtime detected from PyTorch. The flash-attn CUDA kernels require a CUDA-enabled PyTorch build."
    fi
    info "Detected PyTorch: $TORCH_VERSION (CUDA $CUDA_VERSION)"
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
page_url = index_url.rstrip("/") + "/flash-attn/"
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

SDIST_NAME="flash_attn-${FLASH_ATTN_VERSION}.tar.gz"
DOWNLOAD_DIR="$(mktemp -d)"
info "Fetching $SDIST_NAME from $INDEX_URL ..."
fetch_sdist "$INDEX_URL" "$SDIST_NAME" "$FLASH_ATTN_SDIST_SHA256" "${DOWNLOAD_DIR}/${SDIST_NAME}" \
    || error "Fetch of $SDIST_NAME from $INDEX_URL failed (route unreachable, file not listed, or the index-declared sha256 differs from the pinned constant)."
SDIST_PATH="${DOWNLOAD_DIR}/${SDIST_NAME}"
[ -f "$SDIST_PATH" ] || error "Fetch did not produce $SDIST_NAME in $DOWNLOAD_DIR."

# --- 2. Hash gate: fail closed BEFORE any extraction, patch, or build ------
info "Verifying sha256 of $SDIST_NAME ..."
ACTUAL_SHA=$(sha256sum "$SDIST_PATH" | awk '{print $1}')
if [ "$ACTUAL_SHA" != "$FLASH_ATTN_SDIST_SHA256" ]; then
    error "sha256 MISMATCH for $SDIST_NAME — refusing to extract, patch, or build.
  expected: $FLASH_ATTN_SDIST_SHA256
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
SRC_DIR="${EXTRACT_DIR}/flash_attn-${FLASH_ATTN_VERSION}"
SETUP_PY="${SRC_DIR}/setup.py"
[ -f "$SETUP_PY" ] || error "Extracted tree does not contain flash_attn-${FLASH_ATTN_VERSION}/setup.py."

# --- 4. Patch: add the architecture-121 gencode branch ----------------------
# The upstream setup.py emits exactly one '-gencode arch=compute_XX,code=sm_XX'
# occurrence per existing architecture branch (80/90/100/120, guarded by the
# FLASH_ATTN_CUDA_ARCHS env membership tests). This patch inserts ONE new
# branch for 121 after the 120 branch, in the same style. Occurrence-exact
# assertions on both sides: zero compute_121/sm_121 tokens before, exactly one
# after, and each of the four existing architecture tokens unchanged at one —
# any other count is a hard error (a future version bump that re-pins the hash
# but changes the patch shape must fail loudly, never silently build
# unpatched). An ast.parse pass then proves the edited file is still valid
# Python (stdlib only; setup.py itself is never imported here).
count_occurrences() { # count_occurrences TOKEN FILE
    grep -o -F -- "$1" "$2" | wc -l
}

check_existing_arch_branches() { # each of 80/90/100/120 must appear exactly once
    for arch in 80 90 100 120; do
        n=$(count_occurrences "arch=compute_${arch},code=sm_${arch}" "$SETUP_PY")
        if [ "$n" -ne 1 ]; then
            error "Expected exactly 1 'arch=compute_${arch},code=sm_${arch}' occurrence in setup.py, found $n — upstream patch context changed; refusing to build."
        fi
    done
}

info "Patching setup.py to add the architecture-121 gencode branch ..."
BEFORE_121=$(count_occurrences "arch=compute_121,code=sm_121" "$SETUP_PY")
if [ "$BEFORE_121" -ne 0 ]; then
    error "Expected exactly 0 'arch=compute_121,code=sm_121' occurrences in setup.py before patching, found $BEFORE_121 — upstream already ships sm_121; the pinned patch shape is stale."
fi
check_existing_arch_branches
sed -i '/cc_flag.append("arch=compute_120,code=sm_120")$/a\        if bare_metal_version >= Version("12.8") and "121" in cuda_archs():\n            cc_flag.append("-gencode")\n            cc_flag.append("arch=compute_121,code=sm_121")' "$SETUP_PY"
AFTER_121=$(count_occurrences "arch=compute_121,code=sm_121" "$SETUP_PY")
BRANCH_121=$(count_occurrences '"121" in cuda_archs()' "$SETUP_PY")
if [ "$AFTER_121" -ne 1 ] || [ "$BRANCH_121" -ne 1 ]; then
    error "Patch assertions failed: arch=compute_121,code=sm_121 occurrences: $AFTER_121 (want 1), '121'-branch conditions: $BRANCH_121 (want 1)."
fi
check_existing_arch_branches
"$PYTHON_BIN" -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$SETUP_PY" \
    || error "Patched setup.py no longer parses as Python — the gencode insertion corrupted the file; refusing to build."
info "Patch applied: one architecture-121 branch emitting one compute_121/sm_121 gencode; 80/90/100/120 branches unchanged."

if [ "$PATCH_ONLY" = true ]; then
    echo "PATCH VERIFIED: sha256 ok, sm_121 gencode patch verified"
    exit 0
fi

# --- 5. Build the patched flash_attn wheel ----------------------------------
# pip enters the pipeline here (the fetch never used it). uv-created venvs
# ship without pip, hence the bootstrap hint. Build flags:
#   FLASH_ATTENTION_FORCE_BUILD=TRUE  — the canonical upstream gate that
#                                       bypasses setup.py's own
#                                       guessed-wheel download attempt
#   FLASH_ATTN_CUDA_ARCHS=121         — select ONLY the 121 branch (the
#                                       default "80;90;100;120" list would
#                                       embed four architecture families,
#                                       roughly 4x the compile time)
#   MAX_JOBS (default 4, caller-raisable) — setup.py honors a pre-set
#                                       MAX_JOBS; 4 keeps peak memory low
#                                       enough for shared boxes
# --no-build-isolation is mandatory: the sdist ships no pyproject.toml and
# its setup.py imports torch unconditionally, so an isolated PEP 517 build
# dies with ModuleNotFoundError: torch. Build deps (torch, ninja, packaging)
# must pre-exist in the target environment — every dnallm install carries
# them.
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    error "pip is not importable by '$PYTHON_BIN'. Bootstrap it first, e.g.:
  uv pip install --python <venv> pip      (uv lane)
  $PYTHON_BIN -m ensurepip                (plain venv lane)"
fi
if [ -z "$WHEELHOUSE_DIR" ]; then
    WHEELHOUSE_DIR="$(mktemp -d)"
else
    mkdir -p "$WHEELHOUSE_DIR"
fi
info "Building patched flash_attn wheel into $WHEELHOUSE_DIR (MAX_JOBS=${MAX_JOBS:-4}) ..."
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=121 MAX_JOBS="${MAX_JOBS:-4}" \
    "$PYTHON_BIN" -m pip wheel "$SRC_DIR" \
    --no-build-isolation --no-deps --no-cache-dir -w "$WHEELHOUSE_DIR"

# --- 6. Install exactly the resulting wheel ---------------------------------
# Built from the PATCHED extracted directory, never from the tarball path.
# --no-deps keeps pip from touching the environment's torch: flash-attn's
# runtime deps (torch, einops) are unpinned and already satisfied by any
# dnallm install (the environment check above proved torch importable), and
# a re-resolution from an arbitrary index could downgrade a CUDA build.
if [ "$NO_INSTALL" = true ]; then
    info "--no-install: wheel is in $WHEELHOUSE_DIR; skipping install."
    exit 0
fi
FLASH_ATTN_WHEEL=$(find "$WHEELHOUSE_DIR" -maxdepth 1 -name "flash_attn-${FLASH_ATTN_VERSION}-*.whl" | head -n 1)
[ -n "$FLASH_ATTN_WHEEL" ] || error "No flash_attn wheel found in $WHEELHOUSE_DIR."
info "Installing $FLASH_ATTN_WHEEL ..."
"$PYTHON_BIN" -m pip install --no-cache-dir --no-deps "$FLASH_ATTN_WHEEL"
info "flash-attn $FLASH_ATTN_VERSION (sm_121-patched) installed successfully."
