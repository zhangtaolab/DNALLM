#!/bin/bash
#
# This script automates the installation of causal_conv1d and mamba_ssm
# packages by detecting Python/PyTorch/CUDA versions and constructing
# the appropriate .whl file URL for direct installation via pip.
# It can be run with the --default flag to skip all prompts.

# Exit immediately if a command exits with a non-zero status.
set -e

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

# --- Argument Parsing for Non-Interactive Mode ---
USE_DEFAULTS=false
if [ "$1" == "--default" ]; then
    info "Running in non-interactive mode with default settings."
    USE_DEFAULTS=true
fi

# --- 1. Dependency and Environment Check ---
info "Starting environment check..."

# Check for Python and pip
command -v python &>/dev/null || error "Python is not installed. Please install Python and try again."
command -v pip &>/dev/null || error "pip is not installed. Please install pip and try again."

# Check for PyTorch and detect versions
info "Detecting Python, PyTorch, and CUDA versions..."
PY_VERSION_RAW=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_VERSION_URL=$(echo "$PY_VERSION_RAW" | tr -d '.')

TORCH_VERSION_RAW=$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null)
if [ -z "$TORCH_VERSION_RAW" ]; then
    error "PyTorch is not installed or could not be detected. This script requires PyTorch to be installed to determine the correct package versions."
fi
TORCH_VERSION_URL=$(echo "$TORCH_VERSION_RAW" | cut -d. -f1,2)

CUDA_VERSION_RAW=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)
if [ -z "$CUDA_VERSION_RAW" ]; then
    error "Could not detect a CUDA version from PyTorch. The target packages require a CUDA-enabled build of PyTorch."
fi
CUDA_VERSION_URL=$(echo "$CUDA_VERSION_RAW" | cut -d. -f1)
CUDA_PART="cu${CUDA_VERSION_URL}"

info "Detected Python version: $PY_VERSION_RAW (for URL: cp$PY_VERSION_URL)"
info "Detected PyTorch version: $TORCH_VERSION_RAW (for URL: torch$TORCH_VERSION_URL)"
info "Detected CUDA version: $CUDA_VERSION_RAW (for URL: $CUDA_PART)"
echo ""

# --- Interactive Configuration or Default Setup ---
if [ "$USE_DEFAULTS" = false ]; then
    # --- 2. Configure GitHub Access (Interactive) ---
    PROXY_CHOICE=""
    GIT_REPO_PREFIX="https://github.com" # Base URL for git operations

    info "Due to potential network issues, you can use a GitHub proxy for downloads and version fetching."
    PS3="Do you want to use a GitHub proxy? "
    select use_proxy in "Yes, select a proxy" "No, use github.com directly (default)"; do
        case $REPLY in
            1)
                PS3="Please select a proxy server: "
                select proxy in "gh-proxy.com" "ghfast.top" "gh.bioinfor.eu.org" "Cancel"; do
                    if [ "$proxy" = "Cancel" ]; then
                        info "Proxy selection canceled. Using github.com directly."
                        break
                    elif [ -n "$proxy" ]; then
                        PROXY_CHOICE="https://${proxy}"
                        GIT_REPO_PREFIX="${PROXY_CHOICE}/github.com"
                        info "Using proxy: $PROXY_CHOICE"
                        break
                    else
                        warn "Invalid selection. Please choose a number from the list."
                    fi
                done
                break
                ;;
            2|*)
                info "Proceeding without a proxy."
                break
                ;;
        esac
    done
    echo ""

    # --- 3. Fetch and Select Package Versions (Interactive) ---
    info "Fetching available package versions..."
    command -v git &>/dev/null || error "'git' command is not installed. It is required to fetch available package versions."

    fetch_and_display_versions() {
        local repo_url=$1
        local package_name=$2
        info "Latest available versions for $package_name (newest first):"
        local versions
        if ! versions=$(git ls-remote --tags "$repo_url" 2>/dev/null | awk '{print $2}' | sed 's#refs/tags/v##' | grep -E '^[0-9]+\.[0-9]' | grep -v '{}' | sort -V -r | head -n 15); then
            warn "Could not fetch versions for $package_name. This might be due to network issues or an incompatible proxy."
            echo ""
            return
        fi
        if [ -z "$versions" ]; then
            warn "Could not find any valid versions for $package_name. Please enter the version manually."
        else
            echo "$versions"
        fi
        echo ""
    }

    CAUSAL_REPO_URL="${GIT_REPO_PREFIX}/Dao-AILab/causal-conv1d.git"
    MAMBA_REPO_URL="${GIT_REPO_PREFIX}/state-spaces/mamba.git"

    fetch_and_display_versions "$CAUSAL_REPO_URL" "causal_conv1d"
    fetch_and_display_versions "$MAMBA_REPO_URL" "mamba_ssm"

    DEFAULT_CAUSAL_VERSION="1.5.0.post8"
    DEFAULT_MAMBA_VERSION="2.2.4"
    read -p "Enter causal_conv1d version [default: $DEFAULT_CAUSAL_VERSION]: " CAUSAL_VERSION
    CAUSAL_VERSION=${CAUSAL_VERSION:-$DEFAULT_CAUSAL_VERSION}
    read -p "Enter mamba_ssm version [default: $DEFAULT_MAMBA_VERSION]: " MAMBA_VERSION
    MAMBA_VERSION=${MAMBA_VERSION:-$DEFAULT_MAMBA_VERSION}
    echo ""
else
    # --- Non-Interactive Default Setup ---
    info "Using default package versions and no proxy."
    CAUSAL_VERSION="1.5.0.post8"
    MAMBA_VERSION="2.2.4"
    PROXY_CHOICE="" # No proxy
fi

# --- 4. Construct Download URLs ---
info "Constructing package download URLs..."

CAUSAL_BASE_URL="github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_VERSION}"
MAMBA_BASE_URL="github.com/state-spaces/mamba/releases/download/v${MAMBA_VERSION}"

if [ -n "$PROXY_CHOICE" ]; then
  CAUSAL_PREFIX_URL="${PROXY_CHOICE}/${CAUSAL_BASE_URL}"
  MAMBA_PREFIX_URL="${PROXY_CHOICE}/${MAMBA_BASE_URL}"
else
  CAUSAL_PREFIX_URL="https://${CAUSAL_BASE_URL}"
  MAMBA_PREFIX_URL="https://${MAMBA_BASE_URL}"
fi

BUILD_STRING="${CUDA_PART}torch${TORCH_VERSION_URL}cxx11abiFALSE"
PLATFORM_STRING="cp${PY_VERSION_URL}-cp${PY_VERSION_URL}-linux_x86_64.whl"
CAUSAL_WHL_URL="${CAUSAL_PREFIX_URL}/causal_conv1d-${CAUSAL_VERSION}+${BUILD_STRING}-${PLATFORM_STRING}"
MAMBA_WHL_URL="${MAMBA_PREFIX_URL}/mamba_ssm-${MAMBA_VERSION}+${BUILD_STRING}-${PLATFORM_STRING}"
PACKAGES_TO_INSTALL=("$CAUSAL_WHL_URL" "$MAMBA_WHL_URL")

# --- 5. Installation ---
info "The following packages will be installed from the generated URLs:"
echo "1. ${PACKAGES_TO_INSTALL[0]}"
echo "2. ${PACKAGES_TO_INSTALL[1]}"
echo ""

if [ "$USE_DEFAULTS" = false ]; then
    read -p "Press Enter to continue with the installation, or Ctrl+C to abort."
fi

for pkg_url in "${PACKAGES_TO_INSTALL[@]}"; do
    info "Installing from: $pkg_url"
    pip install --no-cache-dir --no-build-isolation "$pkg_url"
    if [ $? -eq 0 ]; then
        info "Successfully installed package."
    else
        error "Failed to install package from $pkg_url. Please manually check the generated URL and your environment. It's possible a wheel for your specific combination of Python/PyTorch/CUDA does not exist."
    fi
    echo ""
done

info "All packages have been installed successfully!"

