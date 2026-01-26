#!/bin/bash
set -e

# Add Homebrew to PATH if not present
export PATH="/opt/homebrew/bin:$PATH"

# Extract version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | head -1 | cut -d'"' -f2)

# --- Helper Functions ---
log() {
    echo "[$1] $2..."
    echo "--------------------------------------"
}

build_package() {
    local dir=$1
    local name=$2
    local should_sync=$3

    pushd "$dir" > /dev/null

    # Sync dependencies only for uv-managed projects
    if [ "$should_sync" = "true" ]; then
        uv sync --no-dev
    fi

    # Clean and build
    rm -rf dist/ build/ *.egg-info/
    uv build

    echo "Done: Built $name package"
    ls -lh dist/
    echo ""
    popd > /dev/null
}

# --- Main Execution ---
echo "============================================"
echo "Building Darts $VERSION for PyPI"
echo "============================================"
echo ""

# 1. Build Main Package
log "1/2" "Building main 'darts' package"
build_package "." "darts" "true"

# 2. Build Compatibility Package
log "2/2" "Building 'u8darts' compatibility package"
build_package "u8darts" "u8darts" "false"

# --- Summary ---
echo "============================================"
echo "Build Complete! Version $VERSION"
echo "============================================"
echo ""
echo "To upload all packages to PyPI:"
echo "twine upload dist/* u8darts/dist/*"
