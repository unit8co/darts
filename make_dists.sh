#!/bin/bash
set -e

# Add Homebrew to PATH if not present
export PATH="/opt/homebrew/bin:$PATH"

# Extract version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | head -1 | cut -d'"' -f2)

echo "============================================"
echo "Building Darts Packages v$VERSION"
echo "============================================"
echo ""

# --- 1. Build main darts package ---
echo "[1/2] Building 'darts' package..."
echo "--------------------------------------"
rm -rf dist/ build/ *.egg-info/
uv build
echo "✓ Built darts package:"
ls -lh dist/
echo ""

# --- 2. Build u8darts redirect package ---
echo "[2/2] Building 'u8darts' redirect package..."
echo "--------------------------------------"

# Swap pyproject files
cp pyproject.toml pyproject_darts.toml
cp pyproject_u8darts.toml pyproject.toml

# Swap README files
cp README.md README_darts.md
cp README_u8darts.md README.md

# Build u8darts (metadata-only package)
uv build

# Restore original files
mv pyproject_darts.toml pyproject.toml
mv README_darts.md README.md

echo "✓ Built u8darts package:"
ls -lh dist/
echo ""

# --- Summary ---
echo "============================================"
echo "Build Complete! Version $VERSION"
echo "============================================"
echo ""
echo "Packages built:"
ls -1 dist/
echo ""
echo "To upload to TestPyPI:"
echo "uv run twine upload --repository testpypi dist/darts-$VERSION*"
echo "uv run twine upload --repository testpypi dist/u8darts-$VERSION*"
echo ""
echo "To upload to PyPI:"
echo "uv run twine upload dist/darts-$VERSION*"
echo "uv run twine upload dist/u8darts-$VERSION*"
