#!/bin/bash
set -e

# Add Homebrew to PATH if not present
export PATH="/opt/homebrew/bin:$PATH"

# Extract version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | head -1 | cut -d'"' -f2)

echo "============================================"
echo "Building Darts Package Version $VERSION"
echo "============================================"
echo ""

rm -rf dist/ build/ *.egg-info/
uv build
echo "✓ Built darts package:"
ls -lh dist/
echo ""

# --- Summary ---
echo "============================================"
echo "Build Complete! Version $VERSION"
echo "============================================"
echo ""
echo "Package built:"
ls -1 dist/
echo ""
echo "To upload to PyPI:"
echo "uv run twine upload dist/darts-$VERSION*"
