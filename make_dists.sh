#!/bin/bash
# Build darts package using uv
# Produces both source distribution (.tar.gz) and wheel (.whl) in dist/

# Clean previous builds
rm -rf dist/ darts.egg-info/

# Build package (reads from pyproject.toml)
uv build

echo ""
echo "Build complete! Packages created in dist/:"
ls -lh dist/
