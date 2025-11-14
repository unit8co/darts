#!/usr/bin/env python3
"""
Post-process generated RST files to replace package path titles with docstring titles.
"""

import re
import sys
from pathlib import Path


def extract_docstring_title(module_path):
    """Extract the title from a module's docstring."""
    try:
        # Read the __init__.py file
        init_file = Path(module_path) / "__init__.py"
        if not init_file.exists():
            return None

        with open(init_file, encoding="utf-8") as f:
            content = f.read()

        # Extract docstring
        docstring_match = re.match(r'^\s*"""(.*?)"""', content, re.DOTALL)
        if not docstring_match:
            docstring_match = re.match(r"^\s*'''(.*?)'''", content, re.DOTALL)

        if not docstring_match:
            return None

        docstring = docstring_match.group(1).strip()

        # Extract title (first non-empty line)
        lines = docstring.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("-") and not line.startswith("="):
                return line

        return None
    except Exception as e:
        print(f"Error extracting docstring from {module_path}: {e}", file=sys.stderr)
        return None


def fix_rst_file(rst_path, source_root):
    """Replace package path title with docstring title in an RST file."""
    with open(rst_path, encoding="utf-8") as f:
        content = f.read()

    # Check if this is a package RST file (contains ".. currentmodule::")
    if ".. currentmodule::" not in content:
        return False

    # Extract package name from ".. currentmodule:: package.name"
    module_match = re.search(r"\.\. currentmodule:: (.+)", content)
    if not module_match:
        return False

    pkg_name = module_match.group(1).strip()

    # Convert package name to file path
    # The package name is like "darts.models.forecasting"
    # source_root is the darts package directory (../darts)
    # So we need to strip the first "darts." from the package name
    parts = pkg_name.split(".")
    if parts[0] == "darts":
        # Remove the 'darts' prefix since source_root is already the darts package
        rel_path = "/".join(parts[1:]) if len(parts) > 1 else ""
        pkg_path = Path(source_root) / rel_path if rel_path else Path(source_root)
    else:
        # Not a darts.* package, use full path
        pkg_path = Path(source_root) / pkg_name.replace(".", "/")

    # Extract docstring title
    doc_title = extract_docstring_title(pkg_path)
    if not doc_title:
        print(f"Could not extract title for {pkg_name}", file=sys.stderr)
        return False

    # Find and replace the title
    # Pattern: look for a line followed by a line of ===
    # This pattern finds: pkg_name\n====
    title_pattern = re.compile(
        r"^(" + re.escape(pkg_name) + r")\n([=]+)\n", re.MULTILINE
    )

    match = title_pattern.search(content)
    if match:
        # Calculate the underline length based on the new title
        underline = "=" * len(doc_title)
        replacement = f"{doc_title}\n{underline}\n"
        new_content = title_pattern.sub(replacement, content)

        with open(rst_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"Fixed {rst_path}: {pkg_name} -> {doc_title}")
        return True

    return False


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: fix_package_titles.py <generated_api_dir> <source_root>",
            file=sys.stderr,
        )
        sys.exit(1)

    api_dir = Path(sys.argv[1])
    source_root = Path(sys.argv[2])

    if not api_dir.exists():
        print(f"Directory not found: {api_dir}", file=sys.stderr)
        sys.exit(1)

    if not source_root.exists():
        print(f"Source root not found: {source_root}", file=sys.stderr)
        sys.exit(1)

    # Process all RST files in the generated API directory
    fixed_count = 0
    for rst_file in api_dir.glob("*.rst"):
        if fix_rst_file(rst_file, source_root):
            fixed_count += 1

    print(f"\nFixed {fixed_count} RST files")


if __name__ == "__main__":
    main()
