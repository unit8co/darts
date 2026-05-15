"""
Package title and docstring extraction for Sphinx API documentation.

This module processes what is given by the *.rst files and provides utilities to:
1. Replace package path titles (e.g., "darts.models.forecasting") with
   descriptive titles from package docstrings (e.g., "Forecasting Models")
2. Insert the full docstring content from package __init__.py files
3. Fix inline :doc: link titles to use descriptive names

Note: the *.rst files for packages and modules are built using the templates in `docs/templates`
Used by conf.py via Sphinx's 'source-read' event.
"""

import re
from pathlib import Path


def extract_docstring_from_file(file_path):
    """Extract the full docstring from a Python file.

    Returns:
        Tuple of (title, body) where body is the docstring content without the title section
    """
    try:
        if not file_path.exists():
            return None, None

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Extract docstring (enclosed by `"""lorem ipsum"""` or `'''lorem ipsum'''`)
        docstring_match = re.match(r'^\s*"""(.*?)"""', content, re.DOTALL)
        if not docstring_match:
            docstring_match = re.match(r"^\s*'''(.*?)'''", content, re.DOTALL)

        if not docstring_match:
            return None, None

        docstring = docstring_match.group(1).strip()
        lines = docstring.split("\n")

        # Find the title (first non-empty line)
        # Titles are expected to be at the top of the module, and be underlined with `===` or `---`
        # """
        # Forecasting Models
        # ==================
        # """

        title = None
        title_line_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("-")
                and not stripped.startswith("=")
            ):
                title = stripped
                title_line_idx = i
                break

        if not title:
            return None, None

        # Skip the title and its underline (if present)
        body_start_idx = title_line_idx + 1

        # Skip underline lines (lines with only = or -)
        while body_start_idx < len(lines):
            stripped = lines[body_start_idx].strip()
            if stripped and not all(c in "=-" for c in stripped):
                break
            body_start_idx += 1

        # Get the rest of the docstring as the body
        body_lines = lines[body_start_idx:]
        body = "\n".join(body_lines).strip()

        return title, body if body else None

    except Exception:
        return None, None


def get_module_info(module_name, source_root):
    """Get the title and docstring body for a module or package.

    Args:
        module_name: Full module name like "darts.models.forecasting"
        source_root: Path to the darts package directory

    Returns:
        Tuple of (title, body) where body is the docstring content without title
    """
    parts = module_name.split(".")
    if parts[0] == "darts":
        # Remove the 'darts' prefix since source_root is already the darts package
        rel_path = "/".join(parts[1:]) if len(parts) > 1 else ""
        base_path = source_root / rel_path if rel_path else source_root
    else:
        # Not a darts.* package, use full path
        base_path = source_root / module_name.replace(".", "/")

    # Check if it's a package (has __init__.py)
    if (base_path / "__init__.py").exists():
        return extract_docstring_from_file(base_path / "__init__.py")
    # Check if it's a module (has .py file)
    elif base_path.with_suffix(".py").exists():
        return extract_docstring_from_file(base_path.with_suffix(".py"))

    return None, None


def process_package_docstrings(app, docname, source):
    """Process package RST files to replace titles and add docstring content.

    This is connected to the 'source-read' event in Sphinx.

    Args:
        app: Sphinx application object
        docname: Name of the document being read
        source: List with single string element containing the source RST content
    """
    # Only process files in generated_api directory
    if not docname.startswith("generated_api/"):
        return

    content = source[0]

    # Check if this is a package RST file (contains ".. currentmodule::")
    if ".. currentmodule::" not in content:
        return

    # Extract package name from ".. currentmodule:: package.name"
    module_match = re.search(r"\.\. currentmodule:: (.+)", content)
    if not module_match:
        return

    pkg_name = module_match.group(1).strip()

    # Get the source root (darts package directory)
    source_root = Path(app.confdir).parent.parent / "darts"

    # 1. Fix the main title and add docstring body
    doc_title, doc_body = get_module_info(pkg_name, source_root)
    if not doc_title:
        return

    # Find and replace the title
    # titles are given in package.rst with format:
    # ```
    # darts.models
    # ============
    # ```
    title_pattern = re.compile(
        r"^(" + re.escape(pkg_name) + r")\n([=]+)\n", re.MULTILINE
    )

    match = title_pattern.search(content)
    if match:
        # Calculate the underline length based on the new title
        underline = "=" * len(doc_title)

        # Build the replacement with title and optional docstring body
        replacement = f"{doc_title}\n{underline}\n"
        if doc_body:
            replacement += f"\n{doc_body}\n"

        content = title_pattern.sub(replacement, content)

    # 2. Fix inline link titles
    # links are given in package.rst with format:
    # ```
    # - :doc:`darts.models <darts.models>`
    # - :doc:`darts.utils <darts.utils>`
    # ```
    link_pattern = re.compile(r":doc:`([^`]+)\s*<([^>]+)>`")

    def replace_link(match):
        target_module = match.group(2).strip()
        # Get the proper title for the target module
        target_title, _ = get_module_info(target_module, source_root)
        if target_title:
            return f":doc:`{target_title} <{target_module}>`"
        return match.group(0)

    content = link_pattern.sub(replace_link, content)

    # Update the source
    source[0] = content
