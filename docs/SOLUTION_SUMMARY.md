# Sphinx Documentation Package Naming Solution

## Problem
The Sphinx documentation was displaying package names as full paths (e.g., `darts.models.forecasting`) in the navigation bar instead of using the descriptive titles from package docstrings (e.g., "Forecasting Models").

## Solution
A post-processing approach that extracts titles from package `__init__.py` docstrings and replaces the package path titles in generated RST files.

### Components

#### 1. Post-Processing Script: `docs/fix_package_titles.py`
- Reads generated RST files
- Extracts the title from each package's `__init__.py` docstring
- Replaces the package path title with the docstring title
- Maintains the RST heading structure for proper toctree hierarchy

#### 2. Updated Makefile: `docs/Makefile`
The `generate-api` target now includes:
```makefile
generate-api:
	@echo "[Makefile] generating API using sphinx-apidoc..."
	@sphinx-apidoc -e -f --templatedir templates -o "$(SOURCEDIR)/generated_api" ../darts ../darts/logging.py ../darts/tests/*
	@cp -r $(GENAPIDIR)/*.rst "$(SOURCEDIR)/generated_api/"
	@echo "[Makefile] fixing package titles..."
	@python fix_package_titles.py "$(SOURCEDIR)/generated_api" ../darts
```

#### 3. Updated Template: `docs/templates/package.rst_t`
- Keeps an explicit RST title for proper toctree structure
- Removes the `automodule` directive to avoid duplicate headings
- The title is initially set to the package path, then replaced by the script

## Results

### Before
```
- darts.models
  - darts.models.forecasting
    - ARIMA
    - Baseline Models
```

### After
```
- Models
  - Forecasting Models
    - ARIMA
    - Baseline Models
```

## Usage

The solution is integrated into the build process. Simply run:

```bash
make --directory=./docs build-api
```

Or for full documentation:

```bash
make --directory=./docs build-all-docs
```

## How It Works

1. **sphinx-apidoc** generates RST files using the template
   - Package RST files start with titles like `darts.models`

2. **fix_package_titles.py** post-processes the files
   - Extracts "Models" from `darts/models/__init__.py` docstring
   - Replaces `darts.models` with `Models` in the RST file

3. **sphinx-build** generates HTML
   - Uses the updated titles from RST files
   - Maintains proper toctree hierarchy (l1, l2, l3)

## Docstring Format

For the script to work correctly, package `__init__.py` files should have docstrings with a title on the first line:

```python
"""
Package Title
-------------
"""
```

or

```python
"""
Package Title
=============
"""
```

The script extracts the first non-empty, non-underline line as the title.
