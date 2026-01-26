# u8darts (DEPRECATED)

⚠️ **This package has been renamed to `darts`.** ⚠️

## Quick Migration

```bash
pip uninstall u8darts
pip install "darts[all]"  # or darts[torch], darts[notorch]
```

**No code changes needed** - both packages import as `import darts`.

---

## Why This Package Exists

This is a **compatibility redirect package** that automatically installs `darts` and shows a deprecation warning.

When you install `u8darts`, you're actually getting the `darts` package with a deprecation notice.

## Timeline

| Date | Status |
|------|--------|
| **Now - June 2027** | Both packages work, u8darts shows warnings |
| **June 2027** | u8darts discontinued, only darts receives updates |

## Migration Instructions

### 1. Update Installation

**Before:**
```bash
pip install u8darts[torch]
```

**After:**
```bash
pip install "darts[torch]"
```

### 2. Update Requirements Files

**requirements.txt:**
```diff
- u8darts[torch]>=0.40.0
+ darts[torch]>=0.40.0
```

**pyproject.toml:**
```diff
[project]
dependencies = [
-    "u8darts[torch]>=0.40.0",
+    "darts[torch]>=0.40.0",
]
```

### 3. No Code Changes!

Your Python code stays the same:
```python
import darts  # ✓ Same import
from darts import TimeSeries  # ✓ Same imports
```

---

## Why the Change?

The "u8" prefix was a historical artifact from our company name (Unit8). As the project has matured into a widely-used open-source library, we're simplifying to just "darts" for better discoverability and cleaner branding.

## Full Migration Guide

**Complete guide:** [MIGRATION.md](https://github.com/unit8co/darts/blob/master/MIGRATION.md)

Includes:
- Detailed migration steps
- CI/CD pipeline updates
- Docker configuration
- Troubleshooting
- FAQ

## Questions?

- 📖 **Documentation**: https://unit8co.github.io/darts/
- 🐙 **GitHub**: https://github.com/unit8co/darts
- 💬 **Gitter**: https://gitter.im/unit8co/darts
- 🐛 **Issues**: https://github.com/unit8co/darts/issues

---

**Package maintained until June 2027** for backward compatibility.

Use `darts` for all new installations.
