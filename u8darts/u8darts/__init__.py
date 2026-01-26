"""
u8darts - DEPRECATED LEGACY PACKAGE

This package has been renamed to 'darts'. Please update your installation:

    pip uninstall u8darts
    pip install "darts[torch]"  # or darts[all], darts[notorch]

This compatibility shim will be maintained until June 2027.

For more information, see:
https://github.com/unit8co/darts/blob/master/MIGRATION.md
"""

import sys
import warnings

# Issue deprecation warning ONCE per Python session
if "u8darts._warned" not in sys.modules:
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "⚠️  DEPRECATION: 'u8darts' package is deprecated\n"
        "=" * 70 + "\n"
        "Migrate to 'darts' package:\n\n"
        "  pip uninstall u8darts\n"
        "  pip install darts[torch]  # or darts[all], darts[notorch]\n\n"
        "Your code works as-is - only the package name changes.\n\n"
        "Support timeline:\n"
        "  • Until June 2027: Both packages maintained\n"
        "  • After June 2027: Only 'darts' receives updates\n\n"
        "Migration guide: https://github.com/unit8co/darts/blob/master/MIGRATION.md\n"
        "=" * 70,
        FutureWarning,
        stacklevel=2,
    )
    sys.modules["u8darts._warned"] = True

# Re-export everything from darts transparently
from darts import *  # noqa: F401, F403
from darts import __version__ as _darts_version

__version__ = _darts_version

# Add metadata to help tools detect this is a redirect package
__deprecated__ = True
__redirect_to__ = "darts"
__sunset_date__ = "2027-06-30"
