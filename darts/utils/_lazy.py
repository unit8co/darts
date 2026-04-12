"""Shared helper for PEP 562 lazy module loading in ``__init__.py`` files."""

import importlib
from collections.abc import Callable
from typing import Any


def setup_lazy_imports(
    lazy_imports: dict[str, str | tuple[str, str | None]],
    module_name: str,
    module_globals: dict[str, Any],
    extra_attrs: dict[str, Callable[[], Any]] | None = None,
) -> tuple[list[str], Callable[[str], Any], Callable[[], list[str]]]:
    """Create ``__all__``, ``__getattr__``, and ``__dir__`` for a lazy package.

    Parameters
    ----------
    lazy_imports
        Mapping of public attribute name to either:

        - a module path string (required dependency), or
        - a 2-tuple ``(module_path, optional_dep_name)`` where
          *optional_dep_name* is a human-readable dependency label
          (e.g. ``"(Py)Torch"``).  When the import fails and
          *optional_dep_name* is not ``None``, a
          :class:`~darts.utils.utils.NotImportedModule` sentinel is
          returned instead of raising.
    module_name
        The ``__name__`` of the calling package (for error messages).
    module_globals
        The ``globals()`` dict of the calling package.  Resolved
        attributes are cached here so ``__getattr__`` is only invoked
        once per name.
    extra_attrs
        Optional mapping of additional attribute names to zero-arg
        callables that produce their values on first access.  Useful
        for derived attributes (e.g. ``TIME_DEPENDENT_METRICS``).

    Returns
    -------
    tuple[list[str], Callable, Callable]
        ``(__all__, __getattr__, __dir__)`` ready to be assigned at
        module level.
    """
    # attrs are added to `__all__` of `__init__.py`
    all_names = list(lazy_imports.keys())
    if extra_attrs:
        all_names.extend(extra_attrs.keys())

    # `__getattr__` is called when accessing module / package attributes
    # e.g., `from darts.models import XXX` for the `darts.models` package
    def __getattr__(name: str) -> Any:
        if name in lazy_imports:
            # import attrs from darts package and cache it
            entry = lazy_imports[name]
            if isinstance(entry, str):
                module_path, optional_dep = entry, None
            else:
                module_path, optional_dep = entry

            try:
                module = importlib.import_module(module_path)
                value = getattr(module, name)
            except (ModuleNotFoundError, ImportError):
                # attr is not available; if it is an optional dependency, wrap it in
                # the `NotImportedModule`. Otherwise, raise an exception
                if optional_dep is not None:
                    from darts.utils.utils import NotImportedModule

                    value = NotImportedModule(module_name=optional_dep, warn=False)
                else:
                    raise
        elif extra_attrs is not None and name in extra_attrs:
            # special attrs that are not imported but computed within `__init__.py`
            value = extra_attrs[name]()
        else:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

        # cache the attr under globals to only invoke `__getattr__` only once per name
        module_globals[name] = value
        return value

    # attrs are added to `__dir__` of `__init__.py`
    def __dir__() -> list[str]:
        return all_names

    return all_names, __getattr__, __dir__
