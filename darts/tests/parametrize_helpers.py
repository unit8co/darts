from itertools import product as _product


def param_product(*iterables):
    """Return the Cartesian product of *iterables as a list for pytest.mark.parametrize."""
    return list(_product(*iterables))


def param_zip(*iterables, strict=False):
    """Return zip(*iterables) as a list for pytest.mark.parametrize."""
    return list(zip(*iterables, strict=strict))


def param_list(iterable):
    """Materialize an iterable as a list for pytest.mark.parametrize."""
    return list(iterable)
