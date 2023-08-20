import collections.abc as cabc
import importlib
import inspect
import types

__all__ = [
    "import_module",
    "parse_params",
]


def parse_params(func, *args, **kwargs) -> cabc.Mapping:
    """
    Get function parameterization.

    Returns
    -------
    params: ~collections.abc.Mapping
        (key, value) params as seen in body of `func` when called via `func(*args, **kwargs)`.
    """
    sig = inspect.Signature.from_callable(func)
    f_args = sig.bind(*args, **kwargs)
    f_args.apply_defaults()

    params = dict(
        zip(f_args.arguments.keys(), f_args.args),  # positional arguments
        **f_args.kwargs,
    )
    return params


def import_module(name: str, fail_on_error: bool = True) -> types.ModuleType:
    """
    Load a Python module dynamically.
    """
    try:
        pkg = importlib.import_module(name)
    except ModuleNotFoundError:
        if fail_on_error:
            raise
        else:
            pkg = None
    return pkg
