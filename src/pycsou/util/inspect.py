import collections.abc as cabc
import inspect

__all__ = [
    "parse_params",
]


def parse_params(func, *args, **kwargs) -> cabc.Mapping:
    """
    Get function parameterization.

    Returns
    -------
    params: dict
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
