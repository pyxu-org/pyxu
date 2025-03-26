import collections.abc as cabc

import jax
import jax.numpy as jnp

from .dtype import idtype


def broadcast_seq(x, N: int = None) -> tuple:
    """
    Broadcast `x` to a tuple of length `N`.

    If `N` is omitted, then no broadcasting takes place, only tupling.
    """
    if isinstance(x, cabc.Iterable):
        y = tuple(x)
    else:
        y = (x,)

    if N is not None:
        if len(y) == 1:
            y *= N  # broadcast
        assert len(y) == N

    return y


@jax.jit
def next_fast_len(target):
    """
    Find a good FFT length.

    Notes
    -----
    JAX does not expose a ``next_fast_len()`` function to optimize FFT runtimes.
    (See https://github.com/jax-ml/jax/discussions/15200)

    Since all FFT backends support factors (2,3,5), it is safe to use a 5-smooth number as FFT length.

    Parameters
    ----------
    target: int | jax.Array
        Length to start searching from.

    Returns
    -------
    out: jax.Array
        The first 5-smooth number greater than or equal to target.
        Has the same shape as `target`.
    """
    exp_max = 10
    factors = [2, 3, 5]

    exp_mesh = jnp.meshgrid(
        *[jnp.arange(exp_max + 1, dtype=idtype())] * len(factors),
        sparse=True,
    )
    good_size = jax.tree.reduce(
        jnp.multiply,
        [f**e for (f, e) in zip(factors, exp_mesh)],
    )
    good_size = jnp.sort(good_size.ravel())

    idx = jnp.searchsorted(good_size, target, side="left")
    out = good_size[idx]
    return out
