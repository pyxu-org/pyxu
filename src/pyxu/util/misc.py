import collections.abc as cabc
import dataclasses

import jax


@dataclasses.dataclass(frozen=True)
class UniformSpec:
    r"""
    Multi-dimensional uniform mesh specifier.

    Defines points :math:`\bbx_{m} \in \bR^{D}` where each point lies on the regular lattice

    .. math::

       \bbx_{\bbm} = \bbx_{0} + \Delta_{\bbx} \odot \bbm,
       \qquad
       [\bbm]_{d} \in \{0,\ldots,M_{d}-1\},

    """

    start: tuple[float]
    step: tuple[float]
    num: tuple[int]

    def __init__(self, start, step, num):
        start = jax.tree.map(float, broadcast_seq(start, None))

        step = jax.tree.map(float, broadcast_seq(step, None))
        assert jax.tree.all(jax.tree.map(lambda _: _ > 0, step))

        num = jax.tree.map(int, broadcast_seq(num, None))
        assert jax.tree.all(jax.tree.map(lambda _: _ > 0, num))

        D = max(map(len, [start, step, num]))

        object.__setattr__(self, "start", broadcast_seq(start, D))
        object.__setattr__(self, "step", broadcast_seq(step, D))
        object.__setattr__(self, "num", broadcast_seq(num, D))


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
