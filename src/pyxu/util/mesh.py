import equinox as eqx
import jax
import jax.numpy as jnp

from ..typing import Array
from .misc import broadcast_seq


class UniformSpec(eqx.Module):
    r"""
    Multi-dimensional uniform mesh specifier.

    Defines points :math:`\bbx_{m} \in \bR^{D}` where each point lies on the regular lattice

    .. math::

       \bbx_{\bbm} = \bbx_{0} + \Delta_{\bbx} \odot \bbm,
       \qquad
       [\bbm]_{d} \in \{0,\ldots,M_{d}-1\}

    """

    start: tuple[float] = eqx.field(static=True)
    step: tuple[float] = eqx.field(static=True)
    num: tuple[int] = eqx.field(static=True)

    def __init__(self, start, step, num):
        r"""
        Parameters
        ----------
        start: tuple[float]
            \bbx_{0} \in \bR^{D}
        step: tuple[float]
            \Delta_{\bbx} \in \bR_{+}^{D}
        num: tuple[int]
            (M1,...,MD) lattice size

        Scalars are broadcast to all dimensions.
        """
        start = jax.tree.map(float, broadcast_seq(start, None))

        step = jax.tree.map(float, broadcast_seq(step, None))
        assert jax.tree.all(jax.tree.map(lambda _: _ > 0, step))

        num = jax.tree.map(int, broadcast_seq(num, None))
        assert jax.tree.all(jax.tree.map(lambda _: _ > 0, num))

        D = max(map(len, [start, step, num]))

        self.start = broadcast_seq(start, D)
        self.step = broadcast_seq(step, D)
        self.num = broadcast_seq(num, D)

    @property
    def ndim(self) -> int:
        D = len(self.start)
        return D

    def meshgrid(self) -> list[Array]:
        r"""
        Construct lattice coordinates.

        Returns
        -------
        mesh: list[Array]
            (M1,...,MD) coordinates per dimension XYZ...

            mesh[d][m1,...,md] = \bbx_{m1,...,md}[d]
        """
        mesh_1d = [None] * self.ndim
        for d in range(self.ndim):
            mesh_1d[d] = self.start[d] + self.step[d] * jnp.arange(self.num[d])

        mesh = jnp.meshgrid(*mesh_1d, indexing="ij")
        return mesh
