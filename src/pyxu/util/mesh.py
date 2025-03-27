import equinox as eqx
import jax
import jax.numpy as jnp

from ..typing import Array
from .dtype import fdtype
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


class AffineSpec(UniformSpec):
    r"""
    Multi-dimensional uniform sampling of an affine space.

    Defines points :math:`\bbx_{m} \in \bR^{D}` where each point lies on the regular lattice

    .. math::

       \bbx_{\bbm} = \bbA (\Delta_{\bbx} \odot \bbm) + \bbb,
       \qquad
       [\bbm]_{d} \in \{0,\ldots,M_{d}-1\},
       \bbA \in \bR^{K \times D},
       \bbb \in \bR^{D}

    """

    A: Array  # (K, D)
    b: Array  # (K,)

    def __init__(self, A, b, step, num):
        r"""
        Parameters
        ----------
        A: Array
            (K, D) linear transform
        b: Array
            (K,) offset
        step: tuple[float]
            \Delta_{\bbx} \in \bR_{+}^{D}
        num: tuple[int]
            (M1,...,MD) lattice size

        Scalars are broadcast to all dimensions.
        """
        super().__init__(start=0, step=step, num=num)

        assert A.ndim == 2
        K, D = A.shape
        assert D == self.ndim

        assert b.ndim == 1
        assert K == b.shape[0]

        self.A = jnp.asarray(A, dtype=fdtype())
        self.b = jnp.asarray(b, dtype=fdtype())

    def meshgrid(self) -> list[Array]:
        mesh_pre = jnp.stack(super().meshgrid(), axis=-1)  # (M1,...,MD, D)
        mesh_post = (
            jnp.tensordot(mesh_pre, self.A, axes=[[-1], [1]]) + self.b
        )  # (M1,...,MD, K)

        mesh = jnp.moveaxis(mesh_post, -1, 0)  # (K, M1,...,MD)
        return list(mesh)
