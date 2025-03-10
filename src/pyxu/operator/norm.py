import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax.tree_utils as otu

from ..abc import ProximableOperator
from ..typing import Arrays, DimInfo
from ..util import fdtype


class SquaredL2Norm(ProximableOperator):
    r"""
    :math:`\ell^{2}_{2}`-norm.

    .. math::

       \norm{ \bbx }{2}^{2} = \sum_{i} \abs{ \bbx_{i} }^{2}
    """

    def __init__(self, dim_info: DimInfo = None):
        self.dim_info = dim_info
        self.codim_info = jax.ShapeDtypeStruct(shape=(), dtype=fdtype())

    def apply(self, x: Arrays) -> jt.Scalar:
        z = otu.tree_l2_norm(x, squared=True)
        return z

    def prox(self, x: Arrays, tau: jt.Scalar) -> Arrays:
        scale = 1 / (2 * tau + 1)
        z = otu.tree_scalar_mul(scale, x)
        return z


class L2Norm(ProximableOperator):
    r"""
    :math:`\ell_{2}`-norm.

    .. math::

       \norm{ \bbx }{2} = \sqrt{ \sum_{i} \abs{ \bbx_{i} }^{2} }
    """

    def __init__(self, dim_info: DimInfo = None):
        self.dim_info = dim_info
        self.codim_info = jax.ShapeDtypeStruct(shape=(), dtype=fdtype())

    def apply(self, x: Arrays) -> jt.Scalar:
        z = otu.tree_l2_norm(x, squared=False)
        return z

    def prox(self, x: Arrays, tau: jt.Scalar) -> Arrays:
        scale = 1 - tau / jnp.fmax(self.apply(x), tau)
        z = otu.tree_scalar_mul(scale, x)
        return z


class L1Norm(ProximableOperator):
    r"""
    :math:`\ell_{1}`-norm.

    .. math::

       \norm{ \bbx }{1} = \sum_{i} \abs{ \bbx_{i} }
    """

    def __init__(self, dim_info: DimInfo = None):
        self.dim_info = dim_info
        self.codim_info = jax.ShapeDtypeStruct(shape=(), dtype=fdtype())

    def apply(self, x: Arrays) -> jt.Scalar:
        z = otu.tree_l1_norm(x)
        return z

    def prox(self, x: Arrays, tau: jt.Scalar) -> Arrays:
        z = otu.tree_mul(
            jax.tree.map(jnp.sign, x),
            jax.tree.map(lambda _: jnp.clip(jnp.abs(_) - tau, min=0), x),
        )
        return z


class LInfinityNorm(ProximableOperator):
    r"""
    :math:`\ell_{\infty}`-norm.

    .. math::

       \norm{ \bbx }{\infty} = \max_{i} \abs{ \bbx_{i} }
    """

    def __init__(self, dim_info: DimInfo = None):
        self.dim_info = dim_info
        self.codim_info = jax.ShapeDtypeStruct(shape=(), dtype=fdtype())

    def apply(self, x: Arrays) -> jt.Scalar:
        z = otu.tree_linf_norm(x)
        return z

    def prox(self, x: Arrays, tau: jt.Scalar) -> Arrays:
        raise NotImplementedError  # TODO
