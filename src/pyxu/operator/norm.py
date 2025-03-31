import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax.tree_utils as otu
import optimistix

import pyxu.abc as pxa
import pyxu.typing as pxt
import pyxu.util as pxu


class SquaredL2Norm(pxa.ProximableOperator):
    r"""
    :math:`\ell^{2}_{2}`-norm.

    .. math::

       \norm{ \bbx }{2}^{2} = \sum_{i} \abs{ \bbx_{i} }^{2}
    """

    def __init__(self, dim_shape: pxt.DimShape = None):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        z = otu.tree_l2_norm(x, squared=True)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        scale = 1 / (2 * tau + 1)
        z = otu.tree_scalar_mul(scale, x)
        return z


class L2Norm(pxa.ProximableOperator):
    r"""
    :math:`\ell_{2}`-norm.

    .. math::

       \norm{ \bbx }{2} = \sqrt{ \sum_{i} \abs{ \bbx_{i} }^{2} }
    """

    def __init__(self, dim_shape: pxt.DimShape = None):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        z = otu.tree_l2_norm(x, squared=False)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        scale = 1 - tau / jnp.fmax(self.apply(x), tau)
        z = otu.tree_scalar_mul(scale, x)
        return z


class L1Norm(pxa.ProximableOperator):
    r"""
    :math:`\ell_{1}`-norm.

    .. math::

       \norm{ \bbx }{1} = \sum_{i} \abs{ \bbx_{i} }
    """

    def __init__(self, dim_shape: pxt.DimShape = None):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        z = otu.tree_l1_norm(x)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        z = otu.tree_mul(
            jax.tree.map(jnp.sign, x),
            jax.tree.map(lambda _: jnp.clip(jnp.abs(_) - tau, min=0), x),
        )
        return z


class LInfinityNorm(pxa.ProximableOperator):
    r"""
    :math:`\ell_{\infty}`-norm.

    .. math::

       \norm{ \bbx }{\infty} = \max_{i} \abs{ \bbx_{i} }
    """

    def __init__(self, dim_shape: pxt.DimShape = None):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        z = otu.tree_linf_norm(x)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        def fn(s, args) -> jt.Scalar:
            x, tau = args
            y = jax.tree.map(lambda _: jnp.fmax(0, jnp.fabs(_) - s), x)
            y = otu.tree_sum(y) - tau
            return y

        s_min, s_max = 0, self.apply(x)
        solution = optimistix.root_find(
            fn,
            solver=optimistix.Bisection(
                rtol=1e-4,
                atol=1e-6,
            ),
            y0=0.5 * (s_min + s_max),
            args=(x, tau),
            options=dict(
                lower=s_min,
                upper=s_max,
            ),
        )
        s = solution.value

        z = otu.tree_mul(
            jax.tree.map(jnp.sign, x),
            jax.tree.map(lambda _: jnp.fmin(jnp.fabs(_), s), x),
        )
        return z
