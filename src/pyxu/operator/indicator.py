import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax.projections as opj

import pyxu.abc as pxa
import pyxu.typing as pxt
import pyxu.util as pxu

from .norm import L1Norm, L2Norm, LInfinityNorm


class L2Ball(pxa.ProximableOperator):
    r"""
    Indicator function of the :math:`\ell_{2}`-ball.

    .. math::

       \iota_{2}^{r}(\bbx)
       =
       \begin{cases}
           0      & \norm{ \bbx }{2} \le r \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \prox_{\tau\, \iota_{2}^{r}}(\bbx)
       =
       \bbx - \prox_{r\, \ell_{2}}(\bbx)
    """

    radius: jt.Scalar

    def __init__(self, dim_shape: pxt.DimShape = None, radius: jt.Scalar = 1.0):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())
        self.radius = radius

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        norm = L2Norm().apply(x)
        z = jnp.where(norm <= self.radius, 0.0, jnp.inf)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        z = opj.projection_l2_ball(x, self.radius)
        return z


class L1Ball(pxa.ProximableOperator):
    r"""
    Indicator function of the :math:`\ell_{1}`-ball.

    .. math::

       \iota_{1}^{r}(\bbx)
       =
       \begin{cases}
           0      & \norm{ \bbx }{1} \le r \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \prox_{\tau\, \iota_{1}^{r}}(\bbx)
       =
       \bbx - \prox_{r\, \ell_{\infty}}(\bbx)
    """

    radius: jt.Scalar

    def __init__(self, dim_shape: pxt.DimShape = None, radius: jt.Scalar = 1.0):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())
        self.radius = radius

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        norm = L1Norm().apply(x)
        z = jnp.where(norm <= self.radius, 0.0, jnp.inf)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        z = opj.projection_l1_ball(x, self.radius)
        return z


class LInfinityBall(pxa.ProximableOperator):
    r"""
    Indicator function of the :math:`\ell_{\infty}`-ball.

    .. math::

       \iota_{\infty}^{r}(\bbx)
       =
       \begin{cases}
           0      & \norm{ \bbx }{\infty} \le r \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \prox_{\tau\, \iota_{\infty}^{r}}(\bbx)
       =
       \bbx - \prox_{r\, \ell_{1}}(\bbx)
    """

    radius: jt.Scalar

    def __init__(self, dim_shape: pxt.DimShape = None, radius: jt.Scalar = 1.0):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())
        self.radius = radius

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        norm = LInfinityNorm().apply(x)
        z = jnp.where(norm <= self.radius, 0.0, jnp.inf)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        z = opj.projection_linf_ball(x, self.radius)
        return z


class PositiveOrthant(pxa.ProximableOperator):
    r"""
    Indicator function of the positive orthant.

    .. math::

       \iota_{+}(\bbx)
       =
       \begin{cases}
           0      & \min{ \bbx } \ge 0, \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \prox_{\tau\, \iota_{+}}(\bbx)
       =
       \max(\bbx, 0)
    """

    def __init__(self, dim_shape: pxt.DimShape = None):
        self.dim_shape = dim_shape
        self.codim_shape = pxu.ShapeStruct(shape=())

    def apply(self, x: pxt.Arrays) -> jt.Scalar:
        offset = (x < 0).sum()
        z = jnp.where(offset > 0, jnp.inf, 0.0)
        return z

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        z = jax.tree.map(lambda _: jnp.clip(_, min=0), x)
        return z
