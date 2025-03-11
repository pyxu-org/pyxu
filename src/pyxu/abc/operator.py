import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax.tree_utils as otu

from ..typing import Arrays, CoDimInfo, DimInfo


class Operator(eqx.Module):
    r"""
    Abstract operator :math:`f: \cI \to \cO`.
    """

    # Optional fields users can populate in __init__() to provide cues on (\cI, \cO).
    dim_info: DimInfo = eqx.field(  # \cI
        default=None,
        init=False,
        static=True,
    )
    codim_info: CoDimInfo = eqx.field(  # \cO
        default=None,
        init=False,
        static=True,
    )

    @property
    def dim_shape(self) -> jt.PyTree[tuple[int]]:
        return jax.tree.map(lambda _: _.shape, self.dim_info)

    @property
    def dim_rank(self) -> jt.PyTree[int]:
        return jax.tree.map(lambda _: _.ndim, self.dim_info)

    @property
    def dim_size(self) -> jt.PyTree[int]:
        return jax.tree.map(lambda _: _.size, self.dim_info)

    @property
    def codim_shape(self) -> jt.PyTree[tuple[int]]:
        return jax.tree.map(lambda _: _.shape, self.codim_info)

    @property
    def codim_rank(self) -> jt.PyTree[int]:
        return jax.tree.map(lambda _: _.ndim, self.codim_info)

    @property
    def codim_size(self) -> jt.PyTree[int]:
        return jax.tree.map(lambda _: _.size, self.codim_info)

    def apply(self, x: Arrays) -> Arrays:
        r"""
        Evaluate :math:`f: \cI \to \cO` at specified point :math:`x`.
        """
        raise NotImplementedError

    def __call__(self, x: Arrays) -> Arrays:
        r"""
        Evaluate :math:`f: \cI \to \cO` at specified point :math:`x`.

        Alias of ``apply()``.
        """
        return self.apply(x)


class ProximableOperator(Operator):
    r"""
    Abstract proximable functional :math:`f: \cI \to \bR`.

    Notes
    -----
    * For :math:`\tau > 0`, the *proximal operator* of :math:`f: \cI \to \bR` is defined as:

      .. math::

         \prox_{\tau f}: \cI \to \cI
                           x \to \argmin_{z \in \cI} f(z) + \frac{1}{2\tau} \norm{x - z}{2}^{2}

    * The *Fenchel conjugate of :math:`f: \cI \to \cR` is the function :math:`f^{\ast}: \cI \to \bR`, defined as:

      .. math::

         f^{\ast}: \cI \to \bR
                     x \to \max_{z \in \cI} \innerProduct{x}{z} - f(z)

      From Moreau's identity, its proximal operator is given by:

      .. math::

         \prox_{\sigma f^{\ast}}: \cI \to \cI
                                    x \to x - \sigma \prox_{\frac{f}{\sigma}}(\frac{x}{\sigma})

    * The *Moreau envelope* of :math:`f: \cI \to \bR` is the function :math:`f^{\mu}: \cI \to \bR`, defined as:

      .. math::

         f^{\mu}: \cI \to \bR
                    x \to f(y) + \frac{1}{2\mu} \norm{x - y}{2}^{2}
                          y = \prox_{\mu f}(x)

      :math:`f^{\mu}` is differentiable, with gradient closed form:

      .. math::

         \nabla f^{\mu}: \cI \to \cI
                           x \to \frac{1}{\mu} ( x - \prox_{\mu f}(x) ).
    """

    def prox(self, x: Arrays, tau: jt.Scalar) -> Arrays:
        r"""
        Evaluate :math:`\prox_{\tau f}: \cI \to \cI` at specified point :math:`x`.
        """
        raise NotImplementedError

    def fenchel_prox(self, x: Arrays, sigma: jt.Scalar) -> Arrays:
        r"""
        Evaluate :math:`\prox_{\sigma f^{\ast}}: \cI \to \cI` at specified point :math:`x`.
        """
        prox = self.prox(
            x=otu.tree_scalar_mul(1 / sigma, x),
            tau=1 / sigma,
        )
        y = otu.tree_add_scalar_mul(x, -sigma, prox)
        return y

    def moreau_envelope(self, mu: jt.Scalar) -> Operator:
        r"""
        Moreau envelope :math:`f^{\mu}: \cI \to \bR`.
        """

        @jax.custom_vjp
        def apply(
            op: ProximableOperator,
            mu: jt.Scalar,
            x: Arrays,
        ) -> jt.Scalar:
            prox = op.prox(x=x, tau=mu)
            y = otu.tree_sum(
                jax.tree.map(
                    lambda a, b: jnp.sum(jnp.abs(a - b) ** 2),
                    *(x, prox),
                )
            )

            z = op.apply(prox) + y / (2 * mu)
            return z

        # Special gradient rule ===============================================
        def apply_fwd(
            op: ProximableOperator,
            mu: jt.Scalar,
            x: Arrays,
        ):
            return apply(op, mu, x), (op, mu, x)

        def apply_bwd(residual, v: jt.Scalar) -> tuple[Arrays]:
            op, mu, x = residual
            vjp_x = otu.tree_scalar_mul(
                v / mu,
                otu.tree_sub(x, op.prox(x, tau=mu)),
            )
            vjp_op = None
            vjp_mu = None
            return (vjp_op, vjp_mu, vjp_x)

        apply.defvjp(fwd=apply_fwd, bwd=apply_bwd)
        # =====================================================================

        class MoreauEnvelope(Operator):
            op: ProximableOperator
            mu: jt.Scalar

            def __init__(
                self,
                op: ProximableOperator,
                mu: jt.Scalar,
            ):
                # forward information from `op`.
                self.dim_info = op.dim_info
                self.codim_info = op.codim_info

                self.op = op
                self.mu = mu

            def apply(self, x: Arrays) -> jt.Scalar:
                return apply(self.op, self.mu, x)

        return MoreauEnvelope(op=self, mu=mu)


class LinearOperator(Operator):
    r"""
    Abstract linear operator :math:`f: \cI \to \cO`.

    ``LinearOperator`` allows users to (optionally) provide a custom rule to compute ``adjoint()``, the latter being more efficient than using autodiff rules.

    `dim_info` and `codim_info` fields should typically be specified for linear operators.

    When a custom implementation of ``adjoint()`` is provided, the class definition should be decorated using :py:func:`~pyxu.abc.register_linop_vjp` to plug into JAX's autodiff system.
    """

    def adjoint(self, y: Arrays) -> Arrays:
        r"""
        Evaluate :math:`f^{\adjoint}: \cO \to \cI` at specified point :math:`y`.
        """
        f_adjoint = jax.linear_transpose(self.apply, self.dim_info)
        x, *_ = f_adjoint(y)
        return x


def register_linop_vjp(klass: LinearOperator):
    """
    Register a LinearOperator with JAX's autodiff system.

    This class wrapper ensures:
    * :py:meth:`~pyxu.abc.LinearOperator.adjoint` is used when invoking ``jax.vjp(LinearOperator.apply)``.
    * :py:meth:`~pyxu.abc.LinearOperator.apply` is used when invoking ``jax.vjp(LinearOperator.adjoint)``.

    If :py:func:`pyxu.abc.register_linop_vjp` is not used, then `jax.vjp()` calls will trace computations instead.
    """

    # custom_vjp() not necessary if user did not override default adjoint().
    if klass.adjoint == LinearOperator.adjoint:
        return klass

    # custom_vjp() only works with functions, not methods: it cannot be applied to klass.[apply,adjoint]() directly.
    # As a workaround, we define the functions [_apply,_adjoint](), then bind the instance to them via partialmethod().

    # vjp(apply) override =====================================================
    @jax.custom_vjp
    def _apply(self: LinearOperator, x):
        return klass.apply(self, x)

    def _apply_fwd(self: LinearOperator, x):
        primal_out, f_vjp = jax.vjp(klass.apply, self, x)
        return primal_out, (self, f_vjp)

    def _apply_bwd(residual, y):
        self, f_vjp = residual
        vjp_op, _ = f_vjp(y)  # _ optimized away with jit()
        return (vjp_op, klass.adjoint(self, y))

    _apply.defvjp(_apply_fwd, _apply_bwd)
    # =========================================================================

    # vjp(adjoint) override ===================================================
    @jax.custom_vjp
    def _adjoint(self: LinearOperator, y):
        return klass.adjoint(self, y)

    def _adjoint_fwd(self: LinearOperator, y):
        primal_out, f_vjp = jax.vjp(klass.adjoint, self, y)
        return primal_out, (self, f_vjp)

    def _adjoint_bwd(residual, x):
        self, f_vjp = residual
        vjp_op, _ = f_vjp(x)  # _ optimized away with jit()
        return (vjp_op, klass.apply(self, x))

    _adjoint.defvjp(_adjoint_fwd, _adjoint_bwd)
    # =========================================================================

    wrapper_klass = type(
        f"VJPCompatible_{klass.__name__}",
        (klass,),
        dict(
            apply=functools.partialmethod(_apply),
            adjoint=functools.partialmethod(_adjoint),
        ),
    )

    return wrapper_klass
