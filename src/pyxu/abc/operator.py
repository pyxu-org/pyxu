import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax.tree_utils as otu

import pyxu.typing as pxt
import pyxu.util as pxu


class Operator(eqx.Module):
    r"""
    Abstract operator :math:`f: \cI \to \cO`.
    """

    # Optional fields users can populate in __init__() to provide cues on (\cI, \cO).
    dim_shape: pxt.DimShape = eqx.field(  # \cI
        default=None,
        init=False,
    )
    codim_shape: pxt.CoDimShape = eqx.field(  # \cO
        default=None,
        init=False,
    )

    @property
    def dim_rank(self) -> jt.PyTree[int]:
        return jax.tree.map(lambda _: _.ndim, self.dim_shape)

    @property
    def dim_size(self) -> int:
        return otu.tree_sum(jax.tree.map(lambda _: _.size, self.dim_shape))

    @property
    def codim_rank(self) -> jt.PyTree[int]:
        return jax.tree.map(lambda _: _.ndim, self.codim_shape)

    @property
    def codim_size(self) -> int:
        return otu.tree_sum(jax.tree.map(lambda _: _.size, self.codim_shape))

    def apply(self, x: pxt.Arrays) -> pxt.Arrays:
        r"""
        Evaluate :math:`f: \cI \to \cO` at specified point :math:`x`.
        """
        raise NotImplementedError

    def __call__(self, x: pxt.Arrays) -> pxt.Arrays:
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

    def prox(self, x: pxt.Arrays, tau: jt.Scalar) -> pxt.Arrays:
        r"""
        Evaluate :math:`\prox_{\tau f}: \cI \to \cI` at specified point :math:`x`.
        """
        raise NotImplementedError

    def fenchel_prox(self, x: pxt.Arrays, sigma: jt.Scalar) -> pxt.Arrays:
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

        def moreau_apply(
            op: ProximableOperator,
            mu: jt.Scalar,
            x: pxt.Arrays,
        ) -> jt.Scalar:
            fdtype = pxu.TranslateDType(x.dtype).to_float()
            mu = jnp.asarray(mu, dtype=fdtype)

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
        _apply = jax.custom_vjp(moreau_apply)

        def _apply_fwd(
            op: ProximableOperator,
            mu: jt.Scalar,
            x: pxt.Arrays,
        ):
            primal_out, f_vjp = jax.vjp(moreau_apply, op, mu, x)
            return primal_out, ((op, mu, x), f_vjp)

        def _apply_bwd(residual, v: jt.Scalar) -> tuple[pxt.Arrays]:
            (op, mu, x), f_vjp = residual
            vjp_op, vjp_mu, _ = f_vjp(v)  # _ optimized away with jit()
            vjp_x = otu.tree_scalar_mul(
                v / mu,
                otu.tree_sub(x, op.prox(x, tau=mu)),
            )
            return (vjp_op, vjp_mu, vjp_x)

        _apply.defvjp(fwd=_apply_fwd, bwd=_apply_bwd)
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
                self.dim_shape = op.dim_shape
                self.codim_shape = op.codim_shape

                self.op = op
                self.mu = mu

            def apply(self, x: pxt.Arrays) -> jt.Scalar:
                return _apply(self.op, self.mu, x)

        return MoreauEnvelope(op=self, mu=mu)


class LinearOperator(Operator):
    r"""
    Abstract linear operator :math:`f: \cI \to \cO`.

    ``LinearOperator`` allows users to (optionally) provide a custom rule to compute ``adjoint()``, the latter being more efficient than using autodiff rules.

    `dim_shape` and `codim_shape` fields should typically be specified for linear operators.

    When a custom implementation of ``adjoint()`` is provided, the class definition should be decorated using :py:func:`~pyxu.abc.register_linop_vjp` to plug into JAX's autodiff system.
    """

    def adjoint(self, y: pxt.Arrays) -> pxt.Arrays:
        r"""
        Evaluate :math:`f^{\adjoint}: \cO \to \cI` at specified point :math:`y`.

        Notes
        -----
        The auto-inferred adjoint rule works for both real- and complex-valued operators.
        In the latter case however, ``adjoint()`` only accepts complex-valued inputs.
        """
        dtype = otu.tree_dtype(y, mixed_dtype_handler="promote")
        dim_info = jax.tree.map(
            lambda _: jax.ShapeDtypeStruct(shape=_.shape, dtype=dtype),
            self.dim_shape,
        )
        f_adjoint = jax.linear_transpose(self.apply, dim_info)

        # f_adjoint() computes (A.T y) instead of (A.H y).
        # But (A.H y) = conj(A.T conj(y)), so the true adjoint can still be computed for complex-valued operators.
        y_conj = jax.tree.map(jnp.conjugate, y)
        x_conj, *_ = f_adjoint(y_conj)
        x = jax.tree.map(jnp.conjugate, x_conj)
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
