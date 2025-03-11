import jax
import jax.numpy as jnp
import jaxtyping as jt
import lineax as lx
import optax
import optax.tree_utils as otu

from ..abc import LinearOperator
from ..typing import Arrays


def svdvals(op: LinearOperator, k: int, **kwargs) -> jt.Scalar:
    """
    Compute leading singular values of a linear operator.

    The computation is based on the power method.

    Parameters
    ----------
    op: LinearOperator
        Operator for which singular values are desired.
    k: int
        Number of singular values to compute.

    Returns
    -------
    D: Array
        (k,) leading singular values of ``op``.
    """
    # TODO: upgrade to LOBPCG method; converges faster & computes (k >= 1) singular values.
    # Needs flattening/etc to be performed since jax.experimental.sparse.linalg.lobpcg() accepts a callable with a single array in/out.

    assert k == 1, "Only leading singular value can be computed (for now)."

    assert (op.dim_shape is not None) and (op.codim_shape is not None), (
        "in/out shapes must be known."
    )

    if op.codim_size <= op.dim_size:

        def f(y: Arrays) -> Arrays:
            return op.apply(op.adjoint(y))

        v0 = jax.tree.map(lambda sh: jnp.ones(sh.shape), op.codim_shape)
    else:

        def f(x: Arrays) -> Arrays:
            return op.adjoint(op.apply(x))

        v0 = jax.tree.map(lambda sh: jnp.ones(sh.shape), op.dim_shape)

    if "v0" in kwargs:
        pass
    else:
        kwargs.update(v0=v0)

    D_eig, _ = optax.power_iteration(f, **kwargs)

    D = jnp.sqrt(D_eig)
    return D


def pinv(op: LinearOperator, y: Arrays, tau: jt.Scalar = 0, **kwargs) -> Arrays:
    r"""
    Find the least-squares solution to the system :math:`\bbA \bbx = \bby`, where :math:`\bbA: \cI \to \cO` is a linear operator.

    The solution is estimated via the :math:`\tau`-dampened normal equation:

    .. math::

       (\bbA^{\ast} \bbA + \tau I) \bbx = \bbA^{\ast} \bby

    Parameters
    ----------
    y: Arrays
        Co-domain vector :math:`\bby \in \cO`.
    tau: Scalar
        Positive dampening factor regularizing the pseudo-inverse.
    kwargs: dict
        Extra parameters passed on to :py:class:`~lineax.CG`.
        You should specify `rtol` and `atol` at minimum.

    Returns
    -------
    x: Arrays
        t Pseudo-inverse solution :math:`\hat{\bbx} \in \cI`.
    diagnostics: lineax.Solution
        CG diagnostic information.
        Can be queried to learn more about the solution.
    """
    assert op.dim_info is not None

    lx_op = lx.FunctionLinearOperator(
        fn=lambda x: otu.tree_add_scalar_mul(
            op.adjoint(op.apply(x)),
            tau,
            x,
        ),
        input_structure=op.dim_info,
        tags=lx.positive_semidefinite_tag,
    )
    solver = lx.CG(**kwargs)
    solution = lx.linear_solve(lx_op, op.adjoint(y), solver)

    x = solution.value
    return x, solution
