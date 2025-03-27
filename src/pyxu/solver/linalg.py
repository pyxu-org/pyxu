import jax
import jaxtyping as jt
import lineax as lx
import optax.tree_utils as otu

from ..abc import LinearOperator
from ..typing import Arrays


def pinv(
    op: LinearOperator,
    y: Arrays,
    tau: jt.Scalar = 0,
    **kwargs,
) -> Arrays:
    r"""
    Find the least-squares solution to the system :math:`\bbA \bbx = \bby`, where :math:`\bbA: \cI \to \cO` is a linear operator.

    The solution is estimated via the :math:`\tau`-dampened normal equation:

    .. math::

       (\bbA^{\ast} \bbA + \tau I) \bbx = \bbA^{\ast} \bby

    Parameters
    ----------
    y: Arrays
        Co-domain vector :math:`\bby \in \cO`.

        If `op` is a complex-valued operator, then `y` must be complex-valued: an error will occur otherwise.
    tau: Scalar
        Positive dampening factor regularizing the pseudo-inverse.
    kwargs: dict
        Extra parameters passed on to :py:class:`~lineax.CG`.
        You should specify `rtol` and `atol` at minimum.

    Returns
    -------
    x: Arrays
        Pseudo-inverse solution :math:`\hat{\bbx} \in \cI`.
    diagnostics: lineax.Solution
        CG diagnostic information.
        Can be queried to learn more about the solution.
    """
    assert op.dim_shape is not None
    dtype = otu.tree_dtype(y, mixed_dtype_handler="promote")
    dim_info = jax.tree.map(
        lambda _: jax.ShapeDtypeStruct(shape=_.shape, dtype=dtype),
        op.dim_shape,
    )

    lx_op = lx.FunctionLinearOperator(
        fn=lambda x: otu.tree_add_scalar_mul(
            op.adjoint(op.apply(x)),
            tau,
            x,
        ),
        input_structure=dim_info,
        tags=lx.positive_semidefinite_tag,
    )
    solver = lx.CG(**kwargs)
    diagnostics = lx.linear_solve(lx_op, op.adjoint(y), solver)

    x = diagnostics.value
    return x, diagnostics
