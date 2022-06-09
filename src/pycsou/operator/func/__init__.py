import typing as typ

import pycsou.abc.operator as pyco
import pycsou.util.ptype as pyct


def shift_loss(
    op: pyco.Map,
    data: typ.Optional[pyct.NDArray] = None,
) -> pyco.Func:
    """
    Shift a functional into a loss functional.

    Parameters
    ----------
    data: NDArray
        (N,) input data.

    Returns
    -------
    :py:class:`~pycsou.abc.operator.Func`
        Loss function.
        If `data = None`, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)
