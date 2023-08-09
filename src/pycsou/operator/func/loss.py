import pycsou.util.ptype as pyct

__all__ = [
    "shift_loss",
]


def shift_loss(
    op: pyct.OpT,
    data: pyct.NDArray = None,
) -> pyct.OpT:
    r"""
    Shift a functional :math:`f(x)` to a loss functional :math:`g(x) = f(x - c)`.

    Parameters
    ----------
    data: pyct.NDArray
        (M,) input data.

    Returns
    -------
    op: pyct.OpT
        (1, M) Loss functionial.
        If `data` is omitted, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)
