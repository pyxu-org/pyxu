import pycsou.util.ptype as pyct

__all__ = [
    "shift_loss",
]


def shift_loss(
    op: pyct.OpT,
    data: pyct.NDArray = None,
) -> pyct.OpT:
    """
    Shift a functional into a loss functional.

    Parameters
    ----------
    data: pyct.NDArray
        (N,) input data.

    Returns
    -------
    op: pyct.OpT
        Loss functionial.
        If `data = None`, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)
