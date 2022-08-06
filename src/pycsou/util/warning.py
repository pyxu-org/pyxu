"""
Custom warnings used inside Pycsou.
"""


class PrecisionWarning(UserWarning):
    """
    Use for precision-related warnings.
    """

    pass


class DenseWarning(UserWarning):
    """
    Use for sparse-based algos which revert to dense arrays.
    """


class NonTransparentWarning(UserWarning):
    """
    [Dev-team only] Inform test suite runner of (safe) non-transparent function call.
    """
