"""
Custom warnings used inside Pycsou.
"""


class PrecisionWarning(UserWarning):
    """
    Use for precision-related warnings.
    """


class DenseWarning(UserWarning):
    """
    Use for sparse-based algos which revert to dense arrays.
    """


class NonTransparentWarning(UserWarning):
    """
    [Dev-team only] Inform test suite runner of (safe) non-transparent function call.
    """


class BackendWarning(UserWarning):
    """
    Inform user of a backend-specific problem to be aware of.
    """
