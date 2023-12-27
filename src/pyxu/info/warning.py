# Custom warnings used inside Pyxu.
import inspect
import warnings


class PyxuWarning(UserWarning):
    """
    Parent class of all warnings raised in Pyxu.
    """


class AutoInferenceWarning(PyxuWarning):
    """
    Use when a quantity was auto-inferenced with possible caveats.
    """


class PerformanceWarning(PyxuWarning):
    """
    Use for performance-related warnings.
    """


def warn_dask_perf(msg: str = None):
    """
    Issue a warning for DASK-related performance issues.

    This method is aware of its context and prints the name of the enclosing function/method which invoked it.

    Parameters
    ----------
    msg: str
        Custom warning message.
    """
    if msg is None:
        msg = "Sub-optimal performance for DASK inputs."

    # Get context
    my_frame = inspect.currentframe()
    up_frame = inspect.getouterframes(my_frame)[1]
    header = f"{up_frame.filename}:{up_frame.function}"

    msg = f"[{header}] {msg}"
    warnings.warn(msg, PerformanceWarning)


class PrecisionWarning(PyxuWarning):
    """
    Use for precision-related warnings.
    """


class DenseWarning(PyxuWarning):
    """
    Use for sparse-based algos which revert to dense arrays.
    """


class NonTransparentWarning(PyxuWarning):
    """
    Inform test suite runner of (safe) non-transparent function call.
    """


class BackendWarning(PyxuWarning):
    """
    Inform user of a backend-specific problem to be aware of.
    """


class ContributionWarning(PyxuWarning):
    """
    Use for warnings related to Pyxu plugins.
    """
