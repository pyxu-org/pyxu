import importlib.util

CUPY_ENABLED = importlib.util.find_spec("cupy") is not None
