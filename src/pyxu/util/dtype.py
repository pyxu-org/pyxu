import jax.numpy as jnp
import numpy.typing as npt


class TranslateDType:
    """
    float/complex dtype translator.
    """

    map_to_float: dict = {
        jnp.dtype(jnp.int32): jnp.dtype(jnp.float32),
        jnp.dtype(jnp.int64): jnp.dtype(jnp.float64),
        jnp.dtype(jnp.float32): jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float64): jnp.dtype(jnp.float64),
        jnp.dtype(jnp.complex64): jnp.dtype(jnp.float32),
        jnp.dtype(jnp.complex128): jnp.dtype(jnp.float64),
    }
    map_from_float: dict = {
        (jnp.dtype(jnp.float32), "i"): jnp.dtype(jnp.int32),
        (jnp.dtype(jnp.float64), "i"): jnp.dtype(jnp.int64),
        (jnp.dtype(jnp.float32), "f"): jnp.dtype(jnp.float32),
        (jnp.dtype(jnp.float64), "f"): jnp.dtype(jnp.float64),
        (jnp.dtype(jnp.float32), "c"): jnp.dtype(jnp.complex64),
        (jnp.dtype(jnp.float64), "c"): jnp.dtype(jnp.complex128),
    }

    def __init__(self, dtype: npt.DTypeLike):
        dtype = jnp.dtype(dtype)
        assert dtype in self.map_to_float
        self._fdtype = self.map_to_float[dtype]

    def to_int(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "i")]

    def to_float(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "f")]

    def to_complex(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "c")]


def fdtype() -> np.dtype:
    """
    Determine the floating point type used by JAX.

    The value depends on the `jax_enable_x64` config.
    """
    return jnp.result_type(0.0)


def cdtype() -> np.dtype:
    """
    Determine the complex type used by JAX.

    The value depends on the `jax_enable_x64` config.
    """
    return jnp.result_type(0.0j)
