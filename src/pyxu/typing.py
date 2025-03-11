"""
Type annotations used throughout the project.
"""

import jax
import jaxtyping as jt

Array = jt.Inexact[jt.Array, "..."]
Arrays = jt.PyTree[Array]
DimInfo = CoDimInfo = jt.PyTree[jax.ShapeDtypeStruct]
