"""
Type annotations used throughout the project.
"""

import jaxtyping as jt

from .util.shape import ShapeStruct

Array = jt.Inexact[jt.Array, "..."]
Arrays = jt.PyTree[Array]
DimShape = CoDimShape = jt.PyTree[ShapeStruct]
DType = jt.DTypeLike
