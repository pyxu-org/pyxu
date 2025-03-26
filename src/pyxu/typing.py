"""
Type annotations used throughout the project.
"""

import typing as typ

import jaxtyping as jt

ShapeStruct = typ.TypeVar("ShapeStruct")

Array = jt.Inexact[jt.Array, "..."]
Arrays = jt.PyTree[Array]
DimShape = CoDimShape = jt.PyTree[ShapeStruct]
DType = jt.DTypeLike
