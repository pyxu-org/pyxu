import math


# Basically a stripped-down version of jax.ShapeDtypeStruct().
class ShapeStruct:
    """
    A container for the shape attribute of an array.
    """

    __slots__ = ["shape"]

    def __init__(self, shape):
        self.shape = tuple(shape)

    size = property(lambda self: math.prod(self.shape))
    ndim = property(lambda self: len(self.shape))

    def __len__(self):
        return self.ndim

    def __repr__(self):
        return f"{type(self).__name__}(shape={self.shape})"

    __str__ = __repr__

    def __eq__(self, other):
        if not isinstance(other, ShapeStruct):
            return False
        else:
            return self.shape == other.shape

    def __hash__(self):
        return hash(self.shape)

    def __getitem__(self, key) -> int:
        return self.shape[key]
