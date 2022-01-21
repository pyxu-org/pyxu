import numpy as np

import pycsou.abc.operator as pyco


class M:  # A pyco.Map-like
    # f: \bR -> \bR^{3}
    #      x -> [x, x, x]

    @property
    def shape(self) -> tuple[int, int]:
        return (3, 1)

    @property
    def dim(self) -> int:
        return self.shape[1]

    @property
    def codim(self) -> int:
        return self.shape[0]

    def lipschitz(self) -> float:
        return np.sqrt(3)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        y = arr * np.ones((3,))
        return y


class TestM(MapT):
    pass
