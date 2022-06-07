import typing as typ
import warnings

import dask.array as da
import numpy as np
import scipy.ndimage as snd
import scipy.sparse.linalg as spls

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

if pycd.CUPY_ENABLED:
    import cupy as cp


class SquaredL2Norm(pyco.DiffFunc):
    def __init__(self, shape: pyct.ShapeOrDim = None):
        super(SquaredL2Norm, self).__init__(shape=(1, None))
        self._diff_lipschitz = 2

    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)
        return xp.linalg.norm(arr, axis=-1, keepdims=True) ** 2

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return 2 * arr

    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.DiffFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)


class L1Norm(pyco.ProxFunc):
    def __init__(self, shape=None):
        super(L1Norm, self).__init__(shape=(1, None))
        self._lipschitz = 1

    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)
        return xp.linalg.norm(arr, ord=1, axis=-1, keepdims=True)

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return (abs(arr) - tau).clip(0, None) * xp.sign(arr)

    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.ProxFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)


class FirstDerivative(pyco.LinOp):
    def __init__(self, size: int, axis: int = -1, sampling: float = 1.0, edge: bool = True, kind: str = "forward"):
        super(FirstDerivative, self).__init__((size, size))
        self.axis = axis
        self.sampling = sampling
        self.edge = edge
        self.kind = kind

        # choose apply and adjoint kind
        if kind == "forward":
            self.apply = self._apply_forward
            self.adjoint = self._adjoint_forward
        elif kind == "centered":
            self.apply = self._apply_centered
            self.adjoint = self._adjoint_centered
        elif kind == "backward":
            self.apply = self._apply_backward
            self.adjoint = self._adjoint_backward
        else:
            raise NotImplementedError("kind must be forward, centered, " "or backward")

    def _apply_forward(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.swapaxes(arr, self.axis, -1)
        y = xp.zeros_like(arr)
        y[..., :-1] = (arr[..., 1:] - arr[..., :-1]) / self.sampling
        return y

    def _adjoint_forward(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.swapaxes(arr, self.axis, -1)
        y = xp.zeros_like(arr)
        y[..., :-1] -= arr[..., :-1] / self.sampling
        y[..., 1:] += arr[..., :-1] / self.sampling
        return y

    def _apply_centered(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.swapaxes(arr, self.axis, -1)
        y = xp.zeros_like(arr)
        y[..., 1:-1] = 0.5 * (arr[..., 2:] - arr[..., :-2]) / self.sampling
        if self.edge:
            y[..., 0] = (arr[..., 1] - arr[..., 0]) / self.sampling
            y[..., 0] = (arr[..., -1] - arr[..., -2]) / self.sampling
        return y

    def _adjoint_centered(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.swapaxes(arr, self.axis, -1)
        y = xp.zeros_like(arr)
        y[..., :-2] -= 0.5 * arr[..., 1:-1] / self.sampling
        y[..., 2:] += 0.5 * arr[..., 1:-1] / self.sampling
        if self.edge:
            y[..., 0] -= arr[..., 0] / self.sampling
            y[..., 1] += arr[..., 0] / self.sampling
            y[..., -2] -= arr[..., -1] / self.sampling
            y[..., -1] += arr[..., -1] / self.sampling
        return y

    def _apply_backward(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.swapaxes(arr, self.axis, -1)
        y = xp.zeros_like(arr)
        y[..., 1:] = (arr[..., 1:] - arr[..., :-1]) / self.sampling
        return y

    def _adjoint_backward(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.swapaxes(arr, self.axis, -1)
        y = xp.zeros_like(arr)
        y[..., :-1] -= arr[..., 1:] / self.sampling
        y[..., 1:] += arr[..., 1:] / self.sampling
        return y


class Masking(pyco.LinOp):
    def __init__(self, size: int, sampling_bool: typ.Union[pyct.NDArray, list]):
        if isinstance(sampling_bool, list):
            import numpy as xp
        else:
            xp = pycu.get_array_module(sampling_bool)

        self.sampling_bool = xp.asarray(sampling_bool).reshape(-1).astype(bool)
        self.input_size = size
        self.nb_of_samples = self.sampling_bool[self.sampling_bool == True].size
        if self.sampling_bool.size != size:
            raise ValueError("Invalid size of boolean sampling array.")
        super(Masking, self).__init__(shape=(self.nb_of_samples, self.input_size))

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr[..., self.sampling_bool]

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(1, -1) if (arr.ndim == 1) else arr
        y = xp.zeros((*arr.shape[:-1], self.input_size), dtype=arr.dtype)
        y[..., self.sampling_bool] = arr
        return y


class DownSampling(Masking):
    def __init__(
        self,
        size: int,
        downsampling_factor: typ.Union[int, tuple, list],
        shape: typ.Optional[tuple] = None,
        axis: typ.Optional[int] = None,
    ):
        if type(downsampling_factor) is int:
            if (shape is not None) and (axis is None):
                self.downsampling_factor = len(shape) * (downsampling_factor,)
            else:
                self.downsampling_factor = (downsampling_factor,)
        else:
            self.downsampling_factor = tuple(downsampling_factor)
        if shape is not None:
            if size != np.prod(shape):
                raise ValueError(f"Array size {size} is incompatible with array shape {shape}.")
            if (axis is not None) and (axis > len(shape) - 1):
                raise ValueError(f"Array size {size} is incompatible with array shape {shape}.")
        if (shape is None) and (len(self.downsampling_factor) > 1):
            raise ValueError("Please specify an array shape for multidimensional downsampling.")
        elif (shape is not None) and (axis is None) and (len(shape) != len(self.downsampling_factor)):
            raise ValueError(f"Inconsistent downsampling factors {downsampling_factor} for array of shape {shape}.")
        self.input_size = size
        self.input_shape = shape
        self.axis = axis
        self.downsampling_mask = self.compute_downsampling_mask()
        if self.input_shape is None:
            self.output_shape = None
        else:
            if len(self.downsampling_factor) > 1:
                output_shape = []
                for ax in range(len(self.input_shape)):
                    axis_indices = np.arange(self.input_shape[ax])
                    downsampled_axis_indices = axis_indices % self.downsampling_factor[ax]
                    output_shape.append(downsampled_axis_indices[downsampled_axis_indices == 0].size)
                self.output_shape = tuple(output_shape)
            else:
                output_shape = list(self.input_shape)
                downsampled_axis_indices = np.arange(self.input_shape[self.axis])
                downsampled_axis_indices = downsampled_axis_indices % self.downsampling_factor
                output_shape[self.axis] = downsampled_axis_indices[downsampled_axis_indices == 0].size
                self.output_shape = tuple(output_shape)

        super(DownSampling, self).__init__(size=self.input_size, sampling_bool=self.downsampling_mask)

    def compute_downsampling_mask(self) -> np.ndarray:
        if self.input_shape is None:
            indices = np.arange(self.input_size)
            downsampled_mask = (indices % self.downsampling_factor) == 0
        else:
            if len(self.downsampling_factor) > 1:
                downsampled_mask = True
                for ax in range(len(self.input_shape)):
                    axis_indices = np.arange(self.input_shape[ax])
                    downsampled_axis_indices = axis_indices % self.downsampling_factor[ax]
                    downsampled_axis_indices = downsampled_axis_indices.reshape(
                        downsampled_axis_indices.shape + (len(self.input_shape) - 1) * (1,)
                    )
                    downsampled_axis_indices = np.swapaxes(downsampled_axis_indices, 0, ax)
                    downsampled_mask = downsampled_mask * (downsampled_axis_indices == 0)
            else:
                downsampled_mask = np.zeros(shape=self.input_shape, dtype=bool)
                downsampled_mask = np.swapaxes(downsampled_mask, 0, self.axis)
                downsampled_axis_indices = np.arange(self.input_shape[self.axis])
                downsampled_axis_indices = downsampled_axis_indices % self.downsampling_factor
                downsampled_mask[downsampled_axis_indices == 0, ...] = True
                downsampled_mask = np.swapaxes(downsampled_mask, 0, self.axis)
        return downsampled_mask.reshape(-1)


class Convolve(pyco.LinOp):
    conv_or_corr = {0: "correlate", 1: "convolve"}

    def __init__(self, data_shape, filter, mode="reflect", cval=0.0, origin=0):
        if filter.ndim != len(data_shape):
            raise ValueError(
                f"number of filter dimensions does not match number of data_shape dimensions: {filter.ndim} != {len(data_shape)}"
            )
        super().__init__(shape=(np.product(data_shape), np.product(data_shape)))
        self.filter = filter
        self.mode = mode
        self.cval = cval
        self.origin = origin
        self.data_shape = data_shape

    @pycrt.enforce_precision(i="arr")
    def _apply(self, arr, apply: bool):
        input_shape = arr.shape
        xp = pycu.get_array_module(arr)

        if self.filter.dtype != pycrt.getPrecision():
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)

        # arr = arr.reshape(*input_shape[:-1], *self.data_shape)

        if pycu.deps.CUPY_ENABLED and xp == cp:
            import cupyx.scipy.ndimage as sndx
        elif xp == da:
            import dask_image.ndfilters as sndx
        else:
            sndx = snd
        # flatten stacking dimension, reshape data dimension
        arr = arr.reshape(-1, *self.data_shape)
        data = []
        # iterate over stacking dimension
        for i in range(arr.shape[0]):
            data.append(
                getattr(sndx, self.conv_or_corr[int(apply)])(
                    arr[i], self.filter, mode=self.mode, cval=self.cval, origin=self.origin
                )
            )
        # TODO monkey fix because input shape is wrong
        # input_shape = (3, 2666000)
        return np.stack(data).reshape(input_shape)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._apply(arr, True)

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._apply(arr, False)


class GradientOp(pyco.LinOp):
    def __init__(self, shape, size, step=1.0, edge=True, dtype="float64", kind="centered"):
        import pylops

        super(GradientOp, self).__init__((2 * size, size))
        self.dtype = dtype
        self.data_shape = shape
        self.op = pylops.Gradient(dims=self.data_shape, sampling=step, edge=edge, dtype=dtype, kind=kind)
        self.op_scipy = spls.aslinearoperator(self.op)

    def apply(self, arr):
        arr = arr.reshape(-1, arr.shape[-1])
        data = []
        for i in range(arr.shape[0]):
            data.append(self.op * arr[i])
        return np.stack(data).reshape(*arr.shape[:-1], -1)

    # Input of (2*shape[0], shape[1]) get back to original (shape).
    # Note this is another derivative operation
    def adjoint(self, arr):
        # TODO monkey fix
        # arr = arr.reshape(3, 5332000)
        arr = arr.reshape(-1, arr.shape[-1])
        data = []
        for i in range(arr.shape[0]):
            data.append(self.op.H * arr[i])
        return np.stack(data).reshape(*arr.shape[:-1], -1)

    # convert to scipy in order to compute lipschitz constant
    def compute_lipschitz(self) -> float:
        return spls.svds(A=self.op_scipy, k=1, which="LM", return_singular_vectors=False, tol=1e-3).item()


class GradientOp(pyco.LinOp):
    def __init__(self, shape, size, step=1.0, edge=True, dtype="float64", kind="centered"):
        import pylops

        super(GradientOp, self).__init__((2 * size, size))
        self.dtype = dtype
        self.data_shape = shape
        self.op = pylops.Gradient(dims=self.data_shape, sampling=step, edge=edge, dtype=dtype, kind=kind)
        self.op_scipy = spls.aslinearoperator(self.op)

    def apply(self, arr):
        input_shape = arr.shape
        arr = arr.reshape(-1, arr.shape[-1])
        data = []
        for i in range(arr.shape[0]):
            data.append(self.op * arr[i])
        return np.stack(data).reshape(*input_shape[:-1], -1)

    # Input of (2*shape[0], shape[1]) get back to original (shape).
    # Note this is another derivative operation
    def adjoint(self, arr):
        input_shape = arr.shape
        # TODO monkey fix
        # arr = arr.reshape(3, 5332000)
        arr = arr.reshape(-1, arr.shape[-1])
        data = []
        for i in range(arr.shape[0]):
            data.append(self.op.H * arr[i])
        return np.stack(data).reshape(*input_shape[:-1], -1)

    # convert to scipy in order to compute lipschitz constant
    def compute_lipschitz(self) -> float:
        return spls.svds(A=self.op_scipy, k=1, which="LM", return_singular_vectors=False, tol=1e-3).item()
