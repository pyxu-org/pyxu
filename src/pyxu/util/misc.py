import collections.abc as cabc
import importlib
import inspect
import os
import types
import typing as typ

import dask.array as da
import zarr

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt

__all__ = [
    "copy_if_unsafe",
    "import_module",
    "parse_params",
    "read_only",
    "save_zarr",
    "load_zarr",
]


def peaks(x: pxt.NDArray, y: pxt.NDArray) -> pxt.NDArray:
    r"""
    Matlab 2D peaks function.

    Peaks is a function of two variables, obtained by translating and scaling Gaussian distributions.  (See `Matlab's
    peaks function <https://www.mathworks.com/help/matlab/ref/peaks.html>`_.)

    This function is useful for testing purposes.

    Parameters
    ----------
    x: NDArray
        X coordinates.
    y: NDArray
        Y coordinates.

    Returns
    -------
    z: NDArray
        Values of the 2D function ``peaks`` at the points specified by the entries of `x` and `y`.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pyxu.util.misc import peaks

       x = np.linspace(-3, 3, 1000)
       xx, yy = np.meshgrid(x, x)
       z = peaks(xx, yy)
       plt.figure()
       plt.imshow(z)
    """
    ndi = pxd.NDArrayInfo.from_obj(x)
    xp = ndi.module()

    a = 3 * ((1 - x) ** 2) * xp.exp(-(x**2) - (y + 1) ** 2)
    b = -10 * ((x / 5) - x**3 - y**5) * xp.exp(-(x**2) - (y**2))
    c = -xp.exp(-((x + 1) ** 2) - (y**2)) / 3
    z = a + b + c
    return z


def star_like_sample(
    N: pxt.Integer,
    w: pxt.Integer,
    s: pxt.Real,
    po: pxt.Integer,
    x0: pxt.Real,
    ndi: pxd.NDArrayInfo = pxd.NDArrayInfo.NUMPY,
) -> pxt.NDArray:
    r"""
    Star-like test image.

    Generates a (N, N) square image of a star-like object normalized between 0 and 1.  Based on `GlobalBioIm's
    StarLikeSample function
    <https://github.com/Biomedical-Imaging-Group/GlobalBioIm/blob/master/Util/StarLikeSample.m>`_.  This function is
    useful for testing purposes as it contains high-frequency information.

    Parameters
    ----------
    N: Integer
        Size of the image (must be an even number).
    w: Integer
        The number of branches of the sample will be 4*w.
    s: Real
        Slope of the sigmoid function :math:`\frac1{1+\exp[s (x-x_{0})]}` attenuating the boundaries.
    po: Integer
        Power-raising factor for the final image (to have smoother edges).
    x0: Real
        Radial shift of the sigmoid function :math:`\frac1{1+\exp[s (x-x_{0})]}`.
    ndi: NDArrayInfo
        Desired array module for the output.

    Returns
    -------
    image: NDArray
        (N, N) image of star-like sample.

    Examples
    --------
    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pyxu.util.misc import star_like_sample

       star = star_like_sample(N=256, w=8, s=20, po=3, x0=0.7)
       plt.figure()
       plt.imshow(star)
    """
    assert N % 2 == 0
    xp = ndi.module()

    grid = xp.linspace(-1, 1, N)
    x, y = xp.meshgrid(grid, grid)
    theta = xp.arctan2(y, x)
    image = 1 + xp.cos(4 * w * theta)
    image /= 1 + xp.exp(s * (xp.sqrt(x**2 + y**2) - x0))
    image = xp.maximum(image, 0) / 2
    image **= po
    image /= xp.amax(image)
    return image


def parse_params(func, *args, **kwargs) -> cabc.Mapping:
    """
    Get function parameterization.

    Returns
    -------
    params: ~collections.abc.Mapping
        (key, value) params as seen in body of `func` when called via `func(*args, **kwargs)`.
    """
    sig = inspect.Signature.from_callable(func)
    f_args = sig.bind(*args, **kwargs)
    f_args.apply_defaults()

    params = dict(
        zip(f_args.arguments.keys(), f_args.args),  # positional arguments
        **f_args.kwargs,
    )
    return params


def import_module(name: str, fail_on_error: bool = True) -> types.ModuleType:
    """
    Load a Python module dynamically.
    """
    try:
        pkg = importlib.import_module(name)
    except ModuleNotFoundError:
        if fail_on_error:
            raise
        else:
            pkg = None
    return pkg


def copy_if_unsafe(x: pxt.NDArray) -> pxt.NDArray:
    """
    Copy array if it is unsafe to do in-place updates on it.

    In-place updates are unsafe if:

    * the array is read-only, OR
    * the array is a view onto another array.

    Parameters
    ----------
    x: NDArray

    Returns
    -------
    y: NDArray
    """
    N = pxd.NDArrayInfo
    ndi = N.from_obj(x)
    if ndi == N.DASK:
        # Dask operations span a graph -> always safe.
        y = x
    elif ndi == N.NUMPY:
        read_only = not x.flags.writeable
        reference = not x.flags.owndata
        y = x.copy() if (read_only or reference) else x
    elif ndi == N.CUPY:
        read_only = False  # https://github.com/cupy/cupy/issues/2616
        reference = not x.flags.owndata
        y = x.copy() if (read_only or reference) else x
    else:
        msg = f"copy_if_unsafe() not yet defined for {ndi}."
        raise NotImplementedError(msg)
    return y


def read_only(x: pxt.NDArray) -> pxt.NDArray:
    """
    Make an array read-only.

    Parameters
    ----------
    x: NDArray

    Returns
    -------
    y: NDArray
    """
    N = pxd.NDArrayInfo
    ndi = N.from_obj(x)
    if ndi == N.DASK:
        # Dask operations are read-only since graph-backed.
        y = x
    elif ndi == N.NUMPY:
        y = x.view()
        y.flags.writeable = False
    elif ndi == N.CUPY:
        y = x.view()
        # y.flags.writeable = False  # https://github.com/cupy/cupy/issues/2616
    else:
        msg = f"read_only() not yet defined for {ndi}."
        raise NotImplementedError(msg)
    return y


def save_zarr(filedir: pxt.Path, kw_in: dict[str, pxt.NDArray]) -> None:
    """
    Saves an array to a Zarr file. If the array is a Dask array, it is saved with a
    filename prefix "dask_". Otherwise, it is saved directly using Zarr's save function.

    Parameters
    ----------
    filedir : Path
        The directory path where the file will be saved.
    kw_in : Dict[str, NDArray]
        A dictionary where keys are the filenames and values are the arrays to be saved.

    Returns
    -------
    None
    """

    for filename, array in kw_in.items():
        try:
            if array is not None:
                if pxd.NDArrayInfo.from_obj(array) == pxd.NDArrayInfo.DASK:
                    array.to_zarr(
                    	filedir / ("dask_" + filename),
                    	overwrite=True,
                    	compute=True,
                    )
                else:
                    zarr.save(filedir / filename, array)
        except Exception as e:
            print(f"Failed to save {filename}: {e}")


def load_zarr(filepath: pxt.Path) -> dict[str, pxt.NDArray]:
    """
    Loads arrays from Zarr files within a specified directory. If a file is prefixed with "dask_",
    it is loaded as a Dask array. Otherwise, it is loaded using Zarr's load function.

    Parameters
    ----------
    filepath : pathlib.Path
        The directory path from where the Zarr files will be loaded.

    Returns
    -------
    kw_out : dict[str, NDArray]
        A dictionary where keys are the filenames (with "dask_" prefix removed if present)
        and values are the loaded arrays.
    """

    if not filepath.exists():
        raise FileNotFoundError(f"The directory {filepath} does not exist.")

    if not filepath.is_dir():
        raise NotADirectoryError(f"{filepath} is not a directory.")

    kw_out = {}
    try:
        files = os.listdir(filepath)
    except Exception as e:
        raise Exception(f"Failed to list directory contents: {e}")

    for file in files:
        try:
            if file.startswith("dask_"):
                array = da.from_zarr(filepath / file)
                kw_out[file.replace("dask_", "")] = array
            else:
                array = zarr.load(filepath / file)
                kw_out[file] = array
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    return kw_out
