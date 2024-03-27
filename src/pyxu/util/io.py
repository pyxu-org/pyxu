import os

import dask.array as da
import zarr

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util.array_module as pxam

__all__ = [
    "save_zarr",
    "load_zarr",
]


def save_zarr(filedir: pxt.Path, kw_in: dict[str, pxt.NDArray]) -> None:
    """
    Saves an array to a Zarr file. If the array is a Dask array, it is saved with a
    filename prefix "dask_". Otherwise, it is saved directly using Zarr's save function.

    Parameters
    ----------
    filedir : Path
        The directory path where the file will be saved.
    kw_in : dict[str, NDArray]
        A dictionary where keys are the filenames and values are the arrays to be saved.
    """
    for filename, array in kw_in.items():
        try:
            if array is not None:
                ndi = pxd.NDArrayInfo.from_obj(array)
                if ndi == pxd.NDArrayInfo.DASK:
                    array.to_zarr(
                        filedir / ("dask_" + filename),
                        overwrite=True,
                        compute=True,
                    )
                else:
                    zarr.save(filedir / filename, pxam.to_NUMPY(array))
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
