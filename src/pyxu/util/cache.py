import hashlib

import pyxu.info.config as pxcfg

__all__ = [
    "cache_module",
]


def cache_module(code: str) -> str:
    """
    Save `code` as an importable module in Pyxu's dynamic module cache.

    The cached module is updated only if changes are detected.

    Parameters
    ----------
    code: str
        Contents of the module.
        When stored in a file, `code` should be a valid Python module.

    Returns
    -------
    module_name: str
        Name of the module in :py:func:`~pyxu.info.config.cache_dir`.

    Notes
    -----
    `module_name` is chosen automatically based on the file's contents.
    """
    # Compute a unique name
    h = hashlib.blake2b(code.encode("utf-8"), digest_size=8)
    module_name = "cached_" + h.hexdigest()

    pxcfg.cache_dir().mkdir(parents=True, exist_ok=True)
    module_path = pxcfg.cache_dir() / f"{module_name}.py"

    # Do we overwrite?
    write = True
    if module_path.exists():
        with open(module_path, mode="r") as f:
            old_content = f.read()
        if old_content == code:
            write = False

    if write:
        with open(module_path, mode="w") as f:
            f.write(code)

    return module_name
