import pyxu.info.config as config

__all__ = [
    "cache_module",
]


def cache_module(name: str, code: str):
    """
    Update a specific module in the dynamic module cache.

    The cached module is updated only if changes are detected.

    Parameters
    ----------
    name: str
        Name of the module to update.
    code: str
        Contents of the module.
        When stored in a file, `code` should be a valid Python module.
    """
    module_path = config.cache_dir / f"{name}.py"

    write = True
    if module_path.exists():
        with open(module_path, mode="r") as f:
            old_content = f.read()
        if old_content == code:
            write = False

    if write:
        with open(module_path, mode="w") as f:
            f.write(code)
