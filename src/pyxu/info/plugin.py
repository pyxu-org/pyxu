import warnings

from importlib_metadata import entry_points

import pyxu.info.warning as pxw


def _load_entry_points(glob, group, names=None):
    r"""
    Load (if any) entry point Pyxu plugins.

    Pyxu accepts contributions in the form of Pyxu-plugins, which can be developed by third-party contributors and are
    not necessarily tested or verified by the team of Pyxu core developers. While we strive to ensure the safety and
    security of our software framework and its plugins, we cannot guarantee their functionality or safety.  Users should
    exercise caution when installing and using plugins and assume full responsibility for any damages or issues that may
    arise from their use. The developers of this software framework are not liable for any harm caused by the use of
    third-party plugins.

    NOTE
    ----
    There might be duplicated extensions when installing a plugin in editable mode (i.e. with pip install -e). This does
    not represent a problem in practice but a warning that an attempt at overloading a Pyxu base object might arise. See
    the issue https://github.com/pypa/setuptools/issues/3649 for further information.

    """
    eps = tuple(entry_points(group=group))

    # Check for duplicated entry points
    seen = set()
    duplicated = [ep.name for ep in eps if ep in seen or seen.add(ep.name)]
    if len(duplicated):
        warnings.warn(f"Found duplicated entry points: {duplicated}.", pxw.ContributionWarning)

    # Load entry points
    try:
        for i, ep in enumerate(eps):
            ep_load = ep.load()
            name = ep.name
            # If plugin can overload, load directly
            if name.startswith("_"):
                if name[1:] in glob:
                    warnings.warn(
                        f"Plugin `{name}` overloaded an existing Pyxu base class/function, use with " f"caution.",
                        pxw.ContributionWarning,
                    )
                    glob[name[1:]] = ep_load
                    if names is not None:
                        names.append(name[1:])

                else:
                    warnings.warn(
                        f"Attempted to overload a non existing Pyxu base class/function `{name}`."
                        f"Do not use the prefix `_`.",
                        pxw.ContributionWarning,
                    )

            # Else, check if class/function already exists in Pyxu
            else:
                if name in glob:
                    warnings.warn(
                        f"Attempting to overload Pyxu base class/function `{name}`.\n"
                        + "Overloading plugins must start with underscore `_`.\n"
                        + f"Defaulting to base class/function `{name}`.\n",
                        pxw.ContributionWarning,
                    )
                else:
                    glob[name] = ep_load
                    if names is not None:
                        names.append(name)
                    warnings.warn(f"Plugin `{name}` loaded, use with caution.", pxw.ContributionWarning)
    except Exception as exc:
        warnings.warn(f"Failed to load plugin `{name}`: {exc}.", pxw.ContributionWarning)

    return names
