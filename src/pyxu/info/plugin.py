import warnings
from importlib.metadata import entry_points

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
    duplicated = [ep.name for ep in eps if (ep in seen) or seen.add(ep.name)]
    if len(duplicated) > 0:
        warnings.warn(f"Found duplicated entry points: {duplicated}.", pxw.ContributionWarning)

    path = lambda name: f"{glob[name].__module__}.{glob[name].__name__}"

    # Load entry points
    for ep in eps:
        try:
            # plugin can overload -> load directly
            if ep.name.startswith("_"):
                if (core_name := ep.name[1:]) in glob:
                    msg = f"Plugin `{ep.value}` overloaded Pyxu base class/function `{path(core_name)}`."

                    glob[core_name] = ep.load()
                    if names is not None:
                        names.append(core_name)
                else:
                    msg = "\n".join(
                        [
                            f"Attempted to overload non-existing Pyxu base class/function `{ep.name}`.",
                            "Do not use the prefix `_`.",
                        ]
                    )

            # check if class/function already exists in Pyxu
            else:
                if ep.name in glob:
                    msg = "\n".join(
                        [
                            f"Attempting to overload Pyxu base class/function `{path(ep.name)}` with `{ep.value}`.",
                            "Overloading plugins must start with underscore `_`.",
                            f"Defaulting to base class/function `{path(ep.name)}`.",
                        ]
                    )
                else:
                    msg = f"Plugin `{ep.value}` loaded."

                    glob[ep.name] = ep.load()
                    if names is not None:
                        names.append(ep.name)
            warnings.warn(msg, pxw.ContributionWarning)
        except Exception as exc:
            warnings.warn(f"Failed to load plugin `{ep.name}`: {exc}.", pxw.ContributionWarning)

    return names
