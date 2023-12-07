How to use plugins in Pyxu?
===========================

For Pyxu Users
--------------

* Jump onto the `catalogue website <./index.html>`_ and simply search for the plugins you need.
* Look out for plugins with a high `Pyxu score <./score.html>`_ for a seamless experience.
* Once selected, follow the given instructions to integrate them into your Pyxu environment. It shouldn't be harder than
  a simple ``pip install`` command!
* Once installed, the plugin will be auto-discovered by Pyxu and will be available for use in your Pyxu environment.

For instance:

- The gradient descent algorithm can be run with the `PGD <../api/opt.html#pyxu.opt.solver.pgd.PGD>`_ algorithm without a providing proximable functional.

- While the ``GradientDescent`` class is not included in the Pyxu core (refer to the `API documentation <../api/index.html>`_), a dedicated plugin is available. You can find it on the `catalogue website <./plugins/index.html>`_.

- To install the plugin, execute the following command in your terminal:

  .. code-block:: bash

     pip install pyxu-gradient-descent

- Once installed, you can import the ``GradientDescent`` class in your Python scripts as if it was part of Pyxu's `solver <../api/opt.html#pyxu-opt-solver>`_ module, using the following line of code:

  .. code-block:: python

     from pyxu.opt.solver import GradientDescent

- Without the installation of the plugin, the previous line of code would have raised an error.

  .. code-block:: python

     ModuleNotFoundError: No module named 'pyxu.opt.solver.GradientDescent'

How does this happens?
----------------------

Pyxu uses `Python's entry point mechanism <https://packaging.python.org/en/latest/specifications/entry-points/>`_ to auto-discover plugins.
The entry points are defined in the ``setup.py`` file of the plugin.
In Pyxu, each module imports entry points from all the installed plugins and registers them as if they were part of the
Pyxu core. This allows Pyxu to be extensible without the need to modify the core code.

For Developers
--------------

Kick-start your plugin development with the `Pyxu cookie-cutter <https://github.com/pyxu-org/cookiecutter-pyxu>`_ to
generate a plugin structure to be filled with your awesome code! You can contribute different types of entry points, such
as:

* `pyxu.operator <../api/operator/index.html>`_
* `pyxu.opt.solver <../api/opt.html#pyxu-opt-solver>`_
* `pyxu.opt.stop <../api/opt.html#pyxu-opt-stop>`_
* `pyxu.math <../api/math.html>`_
* pyxu.contrib: for any other contribution

To override a class or function in the core Pyxu framework, simply use the identical name as the target you wish to override.
This process involves a critical step: specifying your intention in the setup.cfg file by prefixing the new entry point
with an underscore (``_``).

For example, to override the ``NullFunc`` class housed within the ``pyxu.operator.func`` namespace, insert the following
directive in your setup.cfg file:

.. code-block:: ini

    [options.entry_points]
       pyxu.operator.func =
        _NullFunc = your_module_name:NullFunc


Once ready, publish your plugin on `PyPI <https://pypi.org/>`_. The Pyxu-FAIR periodically scans PyPI for new plugins and
`scores <./score.html>`_ them based on their popularity and quality. The higher the score, the higher the chances of your
plugin being discovered by Pyxu users.

If you believe your plugin enhances the Pyxu core codebase, we encourage you to reach out so we can evaluate the possibility
of incorporating it. Even if you're uncertain, don't hesitate to open an issue on the `Pyxu GitHub repository <https://github.com/pyxu-org/pyxu>`_
for feedback and guidance.

🌟 No matter your expertise level, the Pyxu-FAIR ensures a smooth experience. Dive in, and may your Pyxu journey be
FAIR and fantastic! 🌟
