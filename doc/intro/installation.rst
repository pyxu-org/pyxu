.. _installation-instructions:

Installation
============

This guide will walk you through the three levels of installation to meet your specific needs:

1. **Basic Installation** - For users who just want to get started.
2. **Installation with Optional Dependencies** - For those looking for extended functionality.
3. **Developer Installation** - For contributors or users who need to dive deep into the codebase.

Pyxu is designed to be accessible and is compatible with Linux, macOS, and Windows, although it has not been extensively
tested on the latter.

Basic Installation
------------------

The core of **Pyxu** is lightweight and straightforward to install. You'll need Python (>= 3.9, < 3.12) and a few
mandatory dependencies. While these dependencies will be automatically installed via ``pip``, we highly recommend
installing NumPy and SciPy via ``conda`` to benefit from optimized math libraries.

First, to install NumPy and SciPy from conda-forge:

.. code-block:: bash

   conda install -c conda-forge numpy scipy

And then install Pyxu:

.. code-block:: bash

   pip install pyxu

That's it for the basic installation; you're ready to go!

Installation with Optional Dependencies
---------------------------------------

For extended features, you can install Pyxu with its optional dependencies:

.. code-block:: bash

   pip install pyxu[complete-cpu]  # full CPU-only user install.
   pip install pyxu[complete]      # full CPU/GPU  user install.

More fine-grained extras can be installed by looking at the ``project.optional-dependencies`` field in
``pyproject.toml`` at the root of the repository.

.. warning::

   The host system must have `CUDA 12.x <https://docs.nvidia.com/cuda/>`_ installed to use the GPU. If its CUDA version
   is different (or CUDA is not installed), then the `cudatoolkit` optional dependency will pull the required libraries
   from PyPI:

   .. code-block:: bash

      pip install pyxu[complete,cudatoolkit]

   Once installed, run the code below and add the printed lines to your shell's config file to complete the setup.
   (Typically `~/.bashrc` or `~/.zshrc`.)

   .. code-block:: python3

      # python3

      import pathlib as plib
      import sys

      site_path = filter(lambda _: _.endswith("/site-packages"), sys.path)
      site_path = next(site_path)  # there is only one

      nv_path = plib.Path(site_path) / "nvidia"
      lib_path = nv_path.glob("*/lib/")

      for f in sorted(lib_path):
          print(f'LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}":"{f}"')
      print("export LD_LIBRARY_PATH")


Developer Installation
----------------------

If you're interested in contributing to Pyxu or experimenting with its codebase, you **must** clone the repository and
install it *with developer dependencies*:

.. code-block:: bash

   git clone https://github.com/matthieumeo/pyxu.git
   cd pyxu
   pip install -e ".[dev,complete]"
   pre-commit install

Pyxu uses `tox` to automate common operations during the development process. The commands below summarize the main
operations:

.. code-block:: bash

   tox run -e py311-test         # run test suite. (fast subset: ~5[min])
   tox run -e py311-test -- all  # run test suite. (full suite; much longer)
   tox run -e pre-commit         # run pre-commit hooks.
   tox run -e doc                # build HTML docs. (incremental update)
   tox run -e doc -- clean       # build HTML docs. (from scratch)
   tox run -e dist               # build universal wheels for distribution.

All available tox environments can be viewed by running:

.. code-block:: bash

   tox list

Interoperation with Deep Learning Frameworks
--------------------------------------------

If you wish to use Pyxu in combination with deep learning frameworks like JAX and PyTorch, you'll need to install them
separately. For more information, consult the installation guides for `JAX
<https://github.com/google/jax#installation>`_ and `PyTorch <https://pytorch.org/get-started/locally/>`_.

You're All Set!
---------------

You are now ready to harness the capabilities of Pyxu for your projects and research. If you have any questions or
contributions, we would be happy to hear from you!
