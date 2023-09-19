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

   pip install pyxu[complete_no_gpu]  # full CPU-only user install
   pip install pyxu[complete_gpu]     # full CPU/GPU  user install. (CUDA 12.* required.)

More fine-grained Pyxu installs can be obtained by looking at the ``extras_require`` field in ``setup.cfg`` at the root
of the repository.

Developer Installation
----------------------

If you're interested in contributing to Pyxu or experimenting with its codebase, you *must* clone the repository and
install it manually *with developer dependencies*:

.. code-block:: bash

   git clone https://github.com/matthieumeo/pyxu.git
   cd pyxu
   pip install -e ".[dev,complete_gpu]"

To run tests, you can execute:

.. code-block:: bash

   tox run  # setup and run a short test suite.

For building documentation and running pre-commit hooks:

.. code-block:: bash

   tox run -e doc-fast  # build HTML docs
   tox run -e pre-commit  # run pre-commit hooks

All tox environments available can be viewed by running:

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
