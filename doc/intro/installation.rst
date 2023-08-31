.. _installation-instructions:

Installation
============

Welcome to the installation guide for Pyxu! If you're interested in solving inverse problems with state-of-the-art proximal algorithms, you're in the right place. This guide will walk you through the three levels of installation to meet your specific needs:

1. **Basic Installation** - For users who just want to get started.
2. **Installation with Optional Dependencies** - For those looking for extended functionality.
3. **Developer Installation** - For contributors or users who need to dive deep into the codebase.

Pyxu is designed to be accessible and is compatible with Linux, macOS, and Windows, although it has not been extensively tested on the latter. 

Basic Installation
------------------

The core of **Pyxu** is lightweight and straightforward to install. You'll need Python (>= 3.9, < 3.12) and a few mandatory dependencies. While these dependencies will be automatically installed via ``pip``, we highly recommend installing NumPy and SciPy via ``conda`` to benefit from Intel MKL bindings and speed optimizations.

First, to install NumPy and SciPy from the conda-forge channel:

.. code-block:: bash

    conda install -c conda-forge numpy scipy

And then install Pyxu:

.. code-block:: bash

    pip install pyxu

That's it for the basic installation; you're ready to go!

Installation with Optional Dependencies
---------------------------------------

For extended features, you can install optional dependencies:

- ``dev``: For development tools like Sphinx, pre-commit, etc.
- ``fft``: For faster Fourier transforms.

Additionally, you can use aggregate targets based on your platform's GPU capabilities:

- ``complete_no_gpu``: For platforms without GPU support.
- ``complete_gpu``: For platforms with GPU support (CUDA 12.0 and above).

To install with a specific optional dependency, you can use:

.. code-block:: bash

    pip install pyxu[dev]   # for development tools
    pip install pyxu[fft]   # for FFT support

.. note::
    
    You can combine different targets by separating them with commas. For example, to install all GPU dependencies alongside the development tools, use:

    .. code-block:: bash

        pip install pyxu[dev,complete_gpu]

Developer Installation
----------------------

If you're interested in contributing to Pyxu or experimenting with its codebase, you can clone the repository and install it manually.

.. code-block:: bash

    git clone https://github.com/matthieumeo/pyxu.git
    cd pyxu
    pip install -e ".[dev,complete_gpu]" # complete_no_gpu also available

To run tests, you can execute:

.. code-block:: bash

    tox run

.. warning::
    
    Running the tests may take a significant amount of time. Please be patient.

For building documentation and running pre-commit hooks:

.. code-block:: bash

    tox run -e doc # Builds documentation with Sphinx
    tox run -e pre-commit # Run pre-commit hooks

Interoperation with Deep Learning Frameworks
--------------------------------------------

If you wish to use Pyxu in combination with major deep learning frameworks like JAX and PyTorch, you'll need to install them separately. For more information, consult the installation guides for `JAX <https://github.com/google/jax#installation>`_ and `PyTorch <https://pytorch.org/get-started/locally/>`_.

You're All Set!
---------------

You are now ready to harness the capabilities of Pyxu for your projects and research. If you have any questions or contributions, we would be happy to hear from you!