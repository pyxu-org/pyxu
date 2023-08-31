
.. raw:: html

   <p align="center"> 
   <img align="center" src="doc/_static/logo.png" alt="Pyxu logo" width=25%>
   </p> 

**Pyxu** (pronounced [piksu], formerly known as Pycsou) is an open-source Python framework 
allowing scientists at any level to quickly prototype/deploy *hardware accelerated and out-of-core* computational imaging pipelines at scale.
Thanks to its hardware-agnostic **microservice architecture** and its tight integration with the PyData ecosystem, 
Pyxu supports a wide range of imaging applications, scales, and computation architectures.

.. raw:: html

   <p align="center"> 
   <img align="center" src="doc/_static/banner.jpg" alt="Banner" width=80%>
   </p> 

What Makes Pyxu Special?
------------------------

* **Universal & Modular 🌐:** Unlike other platforms that are specialized for particular imaging types, Pyxu is a general-purpose computational imaging tool. No more wrestling with one-size-fits-all solutions that don't quite fit!
* **Plug-and-Play Functionality 🎮:** Craft bespoke imaging pipelines effortlessly with advanced operator algebra logic. Pyxu automates the tedious bits, like computing gradients, proximal operators, and hyperparameters.
* **High-Performance Computing 🚀:** Whether you're working with CPUs or GPUs, Pyxu is optimized for both. It employs `Duck arrays <https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html>`_ and just-in-time compiling with `Numba <https://numba.pydata.org/>`_, supporting frameworks like `CuPy <https://cupy.dev/>`_ for GPU and `Dask <https://dask.org/>`_ for distributed computing.
* **Flexible & Adaptable 🛠️:** Combat the common woes of software rigidity with Pyxu's ultra-flexible framework. Bayesian techniques requiring extensive software flexibility are a breeze here.
* **Hardware Acceleration 🖥️:** Leverage built-in support for hardware acceleration to ramp up your computational speed, all thanks to our module-agnostic codebase.
* **Distributed Computing 🔗:** Got a lot of data? No worries! Pyxu is built for scaling, easily deployable on institutional clusters using industry-standard technologies like `Kubernetes <https://kubernetes.io/>`_ and `Docker <https://www.docker.com/>`_.
* **Deep Learning Interoperability 🤖:**  Integrate with major deep learning frameworks like `PyTorch <https://pytorch.org/>`_ and `JAX <https://jax.readthedocs.io/en/latest/jax.html>`_ for state-of-the-art computational imaging techniques.

Why is Pyxu Necessary?
----------------------

In the realm of computer vision 📷, digital image restoration and enhancement techniques have established themselves as indispensable pillars. 
These techniques, aiming to restore and elevate the quality of degraded or partially observed images, have witnessed unprecedented progress 📈 in recent times. 
Thanks to potent image priors, we've now reached an era where image restoration and enhancement methods are incredibly advanced ✨ —so much so that we might be approaching a pinnacle in terms of performance and accuracy.

However, it's not all roses 🌹.

Despite their leaps in progress, advanced image reconstruction methods often find themselves trapped in a vicious cycle of limited adaptability, usability, and reproducibility. 
Many advanced computational imaging solutions, while effective, are tailored for specific use-cases and seldom venture beyond the confines of a proof-of-concept 🚧. 
These niche solutions demand deep expertise to customize and deploy, making their adoption in production pipelines challenging.

In essence, the imaging domain is desperately seeking what the deep learning community found in frameworks like `PyTorch <https://pytorch.org/>`_, `TensorFlow <https://www.tensorflow.org/>`_, or `Keras <https://keras.io/>`_ —a flexible, modular, and powerful environment that accelerates the adoption of cutting-edge methods in real-world settings.
Pyxu stands as an answer to this call: a groundbreaking, open-source computational imaging software framework tailored for Python enthusiasts 🐍. 

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

Comparison with other Frameworks
--------------------------------

Pyxu offers a comprehensive suite of algorithms, including the latest primal-dual splitting methods for nonsmooth optimization. 
The feature set is robust and mature, positioning it as a leader in the computational imaging arena.

.. list-table:: Feature Maturity Comparison
    :header-rows: 1
    :stub-columns: 1
    :widths: auto

    * - Package Name 📦
      - Operator Types 🛠️
      - Operator Algebra 🎯
      - Algorithmic Suite 📚
      - Application Focus 🎯
      - Remarks 💬

    * - PyLops
      - 🔴 Linear ops
      - 🟡 Partial
      - 🔴 Least-squares & sparse rec.
      - 🟡 Wave-processing, geophysics
      - 🔴 Linear ops. based on old NumPy's matrix interface

    * - PyProximal
      - 🔴 Prox. funcs
      - 🔴 None
      - 🔴 Non-smooth cvx opt.
      - 🟢 None
      - 🔴 Under early development, unstable API

    * - Operator Discretization Library (ODL)
      - 🟡 Linear ops, diff./prox. funcs
      - 🟢 Full
      - 🟡 Smooth & non-smooth cvx opt.
      - 🟡 Tomography
      - 🔴 Domain-specific language for mathematicians

    * - GlobalBioIm
      - 🟢 (Non)linear ops, diff./prox. funcs
      - 🟢 Full
      - 🟢 Smooth, non-smooth & hybrid cvx opt.
      - 🟢 None
      - 🔴 MATLAB-based, unlike most DL fmwks

    * - SigPy
      - 🟡 Linear ops, prox. funcs
      - 🟡 Partial
      - 🟡 Smooth & non-smooth cvx opt.
      - 🔴 MRI
      - 🔴 Very limited suite of ops, funcs, algs

    * - SCICO
      - 🟢 (Non)linear ops, diff./prox. funcs
      - 🟢 Full
      - 🟢 Smooth, non-smooth & hybrid (non)cvx opt.
      - 🟢 None
      - 🟡 JAX-based (pure functions only, no mutation, etc)

    * - DeepInv
      - 🟢 (Non)linear ops, diff./prox. funcs
      - 🟡 Partial
      - 🟢 Smooth, non-smooth & hybrid (non)cvx opt.
      - 🟡 Deep Learning
      - 🟡 PyTorch-based (lots of dependencies)

    * - Pyxu
      - 🟢 (Non)linear ops, diff./prox. funcs
      - 🟢 Full
      - 🟢 Smooth, non-smooth, hybrid & stochastic (non)cvx opt.
      - 🟢 None
      - 🟢 Very rich suite of ops, funcs, algs & HPC features


Pyxu is unique in supporting both out-of-core and distributed computing. Additionally, it offers robust support for JIT compilation abd GPU computing 
via Numba and CuPy, respectively. Most contenders either offer partial support or lack these features altogether.

.. list-table:: HPC Features Comparison
    :header-rows: 1
    :stub-columns: 1
    :widths: auto

    * - Package Name 📦
      - Auto Diff/Prox ⚙️
      - GPU Computing 🖥️
      - Out-of-core Computing 🌐
      - JIT Compiling ⏱️

    * - PyLops
      - 🔴 No
      - 🟢 Yes (CuPy)
      - 🔴 No
      - 🟡 Partial (LLVM via Numba)

    * - PyProximal
      - 🔴 No
      - 🔴 No
      - 🔴 No
      - 🔴 No

    * - Operator Discretization Library (ODL)
      - 🟢 Yes
      - 🟡 Very limited (CUDA)
      - 🔴 No
      - 🔴 No

    * - GlobalBioIm
      - 🟢 Yes
      - 🟢 Yes (MATLAB)
      - 🔴 No
      - 🔴 No

    * - SigPy
      - 🔴 No
      - 🟢 Yes (CuPy)
      - 🟡 Manual (MPI)
      - 🔴 No

    * - SCICO
      - 🟢 Yes
      - 🟢 Yes (JAX) (GPU/TPU)
      - 🔴 No
      - 🟢 Yes (XLA via JAX)

    * - DeepInv
      - 🟢 Autodiff support
      - 🟢 Yes (PyTorch)
      - 🔴 No
      - 🟡 Partial(XLA via torch.compile)

    * - Pyxu
      - 🟢 Yes
      - 🟢 Yes (CuPy)
      - 🟢 Yes(Dask)
      - 🟢 Yes (LLVM and CUDA via Numba)


Get Started Now!
----------------

Ready to dive in? 🏊‍♀️ Our tutorial kicks off with an introductory overview of computational imaging and Bayesian reconstruction. 
It then provides an in-depth tour of Pyxu's multitude of features through concrete examples.

So, gear up to embark on a transformative journey in computational imaging. 

Join Our Community
------------------
Pyxu is open-source and ever-evolving 🚀. Your contributions, whether big or small, can make a significant impact. 
So come be a part of the community that's setting the pace for computational imaging 🌱.

Let's accelerate the transition from research prototypes to production-ready solutions. 
Dive into Pyxu today and make computational imaging more powerful, efficient, and accessible for everyone! 🎉
