.. image:: https://matthieumeo.github.io/pycsou/html/_images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://matthieumeo.github.io/pycsou/html/index


.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581

*Pycsou* is a Python 3 package for solving linear inverse problems with state-of-the-art proximal
algorithms. The software implements the main building blocks -cost functionals, penalty terms and
linear operators- of generic penalised convex optimisation problems.

Pycsou's documentation is available at https://matthieumeo.github.io/pycsou/html/index

This Python library is inspired by the MATLAB `GlobalBioIm
<https://github.com/Biomedical-Imaging-Group/GlobalBioIm>`_ project.

Features
========

Pycsou makes it easy to construct and solve penalised optimisation problems:

1. It offers a rich collection of operators and functionals commonly used in practice.
2. It implements *operator arithmetic*: *add/scale/compose/stack* operators together to quickly
   design *complex optimisation problems*.
3. It implements a rich collection of **state-of-the-art iterative proximal algorithms**, including
   **efficient primal-dual splitting methods** which involve only gradient, proximal and simple
   linear evaluations.
4. It supports *matrix-free linear operators*, making it easy to work with large scale linear
   operators that *may not fit in memory*.


Developper Install
------------------

.. code-block:: bash

   $ my_env=<CONDA ENVIRONMENT NAME>
   $ my_branch=<PYCSOU BRANCH NAME>
   $ git clone https://github.com/matthieumeo/pycsou && cd pycsou/
   $ git checkout "${my_branch}"
   $ conda create --name "${my_env}"            \
                  --strict-channel-priority     \
                  --channel=conda-forge         \
                  --file=conda/requirements.txt
   $ conda activate "${my_env}"
   $ python3 -m pip install -e ".[dev,complete_gpu]"  # 'complete_no_gpu' also available
   $ pre-commit install
   $ tox  # to run tests


To enable JAX on GPU, run this afterwards:

.. code-block:: bash

   $ conda install "jaxlib=*=*cuda*" jax cuda-nvcc -c conda-forge -c nvidia


Cite
----
For citing this package, please see: http://doi.org/10.5281/zenodo.4486431
