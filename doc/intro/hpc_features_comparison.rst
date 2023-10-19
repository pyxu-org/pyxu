.. list-table:: HPC Features - Comparison
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
      - 🟢 Yes + TPU (JAX)
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
      - 🟢 Yes (Dask)
      - 🟢 Yes (LLVM and CUDA via Numba)