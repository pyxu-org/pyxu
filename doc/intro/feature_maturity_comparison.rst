.. list-table:: Feature Maturity - Comparison
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
      - 🔴 Linear oeprators
      - 🟡 Partial
      - 🔴 Least-squares & sparse reconstructions
      - 🟡 Wave-processing, geophysics
      - 🔴 Linear operators based on NumPy's old matrix interface

    * - PyProximal
      - 🔴 Proximable functionals
      - 🔴 None
      - 🔴 Non-smooth convex optimization
      - 🟢 None
      - 🔴 Under early development, unstable API

    * - Operator Discretization Library (ODL)
      - 🟡 Linear operators, differentiable/proximable functionals
      - 🟢 Full
      - 🟡 Smooth & non-smooth convex optimization
      - 🟡 Tomography
      - 🔴 Domain-specific language for mathematicians

    * - GlobalBioIm
      - 🟢 (Non)linear operators, differentiable/proximable functionals
      - 🟢 Full
      - 🟢 Smooth, non-smooth & hybrid convex optimization
      - 🟢 None
      - 🔴 MATLAB-based, unlike most DL frameworks

    * - SigPy
      - 🟡 Linear operators, proximable functionals
      - 🟡 Partial
      - 🟡 Smooth & non-smooth convex optimization
      - 🔴 MRI
      - 🔴 Very limited suite of operators, functionals, and algorithms

    * - SCICO
      - 🟢 (Non)linear operators, differentiable/proximable functionals
      - 🟢 Full
      - 🟢 Smooth, non-smooth & hybrid (non-)convex optimization
      - 🟢 None
      - 🟡 JAX-based (pure functions only, no mutation, etc.)

    * - DeepInv
      - 🟢 (Non)linear operators, differentiable/proximable functionals
      - 🟡 Partial
      - 🟢 Smooth, non-smooth & hybrid (non-)convex optimization
      - 🟡 Deep Learning
      - 🟡 PyTorch-based (lots of dependencies)

    * - Pyxu
      - 🟢 (Non)linear operators, differentiable/proximable functionals
      - 🟢 Full
      - 🟢 Smooth, non-smooth & hybrid (non-)convex optimization
      - 🟢 None
      - 🟢 Very rich suite of operators, functionals, algorithms & HPC features