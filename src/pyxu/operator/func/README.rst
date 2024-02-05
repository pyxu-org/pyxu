FFT Notes
=========

FFT operators operate on real and complex numbers.
FFT-based operators hence have a more expansive interface compared to standard operators:

* Arithmetic methods all take (..., 2) inputs and return (..., 2) outputs.
  These correspond to real-valued views of complex-valued inputs.
  This means that FFT.[co]dim_shape = (..., 2): we do NOT support hybrid (real-in -> complex-out) dim/codim pairs. Users
  wanting this can achieve so by subsampling the output themselves.

* Convenience methods such as [capply,adjoint,...]() are sometimes provided to ease use of FFT operators in other
  contexts. [Ex: perform an FFT as intermediate step in a chain of operations, without having to re-interpret bytes as
  [apply,adjoint]() must do to abide by the Pyxu Operator API.]
