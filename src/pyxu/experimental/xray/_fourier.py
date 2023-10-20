import pyxu.abc as pxa


class FourierXRT(pxa.LinOp):
    # See XRayTransform() for instantiation details.
    def __init__(
        self,
        arg_shape,
        origin,
        pitch,
        n_spec,
        t_spec,
    ):
        raise NotImplementedError
