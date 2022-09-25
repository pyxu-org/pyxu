import datetime as dt
import functools

import matplotlib.pyplot as plt
import numpy as np

import pycsou._dev as dev
import pycsou._dev.stochastic.stoc_utils as devs
import pycsou.abc.solver as pycs
import pycsou.operator.func.norm as pyofn
import pycsou.opt.solver.pgd as pgd
import pycsou.opt.stochastic as pystoc
import pycsou.opt.stop as pycos

if __name__ == "__main__":

    img_shape = (800, 600)
    channels = (3,)
    img_to_pycsou_format = functools.partial(np.moveaxis, source=-1, destination=0)
    pycsou_to_img_format = functools.partial(np.moveaxis, source=0, destination=-1)

    # define blur kernal and Convolution operator
    oof_blur = devs.out_of_focus_kernel(r=6)
    overlap = oof_blur.shape[0] // 2
    Cop = dev.Convolve(arg_shape=img_shape, filter=oof_blur)
    Cop._lipschitz = np.array([0.99913936])
    # Gop.lipschitz() - Lipschitz has been precomputed

    # ==================================================================

    load = pystoc.NpzDataset("tour_de_france_color.npy")
    stacking_dim = load.shape[:-1]
    data_dim = load.shape[-1]

    ground_truth = np.load("tour_de_france_color_truth.npy")

    # ==================================================================

    mini_batch = (100, 100)
    Cop_batch = pystoc.ChunkOp(op=Cop, depth={0: overlap, 1: overlap}, boundary={0: "reflect", 1: "reflect"})
    c_dataset = pystoc.ChunkDataloader(load=load, arg_shape=img_shape, chunks=mini_batch)

    batch = pystoc.Batch(c_dataset, Cop_batch, shuffle=False)

    # ==================================================================

    grad_strategy = pystoc.SGD()
    stoc_func = pystoc.Stochastic(f=pyofn.SquaredL2Norm(), batch=batch, strategy=grad_strategy)

    # ==================================================================

    Gop = dev.GradientOp(img_shape, load.shape[-1], kind="forward")
    Gop._lipschitz = np.array([2.82841955])
    # Dop.lipschitz() - Lipschitz has been precomputed

    mu = 1 / (2 * np.prod(mini_batch))
    reg = mu * pyofn.SquaredL2Norm() * Gop

    F = stoc_func + reg

    # ==================================================================

    pgd = pgd.PGD(F)

    def loss(x):
        return np.sum((x - ground_truth) ** 2, keepdims=True)

    stop_crit = (
        pycos.MaxIter(n=200) | pycos.AbsError(eps=1e-10, f=loss) | pycos.MaxDuration(t=dt.timedelta(seconds=6000))
    )

    x0 = np.random.random(load.shape)

    pgd.fit(x0=x0, stop_crit=stop_crit, mode=pycs.Mode.MANUAL, acceleration=False, d=75)

    for i, data in enumerate(pgd.steps()):
        if i % 30 == 0:
            plt.imshow(pycsou_to_img_format(data["x"].reshape(*channels, *img_shape)))
            plt.title(f"Reconstruct Iteration: {i}")
            plt.show()

    xs = pgd.solution()

    plt.imshow(pycsou_to_img_format(xs.reshape(*channels, *img_shape)))
    plt.title("Solution")
    plt.show()
