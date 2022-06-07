import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

import pycsou._dev as dev
import pycsou._dev.stochastic.stoc_utils as devs
import pycsou.abc.solver as pycs
import pycsou.opt.solver.pgd as pgd
import pycsou.opt.stochastic as pystoc
import pycsou.opt.stop as pycos

if __name__ == "__main__":

    img_shape = (800, 600)

    # define blur kernal and Convolution operator
    oof_blur = devs.out_of_focus_kernel(r=6)
    overlap = oof_blur.shape[0] // 2
    Cop = dev.Convolve(data_shape=img_shape, filter=oof_blur, mode="reflect")
    Cop._lipschitz = np.array([0.99913936])
    # Gop.lipschitz() - Lipschitz has been precomputed

    # ==================================================================

    load = pystoc.NpzLoad("tour_de_france.npy")
    stacking_dim = load.shape[:-1]
    data_dim = load.shape[-1]

    ground_truth = np.load("tour_de_france_truth.npy")

    # ==================================================================

    mini_batch = (100, 100)
    loader = pystoc.ConvolveLoader(
        load=load, blocks=mini_batch, data_shape=img_shape, operator=Cop, depth=(overlap, overlap), mode="reflect"
    )
    batch = pystoc.Batch(loader, shuffle=True)

    # ==================================================================

    grad_strategy = pystoc.SGD()
    stoc_func = pystoc.Stochastic(f=dev.SquaredL2Norm(), batch=batch, strategy=grad_strategy)

    # ==================================================================

    Gop = dev.GradientOp(img_shape, load.shape[-1], kind="forward")
    Gop._lipschitz = np.array([2.82841955])
    # Dop.lipschitz() - Lipschitz has been precomputed

    mu = 1 / (2 * np.prod(mini_batch))
    reg = mu * dev.SquaredL2Norm() * Gop

    F = stoc_func + reg

    # ==================================================================

    pgd = pgd.PGD(F)

    def loss(x):
        return np.sum((x - ground_truth) ** 2, keepdims=True)

    stop_crit = (
        pycos.MaxIter(n=1000) | pycos.AbsError(eps=1e-10, f=loss) | pycos.MaxDuration(t=dt.timedelta(seconds=6000))
    )

    x0 = np.random.random(load.shape)

    pgd.fit(x0=x0, stop_crit=stop_crit, mode=pycs.Mode.MANUAL, acceleration=False, d=75)

    for i, data in enumerate(pgd.steps()):
        if i % 30 == 0:
            plt.imshow(data["x"].reshape(img_shape), cmap="gray", vmin=0, vmax=1)
            plt.title(f"Reconstruct Iteration: {i}")
            plt.show()

    xs = pgd.solution()

    plt.imshow(xs.reshape(img_shape), cmap="gray", vmin=0, vmax=1)
    plt.title("Solution")
    plt.show()
