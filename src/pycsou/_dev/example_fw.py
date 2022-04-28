import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

import pycsou.opt.stop as pycos
from pycsou._dev import L1Norm, SquaredL2Norm, SubSampling
from pycsou.operator.linop.base import ExplicitLinOp
from pycsou.opt.solver.fw_lasso import PolyatomicFWforLasso, VanillaFWforLasso
from pycsou.opt.solver.pgd import PGD

seed = None
L = 40
N = 100
k = 10
psnr = 60

lambda_factor = 0.1
remove = True
eps = 1e-4

track_ofv = pycos.AbsError(var="ofv", eps=1e-16)

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))
    indices = rng.choice(N, size=k)
    injection = SubSampling(size=N, sampling_indices=indices).T
    source = injection(rng.normal(size=k))

    op = ExplicitLinOp(mat)
    op.lipschitz()
    noiseless_measurements = op(source)
    std = np.max(np.abs(noiseless_measurements)) * 10 ** (-psnr / 20)
    noise = rng.normal(0, std, size=L)
    measurements = noiseless_measurements + noise

    lambda_ = lambda_factor * np.linalg.norm(op.T(measurements), np.infty)

    vfw = VanillaFWforLasso(measurements, op, lambda_, step_size="optimal", show_progress=False)
    pfw = PolyatomicFWforLasso(
        measurements, op, lambda_, ms_threshold=0.9, remove_positions=remove, show_progress=False
    )

    print("\nVanilla FW: Solving ...")
    start = time.time()
    vfw.fit(compute_ofv=True, stop_crit=vfw.default_stop_crit() | track_ofv)
    data_o, hist_o = vfw.stats()
    time_o = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_o))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    pfw.fit(compute_ofv=True, stop_crit=pfw.default_stop_crit() | track_ofv)
    data, hist = pfw.stats()
    time_r = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_r))

    data_fid = 0.5 * SquaredL2Norm().asloss(data=measurements) * op
    regul = lambda_ * L1Norm()

    print("Solving with APGD: ...")
    pgd = PGD(data_fid, regul, show_progress=False)
    start = time.time()
    pgd.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=pgd.default_stop_crit() | pycos.AbsError(var="x", eps=1e-16, f=data_fid + regul),
    )
    data_apgd, hist_apgd = pgd.stats()
    time_pgd = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_pgd))

    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}".format(data_o["dcv"], data["dcv"]))
    print(
        "Final value of objective:\n\tAPGD: {:.4f}\n\tVFW : {:.4f}\n\tPFW : {:.4f}".format(
            hist_apgd[-1][-1], data_o["ofv"], data["ofv"]
        )
    )

    plt.figure(figsize=(10, 12))
    plt.subplot(311)
    plt.stem(source, label="source", linefmt="C0:", markerfmt="C0o")
    plt.stem(data_apgd["x"], label="apgd", linefmt="C1-", markerfmt="C1d")
    plt.legend()
    plt.title("APGD")

    plt.subplot(312)
    plt.stem(source, label="source", linefmt="C0:", markerfmt="C0o")
    plt.stem(data_o["x"], label="VFW", linefmt="C2-", markerfmt="C2d")
    plt.legend()
    plt.title("Vanilla FW")

    plt.subplot(313)
    plt.stem(source, label="source", linefmt="C0:", markerfmt="C0o")
    plt.stem(data["x"], label="PFW", linefmt="C3-", markerfmt="C3d")
    plt.legend()
    plt.title("Polyatomic FW")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.stem(source, label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(data_apgd["x"], label="apgd", linefmt="C1:", markerfmt="C1x")
    plt.stem(data_o["x"], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(data["x"], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Compare the reconstructions")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.yscale("log")
    plt.plot(hist["AbsError[ofv]"], label="PFW")
    plt.plot(hist_o["AbsError[ofv]"], label="VFW")
    plt.title("Objective Function Value")
    plt.legend()
    plt.show()
