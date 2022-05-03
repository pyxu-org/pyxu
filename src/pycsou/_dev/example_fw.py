import time

import matplotlib.pyplot as plt
import numpy as np

import pycsou._dev.fw_utils as pycdevu
import pycsou.opt.solver.fw_lasso as pycfw
import pycsou.opt.stop as pycos
from pycsou.operator.linop.base import ExplicitLinOp
from pycsou.opt.solver.pgd import PGD

seed = None  # for reproducibility

# Dimensions of the problem
L = 40
N = 100
k = 10
psnr = 60

# Parameters for reconstruction
lambda_factor = 0.1
remove = True
eps = 1e-6
min_iterations = 100

# quick fix to keep track of the objective function value
track_ofv = pycos.AbsError(var="ofv", eps=1e-16)

# alternative stopping criteria
dcv = pycfw.dcvStoppingCrit(1e-4)
stop_crit = pycos.RelError(
    eps=eps,
    var="ofv",
    f=None,
    norm=2,
    satisfy_all=True,
)

# Minimum number of iterations
min_iter = pycos.MaxIter(n=min_iterations)

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))  # forward matrix
    indices = rng.choice(N, size=k)  # indices of active components in the source
    injection = pycdevu.SubSampling(size=N, sampling_indices=indices).T
    source = injection(rng.normal(size=k))  # sparse source

    op = ExplicitLinOp(mat)
    op.lipschitz()
    noiseless_measurements = op(source)
    std = np.max(np.abs(noiseless_measurements)) * 10 ** (-psnr / 20)
    noise = rng.normal(0, std, size=L)
    measurements = noiseless_measurements + noise

    lambda_ = lambda_factor * np.linalg.norm(op.T(measurements), np.infty)  # rule of thumb to define lambda

    vfw = pycfw.VanillaFWforLasso(measurements, op, lambda_, step_size="optimal", show_progress=False)
    pfw = pycfw.PolyatomicFWforLasso(
        measurements, op, lambda_, ms_threshold=0.9, remove_positions=remove, show_progress=False
    )

    print("\nVanilla FW: Solving ...")
    start = time.time()
    # vfw.fit(compute_ofv=True, stop_crit=min_iter & (vfw.default_stop_crit() | track_ofv))
    vfw.fit(compute_ofv=True, stop_crit=min_iter & (stop_crit | track_ofv))
    data_v, hist_v = vfw.stats()
    time_v = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_v))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    # pfw.fit(compute_ofv=True, stop_crit=min_iter & (pfw.default_stop_crit() | track_ofv))
    pfw.fit(compute_ofv=True, stop_crit=min_iter & (stop_crit | track_ofv))
    data_p, hist_p = pfw.stats()
    time_p = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_p))

    # Explicit definition of the objective function for APGD
    data_fid = 0.5 * pycdevu.SquaredL2Norm().asloss(data=measurements) * op
    regul = lambda_ * pycdevu.L1Norm()

    print("Solving with APGD: ...")
    pgd = PGD(data_fid, regul, show_progress=False)
    start = time.time()
    pgd.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=min_iter & (pgd.default_stop_crit() | pycos.AbsError(var="x", eps=1e-16, f=data_fid + regul)),
    )
    data_apgd, hist_apgd = pgd.stats()
    time_pgd = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_pgd))

    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}".format(data_v["dcv"], data_p["dcv"]))
    print(
        "Final value of objective:\n\tAPGD: {:.4f}\n\tVFW : {:.4f}\n\tPFW : {:.4f}".format(
            hist_apgd[-1][-1], data_v["ofv"], data_p["ofv"]
        )
    )

    # Solving the same problems with another stopping criterion: DCV
    vfw.fit(compute_ofv=True, stop_crit=min_iter & (dcv | track_ofv))
    data_v_dcv, hist_v_dcv = vfw.stats()

    pfw.fit(compute_ofv=True, stop_crit=min_iter & (dcv | track_ofv))
    data_p_dcv, hist_p_dcv = pfw.stats()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Compare the reconstructions")
    plt.subplot(211)
    plt.stem(source, label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(data_apgd["x"], label="apgd", linefmt="C1:", markerfmt="C1x")
    plt.stem(data_v["x"], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(data_p["x"], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Relative improvement as stopping criterion")
    plt.subplot(212)
    plt.stem(source, label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(data_apgd["x"], label="apgd", linefmt="C1:", markerfmt="C1x")
    plt.stem(data_v_dcv["x"], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(data_p_dcv["x"], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Dual certificate value as stopping criterion")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Objective function values")
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(hist_p["AbsError[ofv]"], label="PFW")
    plt.plot(hist_v["AbsError[ofv]"], label="VFW")
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Stop: Relative improvement ofv")

    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_dcv["AbsError[ofv]"], label="PFW")
    plt.plot(hist_v_dcv["AbsError[ofv]"], label="VFW")
    plt.ylabel("OFV")
    plt.legend()
    plt.title("Stop: Absolute error dcv")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Stopping criterion values")
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(hist_p["RelError[ofv]"], label="PFW")
    plt.plot(hist_v["RelError[ofv]"], label="VFW")
    plt.title("Stop: Relative improvement ofv")
    plt.legend()
    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_dcv["AbsError[dcv]"], label="PFW")
    plt.plot(hist_v_dcv["AbsError[dcv]"], label="VFW")
    plt.title("Stop: Absolute error dcv")
    plt.legend()
    plt.show()
