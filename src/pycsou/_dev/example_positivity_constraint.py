import datetime as dt
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pycsou._dev.fw_utils as pycdevu
import pycsou.abc as pyca
import pycsou.opt.solver.fw_lasso as pycfw
import pycsou.opt.stop as pycos
from pycsou.abc.operator import LinOp
from pycsou.opt.solver.pds import CondatVu
from pycsou.opt.solver.pgd import PGD

matplotlib.use("Qt5Agg")

seed = None  # for reproducibility

# Dimensions of the problem
L = 30
N = 100
k = 10
psnr = 20

# Parameters for reconstruction
lambda_factor = 0.1
remove = True
eps = 1e-5
min_iterations = 100
tmax = 5.0

stop_crit = pycos.RelError(
    eps=eps,
    var="objective_func",
    f=None,
    norm=2,
    satisfy_all=True,
)

# Minimum number of iterations
min_iter = pycos.MaxIter(n=min_iterations)

full_stop = (min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax))

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))  # forward matrix
    indices = rng.choice(N, size=k)  # indices of active components in the source
    injection = pycdevu.SubSampling(size=N, sampling_indices=indices).T
    source = injection((rng.normal(size=k)))  # sparse source

    op = pyca.LinOp.from_array(mat)
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
    # vfw.fit(stop_crit=min_iter & vfw.default_stop_crit())
    vfw.fit(stop_crit=full_stop)
    data_v, hist_v = vfw.stats()
    time_v = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_v))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    # pfw.fit(stop_crit=min_iter & pfw.default_stop_crit())
    pfw.fit(stop_crit=full_stop)
    data_p, hist_p = pfw.stats()
    time_p = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_p))

    # Explicit definition of the objective function for APGD
    data_fid = 0.5 * pycdevu.SquaredL2Norm().argshift(-measurements) * op
    # it seems necessary to manually lunch the evaluation of the diff lipschitz constant
    data_fid.diff_lipschitz()
    regul = lambda_ * pycdevu.L1Norm()

    print("Solving with APGD: ...")
    pgd = PGD(data_fid, regul, show_progress=False)
    start = time.time()
    pgd.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=(min_iter & pgd.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
        track_objective=True,
    )
    data_apgd, hist_apgd = pgd.stats()
    time_pgd = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_pgd))

    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}".format(data_v["dcv"], data_p["dcv"]))
    print(
        "Final value of objective:\n\tAPGD: {:.4f}\n\tVFW : {:.4f}\n\tPFW : {:.4f}".format(
            hist_apgd[-1][-1], hist_v[-1][-1], hist_p[-1][-1]
        )
    )

    print("Positivity constraint:")
    print("\tVanilla FW: Solving ...")
    # Solving the same problems with positivity constraint
    vfw.fit(stop_crit=full_stop, positivity_constraint=True)
    data_v_pos, hist_v_pos = vfw.stats()
    print("\tSolved in {:.3f} seconds".format(hist_v_pos["duration"][-1]))

    print("\tPolyatomic FW: Solving ...")
    pfw.fit(stop_crit=full_stop, positivity_constraint=True)
    data_p_pos, hist_p_pos = pfw.stats()
    print("\tSolved in {:.3f} seconds".format(hist_p_pos["duration"][-1]))

    print("\tSolving with PGD: ...")
    posRegul = lambda_ * pycdevu.L1NormPositivityConstraint(shape=(1, None))
    pgd_pos = PGD(data_fid, posRegul, show_progress=False)
    pgd_pos.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=(min_iter & pgd_pos.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
        track_objective=True,
    )
    # cv = CondatVu(f=data_fid, g=regul, h=pycdevu.NonNegativeOrthant(shape=(1, N)), show_progress=True, verbosity=100)
    # cv.fit(
    #     x0=np.zeros(N, dtype="float64"),
    #     stop_crit=(min_iter & cv.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
    #     track_objective=True,
    # )
    # data_pds, hist_pds = cv.stats()
    data_pgd_pos, hist_pgd_pos = pgd_pos.stats()
    print("\tSolved in {:.3f} seconds".format(hist_pgd_pos["duration"][-1]))

    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}".format(data_v_pos["dcv"], data_p_pos["dcv"]))
    print(
        "Final value of objective:\n\tPDS: {:.4f}\n\tVFW : {:.4f}\n\tPFW : {:.4f}".format(
            hist_pgd_pos[-1][-1], hist_v_pos[-1][-1], hist_p_pos[-1][-1]
        )
    )
    ###########################################################################################

    # Bugs PDS:
    # *init IdentityOp(dim instead of shape)
    # *NullFunc -> NullOp
    # *Quadratic function to import from another file
    # * dimension of ourputof PDS
    # * does not return a positive valued variable...

    plt.figure(figsize=(10, 8))
    plt.suptitle("Compare the reconstructions")
    plt.subplot(211)
    plt.stem(source, label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(data_apgd["x"], label="APGD", linefmt="C1:", markerfmt="C1x")
    plt.stem(data_v["x"], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(data_p["x"], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Unconstrained reconstruction")
    plt.subplot(212)
    plt.stem(source, label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(data_pgd_pos["x"], label="PGD", linefmt="C1:", markerfmt="C1s")
    plt.stem(data_v_pos["x"], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(data_p_pos["x"], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Positivity constrained solution")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Objective function values")
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(hist_p["duration"], hist_p["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v["duration"], hist_v["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_apgd["duration"], hist_apgd["Memorize[objective_func]"], label="APGD")
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Unconstrained reconstruction")
    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_pos["duration"], hist_p_pos["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v_pos["duration"], hist_v_pos["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_pgd_pos["duration"], hist_pgd_pos["Memorize[objective_func]"], label="PGD")
    plt.ylabel("OFV")
    plt.legend()
    plt.title("Positivity constrained solution")
    plt.show()
