import datetime as dt
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pycsou._dev.fw_utils as pycdevu
import pycsou.abc as pyca
import pycsou.operator
import pycsou.operator as pycop
import pycsou.opt.solver.fw_lasso as pycfw
import pycsou.opt.stop as pycos
from pycsou.abc.operator import LinOp
from pycsou.opt.solver.pgd import PGD

matplotlib.use("Qt5Agg")

seed = 496  # for reproducibility

# Dimensions of the problem
L = 1000
N = 10000
k = 150
psnr = 20

# Parameters for reconstruction
lambda_factor = 0.1
remove = True
eps = 1e-4
min_iterations = 1
tmax = 15.0
ms_threshold = 0.9
init_correction = 1e-1
final_correction = 1e-6

# alternative stopping criteria
dcv = pycfw.dcvStoppingCrit(1e-3)
stop_crit = pycos.RelError(
    eps=eps,
    var="objective_func",
    f=None,
    norm=2,
    satisfy_all=True,
)

# Minimum number of iterations
min_iter = pycos.MaxIter(n=min_iterations)

# track DCV
track_dcv = pycos.AbsError(eps=1e-10, var="dcv", f=None, norm=2, satisfy_all=True)

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))  # forward matrix
    indices = rng.choice(N, size=k)  # indices of active components in the source
    injection = pycop.SubSample(N, indices).T  # pycdevu.SubSampling(size=N, sampling_indices=indices).T
    source = injection(rng.normal(size=k))  # sparse source

    op = pyca.LinOp.from_array(mat)
    op.lipschitz()
    noiseless_measurements = op(source)
    std = np.max(np.abs(noiseless_measurements)) * 10 ** (-psnr / 20)
    noise = rng.normal(0, std, size=L)
    measurements = noiseless_measurements + noise

    lambda_ = lambda_factor * np.linalg.norm(op.T(measurements), np.infty)  # rule of thumb to define lambda

    vfw = pycfw.VanillaFWforLasso(measurements, op, lambda_, step_size="optimal", show_progress=False)
    pfw = pycfw.PolyatomicFWforLasso(
        measurements,
        op,
        lambda_,
        ms_threshold=ms_threshold,
        remove_positions=remove,
        show_progress=False,
        init_correction_prec=init_correction,
        final_correction_prec=final_correction,
    )

    print("\nVanilla FW: Solving ...")
    start = time.time()
    # vfw.fit(stop_crit=min_iter & vfw.default_stop_crit())
    vfw.fit(stop_crit=(min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv)
    data_v, hist_v = vfw.stats()
    time_v = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_v))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    # pfw.fit(stop_crit=min_iter & pfw.default_stop_crit())
    pfw.fit(stop_crit=(min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv)
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

    plt.figure()
    plt.subplot(121)
    plt.plot(pfw._mstate["N_indices"], label="Support size")
    plt.plot(pfw._mstate["N_candidates"], label="candidates")
    plt.legend()
    plt.subplot(122)
    plt.plot(pfw._mstate["correction_iterations"], label="iterations")
    plt.twinx()
    plt.plot(pfw._mstate["correction_durations"], label="duration", c="r")
    plt.title("Correction iterations")
    plt.legend()
    plt.show()

    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}".format(data_v["dcv"], data_p["dcv"]))
    print(
        "Final value of objective:\n\tAPGD: {:.4f}\n\tVFW : {:.4f}\n\tPFW : {:.4f}".format(
            hist_apgd[-1][-1], hist_v[-1][-1], hist_p[-1][-1]
        )
    )

    # Solving the same problems with another stopping criterion: DCV
    vfw.fit(stop_crit=(min_iter & dcv) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)))
    data_v_dcv, hist_v_dcv = vfw.stats()

    pfw.fit(stop_crit=(min_iter & dcv) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)))
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
    plt.plot(hist_p["duration"], hist_p["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v["duration"], hist_v["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_apgd["duration"], hist_apgd["Memorize[objective_func]"], label="APGD")
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Stop: Relative improvement ofv")

    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_dcv["duration"], hist_p_dcv["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v_dcv["duration"], hist_v_dcv["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_apgd["duration"], hist_apgd["Memorize[objective_func]"], label="APGD")
    plt.ylabel("OFV")
    plt.legend()
    plt.title("Stop: Absolute error dcv")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Stopping criterion values")  # TO CHAAAAAAAAAAANGE + time x-axis + push
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(hist_p["duration"], hist_p["RelError[objective_func]"], label="PFW")
    plt.plot(hist_v["duration"], hist_v["RelError[objective_func]"], label="VFW")
    plt.title("Stop: Relative improvement ofv")
    plt.legend()
    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_dcv["duration"], hist_p_dcv["AbsError[dcv]"], label="PFW")
    plt.plot(hist_v_dcv["duration"], hist_v_dcv["AbsError[dcv]"], label="VFW")
    plt.title("Stop: Absolute error dcv")
    plt.legend()
    plt.show()

    # print("Solving with APGD and specified support: ...")
    # import pycsou.operator as pycop
    # ss = pycop.SubSample(data_fid.shape[1], np.nonzero(data_apgd['x'])[0])
    # rspgd = PGD(data_fid * ss.T, lambda_ * pycop.L1Norm(), show_progress=False)
    # start = time.time()
    # rspgd.fit(
    #     x0=np.zeros(ss.shape[0], dtype="float64"),
    #     stop_crit=(min_iter & pgd.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
    #     track_objective=True,
    # )
    # print("Done.")
    # data_rsapgd, hist_rsapgd = rspgd.stats()
    #
    # plt.figure(figsize=(10, 8))
    # plt.suptitle("Compare the reconstructions")
    # plt.stem(source, label="source", linefmt="C0-", markerfmt="C0o")
    # plt.stem(data_apgd["x"], label="apgd", linefmt="C1:", markerfmt="C1x")
    # plt.stem(ss.T(data_rsapgd["x"]), label="rs apgd", linefmt="C2:", markerfmt="C2x")
    # plt.legend()
    # plt.show()

# todo change the reweighting with specified number of iterations: 1/2/5/10 ? More iterations for the last one ?
