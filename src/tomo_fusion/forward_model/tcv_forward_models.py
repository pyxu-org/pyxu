import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import pycsou.abc as pyca
import pycsou.operator.linop.base as pycb
import tomo_fusion.forward_model.chord_geometry as geom
import tomo_fusion.forward_model.LoS_handling as LoS
import tomo_fusion.forward_model.reaction as pyreaction


def plot_diagnostics():
    """
    Plot LoS (line of sight) configuration for bolo, axuv and sxr diagnostics
    """
    Lr = 0.5
    Lz = 1.5
    center = [Lr / 2, Lz / 2]
    radcam = geom.RADCAM_system()
    fig, ax = plt.subplots(1, 3, figsize=(8, 5), sharey=True)
    # bolo_plot = LoS.plot_LoS_ax(radcam.bolo_LoS_params, ax[0], Lr=Lr, Lz=Lz, center=center)
    LoS.plot_LoS_ax(radcam.axuv_LoS_params, ax[1], Lr=Lr, Lz=Lz, center=center)
    LoS.plot_LoS_ax(radcam.sxr_LoS_params, ax[2], Lr=Lr, Lz=Lz, center=center)
    ax[0].set_yticks([0, 1.5 / 2, 1.5])
    ax[0].set_yticklabels([r"$-Lz/2$", r"$0$", r"$Lz/2$"], fontsize=20)
    ax[0].set_xticks([0, 0.5 / 2, 0.5])
    ax[0].set_xticklabels([r"$-Lr/2$", r"$0$", r"$Lr/2$"], fontsize=20)
    ax[1].set_xticks([0, 0.5 / 2, 0.5])
    ax[1].set_xticklabels([r"$-Lr/2$", r"$0$", r"$Lr/2$"], fontsize=20)
    ax[2].set_xticks([0, 0.5 / 2, 0.5])
    ax[2].set_xticklabels([r"$-Lr/2$", r"$0$", r"$Lr/2$"], fontsize=20)
    matplotlib.rcParams["text.usetex"] = True
    plt.tight_layout()
    ax[0].set_title("Bolometers", fontsize=30)
    ax[1].set_title("AXUV", fontsize=30)
    ax[2].set_title("SXR", fontsize=30)


def SpcForwardModel(diagnostic="bolo", model="lines", **other_kwargs):
    """

    Parameters
    ----------
    diagnostic
        Diagnostic considered. Admissible values are "bolo", "axuv", "sxr", "artificial" (bolometers, extreme ultra violet, soft x-ray
         or an articial one with uniform (p,theta) parameter space coverage).
    model
        Type of model. Admissible values are "lines" (interpolation based line integration, matrix-free),
        "tubes" (integration over tubes of given width, matrix-free),
        "spc_lines" (integration over lines, matrix-based),
        "spc_beams" (integration over fan beam, matrix-based)
    """

    if model == "lines":
        # get relevant other_kwargs
        Lr = other_kwargs.get("Lr", 0.5)
        Lz = other_kwargs.get("Lz", 1.5)
        sampling_step = other_kwargs.get("sampling_step", 0.0125)
        integration_step_lines = other_kwargs.get("integration_step_lines", 1)
        # load geometry information
        radcam = geom.RADCAM_system()
        if diagnostic == "bolo":
            LoS_params = radcam.bolo_LoS_params
            bolo_startpoints, bolo_endpoints, _ = LoS.LoS_extrema_from_LoS_params(
                LoS_params, Lr=Lr, Lz=Lz, h=sampling_step
            )
            Op = pyreaction.RadonOpLines(
                bolo_startpoints,
                bolo_endpoints,
                np.array([round(Lz / sampling_step), round(Lr / sampling_step)]),
                h=sampling_step,
                integration_step=integration_step_lines,
            )
        # axuv
        elif diagnostic == "axuv":
            LoS_params = radcam.axuv_LoS_params
            axuv_startpoints, axuv_endpoints, _ = LoS.LoS_extrema_from_LoS_params(
                LoS_params, Lr=Lr, Lz=Lz, h=sampling_step
            )
            Op = pyreaction.RadonOpLines(
                axuv_startpoints,
                axuv_endpoints,
                np.array([round(Lz / sampling_step), round(Lr / sampling_step)]),
                h=sampling_step,
                integration_step=integration_step_lines,
            )
        # sxr
        elif diagnostic == "sxr":
            LoS_params = radcam.sxr_LoS_params
            sxr_startpoints, sxr_endpoints, _ = LoS.LoS_extrema_from_LoS_params(
                LoS_params, Lr=Lr, Lz=Lz, h=sampling_step
            )
            Op = pyreaction.RadonOpLines(
                sxr_startpoints,
                sxr_endpoints,
                np.array([round(Lz / sampling_step), round(Lr / sampling_step)]),
                h=sampling_step,
                integration_step=integration_step_lines,
            )
        # artificial
        elif diagnostic == "artificial":
            # get relevant other_kwargs
            nP = other_kwargs.get("nP", 13)
            nTheta = other_kwargs.get("nTheta", 13)
            pMax = other_kwargs.get("pMax", Lr / 2)
            LoS_params, startpoints, endpoints = LoS.generate_LoS_uniformly_spaced(
                nP=nP, nTheta=nTheta, Pmin=-pMax, Pmax=pMax, Lr=Lr, Lz=Lz, h=sampling_step
            )
            Op = pyreaction.RadonOpLines(
                startpoints,
                endpoints,
                np.array([round(Lz / sampling_step), round(Lr / sampling_step)]),
                h=sampling_step,
                integration_step=integration_step_lines,
            )
        else:
            raise ValueError(
                "Diagnostic `{}` not recognized. Admissible values are `bolo`, `axuv`, `sxr`.".format(diagnostic)
            )
        LoS.plot_LoS(LoS_params, Lr=0.5, Lz=1.5, center=[0.25, 0.75])

    elif model == "tubes":
        raise ValueError("Tubes not handled yet by this function")
    elif model == "spc_lines":
        if diagnostic == "bolo":
            Tmat = scio.loadmat("forward_model/matrices/Tmatline_py.mat")["Tmatline_py"]
            Op = pycb._ExplicitLinOp(cls=pyca.LinOp, mat=Tmat)
        else:
            raise ValueError("SPC lines model currently only handled for `bolo` diagnostic.")
    elif model == "spc_beams":
        if diagnostic == "bolo":
            Tmat = scio.loadmat("forward_model/matrices/Tmatvol_py.mat")["Tmatvol_py"]
            Op = pycb._ExplicitLinOp(cls=pyca.LinOp, mat=Tmat)
        else:
            raise ValueError("SPC beams model currently only handled for `bolo` diagnostic.")
    else:
        raise ValueError(
            "Model `{}` not recognized. Admissible values are `lines`, `tubes`, `spc_lines`, `spc_beams`.".format(model)
        )

    return Op
