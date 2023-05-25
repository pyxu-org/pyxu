import numpy as np


def generate_phantoms(
    R: np.ndarray,
    Z: np.ndarray,
    phantom_type: str = "gaussian",
    Rp: float = 6.353,
    Zp: float = 0.6447,
    rho2: float = 1,
    aspect: float = 1.5,
):
    """
    Generate realistic phantoms for SXR tomography.
    :param R: grid R on which the phantom is sampled.
    :param Z: grid Z on which the phantom is sampled.
    :param phantom_type: Desired type of phantom. Valid types are ['gaussian','hollow','hfs','lfs'].
    :param Rp: coordinate r of phantom's center.
    :param Zp: coordinate z of phantom's center.
    :param rho2: spread of phantom in z direction.
    :param aspect: aspect parameter.
    :return: Phantom image taking the form of an np.ndarray.
    """
    rho1 = rho2 / aspect
    Gaussian_Phantom = np.exp(-((R - Rp) ** 2) / (2 * rho1**2) - (Z - Zp) ** 2 / (2 * rho2**2))
    Hollow_Phantom = Gaussian_Phantom - 1.0 * np.exp(-((R - Rp) ** 2) / (rho1**2) - (Z - Zp) ** 2 / (rho2**2))
    LFS_Phantom = Hollow_Phantom * np.exp(-((R - Rp - rho1) ** 2) / (6 * rho1**2))  # -(z-Zp)^2/(6*rho2^2))
    HFS_Phantom = Hollow_Phantom * np.exp(-((R - Rp + rho1) ** 2) / (6 * rho1**2))  # -(z-Zp)^2/(6*rho2^2))
    if phantom_type == "gaussian":
        return Gaussian_Phantom
    elif phantom_type == "hollow":
        return Hollow_Phantom
    elif phantom_type == "lfs":
        return LFS_Phantom
    elif phantom_type == "hfs":
        return HFS_Phantom
    else:
        print("Invalid phantom type.")
        return None
