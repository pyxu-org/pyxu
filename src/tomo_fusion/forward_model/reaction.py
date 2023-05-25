import numba
import numpy as np
import numpy_indexed as npi

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct


# then ReactionOp as RadonOp.gram!!
class RadonOpTubes(pyca.LinOp):
    def __init__(
        self,
        p: pyct.NDArray,
        theta: pyct.NDArray,
        R: np.ndarray,
        Z: np.ndarray,
        r_center: float,
        z_center: float,
        pixel_diag: float,
        tube_width: float = 1.0,
    ):
        """

        Parameters
        ----------
        p
        theta
        R
        Z
        r_center
        z_center
        pixel_diag
        tube_width
        """
        self.nbLoS = p.size
        self.p = p
        self.theta = theta
        self.R, self.Z = R, Z
        self.r_center = r_center
        self.z_center = z_center
        self.tube_width = tube_width
        self.epsilon = self.tube_width * pixel_diag
        super().__init__(shape=(self.nbLoS, self.R.size))

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        image = arr.reshape(self.R.shape)
        sinogram = np.zeros(shape=(self.nbLoS,))
        for i in range(self.nbLoS):
            mask = (
                np.abs(
                    self.p[i]
                    - (
                        np.cos(self.theta[i]) * (self.R - self.r_center)
                        + np.sin(self.theta[i]) * (self.Z - self.z_center)
                    )
                )
                <= self.epsilon
            )
            sinogram[i] = image[mask].sum()
        return sinogram

    def adjoint(self, sinogram: pyct.NDArray) -> pyct.NDArray:
        sinogram = sinogram.flatten()
        image = np.zeros(shape=self.R.shape)
        for i in range(self.nbLoS):
            mask = (
                np.abs(
                    self.p[i]
                    - (
                        np.cos(self.theta[i]) * (self.R - self.r_center)
                        + np.sin(self.theta[i]) * (self.Z - self.z_center)
                    )
                )
                <= self.epsilon
            )
            image[mask] += sinogram[i]
        return image


def TestRadonOpTubes(radon_op):
    print("Testing RadonOpTubes\n")
    tol = 1e-10
    # test adjoint implementation
    np.random.seed(0)
    test_vec = np.random.rand(radon_op.nbLoS)
    test_image = np.random.rand(radon_op.R.shape[0], radon_op.R.shape[1])
    test_sinogram = radon_op.apply(test_image)
    test_backproject = radon_op.adjoint(test_vec)
    scalar_product_forward = np.dot(test_sinogram, test_vec)
    scalar_product_adjoint = np.dot(test_image.reshape(-1), test_backproject.reshape(-1))
    assert np.abs(scalar_product_forward - scalar_product_adjoint) < tol, "Adjoint does not satisfy (Rx)^Ty=x^T(R^*y)"
    print("Adjoint satisfies (Rx)^Ty=x^T(R^*y) on random x, y")


class RadonOpLines(pyca.LinOp):
    def __init__(
        self,
        startpoints: np.ndarray,
        endpoints: np.ndarray,
        arg_shape: np.ndarray,
        integration_step: float,
        h: float,
        Lz: float = 1.5,
    ):
        super().__init__(shape=(startpoints.shape[0], round(arg_shape[0] * arg_shape[1])))
        # following if condition was introduced to handle case of actual LoS from radcam but is wrong since chosen
        # condition is wrong. think and fix.
        # if startpoints[0, 0] >= 0 and endpoints[0, 0] >= 0:
        # convert in image coordinates if not done yet
        # startpoints, endpoints = los.cartesian_to_image_coordinates(startpoints, endpoints, Lz, h)
        self.nbLoS = startpoints.shape[0]
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.nrows = arg_shape[0]
        self.ncols = arg_shape[1]
        nbEval = np.zeros(self.nbLoS, dtype=int)
        hstep = np.zeros((self.nbLoS, 2))
        for i in range(self.nbLoS):
            # no need to rescale to compute nbEval, points coordinates already rescaled
            nbEval[i] = np.ceil(np.linalg.norm(startpoints[i] - endpoints[i]) / integration_step)
            locs = np.linspace(startpoints[i], endpoints[i], nbEval[i], endpoint=False)
            hstep[i] = locs[1, :] - locs[0, :]
        self.nbEval = nbEval
        self.hstep = hstep
        self.integration_step = integration_step
        self.h = h

    def _get_locs_and_weights(self, n: int):
        eval_locs = (
            np.linspace(self.startpoints[n], self.endpoints[n], self.nbEval[n], endpoint=False) + 0.5 * self.hstep[n]
        )
        minz = np.floor(eval_locs[:, 0]).astype("int")
        maxz = np.ceil(eval_locs[:, 0]).astype("int")
        minr = np.floor(eval_locs[:, 1]).astype("int")
        maxr = np.ceil(eval_locs[:, 1]).astype("int")
        dz = eval_locs[:, 0] - minz
        dr = eval_locs[:, 1] - minr
        top_left = np.vstack((minz, minr))
        top_right = np.vstack((minz, maxr))
        bottom_left = np.vstack((maxz, minr))
        bottom_right = np.vstack((maxz, maxr))
        pixel_locs = np.hstack((top_left, top_right, bottom_left, bottom_right))
        pixel_locs[0, :] = np.clip(pixel_locs[0, :], 0, self.nrows - 1)
        pixel_locs[1, :] = np.clip(pixel_locs[1, :], 0, self.ncols - 1)
        # necessary to avoid picking pixels that are out of bounds
        coeff_vals = np.hstack(((1 - dz) * (1 - dr), (1 - dz) * dr, dz * (1 - dr), dz * dr))
        pixel_locs_flat = self.ncols * pixel_locs[0, :] + pixel_locs[1, :]
        # new code: terribly slow
        # sorting_indices = pixel_locs_flat.argsort()
        # sorted_pixel_locs = pixel_locs_flat[sorting_indices]
        # sorted_coeff_vals = coeff_vals[sorting_indices]
        # unique_sorting = np.unique(sorted_pixel_locs, return_index=True)
        # pixel_locs_unique = unique_sorting[0]
        # grouped_coeff_vals = np.split(sorted_coeff_vals, unique_sorting[1][1:])
        # coeff_vals_unique = np.array([np.sum(group) for group in grouped_coeff_vals])
        # old code
        pixel_locs_unique, coeff_vals_unique = npi.group_by(pixel_locs_flat).sum(coeff_vals)
        return pixel_locs_unique, coeff_vals_unique * np.linalg.norm(self.hstep[n]) * self.h
        # sort_indices = np.argsort(pixel_locs_flat)
        # pixel_locs_flat = pixel_locs_flat[sort_indices]
        # coeff_vals = coeff_vals[sort_indices]
        # pixel_locs_unique = np.unique(pixel_locs_flat, return_index=True)
        # coeffs_grouped = np.split(coeff_vals, pixel_locs_unique[1][1:])
        # coeffs_reduced = [sum(group_i) for group_i in zip(*coeffs_grouped)]
        # return pixel_locs_unique[0], np.array(coeffs_reduced) * np.linalg.norm(self.hstep[n]) * self.h
        # pixel_locs_unique, coeff_vals_unique = npi.group_by(pixel_locs.T).sum(coeff_vals)
        # pixel_locs_flat = self.ncols * pixel_locs_unique[:, 0] + pixel_locs_unique[:, 1]
        # return pixel_locs_flat, coeff_vals_unique*np.linalg.norm(self.hstep[n])*self.h

    @pycu.vectorize("arr")
    # @numba.jit(parallel=True, fastmath=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # could be easily vectorized in a different way.
        arr = arr.flatten()
        sinogram = np.zeros(self.nbLoS)
        for i in numba.prange(self.nbLoS):
            pixel_locs, coeff_vals = self._get_locs_and_weights(i)
            sinogram[i] = np.sum(arr[pixel_locs] * coeff_vals)
        return sinogram.reshape(1, -1)

    @pycu.vectorize("sinogram")
    # @numba.jit(parallel=True, fastmath=True)
    def adjoint(self, sinogram: pyct.NDArray) -> pyct.NDArray:
        # easily vectorizable in a different way.
        sinogram = sinogram.flatten()
        image = np.zeros(round(self.nrows * self.ncols))
        for i in numba.prange(self.nbLoS):
            pixel_locs, coeff_vals = self._get_locs_and_weights(i)
            image[pixel_locs] += coeff_vals * sinogram[i]
        return image.reshape(1, -1)


def TestRadonOpLines(radon_op):
    # write actual test in pycsou
    print("Testing RadonOpLines\n")
    tol = 1e-10
    # test apply method on constant image
    test_image = np.ones((radon_op.nrows, radon_op.ncols))
    test_sinogram = radon_op.apply(test_image.reshape(1, -1))
    test_sinogram = test_sinogram.squeeze()
    error = np.zeros(test_sinogram.size)
    for i in range(test_sinogram.size):
        coord_length = np.linalg.norm(radon_op.startpoints[i, :] - radon_op.endpoints[i, :]) * radon_op.h
        error[i] = np.abs(test_sinogram[i] - coord_length)
    assert np.max(error) < tol, "Apply method on 1-constant image does not output coord-lengths"
    print("Apply method correctly outputs coord lengths on 1-constant image\n")
    # test adjoint implementation
    np.random.seed(0)
    test_vec = np.random.rand(1, radon_op.nbLoS)
    test_image = np.random.rand(radon_op.nrows, radon_op.ncols)
    test_sinogram = radon_op.apply(test_image.reshape(1, -1))
    test_backproject = radon_op.adjoint(test_vec)
    scalar_product_forward = np.dot(test_sinogram.reshape(-1), test_vec.reshape(-1))
    scalar_product_adjoint = np.dot(test_image.reshape(-1), test_backproject.reshape(-1))
    assert np.abs(scalar_product_forward - scalar_product_adjoint) < tol, "Adjoint does not satisfy (Rx)^Ty=x^T(R^*y)"
    print("Adjoint satisfies (Rx)^Ty=x^T(R^*y) on random x, y")
