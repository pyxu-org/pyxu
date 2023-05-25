import numpy as np
import scipy.io as scio

import tomo_fusion.forward_model.LoS_handling as pyrd


class RADCAM_system:
    def __init__(self):
        bolo_chords = scio.loadmat("forward_model/chords/bolo_chord_start_end_points.mat")["B"]
        sxr_chords = scio.loadmat("forward_model/chords/sxr_chord_start_end_points.mat")["A"]
        axuv_chords = scio.loadmat("forward_model/chords/axuv_chord_start_end_points.mat")["C"]
        self.bolo_xchords = bolo_chords[: round(bolo_chords.shape[0] / 2), :]
        self.bolo_ychords = bolo_chords[round(bolo_chords.shape[0] / 2) :, :]
        self.sxr_xchords = sxr_chords[: round(sxr_chords.shape[0] / 2), :]
        self.sxr_ychords = sxr_chords[round(sxr_chords.shape[0] / 2) :, :]
        self.axuv_xchords = axuv_chords[: round(axuv_chords.shape[0] / 2), :]
        self.axuv_ychords = axuv_chords[round(axuv_chords.shape[0] / 2) :, :]
        # tiles coordinates
        self.x_left_tile = 0.624
        self.x_right_tile = 1.124  # 1.138, approximated for now
        self.y_lower_tile = -0.75
        self.y_upper_tile = 0.75
        self.tile_extent = np.array(
            [
                [0.972, -0.75],
                [1.124, -0.555],
                [1.124, 0.555],
                [0.972, 0.75],
                [0.67, 0.75],
                [0.624, 0.704],
                [0.624, -0.704],
                [0.67, -0.75],
                [0.972, -0.75],
            ]
        )
        self.tile_extent[:, 0] -= self.x_left_tile
        self.tile_extent[:, 1] -= self.y_lower_tile
        self.tile_extent_plot = self.tile_extent
        # for plotting purposes, we flip to have origin at upper left corner and normalize. We will also need to shift by -0.5*h, with h discretization step
        self.tile_extent_plot[:, 0] /= 0.5
        self.tile_extent_plot[:, 1] = (1.5 - self.tile_extent[:, 1]) / 1.5
        # shift coordinates to have origin at lower left corner, corresponding to (r,z)=(0.624,-0.75)
        self.bolo_xchords -= self.x_left_tile
        self.bolo_ychords -= self.y_lower_tile
        self.sxr_xchords -= self.x_left_tile
        self.sxr_ychords -= self.y_lower_tile
        self.axuv_xchords -= self.x_left_tile
        self.axuv_ychords -= self.y_lower_tile
        # compute LoS_parametrizations
        center = np.array([(self.x_right_tile - self.x_left_tile) / 2, (self.y_upper_tile - self.y_lower_tile) / 2])
        self.bolo_LoS_params, self.bolo_startpoints, self.bolo_endpoints = pyrd.generate_LoS_from_point_couples(
            self.bolo_xchords, self.bolo_ychords, center
        )
        self.sxr_LoS_params, self.sxr_startpoints, self.sxr_endpoints = pyrd.generate_LoS_from_point_couples(
            self.sxr_xchords, self.sxr_ychords, center
        )
        self.axuv_LoS_params, self.axuv_startpoints, self.axuv_endpoints = pyrd.generate_LoS_from_point_couples(
            self.axuv_xchords, self.axuv_ychords, center
        )
