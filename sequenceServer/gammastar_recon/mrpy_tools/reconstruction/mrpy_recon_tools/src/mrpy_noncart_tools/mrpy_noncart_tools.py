"""!
@brief Collection of tools which are used for non-cartesian reconstruction tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import mrinufft
from mrinufft.density import voronoi


def calc_equidistant_propeller_trajectory(
        ksp_data: np.ndarray,
        ro_axis: int,
        pe_axis: int,
        acq_axis: int | None,
        acq_angle_incr: float,
) -> np.ndarray:
    """!
    @brief Calculates the trajectory of propeller samples. Assumes equidistant spacing in sampled k-space data.

    @param ksp_data: k-space data array for which the gridding trajectory shall be calculated
    @param ro_axis: Integer, indicating the readout axis of the k-space array
    @param pe_axis: Integer, indicating the phase encoded axis of the k-space array
    @param acq_axis: Integer, indicating the acquisition axis of the kspace array
    @param acq_angle_incr: Angular increment

    @return
        - Numpy array with same dimensions as ksp_data with additional dimensions at -1 of size 2, containing grid
          positions

    @author Jörn Huber
    """

    traj_shape = list(ksp_data.shape)
    traj_shape.append(2)
    traj_shape = tuple(traj_shape)
    traj = np.zeros(traj_shape)

    center_x = traj.shape[ro_axis] / 2
    center_y = traj.shape[pe_axis] / 2

    traj_x = np.arange(-center_x, center_x)
    traj_y = np.arange(-center_y, center_y)

    traj_orig = np.zeros((2, traj.shape[ro_axis], traj.shape[pe_axis]))
    traj_orig[0, :, :] = np.repeat(traj_x[:, None], traj.shape[pe_axis], 1)
    traj_orig[1, :, :] = np.repeat(traj_y[None, :], traj.shape[ro_axis], 0)

    traj_reshaped = np.expand_dims(np.moveaxis(traj, ro_axis, -1), ro_axis)
    traj_reshaped = np.expand_dims(np.moveaxis(traj_reshaped, pe_axis, -1), pe_axis)
    traj_reshaped = np.expand_dims(np.moveaxis(traj_reshaped, acq_axis, -1), acq_axis)
    moved_shape = traj_reshaped.shape
    traj_reshaped = np.reshape(
        traj_reshaped, [-1, 2, traj_reshaped.shape[-3], traj_reshaped.shape[-2], traj_reshaped.shape[-1]]
    )

    traj_rot = np.zeros_like(traj_orig)
    for i_acq in range(traj_reshaped.shape[-1]):
        rad_ang = (acq_angle_incr * float(i_acq))/180.0*np.pi
        traj_rot[0, :, :] = traj_orig[0, :, :] * np.cos(rad_ang) - traj_orig[1, :, :] * np.sin(rad_ang)
        traj_rot[1, :, :] = traj_orig[0, :, :] * np.sin(rad_ang) + traj_orig[1, :, :] * np.cos(rad_ang)
        traj_reshaped[:, 0, :, :, i_acq] = np.repeat(traj_rot[None, 0, :, :], traj_reshaped.shape[0], 0)
        traj_reshaped[:, 1, :, :, i_acq] = np.repeat(traj_rot[None, 1, :, :], traj_reshaped.shape[0], 0)

    traj_reshaped = np.reshape(traj_reshaped, moved_shape)
    traj_reshaped = np.squeeze(np.moveaxis(traj_reshaped, -1, acq_axis), acq_axis + 1)
    traj_reshaped = np.squeeze(np.moveaxis(traj_reshaped, -1, pe_axis), pe_axis + 1)
    traj_reshaped = np.squeeze(np.moveaxis(traj_reshaped, -1, ro_axis), ro_axis + 1)

    return traj_reshaped


def apply_nufft(
        ksp_data: np.ndarray,
        traj_data: np.ndarray,
        ro_axis: int,
        ch_axis: int,
        acq_axis: int,
        mat_size: int,
        pe1_axis: int | None = None,
        pe2_axis: int | None = None,
        density_method = 'voronoi'
) -> np.ndarray:
    """!
    @brief Applies non-Cartesian image reconstruction using functionality from MRI Nufft

    @param ksp_data: k-space data array which contains non-Cartesian raw data which shall be used during image recon.
    @param traj_data: Trajectory data array which contains the sample points for corresponding data samples. Note that
                      the sample positions need to be stored in the "channel" axis of the kspace data array. All other
                      dims need to correspond to the ksp_data dims.
    @param ro_axis: RO axis of the k-space data.
    @param ch_axis: Channel axis of the k-space data.
    @param acq_axis: Acquisition axis of the k-space data.
    @param mat_size: Size of the target matrix after reconstruction.
    @param pe1_axis: PE1 axis of the k-space data (if applicable e.g. in 2D PROPELLER readouts).
    @param pe2_axis: PE2 axis of the k-space data (if applicable e.g. in 3D PROPELLER readouts).
    @param density_method: Voronoi density will be used automatically, if "cell_count" is not provided. "cell_count"
                           should be used for PROPELLER applications.

    @return
        - Reconstructed image space data.


    @author Jörn Huber
    """

    nufft_operator = mrinufft.get_operator("finufft")

    # We create a data batch to avoid nested loops
    source_data = np.copy(ksp_data)
    source_data = np.expand_dims(np.moveaxis(source_data, ro_axis, -1), ro_axis)
    if pe1_axis is not None:
        source_data = np.expand_dims(np.moveaxis(source_data, pe1_axis, -1), pe1_axis)
    if pe2_axis is not None:
        source_data = np.expand_dims(np.moveaxis(source_data, pe2_axis, -1), pe2_axis)
    pe_merged_shape = list(source_data.shape)
    if pe1_axis is not None:
        pe_merged_shape.pop()
    if pe2_axis is not None:
        pe_merged_shape.pop()
    if pe1_axis is not None and pe2_axis is not None:
        pe_merged_shape[-1] = ksp_data.shape[ro_axis] * ksp_data.shape[pe1_axis] * ksp_data.shape[pe2_axis]
    elif pe1_axis is not None and pe2_axis is None:
        pe_merged_shape[-1] = ksp_data.shape[ro_axis] * ksp_data.shape[pe1_axis]
    elif pe1_axis is None and pe2_axis is not None:
        pe_merged_shape[-1] = ksp_data.shape[ro_axis] * ksp_data.shape[pe2_axis]

    source_data = np.reshape(source_data, pe_merged_shape)
    source_data = np.expand_dims(np.moveaxis(source_data, ch_axis, -1), ch_axis)
    source_data = np.expand_dims(np.moveaxis(source_data, acq_axis, -1), acq_axis)
    batch_data = np.reshape(
        source_data,
        shape = [-1, source_data.shape[-3], source_data.shape[-2], source_data.shape[-1]]
    ) # Format: [BATCHDIM, RO, CHA, ACQ]
    batch_data = np.transpose(batch_data, [0, 2, 3, 1])  # Format expected by mrinufft: [BATCHDIM, CHA, ACQ, RO]

    # We create the corresponding trajectory batch
    source_traj = np.copy(traj_data)
    source_traj = np.expand_dims(np.moveaxis(source_traj, ro_axis, -1), ro_axis)
    if pe1_axis is not None:
        source_traj = np.expand_dims(np.moveaxis(source_traj, pe1_axis, -1), pe1_axis)
    if pe2_axis is not None:
        source_traj = np.expand_dims(np.moveaxis(source_traj, pe2_axis, -1), pe2_axis)
    pe_merged_shape = list(source_traj.shape)
    if pe1_axis is not None:
        pe_merged_shape.pop()
    if pe2_axis is not None:
        pe_merged_shape.pop()
    if pe1_axis is not None and pe2_axis is not None:
        pe_merged_shape[-1] = traj_data.shape[ro_axis] * traj_data.shape[pe1_axis] * traj_data.shape[pe2_axis]
    elif pe1_axis is not None and pe2_axis is None:
        pe_merged_shape[-1] = traj_data.shape[ro_axis] * traj_data.shape[pe1_axis]
    elif pe1_axis is None and pe2_axis is not None:
        pe_merged_shape[-1] = traj_data.shape[ro_axis] * traj_data.shape[pe2_axis]
    source_traj = np.reshape(source_traj, pe_merged_shape)

    source_traj = np.expand_dims(np.moveaxis(source_traj, ch_axis, -1), ch_axis)
    source_traj = np.expand_dims(np.moveaxis(source_traj, acq_axis, -3), acq_axis)
    batch_traj = np.reshape(
        source_traj,
        shape = [-1, source_traj.shape[-3], source_traj.shape[-2], source_traj.shape[-1]]
    ) # Format expected by mrinufft: [BATCHDIM, ACQ, RO, POS]

    # Prepare nufft arrays
    nufft_dim = batch_traj.shape[-1]
    if nufft_dim == 2:
        target_batch_shape = [batch_data.shape[0], batch_data.shape[1], mat_size, mat_size]
    else:
        target_batch_shape = [batch_data.shape[0], batch_data.shape[1], mat_size, mat_size, mat_size]
    target_data = np.zeros(tuple(target_batch_shape), dtype=complex)
    nufft_shape = (mat_size, mat_size) if nufft_dim == 2 else (mat_size, mat_size, mat_size)

    # Process batch
    for batch_idx in range(batch_data.shape[0]):

        grid_data = np.reshape(batch_data[batch_idx,], (batch_data.shape[1], -1))
        samples_loc = batch_traj[batch_idx,] / np.max(batch_traj[batch_idx,]) * np.pi

        if density_method == 'cell_count':
            density = mrinufft.density.geometry_based.cell_count(
                samples_loc.reshape(-1, nufft_dim),
                shape=nufft_shape,
                osf=0.5
            )
        else:
            density = voronoi(samples_loc.reshape(-1, nufft_dim))

        nufft = nufft_operator(
            samples_loc.reshape(-1, nufft_dim),
            shape = nufft_shape,
            density=density,
            n_coils=ksp_data.shape[ch_axis]
        )

        if ksp_data.shape[ch_axis] == 1:
            target_data[batch_idx,] = np.expand_dims(nufft.adj_op(grid_data), 0)
        else:
            target_data[batch_idx,] = nufft.adj_op(grid_data)

    # Back to unbatched structure
    if nufft_dim == 2:
        target_data = np.flip(np.transpose(target_data, [0, 1, 3, 2]), 2)
    else:
        target_data = np.flip(np.transpose(target_data, [0, 1, 4, 3, 2]), 3)
    unbatched_shape = list(ksp_data.shape)
    unbatched_shape[ro_axis] = mat_size

    if pe1_axis is None:
        unbatched_shape[acq_axis] = mat_size
    else:
        unbatched_shape[acq_axis] = 1
        unbatched_shape[pe1_axis] = mat_size
    if nufft_dim == 3:
        unbatched_shape[pe2_axis] = mat_size

    if nufft_dim == 2:
        target_resh_data = np.reshape(np.transpose(target_data, [3, 1, 2, 0]), unbatched_shape)
    else:
        target_resh_data = np.reshape(np.transpose(target_data, [4, 1, 3, 2, 0]), unbatched_shape)

    return target_resh_data


def calc_propeller_blade_increment_from_trajs(
        trajectory_line_blade_1: np.ndarray,
        trajectory_line_blade_2: np.ndarray
) -> float:
    """!
    @brief Calculates the PROPELLER blade angle from two trajectory lines between two PROPELLER blades based on the
           scalar product of two direction vectors which are extracted from the trtajectory.

    @param trajectory_line_blade_1: 2D numpy array of size (num_col, 2) containing real valued
                                    trajectory data for first blade.
    @param trajectory_line_blade_2: 2D numpy array of size (num_col, 2) containing real valued
                                    trajectory data for second blade.

    @return
        - Angle between the two blades in degrees.

    @author Jörn Huber
    """

    vec_1 = trajectory_line_blade_1[0, :]
    vec_2 = trajectory_line_blade_2[0, :]
    return np.arccos(vec_1 @ vec_2/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))/np.pi*180.0
