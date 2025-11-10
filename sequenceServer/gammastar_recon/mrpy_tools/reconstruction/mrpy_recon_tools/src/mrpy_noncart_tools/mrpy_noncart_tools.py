"""!
@brief Collection of tools which are used for non-cartesian reconstruction tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import math
import numpy as np
from typing import Tuple, List

def prep_kaiser_bessel_kernel(n_samples: int, beta: float) -> np.ndarray:
    """!
    @brief Function which prepares kaiser-bessel values as used during gridding routine. Note: A beta value of 18.557 is
           recommended for an oversampling factor of 2 and a window width of 4.

    @param n_samples: (int) Number of desired samples
    @param beta: (float) Beta value to control kernel shape

    @return
        - (np.ndarray) prepared kaiser-bessel values in array of size n_samples

    @author Jörn Huber
    """
    if n_samples == 0:
        raise ValueError("Number of input samples needs to be larger than 0")
    if beta <= 0:
        raise ValueError("Beta needs to be positive > 0")
    return np.kaiser(n_samples, beta)[int(n_samples/2):-1]


def calc_equidistant_radial_trajectory_2D(radial_data_dims: np.ndarray, radial_acq_angle: float) -> np.ndarray:
    """!
    @brief Calculates the trajectory of radial samples. Assumes equidistant spacing in sampled k-space data.

    @param radial_data_dims: (np.ndarray) 2D vector (num_col, num_lin) containing the dimensions of the acquired
                             spoke/blade e.g., (64, 1) for a radial spoke with 64 equidistant samples or (64, 16) for a
                             propeller blade of size 64 (RO)x16 (PE).
    @param radial_acq_angle: (float) Angle of acquisition in k-space in radians

    @return
        - (np.ndarray) 3D vector of size (num_col, num_lin, 2), containing x- and y- sampling positions of the spoke
                       blade, e.g., (:,0,0) contains x coordinates for a radial spoke and (:,:,0) contains x-coordinates
                       of a 2D blade.

    @author Jörn Huber
    """
    if not isinstance(radial_data_dims, np.ndarray) or not len(radial_data_dims) == 2:
        raise ValueError("Expected radial_data_dims array with two entries")
    if radial_data_dims.dtype == complex:
        raise ValueError("Expected real values for radial_data_dims array")
    if isinstance(radial_acq_angle, complex):
        raise ValueError("Expected a real value for radial_acq_angle")

    traj = np.zeros((radial_data_dims[0], radial_data_dims[1], 2))
    center_x = radial_data_dims[0] / 2
    if np.mod(radial_data_dims[1],2) == 1:
        center_y = (radial_data_dims[1]-1) / 2 #Radial Spoke, we do not want the center to be located at float positions
    else:
        center_y = radial_data_dims[1] / 2
    for i_x in range(0, radial_data_dims[0]):
        for i_y in range(0, radial_data_dims[1]):
            traj[i_x, i_y, 0] = (np.cos(radial_acq_angle)*(i_x - center_x)
                                 - np.sin(radial_acq_angle)*(i_y - center_y) + center_x)
            traj[i_x, i_y, 1] = (np.sin(radial_acq_angle)*(i_x - center_x)
                                 + np.cos(radial_acq_angle)*(i_y - center_y) + center_x)
    return traj


def grid_data_to_matrix_2D(
    data: np.ndarray,
    traj: np.ndarray,
    os_factor: float,
    window_width: float,
    kernel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """!
    @brief Applies gridding routine to acquired blade data.

    @param data: (np.ndarray) containing complex k-space data of size (num_col, num_lin, num_acq)
    @param traj: (np.ndarray) 4D numpy array of size (num_col, num_lin, 2, num_acq) containing real valued
                 trajectory data
    @param os_factor: (float) Oversampling factor of target grid, 2.0 is recommended
    @param window_width: (float) Width of gridding window (4 recommended for best quality)
    @param kernel: (np.ndarray) Precalculated kernel values

    @return
        - (np.ndarray) gridded data of size (num_col*os_factor, num_col*os_factor)

    @author Jörn Huber
    """
    if not isinstance(data, np.ndarray) or not len(data.shape) == 3:
        raise ValueError("data needs to be a 3D numpy ndarray with complex entries")
    if not isinstance(traj, np.ndarray) or data.dtype is complex or not len(traj.shape) == 4:
        raise ValueError("traj needs to be a 4D numpy ndarray with real-valued entries")
    if not isinstance(os_factor, float):
        raise ValueError("os_factor needs to be of type float (e.g. 2.0)")
    if not isinstance(window_width, float):
        raise ValueError("window_width factor needs to be of type float (e.g. 4.0)")
    if not isinstance(kernel, np.ndarray) or data.dtype is complex:
        raise ValueError("kernel needs to be a numpy array with real-valued entries!")

    num_ro, num_pe, num_acq = data.shape
    gridded_cplx = np.zeros((int(num_ro*os_factor), int(num_ro*os_factor)), dtype=complex)
    post_dens = np.zeros((int(num_ro*os_factor), int(num_ro*os_factor)))
    for i_acq in range(0, num_acq):
        for i_ro in range(0, num_ro):
            for i_pe in range(0, num_pe):
                x_coord = traj[i_ro, i_pe, 0, i_acq] * os_factor
                y_coord = traj[i_ro, i_pe, 1, i_acq] * os_factor
                x_min = int(np.ceil(x_coord - window_width * os_factor / 2))
                x_max = int(np.ceil(x_coord + window_width * os_factor / 2))
                y_min = int(np.ceil(y_coord - window_width * os_factor / 2))
                y_max = int(np.ceil(y_coord + window_width * os_factor / 2))

                x_min = max(x_min, 0)
                x_max = min(x_max, gridded_cplx.shape[0])
                y_min = max(y_min, 0)
                y_max = min(y_max, gridded_cplx.shape[0])

                for x_counter in range(x_min, x_max):
                    dist_x = np.abs(x_coord - x_counter)
                    weight_ind_x = int(dist_x / (window_width * os_factor / 2) * kernel.shape[0])
                    if weight_ind_x >= kernel.shape[0]:
                        weight_ind_x = kernel.shape[0]-1

                    for y_counter in range(y_min, y_max):
                        dist_y = np.abs(y_coord - y_counter)
                        weight_ind_y = int(dist_y / (window_width * os_factor / 2) * kernel.shape[0])
                        if weight_ind_y >= kernel.shape[0]:
                            weight_ind_y = kernel.shape[0]-1

                        weight = kernel[weight_ind_x]*kernel[weight_ind_y]
                        gridded_cplx[x_counter, y_counter] = (gridded_cplx[x_counter, y_counter] +
                                                              weight*data[i_ro, i_pe, i_acq])
                        post_dens[x_counter, y_counter] = post_dens[x_counter, y_counter] + weight

    # Post density compensation
    gridded_cplx_dens = np.copy(gridded_cplx)
    for i_ro in range(0, gridded_cplx.shape[0]):
        for i_pe in range(0, gridded_cplx.shape[1]):
            if post_dens[i_ro, i_pe] > 0.0:
                gridded_cplx_dens[i_ro, i_pe] = gridded_cplx[i_ro, i_pe]/post_dens[i_ro, i_pe]

    return gridded_cplx_dens, gridded_cplx


def get_deconvolution_matrix_2D(
    matrix_size: int,
    os_factor: float,
    window_width: int,
    kernel: np.ndarray
) -> np.ndarray:
    """!
    @brief Calculates the deconvolution matrix by gridding a delta pulse and calculating the FFT.

    @param matrix_size: (int) Original size of matrix e.g. 64 for spokes of size (64, 1)
    @param os_factor: (float) Oversampling factor of target grid, 2.0 is recommended
    @param window_width: (int) Width of gridding window (4 recommended for best quality)
    @param kernel: (np.ndarray) Precalculated kernel values

    @return
        - (np.ndarray) real-valued deconvolution matrix of size (num_col*os_factor, num_col*os_factor)

    @author Jörn Huber
    """
    traj = np.ones((matrix_size, 1, 2, 1))
    for i_sample in range(0, matrix_size):
        traj[i_sample, 0, 0, 0] = i_sample
        traj[i_sample, 0, 1, 0] = matrix_size/2
    data = np.zeros((matrix_size, 1, 1), dtype=complex)
    data[int(matrix_size/2), 0, 0] = 1
    delta_pulse_gridded_dens, delta_pulse_gridded = grid_data_to_matrix_2D(data, traj, os_factor, window_width, kernel)
    decon_mat = np.abs(np.fft.fftshift(np.fft.ifft2(np.squeeze(delta_pulse_gridded))))
    #decon_mat = decon_mat / np.max(decon_mat)
    return decon_mat


def apply_deconvolution_2D(
    data_grid: np.ndarray,
    decon_mat: np.ndarray,
    os_factor: float
) -> np.ndarray:
    """!
    @brief Applies deconvolution to gridded data using a precalculated deconvolution matrix.

    @param data_grid: (np.ndarray) gridded complex k-space data (num_col, num_lin)
    @param decon_mat: (np.ndarray) real_valued deconvolution matrix
    @param os_factor: (float) Applied oversampling factor during gridding

    @return
        - (np.ndarray) image space (num_col/os_factor, num_lin/os_factor) with applied deconvolution

    @author Jörn Huber
    """
    num_col, num_lin = data_grid.shape
    im = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_grid)))
    im_origfov = im[int(num_col / os_factor / 2):-int(num_col / os_factor / 2),
                    int(num_lin / os_factor / 2):-int(num_lin / os_factor / 2)]
    decon_mat_origfov = decon_mat[int(num_col / os_factor / 2):-int(num_col / os_factor / 2),
                                  int(num_lin / os_factor / 2):-int(num_lin / os_factor / 2)]
    num_col_orig, num_lin_orig = im_origfov.shape
    for i_col in range(0, num_col_orig):
        for i_lin in range(0, num_lin_orig):
            im_origfov[i_col, i_lin] = im_origfov[i_col, i_lin] / decon_mat_origfov[i_col, i_lin]
    return im_origfov


def prop_cut_kspace_edges_2D(data: np.ndarray) -> None:
    """!
    @brief Cuts outer edges of k-space. Needed for e.g. rotation correction.

    @param data: (np.ndarray) 3D ndarray of gridded blade data (num_col, num_col, num_acq)

    @return
        - (np.ndarray) data but with edges set to zero

    @author Jörn Huber
    """
    num_col, num_lin, num_acq = data.shape
    if not num_col == num_lin:
        raise ValueError("Number of columns and lines in data must match")
    for i_col in range(0, num_col):
        for i_lin in range(0, num_lin):
            dist = np.sqrt((i_col - num_col/2)**2 + (i_lin - num_lin/2)**2)
            if dist > num_col/2:
                data[i_col, i_lin, :] = np.zeros(num_acq)


def prop_phase_correction_2D(
    data: np.ndarray,
    filter_type: str = "rhomb",
    fov: List[float] = [1, 1]
) -> np.ndarray:
    """!
    @brief Removes low order phase shifts from blade image data, which is the result from eddy currents etc.
    @details Phase correction is needed as blades need to share a common k-space center before gridding. Therefore,
             k-space frequencies are low-pass filtered such that low order phase variation is preserved in image space
             and such phase variation is removed from individual blades by multiplication with the filtered complex
             conjugate.

    @param data: (np.ndarray) 2D (num_col, num_lin) numpy array of complex kspace blade data
    @param filter_type: (str) Type of filter to be used. Either "rhomb" or "square".
    @param fov: (list) Field of view for square filter calculation (only relevant if FOV is not quadratic).

    @return
        - (np.ndarray) 2D complex numpy array of phase corrected k-space data

    @author Jörn Huber, Tom Lütjen
    """
    if len(data.shape) != 2:
        raise ValueError("Data needs to be a 2D numpy ndarray with complex entries")

    num_col, num_lin = data.shape
    triang_filter = np.zeros((num_col, num_lin))
    for i_col in range(0, num_col):
        for i_lin in range(0, num_lin):
            if filter_type == "rhomb":
                triang_filter[i_col, i_lin] = ((num_col / 2 - np.abs(i_col - num_col / 2))
                                            * (num_lin / 2 - np.abs(i_lin - num_lin / 2)))
            elif filter_type == "square":
                triang_filter[i_col, i_lin] = ((max(num_col/fov[0], num_lin/fov[1])/(num_col/fov[0]))*num_col / 2 - np.abs(i_col - num_col / 2)) \
                                            * ((max(num_col/fov[0], num_lin/fov[1])/(num_lin/fov[1]))*num_lin / 2 - np.abs(i_lin - num_lin / 2))
            else:
                raise ValueError("Unknown filter type. Only rhomb and square are supported.")
    
    triang_filter = triang_filter / np.amax(triang_filter)

    im = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.squeeze(data))))
    data_filt = np.multiply(np.squeeze(data), triang_filter)
    im_filt = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_filt)))
    im_filt = im_filt / np.abs(im_filt)
    im = np.multiply(im, np.conj(im_filt))
    data_corr = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im)))

    return data_corr

def prop_calc_ksp_coverage(
    traj: np.ndarray,
    num_grid_points_per_axis: int
) -> float:
    """!
    @brief Calculates k-space coverage of PROPELLER trajectory.
    @details Calculates fraction of k-space covered by all PROPELLER blades compared to a full k-space coverage, i.e. infinite number of blades
    with arbitraly small blade angle increment.

    @param traj: (np.ndarray) 4D numpy array of size (num_col, num_lin, 2, num_acq) containing real valued
                 trajectory data centered around 0.
    @param num_grid_points_per_axis: (int) Number of grid points per axis to determine the accuracy of the k-space coverage.
    
    @return
        - (float) k-space coverage in percent

    @author Tom Lütjen
    """
    num_col = traj.shape[0]
    num_lin = traj.shape[1]
    num_acq = traj.shape[3]
    grid_points_x = np.linspace(-num_col//2, num_col//2, num_grid_points_per_axis)
    grid_points_y = np.linspace(-num_col//2, num_col//2, num_grid_points_per_axis)
    rad = np.sqrt((num_col//2)**2 + (num_lin//2)**2)
    
    dummy_part = np.zeros((num_grid_points_per_axis, num_grid_points_per_axis))
    dummy_full = np.zeros((num_grid_points_per_axis, num_grid_points_per_axis))
    for count_x, x_idx in enumerate(grid_points_x):
        for count_y, y_idx in enumerate(grid_points_y):
            coor_cur = np.array([x_idx, y_idx])
            if np.sqrt((x_idx) ** 2 + (y_idx) ** 2) <= rad:
                dummy_full[count_x, count_y] = 1
            for blade_idx in range(num_acq):
                if ((len(tuple(np.argwhere(traj[:,:, 1, blade_idx] == traj[:,:, 1, blade_idx].min()))) == num_col) and (
                    len(tuple(np.argwhere(traj[:,:, 1, blade_idx] == traj[:,:, 1, blade_idx].max()))) == num_col) and (
                    len(tuple(np.argwhere(traj[:,:, 0, blade_idx] == traj[:,:, 0, blade_idx].min()))) == num_lin) and (
                    len(tuple(np.argwhere(traj[:,:, 0, blade_idx] == traj[:,:, 0, blade_idx].max()))) == num_lin)) or (
                    (len(tuple(np.argwhere(traj[:,:, 1, blade_idx] == traj[:,:, 1, blade_idx].min()))) == num_lin) and (
                    len(tuple(np.argwhere(traj[:,:, 1, blade_idx] == traj[:,:, 1, blade_idx].max()))) == num_lin) and (
                    len(tuple(np.argwhere(traj[:,:, 0, blade_idx] == traj[:,:, 0, blade_idx].min()))) == num_col) and (
                    len(tuple(np.argwhere(traj[:,:, 0, blade_idx] == traj[:,:, 0, blade_idx].max()))) == num_col)):

                    min_idx_x = (0, 0)
                    max_idx_x = (num_col-1, num_lin-1)
                    min_idx_y = (num_col-1, 0)
                    #max_idx_y = (0, num_lin-1)
                else:
                    min_idx_x = tuple(np.argwhere(traj[:,:, 1, blade_idx] == traj[:,:, 1, blade_idx].min())[0])
                    max_idx_x = tuple(np.argwhere(traj[:,:, 1, blade_idx] == traj[:,:, 1, blade_idx].max())[-1])
                    min_idx_y = tuple(np.argwhere(traj[:,:, 0, blade_idx] == traj[:,:, 0, blade_idx].min())[0])
                    #max_idx_y = tuple(np.argwhere(traj[bin_idx, 0, :, :, blade_idx] == traj[bin_idx, 0, :, :, blade_idx].max())[-1])
                

                x_min_x = traj[:,:, 0, blade_idx][min_idx_x]
                x_min_y = traj[:,:, 1, blade_idx][min_idx_x]
                x_max_x = traj[:,:, 0, blade_idx][max_idx_x]
                x_max_y = traj[:,:, 1, blade_idx][max_idx_x]
                y_min_x = traj[:,:, 0, blade_idx][min_idx_y]
                y_min_y = traj[:,:, 1, blade_idx][min_idx_y]
                #y_max_x = traj[bin_idx, 0, :, :, blade_idx][max_idx_y]
                #y_max_y = traj[bin_idx, 1, :, :, blade_idx][max_idx_y]

                a = np.array([y_min_x, y_min_y])
                b = np.array([x_min_x, x_min_y])
                #c = np.array([y_max_x, y_max_y])
                d = np.array([x_max_x, x_max_y])

            
                if (0 < np.dot(coor_cur - a, b - a) < np.dot(b - a, b - a)) and (0 < np.dot(coor_cur - a, d - a) < np.dot(d - a, d - a)):
                    dummy_part[count_x, count_y] = 1
    
    cover = 100 * np.count_nonzero(dummy_part) / np.count_nonzero(dummy_full)
    return cover

def calc_propeller_blade_increment_from_trajs(
    trajectory_line_blade_1: np.ndarray,
    trajectory_line_blade_2: np.ndarray
) -> float:
    """!
    @brief Calculates the PROPELLER blade angle from two trajectory lines between two PROPELLER blades based on the
           scalar product of two direction vectors which are extracted from the trtajectory.

    @param trajectory_line_blade_1: (np.ndarray) 2D numpy array of size (num_col, 2) containing real valued
                                                 trajectory data for first blade.
    @param trajectory_line_blade_2: (np.ndarray) 2D numpy array of size (num_col, 2) containing real valued
                                                 trajectory data for second blade.

    @return
        - (float) Angle between the two blades in degrees.

    @author Jörn Huber
    """

    vec_1 = trajectory_line_blade_1[0, :]
    vec_2 = trajectory_line_blade_2[0, :]
    return np.arccos(vec_1 @ vec_2/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))/math.pi*180.0
