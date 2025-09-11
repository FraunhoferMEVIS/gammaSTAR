"""!
@brief Collection of tools which are used for coil/channel reconstruction tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np


def sum_of_squares_combination(im):
    """!
    @brief Combines multi-coil MRI data using the sum of squares (SOS) method.

    @param im: (np.ndarray): Multi-coil MRI data of shape (nx, ny, nc), where nx is the number of columns, ny is the
                             number of lines, and nc is the number of channels.

    @return
        - (np.ndarray) Combined image of shape (nx, ny), where nx is the number of columns and ny is the number of lines.

    @author Jörn Huber
    """

    return np.sqrt(np.sum(np.abs(im) ** 2, axis=-1))

def estimate_sens(ksp, interpol_fact = 1):
    """!
    @brief Calculates the coil sensitivity maps from data using an eigenvalue approach as described by Walsh et. al

    @param ksp: (np.ndarray) 2D (num_col, num_lin) or 3D (num_col, num_lin, num_channel) measured k-space data.
    @param interpol_fact: (int) Interpolation factor, which determines the size of the estimated sensitivity map. If
                          interpol_fact = 2, only every second pixel is used for value estimation and the resulting
                          map has half the resolution of the original image. In the end, the map is interpolated to
                          the full resolution again. In practice, factor of 2, 4 or even 8 should be feasible as the
                          sensitivity maps are slowly variying.

    @return
        - (np.ndarray) Estimated sensitivity maps of size (num_col, num_lin, num_channel)

    @author Jörn Huber
    """
    shape = ksp.shape
    if len(shape) == 2:
        num_col, num_lin = shape
        num_cha = 1
        ksp = ksp[..., np.newaxis]
    elif len(shape) == 3:
        num_col, num_lin, num_cha = shape
    else:
        raise ValueError("Input data must have shape (Nx, Ny, Ncha).")

    im = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ksp, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    block_size_lin = max(num_lin/16, 3) #Empirically chosen
    if block_size_lin%2 == 0:
        block_size_lin = block_size_lin - 1
    block_size_col = max(num_col/16, 3) #Empirically chosen
    if block_size_col%2 == 0:
        block_size_col = block_size_col - 1

    sens_map = np.zeros((int(num_col/interpol_fact), int(num_lin/interpol_fact), num_cha), dtype=complex)
    for ix in range(0, num_col, interpol_fact):
        i_col_min = int(max(ix - (block_size_col - 1) / 2, 0))
        i_col_max = int(min(ix + (block_size_col - 1) / 2, num_col - 1))
        for iy in range(0, num_lin, interpol_fact):
            i_min_lin = int(max(iy - (block_size_lin - 1) / 2, 0))
            i_max_lin = int(min(iy + (block_size_lin - 1) / 2, num_lin - 1))

            sig_mat = im[i_col_min:i_col_max + 1, i_min_lin:i_max_lin + 1].reshape(-1, num_cha).T
            sig_mat_quad = sig_mat @ sig_mat.conj().T
            w, v = np.linalg.eigh(sig_mat_quad)
            max_eig_val = np.argmax(np.abs(w))
            max_eig_vec = v[:, max_eig_val]
            sens_map[ix // interpol_fact, iy // interpol_fact] = max_eig_vec * np.exp(-1j * np.angle(max_eig_vec[0]))

    return sens_map

def combine_channels(ksp, sens):
    """!
    @brief Combines k-space data from multiple channels using provided sensitivity maps.
    @details Performs optimal channel combination in image space using the sensitivity maps, then transforms back to k-space.

    @param ksp: (np.ndarray) K-space data of shape (num_col, num_lin, num_cha).
    @param sens: (np.ndarray) Coil sensitivity maps of shape (num_col, num_lin, num_cha).

    @return
        - (np.ndarray) Combined k-space data of shape (num_col, num_lin).
        - (np.ndarray) Combined image of shape (num_col, num_lin).

    @author Jörn Huber
    """

    if not ksp.shape == sens.shape:
        raise ValueError("Input dimensions of kspace and coil sensitivities must match.")
    if len(ksp.shape) == 2:
        im = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ksp)))
        return ksp, im

    im = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ksp, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    sens_conj = np.conjugate(sens)
    scl = np.sum(sens_conj * sens, axis=2)
    opt_coeff = sens_conj / np.sqrt(scl[:, :, np.newaxis])

    im_combine = np.sum(opt_coeff * im, axis=2)

    ksp_combine = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im_combine)))

    return ksp_combine, im_combine
