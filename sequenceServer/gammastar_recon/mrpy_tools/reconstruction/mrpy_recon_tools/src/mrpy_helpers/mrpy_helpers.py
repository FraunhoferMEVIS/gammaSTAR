"""!
@brief Collection of tools which provide helpful functionality for e.g. testing purposes.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import math
import numpy as np
import sigpy
import mrpy_parallel_tools as parallel_tools

def remove_readout_os(ksp_data, dim, os_factor):
    """!
    @brief Removes the readout oversampling which is applied by default.
    @details Might not correctly preserve the phase information, additional investigation is needed.

    @param ksp_data: (np.ndarray) N-dimensional kspace data, with readout dimensions in the first index.

    @return
        - (nd.array) 1D readout line of kspace with removed oversampling

    @author Jörn Huber
    """
    num_samples = ksp_data.shape[dim]
    samples_cutoff = int((num_samples-num_samples/os_factor)/2)

    ksp_data_fft = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(ksp_data, axes=dim), axis=dim), axes=dim)
    ksp_data_fft_os_removed = np.take(ksp_data_fft, range(samples_cutoff, int(num_samples-samples_cutoff)), axis=dim)
    ksp_data_os_removed = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ksp_data_fft_os_removed, axes=dim), axis=dim), axes=dim)
    return ksp_data_os_removed


def gauss_kern_val(pos_x, pos_y, pos_z, sigma):
    """!
    @brief Calculates gaussian kernel value for sensitivities

    @param pos_x: (float) column position
    @param pos_y: (float) line position
    @param pos_z: (float) z position
    @param sigma: (float) gaussian kernel sigma

    @return
        - (float) kernel value

    @author Jörn Huber
    """
    gauss_x = np.exp(-0.5 * np.square(pos_x) / np.square(sigma))
    gauss_y = np.exp(-0.5 * np.square(pos_y) / np.square(sigma))
    gauss_z = np.exp(-0.5 * np.square(pos_z) / np.square(sigma))
    return gauss_x*gauss_y*gauss_z

def create_sensitivity_test_data_2D(num_col, num_lin):
    """!
    @brief Creates nine artifical shepp-logan channels for testing purposes

    @param num_col: (int) number of columns
    @param num_lin: (int) number of lines

    @return
        - (np.ndarray) Multichannel kspaces of size (num_col, num_lin, num_cha)
        - (np.ndarray) Original shepp-logan phantom of size (num_col, num_lin)
        - (np.ndarray) Original sensitivities of size (num_col, num_lin, num_cha)

    @author Jörn Huber
    """
    sl = sigpy.shepp_logan((num_col, num_lin), dtype=complex)
    sens = np.zeros((num_col, num_lin, 9), dtype=complex)
    sl_sens = np.zeros((num_col, num_lin, 9), dtype=complex)
    map_index = 0
    for center_x in range(0, num_col + 1, int(num_col / 2)):
        for center_y in range(0, num_lin + 1, int(num_lin / 2)):
            for ix in range(0, num_col):
                for iy in range(0, num_lin):
                    sens[ix, iy, map_index] = gauss_kern_val((ix-center_x)/num_col, (iy-center_y)/num_lin, 0.0, 0.3)
                    sl_sens[ix, iy, map_index] = sens[ix, iy, map_index]*sl[ix, iy]
            map_index = map_index + 1
    sl_ksp_sens = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sl_sens, axes=(0, 1)),axes=(0, 1)), axes=(0, 1))
    return sl_ksp_sens, sl, sens

def create_partial_fourier_test_data_2D(num_col, num_lin, pf_fraction):
    """!
    @brief Creates nine artifical shepp-logan channels for testing purposes.
    @details k-space data is created which serves the purpose of testing partial fourier reconstruction algorithms.
             Therefore, data containing only a fraction of the sampled data is created for partial fourier
             applications. Note that this currently only supports 2D data.

    @param num_col: (int) number of columns
    @param num_lin: (int) number of lines
    @param pf_fraction: (float) Fraction of partial fourier lines. Should be >= 0.5.

    @return
        - (np.ndarray) Multichannel fractional sampled kspaces of size (num_col, num_lin, 9) based on
                       real valued channel data.
        - (np.ndarray) Multichannel fractional sampled kspaces of size (num_col, num_lin, 9) based on
                       cplx valued channel data.
        - (np.ndarray) Multichannel fully sampled kspaces of size (num_col, num_lin, num_cha) based on
                       real valued channel data.
        - (np.ndarray) Multichannel fully sampled kspaces of size (num_col, num_lin, num_cha) based on
                       cplx valued channel data.
        - (np.ndarray) Original shepp-logan phantom of size (num_col, num_lin)

    @author Jörn Huber
    """
    if num_col < 32 or num_lin < 32:
        raise ValueError("Simulated data should at least have a size of 32x32!")
    if pf_fraction < 0.5 or pf_fraction > 1.0:
        raise ValueError("Partial fourier fraction must lie within [0.5, 1.0]")

    # Generate sensitivities
    sl = sigpy.shepp_logan((num_col, num_lin), dtype=complex)
    sens = np.zeros((num_col, num_lin, 9), dtype=complex)
    sl_sens = np.zeros((num_col, num_lin, 9), dtype=complex)
    map_index = 0
    for center_x in range(0, num_col + 1, int(num_col / 2)):
        for center_y in range(0, num_lin + 1, int(num_lin / 2)):
            for ix in range(0, num_col):
                for iy in range(0, num_lin):
                    sens[ix, iy, map_index] = (gauss_kern_val((ix-center_x)/num_col, (iy-center_y)/num_lin, 0,0.3)
                                        + 1j * gauss_kern_val((ix-center_x)/num_col, (iy-center_y)/num_lin, 0,0.3))
                    sl_sens[ix, iy, map_index] = sens[ix, iy, map_index]*sl[ix, iy]
            map_index = map_index + 1
    ksp_sens_cplx = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sl_sens, axes=(0, 1)), axes=(0, 1)),
                                    axes=(0, 1))
    ksp_sens_real = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(np.real(sl_sens), axes=(0, 1)), axes=(0, 1)),
                                    axes=(0, 1))

    # Generate real and complex partial fourier data
    ksp_sens_real_pf = np.copy(ksp_sens_real)
    ksp_sens_cplx_pf = np.copy(ksp_sens_cplx)
    for i_lin in range(0, num_lin):
        if i_lin > int(pf_fraction*num_lin):
            ksp_sens_real_pf[:, i_lin, :] = np.zeros((num_col, 9), dtype=complex)
            ksp_sens_cplx_pf[:, i_lin, :] = np.zeros((num_col, 9), dtype=complex)

    return ksp_sens_real_pf, ksp_sens_cplx_pf, ksp_sens_real, ksp_sens_cplx, sl

def create_partial_fourier_test_data_3D(num_col, num_lin, num_par, pf_fraction_pe1, pf_fraction_pe2):
    """!
    @brief Creates nine artifical shepp-logan channels for testing purposes.
    @details k-space data is created which serves the purpose of testing partial fourier reconstruction algorithms.
             Therefore, data containing only a fraction of the sampled data is created for partial fourier
             applications. Note that this currently only supports 2D data. 

    @param num_col: (int) number of columns
    @param num_lin: (int) number of lines
    @param pf_fraction: (float) Fraction of partial fourier lines. Should be >= 0.5.

    @return
        - (np.ndarray) Multichannel fractional sampled kspaces of size (num_col, num_lin, 9) based on
                       real valued channel data.
        - (np.ndarray) Multichannel fractional sampled kspaces of size (num_col, num_lin, 9) based on
                       cplx valued channel data.
        - (np.ndarray) Multichannel fully sampled kspaces of size (num_col, num_lin, num_cha) based on
                       real valued channel data.
        - (np.ndarray) Multichannel fully sampled kspaces of size (num_col, num_lin, num_cha) based on
                       cplx valued channel data.
        - (np.ndarray) Original shepp-logan phantom of size (num_col, num_lin)

    @author Jörn Huber
    """
    if num_col < 32 or num_lin < 32 or num_par < 32:
        raise ValueError("Simulated data should at least have a size of 32x32!")
    if pf_fraction_pe1 < 0.5 or pf_fraction_pe1 > 1.0 or pf_fraction_pe2 < 0.5 or pf_fraction_pe2 > 1.0:
        raise ValueError("Partial fourier fraction must lie within [0.5, 1.0]")

    sl_mag = sigpy.shepp_logan((num_col, num_lin, num_par))

    sl_phase = np.zeros((num_col, num_lin, num_par))
    center_col = num_col/2
    center_lin = num_lin/2
    center_par = num_par/2
    for i_col in range(0, num_col):
        for i_pe1 in range(0, num_lin):
            for i_pe2 in range(0, num_par):
                sl_phase[i_col, i_pe1, i_pe2] = gauss_kern_val((i_col-center_col)/num_col, (i_pe1-center_lin)/num_lin, (i_pe2-center_par)/num_par,0.3)*np.pi
    sl_ksp = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sl_mag*np.exp(sl_phase))))

    for i_pe2 in range(0, num_par):
        for i_pe1 in range(0, num_lin):
            if i_pe1 > int(pf_fraction_pe1*num_lin) or i_pe2 > int(pf_fraction_pe2*num_par):
                sl_ksp[:, i_pe1, i_pe2] = np.zeros(num_col, dtype=complex)

    return sl_ksp, sl_mag*np.exp(sl_phase)
