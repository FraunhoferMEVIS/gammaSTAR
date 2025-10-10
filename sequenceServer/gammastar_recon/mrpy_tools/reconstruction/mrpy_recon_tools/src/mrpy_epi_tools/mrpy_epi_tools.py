"""!
@brief Collection of tools which are used for epi reconstruction tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import math
import numpy as np
import sigpy

def calc_linear_phase_correction(three_lines_ksp_phasecor):
    """!
    @brief Calculates the phase information which needs to be applied to every second readout to correct for the EPI
           phase drift.

    @param three_lines_ksp_phasecor: (np.ndarray) 3D (num_col, 3) measured k-space phase correction data.

    @return
        - (np.ndarray) Linear complex valued phase correction

    @author Jörn Huber
    """
    try:
        num_col, num_lin = three_lines_ksp_phasecor.shape
    except:
        raise ValueError("Input data must be 2D with shape (num_col, 3).")

    if num_lin != 3:
        raise ValueError("Only 3-lines reference scans are supported but detected " + str(num_lin) + "lines!")

    # Calculate product of projections
    read_up = (three_lines_ksp_phasecor[:, 0] + three_lines_ksp_phasecor[:, 2]) / 2.0
    read_down = three_lines_ksp_phasecor[:, 1]
    proj_up = np.fft.fftshift(np.fft.ifft(read_up))
    proj_down = np.fft.fftshift(np.fft.ifft(read_down))
    proj_prod = np.multiply(proj_up, np.conj(proj_down))

    # Estimate a suitable range with signal which can be used for fitting and extract the phase information
    center_val = int(num_col / 2)
    fit_range = int(num_col / 2 - 1)
    for i_range in range(0, int(num_col / 2 - 1)):
        fit_range = i_range
        perc_sig = np.sum(np.abs(proj_prod[center_val - i_range:center_val + i_range])) / np.sum(np.abs(proj_prod))
        if perc_sig > 0.7:
            break
    fit_data = np.unwrap(np.angle(proj_prod[center_val - fit_range:center_val + fit_range]))
    fit_pos = list(range(center_val - fit_range, center_val + fit_range, 1))

    # Linear fit of extracted region
    coef = np.polyfit(fit_pos, fit_data, 1)
    poly1d_fn = np.poly1d(coef)
    corr_pos = list(range(0, num_col))
    corr_val = -poly1d_fn(corr_pos)
    corr_phase_cplx = np.empty(num_col, dtype=complex)
    for pos in range(num_col):
        corr_phase_cplx[pos] = math.cos(corr_val[pos]) + 1j * math.sin(corr_val[pos])

    return corr_phase_cplx


def correct_linear_phase_drift(ksp_data, phase_corr_drift_cplx):
    """!
    @brief Applies phase correction to even/odd EPI lines based on three-lines reference scan.

    @param ksp_data: (np.ndarray) 2D (num_col, num_lin) measured k-space segment.
    @param phase_corr_drift_cplx: (np.ndarray) 1D (num_col) Calculated phase correction drift from 3 lines reference data.
    @param even: (bool) If True, apply phase correction to even lines, else to odd lines.

    @return
        - (np.ndarray) Linear drift corrected kspace segment (num_col, num_lin)

    @author Jörn Huber
    """
    try:
        num_col, num_lin = ksp_data.shape
    except:
        raise ValueError("Input data must be 2D with shape (num_col, 3).")

    ksp_lines_corrected = np.copy(ksp_data)
    for i_lin in range(0, num_lin):
        im_line = np.fft.fftshift(np.fft.ifft(ksp_data[:, i_lin]))
        im_line = np.multiply(im_line, np.conj(phase_corr_drift_cplx))
        ksp_lines_corrected[:, i_lin] = np.fft.fft(np.fft.fftshift(im_line))

    return ksp_lines_corrected
