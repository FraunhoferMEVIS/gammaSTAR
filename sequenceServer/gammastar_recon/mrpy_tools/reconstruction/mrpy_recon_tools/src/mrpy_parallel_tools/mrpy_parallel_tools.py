"""!
@brief Collection of tools which are used for parallel imaging reconstruction tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np

def fill_partial_fourier_pocs_2D(pf_data: np.ndarray, max_num_iter: int) -> np.ndarray:
    """!
    @brief Reconstruct missing lines in partial fourier acquisitions using the pocs approach.
    @details In contrast to the simple conjgate symmetry algorithm, pocs is able to estimate the phase of the object,
             which means that it does not rely on real valued image data which is reality is also never the case. For a
             detailed describtion of the algorithm check "A fast, iterative, partial-fourier technique capable of local
             phase recovery" (1991) by Haacke et al.

    @param pf_data: (np.ndarray) Fractionally sampled complex multi-channel k-space data (num_col, num_lin)
    @param max_num_iter: (int) Maximum number of desired iterations of iterative part

    @return
        - (np.ndarray) Reconstructed k-space data of size (num_col, num_lin)

    @author Jörn Huber
    """
    if len(pf_data.shape) != 2:
        raise ValueError("Partial fourier data must have dimensions (num_col, num_lin)!")
    if max_num_iter < 1:
        raise ValueError("Maximum number of iterations must not be less than 1")

    num_ro, num_pe = pf_data.shape

    # We estimate the area around the central k-space line which can be used for phase operations (m-points)
    _, center_pe = np.unravel_index(np.abs(pf_data).argmax(), pf_data.shape)
    m = 0
    for i_pe in range(center_pe + 1, num_pe):
        if np.abs(pf_data[:, i_pe]).any() != 0.0:
            m = m + 1

    # We estimate the phase information from the central part first
    pf_data_central = np.copy(pf_data)
    pf_data_central[:, 0:center_pe - m] = np.zeros((num_ro, center_pe - m), dtype=complex)
    rho = np.angle(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(pf_data_central))))

    # Iterative part, updating magnitude and phase estimates in image space
    s = np.copy(pf_data)
    s_snake = np.zeros((num_ro, num_pe), dtype=complex)
    iter_err = []
    for i_iter in range(0, max_num_iter):

        # Estimate new k-space guess
        p_j = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(s))))
        p_j_central = np.copy(s)
        p_j_central[:, 0:center_pe - m] = np.zeros((num_ro, center_pe - m), dtype=complex)
        p_j_central[:, center_pe + m:num_pe] = np.zeros((num_ro, num_pe-(center_pe + m)), dtype=complex)
        rho_j = np.angle(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(p_j_central))))
        p_j_snake = p_j*np.cos(rho-rho_j)*np.exp(1j*rho)
        s_j = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(p_j_snake)))

        # Linear window combination to avoid discontinuities in k-space
        q = 2
        s_snake[:,0:center_pe+m-q] = s[:,0:center_pe+m-q]
        lin_filter_left = np.arange(0, 1+1/float(2*q), 1/float(2*q))
        lin_filter_right = np.arange(1, 0-1 / float(2 * q), -1 / float(2 * q))
        line_offset = center_pe + m
        filter_index = 0
        for i_q in range(line_offset-q, line_offset+q):
            s_snake[:, i_q] = lin_filter_left[filter_index]*s[:,i_q] + lin_filter_right[filter_index]*s_j[:, i_q]
            filter_index = filter_index + 1
        s_snake[:, center_pe + m + q:num_pe] = s_j[:, center_pe + m + q:num_pe]

        # Measure the change between s_snake and s and see if convergence is reached
        im_prev = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(s)))
        im_update = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(s_snake)))
        iter_err.append( np.sum(np.abs(im_update - im_prev).flatten()) )
        s = np.copy(s_snake)

        if iter_err[i_iter]/iter_err[0] < 0.05:
            break

    return s


def fill_partial_fourier_pocs_3D(
    pf_data: np.ndarray,
    max_num_iter: int,
    q_pe1: int,
    q_pe2: int
) -> np.ndarray:
    """!
    @brief Reconstruct missing lines in 3D partial fourier acquisitions using the pocs approach.
    @details In contrast to the simple conjgate symmetry algorithm, pocs is able to estimate the phase of the object,
             which means that it does not rely on real valued image data which is reality is also never the case. For a
             detailed describtion of the algorithm check "A fast, iterative, partial-fourier technique capable of local
             phase recovery" (1991) by Haacke et al.

    @param pf_data: (np.ndarray) Fractionally sampled complex multi-channel k-space data (num_col, num_lin)
    @param max_num_iter: (int) Maximum number of desired iterations of iterative part
    @param q_pe1: (int) Transition zone between measured data and estimated samples in pe1 direction. A smooth
                        transition in this area will be aimed for.
    @param q_pe2: (int) Transition zone between measured data and estimated samples in pe2 direction. A smooth
                        transition in this area will be aimed for.

    @return
        - (np.ndarray) Reconstructed k-space data of size (num_col, num_pe1, num_pe2)

    @author Jörn Huber
    """

    if len(pf_data.shape) != 3:
        raise ValueError("Partial fourier data must have dimensions (num_col, num_lin, num_par)!")
    if max_num_iter < 1:
        raise ValueError("Maximum number of iterations must not be less than 1")

    num_ro, num_pe1, num_pe2 = pf_data.shape

    # We estimate the area around the central k-space line which can be used for phase operations (m-points)
    _, center_pe1, center_pe2 = np.unravel_index(np.abs(pf_data).argmax(), pf_data.shape)
    m_center_offset_pe1 = 0
    for i_pe1 in range(center_pe1 + 1, num_pe1):
        if np.abs(pf_data[:, i_pe1, :]).any() != 0.0:
            m_center_offset_pe1 = m_center_offset_pe1 + 1

    m_center_offset_pe2 = 0
    for i_pe2 in range(center_pe2 + 1, num_pe2):
        if np.abs(pf_data[:, :, i_pe2]).any() != 0.0:
            m_center_offset_pe2 = m_center_offset_pe2 + 1

    b_is_pf_pe1 = False
    b_is_pf_pe2 = False
    if m_center_offset_pe1 + center_pe1 + 1 < num_pe1:
        b_is_pf_pe1 = True
    if m_center_offset_pe2 + center_pe2 + 1 < num_pe2:
        b_is_pf_pe2 = True

    # Early return: Nothing to do
    if not b_is_pf_pe1 and not b_is_pf_pe2:
        return pf_data

    # Calculate the phase from the central part of the sampled data
    p_1_central = np.copy(pf_data)

    # PE1
    if b_is_pf_pe1:
        p_1_central[:, 0:center_pe1 - m_center_offset_pe1, :] = 0
        p_1_central[:, center_pe1 + m_center_offset_pe1 - q_pe1:num_pe1, :] = 0

    # PE2
    if b_is_pf_pe2:
        p_1_central[:, :, 0:center_pe2 - (m_center_offset_pe2 - q_pe2)] = 0
        p_1_central[:, :, center_pe2 + m_center_offset_pe2 - q_pe2:num_pe2] = 0

    rho = np.angle(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(p_1_central))))

    # Iterative part, updating magnitude and phase estimates in image space
    sj_snake = np.zeros((num_ro, num_pe1, num_pe2), dtype=complex)
    s = pf_data.copy()

    # We remove the transition band from s
    if b_is_pf_pe1:
        s[:, center_pe1 + m_center_offset_pe1 - q_pe1 + 1:, :] = 0
    if b_is_pf_pe2:
        s[:, :, center_pe2 + m_center_offset_pe2 - q_pe2 + 1:] = 0

    for i_iter in range(max_num_iter):

        p_jm = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(s)))

        rho_j = np.angle(p_jm)
        pj_snake = np.abs(p_jm) * (np.cos(rho - rho_j) if i_iter > 0 else 1) * np.exp(1j * rho)

        sj = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(pj_snake)))

        sj_snake[:] = s
        if b_is_pf_pe1:
            sj_snake[:, center_pe1 + m_center_offset_pe1 - q_pe1 + 1:num_pe1, :] = sj[:,
                                                                                      center_pe1 + m_center_offset_pe1 - q_pe1 + 1:num_pe1,
                                                                                      :]
        if b_is_pf_pe2:
            sj_snake[:, :, center_pe2 + m_center_offset_pe2 - q_pe2 + 1:num_pe2] = sj[:,
                                                                                      :,
                                                                                      center_pe2 + m_center_offset_pe2 - q_pe2 + 1:num_pe2]

        s = sj_snake.copy()

    # The new k-space guess probably has a sharp boundary between measured and estimated samples. We get rid of that
    # by smoothing respecitve areas in k-space.

    # Measured data before transition zone
    s_final = np.copy(pf_data)
    if b_is_pf_pe1:
        s_final[:, center_pe1 + m_center_offset_pe1 - q_pe1 + 1:, :] = 0
    if b_is_pf_pe2:
        s_final[:, :, center_pe2 + m_center_offset_pe2 - q_pe2 + 1:] = 0

    # Transition zone
    if b_is_pf_pe1:
        trans_start_pe1 = center_pe1 + m_center_offset_pe1 - (q_pe1 - 1)
        trans_end_pe1 = center_pe1 + m_center_offset_pe1
        filter_left = np.cos(np.pi * np.arange(q_pe1) / (2 * (q_pe1 - 1)))
        filter_right = 1 - filter_left
        for i_pe1 in range(trans_start_pe1, trans_end_pe1 + 1):
            i_filt = i_pe1 - trans_start_pe1
            s_final[:, i_pe1, :] = filter_left[i_filt] * pf_data[:, i_pe1, :] + filter_right[i_filt] * s[:, i_pe1, :]

    if b_is_pf_pe2:
        trans_start_pe2 = center_pe2 + m_center_offset_pe2 - (q_pe2 - 1)
        trans_end_pe2 = center_pe2 + m_center_offset_pe2
        filter_left = np.cos(np.pi * np.arange(q_pe2) / (2 * (q_pe2 - 1)))
        filter_right = 1 - filter_left
        for i_pe2 in range(trans_start_pe2, trans_end_pe2 + 1):
            i_filt = i_pe2 - trans_start_pe2
            s_final[:, :, i_pe2] = filter_left[i_filt] * pf_data[:, :, i_pe2] + filter_right[i_filt] * s[:, :, i_pe2]

    # Estimated data after transition zone
    if b_is_pf_pe1:
        s_final[:, center_pe1 + m_center_offset_pe1 + 1:num_pe1, :] = s[:, center_pe1 + m_center_offset_pe1 + 1:num_pe1, :]
    if b_is_pf_pe2:
        s_final[:, :, center_pe2 + m_center_offset_pe2 + 1:num_pe2] = s[:, :, center_pe2 + m_center_offset_pe2 + 1:num_pe2]

    return s_final
