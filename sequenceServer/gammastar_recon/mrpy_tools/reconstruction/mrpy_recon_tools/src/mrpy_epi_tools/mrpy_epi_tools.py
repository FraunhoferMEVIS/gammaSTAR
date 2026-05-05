"""!
@brief Collection of tools which are used for epi reconstruction tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np


def calculate_phase_shifts(
        phasecor_data: np.ndarray,
        phasecor_data_reverse: np.ndarray = None,
        ro_axis: int = 0,
        rep_axis:int|None = None,
        dir_axis:int = 10
) -> np.ndarray:
    """!
    @brief Calculate EPI phase shifts.
    @details Ahn CB, Cho ZH. A new phase correction method in NMR imaging based on autocorrelation and histogram
             analysis. IEEE Trans Med Imaging. 1987;6(1):32-6. doi: 10.1109/TMI.1987.4307795. PMID: 18230424.

    @param phasecor_data: Phase correction data, consisting of three lines (num_col, num_cha, Na, Nb, Nc....) acquired
                          with the same EPI sequence as the actual measurement.
    @param phasecor_data_reverse: Reverse phase correction data, consisting of three lines (num_col, num_cha, Na, Nb, Nc....)
                                  acquired with the same EPI sequence as the actual measurement. This array is optional
                                  because in some cases, phase cor data is not stored in a single array but in seperate
                                  arrays with different flags.
    @param ro_axis: Axis along which the readout direction is defined.
    @param rep_axis: Axis corresponding to the repetition of the reference lines in the phasecor. If reverse array is
                     provided, this entry is ignored.
    @param dir_axis: Axis corresponding to the direction of the reference lines in phasecor.

    @return
        - Complex phase shift array.

    @author Jörn Huber, Tom Lütjen
    """

    if phasecor_data_reverse is not None:

        if phasecor_data_reverse.shape[dir_axis] == 2:

            # Average two lines with the same direction to account for relaxation
            line_up_0 = np.expand_dims(np.take(phasecor_data_reverse, 0, dir_axis), dir_axis)
            line_up_1 = np.expand_dims(np.take(phasecor_data_reverse, 1, dir_axis), dir_axis)
            line_up = (line_up_0 + line_up_1) / 2

            line_down = np.expand_dims(np.take(phasecor_data, 0, dir_axis), dir_axis)

        elif phasecor_data.shape[dir_axis] == 2:

            # Average two lines with the same direction to account for relaxation
            line_up_0 = np.expand_dims(np.take(phasecor_data, 0, dir_axis), dir_axis)
            line_up_1 = np.expand_dims(np.take(phasecor_data, 1, dir_axis), dir_axis)
            line_up = (line_up_0 + line_up_1) / 2

            line_down = np.expand_dims(np.take(phasecor_data_reverse, 0, dir_axis), dir_axis)

        else:

            raise ValueError("Neither the phasecor_data array not the phasecor_data_reverse array have 2 elements"
                             "in the dir_axis dimension.")

    else:

        # Average two lines with the same direction to account for relaxation
        idx = [slice(None)] * phasecor_data.ndim
        idx[rep_axis] = 0
        idx[dir_axis] = 1
        line_up_0 = np.expand_dims(phasecor_data[tuple(idx)], axis=(rep_axis, dir_axis))

        idx = [slice(None)] * phasecor_data.ndim
        idx[rep_axis] = 0
        idx[dir_axis] = 0
        line_down = np.expand_dims(phasecor_data[tuple(idx)], axis=(rep_axis, dir_axis))

        idx = [slice(None)] * phasecor_data.ndim
        idx[rep_axis] = 1
        idx[dir_axis] = 1
        line_up_1 = np.expand_dims(phasecor_data[tuple(idx)], axis=(rep_axis, dir_axis))
        line_up = (line_up_0 + line_up_1) / 2

    # Compute projections
    proj_down = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(line_down, axes=ro_axis), axis=ro_axis),
        axes=ro_axis,
    )
    proj_up = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(line_up, axes=ro_axis), axis=ro_axis), axes=ro_axis
    )

    # Extract phase shifts
    proj_prod = proj_up * np.conj(proj_down)

    # Linear fitting (we cannot exploit broadcasting here, but we flatten to avoid nested loops)
    proj_prod = np.moveaxis(proj_prod, ro_axis, -1)
    proj_prod_shape = proj_prod.shape
    proj_prod = proj_prod.reshape(-1, proj_prod.shape[-1])
    phase_shifts = np.zeros_like(proj_prod)

    for batch_idx in range(proj_prod.shape[0]):

        cur_proj_prod = proj_prod[batch_idx, :]
        center_val = int(phasecor_data.shape[ro_axis] / 2)

        low_pix = 0
        high_pix = phasecor_data.shape[ro_axis] - 1

        for idx in range(int(phasecor_data.shape[ro_axis] / 2)):

            low_pix = center_val - idx
            high_pix = center_val + idx

            if low_pix < 0 or high_pix >= phasecor_data.shape[ro_axis] - 1:
                break

            perc_sig = np.sum(np.abs(cur_proj_prod[low_pix:high_pix])) / np.sum(np.abs(cur_proj_prod))
            if perc_sig > 0.7:
                break

        fit_data = np.unwrap(np.angle(cur_proj_prod[low_pix:high_pix]))
        fit_pos = np.arange(low_pix, high_pix)

        coef = np.polyfit(fit_pos, fit_data, 1)
        poly1d_fn = np.poly1d(coef)
        shift_pos = np.arange(phasecor_data.shape[ro_axis])
        shift_val = -poly1d_fn(shift_pos)

        # Apply Fourier shift theorem (shift_val is already in radians), we need the conjugate because the method
        # above calculates the "correcting" phase shift.
        phase_shifts[batch_idx, :] = np.exp(-1j * shift_val).conj()

    phase_shifts = phase_shifts.reshape(proj_prod_shape)
    phase_shifts = np.moveaxis(phase_shifts, -1, ro_axis)

    return phase_shifts


def remove_phase_shifts(data:np.ndarray, phase_shifts:np.ndarray, ro_axis:int = 0):
    """!
    @brief Remove EPI phase shifts from k-space data.

    @param data: k-space data.
    @param phase_shifts: Phase shifts calculated from the phase correction data.
    @param ro_axis: Axis corresponding to the frequency encoding (k-space) dimension (Must match in data and phasecor).

    @return
        - k-space data with removed EPI phase shifts.

    @author Jörn Huber, Tom Lütjen
    """

    # Project data
    proj_reco = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(data, axes=ro_axis), axis=ro_axis), axes=ro_axis
    )

    # We need the conjugate because the method above calculates the true phase shift
    proj_reco *= np.conj(phase_shifts)
    data = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(proj_reco, axes=ro_axis), axis=ro_axis),
        axes=ro_axis,
    )

    return data
