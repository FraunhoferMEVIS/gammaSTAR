"""!
@brief Collection of unit tests for mrpy_epi_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import math
import numpy as np
import sigpy
import mrpy_epi_tools as epi_tools
import matplotlib.pyplot as plt


class TestEPITools(unittest.TestCase):
    """!
    Unit test class for mrpy_epi_tools
    """

    def test_calc_linear_phase_correction(self):
        """!
        @brief UT which validates the correct functionality of calc_linear_phase_correction
        @details We first create an artifical shepp logan dataset. We manipulate every other k-space line of that
                 dataset to contain a small linear phase drift. Afterwards, we estimate the phase drift.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_epi_tools: calc_linear_phase_correction")

        # Create a shepp logan k-space dataset which suffers from epi specific nyquist ghosting
        num_col = 64
        num_lin = 64

        sl = sigpy.shepp_logan((num_col, num_lin), dtype=complex)
        sl_ksp = np.fft.fftshift(np.fft.fftn(sl))
        sl_ksp_up = np.zeros(sl.shape, dtype=complex)
        sl_ksp_down = np.zeros(sl.shape, dtype=complex)
        sl_ksp_up[:, 0::2] = sl_ksp[:, 0::2]
        sl_ksp_down[:, 1::2] = sl_ksp[:, 1::2]
        sl_phasecor_up = sl_ksp_up[:, int(num_col/2)]
        sl_phasecor_down = sl_ksp_up[:, int(num_lin/2)]

        # Apply epi-like linear phase drifts
        phase_drift = np.empty(sl.shape[0], dtype=complex)
        for pos in range(num_col):
            arg = -math.pi*(pos-sl.shape[0]/2)/sl.shape[0]
            phase_drift[pos] = math.cos(arg) + 1j*math.sin(arg)
        for i_line in range(0, num_lin):
            proj_line = np.fft.fftshift(np.fft.ifft(sl_ksp_down[:, i_line]))
            proj_line = np.multiply(proj_line, phase_drift)
            sl_ksp_down[:, i_line] = np.fft.fft(np.fft.fftshift(proj_line))

            proj_line = np.fft.fftshift(np.fft.ifft(sl_ksp_up[:, i_line]))
            sl_ksp_up[:, i_line] = np.fft.fft(np.fft.fftshift(proj_line))

        sl_phasecor_down = np.fft.fftshift(np.fft.ifft(sl_phasecor_down))
        sl_phasecor_down = np.multiply(sl_phasecor_down, phase_drift)
        sl_phasecor_down = np.fft.fft(np.fft.fftshift(sl_phasecor_down))

        sl_phasecor_up = np.fft.fftshift(np.fft.ifft(sl_phasecor_up))
        sl_phasecor_up = np.fft.fft(np.fft.fftshift(sl_phasecor_up))

        # Combine to final data
        data_sl = np.zeros((num_col, num_lin, 2), dtype=complex)  # 2 due to up/down
        data_sl[:, :, 0] = sl_ksp_up
        data_sl[:, :, 1] = sl_ksp_down
        data_sl_phasecor = np.zeros((num_col, 3), dtype=complex)
        data_sl_phasecor[:, 0] = sl_phasecor_up
        data_sl_phasecor[:, 1] = sl_phasecor_down
        data_sl_phasecor[:, 2] = sl_phasecor_up
        epi_drift_corr = epi_tools.calc_linear_phase_correction(data_sl_phasecor)

        self.assertLess(np.mean(np.abs(np.angle(epi_drift_corr)-np.angle(phase_drift))), 0.1)

    def test_correct_linear_phase_drift(self):
        """!
        @brief UT which validates the correct functionality of correct_linear_phase_drift
        @details We first create an artifical shepp logan dataset. We manipulate every other k-space line of that
                 dataset to contain a small linear phase drift. Afterwards, we correct that linear drift. Finally, we
                 validate whether the amount of ghosting signal is lower after the correction routine.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_epi_tools: correct_linear_phase_drift")

        # Create a shepp logan k-space dataset which suffers from epi specific nyquist ghosting
        num_col = 64
        num_lin = 64

        sl = sigpy.shepp_logan((num_col, num_lin), dtype=complex)
        sl_ksp = np.fft.fftshift(np.fft.fftn(sl))
        sl_ksp_up = np.zeros(sl.shape, dtype=complex)
        sl_ksp_down = np.zeros(sl.shape, dtype=complex)
        sl_ksp_up[:, 0::2] = sl_ksp[:, 0::2]
        sl_ksp_down[:, 1::2] = sl_ksp[:, 1::2]
        sl_phasecor_up = sl_ksp_up[:, int(num_col/2)]
        sl_phasecor_down = sl_ksp_up[:, int(num_lin/2)]

        # Apply epi-like linear phase drifts
        phase_drift = np.empty(sl.shape[0], dtype=complex)
        for pos in range(num_col):
            arg = -math.pi*(pos-sl.shape[0]/2)/sl.shape[0]
            phase_drift[pos] = math.cos(arg) + 1j*math.sin(arg)
        for i_line in range(0, num_lin):
            proj_line = np.fft.fftshift(np.fft.ifft(sl_ksp_down[:, i_line]))
            proj_line = np.multiply(proj_line, phase_drift)
            sl_ksp_down[:, i_line] = np.fft.fft(np.fft.fftshift(proj_line))

            proj_line = np.fft.fftshift(np.fft.ifft(sl_ksp_up[:, i_line]))
            sl_ksp_up[:, i_line] = np.fft.fft(np.fft.fftshift(proj_line))

        sl_phasecor_down = np.fft.fftshift(np.fft.ifft(sl_phasecor_down))
        sl_phasecor_down = np.multiply(sl_phasecor_down, phase_drift)
        sl_phasecor_down = np.fft.fft(np.fft.fftshift(sl_phasecor_down))

        sl_phasecor_up = np.fft.fftshift(np.fft.ifft(sl_phasecor_up))
        sl_phasecor_up = np.fft.fft(np.fft.fftshift(sl_phasecor_up))

        # Combine to final data
        data_sl = np.zeros((num_col, num_lin, 2), dtype=complex)  # 2 due to up/down
        data_sl[:, :, 0] = sl_ksp_up
        data_sl[:, :, 1] = sl_ksp_down
        data_sl_phasecor = np.zeros((num_col, 3), dtype=complex)
        data_sl_phasecor[:, 0] = sl_phasecor_up
        data_sl_phasecor[:, 1] = sl_phasecor_down
        data_sl_phasecor[:, 2] = sl_phasecor_up

        epi_drift_corr = epi_tools.calc_linear_phase_correction(data_sl_phasecor)
        data_sl_cor = epi_tools.correct_linear_phase_drift(data_sl[:, :, 1], epi_drift_corr)

        im_sl_corr = np.fft.fftn(np.squeeze(data_sl[:, :, 0]) + np.squeeze(data_sl_cor))
        im_sl_uncorr = np.fft.fftn(np.squeeze(data_sl[:, :, 0] + data_sl[:, :, 1]))

        # Plot results
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(np.log(np.abs(data_sl[:, :, 0] + data_sl[:, :, 1])), cmap='gray')
        axs[0, 0].title.set_text('Uncorrected k-space')
        axs[0, 1].imshow(np.abs(im_sl_uncorr), cmap='gray')
        axs[0, 1].title.set_text('Uncorrected image space')
        axs[1, 0].imshow(np.log(np.abs(data_sl[:, :, 0] + data_sl_cor)), cmap='gray')
        axs[1, 0].title.set_text('Corrected k-space')
        axs[1, 1].imshow(np.abs(im_sl_corr), cmap='gray')
        axs[1, 1].title.set_text('Corrected image space')
        plt.show()

        # We measure the energy of the nyquist ghosts in uncorrected and corrected data. In the corrected case,
        # the energy should be less
        sig_ghost_uncorr = np.sum(np.sum(np.abs(im_sl_uncorr[:, 0:5]),1),0)
        sig_ghost_corr = np.sum(np.sum(np.abs(im_sl_corr[:, 0:5]), 1), 0)
        self.assertLess(sig_ghost_corr, sig_ghost_uncorr)


if __name__ == '__main__':
    unittest.main()
