"""!
@brief Collection of unit tests for mrpy_noncart_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import math
import numpy as np
import mrpy_noncart_tools as noncart_tools
import sigpy
import matplotlib.pyplot as plt


class TestNonCartTools(unittest.TestCase):
    """!
    Unit test class for mrpy_noncart_tools
    """

    def test_prep_kaiser_bessel_kernel(self):
        """!
        @brief UT which validates the correct functionality of prep_kaiser_bessel_kernel
        @details The test creates kernel values using valid inputs and compares them to the expected kernel values.
                 In addition, invalid inputs are presented and the error case is validated.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: prep_kaiser_bessel_kernel")
        kernel_vals = noncart_tools.prep_kaiser_bessel_kernel(10000, 18.557)
        self.assertEqual(np.max((kernel_vals - np.kaiser(10000, 18.557)[int(10000 / 2):-1])), 0.0)
        self.assertRaises(ValueError, noncart_tools.prep_kaiser_bessel_kernel, 0, 18.557)
        self.assertRaises(ValueError, noncart_tools.prep_kaiser_bessel_kernel, 10000, 0)

    def test_calc_equidistant_radial_trajectory_2D(self):
        """!
        @brief UT which validates the correct functionality of calc_equidistant_radial_trajectory_2D
        @details The test creates radial trajectories using different types of valid and invalid inputs. It is checked
                 whether the correct error is risen in case of invalid input and whether some sample points of the
                 calculated trajectories correspond to the expected values.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: calc_equidistant_radial_trajectory_2D")
        self.assertRaises(ValueError, noncart_tools.calc_equidistant_radial_trajectory_2D, 1, 0)
        self.assertRaises(ValueError, noncart_tools.calc_equidistant_radial_trajectory_2D, np.array((1, 1, 1)), 0)
        self.assertRaises(ValueError, noncart_tools.calc_equidistant_radial_trajectory_2D,
                          np.array((1, 1, 1), dtype=complex), 0)
        self.assertRaises(ValueError, noncart_tools.calc_equidistant_radial_trajectory_2D,
                          np.array((1, 1)), 1 + 1j)

        traj = noncart_tools.calc_equidistant_radial_trajectory_2D(np.array((64, 1)), 0)
        self.assertEqual(np.min(traj[:, 0, 0]), 0)
        self.assertEqual(np.max(traj[:, 0, 0]), 63)

        traj = noncart_tools.calc_equidistant_radial_trajectory_2D(np.array((64, 1)), 90 / 180 * math.pi)
        self.assertEqual(np.min(traj[:, 0, 1]), 0)
        self.assertEqual(np.max(traj[:, 0, 1]), 63)

        traj = noncart_tools.calc_equidistant_radial_trajectory_2D(np.array((64, 1)), 45 / 180 * math.pi)
        self.assertLess(np.min(traj[:, 0, 0]) - 9.372583, 0.0001)
        self.assertLess(np.max(traj[:, 0, 0]) - 53.92031022, 0.0001)
        self.assertLess(np.min(traj[:, 0, 1]) - 9.372583, 0.0001)
        self.assertLess(np.max(traj[:, 0, 1]) - 53.92031022, 0.0001)

    def test_grid_data_to_matrix_2D(self):
        """!
        @brief UT which validates the correct functionality of grid_data_to_matrix_2D
        @details The test validates that correct errors are risen in case of invalid input data to the gridding
                 routine. In addition, gridded values are validated by comparison to expected values for a valid input
                 setup.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: grid_data_to_matrix_2D")
        kernel_vals = np.ones(100000)

        # We validate the gridding subroutine by gridding a set of radial "ones" and by checking that the density etc.
        # is correct afterwards.
        traj = np.ones((64, 1, 2, 2))
        for i_sample in range(0, 64):
            traj[i_sample, 0, 0, 0] = i_sample
            traj[i_sample, 0, 1, 0] = 32
            traj[i_sample, 0, 0, 1] = 32
            traj[i_sample, 0, 1, 1] = i_sample
        data = np.ones((64, 1, 2), dtype=complex)
        gridded_data_dens, _ = noncart_tools.grid_data_to_matrix_2D(data, traj, 2.0,
                                                                    4.0, kernel_vals)
        self.assertEqual(np.max(np.real(gridded_data_dens.flatten())), 1.0)
        self.assertEqual(np.min(np.real(gridded_data_dens.flatten())), 0.0)
        self.assertEqual(np.max(np.imag(gridded_data_dens.flatten())), 0.0)
        self.assertEqual(np.min(np.imag(gridded_data_dens.flatten())), 0.0)
        self.assertEqual(np.sum(gridded_data_dens[:, 64] - np.ones(128)), 0.0)
        self.assertEqual(np.sum(gridded_data_dens[64, :] - np.ones(128)), 0.0)

        self.assertRaises(ValueError, noncart_tools.grid_data_to_matrix_2D, 0, 0, 0, 0, 0)
        self.assertRaises(ValueError, noncart_tools.grid_data_to_matrix_2D, data, 0, 0, 0, 0)
        self.assertRaises(ValueError, noncart_tools.grid_data_to_matrix_2D, data, traj, 0, 0, 0)
        self.assertRaises(ValueError, noncart_tools.grid_data_to_matrix_2D, data, traj, 2.0, 0, 0)
        self.assertRaises(ValueError, noncart_tools.grid_data_to_matrix_2D, data, traj, 2.0, 4.0, 0)

    def test_get_deconvolution_matrix_2D(self):
        """!
        @brief UT which validates the correct functionality of get_deconvolution_matrix_2D
        @details Calculates the deconvolution matrix and validates that minimal and maximal values are close to
                 expected values.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: get_deconvolution_matrix_2D")
        decon_mat = noncart_tools.get_deconvolution_matrix_2D(64, 2.0, 4.0, np.kaiser(10000, 18.557)[int(10000 / 2):-1])
        self.assertEqual(decon_mat.shape, (128, 128))
        self.assertLess(np.min(decon_mat), 1.4e-07)
        self.assertLess(0.00031, np.max(decon_mat))

    def test_apply_deconvolution_2D(self):
        """!
        @brief UT which validates the correct functionality of apply_deconvolution_2D
        @details Calculates the deconvolution matrix and applies it to the same matrix to validate that minimal and
                 maximal values of the result are close to one.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: apply_deconvolution_2D")
        decon_mat = noncart_tools.get_deconvolution_matrix_2D(64, 2.0, 4.0, np.kaiser(10000, 18.557)[int(10000 / 2):-1])
        decon_mat_ksp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(decon_mat)))
        corr_im = noncart_tools.apply_deconvolution_2D(decon_mat_ksp, decon_mat, 2.0)
        corr_im = corr_im / np.max(np.abs(corr_im))
        self.assertLess(0.99999, np.min(corr_im))
        self.assertEqual(corr_im.shape, (64, 64))

    def test_prop_cut_kspace_edges_2D(self):
        """!
        @brief UT which validates the correct functionality of prop_cut_kspace_edges
        @details Cuts outer values in an array of ones. Validates the resulting by comparing edge values to zero.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: prop_cut_kspace_edges_2D")
        data = np.ones((64, 64, 8))
        noncart_tools.prop_cut_kspace_edges_2D(data)
        self.assertEqual(data[0, 0, 0], 0.0)
        self.assertEqual(data[63, 0, 0], 0.0)
        self.assertEqual(data[0, 63, 0], 0.0)
        self.assertEqual(data[63, 63, 0], 0.0)
        self.assertRaises(ValueError, noncart_tools.prop_cut_kspace_edges_2D, np.ones((64, 65, 8)))

    def test_prop_phase_correction_2D(self):
        """
        @brief Tests whether various strengths of 1. order phase variation is removed from image space by
               using PROPELLER phase correction.
        @details Creates radial test data and applies a known linear phase variation in image space to the data.
                 Afterwards, the phase variation is corrected and it is validated that the mean phase variation in
                 the area of signal is lower than before.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: prop_phase_correction_2D")
        radial_data = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sigpy.shepp_logan((32, 64), dtype=complex))))
        radial_data = np.squeeze(radial_data)
        radial_im = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(radial_data)))
        phase_factor_x = 3
        phase_factor_y = 3
        for i_col in range(0, 32):
            for i_lin in range(0, 64):
                radial_im[i_col, i_lin] = (np.abs(radial_im[i_col, i_lin])
                                           * np.exp(1j * phase_factor_x * (i_col - 32 / 2) / 32 * 2 * np.pi)
                                           * np.exp(1j * phase_factor_y * (i_lin - 64 / 2) / 64 * 2 * np.pi))
        radial_data = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(radial_im)))
        radial_data_corr = noncart_tools.prop_phase_correction_2D(radial_data)
        radial_im_corr = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(radial_data_corr)))

        fig, axs = plt.subplots(3, 2)
        fig.suptitle('Effect of Phasecorrection in PROPELLER Imaging')
        axs[0, 0].imshow(np.log(np.abs(radial_data)), 'gray')
        axs[0, 0].title.set_text('k-space without phase corr (abs)')
        axs[1, 0].imshow(np.abs(radial_im), 'gray')
        axs[1, 0].title.set_text('image without phase corr (abs)')
        axs[2, 0].imshow(np.angle(radial_im), 'gray')
        axs[2, 0].title.set_text('image without phase corr (phase)')
        axs[0, 1].imshow(np.log(np.abs(radial_data_corr)), 'gray')
        axs[0, 1].title.set_text('k-space with phase corr (abs)')
        axs[1, 1].imshow(np.abs(radial_im_corr), 'gray')
        axs[1, 1].title.set_text('image with phase corr (abs)')
        axs[2, 1].imshow(np.angle(radial_im_corr), 'gray')
        axs[2, 1].title.set_text('image with phase corr (phase)')
        plt.show()

        mean_phase_no_corr = np.mean(np.abs(np.angle(radial_im[:, 15:45]).flatten()))
        mean_phase_corr = np.mean(np.abs(np.angle(radial_im_corr[:, 15:45]).flatten()))

        self.assertLess(mean_phase_corr, mean_phase_no_corr)

    def test_prop_calc_ksp_coverage(self):
        """
        @brief Validates that the caulculated k-space coverage is close to 100% for a fully sampled propeller trajectory.
        @details We create artifical propeller trajectories with 32 bladea and a 96x32 k-space matrix each first.
                 Then, we calculate the k-space coverage and validate that it is close to 100%.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: prop_calc_ksp_coverage")
        num_col = 96
        num_lin = 32
        num_acq = 32

        traj = np.zeros((num_col, num_lin, 2, num_acq))
        center = np.array([num_col // 2, num_lin // 2])
        for i in range(num_acq):
            angle = 2 * np.pi * i / num_acq
            for x in range(num_col):
                for y in range(num_lin):
                    traj[x, y, 0, i] = (x - center[0]) * np.cos(angle) - (y - center[1]) * np.sin(angle)
                    traj[x, y, 1, i] = (x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle)
        coverage = noncart_tools.prop_calc_ksp_coverage(traj, 32)
        self.assertLess(100.0-coverage, 1.0)

    def test_calc_propeller_blade_increment_from_trajs(self):
        """
        @brief Tests whether the angle between blades is calculated correctly from a set of trajectories.
        @details Calculates the anggle between two vectors with a 90° angular difference and verifies the outcome.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: calc_propeller_blade_increment_from_trajs")

        traj_blade_1 = np.zeros((64, 2))
        traj_blade_2 = np.zeros((64, 2))
        traj_blade_1[0, 0] = 1.0
        traj_blade_2[0, 1] = 1.0

        angle = noncart_tools.calc_propeller_blade_increment_from_trajs(traj_blade_1, traj_blade_2)
        self.assertEqual(angle, 90.0)


if __name__ == '__main__':
    unittest.main()
