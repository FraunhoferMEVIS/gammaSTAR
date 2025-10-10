"""!
@brief Collection of unit tests for mrpy_coil_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import mrpy_coil_tools as coil_tools
import mrpy_helpers as mr_helpers
import matplotlib.pyplot as plt


class TestCoilTools(unittest.TestCase):
    """!
    Unit test class for mrpy_coil_tools
    """

    def test_estimate_sens(self):
        """!
        @brief UT which validates the correct functionality of estimate_sens
        @details We first create a set of artificial sensitivity data. Part 1: Afterwards, we estimate the sensitivity
                 profiles from that data. We finally compare estimated sensitivites with simulated sensitivities in the
                 area of the shepp-logan phantom to make sure that the mean absolute difference is lower than 0.05.
                 Part 2: We validate the estimation of a single channel sensitivity map. Part 3: We validate the same
                 experiment as Part 1 but with a 2-fold interpolation of the sensitivity maps.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_coil_tools: estimate_sens")

        # Part 1 - Create artificial sensitivity data and estimate sensitivities
        sl_ksp_sens, sl, sens = mr_helpers.create_sensitivity_test_data_2D(64, 64)
        sens_maps = coil_tools.estimate_sens(sl_ksp_sens)
        fig, axs = plt.subplots(3, sl_ksp_sens.shape[2])
        for i_channel in range(0, sl_ksp_sens.shape[2]):
            axs[0, i_channel].imshow(
                np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(sl_ksp_sens[:, :, i_channel])))), 'gray')
            axs[0, i_channel].title.set_text('Channel Signal ' + str(i_channel))
            axs[1, i_channel].imshow(np.abs(sens_maps[:, :, i_channel]), 'gray')
            axs[1, i_channel].title.set_text('Estimated Sensitivity ' + str(i_channel))
            axs[2, i_channel].imshow(np.abs(sens[:, :, i_channel]), 'gray')
            axs[2, i_channel].title.set_text('Simulated Sensitivity ' + str(i_channel))
        plt.show()

        # Validate that the difference signal between estimated and simulated absolute sensitivities is low
        abs_diff = np.abs(sens_maps - sens)
        for i_channel in range(0, sl_ksp_sens.shape[2]):
            channel_diff = 0.0
            num_elem = 0
            for ix in range(0, sl_ksp_sens.shape[0]):
                for iy in range(0, sl_ksp_sens.shape[1]):
                    if sl[ix, iy] > 0.0:
                        channel_diff = channel_diff + abs_diff[ix, iy, i_channel]
                        num_elem = num_elem + 1
            channel_diff = channel_diff / num_elem
            self.assertLess(channel_diff, 0.05)

        # Validate error raising in case of wrong input
        wrong_dimension_data = np.ones((10, 10, 10, 10, 10, 10), dtype=complex)
        self.assertRaises(ValueError, coil_tools.estimate_sens, wrong_dimension_data)

        # Part 2: Validate single channel sensitivity estimation
        ksp_single = np.ones((64, 64), dtype=complex)
        sens_maps_single = coil_tools.estimate_sens(ksp_single)
        self.assertEqual(sens_maps_single.shape, (64, 64, 1))
        self.assertEqual((sens_maps_single == np.ones((64, 64), dtype=complex)).all(), True)

    def test_combine_channels(self):
        """!
        @brief UT which validates the correct functionality of combine_channels
        @details We first create a set of artificial sensitivity data. Afterwards, we use the adaptive combination
                 routine to combine the channel data to a final dataset. We verify the result by validating the mean
                 absolute error between the ground truth shepp logan image and the reconstructed adaptive combination.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_coil_tools: combine_channels")
        sl_ksp_sens, sl, sens = mr_helpers.create_sensitivity_test_data_2D(64, 64)
        _, im_comb = coil_tools.combine_channels(sl_ksp_sens, sens)

        fig, axs = plt.subplots(1, sl_ksp_sens.shape[2] + 1)
        for i_channel in range(0, sl_ksp_sens.shape[2]):
            axs[i_channel].imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(sl_ksp_sens[:, :, i_channel])))),
                                  'gray')
            axs[i_channel].title.set_text('Ch. ' + str(i_channel))
        axs[sl_ksp_sens.shape[2]].imshow(np.abs(im_comb), 'gray')
        axs[sl_ksp_sens.shape[2]].title.set_text('Adaptive Combination')
        plt.show()

        # Validate that the difference signal between estimated and simulated absolute images is low
        mean_error = np.mean((np.abs(im_comb) - np.abs(sl)).flatten())
        self.assertLess(mean_error, 0.01)

        # Validate error raising in case of wrong input
        wrong_dimension_data = np.ones((10, 10, 10, 10, 10, 10), dtype=complex)
        self.assertRaises(ValueError, coil_tools.combine_channels, wrong_dimension_data, sens)

    def test_sos_combination(self):
        """!
        @brief UT which validates the correct functionality of sos_combination
        @details We first create a set of artificial sensitivity data. Afterwards, we apply the sum-of-squares
                 combination to the generated data. We verify the result by comparing calculating the maximum difference
                 to the ground truth reference.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_coil_tools: sos_combination")
        sl_ksp_sens, sl, _ = mr_helpers.create_sensitivity_test_data_2D(64, 64)
        original_sos = coil_tools.sum_of_squares_combination(np.fft.ifftshift(np.fft.ifftn(sl_ksp_sens, axes=(0, 1))))
        original_sos = original_sos/np.max(np.abs(original_sos).flatten())

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(np.abs(original_sos), 'gray')
        axs[0].title.set_text('SOS Combination')
        axs[1].imshow(np.abs(sl), 'gray')
        axs[1].title.set_text('Original')
        plt.show()

        self.assertLess(np.max(np.abs(original_sos - sl).flatten()), 0.1)

if __name__ == '__main__':
    unittest.main()
