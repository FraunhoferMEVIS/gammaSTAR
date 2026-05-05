"""!
@brief Collection of unit tests for mrpy_helpers.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import sigpy
import mrpy_helpers
import mrpy_helpers as mr_helpers
from typing import Tuple
import matplotlib.pyplot as plt


def generate_test_data(
        shape: Tuple,
        k_axes: Tuple
) -> np.ndarray:
    """!
    @brief Create artificial Shepp-Logan k-space dataset with coil sensitivities.

    @param shape: Shape of the test data
    @param k_axes: Axes corresponding to the frequency (k-space) dimensions.

    @return
        - k-space data

    @author Jörn Huber, Tom Lütjen
    """

    # Create phantom
    phantom_shape = [shape[k] for k in k_axes]
    if shape[k_axes[-1]] == 1:
        phantom = sigpy.shepp_logan(phantom_shape[:2], dtype=complex)
        phantom = np.expand_dims(phantom, axis=-1)
    else:
        phantom = sigpy.shepp_logan(phantom_shape, dtype=complex)

    # Expand phantom to shape
    im = np.expand_dims(
        phantom, axis=[ax for ax in range(len(shape)) if ax not in k_axes]
    )
    im = np.broadcast_to(im, shape)

    # Generate k-space data
    ksp = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(im, axes=k_axes), axes=k_axes),
        axes=k_axes,
    )

    return ksp


class TestHelpers(unittest.TestCase):
    """!
    Unit test class for mrpy_helpers
    """

    def test_remove_os(self):
        """!
        @brief UT which validates the correct functionality of remove_os
        @details The test verifies that oversampling is removed in a test dataset. Therefore, test data is first created
                 with doubled FoV readout sizes. Afterwards, the FoV is halfed and it is validated that the resulting
                 image data corresponds to the original data with the correct FoV. Criteria: Maximum difference between
                 simulated and processed images shall be below 0.005.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_helpers: remove_os")
        test_image_true_fov = sigpy.shepp_logan((64, 64, 8), dtype=complex)

        test_image_ro_os = np.zeros((128, 64, 8), dtype=complex)
        test_image_ro_os[32:96, :, :] = test_image_true_fov
        test_ksp_ro_os = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(test_image_ro_os), axes=(0, 1, 2)))

        test_ksp_corrected = mr_helpers.remove_os(test_ksp_ro_os, 2, (0,))
        test_im_corrected = np.fft.fftshift(np.fft.ifftn(test_ksp_corrected, axes=(0, 1, 2)))

        self.assertLess(np.max(np.abs(test_im_corrected)-np.abs(test_image_true_fov)), 0.005)


    def test_filter_low_order_phase(self, debug_plot:bool = True):
        """
        @brief UT which validates the correct functionality of filter_low_order_phase
        @details Part 1: Create a 3D square filter. Criteria: Validate that resulting filter has the correct dimensions
                 as well as values along all non k-space axes.
                 Part 2: Create a 2D square filter. Criteria: Validate that resulting filter has the correct dimensions
                 as well as values along all non k-space axes.
                 Part 3: Create a 2D square filter and filter k-space data with artificial linear phase information.
                 Criteria: Mean phase in filtered image needs to be less than in unfiltered image.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_helpers: filter_low_order_phase")

        # Part 1: Create a 3D square filter. Criteria: Validate that resulting filter has the correct dimensions as
        #         well as values along all non k-space axes.
        _, triang_filter = mrpy_helpers.filter_low_order_phase(
            ksp_data = np.zeros((64, 32, 24, 2, 3)),
            k_axes = (0, 1, 2)
        )

        self.assertEqual(triang_filter.shape, (64, 32, 24, 2, 3))
        self.assertAlmostEqual(np.max(triang_filter), 1.0, 2)

        triang_filter = np.reshape(triang_filter, [64, 32, 24, -1])

        for i_filt in range(triang_filter.shape[3]):
            self.assertAlmostEqual(triang_filter[0, 0, 0, i_filt], 0.00, 1)
            self.assertAlmostEqual(triang_filter[32, 0, 0, i_filt], 0.00, 1)
            self.assertAlmostEqual(triang_filter[63, 0, 0, i_filt], 0.00, 1)
            self.assertAlmostEqual(triang_filter[32, 16, 12, i_filt], 1.00, 1)
            self.assertAlmostEqual(triang_filter[63, 23, 11, i_filt], 0.00, 1)

        # Part 2: Create a 2D square filter. Criteria: Validate that resulting filter has the correct dimensions as well
        #         as values along all non k-space axes.
        _, triang_filter = mrpy_helpers.filter_low_order_phase(
            ksp_data=np.zeros((64, 32, 32, 2, 3)),
            k_axes=(0, 1)
        )

        self.assertEqual(triang_filter.shape, (64, 32, 32, 2, 3))
        self.assertAlmostEqual(np.max(triang_filter), 1.0, 2)

        triang_filter = np.reshape(triang_filter, [64, 32, -1])

        for i_filt in range(triang_filter.shape[2]):
            self.assertAlmostEqual(triang_filter[0, 0, i_filt], 0.00, 1)
            self.assertAlmostEqual(triang_filter[32, 0, i_filt], 0.05, 1)
            self.assertAlmostEqual(triang_filter[63, 0, i_filt], 0.00, 1)
            self.assertAlmostEqual(triang_filter[32, 16, i_filt], 1.00, 1)
            self.assertAlmostEqual(triang_filter[63, 23, i_filt], 0.00, 1)

        # Part 3: Create a 2D square filter and filter k-space data with artificial linear phase information. Criteria:
        #         Mean phase in filtered image needs to be less than in unfiltered image.
        data = generate_test_data(
            (64, 32, 24),
            k_axes=(0, 1, 2),
        )

        im = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(data, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=(0, 1, 2)
        )

        phase_factor_x = 3
        phase_factor_y = 3
        for i_par in range(0, 24):
            for i_col in range(0, 64):
                for i_lin in range(0, 32):
                    im[i_col, i_lin, i_par] = (np.abs(im[i_col, i_lin, i_par])
                                               * np.exp(1j * phase_factor_x * (i_col - 64 / 2) / 64 * 2 * np.pi)
                                               * np.exp(1j * phase_factor_y * (i_lin - 32 / 2) / 32 * 2 * np.pi))

        ksp = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(im, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
        )

        data_filt, _ = mrpy_helpers.filter_low_order_phase(
            ksp_data=ksp,
            k_axes=(0, 1)
        )

        im_filt = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(data_filt, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
        )

        if debug_plot:
            fig, axs = plt.subplots(3, 2)
            fig.suptitle('Effect of Phasecorrection in PROPELLER Imaging')
            axs[0, 0].imshow(np.log(np.abs(ksp[:,:,12])), 'gray')
            axs[0, 0].title.set_text('k-space without phase corr (abs)')
            axs[1, 0].imshow(np.abs(im[:,:,12]), 'gray')
            axs[1, 0].title.set_text('image without phase corr (abs)')
            axs[2, 0].imshow(np.angle(im[:,:,12]), 'gray')
            axs[2, 0].title.set_text('image without phase corr (phase)')
            axs[0, 1].imshow(np.log(np.abs(data_filt[:,:,12])), 'gray')
            axs[0, 1].title.set_text('k-space with phase corr (abs)')
            axs[1, 1].imshow(np.abs(im_filt[:,:,12]), 'gray')
            axs[1, 1].title.set_text('image with phase corr (abs)')
            axs[2, 1].imshow(np.angle(im_filt[:,:,12]), 'gray')
            axs[2, 1].title.set_text('image with phase corr (phase)')
            plt.show()

        mean_phase_im = np.mean(np.abs(np.angle(im[:, 8:24,:]).flatten()))
        mean_phase_im_filt = np.mean(np.abs(np.angle(im_filt[:, 8:24,:]).flatten()))

        self.assertLess(mean_phase_im_filt, mean_phase_im)


if __name__ == '__main__':
    unittest.main()
