"""!
@brief Collection of unit tests for mrpy_metrics.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import sigpy
import mrpy_helpers as mr_helpers

class TestHelpers(unittest.TestCase):
    """!
    Unit test class for mrpy_metrics
    """

    def test_remove_readout_os(self):
        """!
        @brief UT which validates the correct functionality of remove_readout_os
        @details The test verifies that oversampling is removed in a test dataset. Therefore, test data is first created
                 with doubled FoV readout sizes. Afterwards, the FoV is halfed and it is validated that the resulting
                 image data corresponds to the original data with the correct FoV. Note: we currently only compare
                 the absolute values as there might be a problem with the phase information which needs to be
                 investigated.

        @param self: Reference to object

        @author Jörn Huber
        """
        test_image_true_fov = sigpy.shepp_logan((64, 64, 8), dtype=complex)

        test_image_ro_os = np.zeros((128, 64, 8), dtype=complex)
        test_image_ro_os[32:96, :, :] = test_image_true_fov
        test_ksp_ro_os = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(test_image_ro_os), axes=(0, 1, 2)))

        test_ksp_corrected = mr_helpers.remove_readout_os(test_ksp_ro_os, 0, 2)
        test_im_corrected = np.fft.fftshift(np.fft.ifftn(test_ksp_corrected, axes=(0, 1, 2)))

        self.assertLess(np.max(np.abs(test_im_corrected)-np.abs(test_image_true_fov)), 0.005)

    def test_create_artificial_partial_fourier_data(self):
        """!
        @brief UT which validates the correct functionality of create_artificial_parallel_data
        @details The test verifies that correct artifical data is created for 2D and 3D cases and that all error
                 conditions are evoked at least once.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_parallel_tools: create_artificial_partial_fourier_data")

        # Part 1: Validate 2D phantom data
        ksp_r_pf, ksp_cp_pf, ksp_r, ksp_cp, sl = mr_helpers.create_partial_fourier_test_data_2D(64, 64, 0.5)
        self.assertEqual(ksp_r_pf.shape, (64, 64, 9))
        self.assertEqual(ksp_cp_pf.shape, (64, 64, 9))
        self.assertEqual(ksp_r.shape, (64, 64, 9))
        self.assertEqual(ksp_cp.shape, (64, 64, 9))
        self.assertEqual(sl.shape, (64, 64))
        for i_lin in range(33, 64):
            self.assertEqual(np.sum(np.squeeze(ksp_r_pf[:,i_lin])), 0.0)
            self.assertEqual(np.sum(np.squeeze(ksp_cp_pf[:, i_lin])), 0.0 + 1j*0.0)

        # Part 2: Validate error configs
        self.assertRaises(ValueError, mr_helpers.create_partial_fourier_test_data_2D, 28, 28, 0.5)
        self.assertRaises(ValueError, mr_helpers.create_partial_fourier_test_data_2D, 64, 64, 0.4)

if __name__ == '__main__':
    unittest.main()
