"""!
@brief Collection of unit tests for mrpy_parallel_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import mrpy_parallel_tools as parallel_tools
import mrpy_coil_tools as coil_tools
import mrpy_helpers as mr_helpers
import matplotlib.pyplot as plt


class TestParallelTools(unittest.TestCase):
    """!
    Unit test class for mrpy_parallel_tools
    """

    def test_fill_partial_fourier_pocs_2D(self):
        """!
        @brief UT which validates the correct functionality of fill_partial_fourier_pocs_2D
        @details We first generate artificial data with a partial fourier factor of 75%. Note that we use the data
                 with the complex valued image spaces for this purpose (that is, we introduce a phase term). Afterwards,
                 missing samples are synthesized using the pocs approach and final multi-channel data is combined using
                 an sos approach. The final corrected image is compared to the ground truth as well as the erroneous
                 image and it is verified that the difference error with the corrected image is less than before.
                 In case of the grappa corrected results, the difference error needs to be less than without correction.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_parallel_tools: fill_partial_fourier_pocs_2D")

        # Part 1: Validate functional test
        _, data_pf, _, data, image = mr_helpers.create_partial_fourier_test_data_2D(64, 64, 0.75)
        data_recon = np.zeros((64, 64, 9), dtype=complex)
        for i_channel in range(0, 9):
            data_recon[:,:,i_channel] = (
                parallel_tools.fill_partial_fourier_pocs_2D(np.squeeze(data_pf[:,:,i_channel]), 10))

        image_sos_pocs = coil_tools.sum_of_squares_combination(np.fft.ifftshift(np.fft.ifftn(data_recon, axes=(0, 1))))
        image_sos_pf = coil_tools.sum_of_squares_combination(np.fft.ifftshift(np.fft.ifftn(data_pf, axes=(0, 1))))

        fig, axs = plt.subplots(5, data_recon.shape[2])
        for i_channel in range(0, data_recon.shape[2]):
            axs[0, i_channel].imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_pf[:, :, i_channel]))))
                                     , 'gray')
            axs[0, i_channel].title.set_text('PF 6/8 Image ' + str(i_channel))
            axs[0, i_channel].axis('off')
            axs[1, i_channel].imshow(np.power(np.abs(data_pf[:, :, i_channel]), 0.2), 'gray')
            axs[1, i_channel].title.set_text('PF 6/8 kspace ' + str(i_channel))
            axs[1, i_channel].axis('off')
            axs[2, i_channel].imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_recon[:, :, i_channel]))))
                                     , 'gray')
            axs[2, i_channel].title.set_text('POCS Image ' + str(i_channel))
            axs[2, i_channel].axis('off')
            axs[3, i_channel].imshow(np.power(np.abs(data_recon[:, :, i_channel]), 0.2), 'gray')
            axs[3, i_channel].title.set_text('POCS kspace ' + str(i_channel))
            axs[3, i_channel].axis('off')
            axs[4, i_channel].imshow(np.power(np.abs(data[:, :, i_channel]), 0.2), 'gray')
            axs[4, i_channel].title.set_text('True kspace ' + str(i_channel))
            axs[4, i_channel].axis('off')

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image_sos_pf, 'gray')
        axs[0].title.set_text('Sum-Of-Squares 6/8 PF Zerofilled')
        axs[1].imshow(image_sos_pocs, 'gray')
        axs[1].title.set_text('Sum-Of-Squares 6/8 PF POCS')
        axs[2].imshow(np.abs(image), 'gray')
        axs[2].title.set_text('Ground Truth')
        plt.show()

        mean_abs_err_conjs = np.mean(np.abs((np.abs(image) - image_sos_pocs).flatten()))
        mean_abs_err_pf = np.mean(np.abs((np.abs(image) - image_sos_pf).flatten()))
        self.assertLess(mean_abs_err_conjs, mean_abs_err_pf)

        # Part 2: Validate error configs
        self.assertRaises(ValueError, parallel_tools.fill_partial_fourier_pocs_2D, np.ones((64, 64, 64)), 10)
        self.assertRaises(ValueError, parallel_tools.fill_partial_fourier_pocs_2D, np.ones((64, 64)), -1)

    def test_fill_partial_fourier_pocs_3D(self):
        """!
        @brief UT which validates the correct functionality of fill_partial_fourier_pocs_3D
        @details We first generate artificial data with a partial fourier factor of 75%. Note that we use the data
                 with the complex valued image spaces for this purpose (that is, we introduce a phase term). Afterwards,
                 missing samples are synthesized using the pocs approach and final multi-channel data is combined using
                 an sos approach. The final corrected image is compared to the ground truth as well as the erroneous
                 image and it is verified that the difference error with the corrected image is less than before.
                 In case of the grappa corrected results, the difference error needs to be less than without correction.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_parallel_tools: fill_partial_fourier_pocs_3D")

        # Part 1: Validate functional test
        sl_ksp, sl_truth = mr_helpers.create_partial_fourier_test_data_3D(64, 64, 32, 0.75, 0.75)
        data_recon = parallel_tools.fill_partial_fourier_pocs_3D(sl_ksp, 10, 3, 3)

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(np.log(np.abs(sl_ksp[:,:,16])), 'gray')
        axs[0, 0].title.set_text('Uncorrected PE1 PF')
        axs[0, 1].imshow(np.log(np.abs(data_recon[:,:,16])), 'gray')
        axs[0, 1].title.set_text('POCS PE1 PF')
        axs[1, 0].imshow(np.log(np.abs(sl_ksp[:, 32, :])), 'gray')
        axs[1, 0].title.set_text('Uncorrected PE2 PF')
        axs[1, 1].imshow(np.log(np.abs(data_recon[:, 32, :])), 'gray')
        axs[1, 1].title.set_text('POCS PE2 PF')

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sl_ksp)))[:, :, 16]), 'gray')
        axs[0, 0].title.set_text('Uncorrected PE1 PF')
        axs[0, 1].imshow(np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data_recon)))[:, :, 16]), 'gray')
        axs[0, 1].title.set_text('POCS PE1 PF')
        axs[1, 0].imshow(np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sl_ksp)))[:, 32, :]), 'gray')
        axs[1, 0].title.set_text('Uncorrected PE2 PF')
        axs[1, 1].imshow(np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data_recon)))[:, 32, :]), 'gray')
        axs[1, 1].title.set_text('POCS PE2 PF')
        plt.show()

        mean_abs_err_fft = np.mean(np.abs((np.abs(sl_truth) - np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sl_ksp))))).flatten()))
        mean_abs_err_pocs = np.mean(np.abs((np.abs(sl_truth) - np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data_recon))))).flatten()))

        self.assertLess(mean_abs_err_pocs, mean_abs_err_fft)

if __name__ == '__main__':
    unittest.main()
