"""!
@brief Collection of unit tests for mrpy_epi_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import sigpy
import mrpy_epi_tools as epi_tools
import matplotlib.pyplot as plt


def generate_test_data(shift, shape, k_axes, ro_axis=0, pe_axis=2, rep_axis=5, dir_axis=10):
    """!
    @brief Create artificial Shepp-Logan k-space dataset with Nyquist-ghosting.

    @param shape: (tuple) Shape of the test data
    @param k_axes: (tuple) Axes corresponding to the frequency (k-space) dimensions.

    @return
        - (np.ndarray) k-space data
        - (np.ndarray) image data
        - (np.ndarray) Coil sensitivities

    @author Jörn Huber, Tom Lütjen
    """

    # Create phantom
    phantom_shape = [shape[k] for k in k_axes]
    if shape[k_axes[-1]] == 1:
        phantom = sigpy.shepp_logan(phantom_shape[:2], dtype=complex)
        phantom = np.expand_dims(phantom, axis=-1)
    else:
        phantom = sigpy.shepp_logan(phantom_shape, dtype=complex)

    # Expand phamtom to shape
    im = np.expand_dims(
        phantom, axis=[ax for ax in range(len(shape)) if ax not in k_axes]
    )
    im = np.broadcast_to(im, shape)

    # Simulate k-space data
    data = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(im, axes=k_axes), axes=k_axes),
        axes=k_axes,
    )

    # Compute projections for phase shift simulation
    data_proj = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(data, axes=ro_axis), axis=ro_axis),
        axes=ro_axis,
    )

    # Simulate phase correction data (data_proj*phase_shift)
    phasecor = data_proj.copy()

    broadcast_shape = [
        data_proj.shape[ax]
        if ax in range(len(data_proj.shape)) and ax not in [rep_axis, dir_axis]
        else 2
        for ax in range(len(data_proj.shape))
    ]

    phasecor = np.broadcast_to(phasecor, broadcast_shape) # Broadcast data to new shape which has 2 at rep_axis and
                                                          # dir_axis

    for ax in k_axes:
        if ax != ro_axis:

            if ax == pe_axis:
                phasecor = np.expand_dims(np.take(phasecor, indices=phasecor.shape[pe_axis]/2, axis=ax), axis=ax)
            else:
                phasecor = np.expand_dims(np.take(phasecor, indices=0, axis=ax), axis=ax)

    # Linear phase shift
    phase_shift = np.exp(
        -2j
        * shift
        * np.pi
        * (np.arange(-data_proj.shape[ro_axis] // 2, data_proj.shape[ro_axis] // 2))
        / data_proj.shape[ro_axis]
    )
    phase_shift = np.expand_dims(
        phase_shift, axis=[ax for ax in range(len(data_proj.shape)) if ax != ro_axis]
    )
    slices = [slice(None)] * phasecor.ndim
    slices[dir_axis] = 0
    phasecor[tuple(slices)] *= phase_shift[tuple(slices)]

    # Back projection to k-space
    phasecor = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(phasecor, axes=ro_axis), axis=ro_axis),
        axes=ro_axis,
    )
    slices = [slice(None)] * data.ndim
    slices[pe_axis] = slice(None, None, 2)
    data[tuple(slices)] = np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(data_proj[tuple(slices)] * phase_shift, axes=ro_axis),
            axis=ro_axis,
        ),
        axes=ro_axis,
    )

    return data, im, phasecor, phase_shift


class TestEPITools(unittest.TestCase):
    """!
    Unit test class for mrpy_epi_tools
    """

    def test_calculate_phase_shifts(self, debug_plot:bool = True):
        """!
        @brief Unit Test for calculate_phase_shifts
        @details We simulate phase shifted k-space data and calculate the phase shifts and compare to the
                 simulated phase shifts. Criteria: Calculated and simulated shifts need to be close with a tolerance of
                 0.06.
        @param self: Reference to object

        @author: Jörn Huber, Tom Lütjen
        """

        print("Testing calculate_phase_shifts")

        _, _, phasecor_one_vox, phase_shift_one_vox = generate_test_data(
            1.0,
            (64, 1, 64, 1, 1, 1, 1),
            k_axes=(0, 2, 3),
            ro_axis=0,
            pe_axis=2,
            rep_axis=5,
            dir_axis=6,
        )

        _, _, phasecor_half_vox, phase_shift_half_vox = generate_test_data(
            0.5,
            (64, 1, 64, 1, 1, 1, 1),
            k_axes=(0, 2, 3),
            ro_axis=0,
            pe_axis=2,
            rep_axis=5,
            dir_axis=6,
        )

        _, _, phasecor_quarter_vox, phase_shift_quarter_vox = generate_test_data(
            0.25,
            (64, 1, 64, 1, 1, 1, 1),
            k_axes=(0, 2, 3),
            ro_axis=0,
            pe_axis=2,
            rep_axis=5,
            dir_axis=6,
        )

        phasecor_processing = np.zeros((64, 1, 1, 3, 1, 2, 2), dtype=complex)
        phasecor_processing[:, 0, 0, 0, 0, 0, 1] = phasecor_one_vox[:, 0, 0, 0, 0, 0, 1]
        phasecor_processing[:, 0, 0, 0, 0, 0, 0] = phasecor_one_vox[:, 0, 0, 0, 0, 0, 0]
        phasecor_processing[:, 0, 0, 0, 0, 1, 1] = phasecor_one_vox[:, 0, 0, 0, 0, 1, 1]

        phasecor_processing[:, 0, 0, 1, 0, 0, 1] = phasecor_half_vox[:, 0, 0, 0, 0, 0, 1]
        phasecor_processing[:, 0, 0, 1, 0, 0, 0] = phasecor_half_vox[:, 0, 0, 0, 0, 0, 0]
        phasecor_processing[:, 0, 0, 1, 0, 1, 1] = phasecor_half_vox[:, 0, 0, 0, 0, 1, 1]

        phasecor_processing[:, 0, 0, 2, 0, 0, 1] = phasecor_quarter_vox[:, 0, 0, 0, 0, 0, 1]
        phasecor_processing[:, 0, 0, 2, 0, 0, 0] = phasecor_quarter_vox[:, 0, 0, 0, 0, 0, 0]
        phasecor_processing[:, 0, 0, 2, 0, 1, 1] = phasecor_quarter_vox[:, 0, 0, 0, 0, 1, 1]

        estimated_phase_shift = epi_tools.calculate_phase_shifts(phasecor_processing, ro_axis=0, rep_axis=5, dir_axis=6)

        self.assertEqual(np.allclose(
            np.abs(estimated_phase_shift[:,0,0,0,0,0,0]), np.abs(phase_shift_one_vox), atol=0.06
        ), True)
        self.assertEqual(np.allclose(
            np.abs(estimated_phase_shift[:,0,0,1,0,0,0]), np.abs(phase_shift_half_vox), atol=0.06
        ), True)
        self.assertEqual(np.allclose(
            np.abs(estimated_phase_shift[:,0,0,2,0,0,0]), np.abs(phase_shift_quarter_vox), atol=0.06
        ), True)


    def test_remove_phase_shifts(self, debug_plot=True):
        """!
        @brief Unit Test for remove_phase_shifts
        @details We simulate phase shifted k-space data and remove the phase shifts and compare to the original data.
                 Criteria: All elements in reconstructed in simulated images need to be close with a tolerance of 2e-2.
        @param self: Reference to object

        @author: Jörn Huber, Tom Lütjen
        """
        print("Testing remove_phase_shifts")

        data, im, phasecor, phase_shift = generate_test_data(
            1.0,
            (64, 1, 64, 1, 1, 1, 1),
            k_axes=(0, 2, 3),
            ro_axis=0,
            pe_axis=2,
            rep_axis=5,
            dir_axis=6,
        )

        estimated_phase_shift = epi_tools.calculate_phase_shifts(
            phasecor, ro_axis=0, rep_axis=5, dir_axis=6
        )

        slices = [slice(None)] * data.ndim
        slices[2] = slice(None, None, 2)
        data_corrected = data.copy()

        data_corrected[tuple(slices)] = epi_tools.remove_phase_shifts(
            data[tuple(slices)], estimated_phase_shift, ro_axis=0
        )

        reco = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(data, axes=(0, 2, 3)), axes=(0, 2, 3)),
            axes=(0, 2, 3),
        )
        reco_corrected = np.fft.fftshift(
            np.fft.ifftn(
                np.fft.ifftshift(data_corrected, axes=(0, 2, 3)), axes=(0, 2, 3)
            ),
            axes=(0, 2, 3),
        )

        if debug_plot:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(np.abs(reco[:, 0, :, 0, 0, 0, 0]), cmap="gray")
            axs[0, 0].set_title("Uncorrected Reconstruction")
            axs[0, 1].imshow(np.abs(reco_corrected[:, 0, :, 0, 0, 0, 0]), cmap="gray")
            axs[0, 1].set_title("Corrected Reconstruction")
            axs[1, 0].imshow(np.log(np.abs(data[:, 0, :, 0, 0, 0, 0])), cmap="gray")
            axs[1, 0].set_title("Uncorrected k-space")
            axs[1, 1].imshow(np.log(np.abs(data_corrected[:, 0, :, 0, 0, 0, 0])), cmap="gray")
            axs[1, 1].set_title("Corrected k-space")
            plt.show()

        self.assertEqual(np.allclose(reco_corrected, im, atol=2e-2), True)


if __name__ == '__main__':
    unittest.main()
