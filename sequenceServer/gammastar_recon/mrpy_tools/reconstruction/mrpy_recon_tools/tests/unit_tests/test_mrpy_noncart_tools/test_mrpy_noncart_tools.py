"""!
@brief Collection of unit tests for mrpy_noncart_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import mrpy_noncart_tools as noncart_tools


class TestNonCartTools(unittest.TestCase):
    """!
    Unit test class for mrpy_noncart_tools
    """

    def test_calc_propeller_blade_increment_from_trajs(self):

        """
        @brief UT which validates the correct functionality of calc_propeller_blade_increment_from_trajs.
        @details Tests whether the angle between blades is calculated correctly from a set of trajectories.
                 Criteria. Calculates the angle between two vectors with a 90° angular difference and verifies the
                 outcome.

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


    def test_calc_equidistant_propeller_trajectory(self):
        """!
        @brief UT which validates the correct functionality of calc_equidistant_propeller_trajectory
        @details The test creates a numpy array and chosen trajectory entries are validated to comply with target
                 values. Criteria: Target and calculated samples need to agree within three places.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: calc_equidistant_propeller_trajectory")

        test_array = np.zeros((64, 32, 2), dtype=complex)

        traj = noncart_tools.calc_equidistant_propeller_trajectory(
            ksp_data=test_array,
            ro_axis=0,
            pe_axis=1,
            acq_axis=2,
            acq_angle_incr=180.0 / 2.0
        )

        self.assertAlmostEqual(traj[32, 15, 1, 0], 1.0, 3)
        self.assertAlmostEqual(traj[32, 15, 0, 0], 0.0, 3)
        self.assertAlmostEqual(traj[63, 15, 1, 0], 1.0, 3)
        self.assertAlmostEqual(traj[63, 15, 0, 0], 31.0, 3)
        self.assertAlmostEqual(traj[0, 15, 0, 0], -32.0, 3)


    def test_apply_nufft(self):
        """!
        @brief UT which validates the correct functionality of apply_nufft
        @details Part 1: We validate the functionality with a 2D radial configuration. Criteria: Dimensions of nufft
                 object must comply to the estimated size.
                 Part 2: We validate the functionality with a 2D PROPELLER configuration. Criteria: Dimensions of
                 nufft object must comply to the estimated size.

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_noncart_tools: apply_nufft")

        # Part 1: We validate the functionality with a 2D radial configuration. Criteria: Dimensions of nufft object
        # must comply to the estimated size.

        test_data = np.ones((64, 1, 32, 1000), dtype=complex)
        test_data[0:16, :, :, :] = 0
        test_data[48:, :, :, :] = 0
        test_data = np.fft.fftshift(np.fft.fftn(test_data, axes=(0,)), axes=0)

        traj_data = noncart_tools.calc_equidistant_propeller_trajectory(
            ksp_data=test_data,
            ro_axis=0,
            pe_axis=2,
            acq_axis=3,
            acq_angle_incr=180.0 / 1000
        )
        traj_data = np.moveaxis(traj_data, -1, 1)
        traj_data = np.squeeze(traj_data, 2)

        nufft_data = noncart_tools.apply_nufft(
            ksp_data=test_data[:, :, 16, :],
            traj_data=traj_data[:, :, 16, :],
            mat_size=64,
            ro_axis=0,
            ch_axis=1,
            acq_axis=2
        )

        self.assertEqual(nufft_data.shape, (64, 1, 64))

        # Part 2: We validate the functionality with a 2D PROPELLER configuration. Criteria: Dimensions of nufft object
        # must comply to the estimated size.

        test_data = np.ones((64, 1, 32, 16), dtype=complex)
        traj_data = noncart_tools.calc_equidistant_propeller_trajectory(
            ksp_data=test_data,
            ro_axis=0,
            pe_axis=2,
            acq_axis=3,
            acq_angle_incr=180.0 / 16.0
        )
        traj_data = np.moveaxis(traj_data, -1, 1)
        traj_data = np.squeeze(traj_data, 2)

        nufft_data = noncart_tools.apply_nufft(
            ksp_data=test_data,
            traj_data=traj_data,
            mat_size=64,
            ro_axis=0,
            ch_axis=1,
            acq_axis=3,
            pe1_axis=2
        )

        self.assertEqual(nufft_data.shape, (64, 1, 64, 1))


if __name__ == '__main__':
    unittest.main()
