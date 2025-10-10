"""!
@brief Collection of unit tests for mrpy_gammastar_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import numpy as np
import mrpy_gammastar_tools as gammastar_tools


class TestGammstarTools(unittest.TestCase):
    """!
    Unit test class for mrpy_gammastar_tools
    """

    def test_reorder_twix_with_raw_reps(self):
        """!
        @brief UT which validates the correct functionality of reorder_twix_with_raw_reps
        @details

        @param self: Reference to object

        @author Jörn Huber
        """
        print("\nTesting mrpy_gammastar_tools: reorder_twix_with_raw_reps")

        phase_cor_raw_rep = dict()
        phase_cor_raw_rep['adc_header'] = dict()
        phase_cor_raw_rep['adc_header']['ACQ_IS_PHASECORR_DATA'] = True
        phase_cor_np = 1*np.ones((64, 20, 1), dtype=complex)

        rt_feedback_raw_rep = dict()
        rt_feedback_raw_rep['adc_header'] = dict()
        rt_feedback_raw_rep['adc_header']['ACQ_IS_RTFEEDBACK_DATA'] = True
        rt_feedback_np = 2*np.ones((64, 20, 1), dtype=complex)

        para_calib_raw_rep = dict()
        para_calib_raw_rep['adc_header'] = dict()
        para_calib_raw_rep['adc_header']['ACQ_IS_PARALLEL_CALIBRATION'] = True
        para_calib_np = 3*np.ones((64, 20, 1), dtype=complex)

        imaging_raw_rep = dict()
        imaging_raw_rep['adc_header'] = dict()
        imaging_np = 4*np.ones((64, 20, 2), dtype=complex)

        raw_reps = [rt_feedback_raw_rep, phase_cor_raw_rep, para_calib_raw_rep, imaging_raw_rep, imaging_raw_rep]
        reorder_np = gammastar_tools.reorder_twix_with_raw_reps(raw_reps, phase_cor_np, rt_feedback_np, para_calib_np, imaging_np)

        self.assertTrue(np.all(reorder_np[:, :, 0] == (2. + 0.j)))
        self.assertTrue(np.all(reorder_np[:, :, 1] == (1. + 0.j)))
        self.assertTrue(np.all(reorder_np[:, :, 2] == (3. + 0.j)))
        self.assertTrue(np.all(reorder_np[:, :, 3] == (4. + 0.j)))
        self.assertTrue(np.all(reorder_np[:, :, 4] == (4. + 0.j)))

if __name__ == '__main__':
    unittest.main()
