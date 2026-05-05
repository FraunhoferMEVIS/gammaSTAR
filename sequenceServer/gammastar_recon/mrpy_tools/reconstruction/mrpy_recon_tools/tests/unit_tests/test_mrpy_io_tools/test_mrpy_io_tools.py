"""!
@brief Collection of unit tests for mrpy_io_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import unittest
import os
import numpy as np
import pydicom
import shutil
import ismrmrd
import mrpy_io_tools as io_tools
import mrpy_ismrmrd_tools as ismrmrd_tools


class TestIOTools(unittest.TestCase):
    """!
    Unit test class for mrpy_io_tools
    """


    def test_get_folder_name(self):
        """!
        @brief UT which validates the correct functionality of get_folder_name
        @details The test writes a dicom file with defined header entries. The created dicom file is loaded and
                 correct header entries are analyzed. Criteria: folder name needs to correspond to target string.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_io_tools: get_folder_name")
        folder_name = io_tools.get_folder_name("test_protocol_123")
        self.assertEqual(folder_name, "test_protocol_123/0")


    def test_write_dcm_from_ismrmrd_image(self):
        """!
        @brief UT which validates the correct functionality of write_dcm_from_ismrmrd_image
        @details The test first creates an ismrmrd image from a numpy array. Afterwards, the ismrmrd image is written
                 to DICOM. Criteria: Target dicom file exists.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_io_tools: write_dcm_from_ismrmrd_image")

        if os.path.exists("DummyProtocol") and os.path.isdir("DummyProtocol"):
            shutil.rmtree("DummyProtocol")

        # We first create an ismrmrd image
        test_np_acq = np.ones((64, 16))
        test_np_image = np.ones((64, 64, 8))
        test_acq = [ismrmrd.Acquisition.from_array(test_np_acq)]
        test_acq[0].idx.average = 0  # pylint: disable=maybe-no-member
        test_acq[0].idx.contrast = 0  # pylint: disable=maybe-no-member
        test_acq[0].idx.phase = 0  # pylint: disable=maybe-no-member
        test_acq[0].idx.repetition = 0  # pylint: disable=maybe-no-member
        test_acq[0].idx.set = 0  # pylint: disable=maybe-no-member
        test_acq[0].idx.slice = 0  # pylint: disable=maybe-no-member
        test_acq[0].read_dir[0] = 1  # pylint: disable=maybe-no-member
        test_acq[0].read_dir[1] = 0  # pylint: disable=maybe-no-member
        test_acq[0].read_dir[2] = 0  # pylint: disable=maybe-no-member
        test_acq[0].phase_dir[0] = 0  # pylint: disable=maybe-no-member
        test_acq[0].phase_dir[1] = 1  # pylint: disable=maybe-no-member
        test_acq[0].phase_dir[2] = 0  # pylint: disable=maybe-no-member
        test_acq[0].slice_dir[0] = 0  # pylint: disable=maybe-no-member
        test_acq[0].slice_dir[1] = 0  # pylint: disable=maybe-no-member
        test_acq[0].slice_dir[2] = 1  # pylint: disable=maybe-no-member
        test_acq[0].position[0] = 1  # pylint: disable=maybe-no-member
        test_acq[0].position[1] = 1  # pylint: disable=maybe-no-member
        test_acq[0].position[2] = 1  # pylint: disable=maybe-no-member
        test_header = ismrmrd.xsd.CreateFromDocument(ismrmrd_tools.create_dummy_ismrmrd_header())
        meas_idx = ismrmrd_tools.MeasIDX(0, 0, 0, 0, 0)
        ismrmrd_image = ismrmrd_tools.numpy_array_to_ismrmrd_image(test_np_image, test_acq, test_header, 3, meas_idx)

        io_tools.write_dcm_from_ismrmrd_image(ismrmrd_image)
        try:
            ds = pydicom.dcmread("DummyProtocol/series_3.dcm")
            os.remove("DummyProtocol/series_3.dcm")
            is_available = True
        except:
            is_available = False
        self.assertEqual(is_available, True)


if __name__ == '__main__':
    print("---Running unit tests for mrpy_io_tools---")
    unittest.main()
