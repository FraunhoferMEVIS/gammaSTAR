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
import ismrmrd
import mrpy_io_tools as io_tools
import mrpy_ismrmrd_tools as ismrmrd_tools


class TestIOTools(unittest.TestCase):
    """!
    Unit test class for mrpy_io_tools
    """

    # def test_unit_read_siemens_twix(self):
    #     """!@brief UT which validates the correct functionality of read_siemens_twix
    #
    #     @param self: Reference to object
    #
    #     """
    #     print("\nTesting twix data loading of single raid data")
    #     noise, noise_hdr, ksp, ksp_phasecor, ksp_rtfeedback, ksp_paracalib, meas_hdr = io_tools.read_siemens_twix(
    #         path_to_testdata + "/read_siemens_twix/meas_MID00045_FID39253_t2_blade_tra_p2.dat")
    #     self.assertTrue(noise is None)
    #     self.assertTrue(noise_hdr is None)
    #     self.assertTrue(ksp.shape == (320, 16, 28, 10, 28, 18))
    #     self.assertTrue(ksp_phasecor is None)
    #     self.assertTrue(ksp_rtfeedback is None)
    #     self.assertTrue(ksp_paracalib is None)
    #     self.assertTrue(meas_hdr is not None)
    #
    #     print("\nTesting twix data loading of multi raid data")
    #     noise, noise_hdr, ksp, ksp_phasecor, ksp_rtfeedback, ksp_paracalib, meas_hdr = io_tools.read_siemens_twix(
    #         path_to_testdata + "/read_siemens_twix/meas_MID00065_FID39123_t2_blade_tra_p2_single_pos.dat")
    #     self.assertTrue(noise.shape == (256, 10, 128, 2))
    #     self.assertTrue(noise_hdr is not None)
    #     self.assertTrue(ksp.shape == (320, 10, 16, 16, 32))
    #     self.assertTrue(ksp_phasecor is None)
    #     self.assertTrue(ksp_rtfeedback is None)
    #     self.assertTrue(ksp_paracalib is None)
    #     self.assertTrue(meas_hdr is not None)

    def test_write_dcm(self):
        """!
        @brief UT which validates the correct functionality of write_dcm
        @details The test writes a dicom file with defined header entries. The created dicom file is loaded and
                 correct header entries are analyzed.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_io_tools: write_dcm")
        test_im = np.zeros((1, 64, 64))*3000 #Normalize and scale to short int range
        io_tools.write_dcm("temp.dcm", test_im,
                           'UT Study',
                           'UT Series',
                           'UT System',
                           'UT Vendor',
                           'UT Body',
                           'UT Protocol',
                           'UT Sequence',
                           '3D')
        ds = pydicom.dcmread("temp.dcm")
        dcm_elem_vendor = ds[0x00080070]
        self.assertTrue(dcm_elem_vendor.value == 'UT Vendor')
        dcm_elem_bodypart = ds[0x00180015]
        self.assertTrue(dcm_elem_bodypart.value == 'UT Body')
        dcm_elem_protocol = ds[0x00181030]
        self.assertTrue(dcm_elem_protocol.value == 'UT Protocol')
        dcm_elem_sequence = ds[0x00189005]
        self.assertTrue(dcm_elem_sequence.value == 'UT Sequence')
        os.remove("temp.dcm")

    def test_write_dcm_from_ismrmrd_image(self):
        """!
        @brief UT which validates the correct functionality of write_dcm_from_ismrmrd_image
        @details The test first creates an ismrmrd image from a numpy array. Afterwards, the ismrmrd image is written
                 to DICOM. We validate that the dicom file exists. TODO: We should add a test which validates the
                 pixel contents of the DICOM file.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_io_tools: write_dcm_from_ismrmrd_image")

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

    def test_save_to_bart_cfl(self):
        """!
        @brief UT which validates the correct functionality of save_to_bart_cfl
        @details The test writes a numpy array filled with zeros to a bart file. Afterwards, it is validated whether
                 respective files are present.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_io_tools: save_to_bart_cfl")
        test_im = np.zeros((64, 64), dtype=complex)
        self.assertRaises(ValueError, io_tools.save_to_bart_cfl, "test_im", np.real(test_im))

        io_tools.save_to_bart_cfl("test_im", test_im)
        self.assertTrue(os.path.isfile("test_im.cfl"))
        self.assertTrue(os.path.isfile("test_im.hdr"))
        os.remove("test_im.cfl")
        os.remove("test_im.hdr")

    def test_load_from_bart_cfl(self):
        """!
        @brief UT which validates the correct functionality of load_from_bart_cfl.
        @details The test first writes a bart file filled with zeros to the hard disk. Afterwards, the file is loaded
                 and resulting image values are compared to zero.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_io_tools: load_from_bart_cfl")
        io_tools.save_to_bart_cfl("test_im", np.zeros((64, 64), dtype=complex))
        test_im = io_tools.load_from_bart_cfl("test_im")
        self.assertTrue(test_im.shape == (64, 64))
        self.assertEqual(np.max(np.abs(test_im).flatten()), 0.0)

if __name__ == '__main__':
    print("---Running unit tests for mrpy_io_tools---")
    unittest.main()
