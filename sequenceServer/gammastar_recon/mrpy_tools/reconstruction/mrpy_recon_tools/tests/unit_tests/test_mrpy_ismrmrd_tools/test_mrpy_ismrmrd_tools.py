"""!
@brief Collection of unit tests for mrpy_ismrmrd_tools.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""


import unittest
import math
import numpy as np
import threading
import socket
import time
from queue import Queue
import ismrmrd
import logging
import mrpy_ismrmrd_tools as ismrmrd_tools


def test_server():
    """!
    @brief Helper function which implements a test server, which accepts a client connection and sends some data back.
    @details Partly based on AI generated code (GPT-4.1) with adaptions using the following prompt: "Hello, can you give
             me a python code snippet which uses a single script to first opens a socket connection on localhost and
             port 9002 and then connects with a client sending a dummy text message? The call to the server shall be
             non-blocking."

    @author: Jörn Huber
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 9003))
    server_socket.listen(5)
    print("Server started on port 9003...")
    connection, addr = server_socket.accept()

    time.sleep(5)
    con = ismrmrd_tools.ConnectionBuffer(connection)
    acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
    image1 = ismrmrd.Image.from_array(1*np.ones((64, 64)), acquisition=acq, transpose=False)
    image2 = ismrmrd.Image.from_array(2*np.ones((64, 64)), acquisition=acq, transpose=False)
    image3 = ismrmrd.Image.from_array(3*np.ones((64, 64)), acquisition=acq, transpose=False)

    # Send stuff which is catched by the injector
    con.send_image(image1)
    con.send_image(image2)
    con.send_image(image3)
    con.send_close()

    con.shutdown_close()
    server_socket.close()


class TestISMRMRDTools(unittest.TestCase):
    """!
    Unit test class for mrpy_ismrmrd_tools
    """

    def test_numpy_array_to_ismrmrd_image(self):
        """!
        @brief UT which validates the correct functionality of numpy_array_to_ismrmrd_image.
        @details The test first creates numpy arrays for image data and acquisition data. The numpy acquisition data
                 is used to create an acquisition object and the idx entries as well as position information are
                 manipulated. Afterwards, a dummy ismrmrd header file is created. Finally, we create an ismrmrd image
                 from the numpy data together with the information from the acquisition and header objects.

        @param self: Reference to object

        @author: Jörn Huber
        """
        test_np_acq = np.ones((64, 16), dtype=complex)
        test_np_image = np.ones((64, 64, 8), dtype=complex)
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
        ismrmrd_image = ismrmrd_tools.numpy_array_to_ismrmrd_image(np.abs(test_np_image), test_acq, test_header, 3, meas_idx)

        # Validate idx counters of image
        self.assertEqual(ismrmrd_image.average, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.contrast, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.phase, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.repetition, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.set, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.slice, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.image_index, 0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.image_series_index, 3)  # pylint: disable=maybe-no-member

        # Validate position information of image which should be extracted from the acquisition
        self.assertEqual(ismrmrd_image.read_dir[0], 1.0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.read_dir[1], 0.0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.read_dir[2], 0.0)  # pylint: disable=maybe-no-member

        self.assertEqual(ismrmrd_image.phase_dir[0], 0.0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.phase_dir[1], 1.0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.phase_dir[2], 0.0)  # pylint: disable=maybe-no-member

        self.assertEqual(ismrmrd_image.slice_dir[0], 0.0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.slice_dir[1], 0.0)  # pylint: disable=maybe-no-member
        self.assertEqual(ismrmrd_image.slice_dir[2], 1.0)  # pylint: disable=maybe-no-member

        # Validate the image content
        self.assertEqual(np.sum(ismrmrd_image.data.flatten() - np.ones((64 * 64 * 8))), 0.0)

        # Validate image header entries
        self.assertEqual(ismrmrd_image.meta['protocolName'], 'DummyProtocol')
        self.assertEqual(ismrmrd_image.meta['FrameOfReference'], '1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0')
        self.assertEqual(ismrmrd_image.meta['DataRole'], 'Image')

    def test_numpy_array_to_ismrmrd_acqs(self):
        """!
        @brief UT which validates the correct functionality of numpy_array_to_ismrmrd_acqs.
        @details The test creates ismrmrd acquisition objects from a numpy array. We first validate that the data
                 content of individual acquisition objects is correct. Subsequently, the maximum and minimum idx entries
                 are validated. We continue with a validation of the correct ismrmrd flags. Finally, header entries
                 of the acquisition objects are validated.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: numpy_array_to_ismrmrd_acqs")
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        imaging_flags = ["ACQ_USER1"]
        ksp_phasecor = np.ones((128, 3, 3, 16), dtype=complex)
        phasecor_flags = ["ACQ_USER2", "ACQ_IS_PHASECORR_DATA"]
        list_of_data = [ksp_imaging, ksp_phasecor]
        list_of_flags = [imaging_flags, phasecor_flags]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3
        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position)

        # Validate data written into acqs
        self.assertEqual(len(list_of_acqs), 96)
        for acq in list_of_acqs:
            self.assertEqual(np.sum(acq.data.flatten()-np.ones((3*128))), 0)

        # Validate maximum and minimum idx entries
        max_kspace_encoding_pe1 = max(list_of_acqs,
                                      key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1
        max_kspace_encoding_pe2 = max(list_of_acqs,
                                      key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2
        min_kspace_encoding_pe1 = min(list_of_acqs,
                                      key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1
        min_kspace_encoding_pe2 = min(list_of_acqs,
                                      key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2
        self.assertEqual(max_kspace_encoding_pe1, 2)
        self.assertEqual(min_kspace_encoding_pe1, 0)
        self.assertEqual(max_kspace_encoding_pe2, 15)
        self.assertEqual(min_kspace_encoding_pe2, 0)

        # Validate correct flags
        acq_counter = 0
        for acq in list_of_acqs:
            acq_flags = ismrmrd_tools.bitmask_to_flags(acq.flags)

            if acq_counter < 3*16:
                self.assertEqual(("ACQ_USER1" in acq_flags), True)
            else:
                self.assertEqual(("ACQ_USER2" in acq_flags), True)
                self.assertEqual(("ACQ_IS_PHASECORR_DATA" in acq_flags), True)

            if acq.idx.kspace_encode_step_1 == 0:
                self.assertEqual(("ACQ_FIRST_IN_ENCODE_STEP1" in acq_flags), True)

                if acq.idx.kspace_encode_step_2 == 0:
                    self.assertEqual(("ACQ_FIRST_IN_ENCODE_STEP2" in acq_flags), True)

            if acq.idx.kspace_encode_step_1 == 2:
                self.assertEqual(("ACQ_LAST_IN_ENCODE_STEP1" in acq_flags), True)

                if acq.idx.kspace_encode_step_2 == 15:
                    self.assertEqual(("ACQ_LAST_IN_ENCODE_STEP2" in acq_flags), True)

            acq_counter = acq_counter + 1

        # Validate other relevant header entries in acq objects
        for acq in list_of_acqs:
            acq.read_dir[0] = 1  # pylint: disable=maybe-no-member
            acq.read_dir[1] = 0  # pylint: disable=maybe-no-member
            acq.read_dir[2] = 0  # pylint: disable=maybe-no-member
            acq.phase_dir[0] = 0  # pylint: disable=maybe-no-member
            acq.phase_dir[1] = 1  # pylint: disable=maybe-no-member
            acq.phase_dir[2] = 0  # pylint: disable=maybe-no-member
            acq.slice_dir[0] = 0  # pylint: disable=maybe-no-member
            acq.slice_dir[1] = 0  # pylint: disable=maybe-no-member
            acq.slice_dir[2] = 1  # pylint: disable=maybe-no-member
            acq.position[0] = -1  # pylint: disable=maybe-no-member
            acq.position[1] = -2  # pylint: disable=maybe-no-member
            acq.position[2] = -3  # pylint: disable=maybe-no-member

    def test_create_dummy_ismrmrd_header(self):
        """!
        @brief UT which validates the correct functionality of create_dummy_ismrmrd_header.
        @details The test creates the minimal dummy header. Afterwards, header strings are compared to target values.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: create_dummy_ismrmrd_header")
        dummy_header = ismrmrd_tools.create_dummy_ismrmrd_header()
        self.assertEqual('<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">' in dummy_header, True)
        self.assertEqual("<measurementInformation><measurementID>45098_164288619_164288624_176</measurementID><patientPosition>HFS</patientPosition><protocolName>DummyProtocol</protocolName><frameOfReferenceUID>1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0</frameOfReferenceUID></measurementInformation>" in dummy_header, True)
        self.assertEqual("<acquisitionSystemInformation><systemVendor>Vendor</systemVendor><systemModel>Model</systemModel><systemFieldStrength_T>2.89362001</systemFieldStrength_T><receiverChannels>16</receiverChannels><institutionName>Institution</institutionName></acquisitionSystemInformation>" in dummy_header, True)
        self.assertEqual("<experimentalConditions><H1resonanceFrequency_Hz>123251770</H1resonanceFrequency_Hz></experimentalConditions>" in dummy_header, True)
        self.assertEqual("</ismrmrdHeader>" in dummy_header, True)

    def test_convert_acquisitions(self):
        """!
        @brief UT which validates the correct functionality of convert_acquisitions.
        @details The test first creates multiple acquisition objects with different ACQ_IS... flags. In addition,
                 a dummy header is created which contains relevant information like the encoded space etc. Afterwards,
                 convert_acquisitions is called and the acquisition data is converted to corresponding numpy arrays.

        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: ConnectionBuffer.convert_acquisitions")
        inv_flags = {v: k for k, v in ismrmrd_tools.IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.items()}
        np_data = np.ones((128, 20), dtype=complex)

        acq_noise = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_noise.set_flag(inv_flags['ACQ_IS_NOISE_MEASUREMENT'])

        acq_parallel_calib = []
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib[0].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[0].idx.kspace_encode_step_1 = 0
        acq_parallel_calib[0].idx.kspace_encode_step_2 = 0
        acq_parallel_calib[1].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[1].idx.kspace_encode_step_1 = 1
        acq_parallel_calib[1].idx.kspace_encode_step_2 = 0
        acq_parallel_calib[2].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[2].idx.kspace_encode_step_1 = 0
        acq_parallel_calib[2].idx.kspace_encode_step_2 = 1
        acq_parallel_calib[3].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[3].idx.kspace_encode_step_1 = 1
        acq_parallel_calib[3].idx.kspace_encode_step_2 = 1

        acq_navigation = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_navigation.set_flag(inv_flags['ACQ_IS_NAVIGATION_DATA'])

        acq_phasecor = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_phasecor.set_flag(inv_flags['ACQ_IS_PHASECORR_DATA'])

        acq_rt_feedback = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_rt_feedback.set_flag(inv_flags['ACQ_IS_RTFEEDBACK_DATA'])

        con_buffer = ismrmrd_tools.ConnectionBuffer(None)
        con_buffer.meas_data.data['ACQ_IS_NOISE_MEASUREMENT'] = [acq_noise]
        con_buffer.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'] = acq_parallel_calib
        con_buffer.meas_data.data['ACQ_IS_NAVIGATION_DATA'] = [acq_navigation]
        con_buffer.meas_data.data['ACQ_IS_PHASECORR_DATA'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_REVERSE_ACQ_IS_PHASECORR_DATA'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_RTFEEDBACK_DATA'] = [acq_rt_feedback]
        con_buffer.meas_data.data['ACQ_IS_REVERSE_ACQ_IS_IMAGING'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_IMAGING_ACQ_IS_REVERSE'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_IMAGING'] = [acq_phasecor]

        con_buffer.headers.append(ismrmrd.xsd.CreateFromDocument(ismrmrd_tools.create_dummy_ismrmrd_header()))
        con_buffer.convert_acquisitions()

        self.assertTrue('NP_IS_NOISE_MEASUREMENT' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_PARALLEL_CALIBRATION' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_NAVIGATION_DATA' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_PHASECORR_DATA' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_REVERSE_NP_IS_PHASECORR_DATA' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_RTFEEDBACK_DATA' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_IMAGING' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_REVERSE_NP_IS_IMAGING' in con_buffer.meas_data.data)
        self.assertTrue('NP_IS_IMAGING_NP_IS_REVERSE' in con_buffer.meas_data.data)

        self.assertEqual(con_buffer.meas_data.pf_factor_pe1, 1.0)
        self.assertEqual(con_buffer.meas_data.pf_factor_pe2, 1.0)
        self.assertEqual(con_buffer.meas_data.accel_pe1, 1.0)
        self.assertEqual(con_buffer.meas_data.accel_pe2, 1.0)
        self.assertEqual(con_buffer.meas_data.calib_reg_pe1[0], 0)
        self.assertEqual(con_buffer.meas_data.calib_reg_pe1[1], 2)
        self.assertEqual(con_buffer.meas_data.calib_reg_pe2[0], 0)
        self.assertEqual(con_buffer.meas_data.calib_reg_pe2[1], 2)

    def test_send_text(self):
        """!
        @brief UT which validates the correct functionality of send_text.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, a text message is sent
                 and the content which is received on the server side is evaluated for correctness.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_text")
        que = Queue()
        server_thread = threading.Thread(target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)), args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual text and close messages
        client_con.send_text("test_send_message_text")
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_TEXT)
        self.assertEqual(server_con.texts[0], 'test_send_message_text')
        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

    def test_send_config_text(self):
        """!
        @brief UT which validates the correct functionality of send_config_text.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, a config_text message is
                 sent and the content which is received on the server side is evaluated for correctness.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_config_text")
        que = Queue()
        server_thread = threading.Thread(target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)), args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual text and close messages
        client_con.send_config_text("test_send_config_text")
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_TEXT)
        self.assertEqual(server_con.config_texts[0], 'test_send_config_text')
        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

    def test_send_metadata(self):
        """!
        @brief UT which validates the correct functionality of send_metadata.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, a header message is sent
                 and the content which is received on the server side is evaluated for correctness. Finally, we
                 validate that no other messages besides the final close message were received.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_metadata")
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual header and close messages
        header = ismrmrd_tools.create_dummy_ismrmrd_header()
        client_con.send_metadata(header)
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_HEADER)
        self.assertEqual(str(server_con.headers[0].Meta),
                         "<class 'ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader.Meta'>")
        self.assertEqual(server_con.headers[0].experimentalConditions.H1resonanceFrequency_Hz,
                         123251770)
        self.assertEqual(server_con.headers[0].measurementInformation.frameOfReferenceUID,
                         '1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0')
        self.assertEqual(server_con.headers[0].measurementInformation.measurementID,
                         '45098_164288619_164288624_176')
        self.assertEqual(server_con.headers[0].measurementInformation.patientPosition.name,
                         'HFS')
        self.assertEqual(server_con.headers[0].measurementInformation.protocolName,
                         'DummyProtocol')
        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

    def test_send_config_file(self):
        """!
        @brief UT which validates the correct functionality of send_config_file.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, a config file message is
                 sent and the content which is received on the server side is evaluated for correctness. Finally, we
                 validate that no other messages besides the final close message were received.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_config_file")
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual header and close messages
        client_con.send_config_file("test_send_config_file")
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE)
        self.assertEqual(server_con.config_files[0], 'test_send_config_file')
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

    def test_send_image(self):
        """!
        @brief UT which validates the correct functionality of send_message_image.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, an image message is
                 sent and the content which is received on the server side is evaluated for correctness. Finally, we
                 validate that no other messages besides the final close message were received.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_image")
        # Establish Open Recon Server in seperate thread
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual header and close messages
        acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        image = ismrmrd.Image.from_array(np.ones((64, 64)), acquisition=acq, transpose=False)
        client_con.send_image(image)
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        # Validate received images
        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_IMAGE)
        self.assertEqual(server_con.images[0].data.any(), 1.0)
        self.assertEqual(len(server_con.images), 1)

        # Validate that nothing else was received (besides close message)
        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.waveforms), 0)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

    def test_send_close(self):
        """!
        @brief UT which validates the correct functionality of send_close.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, a close message is
                 sent and the content which is received on the server side is evaluated for correctness. Finally, we
                 validate that no other messages besides the final close message were received.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_close")
        # Establish Open Recon Server in seperate thread
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual header and close messages
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        # Validate received images
        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

        # Validate that nothing else was received
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.waveforms), 0)

    def test_send_acquisition(self):
        """!
        @brief UT which validates the correct functionality of send_acquisition.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, an acquisition message is
                 sent and the content which is received on the server side is evaluated for correctness. Finally, we
                 validate that no other messages besides the final close message were received.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: send_acquisition")
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # Send the actual header and close messages
        acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        client_con.send_acquisition(acq)
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        # Validate received images
        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_ACQUISITION)
        self.assertEqual(server_con.meas_data.data['ACQ_IS_IMAGING'][0].data.any(), 1.0)
        self.assertEqual(len(server_con.meas_data.data['ACQ_IS_IMAGING']), 1)

        for acq_key in server_con.meas_data.data:
            self.assertEqual(acq_key, 'ACQ_IS_IMAGING')

        # Validate that nothing else was received (besides close message)
        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE)

    def test_gstar_recon_emitter(self):
        """!
        @brief UT which validates the correct functionality of gstar_recon_emitter.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, ismrmrd messages related
                 to open recon 1.0 and open recon 1.1 protocols are sent to the server and the correct order and
                 identification of messages is validated.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: gstar_recon_emitter")

        # Create a list of acquisition objects, which we want to send
        list_of_acqs = []
        for i in range(0, 10):
            acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
            list_of_acqs.append(acq)
        header = ismrmrd_tools.create_dummy_ismrmrd_header()

        # Validate Protocol 1.0
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        client_con = ismrmrd_tools.gstar_recon_emitter('localhost', 9002, list_of_acqs, header, protocol=1.0, config_message="GstarReconTest")
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        b_correct_protocol = False
        if (server_con.messages_received[0] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE
                and server_con.messages_received[1] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_HEADER
                and all(element == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_ACQUISITION for element in server_con.messages_received[3:-2])
                and server_con.messages_received[-1] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE):
            b_correct_protocol = True
        self.assertEqual(b_correct_protocol, True)

        # Validate Protocol 1.1
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        client_con = ismrmrd_tools.gstar_recon_emitter('localhost', 9002, list_of_acqs, header, protocol=1.1,
                                                       config_message="GstarReconTest")
        client_con.shutdown_close()
        server_thread.join()

        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        b_correct_protocol = False
        if (server_con.messages_received[0] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE
                and server_con.messages_received[1] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_HEADER
                and server_con.messages_received[2] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_TEXT
                and all(element == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_ACQUISITION for element in server_con.messages_received[3:-2])
                and server_con.messages_received[-1] == ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CLOSE
                and server_con.config_files[0] == "openrecon"):
            b_correct_protocol = True
        self.assertEqual(b_correct_protocol, True)

        # Finally, we test an invalid configuration
        self.assertRaises(ValueError, ismrmrd_tools.gstar_recon_emitter, 'localhost', 9002, list_of_acqs, header, 1.2, "GstarReconTest")

    def test_gstar_recon_injector(self):
        """!
        @brief UT which validates the correct functionality of gstar_recon_injector.
        @details The test first opens a test ismrmrd server in a separate thread. Afterwards, a client socket is
                 connected to the server and the open_recon_injector function is called. After 5 seconds, the
                 server will send images to the client, and it is validated whether correct number of images and
                 correct data in images is received.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: gstar_recon_injector")
        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)
        que = Queue()
        server_thread = threading.Thread(target=lambda q: q.put(test_server()), args=(que,))
        server_thread.start()
        time.sleep(1)

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9003))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        ismrmrd_tools.gstar_recon_injector(client_con)
        server_thread.join()

        self.assertEqual(len(client_con.images), 3)
        self.assertEqual(np.all(client_con.images[0].data == 1.0), True)
        self.assertEqual(np.all(client_con.images[1].data == 2.0), True)
        self.assertEqual(np.all(client_con.images[2].data == 3.0), True)
        client_socket.close()

    def test_gstar_recon_server(self):
        """!
        @brief UT which validates the correct functionality of gstar_recon_server.
        @details The test first opens the ismrmrd server in a separate thread. Afterwards, we send defined ismrmrd
                 messages of various types to the server. Finally, we validate that the order of received messages
                 is correct and that the content of messages is correct, too.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: gstar_recon_server")
        # Establish Open Recon Server in seperate thread
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        client_con.send_config_file("test_send_message_config_file")  # Config File
        client_con.send_text("test_send_message_text")  # Text

        header = ismrmrd_tools.create_dummy_ismrmrd_header()
        client_con.send_metadata(header)  # Header

        acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        client_con.send_acquisition(acq)  # Acquisition

        image = ismrmrd.Image.from_array(np.ones((64, 64)), acquisition=acq, transpose=False)
        client_con.send_image(image)  # Image

        client_con.send_close()  # Close

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_TEXT)
        self.assertEqual(server_con.messages_received[2], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_HEADER)
        self.assertEqual(server_con.messages_received[3], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_ACQUISITION)
        self.assertEqual(server_con.messages_received[4], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_IMAGE)

        self.assertEqual(len(server_con.config_files), 1)
        self.assertEqual(len(server_con.texts), 1)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 1)
        self.assertEqual(len(server_con.meas_data.data['ACQ_IS_IMAGING']), 1)
        self.assertEqual(len(server_con.images), 1)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(np.all(server_con.meas_data.data['ACQ_IS_IMAGING'][0].data == 1.0), True)
        self.assertEqual(np.all(server_con.images[0].data == 1.0), True)
        self.assertEqual(server_con.config_files[0], 'test_send_message_config_file')
        self.assertEqual(server_con.texts[0], 'test_send_message_text')
        self.assertEqual(str(server_con.headers[0].Meta),
                         "<class 'ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader.Meta'>")
        self.assertEqual(server_con.headers[0].experimentalConditions.H1resonanceFrequency_Hz,
                         123251770)
        self.assertEqual(server_con.headers[0].measurementInformation.frameOfReferenceUID,
                         '1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0')
        self.assertEqual(server_con.headers[0].measurementInformation.measurementID,
                         '45098_164288619_164288624_176')
        self.assertEqual(server_con.headers[0].measurementInformation.patientPosition.name,
                         'HFS')
        self.assertEqual(server_con.headers[0].measurementInformation.protocolName,
                         'DummyProtocol')

    def test_read_config_file(self):
        """!
        @brief UT which validates the correct functionality of read_config_file.
        @details The test starts a client in a separate thread which tries to send a config_file message. Afterwards,
                 we create an ismrmrd server which accepts the clients connection and reads the specified message.
                 The content of the message is validated afterwards.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_config_file")
        # Establish Open Recon Server in seperate thread
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        client_con.send_config_file("test_send_message_config_file")  # Config File
        client_con.send_close()
        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE)

        self.assertEqual(len(server_con.config_files), 1)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(server_con.config_files[0], 'test_send_message_config_file')

    def test_read_text(self):
        """!
        @brief UT which validates the correct functionality of read_message_text.
        @details The test starts a client in a separate thread which tries to send a text message. Afterwards,
                 we create an ismrmrd server which accepts the clients connection and reads the specified message.
                 The content of the message is validated afterwards.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_text")
        # Establish Open Recon Server in seperate thread
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        client_con.send_text("test_send_message_text")  # Text
        client_con.send_close()

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_TEXT)

        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 1)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(server_con.texts[0], 'test_send_message_text')

    def test_read_config_text(self):
        """!
        @brief UT which validates the correct functionality of read_config_text.
        @details The test starts a client in a separate thread which tries to send a config_text message. Afterwards,
                 we create an ismrmrd server which accepts the clients connection and reads the specified message.
                 The content of the message is validated afterwards.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_config_text")
        # Establish Open Recon Server in seperate thread
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        client_con.send_config_text("test_read_config_text")  # Text
        client_con.send_close()

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_TEXT)

        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 1)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(server_con.config_texts[0], 'test_read_config_text')

    def test_read_metadata(self):
        """!
        @brief UT which validates the correct functionality of read_metadata.
        @details The test starts a client in a separate thread which tries to send a header message. Afterwards,
                 we create an ismrmrd server which accepts the clients connection and reads the specified message.
                 The content of the message is validated afterwards.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_metadata")
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        header = ismrmrd_tools.create_dummy_ismrmrd_header()
        client_con.send_metadata(header)  # Header
        client_con.send_close()  # Close

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_HEADER)

        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 1)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(str(server_con.headers[0].Meta),
                         "<class 'ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader.Meta'>")
        self.assertEqual(server_con.headers[0].experimentalConditions.H1resonanceFrequency_Hz,
                         123251770)
        self.assertEqual(server_con.headers[0].measurementInformation.frameOfReferenceUID,
                         '1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0')
        self.assertEqual(server_con.headers[0].measurementInformation.measurementID,
                         '45098_164288619_164288624_176')
        self.assertEqual(server_con.headers[0].measurementInformation.patientPosition.name,
                         'HFS')
        self.assertEqual(server_con.headers[0].measurementInformation.protocolName,
                         'DummyProtocol')

    def test_read_acquisition(self):
        """!
        @brief UT which validates the correct functionality of read_acquisition.
        @details The test starts a client in a separate thread which tries to send a acquisition message. Afterwards,
                 we create an ismrmrd server which accepts the clients connection and reads the specified message.
                 The content of the message is validated afterwards.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_acquisition")
        # We start a client in a separate thread which tries to connect to the server below
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        inv_flags = {v: k for k, v in ismrmrd_tools.IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.items()}
        acq1 = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        acq1.set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING'])
        client_con.send_acquisition(acq1)  # Acquisition

        acq2 = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        acq2.set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING'])
        acq2.set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        client_con.send_acquisition(acq2)  # Acquisition

        acq3 = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        acq3.set_flag(inv_flags['ACQ_IS_NOISE_MEASUREMENT'])
        client_con.send_acquisition(acq3)  # Acquisition

        client_con.send_close()  # Close

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_ACQUISITION)

        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data['ACQ_IS_NOISE_MEASUREMENT']), 1)
        self.assertEqual(len(server_con.meas_data.data['ACQ_IS_IMAGING']), 2)
        self.assertEqual(len(server_con.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION']), 2)
        self.assertEqual(len(server_con.images), 0)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(np.all(server_con.meas_data.data['ACQ_IS_IMAGING'][0].data == 1.0), True)
        self.assertEqual(np.all(server_con.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'][0].data == 1.0), True)

    def test_read_message_image(self):
        """!
        @brief UT which validates the correct functionality of read_message_image.
        @details The test starts a client in a separate thread which tries to send a acquisition message. Afterwards,
                 we create an ismrmrd server which accepts the clients connection and reads the specified message.
                 The content of the message is validated afterwards.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_message_image")
        # We start a client in a separate thread which tries to connect to the server below
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        image = ismrmrd.Image.from_array(np.ones((64, 64)), acquisition=acq, transpose=False)
        client_con.send_image(image)  # Image
        client_con.send_close()  # Close

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_IMAGE)

        self.assertEqual(len(server_con.config_files), 0)
        self.assertEqual(len(server_con.texts), 0)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 0)
        self.assertEqual(len(server_con.meas_data.data.keys()), 0)
        self.assertEqual(len(server_con.images), 1)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(np.all(server_con.images[0].data == 1.0), True)

    def test_read_close(self):
        """!
        @brief UT which validates the correct functionality of read_close.
        @details The test creates a connection buffer object and populates its acquisition entries first. Afterwards,
                 "read_close" is called and it is validated whether corresponding entries are correctly set in
                 connection buffer.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools.ConnectionBuffer: read_close")
        inv_flags = {v: k for k, v in ismrmrd_tools.IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.items()}
        np_data = np.ones((128, 20), dtype=complex)

        acq_noise = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_noise.set_flag(inv_flags['ACQ_IS_NOISE_MEASUREMENT'])

        acq_parallel_calib = []
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib.append(ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0])))
        acq_parallel_calib[0].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[0].idx.kspace_encode_step_1 = 0
        acq_parallel_calib[0].idx.kspace_encode_step_2 = 0
        acq_parallel_calib[1].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[1].idx.kspace_encode_step_1 = 1
        acq_parallel_calib[1].idx.kspace_encode_step_2 = 0
        acq_parallel_calib[2].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[2].idx.kspace_encode_step_1 = 0
        acq_parallel_calib[2].idx.kspace_encode_step_2 = 1
        acq_parallel_calib[3].set_flag(inv_flags['ACQ_IS_PARALLEL_CALIBRATION'])
        acq_parallel_calib[3].idx.kspace_encode_step_1 = 1
        acq_parallel_calib[3].idx.kspace_encode_step_2 = 1

        acq_navigation = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_navigation.set_flag(inv_flags['ACQ_IS_NAVIGATION_DATA'])

        acq_phasecor = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_phasecor.set_flag(inv_flags['ACQ_IS_PHASECORR_DATA'])

        acq_rt_feedback = ismrmrd.Acquisition.from_array(np.transpose(np_data, [1, 0]))
        acq_rt_feedback.set_flag(inv_flags['ACQ_IS_RTFEEDBACK_DATA'])

        con_buffer = ismrmrd_tools.ConnectionBuffer(None)
        con_buffer.meas_data.data['ACQ_IS_NOISE_MEASUREMENT'] = [acq_noise]
        con_buffer.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'] = acq_parallel_calib
        con_buffer.meas_data.data['ACQ_IS_NAVIGATION_DATA'] = [acq_navigation]
        con_buffer.meas_data.data['ACQ_IS_PHASECORR_DATA'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_REVERSE_ACQ_IS_PHASECORR_DATA'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_RTFEEDBACK_DATA'] = [acq_rt_feedback]
        con_buffer.meas_data.data['ACQ_IS_REVERSE_ACQ_IS_IMAGING'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_IMAGING_ACQ_IS_REVERSE'] = [acq_phasecor]
        con_buffer.meas_data.data['ACQ_IS_IMAGING'] = [acq_phasecor]

        con_buffer.headers.append(ismrmrd.xsd.CreateFromDocument(ismrmrd_tools.create_dummy_ismrmrd_header()))

        con_buffer.read_close()

        self.assertTrue(con_buffer.is_exhausted)

    def test_ismrmrd_flags_to_bitmask(self):
        """!
        @brief UT which validates the correct functionality of ismrmrd_flags_to_bitmask.
        @details The test validates bitmasks for different sets of ismrmrd flags.

        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: ismrmrd_flags_to_bitmask")
        list_of_flags = ["ACQ_USER1", "ACQ_IS_PHASECORR_DATA", "ACQ_USER2"]
        bitmask = ismrmrd_tools.ismrmrd_flags_to_bitmask(list_of_flags)
        self.assertEqual(bitmask, 216172782122172416)

        list_of_flags = ["ACQ_FIRST_IN_REPETITION", "ACQ_IS_PARALLEL_CALIBRATION", "ACQ_IS_RTFEEDBACK_DATA"]
        bitmask = ismrmrd_tools.ismrmrd_flags_to_bitmask(list_of_flags)
        self.assertEqual(bitmask, 134746112)

    def test_bitmask_to_flags(self):
        """!
        @brief UT which validates the correct functionality of bitmask_to_flags.
        @details The test validates bitmasks for different sets of ismrmrd flags.

        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: bitmask_to_flags")
        bitmask = 216172782122172416
        list_of_flags = ismrmrd_tools.bitmask_to_flags(bitmask)
        self.assertEqual(len(list_of_flags), 3)
        self.assertEqual("ACQ_USER1" in list_of_flags, True)
        self.assertEqual("ACQ_IS_PHASECORR_DATA" in list_of_flags, True)
        self.assertEqual("ACQ_USER2" in list_of_flags, True)

        bitmask = 134746112
        list_of_flags = ismrmrd_tools.bitmask_to_flags(bitmask)
        self.assertEqual(len(list_of_flags), 3)
        self.assertEqual("ACQ_FIRST_IN_REPETITION" in list_of_flags, True)
        self.assertEqual("ACQ_IS_PARALLEL_CALIBRATION" in list_of_flags, True)
        self.assertEqual("ACQ_IS_RTFEEDBACK_DATA" in list_of_flags, True)

    def test_receive_messages(self):
        """!
        @brief UT which validates the correct functionality of receive_messages.
        @details The test first opens an ismrmrd client in a separate thread, which sends different messages.
                 Afterwards, we send defined ismrmrd messages of various types to receive them. Finally, we validate
                 that the order of received messages is correct and that the content of messages is correct, too.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: ConnectionBuffer.receive_messages")
        # We start a client in a separate thread which tries to connect to the server below
        que = Queue()
        server_thread = threading.Thread(
            target=lambda q, arg1, arg2: q.put(ismrmrd_tools.gstar_recon_server(arg1, arg2)),
            args=(que, "localhost", 9002))
        server_thread.start()
        time.sleep(1)

        # Connect a client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 9002))
        client_con = ismrmrd_tools.ConnectionBuffer(client_socket)

        # We send a bunch of messages to the server
        client_con.send_config_file("test_send_message_config_file")  # Config File
        client_con.send_text("test_send_message_text")  # Text

        header = ismrmrd_tools.create_dummy_ismrmrd_header()
        client_con.send_metadata(header)  # Header

        acq = ismrmrd.Acquisition.from_array(np.ones((64, 64)))
        client_con.send_acquisition(acq)  # Acquisition

        image = ismrmrd.Image.from_array(np.ones((64, 64)), acquisition=acq, transpose=False)
        client_con.send_image(image)  # Image

        client_con.send_close()  # Close

        client_con.shutdown_close()
        server_thread.join()

        # We evaluate that the server received and stored the correct messages
        server_con, server_socket = que.get()
        server_con.shutdown_close()
        server_socket.close()

        self.assertEqual(server_con.messages_received[0], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE)
        self.assertEqual(server_con.messages_received[1], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_TEXT)
        self.assertEqual(server_con.messages_received[2], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_HEADER)
        self.assertEqual(server_con.messages_received[3], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_ACQUISITION)
        self.assertEqual(server_con.messages_received[4], ismrmrd_tools.IsmrmrdConstants.ID_MESSAGE_IMAGE)

        self.assertEqual(len(server_con.config_files), 1)
        self.assertEqual(len(server_con.texts), 1)
        self.assertEqual(len(server_con.config_texts), 0)
        self.assertEqual(len(server_con.headers), 1)
        self.assertEqual(len(server_con.meas_data.data['ACQ_IS_IMAGING']), 1)
        self.assertEqual(len(server_con.images), 1)
        self.assertEqual(len(server_con.waveforms), 0)

        self.assertEqual(np.all(server_con.meas_data.data['ACQ_IS_IMAGING'][0].data == 1.0), True)
        self.assertEqual(np.all(server_con.images[0].data == 1.0), True)
        self.assertEqual(server_con.config_files[0], 'test_send_message_config_file')
        self.assertEqual(server_con.texts[0], 'test_send_message_text')
        self.assertEqual(str(server_con.headers[0].Meta),
                         "<class 'ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader.Meta'>")
        self.assertEqual(server_con.headers[0].experimentalConditions.H1resonanceFrequency_Hz,
                         123251770)
        self.assertEqual(server_con.headers[0].measurementInformation.frameOfReferenceUID,
                         '1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0')
        self.assertEqual(server_con.headers[0].measurementInformation.measurementID,
                         '45098_164288619_164288624_176')
        self.assertEqual(server_con.headers[0].measurementInformation.patientPosition.name,
                         'HFS')
        self.assertEqual(server_con.headers[0].measurementInformation.protocolName,
                         'DummyProtocol')

    def test_noise_scan_to_acq(self):
        """!
        @brief UT which validates the correct functionality of noise_scan_to_acq.
        @details The test creates an acquisition object from a numpy array. Afterwards, the correct noise flag
                 in the acquisition object is validated.

        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: noise_scan_to_acq")
        test_np_array = np.ones((128, 20), dtype=complex)
        noise_acq = ismrmrd_tools.noise_scan_to_acq(test_np_array)

        acq_flags = ismrmrd_tools.bitmask_to_flags(noise_acq.getHead().flags)
        self.assertEqual(acq_flags[0], 'ACQ_IS_NOISE_MEASUREMENT')

    def test_gstar_to_ismrmrd_hdr(self):
        """!
        @brief UT which validates the correct functionality of gstar_to_ismrmrd_hdr.
        @details The test first creates several dictionaries which correspond to the dictionaries which would be
                 created by the gammaSTAR framework to extract the relevant parameters from that. Afterwards, the
                 header is populated, and it is validated whether all entries are set correctly. As a second step,
                 we perform the same task, but we provide additional information which contain information about
                 the measurement or subject, which we might obtain from e.g. a full body MR system or tabletop system.

        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: gstar_to_ismrmrd_hdr")

        # PART 1: No additional information available

        # Hardcoded protocol information
        prot = {
            'read_oversampling': 1,
            'phase_oversampling': 1,
            'slice_oversampling': 1,
            'read_partial_fourier': 1.0,
            'phase_partial_fourier': 1.0,
            'slice_partial_fourier': 1.0,
            'PAT_factor_phase': 2,
            'PAT_factor_slice': 2,
            'TR': 2000,
            'TE': 30,
            'TI': 1000,
            'flip_angle': 90,
            'PAT_mode': 'SomeMode'
        }

        # Hardcoded additional information (can be empty if not used)
        info = {}

        # Hardcoded experimental conditions (can be empty if not used)
        expo = {}

        # Hardcoded system information
        sys = {
            'frequency': {'1': 123.0}  # Example frequency in MHz
        }

        # Hardcoded root information
        root = {
            'acq_size': {'1': 256, '2': 256, '3': 1},
            'fov': {'1': 0.2, '2': 0.2, '3': 0.1},  # FOV in meters
            'mat_size': {'1': 256, '2': 256, '3': 1}
        }

        # Now you can use these variables with the function
        xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(prot, info, expo, sys, root)
        ismrmrd_header = ismrmrd.xsd.CreateFromDocument(xml_header)

        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.fieldOfView_mm.x, 200)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.fieldOfView_mm.y, 200)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.fieldOfView_mm.z, 100)

        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.matrixSize.x, 256)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.matrixSize.y, 256)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.matrixSize.z, 1)

        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.matrixSize.x, 256)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.matrixSize.y, 256)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.matrixSize.z, 1)

        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.x, 200)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.y, 200)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.z, 100)

        self.assertEqual(ismrmrd_header.experimentalConditions.H1resonanceFrequency_Hz, 123)

        self.assertEqual(ismrmrd_header.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1, 2)
        self.assertEqual(ismrmrd_header.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2, 2)

        self.assertEqual(ismrmrd_header.sequenceParameters.TE[0], 30000.0)
        self.assertEqual(ismrmrd_header.sequenceParameters.TI[0], 1000000.0)
        self.assertEqual(ismrmrd_header.sequenceParameters.TR[0], 2000000.0)

        # PART 2: We have additional information available
        meas_info = dict()
        meas_info["patientName"] = 'TestPatient'
        meas_info["patientWeight_kg"] = '1000'
        meas_info["patientHeight_m"] = '100000'
        meas_info["patientID"] = 'ABC'
        meas_info["patientBirthdate"] = '1.1.0'
        meas_info["studyDate"] = '00:00:00'
        meas_info["studyTime"] = '00:00:00'
        meas_info["studyDescription"] = 'TestStudy'
        meas_info["bodyPartExamined"] = 'Brain'
        meas_info["measurementID"] = '0.0.0'
        meas_info["patientPosition"] = 'HFS'
        meas_info["protocolName"] = 'MRPY_Recon_Tools'
        meas_info["frameOfReferenceUID"] = '0.0.0.0'
        meas_info["seriesDescription"] = 'MRPY_Recon_Tools'
        meas_info["systemVendor"] = 'Python'
        meas_info["systemModel"] = 'Numpy'
        meas_info["systemFieldStrength_T"] = '3.0'
        meas_info["institutionName"] = 'Mevis'
        meas_info["H1resonanceFrequency_Hz"] = '1234'

        xml_header_part2 = ismrmrd_tools.gstar_to_ismrmrd_hdr(prot, info, expo, sys, root, meas_info)
        ismrmrd_header_part2 = ismrmrd.xsd.CreateFromDocument(xml_header_part2)

        self.assertEqual(ismrmrd_header_part2.encoding[0].encodedSpace.fieldOfView_mm.x, 200)
        self.assertEqual(ismrmrd_header_part2.encoding[0].encodedSpace.fieldOfView_mm.y, 200)
        self.assertEqual(ismrmrd_header_part2.encoding[0].encodedSpace.fieldOfView_mm.z, 100)
        self.assertEqual(ismrmrd_header_part2.encoding[0].encodedSpace.matrixSize.x, 256)
        self.assertEqual(ismrmrd_header_part2.encoding[0].encodedSpace.matrixSize.y, 256)
        self.assertEqual(ismrmrd_header_part2.encoding[0].encodedSpace.matrixSize.z, 1)
        self.assertEqual(ismrmrd_header_part2.encoding[0].reconSpace.matrixSize.x, 256)
        self.assertEqual(ismrmrd_header_part2.encoding[0].reconSpace.matrixSize.y, 256)
        self.assertEqual(ismrmrd_header_part2.encoding[0].reconSpace.matrixSize.z, 1)
        self.assertEqual(ismrmrd_header_part2.encoding[0].reconSpace.fieldOfView_mm.x, 200)
        self.assertEqual(ismrmrd_header_part2.encoding[0].reconSpace.fieldOfView_mm.y, 200)
        self.assertEqual(ismrmrd_header_part2.encoding[0].reconSpace.fieldOfView_mm.z, 100)
        self.assertEqual(ismrmrd_header_part2.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1, 2)
        self.assertEqual(ismrmrd_header_part2.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2, 2)
        self.assertEqual(ismrmrd_header_part2.sequenceParameters.TE[0], 30000.0)
        self.assertEqual(ismrmrd_header_part2.sequenceParameters.TI[0], 1000000.0)
        self.assertEqual(ismrmrd_header_part2.sequenceParameters.TR[0], 2000000.0)

        # Additional fields
        self.assertEqual(ismrmrd_header_part2.experimentalConditions.H1resonanceFrequency_Hz, 1234)
        self.assertEqual(ismrmrd_header_part2.subjectInformation.patientName, 'TestPatient')
        self.assertEqual(ismrmrd_header_part2.subjectInformation.patientBirthdate, '1.1.0')
        self.assertEqual(ismrmrd_header_part2.subjectInformation.patientHeight_m, 100000.0)
        self.assertEqual(ismrmrd_header_part2.subjectInformation.patientWeight_kg, 1000.0)
        self.assertEqual(ismrmrd_header_part2.subjectInformation.patientID, 'ABC')
        self.assertEqual(ismrmrd_header_part2.studyInformation.studyDate, '00:00:00')
        self.assertEqual(ismrmrd_header_part2.studyInformation.studyDescription, 'TestStudy')
        self.assertEqual(ismrmrd_header_part2.studyInformation.bodyPartExamined, 'Brain')
        self.assertEqual(ismrmrd_header_part2.measurementInformation.frameOfReferenceUID, '0.0.0.0')
        self.assertEqual(ismrmrd_header_part2.measurementInformation.measurementID, '0.0.0')
        self.assertEqual(ismrmrd_header_part2.measurementInformation.patientPosition.name, 'HFS')
        self.assertEqual(ismrmrd_header_part2.measurementInformation.patientPosition.value, 'HFS')
        self.assertEqual(ismrmrd_header_part2.measurementInformation.protocolName, 'MRPY_Recon_Tools')
        self.assertEqual(ismrmrd_header_part2.measurementInformation.seriesDescription, 'MRPY_Recon_Tools')
        self.assertEqual(ismrmrd_header_part2.acquisitionSystemInformation.systemModel, 'Numpy')
        self.assertEqual(ismrmrd_header_part2.acquisitionSystemInformation.systemVendor, 'Python')
        self.assertEqual(ismrmrd_header_part2.acquisitionSystemInformation.systemFieldStrength_T, 3.0)
        self.assertEqual(ismrmrd_header_part2.acquisitionSystemInformation.institutionName, 'Mevis')

    def test_twix_hdr_to_ismrmrd_hdr(self):
        """!
        @brief UT which validates the correct functionality of twix_hdr_to_ismrmrd_hdr.
        @details The test first creates a hard coded structure, containing all entries which would be present in an
                 hdr header if data was loaded using pymapvbvd. Afterwards, the ismrmrd header is populated and it is
                 validated whether all entries are set correctly.

        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: twix_hdr_to_ismrmrd_hdr")

        # Hardcoded protocol information
        twix_hdr = {
            'Config': {
                'tPatientName': 'John Doe',
                'PatientID': '123456',
                'PatientBirthDay': '1980-01-01',
                'PatientPosition': 'HFS',
                'NImageCols': 256,
                'NLinMeas': 256,
                'NParMeas': 1,
                'FrameOfReference': '1.2.3.4.5'
            },
            'Dicom': {
                'flUsedPatientWeight': 70.0,
                'tStudyDescription': 'Test Study',
                'tBodyPartExamined': 'Brain',
                'Manufacturer': 'ManufacturerName',
                'ManufacturersModelName': 'ModelName',
                'lFrequency': 123.0,  # Frequency in Hz
                'adFlipAngleDegree': 90.0,
                'dThickness': 5.0,
                'lPhaseEncodingLines': 256
            },
            'Meas': {
                'Study': 'Study1',
                'tProtocolName': 'Protocol1',
                'InstitutionName': 'InstitutionName',
                'PrepareTimestamp': '2023-10-01 12:00:00',
                'flPatientHeight': 180.0,
                'ucRefScanMode': 4.0
            },
            'MeasYaps': {
                ('sProtConsistencyInfo', 'flNominalB0'): 3.0  # Field strength in Tesla
            },
            'Protocol': {
                'RoFOV': 200.0,  # Readout FOV in mm
                'PeFOV': 200.0,  # Phase encoding FOV in mm
                'alTR': 2000,  # Repetition time in ms
                'alTE': 30,  # Echo time in ms
                'alTI': 30,  # Inversion time in ms
                'ProdAccelFactorPE': 2,
                'ProdAccelFactor3D': 2
            }
        }

        # Now you can use this variable with the function
        xml_header = ismrmrd_tools.twix_hdr_to_ismrmrd_hdr(twix_hdr)
        ismrmrd_header = ismrmrd.xsd.CreateFromDocument(xml_header)

        self.assertEqual(ismrmrd_header.acquisitionSystemInformation.systemFieldStrength_T, 3.0)
        self.assertEqual(ismrmrd_header.acquisitionSystemInformation.systemFieldStrength_T, 3.0)

        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.fieldOfView_mm.x, 200)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.fieldOfView_mm.y, 200)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.fieldOfView_mm.z, 0.0)

        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.matrixSize.x, 256)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.matrixSize.y, 256)
        self.assertEqual(ismrmrd_header.encoding[0].encodedSpace.matrixSize.z, 1)

        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.matrixSize.x, 256)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.matrixSize.y, 256)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.matrixSize.z, 1)

        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.x, 200)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.y, 200)
        self.assertEqual(ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.z, 5.0)

        self.assertEqual(ismrmrd_header.experimentalConditions.H1resonanceFrequency_Hz, 123)

        self.assertEqual(ismrmrd_header.measurementInformation.frameOfReferenceUID, '1.2.3.4.5')
        self.assertEqual(ismrmrd_header.measurementInformation.measurementID, 'Study1')
        self.assertEqual(ismrmrd_header.measurementInformation.protocolName, 'Protocol1')

        self.assertEqual(ismrmrd_header.sequenceParameters.TE[0], 30.0)
        self.assertEqual(ismrmrd_header.sequenceParameters.TI[0], 30.0)
        self.assertEqual(ismrmrd_header.sequenceParameters.TR[0], 2000.0)
        self.assertEqual(ismrmrd_header.sequenceParameters.flipAngle_deg[0], 90.0)

        self.assertEqual(ismrmrd_header.studyInformation.bodyPartExamined, 'Brain')
        self.assertEqual(ismrmrd_header.studyInformation.studyDescription, 'Test Study')

        self.assertEqual(ismrmrd_header.subjectInformation.patientHeight_m, 180.0)
        self.assertEqual(ismrmrd_header.subjectInformation.patientID, '123456')
        self.assertEqual(ismrmrd_header.subjectInformation.patientName, 'John Doe')
        self.assertEqual(ismrmrd_header.subjectInformation.patientWeight_kg, 70.0)

    def test_identify_readout_type_from_acqs(self):
        """!
        @brief UT which validates the correct functionality of identify_readout_type_from_acqs.
        @details This test validates the functionality of identify_readout_type_from_acqs in ismrmrd_tools. It creates
                 various synthetic k-space datasets and trajectories to simulate different MR acquisition types
                 (Cartesian, non-Cartesian 2D/3D, ramp-sampled, and PROPELLER). For each scenario, it checks that the
                 function correctly identifies the readout type, whether ramp sampling or PROPELLER is present, and the
                 blade dimension, ensuring robust detection of acquisition geometry from ISMRMRD acquisitions.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: identify_readout_type_from_acqs")

        # Part 1: Trajectory info is missing -> Assume Cartesian readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position)
        readout_type, is_rampsamp, is_propeller, blade_dim = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)
        self.assertEqual(readout_type, ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN)
        self.assertEqual(is_rampsamp, False)
        self.assertEqual(is_propeller, False)
        self.assertEqual(blade_dim, -1)

        # Part 2: Cartesian trajectory info -> Cartesian readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    ksp_imaging_traj[i, :, j, k, 0] = i
                    ksp_imaging_traj[i, :, j, k, 1] = j
                    ksp_imaging_traj[i, :, j, k, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)
        readout_type, is_rampsamp, is_propeller, blade_dim = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)
        self.assertEqual(readout_type, ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN)
        self.assertEqual(is_rampsamp, False)
        self.assertEqual(is_propeller, False)
        self.assertEqual(blade_dim, -1)

        # Part 3: Cartesian trajectory info with different step size -> Cartesian readout with ramps
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    if 16 > i or i > 108:
                        ksp_imaging_traj[i, :, j, k, 0] = i*0.5
                    else:
                        ksp_imaging_traj[i, :, j, k, 0] = i
                    ksp_imaging_traj[i, :, j, k, 1] = j
                    ksp_imaging_traj[i, :, j, k, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)
        readout_type, is_rampsamp, is_propeller, blade_dim = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)
        self.assertEqual(readout_type, ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN)
        self.assertEqual(is_rampsamp, True)
        self.assertEqual(is_propeller, False)
        self.assertEqual(blade_dim, -1)

        # Part 4: Non-Cartesian trajectory 2D -> Non-Cartesian 2D readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    ksp_imaging_traj[i, :, j, k, 0] = i*math.sin(180.0/3*j)
                    ksp_imaging_traj[i, :, j, k, 1] = i*math.cos(180.0/3*j)
                    ksp_imaging_traj[i, :, j, k, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)
        readout_type, is_rampsamp, is_propeller, blade_dim = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)
        self.assertEqual(readout_type, ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D)
        self.assertEqual(is_rampsamp, False)
        self.assertEqual(is_propeller, False)
        self.assertEqual(blade_dim, -1)

        # Part 5: Non-Cartesian trajectory 3D -> Non-Cartesian 3D readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    ksp_imaging_traj[i, :, j, k, 0] = i * math.sin(180.0 / 3.0 * j)
                    ksp_imaging_traj[i, :, j, k, 1] = i * math.cos(180.0 / 3.0 * j)
                    ksp_imaging_traj[i, :, j, k, 2] = i
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)
        readout_type, is_rampsamp, is_propeller, blade_dim = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)
        self.assertEqual(readout_type, ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D)
        self.assertEqual(is_rampsamp, False)
        self.assertEqual(is_propeller, False)
        self.assertEqual(blade_dim, -1)

        # Part 6: PROPELLER trajectory -> Cartesian Readout with propeller
        ksp_imaging = np.ones((128, 3, 3, 16, 1, 10), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 1, 10, 3))
        for b in range(10):
            for i in range(128):
                for j in range(3):
                    for k in range(16):
                        ksp_imaging_traj[i, :, j, k, 0, b, 0] = i * math.sin(180.0 / 10.0 * b)
                        ksp_imaging_traj[i, :, j, k, 0, b, 1] = j * math.cos(180.0 / 10.0 * b)
                        ksp_imaging_traj[i, :, j, k, 0, b, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)
        readout_type, is_rampsamp, is_propeller, blade_dim = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)
        self.assertEqual(readout_type, ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN)
        self.assertEqual(is_rampsamp, False)
        self.assertEqual(is_propeller, True)
        self.assertEqual(blade_dim, 'SET')

    def test_ismrmrd_acqs_to_numpy_array(self):
        """!
        @brief UT which validates the correct functionality of ismrmrd_acqs_to_numpy_array.
        @details The selected code tests the ismrmrd_acqs_to_numpy_array function from ismrmrd_tools. It verifies that
                 ISMRMRD acquisition objects created from various synthetic k-space datasets (with and without
                 trajectory information, including Cartesian, ramp-sampled, and non-Cartesian 2D/3D cases) can be
                 accurately converted back to their original or expected numpy array representations. The test checks
                 both data integrity and output shapes, ensuring correct handling of different MR acquisition
                 geometries.
        @param self: Reference to object

        @author: Jörn Huber
        """
        print("\nTesting mrpy_ismrmrd_tools: ismrmrd_acqs_to_numpy_array")

        # Part 1: Trajectory info is missing -> Assume Cartesian readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position)

        # We use the acquisition object to recreate the numpy array
        ksp_imaging_recreated = np.squeeze(ismrmrd_tools.ismrmrd_acqs_to_numpy_array(list_of_acqs))
        self.assertEqual(np.all(ksp_imaging_recreated == ksp_imaging), True)

        # Part 2: Cartesian trajectory info -> Cartesian readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    ksp_imaging_traj[i, :, j, k, 0] = i
                    ksp_imaging_traj[i, :, j, k, 1] = j
                    ksp_imaging_traj[i, :, j, k, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)

        # We use the acquisition object to recreate the numpy array
        ksp_imaging_recreated = np.squeeze(ismrmrd_tools.ismrmrd_acqs_to_numpy_array(list_of_acqs))
        self.assertEqual(np.all(ksp_imaging_recreated == ksp_imaging), True)

        # Part 3: Cartesian trajectory info with different step size -> Cartesian readout with ramps
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        step_size = 1.0
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    if 16 > i or i > 108:
                        step_size = 0.5
                    else:
                        step_size = 1.0
                    if i == 0:
                        ksp_imaging_traj[i, :, j, k, 0] = 0.0
                    else:
                        ksp_imaging_traj[i, :, j, k, 0] = ksp_imaging_traj[i-1, :, j, k, 0] + step_size
                    ksp_imaging_traj[i, :, j, k, 1] = j
                    ksp_imaging_traj[i, :, j, k, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)

        # We use the acquisition object to recreate the numpy array
        ksp_imaging_ramp_sampled = np.squeeze(ismrmrd_tools.ismrmrd_acqs_to_numpy_array(list_of_acqs))
        diff_max = np.max(np.abs(ksp_imaging_ramp_sampled - ksp_imaging).flatten())
        self.assertLess(diff_max, 10**-5)

        # Part 4: Non-Cartesian trajectory 2D -> Non-Cartesian 2D readout
        ksp_imaging = np.ones((128, 3, 3, 16), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 16, 3))
        for i in range(128):
            for j in range(3):
                for k in range(16):
                    ksp_imaging_traj[i, :, j, k, 0] = i * math.sin(180.0 / 3 * j)
                    ksp_imaging_traj[i, :, j, k, 1] = i * math.cos(180.0 / 3 * j)
                    ksp_imaging_traj[i, :, j, k, 2] = k
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)

        class MatrixSize:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class EncodedSpace:
            def __init__(self, x, y, z):
                self.matrixSize = MatrixSize(x, y, z)

        encoded_space = EncodedSpace(x=128, y=3, z=16)
        ksp_imaging_regrid = np.squeeze(ismrmrd_tools.ismrmrd_acqs_to_numpy_array(list_of_acqs, encoded_space))
        self.assertEqual(ksp_imaging_regrid.shape, (128, 3, 128, 16))

        # Part 5: Non-Cartesian trajectory 3D -> Non-Cartesian 3D readout
        ksp_imaging = np.ones((128, 3, 3, 1), dtype=complex)
        ksp_imaging_traj = np.zeros((128, 3, 3, 1, 3))
        for i in range(128):
            for j in range(3):
                ksp_imaging_traj[i, :, j, 0, 0] = i * math.sin(180.0 / 3.0 * j)
                ksp_imaging_traj[i, :, j, 0, 1] = i * math.cos(180.0 / 3.0 * j)
                ksp_imaging_traj[i, :, j, 0, 2] = i
        imaging_flags = ["ACQ_USER1"]
        list_of_data = [ksp_imaging]
        list_of_flags = [imaging_flags]
        list_of_trajs = [ksp_imaging_traj]

        read_dir = np.zeros(3)
        read_dir[0] = 1
        read_dir[1] = 0
        read_dir[2] = 0
        phase_dir = np.zeros(3)
        phase_dir[0] = 0
        phase_dir[1] = 1
        phase_dir[2] = 0
        slice_dir = np.zeros(3)
        slice_dir[0] = 0
        slice_dir[1] = 0
        slice_dir[2] = 1
        position = np.zeros(3)
        position[0] = -1
        position[1] = -2
        position[2] = -3

        list_of_acqs = ismrmrd_tools.numpy_array_to_ismrmrd_acqs(list_of_data,
                                                                 list_of_flags,
                                                                 read_dir,
                                                                 phase_dir,
                                                                 slice_dir,
                                                                 position,
                                                                 list_of_trajs)

        encoded_space = EncodedSpace(x=128, y=3, z=16)
        ksp_imaging_regrid = np.squeeze(ismrmrd_tools.ismrmrd_acqs_to_numpy_array(list_of_acqs, encoded_space))
        self.assertEqual(ksp_imaging_regrid.shape, (128, 3, 128, 128))


if __name__ == '__main__':
    print("---Running unit tests for mrpy_ismrmrd_tools---")
    unittest.main()
