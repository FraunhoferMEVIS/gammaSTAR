"""!
@brief Collection of tools which provide functionality to handle ismrmrd data and streaming protocols.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import statistics
import math
import struct
import socket
import ismrmrd
import numpy as np
import time
import xml.etree.ElementTree as xml_elem_tree
import threading
import logging
import random
from datetime import datetime
import mrinufft
from mrinufft.density import voronoi
import mrpy_coil_tools as coil_tools
import mrpy_helpers as helpers
from scipy.interpolate import interp1d


class MeasIDX:
    """!
    @brief Class which capsules idx counters.
    """

    def __init__(self, repetition, contrast, phase, set, slice):
        self.repetition = repetition
        self.contrast = contrast
        self.phase = phase
        self.set = set
        self.slice = slice


class IsmrmrdConstants:
    """!
    @brief Class which capsules some ismrmrd related constants.
    """

    SIZEOF_MRD_MESSAGE_IDENTIFIER = 2
    SIZEOF_CONFIG_FILE_NAME = 1024
    SIZEOF_LENGTH = 4

    ID_MESSAGE_CONFIG_FILE = 1
    ID_MESSAGE_CONFIG_TEXT = 2
    ID_MESSAGE_HEADER = 3
    ID_MESSAGE_CLOSE = 4
    ID_MESSAGE_TEXT = 5
    ID_MESSAGE_ACQUISITION = 1008
    ID_MESSAGE_IMAGE = 1022
    ID_MESSAGE_WAVEFORM = 1026

    READOUT_TYPE_CARTESIAN = 1
    READOUT_TYPE_NONCARTESIAN_2D = 3
    READOUT_TYPE_NONCARTESIAN_3D = 4
    READOUT_TYPE_PROPELLER_2D = 5
    READOUT_TYPE_PROPELLER_3D = 6

    IDX_MAP = {
        "COL": 0,
        "CHA": 1,
        "PE1": 2,
        "PE2": 3,
        "SLC": 4,
        "SET": 5,
        "PHS": 6,
        "CON": 7,
        "REP": 8,
        "AVE": 9,
        "SEG": 10,
    }

    ISMRMRD_ACQ_FLAGS = {
        1: "ACQ_FIRST_IN_ENCODE_STEP1",
        2: "ACQ_LAST_IN_ENCODE_STEP1",
        3: "ACQ_FIRST_IN_ENCODE_STEP2",
        4: "ACQ_LAST_IN_ENCODE_STEP2",
        5: "ACQ_FIRST_IN_AVERAGE",
        6: "ACQ_LAST_IN_AVERAGE",
        7: "ACQ_FIRST_IN_SLICE",
        8: "ACQ_LAST_IN_SLICE",
        9: "ACQ_FIRST_IN_CONTRAST",
        10: "ACQ_LAST_IN_CONTRAST",
        11: "ACQ_FIRST_IN_PHASE",
        12: "ACQ_LAST_IN_PHASE",
        13: "ACQ_FIRST_IN_REPETITION",
        14: "ACQ_LAST_IN_REPETITION",
        15: "ACQ_FIRST_IN_SET",
        16: "ACQ_LAST_IN_SET",
        17: "ACQ_FIRST_IN_SEGMENT",
        18: "ACQ_LAST_IN_SEGMENT",
        19: "ACQ_IS_NOISE_MEASUREMENT",
        20: "ACQ_IS_PARALLEL_CALIBRATION",
        21: "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",
        22: "ACQ_IS_REVERSE",
        23: "ACQ_IS_NAVIGATION_DATA",
        24: "ACQ_IS_PHASECORR_DATA",
        25: "ACQ_LAST_IN_MEASUREMENT",
        26: "ACQ_IS_HPFEEDBACK_DATA",
        27: "ACQ_IS_DUMMYSCAN_DATA",
        28: "ACQ_IS_RTFEEDBACK_DATA",
        29: "ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA",
        30: "ISMRMRD_ACQ_IS_PHASE_STABILIZATION_REFERENCE",
        31: "ISMRMRD_ACQ_IS_PHASE_STABILIZATION",
        53: "ISMRMRD_ACQ_COMPRESSION1",
        54: "ISMRMRD_ACQ_COMPRESSION2",
        55: "ISMRMRD_ACQ_COMPRESSION3",
        56: "ISMRMRD_ACQ_COMPRESSION4",
        57: "ACQ_USER1",
        58: "ACQ_USER2",
        59: "ACQ_USER3",
        60: "ACQ_USER4",
        61: "ACQ_USER5",
        62: "ACQ_USER6",
        63: "ACQ_USER7",
        64: "ACQ_USER8"
    }


class MeasData:
    """!
    @brief Class which measurement data and information
    """

    def __call__(self, data_key, dim_str):
        return self.data[data_key].shape[IsmrmrdConstants.IDX_MAP[dim_str]]

    def __init__(self):

        ## Dictionary, which contains different sub dictionaries which store the received data
        self.data = dict()

        ## ISMRMRD header which holds information about the measurement
        self.meas_header = None

        ## Derived info from header about encoded space for easy access
        self.encoded_space = None

        ## Derived info from header about reconstructed space for easy access
        self.recon_space = None

        ## Derived info from header about partial fourier factor 1 for easy access
        self.pf_factor_pe1 = None

        ## Derived info from header about partial fourier factor 2 for easy access
        self.pf_factor_pe2 = None

        ## Derived info from header about acceleration factor 1 for easy access
        self.accel_pe1 = None

        ## Derived info from header about acceleration factor 2 for easy access
        self.accel_pe2 = None

        ## Numpy array which contains the noise correlation matrix
        self.noise_corr = None

        ## Noise whitening matrix W
        self.W = None

        ## Indicates whether readout direction was already Fourier transformed
        self.is_ro_ft = False

        ## Indicates whether phase-encoding direction was already Fourier transformed
        self.is_pe_ft = False

        ## Indicates whether partition encoding direction was already Fourier transformed
        self.is_par_ft = False

        ## Indicates starting and end indices of the parallel calibration pe1 region
        self.calib_reg_pe1 = (-1, -1)

        ## Indicates starting and end indices of the parallel calibration pe2 region
        self.calib_reg_pe2 = (-1, -1)

        ## Holds information about the imaging readout type, e.g. 2D/3D Non-Cartesian etc.
        self.imaging_readout_type = -1

        ## Indicates whether Cartesian PROPELLER trajectory is applied
        self.is_propeller = False

        ## Indicates the PROPELLER blade dimension
        self.blade_dim = ''


class ConnectionBuffer:
    """!
    @brief Class which capsules all message related handling for the server and client side.
    """

    def __init__(self, socket):
        """!
        @brief Initilization method.

        @param socket: A connected socket object.

        @author Kelvin Chow, Jörn Huber
        """
        ## The socket object
        self.socket = socket

        ## The measurement data object, which handles received acquisitions
        self.meas_data = MeasData()

        ## Indicates whether the connection is exhausted.
        self.is_exhausted = False

        ## List which contains the identifiers of received messages.
        self.messages_received = []

        ## List which contains the received config files
        self.config_files = []

        ## List which contains the received config texts
        self.config_texts = []

        ## List which contains the received headers/metadata
        self.headers = []

        ## List which contains the received texts
        self.texts = []

        ## List which contains the received images.
        self.images = []

        ## List which contains the received waveforms.
        self.waveforms = []

        ## Counts the number of sent acquisitions.
        self.sentAcqs       = 0

        ## Counts the number of sent images.
        self.sentImages     = 0

        ## Counts the number of sent waveforms.
        self.sentWaveforms  = 0

        ## Counts the number of received acquisitions.
        self.recvAcqs       = 0

        ## Counts the number of received images.
        self.recvImages     = 0

        ## Counts the number of received waveforms.
        self.recvWaveforms  = 0

        ## Helper, locking in threading environment
        self.lock           = threading.Lock()

        ## Handlers
        self.handlers       = {
            IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE:         self.read_config_file,
            IsmrmrdConstants.ID_MESSAGE_CONFIG_TEXT:         self.read_config_text,
            IsmrmrdConstants.ID_MESSAGE_HEADER:              self.read_metadata,
            IsmrmrdConstants.ID_MESSAGE_CLOSE:               self.read_close,
            IsmrmrdConstants.ID_MESSAGE_TEXT:                self.read_text,
            IsmrmrdConstants.ID_MESSAGE_ACQUISITION: self.read_acquisition,
            IsmrmrdConstants.ID_MESSAGE_WAVEFORM:    self.read_waveform,
            IsmrmrdConstants.ID_MESSAGE_IMAGE:       self.read_image
        }

    def convert_acquisitions(self):
        """!
        @brief Converts list of ismrmrd acquisition objects to numpy arrays.
        @details Buffered data is handled depending on the type of readout. If non-cartesian trajectories were applied,
                 the data is gridded to a Cartesian matrix first. If ramp sampling was applied, indiviudal readouts
                 are regridded first.

        @author Jörn Huber
        """
        b_is_process = any('ACQ' in acq_key for acq_key in self.meas_data.data.keys()) and len(self.headers) > 0

        if b_is_process:
            logging.info("GSTAR Recon: Processing acquisitions")

            self.meas_data.imaging_readout_type, _, self.meas_data.is_propeller, self.meas_data.blade_dim = identify_readout_type_from_acqs(self.meas_data.data['ACQ_IS_IMAGING'])

            if self.meas_data.imaging_readout_type == IsmrmrdConstants.READOUT_TYPE_CARTESIAN and not self.meas_data.is_propeller:

                self.meas_data.encoded_space = self.headers[0].encoding[0].encodedSpace
                self.meas_data.recon_space = self.headers[0].encoding[0].reconSpace
                self.meas_data.pf_factor_pe1 = self.meas_data.encoded_space.matrixSize.y / self.meas_data.recon_space.matrixSize.y
                self.meas_data.pf_factor_pe2 = self.meas_data.encoded_space.matrixSize.z / self.meas_data.recon_space.matrixSize.z
                self.meas_data.accel_pe1 = self.headers[0].encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
                self.meas_data.accel_pe2 = self.headers[0].encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2

            else:

                self.meas_data.encoded_space = self.headers[0].encoding[0].encodedSpace
                self.meas_data.recon_space = self.headers[0].encoding[0].reconSpace
                self.meas_data.pf_factor_pe1 = 1.0 # No Partial Fourier for non-Cartesian or PROPELLER data
                self.meas_data.pf_factor_pe2 = 1.0 # No Partial Fourier for non-Cartesian or PROPELLER data
                self.meas_data.accel_pe1 = self.headers[0].encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
                self.meas_data.accel_pe2 = self.headers[0].encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2

            os_factor = self.meas_data.encoded_space.fieldOfView_mm.x / self.meas_data.recon_space.fieldOfView_mm.x #== 2 and encoded_space.matrixSize.x / recon_space.matrixSize.x == 2

            self.meas_data.meas_header = self.headers[0]
            logging.info("GSTAR Recon:  Partial Fourier PE1xPE2: " + str(self.meas_data.pf_factor_pe1) + "x" + str(self.meas_data.pf_factor_pe2))
            logging.info("GSTAR Recon:  Parallel Imaging PE1xPE2 " + str(self.meas_data.accel_pe1) + "x" + str(self.meas_data.accel_pe2))

            acq_keys = list(self.meas_data.data.keys())
            for acq_key in acq_keys:
                np_key = acq_key.replace('ACQ', 'NP')
                logging.info('GSTAR Recon:  Checking readout type of ' + acq_key + " data")
                if "IS_IMAGING" in np_key:
                    self.meas_data.data[np_key] = ismrmrd_acqs_to_numpy_array(self.meas_data.data[acq_key],
                                                                              self.meas_data.encoded_space,
                                                                              self.meas_data.recon_space,
                                                                              self.meas_data.W,
                                                                              os_factor)
                else:
                    self.meas_data.data[np_key] = ismrmrd_acqs_to_numpy_array(self.meas_data.data[acq_key],
                                                                              None,
                                                                              None,
                                                                              self.meas_data.W,
                                                                              os_factor)

            if 'NP_IS_PARALLEL_CALIBRATION' in self.meas_data.data:
                self.meas_data.calib_reg_pe1 = (
                    min(self.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1,
                    max(self.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1 + 1)

                self.meas_data.calib_reg_pe2 = (
                    min(self.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2,
                    max(self.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2 + 1)

    def receive_messages(self):
        """!
        @brief Using an established tcp connection, this function receives all incoming messages and handles appropriate
               deserialization.

        @param con: (ConnectionBuffer) ConnectionBuffer object which is used to store incoming messages.

        @author Jörn Huber
        """

        while True:

            # Read the 2-byte message ID and deserialize it
            message_id = self.read_mrd_message_identifier()

            # Handle the type of message based on the previously extracted ID
            if message_id == IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE:
                self.read_config_file()

            elif message_id == IsmrmrdConstants.ID_MESSAGE_CONFIG_TEXT:
                self.read_config_text()

            elif message_id == IsmrmrdConstants.ID_MESSAGE_HEADER:
                self.read_metadata()

            elif message_id == IsmrmrdConstants.ID_MESSAGE_TEXT:
                self.read_text()

            elif message_id == IsmrmrdConstants.ID_MESSAGE_ACQUISITION:
                self.read_acquisition()

            elif message_id == IsmrmrdConstants.ID_MESSAGE_IMAGE:
                self.read_image()

            elif message_id == IsmrmrdConstants.ID_MESSAGE_CLOSE:
                self.read_close()
                self.convert_acquisitions()
                break

    # --> Based on python-ismrmrd-server (see third_party_licenses.txt)
    def read(self, nbytes):
        """!
        @brief Reads a defined amount of bytes from the stream.

        @param nbytes: (int) Number of bytes to be read from the stream.

        @author Kelvin Chow, Jörn Huber
        """
        return self.socket.recv(nbytes, socket.MSG_WAITALL)

    def shutdown_close(self):
        """!
        @brief Close the socket connection.

        @author Kelvin Chow, Jörn Huber
        """
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        self.socket.close()
        logging.info("Socket closed")

    def read_mrd_message_identifier(self):
        """!
        @brief Reads the 22 bytes message identifier from the streamthe stream.

        @return (int) The ISMRMRD message identifier.

        @author Kelvin Chow, Jörn Huber
        """
        try:
            identifier_bytes = self.read(IsmrmrdConstants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            logging.error("Connection closed unexpectedly")
            self.is_exhausted = True
            return

        if (len(identifier_bytes) == 0):
            self.is_exhausted = True
            return

        ident = struct.unpack('<H', identifier_bytes)[0]

        self.messages_received.append(ident)
        return ident

    def send_config_file(self, filename):
        """!
        @brief Send a config file name text message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_CONFIG_FILE (1) ----------------------------------------
                 This message contains the file name of a configuration file used for
                 image reconstruction/post-processing.  The file must exist on the server.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Config file name (1024 bytes, char          )

        @param filename: (string) The text string containing the name of the config file

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CONFIG_FILE (1)")
            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_CONFIG_FILE))
            self.socket.send(struct.Struct('<1024s').pack(filename.encode()))

    def read_config_file(self):
        """!
        @brief Reads an incoming configuration file name from stream.
        @details ----- MRD_MESSAGE_CONFIG_FILE (1) ----------------------------------------
                 This message contains the file name of a configuration file used for
                 image reconstruction/post-processing.  The file must exist on the server.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Config file name (1024 bytes, char          )

        @author Kelvin Chow, Jörn Huber
        """
        logging.info("<-- Received MRD_MESSAGE_CONFIG_FILE (1)")
        config_file_bytes = self.read(IsmrmrdConstants.SIZEOF_CONFIG_FILE_NAME)
        config_file = struct.unpack('<1024s', config_file_bytes)[0].split(b'\x00', 1)[0].decode('utf-8')
        self.config_files.append(config_file)
        logging.info("<--  " + str(config_file))

    def send_config_text(self, contents):
        """!
        @brief Sends a configuration text file to the stream.
        @details ----- MRD_MESSAGE_CONFIG_TEXT (2) --------------------------------------
                 This message contains the configuration information (text contents) used
                 for image reconstruction/post-processing.  Text is null-terminated.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Length           (   4 bytes, uint32_t      )
                 Config text data (  variable, char          )

        @param contents: (string) Text string which is packed and sent to the stream.

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CONFIG_TEXT (2)")
            contents_with_nul = '%s\0' % contents  # Add null terminator
            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_CONFIG_TEXT))
            self.socket.send(struct.Struct('<I').pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())

    def read_config_text(self):
        """!
        @brief Reads an incoming configuration text file from stream.
        @details ----- MRD_MESSAGE_CONFIG_TEXT (2) --------------------------------------
                 This message contains the configuration information (text contents) used
                 for image reconstruction/post-processing.  Text is null-terminated.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Length           (   4 bytes, uint32_t      )
                 Config text data (  variable, char          )

        @author Kelvin Chow, Jörn Huber
        """
        logging.info("<-- Received MRD_MESSAGE_CONFIG_TEXT (2)")
        length_bytes = self.read(IsmrmrdConstants.SIZEOF_LENGTH)
        length = struct.unpack('<I', length_bytes)[0]
        config_text = self.read(length)
        config_text = config_text.split(b'\x00',1)[0].decode('utf-8')  # Strip off null teminator
        self.config_texts.append(config_text)

    def send_metadata(self, contents):
        """!
        @brief Send a header text message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_METADATA_XML_TEXT (3) -----------------------------------
                 This message contains the metadata for the entire dataset, formatted as
                 MRD XML flexible data header text.  Text is null-terminated.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Length           (   4 bytes, uint32_t      )
                 Text xml data    (  variable, char          )

        @param contents: (string) The xml encoded text string containing header entries which shall be sent
               to the stream

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_METADATA_XML_TEXT (3)")
            contents_with_nul = '%s\0' % contents  # Add null terminator
            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_HEADER))
            self.socket.send(struct.Struct('<I').pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())

    def read_metadata(self):
        """!
        @brief Reads metadata header information from the stream.
        @details ----- MRD_MESSAGE_METADATA_XML_TEXT (3) -----------------------------------
                 This message contains the metadata for the entire dataset, formatted as
                 MRD XML flexible data header text.  Text is null-terminated.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Length           (   4 bytes, uint32_t      )
                 Text xml data    (  variable, char          )

        @author Kelvin Chow, Jörn Huber
        """
        logging.info("<-- Received MRD_MESSAGE_METADATA_XML_TEXT (3)")
        length_bytes = self.read(IsmrmrdConstants.SIZEOF_LENGTH)
        length = struct.unpack('<I', length_bytes)[0]
        metadata = self.read(length)
        metadata = metadata.split(b'\x00',1)[0].decode('utf-8')  # Strip off null teminator
        try:
            header = ismrmrd.xsd.CreateFromDocument(metadata)
            self.headers.append(header)
        except:
            logging.warning("Could not deserialize header, maybe the encoding is not valid:\n" + metadata)

    def send_close(self):
        """!
        @brief Send a close message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_CLOSE (4) ----------------------------------------------
                 This message signals that all data has been sent (either from server or client).

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CLOSE (4)")
            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_CLOSE))

    def read_close(self):
        """!
        @brief Reads a close message from the stream. Additionally identifies if data was sent using a specific protocol.
        @details ----- MRD_MESSAGE_CLOSE (4) ----------------------------------------------
                 This message signals that all data has been sent (either from server or client).

        @author Kelvin Chow, Jörn Huber
        """

        num_recv_noise_calib = 0
        num_recv_parallel_calib = 0
        num_recv_navigation = 0
        num_recv_phasecorr = 0
        num_recv_rtfeedback = 0
        num_recv_imaging = 0

        for acq_key in self.meas_data.data:
            if 'ACQ_IS_NOISE_MEASUREMENT' in acq_key:
                num_recv_noise_calib += len(self.meas_data.data[acq_key])
            if 'ACQ_IS_PARALLEL_CALIBRATION' in acq_key:
                num_recv_parallel_calib += len(self.meas_data.data[acq_key])
            if 'ACQ_IS_NAVIGATION_DATA' in acq_key:
                num_recv_navigation += len(self.meas_data.data[acq_key])
            if 'ACQ_IS_PHASECORR_DATA' in acq_key:
                num_recv_phasecorr += len(self.meas_data.data[acq_key])
            if 'ACQ_IS_RTFEEDBACK_DATA' in acq_key:
                num_recv_rtfeedback += len(self.meas_data.data[acq_key])
            if 'ACQ_IS_IMAGING' in acq_key:
                num_recv_imaging += len(self.meas_data.data[acq_key])

        logging.info("<-- Received MRD_MESSAGE_CLOSE (4)")
        logging.info("    Total received acquisitions: %5d", self.recvAcqs)
        logging.info("      Noise calibration:         %5d", num_recv_noise_calib)
        logging.info("      Parallel calibration:      %5d", num_recv_parallel_calib)
        logging.info("      Navigation:                %5d", num_recv_navigation)
        logging.info("      Phase correction:          %5d", num_recv_phasecorr)
        logging.info("      Realtime feedback:         %5d", num_recv_rtfeedback)
        logging.info("      Imaging:                   %5d", num_recv_imaging)
        logging.info("    Total received images:       %5d", self.recvImages)
        logging.info("    Total received waveforms:    %5d", self.recvWaveforms)
        logging.info("------------------------------------------")

        self.is_exhausted = True

    def send_text(self, contents):
        """!
        @brief Send a text message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_TEXT (5) -----------------------------------
                 This message contains arbitrary text data.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Length           (   4 bytes, uint32_t      )
                 Text data        (  variable, char          )

        @param contents: (string) The text string which shall be sent to the stream

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_TEXT (2)")
            contents_with_nul = '%s\0' % contents  # Add null terminator
            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_TEXT))
            self.socket.send(struct.Struct('<I').pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())

    def read_text(self):
        """!
       @brief Reads a text message encoded using the ismrmrd protocol from the stream.
       @details ----- MRD_MESSAGE_TEXT (5) -----------------------------------
                This message contains arbitrary text data.
                Message consists of:
                ID               (   2 bytes, unsigned short)
                Length           (   4 bytes, uint32_t      )
                Text data        (  variable, char          )

       @author Kelvin Chow, Jörn Huber
       """
        logging.info("<-- Received MRD_MESSAGE_TEXT (2)")
        length_bytes = self.read(IsmrmrdConstants.SIZEOF_LENGTH)
        length = struct.unpack('<I', length_bytes)[0]
        config_text = self.read(length)
        config_text = config_text.split(b'\x00',1)[0].decode('utf-8')  # Strip off null teminator
        self.texts.append(config_text)

    def send_acquisition(self, acquisition):
        """!
        @brief Send an acquisition message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) -----------------------------
                 This message contains raw k-space data from a single readout.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Fixed header     ( 340 bytes, mixed         )
                 Trajectory       (  variable, float         )
                 Raw k-space data (  variable, float         )

        @param acquisition: (ismrmrd.Acquisition) The acquisition object which shall be serialized into the stream.

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            self.sentAcqs += 1
            if (self.sentAcqs == 1) or (self.sentAcqs % 100 == 0):
                logging.info("--> Sending MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) (total: %d)", self.sentAcqs)

            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_ACQUISITION))
            acquisition.serialize_into(self.socket.send)

    def read_acquisition(self):
        """!
        @brief Reads an acquisition message encoded using the ismrmrd protocol from the stream.
        @details ----- MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) -----------------------------
                 This message contains raw k-space data from a single readout.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Fixed header     ( 340 bytes, mixed         )
                 Trajectory       (  variable, float         )
                 Raw k-space data (  variable, float         )

        @author Kelvin Chow, Jörn Huber
        """
        self.recvAcqs += 1
        if (self.recvAcqs == 1) or (self.recvAcqs % 100 == 0):
            logging.info("<-- Received MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) (total: %d)", self.recvAcqs)

        acq = ismrmrd.Acquisition.deserialize_from(self.read)
        acq_flags = bitmask_to_flags(acq.getHead().flags)

        key_string = ''
        for key_flag in acq_flags:
            if "IS" in key_flag: # Flags which describe a specific type of acquisition contain an 'IS' keyword
                key_string = key_string + "_" + key_flag
        if key_string == '' or key_string == '_ACQ_IS_REVERSE':
            key_string += '_ACQ_IS_IMAGING'
        key_string = key_string[1:]

        if 'ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING' in key_string:

            test_string = key_string.replace('ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING', '')
            if 'ACQ_IS_PARALLEL_CALIBRATION' in test_string:
                key_string = key_string.replace('ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING', '')
            else:
                key_string = key_string.replace('ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING', 'ACQ_IS_PARALLEL_CALIBRATION')
            if key_string[0] == '_':
                key_string = key_string[1:]
            if key_string[-1] == '_':
                key_string = key_string[:-1]

            if key_string not in self.meas_data.data:
                self.meas_data.data[key_string] = []
            self.meas_data.data[key_string].append(acq)

            key_string = key_string.replace('ACQ_IS_PARALLEL_CALIBRATION', 'ACQ_IS_IMAGING')
            if key_string[0] == '_':
                key_string = key_string[1:]
            if key_string[-1] == '_':
                key_string = key_string[:-1]

            if key_string not in self.meas_data.data:
                self.meas_data.data[key_string] = []
            self.meas_data.data[key_string].append(acq)

        else:

            if key_string not in self.meas_data.data:
                self.meas_data.data[key_string] = []
            self.meas_data.data[key_string].append(acq)

    def send_image(self, images):
        """!
        @brief Send an image message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_ISMRMRD_IMAGE (1022) -----------------------------------
                 This message contains a single [x y z cha] image.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Fixed header     ( 198 bytes, mixed         )
                 Attribute length (   8 bytes, uint64_t      )
                 Attribute data   (  variable, char          )
                 Image data       (  variable, variable      )

        @param images: (ismrmrd.Image or list[ismrmrd.Image]) The image object(s) which shall be serialized into the
                                                              stream.

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            if not isinstance(images, list):
                images = [images]

            logging.info("--> Sending MRD_MESSAGE_ISMRMRD_IMAGE (1022) (%d images)", len(images))
            for image in images:
                if image is None:
                    continue

                self.sentImages += 1
                self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_IMAGE))
                image.serialize_into(self.socket.send)

    def read_image(self):
        """!
        @brief Reads an image message encoded using the ismrmrd protocol from the stream.
        @details ----- MRD_MESSAGE_ISMRMRD_IMAGE (1022) -----------------------------------
                 This message contains a single [x y z cha] image.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Fixed header     ( 198 bytes, mixed         )
                 Attribute length (   8 bytes, uint64_t      )
                 Attribute data   (  variable, char          )
                 Image data       (  variable, variable      )

        @author Kelvin Chow, Jörn Huber
        """
        self.recvImages += 1
        logging.info("<-- Received MRD_MESSAGE_ISMRMRD_IMAGE (1022)")
        self.images.append(ismrmrd.Image.deserialize_from(self.read))

    def send_waveform(self, waveform):
        """!
        @brief Sends a waveform message encoded using the ismrmrd protocol to the stream.
        @details ----- MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) -----------------------------
                 This message contains abitrary (e.g. physio) waveform data.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Fixed header     ( 240 bytes, mixed         )
                 Waveform data    (  variable, uint32_t      )

        @param waveform: (ismrmrd.Waveform) Waveform object, which is serialized to the stream.

        @author Kelvin Chow, Jörn Huber
        """
        with self.lock:
            self.sentWaveforms += 1
            if (self.sentWaveforms == 1) or (self.sentWaveforms % 100 == 0):
                logging.info("--> Sending MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) (total: %d)", self.sentWaveforms)
            self.socket.send(struct.Struct('<H').pack(IsmrmrdConstants.ID_MESSAGE_WAVEFORM))
            waveform.serialize_into(self.socket.send)

    def read_waveform(self):
        """!
        @brief Reads a waveform message encoded using the ismrmrd protocol from the stream.
        @details ----- MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) -----------------------------
                 This message contains abitrary (e.g. physio) waveform data.
                 Message consists of:
                 ID               (   2 bytes, unsigned short)
                 Fixed header     ( 240 bytes, mixed         )
                 Waveform data    (  variable, uint32_t      )

        @author Kelvin Chow, Jörn Huber
        """
        self.recvWaveforms += 1
        if (self.recvWaveforms == 1) or (self.recvWaveforms % 100 == 0):
            logging.info("<-- Received MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) (total: %d)", self.recvWaveforms)

        waveform = ismrmrd.Waveform.deserialize_from(self.read)

        self.waveforms.append(waveform)

    # <-- Based on python-ismrmrd-server (see third_party_licenses.txt)

def gstar_to_ismrmrd_hdr(prot, info, expo, sys, root, opt_meas_info = None):
    """!
    @brief This function transfers header entries from gammastar to the corresponding ismrmrd header.

    @param prot: (dict) Contains protocol information such as the parallel imaging mode (PAT mode)
    @param info: (dict) Contains additional information
    @param expo: (dict) Contains additional information
    @param sys: (dict) Contains system information such as the resonance frequency
    @param root: (dict) Contains information such as the acquisition matrix size
    @param opt_meas_info: (dict) Contains information such as the acquisition matrix size

    @return
        - (string) ISMRMRD header in xml format but serialized into string

    @author Daniel Hoinkiss, Jörn Huber
    """
    xml_root = xml_elem_tree.Element("ismrmrdHeader", xmlns="http://www.ismrm.org/ISMRMRD")

    # subjectInformation
    sub_information = xml_elem_tree.SubElement(xml_root, "subjectInformation")

    patient_name = xml_elem_tree.SubElement(sub_information, "patientName")
    if opt_meas_info and "patientName" in opt_meas_info:
        patient_name.text = opt_meas_info["patientName"]
    else:
        patient_name.text = 'PHANTOM'

    patient_weight = xml_elem_tree.SubElement(sub_information, "patientWeight_kg")
    if opt_meas_info and "patientWeight_kg" in opt_meas_info:
        patient_weight.text = opt_meas_info["patientWeight_kg"]
    else:
        patient_weight.text = '0.02'

    patient_height = xml_elem_tree.SubElement(sub_information, "patientHeight_m")
    if opt_meas_info and "patientHeight_m" in opt_meas_info:
        patient_height.text = opt_meas_info["patientHeight_m"]
    else:
        patient_height.text = '0.03'

    patient_id = xml_elem_tree.SubElement(sub_information, "patientID")
    if opt_meas_info and "patientID" in opt_meas_info:
        patient_id.text = opt_meas_info["patientID"]
    else:
        patient_id.text = '0000000'

    patient_bday = xml_elem_tree.SubElement(sub_information, "patientBirthdate")
    if opt_meas_info and "patientBirthdate" in opt_meas_info:
        patient_bday.text = opt_meas_info["patientBirthdate"]
    else:
        patient_bday.text = '2000-01-01'

    # studyInformation
    study_info = xml_elem_tree.SubElement(xml_root, "studyInformation")

    study_date = xml_elem_tree.SubElement(study_info, "studyDate")
    if opt_meas_info and "studyDate" in opt_meas_info:
        study_date.text = opt_meas_info["studyDate"]
    else:
        study_date.text = datetime.now().strftime('%Y-%m-%d')

    study_time = xml_elem_tree.SubElement(study_info, "studyTime")
    if opt_meas_info and "studyTime" in opt_meas_info:
        study_time.text = opt_meas_info["studyTime"]
    else:
        study_time.text = datetime.now().strftime('%H:%M:%S')

    study_description = xml_elem_tree.SubElement(study_info, "studyDescription")
    if opt_meas_info and "studyDescription" in opt_meas_info:
        study_description.text = opt_meas_info["studyDescription"]
    else:
        study_description.text = 'Phantom_Test'

    body_part = xml_elem_tree.SubElement(study_info, "bodyPartExamined")
    if opt_meas_info and "bodyPartExamined" in opt_meas_info:
        body_part.text = opt_meas_info["bodyPartExamined"]
    else:
        body_part.text = 'GENERIC'

    # measurementInformation
    meas_info = xml_elem_tree.SubElement(xml_root, "measurementInformation")

    meas_id = xml_elem_tree.SubElement(meas_info, "measurementID")
    if opt_meas_info and "measurementID" in opt_meas_info:
        meas_id.text = opt_meas_info["measurementID"]
    else:
        meas_id.text = f"{random.randint(100, 999)}_{random.randint(10000, 99999)}_{random.randint(1000000, 9999999)}_{random.randint(1, 99):02}"

    patient_position = xml_elem_tree.SubElement(meas_info, "patientPosition")
    if opt_meas_info and "patientPosition" in opt_meas_info:
        patient_position.text = opt_meas_info["patientPosition"]
    else:
        patient_position.text = 'HFS'

    protocol_name = xml_elem_tree.SubElement(meas_info, "protocolName")
    if opt_meas_info and "protocolName" in opt_meas_info:
        protocol_name.text = opt_meas_info["protocolName"]
    else:
        protocol_name.text = 'default'

    frame_ref_uid = xml_elem_tree.SubElement(meas_info, "frameOfReferenceUID")
    if opt_meas_info and "frameOfReferenceUID" in opt_meas_info:
        frame_ref_uid.text = opt_meas_info["frameOfReferenceUID"]
    else:
        frame_ref_uid.text = f"1.3.12.2.1107.5.2.19.{random.randint(10000, 99999)}.{random.randint(1, 99999)}.{random.randint(2010000000000000, 2019999999999999)}.0.0.0"

    series_description = xml_elem_tree.SubElement(meas_info, "seriesDescription")
    if opt_meas_info and "seriesDescription" in opt_meas_info:
        series_description.text = opt_meas_info["seriesDescription"]
    else:
        series_description.text = 'Phantom_Series'

    # acquisitionSystemInformation
    acq_info = xml_elem_tree.SubElement(xml_root, "acquisitionSystemInformation")

    vendor = xml_elem_tree.SubElement(acq_info, "systemVendor")
    if opt_meas_info and "systemVendor" in opt_meas_info:
        vendor.text = opt_meas_info["systemVendor"]
    else:
        vendor.text = 'gammaSTAR'

    model = xml_elem_tree.SubElement(acq_info, "systemModel")
    if opt_meas_info and "systemModel" in opt_meas_info:
        model.text = opt_meas_info["systemModel"]
    else:
        model.text = 'mrZero'

    field_strength = xml_elem_tree.SubElement(acq_info, "systemFieldStrength_T")
    if opt_meas_info and "systemFieldStrength_T" in opt_meas_info:
        field_strength.text = opt_meas_info["systemFieldStrength_T"]
    else:
        field_strength.text = str(3)

    institution = xml_elem_tree.SubElement(acq_info, "institutionName")
    if opt_meas_info and "institutionName" in opt_meas_info:
        institution.text = opt_meas_info["institutionName"]
    else:
        institution.text = 'Fraunhofer MEVIS'

    # experimentalConditions
    exp_conditions = xml_elem_tree.SubElement(xml_root, "experimentalConditions")

    h1_freq = xml_elem_tree.SubElement(exp_conditions, "H1resonanceFrequency_Hz")
    if opt_meas_info and "H1resonanceFrequency_Hz" in opt_meas_info:
        h1_freq.text = opt_meas_info["H1resonanceFrequency_Hz"]
    else:
        h1_freq.text = str(int(sys['frequency']['1']))

    # encoding
    encoding = xml_elem_tree.SubElement(xml_root, "encoding")

    encoded_space = xml_elem_tree.SubElement(encoding, "encodedSpace")
    enc_matrix_size = xml_elem_tree.SubElement(encoded_space, "matrixSize")
    enc_mat_x = xml_elem_tree.SubElement(enc_matrix_size, "x")
    enc_mat_x.text = str(
        int(int(root['acq_size']['1']) * int(prot['read_oversampling']) * float(prot['read_partial_fourier'])))
    enc_mat_y = xml_elem_tree.SubElement(enc_matrix_size, "y")
    enc_mat_y.text = str(
        int(int(root['acq_size']['2']) * int(prot['phase_oversampling']) * float(prot['phase_partial_fourier'])))
    enc_mat_z = xml_elem_tree.SubElement(enc_matrix_size, "z")
    enc_mat_z.text = str(
        int(int(root['acq_size']['3']) * int(prot['slice_oversampling']) * float(prot['slice_partial_fourier'])))

    enc_fov = xml_elem_tree.SubElement(encoded_space, "fieldOfView_mm")
    enc_fov_x = xml_elem_tree.SubElement(enc_fov, "x")
    enc_fov_x.text = str(float(root['fov']['1']) * float(prot['read_oversampling']) * 1000)
    enc_fov_y = xml_elem_tree.SubElement(enc_fov, "y")
    enc_fov_y.text = str(float(root['fov']['2']) * float(prot['phase_oversampling']) * 1000)
    enc_fov_z = xml_elem_tree.SubElement(enc_fov, "z")
    enc_fov_z.text = str(float(root['fov']['3']) * float(prot['slice_oversampling']) * 1000)

    recon_space = xml_elem_tree.SubElement(encoding, "reconSpace")
    recon_matrix_size = xml_elem_tree.SubElement(recon_space, "matrixSize")
    recon_size_x = xml_elem_tree.SubElement(recon_matrix_size, "x")
    recon_size_x.text = str(int(root['mat_size']['1']))
    recon_size_y = xml_elem_tree.SubElement(recon_matrix_size, "y")
    recon_size_y.text = str(int(root['mat_size']['2']))
    recon_size_z = xml_elem_tree.SubElement(recon_matrix_size, "z")
    recon_size_z.text = str(int(root['mat_size']['3']))

    recon_fov = xml_elem_tree.SubElement(recon_space, "fieldOfView_mm")
    recon_fov_x = xml_elem_tree.SubElement(recon_fov, "x")
    recon_fov_x.text = str(float(root['fov']['1']) * 1000)
    recon_fov_y = xml_elem_tree.SubElement(recon_fov, "y")
    recon_fov_y.text = str(float(root['fov']['2']) * 1000)
    recon_fov_z = xml_elem_tree.SubElement(recon_fov, "z")
    recon_fov_z.text = str(float(root['fov']['3']) * 1000)

    # parallel_imaging
    parallel_imaging = xml_elem_tree.SubElement(encoding, "parallelImaging")
    acceleration_factor = xml_elem_tree.SubElement(parallel_imaging, "accelerationFactor")
    accel_ksp_1 = xml_elem_tree.SubElement(acceleration_factor, "kspace_encoding_step_1")
    if 'PAT_factor_phase' in prot.keys():
        if 'PAT_mode' in prot.keys() and prot['PAT_mode'] != 'None':
            accel_ksp_1.text = str(prot['PAT_factor_phase'])
        else:
            accel_ksp_1.text = str(1)
    else:
        accel_ksp_1.text = str(1)
    accel_ksp_2 = xml_elem_tree.SubElement(acceleration_factor, "kspace_encoding_step_2")
    if 'PAT_factor_slice' in prot.keys():
        accel_ksp_2.text = str(prot['PAT_factor_slice'])
    else:
        accel_ksp_2.text = str(1)

    trajectory = xml_elem_tree.SubElement(encoding, "trajectory")
    trajectory.text = "cartesian"
    calib_mode = xml_elem_tree.SubElement(parallel_imaging, "calibrationMode")
    calib_mode.text = 'other'

    # sequenceParameters
    sequence_parameters = xml_elem_tree.SubElement(xml_root, "sequenceParameters")
    
    if 'TR' in prot.keys():
        if type(prot['TR']) == dict:
            for key in range(1, len(prot['TR'].keys()) + 1):
                tr = xml_elem_tree.SubElement(sequence_parameters, "TR")
                tr.text = str(float(prot['TR'][str(key)]) * 1000)
        else:
            tr = xml_elem_tree.SubElement(sequence_parameters, "TR")
            tr.text = str(float(prot['TR']) * 1000)
    else:
        tr = xml_elem_tree.SubElement(sequence_parameters, "TR")
        tr.text = str(0)
    
    if 'TE' in prot.keys():
        if type(prot['TE']) == dict:
            for key in range(1, len(prot['TE'].keys()) + 1):
                te = xml_elem_tree.SubElement(sequence_parameters, "TE")
                te.text = str(float(prot['TE'][str(key)]) * 1000)
        else:
            te = xml_elem_tree.SubElement(sequence_parameters, "TE")
            te.text = str(float(prot['TE']) * 1000)
    else:
        te = xml_elem_tree.SubElement(sequence_parameters, "TE")
        te.text = str(0)

    if 'TI' in prot.keys():
        if type(prot['TI']) == dict:
            for key in range(1, len(prot['TI'].keys()) + 1):
                ti = xml_elem_tree.SubElement(sequence_parameters, "TI")
                ti.text = str(float(prot['TI'][str(key)]) * 1000)
        else:
            ti = xml_elem_tree.SubElement(sequence_parameters, "TI")
            ti.text = str(float(prot['TI']) * 1000)
    else:
        ti = xml_elem_tree.SubElement(sequence_parameters, "TI")
        ti.text = str(0)
    flip_angle = xml_elem_tree.SubElement(sequence_parameters, "flipAngle_deg")
    if 'flip_angle' in prot.keys():
        flip_angle.text = str(prot['flip_angle'])
    else:
        flip_angle.text = str(0)

    # userParameters
    user_parameters = xml_elem_tree.SubElement(xml_root, "userParameters")
    if 'bValues' in prot:
        user_parameters_double = xml_elem_tree.SubElement(user_parameters, "userParameterDouble")
        b_values_name = xml_elem_tree.SubElement(user_parameters_double, "name")
        b_values_name.text = "b_values"
        b_values_value = xml_elem_tree.SubElement(user_parameters_double, "value")
        b_values_value.text = str(prot['bValues'])

    xml_str = xml_elem_tree.tostring(xml_root, encoding="utf-8", method="xml").decode()
    return xml_str


def twix_hdr_to_ismrmrd_hdr(twix_hdr):
    """!
    @brief This function transfers header entries from a twix_hdr object to the corresponding ismrmrd header object

    @param twix_hdr: (twix header): Header object as obtained from pymapvbvd

    @return
        - (string) ISMRMRD header in xml format but serialized into string

    @author Jörn Huber
    """
    root = xml_elem_tree.Element("ismrmrdHeader", xmlns="http://www.ismrm.org/ISMRMRD")

    # subjectInformation
    sub_information = xml_elem_tree.SubElement(root, "subjectInformation")

    patient_name = xml_elem_tree.SubElement(sub_information, "patientName")
    patient_name.text = str(twix_hdr['Config']['tPatientName'])

    patient_weight = xml_elem_tree.SubElement(sub_information, "patientWeight_kg")
    patient_weight.text = str(twix_hdr['Dicom']['flUsedPatientWeight'])

    patient_height = xml_elem_tree.SubElement(sub_information, "patientHeight_m")
    patient_height.text = str(twix_hdr['Meas']['flPatientHeight'])

    patient_id = xml_elem_tree.SubElement(sub_information, "patientID")
    patient_id.text = str(twix_hdr['Config']['PatientID'])

    patient_bday = xml_elem_tree.SubElement(sub_information, "patientBirthdate")
    patient_bday.text = str(twix_hdr['Config']['PatientBirthDay'])

    # studyInformation
    study_info = xml_elem_tree.SubElement(root, "studyInformation")

    study_date = xml_elem_tree.SubElement(study_info, "studyDate")
    study_date.text = twix_hdr['Meas']['PrepareTimestamp'][0:10]

    study_time = xml_elem_tree.SubElement(study_info, "studyTime")
    study_time.text = twix_hdr['Meas']['PrepareTimestamp'][11:]

    study_description = xml_elem_tree.SubElement(study_info, "studyDescription")
    study_description.text = str(twix_hdr['Dicom']['tStudyDescription'])

    body_part = xml_elem_tree.SubElement(study_info, "bodyPartExamined")
    body_part.text = twix_hdr['Dicom']['tBodyPartExamined']

    # measurementInformation
    meas_info = xml_elem_tree.SubElement(root, "measurementInformation")

    meas_id = xml_elem_tree.SubElement(meas_info, "measurementID")
    meas_id.text = str(twix_hdr['Meas']['Study'])

    patient_position = xml_elem_tree.SubElement(meas_info, "patientPosition")
    patient_position.text = twix_hdr['Config']['PatientPosition']

    protocol_name = xml_elem_tree.SubElement(meas_info, "protocolName")
    protocol_name.text = str(twix_hdr['Meas']['tProtocolName'])

    frame_ref_uid = xml_elem_tree.SubElement(meas_info, "frameOfReferenceUID")
    frame_ref_uid.text = str(twix_hdr['Config']['FrameOfReference'])

    series_description = xml_elem_tree.SubElement(meas_info, "seriesDescription")
    series_description.text = str(twix_hdr['Meas']['tProtocolName'])

    # acquisitionSystemInformation
    acq_info = xml_elem_tree.SubElement(root, "acquisitionSystemInformation")

    vendor = xml_elem_tree.SubElement(acq_info, "systemVendor")
    vendor.text = str(twix_hdr['Dicom']['Manufacturer'])

    model = xml_elem_tree.SubElement(acq_info, "systemModel")
    model.text = str(twix_hdr['Dicom']['ManufacturersModelName'])

    field_strength = xml_elem_tree.SubElement(acq_info, "systemFieldStrength_T")
    field_strength.text = str(twix_hdr['MeasYaps'][('sProtConsistencyInfo', 'flNominalB0')])

    institution = xml_elem_tree.SubElement(acq_info, "institutionName")
    institution.text = str(twix_hdr['Meas']['InstitutionName'])

    # experimentalConditions
    exp_conditions = xml_elem_tree.SubElement(root, "experimentalConditions")

    h1_freq = xml_elem_tree.SubElement(exp_conditions, "H1resonanceFrequency_Hz")
    h1_freq.text = str(int(twix_hdr['Dicom']['lFrequency']))

    # encoding
    encoding = xml_elem_tree.SubElement(root, "encoding")

    encoded_space = xml_elem_tree.SubElement(encoding, "encodedSpace")
    matrix_size = xml_elem_tree.SubElement(encoded_space, "matrixSize")
    size_x = xml_elem_tree.SubElement(matrix_size, "x")
    size_x.text = str(int(twix_hdr['Config']['NImageCols']))
    size_y = xml_elem_tree.SubElement(matrix_size, "y")
    size_y.text = str(int(twix_hdr['Config']['NLinMeas']))
    size_z = xml_elem_tree.SubElement(matrix_size, "z")
    size_z.text = str(int(twix_hdr['Config']['NParMeas']))

    fov = xml_elem_tree.SubElement(encoded_space, "fieldOfView_mm")
    fov_x = xml_elem_tree.SubElement(fov, "x")
    fov_x.text = str(twix_hdr['Protocol']['RoFOV'])
    fov_y = xml_elem_tree.SubElement(fov, "y")
    fov_y.text = str(twix_hdr['Protocol']['PeFOV'])
    fov_z = xml_elem_tree.SubElement(fov, "z")
    fov_z.text = str(0.0)

    recon_space = xml_elem_tree.SubElement(encoding, "reconSpace")
    recon_matrix_size = xml_elem_tree.SubElement(recon_space, "matrixSize")
    recon_size_x = xml_elem_tree.SubElement(recon_matrix_size, "x")
    recon_size_x.text = str(int(twix_hdr['Config']['NImageCols']))
    recon_size_y = xml_elem_tree.SubElement(recon_matrix_size, "y")
    recon_size_y.text = str(int(twix_hdr['Dicom']['lPhaseEncodingLines']))
    recon_size_z = xml_elem_tree.SubElement(recon_matrix_size, "z")
    recon_size_z.text = str(int(twix_hdr['Config']['NParMeas']))

    recon_fov = xml_elem_tree.SubElement(recon_space, "fieldOfView_mm")
    recon_fov_x = xml_elem_tree.SubElement(recon_fov, "x")
    recon_fov_x.text = str(twix_hdr['Protocol']['RoFOV'])
    recon_fov_y = xml_elem_tree.SubElement(recon_fov, "y")
    recon_fov_y.text = str(twix_hdr['Protocol']['PeFOV'])
    recon_fov_z = xml_elem_tree.SubElement(recon_fov, "z")
    recon_fov_z.text = str(twix_hdr['Dicom']['dThickness'])

    # TODO: Add Encoding Limits

    parallel_imaging = xml_elem_tree.SubElement(encoding, "parallelImaging")

    acceleration_factor = xml_elem_tree.SubElement(parallel_imaging, "accelerationFactor")
    accel_ksp_1 = xml_elem_tree.SubElement(acceleration_factor, "kspace_encoding_step_1")
    accel_ksp_1.text = str(int(twix_hdr['Protocol']['ProdAccelFactorPE']))
    accel_ksp_2 = xml_elem_tree.SubElement(acceleration_factor, "kspace_encoding_step_2")
    accel_ksp_2.text = str(int(twix_hdr['Protocol']['ProdAccelFactor3D']))

    trajectory = xml_elem_tree.SubElement(encoding, "trajectory")
    trajectory.text = "cartesian"

    calib_mode = xml_elem_tree.SubElement(parallel_imaging, "calibrationMode")
    if twix_hdr['Meas']['ucRefScanMode'] == 4.0:
        calib_mode.text = 'separate'
    else:
        calib_mode.text = 'other'

    # sequenceParameters
    sequence_parameters = xml_elem_tree.SubElement(root, "sequenceParameters")

    tr_list = str(twix_hdr['Protocol']['alTR']).split()
    for tr_time in tr_list:
        tr = xml_elem_tree.SubElement(sequence_parameters, "TR")
        tr.text = tr_time

    te_list = str(twix_hdr['Protocol']['alTE']).split()
    for te_time in te_list:
        te = xml_elem_tree.SubElement(sequence_parameters, "TE")
        te.text = te_time

    ti_list = str(twix_hdr['Protocol']['alTI']).split()
    for ti_time in ti_list:
        ti = xml_elem_tree.SubElement(sequence_parameters, "TI")
        ti.text = ti_time

    flip_angle = xml_elem_tree.SubElement(sequence_parameters, "flipAngle_deg")
    flip_angle.text = str(twix_hdr['Dicom']['adFlipAngleDegree'])

    xml_str = xml_elem_tree.tostring(root, encoding="utf-8", method="xml").decode()
    return xml_str


def noise_scan_to_acq(numpy_noise_array):
    """!
    @brief A method, which creates an acquisition object from a numpy array, containing noise correlation scans
           from multiple channels.

    @param numpy_noise_array: (np.ndarray) Numpy array (num_col, num_channel) which contains individual
                                           noise data for different channels.

    @return
        - (ismrmrd.Acquisition) Acquisition object which contains the reformatted noise data.

    @author Jörn Huber
    """

    acq = ismrmrd.Acquisition.from_array(np.transpose(numpy_noise_array, [1, 0]))
    inv_flags = {v: k for k, v in IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.items()}
    acq.set_flag(inv_flags['ACQ_IS_NOISE_MEASUREMENT'])
    # TODO: Add additional flags and idx etc

    return acq


def numpy_and_raw_rep_to_acq(numpy_array, raw_adc_representations, traj_info = None, reverse_lineflip = False):
    """!
    @brief A method, which creates a list of ISMRMRD acquisition objects from a three-dimensional numpy array based
           on the gammastar raw representations and trajectory information which are provided as a list.

    @param numpy_array: (np.ndarray) Numpy array (num_col, num_channel, num_acquisition) which contains individual
                                     readout data.
    @param raw_adc_representations: (list[json]) A list of gammastar raw representation, which correspond to each
                                                 acquisition of the numpy_array.
    @param traj_info: (dict) A dictionary which contains trajectory information (m0x, m0y, m0z) for each data sample.
    @param reverse_lineflip: (bool) Boolean value, indicating whether acq lines with the ACQ_REVERSE_FLAG need to be
                                    flipped back to their original position as some tools which might be used for
                                    reading raw data such as pymapvbvd might already correct this flip which will
                                    result in faulty behaviour on the reconstruction side.

    @return
        - (list) List of ISMRMRD acquisition objects

    @author Jörn Huber, Daniel Hoinkiss
    """

    acq_list = []
    np_index = 0

    if traj_info is not None:
        read_size = numpy_array.shape[0]
        m0x = np.array(list(traj_info['m0x'].values())).reshape(-1, 1, read_size).T # pylint: disable=too-many-function-args
        m0y = np.array(list(traj_info['m0y'].values())).reshape(-1, 1, read_size).T # pylint: disable=too-many-function-args
        m0z = np.array(list(traj_info['m0z'].values())).reshape(-1, 1, read_size).T # pylint: disable=too-many-function-args
        traj_np = np.concatenate((m0x, m0y, m0z), axis=1)

    for i_readout in range(len(raw_adc_representations)):

        raw_rep = raw_adc_representations[i_readout]

        if 'ACQ_IS_REVERSE' in raw_rep['adc_header'] and reverse_lineflip:
            numpy_array[:, :, np_index] = np.flipud(numpy_array[:, :, np_index])

        if traj_info is not None:
            acq = ismrmrd.Acquisition.from_array(np.transpose(numpy_array[:, :, np_index], [1, 0]),
                                                 traj_np[:, :, i_readout])
        else:
            acq = ismrmrd.Acquisition.from_array(np.transpose(numpy_array[:, :, np_index], [1, 0]))

        # Position information
        acq.read_dir[0] = raw_rep['adc_header']['read_dir']['1'] # pylint: disable=maybe-no-member
        acq.read_dir[1] = raw_rep['adc_header']['read_dir']['2'] # pylint: disable=maybe-no-member
        acq.read_dir[2] = raw_rep['adc_header']['read_dir']['3'] # pylint: disable=maybe-no-member

        acq.phase_dir[0] = raw_rep['adc_header']['phase_dir']['1'] # pylint: disable=maybe-no-member
        acq.phase_dir[1] = raw_rep['adc_header']['phase_dir']['2'] # pylint: disable=maybe-no-member
        acq.phase_dir[2] = raw_rep['adc_header']['phase_dir']['3'] # pylint: disable=maybe-no-member

        acq.slice_dir[0] = raw_rep['adc_header']['slice_dir']['1'] # pylint: disable=maybe-no-member
        acq.slice_dir[1] = raw_rep['adc_header']['slice_dir']['2'] # pylint: disable=maybe-no-member
        acq.slice_dir[2] = raw_rep['adc_header']['slice_dir']['3'] # pylint: disable=maybe-no-member

        acq.position[0] = raw_rep['adc_header']['position']['1'] # pylint: disable=maybe-no-member
        acq.position[1] = raw_rep['adc_header']['position']['2'] # pylint: disable=maybe-no-member
        acq.position[2] = raw_rep['adc_header']['position']['3'] # pylint: disable=maybe-no-member

        # IDX
        if 'idx_average' in raw_rep['adc_header']:
            acq.idx.average = raw_rep['adc_header']['idx_average'] # pylint: disable=maybe-no-member
        if 'idx_contrast' in raw_rep['adc_header']:
            acq.idx.contrast = raw_rep['adc_header']['idx_contrast'] # pylint: disable=maybe-no-member
        if 'idx_kspace_encode_step_1' in raw_rep['adc_header']:
            acq.idx.kspace_encode_step_1 = raw_rep['adc_header']['idx_kspace_encode_step_1'] # pylint: disable=maybe-no-member
        if 'idx_kspace_encode_step_2' in raw_rep['adc_header']:
            acq.idx.kspace_encode_step_2 = raw_rep['adc_header']['idx_kspace_encode_step_2'] # pylint: disable=maybe-no-member
        if 'idx_phase' in raw_rep['adc_header']:
            acq.idx.phase = raw_rep['adc_header']['idx_phase'] # pylint: disable=maybe-no-member
        if 'idx_repetition' in raw_rep['adc_header']:
            acq.idx.repetition = raw_rep['adc_header']['idx_repetition'] # pylint: disable=maybe-no-member
        if 'idx_segment' in raw_rep['adc_header']:
            acq.idx.segment = raw_rep['adc_header']['idx_segment'] # pylint: disable=maybe-no-member
        if 'idx_set' in raw_rep['adc_header']:
            acq.idx.set = raw_rep['adc_header']['idx_set'] # pylint: disable=maybe-no-member
        if 'idx_slice' in raw_rep['adc_header']:
            acq.idx.slice = raw_rep['adc_header']['idx_slice'] # pylint: disable=maybe-no-member

        # Flags
        inv_flags = {v: k for k, v in IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.items()}
        for adc_elem in raw_rep['adc_header']:
            if adc_elem in inv_flags:
                acq.set_flag(inv_flags[adc_elem])

        # Other
        acq.center_sample = raw_rep['adc_header']['center_sample'] # pylint: disable=maybe-no-member
        acq.samples_time_us = raw_rep['adc_header']['sample_time_us'] # pylint: disable=maybe-no-member
        acq.acquisition_time_stamp = raw_rep['tstart'] # pylint: disable=maybe-no-member

        acq_list.append(acq)

        np_index += 1

    return acq_list


def create_dummy_ismrmrd_header():
    """!
    @brief A method, which first creates a dummy ismrmrd header in xml format, which is encoded into a string and
           returned.

    @return
        - (string) ISMRMRD header in xml format but serialized into string

    @author Jörn Huber, GitHub Copilot (GPT 4.1)
    """

    root = xml_elem_tree.Element("ismrmrdHeader", xmlns="http://www.ismrm.org/ISMRMRD")

    meas_info = xml_elem_tree.SubElement(root, "measurementInformation")
    meas_id = xml_elem_tree.SubElement(meas_info, "measurementID")
    meas_id.text = "45098_164288619_164288624_176"
    patient_position = xml_elem_tree.SubElement(meas_info, "patientPosition")
    patient_position.text = "HFS"
    protocol_name = xml_elem_tree.SubElement(meas_info, "protocolName")
    protocol_name.text = "DummyProtocol"
    frame_ref_uid = xml_elem_tree.SubElement(meas_info, "frameOfReferenceUID")
    frame_ref_uid.text = "1.3.12.2.1107.5.2.19.45098.1.20210319161139496.0.0.0"

    acq_info = xml_elem_tree.SubElement(root, "acquisitionSystemInformation")
    vendor = xml_elem_tree.SubElement(acq_info, "systemVendor")
    vendor.text = "Vendor"
    model = xml_elem_tree.SubElement(acq_info, "systemModel")
    model.text = "Model"
    field_strength = xml_elem_tree.SubElement(acq_info, "systemFieldStrength_T")
    field_strength.text = "2.89362001"
    receiver_channels = xml_elem_tree.SubElement(acq_info, "receiverChannels")
    receiver_channels.text = "16"
    institution = xml_elem_tree.SubElement(acq_info, "institutionName")
    institution.text = "Institution"

    exp_conditions = xml_elem_tree.SubElement(root, "experimentalConditions")
    h1_freq = xml_elem_tree.SubElement(exp_conditions, "H1resonanceFrequency_Hz")
    h1_freq.text = "123251770"

    sequence_parameters = xml_elem_tree.SubElement(root, "sequenceParameters")
    tr = xml_elem_tree.SubElement(sequence_parameters, "TR")
    tr.text = str(0.0)
    te = xml_elem_tree.SubElement(sequence_parameters, "TE")
    te.text = str(0.0)
    ti = xml_elem_tree.SubElement(sequence_parameters, "TI")
    ti.text = str(0.0)
    flip_angle = xml_elem_tree.SubElement(sequence_parameters, "flipAngle_deg")
    flip_angle.text = str(90.0)
    echo_spacing = xml_elem_tree.SubElement(sequence_parameters, "echo_spacing")
    echo_spacing.text = str(0.0)

    encoding = xml_elem_tree.SubElement(root, "encoding")

    encoded_space = xml_elem_tree.SubElement(encoding, "encodedSpace")
    matrix_size = xml_elem_tree.SubElement(encoded_space, "matrixSize")
    size_x = xml_elem_tree.SubElement(matrix_size, "x")
    size_x.text = str(int(96))
    size_y = xml_elem_tree.SubElement(matrix_size, "y")
    size_y.text = str(int(96))
    size_z = xml_elem_tree.SubElement(matrix_size, "z")
    size_z.text = str(int(1))

    fov = xml_elem_tree.SubElement(encoded_space, "fieldOfView_mm")
    fov_x = xml_elem_tree.SubElement(fov, "x")
    fov_x.text = str(192.0)
    fov_y = xml_elem_tree.SubElement(fov, "y")
    fov_y.text = str(192.0)
    fov_z = xml_elem_tree.SubElement(fov, "z")
    fov_z.text = str(5.0)

    recon_space = xml_elem_tree.SubElement(encoding, "reconSpace")
    recon_matrix_size = xml_elem_tree.SubElement(recon_space, "matrixSize")
    recon_size_x = xml_elem_tree.SubElement(recon_matrix_size, "x")
    recon_size_x.text = str(int(96))
    recon_size_y = xml_elem_tree.SubElement(recon_matrix_size, "y")
    recon_size_y.text = str(int(96))
    recon_size_z = xml_elem_tree.SubElement(recon_matrix_size, "z")
    recon_size_z.text = str(int(1))

    recon_fov = xml_elem_tree.SubElement(recon_space, "fieldOfView_mm")
    recon_fov_x = xml_elem_tree.SubElement(recon_fov, "x")
    recon_fov_x.text = str(192.0)
    recon_fov_y = xml_elem_tree.SubElement(recon_fov, "y")
    recon_fov_y.text = str(192.0)
    recon_fov_z = xml_elem_tree.SubElement(recon_fov, "z")
    recon_fov_z.text = str(5.0)

    parallel_imaging = xml_elem_tree.SubElement(encoding, "parallelImaging")

    acceleration_factor = xml_elem_tree.SubElement(parallel_imaging, "accelerationFactor")
    accel_ksp_1 = xml_elem_tree.SubElement(acceleration_factor, "kspace_encoding_step_1")
    accel_ksp_1.text = str(1)
    accel_ksp_2 = xml_elem_tree.SubElement(acceleration_factor, "kspace_encoding_step_2")
    accel_ksp_2.text = str(1)

    xml_str = xml_elem_tree.tostring(root, encoding="utf-8", method="xml").decode()

    return xml_str


def gstar_recon_emitter(host_address, port, list_of_acqs, ismrmrd_header, protocol = None, config_message = None):
    """!
    @brief ISMRMRD client which sends ismrmrd.Acquisition objects together with respective text messages to the stream.

    @param list_of_acqs: (list[ismrmrd.Acquisition]) A list of acquisition objects which shall be serialized to the
                         stream
    @param protocol: (double) Can be 1.0 or 1.1 and describes the desired series of messages which shall be used for
                              sending.
    @param host_address: (string) Host address of server
    @param port: (int) Port number of server
    @param config_message: (string) Algorithm configuration message.
    @param ismrmrd_header: (string) XML encoded string which contains the ismrmrd header information. If no header is
                           provided, a dummy header is used.

    @return
        - (ConnectionBuffer) Returns the connection buffer object.

    @author Jörn Huber, Kelvin Chow
    """
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    if protocol != 1.0 and protocol != 1.1:
        raise ValueError("Invalid protocol. Allowed are 1.0 or 1.1")

    print("gammaSTAR emitter: Sending data using " + str(protocol) + " protocol")

    print("  Connecting to ISMRMRD server at " + host_address + ":" + str(port))
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    attempt = 0
    max_attempts = 5
    success = False
    while attempt < max_attempts:
        try:
            client_socket.connect((host_address, port))
        except socket.error as error:
            print("Failed to connect")
            time.sleep(1)
            attempt += 1
        else:
            success = True
            break

    if not success:
        client_socket.close()
        print("... Aborting")
        return False

    con = ConnectionBuffer(client_socket)

    if config_message is None:
        config_file_string = "ConfigFile"
    else:
        config_file_string = config_message

    if protocol == 1.0:

        con.send_config_file(config_file_string)
        con.send_metadata(ismrmrd_header)

    elif protocol == 1.1:
        con.send_config_file("openrecon")
        con.send_metadata(ismrmrd_header)
        con.send_text(config_file_string)

    for acq in list_of_acqs:
        con.send_acquisition(acq)

    con.send_close()
    return con


def gstar_recon_injector(con):
    """!
    @brief ISMRMRD client which uses a connected client socket to wait for reconstructed images on the server side.

    @param con: (ConnectionBuffer) The ConnectionBuffer object.

    @author Jörn Huber
    """
    print("GSTAR Injector: Waiting for incoming messages from reconstruction chain")
    con.receive_messages()


def gstar_recon_server(host, port):
    """!
    @brief gammaSTAR server implementation. Use this to catch data streams from a client.

    @param host: (string) Host ip address
    @param port: (int) Port on which the server should listen

    @return
        - (ConnectionBuffer) The connection buffer object.
        - (socket) The server socket connection object.

    @author Jörn Huber
    """
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    logging.info("GSTAR reconstruction server waiting for connection on " + str(host) + ":" + str(port))

    connection, addr = server_socket.accept()
    logging.info("Incoming connection from " + str(addr) + ", let's go!")

    con = ConnectionBuffer(connection)
    con.receive_messages()

    return con, server_socket


def ismrmrd_flags_to_bitmask(list_of_flags):
    """!
    @brief Reverse functionality of bitmask_to_flags. Creates the bitmask value to be set as a property of the
           ismrmrd.Acquistion objects from the list of ismrmrd flags.

    @param list_of_flags: (list) List of active flags as strings, e.g. ["ACQ_IS_REVERSE", "ACQ_IS_PHASECORR_DATA"]

    @return
        - (int) The bitmask value as a 64 bit integer, which is the bitmask of the acquisition flags.

    @author Jörn Huber, GitHub Copilot (GPT-4.1)
    """
    bitmask = 0
    inv_flag_dict = dict(zip(IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.values(), IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.keys()))
    for flag in list_of_flags:
        if flag in inv_flag_dict:
            ident = inv_flag_dict[flag] - 1
            bitmask |= (1 << ident)

    return bitmask


def bitmask_to_flags(flag_value):
    """!
    @brief Function which transforms the acqusition bitmask into corresponding ismrmrd flags such as IS_ACQ_REVERSE etc.

    @param flag_value: (int) The flag value as a 64 bit integer, which is the bitmask of the acquisition flags.

    @return
        - (list) A list of active flags as strings, e.g. ["ACQ_IS_REVERSE", "ACQ_IS_PHASECORR_DATA"]

    @author Jörn Huber, GitHub Copilot (GPT-4.1)
    """
    active_flags = [name for bit, name in IsmrmrdConstants.ISMRMRD_ACQ_FLAGS.items() if
                    (flag_value & (1 << (bit - 1))) != 0]

    return active_flags


def numpy_array_to_ismrmrd_acqs(list_of_np_ksp_arrays, list_of_ksp_array_flags, read_dir, phase_dir, slice_dir, position, list_of_trajectories = None):
    """!
    @brief Transforms k-space data which is given in form of numpy arrays into acquisiton objects, which can be sent
           to a server using client_to_stream.

    @param list_of_np_ksp_arrays: (list[np.ndarray]) A list of numpy arrays, which e.g. contains imaging data and phase-
                                  correction data.
    @param list_of_ksp_array_flags: (list[list]) A list whose entries correspond to the desired static flags of
                                    corresponding entries in the numpy arrays e.g.
                                    [["ACQ_USER1"],["ACQ_USER1", "ACQ_IS_PHASECORR_DATA"]]
    @param read_dir: (np.ndarray[3]) 3D array, containining information about the read direction. It is assumed that
                                     the read direction is the same for all acquisitions.
    @param phase_dir: (np.ndarray[3]) 3D array, containining information about the phase direction. It is assumed that
                                     the phase direction is the same for all acquisitions.
    @param slice_dir: (np.ndarray[3]) 3D array, containining information about the slice direction. It is assumed that
                                     the slice direction is the same for all acquisitions.
    @param position: (np.ndarray[3]) 3D array, containining information about the slice/slab position. It is assumed that
                                     the slice/slab position is the same for all acquisitions.
    @param list_of_trajectories: (list) A list, containing trajectory information for each data point

    @return
        - (list[ismrmrd.Acquisition]) A list of acquisition objects, ready to be sent to the stream.

    @author Jörn Huber
    """

    list_of_acqs = []

    for i_array in range(0, len(list_of_np_ksp_arrays)):
        ismrmrd_basic_flags = list_of_ksp_array_flags[i_array]

        np_array = list_of_np_ksp_arrays[i_array]
        np_ksp_array_expand = np.copy(np_array)
        while len(np_ksp_array_expand.shape) < 11:
            np_ksp_array_expand = np.expand_dims(np_ksp_array_expand, len(np_ksp_array_expand.shape))

        if list_of_trajectories is not None:
            np_traj = list_of_trajectories[i_array]
            np_traj_expand = np.copy(np_traj)
            while len(np_traj_expand.shape) < 12:
                np_traj_expand = np.expand_dims(np_traj_expand, len(np_traj_expand.shape) - 1)

        num_col, num_cha, num_lin, num_par, num_slice, num_set, num_pha, num_contrast, num_rep, num_ave, num_segment \
            = np_ksp_array_expand.shape

        i_acq_time_stamp = 0
        for i_seg in range(num_segment):
            for i_ave in range(num_ave):
                for i_rep in range(num_rep):
                    for i_contrast in range(num_contrast):
                        for i_pha in range(num_pha):
                            for i_set in range(num_set):
                                for i_slc in range(num_slice):
                                    for i_par in range(num_par):
                                        for i_lin in range(num_lin):

                                            acq_data = np.transpose(np_ksp_array_expand[:,
                                                                                        :,
                                                                                        i_lin,
                                                                                        i_par,
                                                                                        i_slc,
                                                                                        i_set,
                                                                                        i_pha,
                                                                                        i_contrast,
                                                                                        i_rep,
                                                                                        i_ave,
                                                                                        i_seg], [1, 0])

                                            if not acq_data.all() == 0:

                                                if list_of_trajectories is not None:
                                                    traj_data = np_traj_expand[:,
                                                                               0,
                                                                               i_lin,
                                                                               i_par,
                                                                               i_slc,
                                                                               i_set,
                                                                               i_pha,
                                                                               i_contrast,
                                                                               i_rep,
                                                                               i_ave,
                                                                               i_seg,
                                                                               :]
                                                    acq = ismrmrd.Acquisition.from_array(acq_data, traj_data)
                                                else:
                                                    acq = ismrmrd.Acquisition.from_array(acq_data)

                                                acq.acquisition_time_stamp = i_acq_time_stamp
                                                acq.idx.repetition = i_rep  # pylint: disable=maybe-no-member
                                                acq.idx.average = i_ave  # pylint: disable=maybe-no-member
                                                acq.idx.contrast = i_contrast  # pylint: disable=maybe-no-member
                                                acq.idx.phase = i_pha  # pylint: disable=maybe-no-member
                                                acq.idx.set = i_set  # pylint: disable=maybe-no-member
                                                acq.idx.segment = i_seg  # pylint: disable=maybe-no-member
                                                acq.idx.slice = i_slc  # pylint: disable=maybe-no-member
                                                acq.idx.kspace_encode_step_2 = i_par  # pylint: disable=maybe-no-member
                                                acq.idx.kspace_encode_step_1 = i_lin  # pylint: disable=maybe-no-member
                                                acq.read_dir[0] = read_dir[0]  # pylint: disable=maybe-no-member
                                                acq.read_dir[1] = read_dir[1]  # pylint: disable=maybe-no-member
                                                acq.read_dir[2] = read_dir[2]  # pylint: disable=maybe-no-member
                                                acq.phase_dir[0] = phase_dir[0]  # pylint: disable=maybe-no-member
                                                acq.phase_dir[1] = phase_dir[1]  # pylint: disable=maybe-no-member
                                                acq.phase_dir[2] = phase_dir[2]  # pylint: disable=maybe-no-member
                                                acq.slice_dir[0] = slice_dir[0]  # pylint: disable=maybe-no-member
                                                acq.slice_dir[1] = slice_dir[1]  # pylint: disable=maybe-no-member
                                                acq.slice_dir[2] = slice_dir[2]  # pylint: disable=maybe-no-member
                                                acq.position[0] = position[0]  # pylint: disable=maybe-no-member
                                                acq.position[1] = position[1]  # pylint: disable=maybe-no-member
                                                acq.position[2] = position[2]  # pylint: disable=maybe-no-member

                                                ismrmrd_flags = ismrmrd_basic_flags.copy()

                                                # Acquisition flags for encode step 1 e.g. in plane phase encoding
                                                # direction
                                                if i_lin == 0:
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_ENCODE_STEP1")
                                                elif i_lin == num_lin - 1:
                                                    ismrmrd_flags.append("ACQ_LAST_IN_ENCODE_STEP1")

                                                # Acquisition flags for encode step 2 e.g. partition phase encoding
                                                # direction
                                                if all(x == 0 for x in (i_lin, i_par)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_ENCODE_STEP2")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_ENCODE_STEP2")

                                                # Acquisition flags for slice encoding
                                                # direction
                                                if all(x == 0 for x in (i_lin, i_par, i_slc)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_SLICE")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_SLICE")

                                                # Acquisition flags for set encoding
                                                # direction
                                                if all(x == 0 for x in (i_lin, i_par, i_slc, i_set)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_SET")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1
                                                      and i_set == num_set - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_SET")

                                                # Acquisition flags for phase encoding
                                                # direction
                                                if all(x == 0 for x in (i_lin, i_par, i_slc, i_set, i_pha)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_PHASE")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1
                                                      and i_set == num_set - 1
                                                      and i_pha == num_pha - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_PHASE")

                                                # Acquisition flags for contrast encoding
                                                # direction
                                                if all(x == 0 for x in
                                                       (i_lin, i_par, i_slc, i_set, i_pha, i_contrast)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_CONTRAST")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1
                                                      and i_set == num_set - 1
                                                      and i_pha == num_pha - 1
                                                      and i_contrast == num_contrast - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_CONTRAST")

                                                # Acquisition flags for repetition encoding
                                                # direction
                                                if all(x == 0 for x in
                                                       (i_lin, i_par, i_slc, i_set, i_pha, i_contrast, i_rep)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_REPETITION")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1
                                                      and i_set == num_set - 1
                                                      and i_pha == num_pha - 1
                                                      and i_contrast == num_contrast - 1
                                                      and i_rep == num_rep - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_REPETITION")

                                                # Acquisition flags for average encoding
                                                # direction
                                                if all(x == 0 for x in
                                                       (i_lin, i_par, i_slc, i_set, i_pha, i_contrast, i_rep, i_ave)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_AVERAGE")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1
                                                      and i_set == num_set - 1
                                                      and i_pha == num_pha - 1
                                                      and i_contrast == num_contrast - 1
                                                      and i_rep == num_rep - 1
                                                      and i_ave == num_ave - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_AVERAGE")

                                                # Acquisition flags for segment encoding
                                                # direction
                                                if all(x == 0 for x in
                                                       (i_lin, i_par, i_slc, i_set, i_pha, i_contrast, i_rep, i_ave, i_seg)):
                                                    ismrmrd_flags.append("ACQ_FIRST_IN_SEGMENT")
                                                elif (i_lin == num_lin - 1
                                                      and i_par == num_par - 1
                                                      and i_slc == num_slice - 1
                                                      and i_set == num_set - 1
                                                      and i_pha == num_pha - 1
                                                      and i_contrast == num_contrast - 1
                                                      and i_rep == num_rep - 1
                                                      and i_ave == num_ave - 1
                                                      and i_seg == num_segment - 1):
                                                    ismrmrd_flags.append("ACQ_LAST_IN_SEGMENT")

                                                acq.flags = ismrmrd_flags_to_bitmask(ismrmrd_flags)
                                                list_of_acqs.append(acq)
                                                i_acq_time_stamp = i_acq_time_stamp + 1
    return list_of_acqs


def identify_readout_type_from_acqs(list_of_acqs):
    """!
    @brief Identifies the type of readout trajectory based on the received list of acquisitions. 

    @param list_of_acqs: (list) List of ismrmrd.Acquisition objects.
    @param encoded_space: (ISMRMRD encoded space object) Encoded space dimensions
    @param encoded_space: (ISMRMRD recon space object) Encoded space dimensions
    @param W: Noise de-correlation matrix of size (num_cha, num_cha).

    @return
        - (int) Integer, indicating the type of readout with
                READOUT_TYPE_CARTESIAN = 1
                READOUT_TYPE_CARTESIAN_RAMP = 2
                READOUT_TYPE_NONCARTESIAN_2D = 3
                READOUT_TYPE_NONCARTESIAN_3D = 4
    """

    readout_type = -1
    is_ramp_samp = False
    is_propeller = False
    blade_dim = -1

    if list_of_acqs[0].traj.shape[1] == 0:

        logging.info("GSTAR Recon:   Trajectory information was not provided, assuming Cartesian")
        readout_type = IsmrmrdConstants.READOUT_TYPE_CARTESIAN
        is_ramp_samp = False

    elif len(list_of_acqs) == 1:

        round_traj = np.round(list_of_acqs[0].traj * 100.0) / 100.0

        traj_grad_x = np.round(np.gradient(list_of_acqs[0].traj[:, 0]) * 100.0) / 100.0
        traj_grad_y = np.round(np.gradient(list_of_acqs[0].traj[:, 1]) * 100.0) / 100.0
        traj_grad_z = np.round(np.gradient(list_of_acqs[0].traj[:, 2]) * 100.0) / 100.0

        if np.count_nonzero(traj_grad_y) == 0 and np.count_nonzero(traj_grad_z) == 0:

            readout_type = IsmrmrdConstants.READOUT_TYPE_CARTESIAN

            if len(np.unique(traj_grad_x)) == 1:
                is_ramp_samp = False
            else:
                is_ramp_samp = True

        elif np.count_nonzero(traj_grad_y) != 0 and np.count_nonzero(traj_grad_z) == 0:

            readout_type = IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D
            is_ramp_samp = False

        elif np.count_nonzero(traj_grad_y) != 0 and np.count_nonzero(traj_grad_z) != 0:

            readout_type = IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D
            is_ramp_samp = False

    else:

        trajectory_dict = dict()
        for acq in list_of_acqs:

            grad_x = np.round(np.gradient(acq.traj[:, 0]) * 100.0) / 100.0
            grad_y = np.round(np.gradient(acq.traj[:, 1]) * 100.0) / 100.0
            grad_z = np.round(np.gradient(acq.traj[:, 2]) * 100.0) / 100.0
            pos_x = np.round(acq.traj[:, 0] * 100.0) / 100.0
            pos_y = np.round(acq.traj[:, 1] * 100.0) / 100.0
            pos_z = np.round(acq.traj[:, 2] * 100.0) / 100.0

            if acq.idx.kspace_encode_step_1 not in trajectory_dict:

                trajectory_dict[acq.idx.kspace_encode_step_1] = dict()
                trajectory_dict[acq.idx.kspace_encode_step_1]['gx'] = grad_x
                trajectory_dict[acq.idx.kspace_encode_step_1]['gy'] = grad_y
                trajectory_dict[acq.idx.kspace_encode_step_1]['gz'] = grad_z
                trajectory_dict[acq.idx.kspace_encode_step_1]['posx'] = pos_x
                trajectory_dict[acq.idx.kspace_encode_step_1]['posy'] = pos_y
                trajectory_dict[acq.idx.kspace_encode_step_1]['posz'] = pos_z
                trajectory_dict[acq.idx.kspace_encode_step_1]['idx'] = acq.idx

            else:

                prev_encoding_traj = trajectory_dict[acq.idx.kspace_encode_step_1]

                if not is_propeller and not ( (prev_encoding_traj['posx'] == pos_x).all() or (prev_encoding_traj['posy'] == pos_y).all()):
                    is_propeller = True
                    if acq.idx.phase != prev_encoding_traj['idx'].phase:
                        blade_dim = 'PHS'
                    elif acq.idx.repetition != prev_encoding_traj['idx'].repetition:
                        blade_dim = 'REP'
                    elif acq.idx.set != prev_encoding_traj['idx'].set:
                        blade_dim = 'SET'

        for ksp_enc_1_ind in trajectory_dict:

            traj = trajectory_dict[ksp_enc_1_ind]

            if np.count_nonzero(traj['gy']) != 0 and np.count_nonzero(traj['gz']) != 0:
                readout_type = IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D
                is_ramp_samp = False
                break

            elif np.count_nonzero(traj['gy']) != 0 and np.count_nonzero(traj['gz']) == 0:
                readout_type = IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D
                is_ramp_samp = False
                break

            if  readout_type == -1 and np.count_nonzero(traj['gy']) == 0 and np.count_nonzero(traj['gz']) == 0:

                readout_type = IsmrmrdConstants.READOUT_TYPE_CARTESIAN

                if len(np.unique(traj['gx'])) == 1:
                    is_ramp_samp = False
                else:
                    is_ramp_samp = True

    return readout_type, is_ramp_samp, is_propeller, blade_dim


def ismrmrd_acqs_to_numpy_array(list_of_acqs, encoded_space = None, recon_space = None, W = None, os_factor = 1):
    """!
    @brief Sort k-space data from acquisitions into numpy array structures for further processing.
    @details The function first analyzes the maximum idx indices which are available in the list of provided
             acquisitions. Maximum indices are used to create the numpy structure and to sort the data into that
             structure. If acquisitions provide non-Cartesian data, the data is first regridded to the Cartesian grid
             as defined by the encoded space. If data was sampled using ramp sampling, regridding of individual readouts
             is first applied.

    @param list_of_acqs: (list) List of ismrmrd.Acquisition objects.
    @param encoded_space: (ISMRMRD encoded space object) Encoded space dimensions
    @param encoded_space: (ISMRMRD recon space object) Encoded space dimensions
    @param W: Noise de-correlation matrix of size (num_cha, num_cha).

    @return
        - (np.ndarray) 11-D Numpy array of size (number_of_samples, max_kspace_encoding_pe1, max_kspace_encoding_pe2,
                       num_active_channels, max_slice, max_set, max_phase, max_contrast, max_repetition, max_average,
                       max_segment) with sorted acquisition.

    @author Jörn Huber, GitHub Copilot (GPT-4.1)
    """

    readout_type, is_ramp_sample, _, _ = identify_readout_type_from_acqs(list_of_acqs)
    ramp_sampe_string = ''
    if is_ramp_sample:
        ramp_sampe_string = " with 1D resampling"
    if readout_type == IsmrmrdConstants.READOUT_TYPE_CARTESIAN:
        logging.info('GSTAR Recon:   Cartesian readout' + ramp_sampe_string)
    elif readout_type == IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:
        logging.info('GSTAR Recon:   Non-Cartesian 2D readout')
    elif readout_type == IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:
        logging.info('GSTAR Recon:   Non-Cartesian 3D readout')

    num_active_channels = list_of_acqs[0].active_channels

    number_of_samples = list_of_acqs[0].number_of_samples  # encoded_space.matrixSize.x
    if encoded_space is not None:
        max_kspace_encoding_pe1 = encoded_space.matrixSize.y
        max_kspace_encoding_pe2 = encoded_space.matrixSize.z
    else:
        max_kspace_encoding_pe1 = max(list_of_acqs, key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1 + 1
        max_kspace_encoding_pe2 = max(list_of_acqs, key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2 + 1

    max_slice = max(list_of_acqs, key=lambda acq: acq.idx.slice).idx.slice + 1
    max_segment = max(list_of_acqs, key=lambda acq: acq.idx.segment).idx.segment + 1
    max_set = max(list_of_acqs, key=lambda acq: acq.idx.set).idx.set + 1
    max_phase = max(list_of_acqs, key=lambda acq: acq.idx.phase).idx.phase + 1
    max_contrast = max(list_of_acqs, key=lambda acq: acq.idx.contrast).idx.contrast + 1
    max_average = max(list_of_acqs, key=lambda acq: acq.idx.average).idx.average + 1
    max_repetition = max(list_of_acqs, key=lambda acq: acq.idx.repetition).idx.repetition + 1

    acq_data_np = np.zeros((number_of_samples,
                            num_active_channels,
                            max_kspace_encoding_pe1,
                            max_kspace_encoding_pe2,
                            max_slice,
                            max_set,
                            max_phase,
                            max_contrast,
                            max_repetition,
                            max_average,
                            max_segment), dtype=complex)

    for acq in list_of_acqs:
        acq_flags = bitmask_to_flags(acq.getHead().flags)
        data = np.transpose(acq.data, (1, 0))

        if any('ACQ_IS_REVERSE' in s for s in acq_flags):
            data = np.flipud(data)

        acq_data_np[:,
                    :,
                    acq.idx.kspace_encode_step_1,
                    acq.idx.kspace_encode_step_2,
                    acq.idx.slice,
                    acq.idx.set,
                    acq.idx.phase,
                    acq.idx.contrast,
                    acq.idx.repetition,
                    acq.idx.average,
                    acq.idx.segment] = data

    if readout_type == IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:

        acq_data_np_grid = np.zeros((encoded_space.matrixSize.x,
                                    num_active_channels,
                                    encoded_space.matrixSize.x,
                                    max_kspace_encoding_pe2,
                                    max_slice,
                                    max_set,
                                    max_phase,
                                    max_contrast,
                                    max_repetition,
                                    max_average,
                                    max_segment), dtype=complex)

        # --> Based on mri-nufft (see third_party_licenses.txt)

        # Prepare sampling trajectory
        samples_loc = mrinufft.initialize_2D_radial(Nc=max_kspace_encoding_pe1, Ns=list_of_acqs[0].number_of_samples)
        traj_dict = dict()
        for i_acq in range(len(list_of_acqs)):
            if list_of_acqs[i_acq].idx.kspace_encode_step_1 not in traj_dict:
                traj_dict[list_of_acqs[i_acq].idx.kspace_encode_step_1] = list_of_acqs[i_acq].traj[:, 0:-1]
                samples_loc[i_acq, :, :] = list_of_acqs[i_acq].traj[:, 0:-1]

        samples_loc = samples_loc / np.max(samples_loc) * math.pi

        # Prepare NUFFT Operator
        density = voronoi(samples_loc.reshape(-1, 2))
        NufftOperator = mrinufft.get_operator("finufft")
        nufft = NufftOperator(
            samples_loc.reshape(-1, 2), shape=(encoded_space.matrixSize.x, encoded_space.matrixSize.x), density=density,
            n_coils=1
        )

        # <-- Based on mri-nufft (see third_party_licenses.txt)

        for i_rep in range(0, max_repetition):
            for i_ave in range(0, max_average):
                for i_con in range(0, max_contrast):
                    for i_phs in range(0, max_phase):
                        for i_set in range(0, max_set):
                            for i_seg in range(0, max_segment):
                                for i_slc in range(0, max_slice):
                                    for i_par in range(0, max_kspace_encoding_pe2):
                                        for i_cha in range(0, max_slice):
                                            grid_data = acq_data_np[:, i_cha, :,
                                                                    i_par, i_slc, i_set, i_phs, i_con, i_rep,
                                                                    i_ave, i_seg]

                                            # --> Based on mri-nufft (see third_party_licenses.txt)
                                            cart_data = nufft.adj_op(np.transpose(grid_data, [1, 0]).flatten())
                                            # <-- Based on mri-nufft (see third_party_licenses.txt)

                                            cart_data = np.fft.fftshift(
                                                np.fft.ifft(np.fft.fftshift(cart_data, axes=0), axis=0), axes=0)
                                            cart_data = np.fft.fftshift(
                                                np.fft.ifft(np.fft.fftshift(cart_data, axes=1), axis=1), axes=1)

                                            acq_data_np_grid[:, i_cha, :,
                                                             i_par, i_slc, i_set, i_phs, i_con, i_rep,
                                                             i_ave, i_seg] = cart_data

        acq_data_np = acq_data_np_grid

    elif readout_type == IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:

        acq_data_np_grid = np.zeros((encoded_space.matrixSize.x,
                                    num_active_channels,
                                    encoded_space.matrixSize.x,
                                    encoded_space.matrixSize.x,
                                    max_slice,
                                    max_set,
                                    max_phase,
                                    max_contrast,
                                    max_repetition,
                                    max_average,
                                    max_segment), dtype=complex)

        # --> Based on mri-nufft (see third_party_licenses.txt)

        # Prepare sampling trajectory
        samples_loc = mrinufft.initialize_3D_golden_means_radial(Nc=max_kspace_encoding_pe1, Ns=list_of_acqs[0].number_of_samples)
        traj_dict = dict()
        for i_acq in range(len(list_of_acqs)):
            if list_of_acqs[i_acq].idx.kspace_encode_step_1 not in traj_dict:
                traj_dict[list_of_acqs[i_acq].idx.kspace_encode_step_1] = list_of_acqs[i_acq].traj
                samples_loc[i_acq, :, :] = list_of_acqs[i_acq].traj

        samples_loc = samples_loc / np.max(samples_loc) * math.pi

        # Prepare NUFFT Operator
        density = voronoi(samples_loc.reshape(-1, 3))
        NufftOperator = mrinufft.get_operator("finufft")
        nufft = NufftOperator(
            samples_loc.reshape(-1, 3), shape=(encoded_space.matrixSize.x, encoded_space.matrixSize.x, encoded_space.matrixSize.x), density=density,
            n_coils=1
        )

        # <-- Based on mri-nufft (see third_party_licenses.txt)

        for i_rep in range(0, max_repetition):
            for i_ave in range(0, max_average):
                for i_con in range(0, max_contrast):
                    for i_phs in range(0, max_phase):
                        for i_set in range(0, max_set):
                            for i_seg in range(0, max_segment):
                                for i_slc in range(0, max_slice):
                                    for i_cha in range(0, max_slice):
                                        grid_data = acq_data_np[:, i_cha, :,
                                                                0, i_slc, i_set, i_phs, i_con, i_rep,
                                                                i_ave, i_seg]

                                        # --> Based on mri-nufft (see third_party_licenses.txt)
                                        cart_data = nufft.adj_op(np.transpose(grid_data, [1, 0]).flatten())
                                        # <-- Based on mri-nufft (see third_party_licenses.txt)

                                        cart_data = np.fft.fftshift(
                                            np.fft.ifft(np.fft.fftshift(cart_data, axes=0), axis=0), axes=0)
                                        cart_data = np.fft.fftshift(
                                            np.fft.ifft(np.fft.fftshift(cart_data, axes=1), axis=1), axes=1)
                                        cart_data = np.fft.fftshift(
                                            np.fft.ifft(np.fft.fftshift(cart_data, axes=2), axis=2), axes=2)

                                        acq_data_np_grid[:, i_cha, :,
                                                         :, i_slc, i_set, i_phs, i_con, i_rep,
                                                         i_ave, i_seg] = cart_data

        acq_data_np = acq_data_np_grid

    elif is_ramp_sample and (readout_type == IsmrmrdConstants.READOUT_TYPE_CARTESIAN):

        regrid_traj = list_of_acqs[0].traj[:, 0]

        regrid_traj = regrid_traj - np.min(regrid_traj)
        regrid_traj = regrid_traj / np.max(regrid_traj)
        regrid_traj = regrid_traj * list_of_acqs[0].number_of_samples

        if regrid_traj[0] > regrid_traj[-1]:
            integer_grid = np.arange(list_of_acqs[0].number_of_samples, 0, -1)
        else:
            integer_grid = np.arange(0, list_of_acqs[0].number_of_samples)

        for i_rep in range(0, max_repetition):
            for i_ave in range(0, max_average):
                for i_con in range(0, max_contrast):
                    for i_phs in range(0, max_phase):
                        for i_set in range(0, max_set):
                            for i_seg in range(0, max_segment):
                                for i_slc in range(0, max_slice):
                                    for i_par in range(0, max_kspace_encoding_pe2):
                                        for i_cha in range(0, max_slice):
                                            for i_pe in range(0, max_kspace_encoding_pe1):

                                                data = acq_data_np[:, i_cha, i_pe, i_par, i_slc, i_set, i_phs,
                                                                   i_con, i_rep, i_ave, i_seg]

                                                regrid_interpolator = interp1d(regrid_traj, np.squeeze(data), kind='cubic', fill_value='extrapolate')
                                                resampled_data = regrid_interpolator(integer_grid)

                                                acq_data_np[:, i_cha, i_pe, i_par, i_slc, i_set, i_phs, i_con, i_rep, i_ave, i_seg] = resampled_data

    # Last step: We want to remove the readout oversampling
    if os_factor > 1:

        acq_data_np = helpers.remove_readout_os(acq_data_np, 0, os_factor)

        if readout_type == IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D: # We gridded, so we need to remove it along ro and pe1
            acq_data_np = helpers.remove_readout_os(acq_data_np, 2, os_factor)

        if readout_type == IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D: # We gridded, so we need to remove it along ro and pe1 and pe2
            acq_data_np = helpers.remove_readout_os(acq_data_np, 2, os_factor)
            acq_data_np = helpers.remove_readout_os(acq_data_np, 3, os_factor)

    return acq_data_np


def numpy_array_to_ismrmrd_image(nd_image, list_of_acqs, xml_header, image_series_index, meas_idx, add_series_string = '', recon_history = '', scaling_factor = 1.0):
    """!
    @brief Sort k-space data from acquisitions into numpy array structures for further processing.
    @details The function first analyzes the maximum idx indices which are available in the list of provided
             acquisitions. Maximum indices are used to create the numpy structure and to sort the data into that
             structure.

    @param nd_image: (np.ndarray) 3D or 4D numpy array of size (num_col, num_lin, num_par, num_cha)
    @param list_of_acqs: (list[ismrmrd.Acquisition]) List of acquisition objects.
    @param xml_header: (ismrmrd.header) Ismrmrd xml header
    @param image_series_index: (int) Image series index, which is written into image information
    @param meas_idx: (mrpy_ismrmrd_tools.MeasIDX) Measurement idx object, indicating respective counters.
    @param add_series_string: (string) Additional series string.

    @return
        - (ismrmrd.Image) ISMRMRD image object.

    @author Jörn Huber
    """

    # First, we need to identify the correct reference acquisition from the list of acquisitions based on idx entries.
    # This will set a first set of entries, e.g. orientation and position information.
    target_acq = -1
    for acq in list_of_acqs:
        if acq.idx.contrast == meas_idx.contrast and acq.idx.phase == meas_idx.phase and acq.idx.repetition == meas_idx.repetition and acq.idx.set == meas_idx.set and acq.idx.slice == meas_idx.slice:
            target_acq = acq
            break

    image = ismrmrd.Image.from_array(nd_image.transpose(), acquisition=target_acq, transpose=False)
    image.image_index = 0
    image.image_series_index = image_series_index

    try:
        image.field_of_view[0] = xml_header.encoding[0].reconSpace.fieldOfView_mm.z #pylint: disable=maybe-no-member
        image.field_of_view[1] = xml_header.encoding[0].reconSpace.fieldOfView_mm.y #pylint: disable=maybe-no-member
        image.field_of_view[2] = xml_header.encoding[0].reconSpace.fieldOfView_mm.x #pylint: disable=maybe-no-member
    except:
        image.field_of_view[0] = 0 #pylint: disable=maybe-no-member
        image.field_of_view[1] = 0 #pylint: disable=maybe-no-member
        image.field_of_view[2] = 0 #pylint: disable=maybe-no-member

    # TODO: Missing entries: measurement_uid, physiology_time_stamp

    # We create the meta information of the image and add it to the image
    meta = ismrmrd.Meta({'DataRole': 'Image',
                         'ImageHistory': ['GSTARRecon', 'PYTHON', recon_history],
                         'WindowCenter': str((np.max(nd_image.flatten()) + 1) / 2),
                         'WindowWidth': str((np.max(nd_image.flatten()) + 1))})

    meta['seriesDescription'] = 'ISMRMRD Series' + add_series_string  # Will be overwritten if additional information is available
    meta['studyDescription'] = 'ISMRMRD Study'  # Will be overwritten if additional information is available
    meta['scalingFactor'] = str(scaling_factor)

    # Write acqusition info to meta object
    if hasattr(xml_header.acquisitionSystemInformation, 'deviceID') and xml_header.acquisitionSystemInformation.deviceID is not None:
        meta['deviceID'] = xml_header.acquisitionSystemInformation.deviceID
    if hasattr(xml_header.acquisitionSystemInformation, 'institutionName') and xml_header.acquisitionSystemInformation.institutionName is not None:
        meta['institutionName'] = xml_header.acquisitionSystemInformation.institutionName
    if hasattr(xml_header.acquisitionSystemInformation, 'systemFieldStrength_T') and xml_header.acquisitionSystemInformation.systemFieldStrength_T is not None:
        meta['systemFieldStrength_T'] = str(xml_header.acquisitionSystemInformation.systemFieldStrength_T)
    if hasattr(xml_header.acquisitionSystemInformation, 'systemModel') and xml_header.acquisitionSystemInformation.systemModel is not None:
        meta['systemModel'] = str(xml_header.acquisitionSystemInformation.systemModel)
    if hasattr(xml_header.acquisitionSystemInformation, 'systemVendor') and xml_header.acquisitionSystemInformation.systemVendor is not None:
        meta['systemVendor'] = str(xml_header.acquisitionSystemInformation.systemVendor)

    # Write experimental info to meta object
    if hasattr(xml_header.experimentalConditions, 'H1resonanceFrequency_Hz') and xml_header.experimentalConditions.H1resonanceFrequency_Hz is not None:
        meta['H1resonanceFrequency_Hz'] = xml_header.experimentalConditions.H1resonanceFrequency_Hz

    # Write measurement info to meta object
    if hasattr(xml_header.measurementInformation, 'frameOfReferenceUID') and xml_header.measurementInformation.frameOfReferenceUID is not None:
        meta['FrameOfReference'] = xml_header.measurementInformation.frameOfReferenceUID
    if hasattr(xml_header.measurementInformation, 'measurementID') and xml_header.measurementInformation.measurementID is not None:
        meta['measurementID'] = xml_header.measurementInformation.measurementID
    if hasattr(xml_header.measurementInformation, 'patientPosition') and xml_header.measurementInformation.patientPosition is not None:
        meta['patientPosition'] = xml_header.measurementInformation.patientPosition.value
    if hasattr(xml_header.measurementInformation, 'protocolName') and xml_header.measurementInformation.protocolName is not None:
        meta['protocolName'] = xml_header.measurementInformation.protocolName
    if hasattr(xml_header.measurementInformation, 'relativeTablePosition') and xml_header.measurementInformation.relativeTablePosition is not None:
        meta['relativeTablePosition'] = str(xml_header.measurementInformation.relativeTablePosition.z)
    if hasattr(xml_header.measurementInformation, 'seriesDescription') and xml_header.measurementInformation.seriesDescription is not None:
        meta['seriesDescription'] = str(xml_header.measurementInformation.seriesDescription) + add_series_string
    if hasattr(xml_header.measurementInformation, 'seriesInstanceUIDRoot') and xml_header.measurementInformation.seriesInstanceUIDRoot is not None:
        meta['seriesInstanceUIDRoot'] = str(xml_header.measurementInformation.seriesInstanceUIDRoot)
    if hasattr(xml_header.measurementInformation, 'sequenceName') and xml_header.measurementInformation.sequenceName is not None:
        meta['SequenceName'] = xml_header.measurementInformation.sequenceName

    # Write sequence parameters to meta object
    if hasattr(xml_header.sequenceParameters, 'TE') and xml_header.sequenceParameters.TE is not None:
        meta['EchoTime'] = str(xml_header.sequenceParameters.TE[0] / 1000.0)
    if hasattr(xml_header.sequenceParameters, 'TR') and xml_header.sequenceParameters.TR is not None:
        meta['RepetitionTime'] = str(xml_header.sequenceParameters.TR[0] / 1000.0)
    if hasattr(xml_header.sequenceParameters, 'echo_spacing') and xml_header.sequenceParameters.echo_spacing is not None:
        try:
            meta['echo_spacing'] = str(xml_header.sequenceParameters.echo_spacing[0])
        except:
            meta['echo_spacing'] = str(xml_header.sequenceParameters.echo_spacing)
    if hasattr(xml_header.sequenceParameters, 'flipAngle_deg') and xml_header.sequenceParameters.flipAngle_deg is not None:
        try:
            meta['flipAngle_deg'] = str(xml_header.sequenceParameters.flipAngle_deg[0])
        except:
            meta['flipAngle_deg'] = str(xml_header.sequenceParameters.flipAngle_deg)

    # Write study parameters to meta object
    if hasattr(xml_header.studyInformation, 'bodyPartExamined') and xml_header.studyInformation.bodyPartExamined is not None:
        meta['bodyPartExamined'] = str(xml_header.studyInformation.bodyPartExamined)
    if hasattr(xml_header.studyInformation, 'referringPhysicianName') and xml_header.studyInformation.referringPhysicianName is not None:
        meta['referringPhysicianName'] = str(xml_header.studyInformation.referringPhysicianName)
    if hasattr(xml_header.studyInformation, 'studyDescription') and xml_header.studyInformation.studyDescription is not None:
        meta['studyDescription'] = str(xml_header.studyInformation.studyDescription)
    if hasattr(xml_header.studyInformation, 'studyID') and xml_header.studyInformation.studyID is not None:
        meta['studyID'] = str(xml_header.studyInformation.studyID)
    if hasattr(xml_header.studyInformation, 'studyInstanceUID') and xml_header.studyInformation.studyInstanceUID is not None:
        meta['studyInstanceUID'] = str(xml_header.studyInformation.studyInstanceUID)

    # Write subject information to meta object
    if hasattr(xml_header.subjectInformation, 'patientBirthdate') and xml_header.subjectInformation.patientBirthdate is not None:
        meta['patientBirthdate'] = str(xml_header.subjectInformation.patientBirthdate)
    if hasattr(xml_header.subjectInformation, 'patientGender') and xml_header.subjectInformation.patientGender is not None:
        meta['patientGender'] = str(xml_header.subjectInformation.patientGender)
    if hasattr(xml_header.subjectInformation, 'patientHeight_m') and xml_header.subjectInformation.patientHeight_m is not None:
        meta['patientHeight_m'] = str(xml_header.subjectInformation.patientHeight_m)
    if hasattr(xml_header.subjectInformation, 'patientID') and xml_header.subjectInformation.patientID is not None:
        meta['patientID'] = str(xml_header.subjectInformation.patientID)
    if hasattr(xml_header.subjectInformation, 'patientName') and xml_header.subjectInformation.patientName is not None:
        meta['patientName'] = str(xml_header.subjectInformation.patientName)
    if hasattr(xml_header.subjectInformation, 'patientWeight_kg') and xml_header.subjectInformation.patientWeight_kg is not None:
        meta['patientWeight_kg'] = str(xml_header.subjectInformation.patientWeight_kg)

    # Write encoding information to meta object
    try:
        meta['encodedFoV_x'] = str(xml_header.encoding[0].encodedSpace.fieldOfView_mm.x)
        meta['encodedFoV_y'] = str(xml_header.encoding[0].encodedSpace.fieldOfView_mm.y)
        meta['encodedFoV_z'] = str(xml_header.encoding[0].encodedSpace.fieldOfView_mm.z)
        meta['encodedMat_x'] = str(xml_header.encoding[0].encodedSpace.matrixSize.x)
        meta['encodedMat_y'] = str(xml_header.encoding[0].encodedSpace.matrixSize.y)
        meta['encodedMat_z'] = str(xml_header.encoding[0].encodedSpace.matrixSize.z)
    except:
        meta['encodedFoV_x'] = str(0)
        meta['encodedFoV_y'] = str(0)
        meta['encodedFoV_z'] = str(0)
        meta['encodedMat_x'] = str(0)
        meta['encodedMat_y'] = str(0)
        meta['encodedMat_z'] = str(0)

    try:
        meta['reconFoV_x'] = str(xml_header.encoding[0].reconSpace.fieldOfView_mm.x)
        meta['reconFoV_y'] = str(xml_header.encoding[0].reconSpace.fieldOfView_mm.y)
        meta['reconFoV_z'] = str(xml_header.encoding[0].reconSpace.fieldOfView_mm.z)
        meta['reconMat_x'] = str(xml_header.encoding[0].reconSpace.matrixSize.x)
        meta['reconMat_y'] = str(xml_header.encoding[0].reconSpace.matrixSize.y)
        meta['reconMat_z'] = str(xml_header.encoding[0].reconSpace.matrixSize.z)
    except:
        meta['reconFoV_x'] = str(0)
        meta['reconFoV_y'] = str(0)
        meta['reconFoV_z'] = str(0)
        meta['reconMat_x'] = str(0)
        meta['reconMat_y'] = str(0)
        meta['reconMat_z'] = str(0)

    meta['ReadPhaseSeqSwap'] = str(0)

    meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]),
                           "{:.18f}".format(image.getHead().read_dir[1]),
                           "{:.18f}".format(image.getHead().read_dir[2])]
    meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]),
                              "{:.18f}".format(image.getHead().phase_dir[1]),
                              "{:.18f}".format(image.getHead().phase_dir[2])]
    meta['ImageSliceNormDir'] = ["{:.18f}".format(image.getHead().slice_dir[0]),
                                 "{:.18f}".format(image.getHead().slice_dir[1]),
                                 "{:.18f}".format(image.getHead().slice_dir[2])]

    meta['SlicePosLightMarker'] = []
    center_par = int(nd_image.shape[2]/2)
    pos_incr = xml_header.encoding[0].reconSpace.fieldOfView_mm.z/xml_header.encoding[0].reconSpace.matrixSize.z
    for i_par in range(nd_image.shape[2]):
        shift_norm = float(i_par-center_par)*pos_incr
        pos_slice_addition = [image.getHead().slice_dir[0] * shift_norm,
                              image.getHead().slice_dir[1] * shift_norm,
                              image.getHead().slice_dir[2] * shift_norm]

        meta['SlicePosLightMarker'].append("{:.18f}".format(image.getHead().position[0] + pos_slice_addition[0]))
        meta['SlicePosLightMarker'].append("{:.18f}".format(image.getHead().position[1] + pos_slice_addition[1]))
        meta['SlicePosLightMarker'].append("{:.18f}".format(image.getHead().position[2] + pos_slice_addition[2]))

    meta_xml = meta.serialize()
    image.attribute_string = meta_xml
    image.meta = meta

    return image
