"""!
@brief Kspace to image module.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging
import mrpy_ismrmrd_tools as ismrmd_tools

class KspaceToImageModule:
    """!
    @brief This module applies FFT along those image dimensions which were not transformed ot image space by
           another algorithm before.
    """

    @staticmethod
    def __call__(connection_buffer):
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """
        return KspaceToImageModule.apply(connection_buffer)

    @staticmethod
    def apply(connection_buffer):
        """!
        @brief Applies FFT along those image dimensions which were not transformed ot image space by
               another algorithm before.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """

        if not connection_buffer.meas_data.is_ro_ft:
            logging.info("GSTAR Recon: RO iFFT")

            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(connection_buffer.meas_data.data['NP_IS_IMAGING'], axes=0), axis=0), axes=0)
            connection_buffer.is_ro_ft = True

        if not connection_buffer.meas_data.is_pe_ft:
            logging.info("GSTAR Recon: PE iFFT")

            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(connection_buffer.meas_data.data['NP_IS_IMAGING'], axes=2), axis=2), axes=2)
            connection_buffer.is_pe_ft = True

        num_rot = 0
        if connection_buffer.headers[0].acquisitionSystemInformation.systemVendor == 'Siemens':
            num_rot = 1
        elif connection_buffer.headers[0].acquisitionSystemInformation.systemVendor == 'gammaSTAR':
            num_rot = 3

        if connection_buffer.meas_data.imaging_readout_type == ismrmd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D or connection_buffer.meas_data.imaging_readout_type == ismrmd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:
            num_rot += 2

        for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
            for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                    for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                        for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                            for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2')):
                                im = np.squeeze(connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, 0])
                                connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, 0] = np.rot90(im, num_rot)

        return connection_buffer
