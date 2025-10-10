"""!
@brief FFT Module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging
import mrpy_ismrmrd_tools as ismrmrd_tools


class IFFTModule:
    """!
    @brief This module applies FFT along a specified image dimension.
    """

    @staticmethod
    def __call__(dim_str, connection_buffer):
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param dim_str: (String) Dimension string as defined in the mrpy_ismrmrd_tools.
        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """
        return IFFTModule.apply(connection_buffer, dim_str)

    @staticmethod
    def apply(connection_buffer, dim_str):
        """!
        @brief Applies FFT along a specified image dimension

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param dim_str: (String) Dimension string as defined in the mrpy_ismrmrd_tools.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """

        if not connection_buffer.meas_data.is_ro_ft:

            if connection_buffer.meas_data('NP_IS_IMAGING', dim_str) > 1:

                logging.info("GSTAR Recon: " + dim_str + " IFFT")
                dim_ind = ismrmrd_tools.IsmrmrdConstants.IDX_MAP[dim_str]

                connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(connection_buffer.meas_data.data['NP_IS_IMAGING'], axes=dim_ind), axis=dim_ind), axes=dim_ind)

                if dim_str == 'COL':
                    connection_buffer.is_ro_ft = True
                if dim_str == 'PE1':
                    connection_buffer.is_pe_ft = True
                if dim_str == 'PE2':
                    connection_buffer.is_par_ft = True

        return connection_buffer
