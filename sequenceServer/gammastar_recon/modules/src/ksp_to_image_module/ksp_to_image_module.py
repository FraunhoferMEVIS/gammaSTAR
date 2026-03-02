"""!
@brief k-space to image space module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import mrpy_ismrmrd_tools as ismrmrd_tools

class KspaceToImageModule:
    """!
    @brief This module applies FFT along those image dimensions which were not transformed ot image space by
           another algorithm before.
    """

    @staticmethod
    def __call__(connection_buffer: ismrmrd_tools.ConnectionBuffer,
                 book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @return
            - (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            - (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @author Jörn Huber
        """
        return KspaceToImageModule.apply(connection_buffer, book_keeper)

    @staticmethod
    def apply(connection_buffer: ismrmrd_tools.ConnectionBuffer,
              book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies FFT along those image dimensions which were not transformed ot image space by
               another algorithm before.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @return
            - (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            - (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @author Jörn Huber
        """

        if not connection_buffer.meas_data.is_ro_ft:
            logging.info("gs-recon: RO iFFT")

            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(
                np.fft.ifft(np.fft.fftshift(connection_buffer.meas_data.data['NP_IS_IMAGING'], axes=0), axis=0), axes=0)
            connection_buffer.meas_data.is_ro_ft = True
            book_keeper.recon_history += "_FFTRO"

        if not connection_buffer.meas_data.is_pe_ft:
            logging.info("gs-recon: PE iFFT")

            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(
                np.fft.ifft(np.fft.fftshift(connection_buffer.meas_data.data['NP_IS_IMAGING'], axes=2), axis=2), axes=2)
            connection_buffer.meas_data.is_pe_ft = True
            book_keeper.recon_history += "_FFTPE"

        for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
            for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                    for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                        for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                            for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2')):

                                im = connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, i_par, i_slc, i_set,
                                                                                       i_phase, i_con, i_rep, 0, 0]

                                connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, i_par, i_slc, i_set, i_phase,
                                                                                  i_con, i_rep, 0, 0] = im

        return connection_buffer, book_keeper
