"""!
@brief k-space to image space module of gammaSTAR Reconstructions
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
    def __call__(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """
        return KspaceToImageModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies FFT along those image dimensions which were not transformed ot image space by
               another algorithm before.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if not con_buff.meas_data.is_ro_ft:
            logging.info("gs-recon: RO iFFT")

            con_buff.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(
                np.fft.ifft(np.fft.fftshift(con_buff.meas_data.data['NP_IS_IMAGING'], axes=0), axis=0), axes=0)
            con_buff.meas_data.is_ro_ft = True
            book_keeper.recon_history += "_FFTRO"

        if not con_buff.meas_data.is_pe_ft:
            logging.info("gs-recon: PE iFFT")

            con_buff.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(
                np.fft.ifft(np.fft.fftshift(con_buff.meas_data.data['NP_IS_IMAGING'], axes=2), axis=2), axes=2)
            con_buff.meas_data.is_pe_ft = True
            book_keeper.recon_history += "_FFTPE"

        return con_buff, book_keeper
