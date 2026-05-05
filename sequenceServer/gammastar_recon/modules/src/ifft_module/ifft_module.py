"""!
@brief FFT Module of gammaSTAR Reconstructions
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
    def __call__(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper",
            dim_str: str
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc
        @param dim_str: String containing desired fft dimension

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """
        return IFFTModule.apply(con_buff, book_keeper, dim_str)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper",
            dim_str: str
    ):
        """!
        @brief Applies FFT along a specified image dimension

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if not con_buff.meas_data.is_ro_ft:

            if con_buff.meas_data('NP_IS_IMAGING', dim_str) > 1:

                logging.info("gs-recon: " + dim_str + " IFFT")
                dim_ind = ismrmrd_tools.IsmrmrdConstants.IDX_MAP[dim_str]

                con_buff.meas_data.data['NP_IS_IMAGING'] = np.fft.fftshift(
                    np.fft.ifft(np.fft.fftshift(con_buff.meas_data.data['NP_IS_IMAGING'], axes=dim_ind),
                                axis=dim_ind), axes=dim_ind)

                if dim_str == 'COL':
                    con_buff.is_ro_ft = True
                if dim_str == 'PE1':
                    con_buff.is_pe_ft = True
                if dim_str == 'PE2':
                    con_buff.is_par_ft = True
                    book_keeper.recon_history += "_FFTPAR"

        return con_buff, book_keeper
