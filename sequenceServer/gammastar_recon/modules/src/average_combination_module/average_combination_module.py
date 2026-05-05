"""!
@brief Average combination module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import mrpy_ismrmrd_tools as ismrmrd_tools


class AverageCombinationModule:
    """!
    @brief This module sums invdividual averages of the measurement
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
        return AverageCombinationModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies the modules functionality by averaging data along the average dimension. 

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        is_applied = False
        for data_key in con_buff.meas_data.data:
            if "NP" in data_key and "PHASECORR" not in data_key and con_buff.meas_data(data_key, 'AVE') > 1:
                logging.info("gs-recon: Averaging " + data_key)
                ave_sig = np.mean(con_buff.meas_data.data[data_key], 9)
                con_buff.meas_data.data[data_key] = np.expand_dims(ave_sig, 9)

                if not is_applied:
                    is_applied = True

        if is_applied:
            book_keeper.recon_history += "_SummedAverages"

        return con_buff, book_keeper
