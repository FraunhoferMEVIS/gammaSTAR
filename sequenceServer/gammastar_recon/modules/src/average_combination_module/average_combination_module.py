"""!
@brief Average combination module
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
        return AverageCombinationModule.apply(connection_buffer, book_keeper)

    @staticmethod
    def apply(connection_buffer: ismrmrd_tools.ConnectionBuffer,
              book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies the modules functionality by averaging data along the average dimension. 

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @return
            - (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            - (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @author Jörn Huber
        """

        is_applied = False
        for data_key in connection_buffer.meas_data.data:
            if "NP" in data_key and "PHASECORR" not in data_key and connection_buffer.meas_data(data_key, 'AVE') > 1:
                logging.info("gs-recon: Averaging " + data_key)
                ave_sig = np.mean(connection_buffer.meas_data.data[data_key], 9)
                connection_buffer.meas_data.data[data_key] = np.expand_dims(ave_sig, 9)

                if not is_applied:
                    is_applied = True

        if is_applied:
            book_keeper.recon_history += "_SummedAverages"

        return connection_buffer, book_keeper
