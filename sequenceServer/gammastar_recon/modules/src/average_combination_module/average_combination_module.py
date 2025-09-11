"""!
@brief Average combination module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging


class AverageCombinationModule:
    """!
    @brief This module sums invdividual averages of the measurement in all data sets except those which contain the
           "PHASECORR" key as this stores the third measurement line in the average dimension.
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
        return AverageCombinationModule.apply(connection_buffer)

    @staticmethod
    def apply(connection_buffer):
        """!
        @brief Applies the modules functionality by averaging data along the average dimension. 

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """

        for data_key in connection_buffer.meas_data.data:
            if "NP" in data_key and "PHASECORR" not in data_key and connection_buffer.meas_data(data_key, 'AVE') > 1:
                logging.info("GSTAR Recon: Averaging " + data_key)
                connection_buffer.meas_data.data[data_key] = np.expand_dims(np.mean(connection_buffer.meas_data.data[data_key], 9), 9)

        return connection_buffer


