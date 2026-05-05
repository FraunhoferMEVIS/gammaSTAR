"""!
@brief Channel combination module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import mrpy_ismrmrd_tools as ismrmrd_tools


class ChannelCombinationModule:
    """!
    @brief This module applies coil reduction to the first channel.
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
        return ChannelCombinationModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies coil reduction to the first channel.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if 'NP_IS_IMAGING' not in con_buff.meas_data.data:
            logging.warning("gs-recon: No imaging data available for channel combination, skipping")
            return con_buff, book_keeper

        if con_buff.meas_data('NP_IS_IMAGING', 'CHA') > 1:

            # Taking just the first channel in sim only version
            con_buff.meas_data.data['NP_IS_IMAGING'] = np.expand_dims(
                np.take(con_buff.meas_data.data['NP_IS_IMAGING'], indices=0, axis=1), axis=1
            )

            book_keeper.recon_history += "_ChannelsCombined"

        return con_buff, book_keeper
