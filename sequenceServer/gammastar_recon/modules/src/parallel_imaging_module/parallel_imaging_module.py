"""!
@brief Parallel imaging module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import mrpy_ismrmrd_tools as ismrmrd_tools


class ParallelImagingModule:
    """!
    @brief This module applies parallel imaging solutions. This public version just prints a hint to the user.
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
        return ParallelImagingModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies parallel imaging solutions. This public version just prints a hint to the user.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if con_buff.meas_data.accel_pe1 > 1 or con_buff.meas_data.accel_pe2 > 1:

            if con_buff.meas_data('NP_IS_IMAGING', 'CHA') == 1:
                logging.warning("gs-recon: Cannot perform parallel imaging tasks in single channel experiments!")
            else:
                logging.warning("gs-recon: Parallel imaging functionality not available in sim only version, "
                                "contact the developers for personalized solutions!")

        return con_buff, book_keeper
