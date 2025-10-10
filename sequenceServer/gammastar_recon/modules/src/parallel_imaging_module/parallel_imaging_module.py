"""!
@brief Parallel imaging module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging


class ParallelImagingModule:
    """!
    @brief This module reconstructs missing data lines in the "NP_IS_IMAGING" data for 2D/3D Cartesian acquisitions
           using appropriate Grappa/Caipirinha functionality and reference lines from the "NP_IS_PARALLEL_CALIBRATION"
           data.
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
        return ParallelImagingModule.apply(connection_buffer)

    @staticmethod
    def apply(connection_buffer):
        """!
        @brief Reconstructs missing data lines in the "NP_IS_IMAGING" data for 2D/3D Cartesian acquisitions
               using appropriate Grappa/Caipirinha functionality and reference lines from the "NP_IS_PARALLEL_CALIBRATION"
               data.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """

        if connection_buffer.meas_data.accel_pe1 > 1 or connection_buffer.meas_data.accel_pe2 > 1:

            if connection_buffer.meas_data('NP_IS_IMAGING', 'CHA') == 1:
                logging.warning("GSTAR Recon: Cannot perform parallel imaging tasks in single channel experiments!")
                return connection_buffer
            else:
                logging.warning("GSTAR Recon: Parallel imaging functionality currently not implemented, contact the developers for personalized solutions!")
                return connection_buffer

        return connection_buffer
