"""!
@brief Phase correction module for PROPELLER data of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_helpers as helpers


class PhaseCorModule:
    """!
    @brief This module applies coil combination along the channel dimension of data with the "NP_IS_IMAGING" key
           by using sum-of-squares or adaptive combination.
    """

    @staticmethod
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
        return PhaseCorModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies phase correction to PROPELLER data, e.g. removes slowly varying phase inconsistencies between
               blades.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if 'NP_IS_IMAGING' not in con_buff.meas_data.data:
            logging.warning("gs-recon: No imaging data available for phase correction, skipping")
            return con_buff, book_keeper

        if con_buff.meas_data.is_propeller:

            logging.info("gs-recon: Applying PROPELLER phase correction")
            con_buff.meas_data.data['NP_IS_IMAGING'], _ = helpers.filter_low_order_phase(
                ksp_data = con_buff.meas_data.data['NP_IS_IMAGING'],
                k_axes = (0, 2)
            )

            book_keeper.recon_history += "_PROPELLERPhaseCorrection"

        return con_buff, book_keeper
