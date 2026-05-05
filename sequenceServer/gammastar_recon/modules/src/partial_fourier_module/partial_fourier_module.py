"""!
@brief Partial Fourier module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging
import mrpy_ismrmrd_tools as ismrmrd_tools


class PartialFourierModule:
    """!
    @brief This module applies partial Fourier functionality to 2D/3D Cartesian data using zerofilling.
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
        return PartialFourierModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies partial Fourier functionality to 2D/3D Cartesian data using the POCS algorithm
               (or simple zerofilling) based on the config string in the ConnectionBuffer object.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if con_buff.meas_data.pf_factor_pe1 < 1.0 or con_buff.meas_data.pf_factor_pe2 < 1.0:

            logging.info("gs-recon: Zerofilling partial Fourier data")

            recon_mat_size_y = con_buff.meas_data.recon_space.matrixSize.y
            recon_mat_size_z = con_buff.meas_data.recon_space.matrixSize.z
            encode_mat_size_y = con_buff.meas_data.encoded_space.matrixSize.y
            encode_mat_size_z = con_buff.meas_data.encoded_space.matrixSize.z

            pad_list = [(0, 0)]*con_buff.meas_data.data['NP_IS_IMAGING'].ndim

            if con_buff.meas_data.pf_factor_pe1 != 1.0:
                pad_list[2] = (0, recon_mat_size_y - encode_mat_size_y)
                book_keeper.recon_history += "_ZeroFilledPartialFourierPE1"

            if con_buff.meas_data.pf_factor_pe2 != 1.0:
                pad_list[3] = (0, recon_mat_size_z - encode_mat_size_z)
                book_keeper.recon_history += "_ZeroFilledPartialFourierPE2"

            con_buff.meas_data.data['NP_IS_IMAGING'] = np.pad(
                con_buff.meas_data.data['NP_IS_IMAGING'],
                pad_list,
                mode='constant'
            )

        return con_buff, book_keeper
