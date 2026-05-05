"""!
@brief PROPELLER module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import mrpy_noncart_tools as noncart_tools
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_helpers as helpers


class PropellerModule:
    """!
    @brief This module applies PROPELLER reconstruction based on received data. If PROPELLER reconstruction needs to be
           applied is decided based on a previous trajectory analysis.
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
        return PropellerModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies PROPELLER reconstruction based on received data. If PROPELLER reconstruction needs to be
               applied is decided based on a previous trajectory analysis.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if 'NP_IS_IMAGING' not in con_buff.meas_data.data:
            logging.warning("gs-recon: No imaging data available for PROPELLER module, skipping")
            return con_buff, book_keeper

        if con_buff.meas_data.is_propeller:
            logging.info("gs-recon: Extracting PROPELLER trajectory from received acquisitions")

            attr_map = {
                "REP": "rep",
                "SET": "set",
                "PHS": "phase"
            }

            traj_blade_1 = np.zeros((1, 1))
            traj_blade_2 = np.zeros((1, 1))
            for acq_key in con_buff.meas_data.data:
                if 'ACQ_IS_IMAGING' in acq_key:
                    for acq in con_buff.meas_data.data[acq_key]:

                        if (traj_blade_1 == 0).all() and getattr(acq.idx, attr_map[con_buff.meas_data.blade_dim]) == 0:
                            traj_blade_1 = acq.traj[:, 0:-1]
                        elif (traj_blade_2 == 0).all() and getattr(acq.idx, attr_map[con_buff.meas_data.blade_dim]) == 1:
                            traj_blade_2 = acq.traj[:, 0:-1]
                            break

            blade_angle_incr = noncart_tools.calc_propeller_blade_increment_from_trajs(traj_blade_1, traj_blade_2)
            logging.info("gs-recon:  PROPELLER angular spacing is %s°", str(blade_angle_incr))

            logging.info("gs-recon: PROPELLER NUFFT")

            traj_data = noncart_tools.calc_equidistant_propeller_trajectory(
                ksp_data = con_buff.meas_data.data['NP_IS_IMAGING'],
                ro_axis = 0,
                pe_axis = 2,
                acq_axis = 5,
                acq_angle_incr = blade_angle_incr
            )
            traj_data = np.moveaxis(traj_data, -1, 1)
            traj_data = np.squeeze(traj_data, 2)

            nufft_data = noncart_tools.apply_nufft(
                ksp_data = con_buff.meas_data.data['NP_IS_IMAGING'],
                traj_data = traj_data,
                mat_size = int(con_buff.headers[0].encoding[0].reconSpace.matrixSize.x * 2.0),
                ro_axis = 0,
                ch_axis = 1,
                acq_axis = 5,
                pe1_axis = 2,
                density_method = 'cell_count',
            )

            # Back to k-space for further processing
            axes = (0, 2)
            nufft_data = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.fftshift(
                        nufft_data,
                        axes=axes
                    ),
                    axes=axes
                ),
                axes=axes
            )
            nufft_data = np.flip(nufft_data, axis=0)
            nufft_data = np.flip(nufft_data, axis=2)

            # Remove OS along RO and PE1
            con_buff.meas_data.data['NP_IS_IMAGING'] = helpers.remove_os(
                nufft_data,
                os_factor = 2,
                k_axes = (0, 2)
            )

            con_buff.meas_data.is_ro_ft = False
            con_buff.meas_data.is_pe_ft = False
            book_keeper.recon_history += "_PROPELLERNufft"

        return con_buff, book_keeper
