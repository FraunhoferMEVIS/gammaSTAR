"""!
@brief PROPELLER module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging
import subprocess
import math
import mrpy_noncart_tools as noncart_tools

class PropellerModule:
    """!
    @brief This module applies PROPELLER reconstruction based on received data. If PROPELLER reconstruction needs to be
           applied is decided based on a previous trajectory analysis.
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
        return PropellerModule.apply(connection_buffer)

    @staticmethod
    def apply(connection_buffer):
        """!
        @brief Applies partial Fourier functionality to 2D/3D Cartesian data using the POCS algorithm
               (or simple zerofilling) based on the config string in the ConnectionBuffer object.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """

        if connection_buffer.meas_data.is_propeller:
            logging.info("GSTAR Recon: Extracting PROPELLER trajectory from received acquisitions")

            attr_map = {
                "REP": "rep",
                "SET": "set",
                "PHS": "phase"
            }

            traj_blade_1 = np.zeros((1, 1))
            traj_blade_2 = np.zeros((1, 1))
            for acq_key in connection_buffer.meas_data.data:
                if 'ACQ_IS_IMAGING' in acq_key:
                    for acq in connection_buffer.meas_data.data[acq_key]:

                        if (traj_blade_1 == 0).all() and getattr(acq.idx, attr_map[connection_buffer.meas_data.blade_dim]) == 0:
                            traj_blade_1 = acq.traj[:, 0:-1]
                        elif (traj_blade_2 == 0).all() and getattr(acq.idx, attr_map[connection_buffer.meas_data.blade_dim]) == 1:
                            traj_blade_2 = acq.traj[:, 0:-1]
                            break

            blade_angle_incr = noncart_tools.calc_propeller_blade_increment_from_trajs(traj_blade_1, traj_blade_2)
            logging.info("GSTAR Recon:  PROPELLER angular spacing is " + str(blade_angle_incr) + "°")

            is_gpu_available = False
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                if result.returncode == 0:
                    is_gpu_available = True
                else:
                    is_gpu_available = False
            except FileNotFoundError:
                is_gpu_available = False

            logging.info("GSTAR Recon: PROPELLER Gridding")
            ro_samples = connection_buffer.meas_data('NP_IS_IMAGING', 'COL')
            pe_samples = connection_buffer.meas_data('NP_IS_IMAGING', 'PE1')
            radial_data_dims = np.zeros(2, dtype=int)
            radial_data_dims[0] = int(ro_samples)
            radial_data_dims[1] = int(pe_samples)

            oversampling_factor = 2.0
            grid_window_width = 4.0
            kaiser_bessel_kernel = noncart_tools.prep_kaiser_bessel_kernel(10000, 18.557)
            decon_mat = noncart_tools.get_deconvolution_matrix_2D(ro_samples, 2.0, 4.0, kaiser_bessel_kernel)

            orig_data_shape = connection_buffer.meas_data.data['NP_IS_IMAGING'].shape
            grid_data_shape = orig_data_shape[:2] + (orig_data_shape[0],) + orig_data_shape[3:] # After gridding, we have an isotropic matrix size
            im_grid_data = np.zeros(grid_data_shape, dtype=complex)

            for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
                for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                    for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                        for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                            for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2')):
                                for i_cha in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CHA')):

                                    traj = np.zeros((ro_samples,
                                                     pe_samples,
                                                     2,
                                                     connection_buffer.meas_data('NP_IS_IMAGING',
                                                                                 connection_buffer.meas_data.blade_dim)))

                                    for i_blade in range(connection_buffer.meas_data('NP_IS_IMAGING', connection_buffer.meas_data.blade_dim)):
                                        traj[:, :, :, i_blade] = noncart_tools.calc_equidistant_radial_trajectory_2D(radial_data_dims, (blade_angle_incr * i_blade) / 180.0 * math.pi)

                                    grid_data = connection_buffer.meas_data.data['NP_IS_IMAGING'][:, i_cha, :,
                                                                                                  i_par, i_slc,
                                                                                                  :, i_phase,
                                                                                                  i_con, i_rep,
                                                                                                  0, 0]

                                    blades_gridded, _ = noncart_tools.grid_data_to_matrix_2D(grid_data,
                                                                                             traj, oversampling_factor,
                                                                                             grid_window_width,
                                                                                             kaiser_bessel_kernel)

                                    blades_gridded_expand = np.reshape(blades_gridded, [blades_gridded.shape[0], blades_gridded.shape[1], 1])
                                    noncart_tools.prop_cut_kspace_edges_2D(blades_gridded_expand)
                                    blades_gridded = np.squeeze(blades_gridded_expand)

                                    im = noncart_tools.apply_deconvolution_2D(blades_gridded, decon_mat, oversampling_factor)
                                    im_grid_data[:, i_cha, :, i_par, i_slc, 0, i_phase, i_con, i_rep, 0, 0] = im

            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.expand_dims(im_grid_data[:, :, :, :, :, 0, :, :, :, :, :], 5)
            connection_buffer.meas_data.is_ro_ft = True
            connection_buffer.meas_data.is_pe_ft = True

        return connection_buffer
