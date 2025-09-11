"""!
@brief Partial Fourier module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging
import mrpy_parallel_tools as parallel_tools


class PartialFourierModule:
    """!
    @brief This module applies partial Fourier functionality to 2D/3D Cartesian data using the POCS algorithm
           (or simple zerofilling) based on the config string in the ConnectionBuffer object.
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
        return PartialFourierModule.apply(connection_buffer)

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

        if connection_buffer.meas_data.pf_factor_pe1 != 1.0 or connection_buffer.meas_data.pf_factor_pe2 != 1.0:

            logging.info("GSTAR Recon: Running POCS partial Fourier routine")

            temp_encoded = connection_buffer.meas_data.data['NP_IS_IMAGING']
            recon_dims = connection_buffer.meas_data.data['NP_IS_IMAGING'].shape
            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.zeros((connection_buffer.meas_data.recon_space.matrixSize.x,
                                                                  recon_dims[1],
                                                                  connection_buffer.meas_data.recon_space.matrixSize.y,
                                                                  connection_buffer.meas_data.recon_space.matrixSize.z,
                                                                  recon_dims[4],
                                                                  recon_dims[5],
                                                                  recon_dims[6],
                                                                  recon_dims[7],
                                                                  recon_dims[8],
                                                                  recon_dims[9],
                                                                  recon_dims[10]), dtype=complex)

            # We check whether we need to flip the data first, to make sure that missing data samples are at the end and
            # not at the beginning of the array.
            test_ksp = connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, :, 0, 0, 0, 0, 0, 0, 0]
            _, center_pe1, center_pe2 = np.unravel_index(np.abs(test_ksp).argmax(), test_ksp.shape)
            b_is_flip_par = False
            b_is_flip_pe = False
            if center_pe2 <= connection_buffer.meas_data('NP_IS_IMAGING', 'PE2') / 2:
                b_is_flip_par = True
            if center_pe1 <= connection_buffer.meas_data('NP_IS_IMAGING', 'PE1') / 2:
                b_is_flip_pe = True

            if connection_buffer.meas_data('NP_IS_IMAGING', 'PE2') > 1:

                # Estimate the size of the transition band
                q_pe1 = max(3, int((connection_buffer.meas_data('NP_IS_IMAGING', 'PE1') - connection_buffer.meas_data('NP_IS_IMAGING', 'PE1') / 2) / 6))
                if q_pe1 % 2 == 0:
                    q_pe1 = q_pe1 + 1
                q_pe2 = max(3, int((connection_buffer.meas_data('NP_IS_IMAGING', 'PE2') - connection_buffer.meas_data('NP_IS_IMAGING', 'PE2') / 2) / 6))
                if q_pe2 % 2 == 0:
                    q_pe2 = q_pe2 + 1

                for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
                    for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                        for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                            for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                                for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                                    for i_cha in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CHA')):
                                        cha_pf_image = temp_encoded[:, i_cha, :, :, i_slc, i_set, i_phase, i_con, i_rep, 0, 0]

                                        if connection_buffer.meas_data.pf_factor_pe1 != 1.0 and b_is_flip_pe:
                                            cha_pf_image = np.flip(cha_pf_image, 1)
                                        if connection_buffer.meas_data.pf_factor_pe2 != 1.0 and b_is_flip_par:
                                            cha_pf_image = np.flip(cha_pf_image, 2)

                                        cha_pf_image_zero = np.zeros((connection_buffer.meas_data.recon_space.matrixSize.x,
                                                                      connection_buffer.meas_data.recon_space.matrixSize.y,
                                                                      connection_buffer.meas_data.recon_space.matrixSize.z), dtype=complex)
                                        cha_pf_image_zero[:,
                                                          0:connection_buffer.meas_data.encoded_space.matrixSize.y,
                                                          0:connection_buffer.meas_data.encoded_space.matrixSize.z] = cha_pf_image
                                        cha_pf_image_filled = parallel_tools.fill_partial_fourier_pocs_3D(cha_pf_image_zero, 5, q_pe1, q_pe2)

                                        if connection_buffer.meas_data.pf_factor_pe1 != 1.0 and b_is_flip_pe:
                                            cha_pf_image_filled = np.flip(cha_pf_image_filled, 1)
                                        if connection_buffer.meas_data.pf_factor_pe2 != 1.0 and b_is_flip_par:
                                            cha_pf_image_filled = np.flip(cha_pf_image_filled, 2)
                                        connection_buffer.meas_data.data['NP_IS_IMAGING'][:, i_cha, :, :, i_slc, i_set, i_phase, i_con, i_rep, 0, 0] = cha_pf_image_filled

            else:

                for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
                    for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                        for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                            for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                                for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                                    for i_cha in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CHA')):

                                        cha_pf = temp_encoded[:, i_cha, :, 0, i_slc, i_set, i_phase, i_con, i_rep, 0, 0]

                                        if b_is_flip_pe:
                                            cha_pf = np.fliplr(cha_pf)

                                        cha_pf_zero = np.zeros((connection_buffer.meas_data.recon_space.matrixSize.x, connection_buffer.meas_data.recon_space.matrixSize.y), dtype=complex)
                                        cha_pf_zero[:, 0:connection_buffer.meas_data.encoded_space.matrixSize.y] = cha_pf
                                        cha_pf_zero = parallel_tools.fill_partial_fourier_pocs_2D(cha_pf_zero,5)

                                        if b_is_flip_pe:
                                            cha_pf_zero = np.fliplr(cha_pf_zero)

                                        connection_buffer.meas_data.data['NP_IS_IMAGING'][:, i_cha, :, 0,
                                        i_slc, i_con, i_set, i_phase, i_rep, 0, 0] = cha_pf_zero

        return connection_buffer
