"""!
@brief Channel combination module.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import mrpy_coil_tools as coil_tools

class ChannelCombinationModule:
    """!
    @brief This module applies coil combination along the channel dimension of data with the "NP_IS_IMAGING" key
           by using sum-of-squares or adaptive combination.
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
        return ChannelCombinationModule.apply(connection_buffer)

    @staticmethod
    def apply(connection_buffer):
        """!
        @brief Applies either sum-of-squares or adaptive coil combination to the data with "NP_IS_IMAGING" key,
               based on the config string as stored in the connection buffer object. Note that the default is
               sum-of-squares if no valid string is received.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber
        """

        if connection_buffer.meas_data('NP_IS_IMAGING', 'CHA') > 1:

            if connection_buffer.meas_data.is_propeller:
                logging.info("GSTAR Recon: Applying adaptive coil combination")
            else:
                logging.info("GSTAR Recon: Applying sum-of-squares coil combination")

            def process_combination(args):

                (i_rep, i_phase, i_set, i_con, i_slc, i_par) = args
                par_image = np.transpose(
                    np.squeeze(
                        connection_buffer.meas_data.data['NP_IS_IMAGING'][:, :, :, i_par, i_slc, i_set, i_phase, i_con,
                        i_rep, 0, 0]
                    ),
                    [0, 2, 1]
                )

                if connection_buffer.meas_data.is_propeller:

                    sens = coil_tools.estimate_sens(par_image, 1)
                    cha_combined, _ = coil_tools.combine_channels(par_image, sens)

                else:
                    par_image = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(par_image, axes=0), axis=0), axes=0)
                    par_image = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(par_image, axes=1), axis=1), axes=1)
                    cha_combined = coil_tools.sum_of_squares_combination(par_image)

                # Write result back to the shared array
                connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, i_par, i_slc, i_set, i_phase, i_con, i_rep,
                0, 0] = cha_combined

            indices = [
                (i_rep, i_phase, i_set, i_con, i_slc, i_par)
                for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP'))
                for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS'))
                for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET'))
                for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON'))
                for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC'))
                for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2'))
            ]

            with ThreadPoolExecutor() as executor:
                executor.map(process_combination, indices)

            if not connection_buffer.meas_data.is_propeller:
                connection_buffer.meas_data.is_ro_ft = True
                connection_buffer.meas_data.is_pe_ft = True

            connection_buffer.meas_data.data['NP_IS_IMAGING'] = connection_buffer.meas_data.data['NP_IS_IMAGING'][:, 0, :, :, :, :, :, :, :, :, :]
            connection_buffer.meas_data.data['NP_IS_IMAGING'] = np.expand_dims(connection_buffer.meas_data.data['NP_IS_IMAGING'], 1)

        return connection_buffer
