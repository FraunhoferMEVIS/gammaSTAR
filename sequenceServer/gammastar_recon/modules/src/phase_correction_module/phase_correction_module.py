"""!
@brief PROPELLER phase correction module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
import mrpy_noncart_tools as noncart_tools


class PhaseCorModule:
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
        return PhaseCorModule.apply(connection_buffer)

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

        if connection_buffer.meas_data.is_propeller:

            logging.info("GSTAR Recon: Applying PROPELLER phase correction")

            def process_phase_correction(args):
                (i_rep, i_phase, i_set, i_con, i_slc, i_par, i_cha, connection_buffer) = args
                data = connection_buffer.meas_data.data['NP_IS_IMAGING']
                data[:, i_cha, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, 0] = (
                    noncart_tools.prop_phase_correction_2D(
                        data[:, i_cha, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, 0]))
                return None

            indices = [
                (i_rep, i_phase, i_set, i_con, i_slc, i_par, i_cha, connection_buffer)
                for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP'))
                for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS'))
                for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET'))
                for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON'))
                for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC'))
                for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2'))
                for i_cha in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CHA'))
            ]

            with ThreadPoolExecutor() as executor:
                list(executor.map(process_phase_correction, indices))

        return connection_buffer
