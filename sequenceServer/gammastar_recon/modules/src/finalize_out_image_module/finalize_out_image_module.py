"""!
@brief Image finalization module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import mrpy_ismrmrd_tools as ismrmrd_tools


class FinalizeOutImageModule:
    """!
    @brief This module finalizes (convert to short + scaling) the reconstructed series of images and appends it to the
           output buffer.
    """

    @staticmethod
    def __call__(connection_buffer, book_keeper):
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (dict) Dictionary which is used to store image processing results.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            -  (dict) Dictionary which is used to store image processing results.

        @author Jörn Huber
        """
        return FinalizeOutImageModule.apply(connection_buffer, book_keeper)

    @staticmethod
    def apply(connection_buffer, book_keeper):
        """!
        @brief Finalizes (convert to short + scaling) the reconstructed series of images and appends it to the
               output buffer.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (dict) Dictionary which is used to store image processing results.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            -  (dict) Dictionary which is used to store image processing results.

        @author Jörn Huber
        """

        recon_images = np.around(np.abs(connection_buffer.meas_data.data['NP_IS_IMAGING']) * book_keeper.last_scaling_factor).astype(np.uint16)
        for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
            for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                    for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                        for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):

                            idx_comment = '_'
                            if connection_buffer.meas_data('NP_IS_IMAGING', 'REP') > 1:
                                idx_comment += 'REP_' + str(i_rep)
                            if connection_buffer.meas_data('NP_IS_IMAGING', 'CON') > 1:
                                idx_comment += 'ECO_' + str(i_con)
                            if connection_buffer.meas_data('NP_IS_IMAGING', 'PHS') > 1:
                                idx_comment += 'PHS_' + str(i_phase)
                            if connection_buffer.meas_data('NP_IS_IMAGING', 'SET') > 1:
                                idx_comment += 'SET_' + str(i_set)
                            if connection_buffer.meas_data('NP_IS_IMAGING', 'SLC') > 1:
                                idx_comment += 'SLC_' + str(i_slc)
                            series_comment = idx_comment

                            meas_idx = ismrmrd_tools.MeasIDX(i_rep, i_con, i_phase, i_set, i_slc)

                            par_image = np.transpose(recon_images[:, 0, :, :,
                                                                  i_slc,
                                                                  i_set,
                                                                  i_phase,
                                                                  i_con,
                                                                  i_rep, 0, 0], [1, 0, 2])

                            ismrmrd_image = ismrmrd_tools.numpy_array_to_ismrmrd_image(par_image,
                                                                                       connection_buffer.meas_data.data['ACQ_IS_IMAGING'],
                                                                                       connection_buffer.headers[0],
                                                                                       book_keeper.image_series_index,
                                                                                       meas_idx,
                                                                                       series_comment,
                                                                                       'gammaSTAR Recon v1.0.1',
                                                                                       book_keeper.last_scaling_factor)
                            book_keeper.outgoing_image_buffer.append(ismrmrd_image)
                            book_keeper.image_series_index = book_keeper.image_series_index + 1

        return book_keeper
