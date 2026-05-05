"""!
@brief Image finalization module of gammaSTAR Reconstructions
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
        return FinalizeOutImageModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Finalizes (convert to short + scaling) the reconstructed series of images and appends it to the
               output buffer.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        recon_images = np.around(
            np.abs(con_buff.meas_data.data['NP_IS_IMAGING']) * book_keeper.last_scaling_factor
        ).astype(np.uint16)
        for i_rep in range(0, con_buff.meas_data('NP_IS_IMAGING', 'REP')):
            for i_phase in range(0, con_buff.meas_data('NP_IS_IMAGING', 'PHS')):
                for i_set in range(0, con_buff.meas_data('NP_IS_IMAGING', 'SET')):
                    for i_con in range(0, con_buff.meas_data('NP_IS_IMAGING', 'CON')):
                        for i_slc in range(0, con_buff.meas_data('NP_IS_IMAGING', 'SLC')):

                            idx_comment = '_'
                            if con_buff.meas_data('NP_IS_IMAGING', 'REP') > 1:
                                idx_comment += 'REP_' + str(i_rep)
                            if con_buff.meas_data('NP_IS_IMAGING', 'CON') > 1:
                                idx_comment += 'ECO_' + str(i_con)
                            if con_buff.meas_data('NP_IS_IMAGING', 'PHS') > 1:
                                idx_comment += 'PHS_' + str(i_phase)
                            if con_buff.meas_data('NP_IS_IMAGING', 'SET') > 1:
                                idx_comment += 'SET_' + str(i_set)
                            if con_buff.meas_data('NP_IS_IMAGING', 'SLC') > 1:
                                idx_comment += 'SLC_' + str(i_slc)
                            series_comment = idx_comment

                            meas_idx = ismrmrd_tools.MeasIDX(i_rep, i_con, i_phase, i_set, i_slc)

                            par_image = recon_images[:, 0, :, :,
                                                     i_slc,
                                                     i_set,
                                                     i_phase,
                                                     i_con,
                                                     i_rep, 0, 0]

                            ismrmrd_image = ismrmrd_tools.numpy_array_to_ismrmrd_image(
                                par_image,
                                con_buff.meas_data.data['ACQ_IS_IMAGING'],
                                con_buff.headers[0],
                                book_keeper.image_series_index,
                                meas_idx,
                                series_comment,
                                book_keeper.recon_history,
                                book_keeper.last_scaling_factor
                            )
                            book_keeper.outgoing_image_buffer.append(ismrmrd_image)
                            book_keeper.image_series_index = book_keeper.image_series_index + 1

        return book_keeper
