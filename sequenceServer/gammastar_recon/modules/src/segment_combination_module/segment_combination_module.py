"""!
@brief Segment combination module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import mrpy_epi_tools as epi_tools
import mrpy_ismrmrd_tools as ismrmrd_tools


class SegmentCombinationModule:
    """!
    @brief This module applies segment combination to individual k-space segments in the "NP_IS_IMAGING" data to form
           the final combined dataset. If the "meas_data" dictionary of the ConnectionBuffer object contains
           any "ACQ_IS_REVERSE" flags, the module applies the corresponding EPI-based phase correction technique using
           a three-lines reference scan if possible. If such data is not available, the module just sums individual
           segments.
    """

    @staticmethod
    def __call__(connection_buffer: ismrmrd_tools.ConnectionBuffer,
                 book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @return
            - (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            - (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @author Jörn Huber
        """
        return SegmentCombinationModule.apply(connection_buffer, book_keeper)

    @staticmethod
    def apply(connection_buffer: ismrmrd_tools.ConnectionBuffer,
              book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies segment combination to individual k-space segments in the "NP_IS_IMAGING" data to form
           the final combined dataset. If the "meas_data" dictionary of the ConnectionBuffer bject contains
           any "ACQ_IS_REVERSE" flags, the module applies the corresponding EPI-based phase correction technique using
           a three-lines reference scan if possible. If such data is not available, the module just sums individual
           segments.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @return
            - (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            - (BookKeeper) BookKeeper object, holding patient information and reconstruction history.

        @author Jörn Huber, Daniel Hoinkiss
        """

        b_is_reverse = any('ACQ_IS_REVERSE' in data_key for data_key in connection_buffer.meas_data.data)

        if b_is_reverse:  # We received reversely sampled data -> correct shifts

            logging.info("gs-recon: Applying reverse segment correction for EPI-like data")

            # We identify the correct keys for reverse and non-reverse imaging and phasecor data
            ph_cor_rev_key = ''
            im_rev_key = ''
            ph_cor_key = ''
            im_key = ''
            for data_key in connection_buffer.meas_data.data:
                if 'NP_IS_REVERSE' in data_key and 'NP_IS_PHASECORR_DATA' in data_key:
                    ph_cor_rev_key = data_key
                if 'NP_IS_REVERSE' not in data_key and 'NP_IS_PHASECORR_DATA' in data_key:
                    ph_cor_key = data_key
                if 'NP_IS_REVERSE' in data_key and 'NP_IS_IMAGING' in data_key:
                    im_rev_key = data_key
                if 'NP_IS_REVERSE' not in data_key and 'NP_IS_IMAGING' in data_key:
                    im_key = data_key

            if ph_cor_key == '' or ph_cor_rev_key == '':

                logging.warning("gs-recon: No full set of phase correction data available for reverse segment "
                                "correction")

            else:

                phase_cor_acq_key = ph_cor_key.replace('NP', 'ACQ')
                corr_line_ind_ksp1 = connection_buffer.meas_data.data[phase_cor_acq_key][0].idx.kspace_encode_step_1
                corr_line_ind_ksp2 = connection_buffer.meas_data.data[phase_cor_acq_key][0].idx.kspace_encode_step_2

                ph_corr_lines = np.zeros((connection_buffer.meas_data(ph_cor_key, 'COL'), 3), dtype=complex)

                for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
                    for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                        for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                            for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                                for i_cha in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CHA')):
                                    for i_seg in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SEG')):

                                        if connection_buffer.meas_data.data[ph_cor_rev_key].shape[9] == 2:

                                            ph_corr_lines[:, 0] = \
                                            connection_buffer.meas_data.data[ph_cor_rev_key][
                                                :, i_cha, corr_line_ind_ksp1, corr_line_ind_ksp2, i_slc, 0, i_phase, 0,
                                                i_rep, 0, i_seg]

                                            ph_corr_lines[:, 1] = connection_buffer.meas_data.data[ph_cor_key][
                                                :, i_cha, corr_line_ind_ksp1, corr_line_ind_ksp2, i_slc, 0, i_phase, 0,
                                                i_rep, 0, i_seg]

                                            ph_corr_lines[:, 2] = \
                                            connection_buffer.meas_data.data[ph_cor_rev_key][
                                                :, i_cha, corr_line_ind_ksp1, corr_line_ind_ksp2, i_slc, 0, i_phase, 0,
                                                i_rep, 1, i_seg]

                                        elif connection_buffer.meas_data.data[ph_cor_key].shape[9] == 2:

                                            ph_corr_lines[:, 0] = connection_buffer.meas_data.data[ph_cor_key][
                                                :, i_cha, corr_line_ind_ksp1, corr_line_ind_ksp2, i_slc, 0, i_phase, 0,
                                                i_rep, 0, i_seg]

                                            ph_corr_lines[:, 1] = \
                                            connection_buffer.meas_data.data[ph_cor_rev_key][
                                                :, i_cha, corr_line_ind_ksp1, corr_line_ind_ksp2, i_slc, 0, i_phase, 0,
                                                i_rep, 0, i_seg]

                                            ph_corr_lines[:, 2] = connection_buffer.meas_data.data[ph_cor_key][
                                                :, i_cha, corr_line_ind_ksp1, corr_line_ind_ksp2, i_slc, 0, i_phase, 0,
                                                i_rep, 1, i_seg]

                                        if np.min(np.abs(ph_corr_lines)) == 0 and np.max(np.abs(ph_corr_lines)) == 0:
                                            logging.warning(
                                                f"gs-recon: Phase correction data contains only zeros. "
                                                f"Skipping epi phase correction for current segment. "
                                                f"Indices: REP={i_rep}, PHS={i_phase}, SET={i_set}, "
                                                f"SLC={i_slc}, CHA={i_cha}, SEG={i_seg}")
                                            continue

                                        phase_corr_drift = epi_tools.calc_linear_phase_correction(ph_corr_lines)

                                        for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                                            for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2')):

                                                if connection_buffer.meas_data.data[ph_cor_rev_key].shape[9] == 2:

                                                    par_ksp_data = connection_buffer.meas_data.data[im_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con,
                                                        i_rep, 0, i_seg]

                                                    par_ksp_data_corr = epi_tools.correct_linear_phase_drift(
                                                        par_ksp_data, phase_corr_drift)

                                                    connection_buffer.meas_data.data[im_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con,
                                                        i_rep, 0, i_seg] = par_ksp_data_corr

                                                elif connection_buffer.meas_data.data[ph_cor_key].shape[9] == 2:

                                                    par_ksp_data = \
                                                    connection_buffer.meas_data.data[im_rev_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con,
                                                        i_rep, 0, i_seg]

                                                    par_ksp_data_corr = epi_tools.correct_linear_phase_drift(
                                                        par_ksp_data, phase_corr_drift)

                                                    connection_buffer.meas_data.data[im_rev_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con,
                                                        i_rep, 0, i_seg] = par_ksp_data_corr

                connection_buffer.meas_data.data[im_key] = connection_buffer.meas_data.data[im_key] + \
                                                                connection_buffer.meas_data.data[im_rev_key]
                del connection_buffer.meas_data.data[ph_cor_rev_key]
                del connection_buffer.meas_data.data[im_rev_key]

                book_keeper.recon_history += "_EPIReverseSegmentCorrection"

        connection_buffer.meas_data.data["NP_IS_IMAGING"] = np.expand_dims(
            np.sum(connection_buffer.meas_data.data["NP_IS_IMAGING"], 10), 10)

        return connection_buffer, book_keeper
