"""!
@brief Segment combination module of gammastar reconstruction
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         **InsertLicense** code

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging
import mrpy_epi_tools as epi_tools


class SegmentCombinationModule:
    """!
    @brief This module applies segment combination to individual k-space segments in the "NP_IS_IMAGING" data to form
           the final combined dataset. If the "meas_data" dictionary of the ConnectionBuffer object contains
           any "ACQ_IS_REVERSE" flags, the module applies the corresponding EPI-based phase correction technique using
           a three-lines reference scan if possible. If such data is not available, the module just sums individual
           segments.
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
        return SegmentCombinationModule.apply(connection_buffer)

    @staticmethod
    def apply(connection_buffer):
        """!
        @brief Applies segment combination to individual k-space segments in the "NP_IS_IMAGING" data to form
           the final combined dataset. If the "meas_data" dictionary of the ConnectionBuffer bject contains
           any "ACQ_IS_REVERSE" flags, the module applies the corresponding EPI-based phase correction technique using
           a three-lines reference scan if possible. If such data is not available, the module just sums individual
           segments.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.

        @author Jörn Huber, Daniel Hoinkiss
        """

        b_is_reverse = any('ACQ_IS_REVERSE' in data_key for data_key in connection_buffer.meas_data.data)

        if b_is_reverse:  # We received reversely sampled data -> correct shifts

            logging.info("GSTAR Recon: Applying reverse segment correction for EPI-like data")

            # We identify the correct keys for reverse and non-reverse imaging and phasecor data
            phase_cor_reverse_key = ''
            imaging_reverse_key = ''
            phase_cor_key = ''
            imaging_key = ''
            for data_key in connection_buffer.meas_data.data:
                if 'NP_IS_REVERSE' in data_key and 'NP_IS_PHASECORR_DATA' in data_key:
                    phase_cor_reverse_key = data_key
                if 'NP_IS_REVERSE' not in data_key and 'NP_IS_PHASECORR_DATA' in data_key:
                    phase_cor_key = data_key
                if 'NP_IS_REVERSE' in data_key and 'NP_IS_IMAGING' in data_key:
                    imaging_reverse_key = data_key
                if 'NP_IS_REVERSE' not in data_key and 'NP_IS_IMAGING' in data_key:
                    imaging_key = data_key

            if phase_cor_key == '' or phase_cor_reverse_key == '':

                logging.warning("GSTAR Recon: No full set of phase correction data available for reverse segment "
                                "correction")

            else:

                phase_cor_acq_key = phase_cor_key.replace('NP', 'ACQ')
                corr_line_ind = connection_buffer.meas_data.data[phase_cor_acq_key][0].idx.kspace_encode_step_1

                phase_corr_lines = np.zeros((connection_buffer.meas_data(phase_cor_key, 'COL'), 3), dtype=complex)

                for i_rep in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'REP')):
                    for i_phase in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PHS')):
                        for i_set in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SET')):
                            for i_slc in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SLC')):
                                for i_cha in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CHA')):
                                    for i_seg in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'SEG')):

                                        if connection_buffer.meas_data.data[phase_cor_reverse_key].shape[9] == 2:

                                            phase_corr_lines[:, 0] = \
                                            connection_buffer.meas_data.data[phase_cor_reverse_key][
                                                :, i_cha, corr_line_ind, 0, i_slc, 0, i_phase, 0, i_rep, 0, i_seg]

                                            phase_corr_lines[:, 1] = connection_buffer.meas_data.data[phase_cor_key][
                                                :, i_cha, corr_line_ind, 0, i_slc, 0, i_phase, 0, i_rep, 0, i_seg]

                                            phase_corr_lines[:, 2] = \
                                            connection_buffer.meas_data.data[phase_cor_reverse_key][
                                                :, i_cha, corr_line_ind, 0, i_slc, 0, i_phase, 0, i_rep, 1, i_seg]

                                        elif connection_buffer.meas_data.data[phase_cor_key].shape[9] == 2:

                                            phase_corr_lines[:, 0] = connection_buffer.meas_data.data[phase_cor_key][
                                                :, i_cha, corr_line_ind, 0, i_slc, 0, i_phase, 0, i_rep, 0, i_seg]

                                            phase_corr_lines[:, 1] = \
                                            connection_buffer.meas_data.data[phase_cor_reverse_key][
                                                :, i_cha, corr_line_ind, 0, i_slc, 0, i_phase, 0, i_rep, 0, i_seg]

                                            phase_corr_lines[:, 2] = connection_buffer.meas_data.data[phase_cor_key][
                                                :, i_cha, corr_line_ind, 0, i_slc, 0, i_phase, 0, i_rep, 1, i_seg]

                                        phase_corr_drift = epi_tools.calc_linear_phase_correction(phase_corr_lines)

                                        for i_con in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'CON')):
                                            for i_par in range(0, connection_buffer.meas_data('NP_IS_IMAGING', 'PE2')):

                                                if connection_buffer.meas_data.data[phase_cor_reverse_key].shape[9] == 2:

                                                    par_ksp_data = connection_buffer.meas_data.data[imaging_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, i_seg]
                                                    par_ksp_data_corr = epi_tools.correct_linear_phase_drift(
                                                        par_ksp_data, phase_corr_drift)
                                                    connection_buffer.meas_data.data[imaging_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, i_seg] = par_ksp_data_corr


                                                elif connection_buffer.meas_data.data[phase_cor_key].shape[9] == 2:

                                                    par_ksp_data = \
                                                    connection_buffer.meas_data.data[imaging_reverse_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, i_seg]
                                                    par_ksp_data_corr = epi_tools.correct_linear_phase_drift(
                                                        par_ksp_data, phase_corr_drift)
                                                    connection_buffer.meas_data.data[imaging_reverse_key][
                                                        :, i_cha, :, i_par, i_slc, i_set, i_phase, i_con, i_rep, 0, i_seg] = par_ksp_data_corr

                connection_buffer.meas_data.data[imaging_key] = connection_buffer.meas_data.data[imaging_key] + \
                                                                connection_buffer.meas_data.data[imaging_reverse_key]
                del connection_buffer.meas_data.data[phase_cor_reverse_key]
                del connection_buffer.meas_data.data[imaging_reverse_key]

        connection_buffer.meas_data.data["NP_IS_IMAGING"] = np.expand_dims(
            np.sum(connection_buffer.meas_data.data["NP_IS_IMAGING"], 10), 10)
        return connection_buffer
