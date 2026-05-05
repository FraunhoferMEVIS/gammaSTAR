"""!
@brief Segment combination module of gammaSTAR Reconstructions
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
        return SegmentCombinationModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies segment combination to individual k-space segments in the "NP_IS_IMAGING" data to form
               the final combined dataset. If the "meas_data" dictionary of the ConnectionBuffer bject contains
               any "ACQ_IS_REVERSE" flags, the module applies the corresponding EPI-based phase correction technique
               using a three-lines reference scan if possible. If such data is not available, the module just sums
               individual segments.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        if 'NP_IS_IMAGING' not in con_buff.meas_data.data:
            logging.warning("gs-recon: No imaging data available for segment combination, skipping")
            return con_buff, book_keeper

        b_is_reverse = any('ACQ_IS_REVERSE' in data_key for data_key in con_buff.meas_data.data)

        if b_is_reverse:  # We received reversely sampled data -> correct shifts

            logging.info("gs-recon: Applying reverse segment correction for EPI-like data")

            # We identify the correct keys for reverse and non-reverse imaging and phasecor data
            phase_cor_reverse_key = ''
            imaging_reverse_key = ''
            phase_cor_key = ''
            imaging_key = ''
            for data_key in con_buff.meas_data.data:
                if 'NP_IS_REVERSE' in data_key and 'NP_IS_PHASECORR_DATA' in data_key:
                    phase_cor_reverse_key = data_key
                if 'NP_IS_REVERSE' not in data_key and 'NP_IS_PHASECORR_DATA' in data_key:
                    phase_cor_key = data_key
                if 'NP_IS_REVERSE' in data_key and 'NP_IS_IMAGING' in data_key:
                    imaging_reverse_key = data_key
                if 'NP_IS_REVERSE' not in data_key and 'NP_IS_IMAGING' in data_key:
                    imaging_key = data_key

            if phase_cor_key == '' or phase_cor_reverse_key == '':

                logging.warning("gs-recon: No full set of phase correction data available for reverse segment "
                                "correction")

            else:

                # Extract the actual phasecor lines
                phase_cor_acq_key = phase_cor_key.replace('NP', 'ACQ')
                corr_line_ind_ksp1 = con_buff.meas_data.data[phase_cor_acq_key][0].idx.kspace_encode_step_1
                corr_line_ind_ksp2 = con_buff.meas_data.data[phase_cor_acq_key][0].idx.kspace_encode_step_2

                phasecor_data = np.expand_dims(
                    np.take(
                        np.expand_dims(
                            np.take(
                                con_buff.meas_data.data[phase_cor_key],
                                corr_line_ind_ksp1,
                                axis=2
                            ),
                            axis=2
                        ),
                        corr_line_ind_ksp2, axis=3
                    ),
                    axis=3)

                # Extract the actual phasecor reverse lines
                phasecor_data_reverse = np.expand_dims(
                    np.take(
                        np.expand_dims(
                            np.take(
                                con_buff.meas_data.data[phase_cor_reverse_key],
                                corr_line_ind_ksp1,
                                axis=2
                            ),
                            axis=2
                        ),
                        corr_line_ind_ksp2, axis=3
                    ),
                    axis=3)

                # Calculate phase shifts
                phase_shifts = epi_tools.calculate_phase_shifts(
                    phasecor_data=phasecor_data,
                    phasecor_data_reverse=phasecor_data_reverse,
                    dir_axis=9
                )

                # Remove phase shifts
                if con_buff.meas_data.data[phase_cor_reverse_key].shape[9] == 2:

                    con_buff.meas_data.data[imaging_key] = epi_tools.remove_phase_shifts(
                        con_buff.meas_data.data[imaging_key],
                        phase_shifts
                    )

                elif con_buff.meas_data.data[phase_cor_key].shape[9] == 2:

                    con_buff.meas_data.data[imaging_reverse_key] = epi_tools.remove_phase_shifts(
                        con_buff.meas_data.data[imaging_reverse_key],
                        phase_shifts
                    )

                # Combine EPI segments
                con_buff.meas_data.data[imaging_key] = (
                        con_buff.meas_data.data[imaging_key] +
                        con_buff.meas_data.data[imaging_reverse_key]
                )

                del con_buff.meas_data.data[phase_cor_reverse_key]
                del con_buff.meas_data.data[imaging_reverse_key]

                book_keeper.recon_history += "_CorrectedEPIShifts"

        # Combine segment dimension
        con_buff.meas_data.data["NP_IS_IMAGING"] = np.expand_dims(
            np.sum(
                con_buff.meas_data.data["NP_IS_IMAGING"], 10
            ), 10
        )

        return con_buff, book_keeper
