"""!
@brief Acquisition conversion module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import ismrmrd
from scipy.interpolate import interp1d
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_noncart_tools as noncart_tools
import mrpy_helpers as helpers
from typing import Tuple


class AcquisitionConversionModule:
    """!
    @brief This module converts received ISMRMRD acquisitions into numpy array structures for further processing.
           Note: This module provides basic functionality for gammaSTAR sequences if used together with the
           sequence information from the frontend.
    """

    @staticmethod
    def ismrmrd_acqs_to_numpy_array(
            list_of_acqs: list[ismrmrd.Acquisition],
            encoded_space: object | None = None,
            os_factor: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """!
        @brief Sort k-space data from acquisitions into numpy array structures for further processing.
        @details The function first analyzes the maximum idx indices which are available in the list of provided
                 acquisitions. Maximum indices are used to create the numpy structure and to sort the data into that
                 structure. If acquisitions provide non-Cartesian data, the data is first regridded to the Cartesian 
                 grid as defined by the encoded space. If data was sampled using ramp sampling, regridding of individual 
                 readouts is first applied.

        @param list_of_acqs: List of ismrmrd.Acquisition objects.
        @param encoded_space: Encoded space dimensions
        @param os_factor: Readout oversampling factor.

        @return
            - 11-D Numpy array of size (number_of_samples, max_kspace_encoding_pe1,
              max_kspace_encoding_pe2, num_active_channels, max_slice, max_set, max_phase, max_contrast,
              max_repetition, max_average, max_segment) with sorted acquisition.

        @author Jörn Huber
        """

        readout_type, is_ramp_sample, _, _ = ismrmrd_tools.identify_readout_type_from_acqs(list_of_acqs)

        ramp_sampe_string = ''
        if is_ramp_sample:
            ramp_sampe_string = " with 1D resampling"
        if readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN:
            logging.info('gs-recon:   Cartesian readout' + ramp_sampe_string)
        elif readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:
            logging.info('gs-recon:   Non-Cartesian 2D readout')
        elif readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:
            logging.info('gs-recon:   Non-Cartesian 3D readout')

        num_active_channels = list_of_acqs[0].active_channels
        number_of_samples = list_of_acqs[0].number_of_samples
        if encoded_space is not None:
            max_kspace_encoding_pe1 = encoded_space.matrixSize.y
            max_kspace_encoding_pe2 = encoded_space.matrixSize.z
        else:
            max_kspace_encoding_pe1 = max(list_of_acqs,
                                          key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1 + 1
            max_kspace_encoding_pe2 = max(list_of_acqs,
                                          key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2 + 1

        max_slice = max(list_of_acqs, key=lambda acq: acq.idx.slice).idx.slice + 1
        max_segment = max(list_of_acqs, key=lambda acq: acq.idx.segment).idx.segment + 1
        max_set = max(list_of_acqs, key=lambda acq: acq.idx.set).idx.set + 1
        max_phase = max(list_of_acqs, key=lambda acq: acq.idx.phase).idx.phase + 1
        max_contrast = max(list_of_acqs, key=lambda acq: acq.idx.contrast).idx.contrast + 1
        max_average = max(list_of_acqs, key=lambda acq: acq.idx.average).idx.average + 1
        max_repetition = max(list_of_acqs, key=lambda acq: acq.idx.repetition).idx.repetition + 1

        acq_data_np = np.zeros((number_of_samples,
                                num_active_channels,
                                max_kspace_encoding_pe1,
                                max_kspace_encoding_pe2,
                                max_slice,
                                max_set,
                                max_phase,
                                max_contrast,
                                max_repetition,
                                max_average,
                                max_segment), dtype=complex)

        acq_traj_np = np.zeros((3,
                                number_of_samples,
                                max_kspace_encoding_pe1,
                                max_kspace_encoding_pe2,
                                max_slice,
                                max_set,
                                max_phase,
                                max_contrast,
                                max_repetition,
                                max_average,
                                max_segment))

        for acq in list_of_acqs:
            acq_flags = ismrmrd_tools.bitmask_to_flags(acq.getHead().flags)
            data = np.transpose(acq.data, (1, 0))
            traj = np.transpose(acq.traj, (1, 0))

            if any('ACQ_IS_REVERSE' in s for s in acq_flags):
                data = np.flipud(data)

            acq_data_np[:,
                        :,
                        acq.idx.kspace_encode_step_1,
                        acq.idx.kspace_encode_step_2,
                        acq.idx.slice,
                        acq.idx.set,
                        acq.idx.phase,
                        acq.idx.contrast,
                        acq.idx.repetition,
                        acq.idx.average,
                        acq.idx.segment] = data

            if traj.shape[0] != 0:
                acq_traj_np[:,
                            :,
                            acq.idx.kspace_encode_step_1,
                            acq.idx.kspace_encode_step_2,
                            acq.idx.slice,
                            acq.idx.set,
                            acq.idx.phase,
                            acq.idx.contrast,
                            acq.idx.repetition,
                            acq.idx.average,
                            acq.idx.segment] = traj

        if readout_type in [ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D,
                            ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D]:

            nufftdim = 2 if readout_type == 3 else 3
            if nufftdim == 2:
                acq_traj_np = np.take(acq_traj_np, axis=0, indices=(0, 1))

            nufft_data = noncart_tools.apply_nufft(
                ksp_data = acq_data_np,
                traj_data = np.moveaxis(acq_traj_np, 0, 1),
                mat_size = encoded_space.matrixSize.x,
                ro_axis = ismrmrd_tools.IsmrmrdConstants.IDX_MAP['COL'],
                ch_axis = ismrmrd_tools.IsmrmrdConstants.IDX_MAP['CHA'],
                acq_axis = ismrmrd_tools.IsmrmrdConstants.IDX_MAP['PE1'],
                pe2_axis = ismrmrd_tools.IsmrmrdConstants.IDX_MAP['PE2'] if nufftdim == 3 else None,
            )

            # Back to k-space for further processing
            axes = (0, 2) if nufftdim == 2 else (0, 2, 3)
            acq_data_np = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.fftshift(
                        nufft_data,
                        axes=axes
                    ),
                    axes=axes
                ),
                axes=axes
            )

            # Remove OS along PE1 and PE2 (if applicable)
            pe_axes = (2,) if nufftdim == 2 else (2, 3)
            acq_data_np = helpers.remove_os(acq_data_np, int(os_factor), pe_axes)

        elif is_ramp_sample and (readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN):

            regrid_traj = list_of_acqs[0].traj[:, 0]

            regrid_traj = regrid_traj - np.min(regrid_traj)
            regrid_traj = regrid_traj / np.max(regrid_traj)
            regrid_traj = regrid_traj * list_of_acqs[0].number_of_samples

            if regrid_traj[0] > regrid_traj[-1]:
                integer_grid = np.arange(list_of_acqs[0].number_of_samples, 0, -1)
            else:
                integer_grid = np.arange(0, list_of_acqs[0].number_of_samples)

            regrid_interpolator = interp1d(
                x=regrid_traj,
                y=acq_data_np,
                kind='cubic',
                fill_value='extrapolate',
                axis=0
            )
            acq_data_np = regrid_interpolator(integer_grid)

        # Last step: We want to remove the readout oversampling
        acq_data_np = helpers.remove_os(acq_data_np, int(os_factor), (0,))

        return acq_data_np, readout_type

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
        return AcquisitionConversionModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies the modules functionality by converting received ISMRMRD acquisitions into numpy array
               structures for further processing.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        is_process = (any('ACQ' in acq_key for acq_key in con_buff.meas_data.data.keys()) and len(con_buff.headers) > 0)

        if is_process:
            logging.info("gs-recon: Processing acquisitions")

            con_buff.meas_data.encoded_space = con_buff.headers[0].encoding[0].encodedSpace
            con_buff.meas_data.recon_space = con_buff.headers[0].encoding[0].reconSpace
            con_buff.meas_data.accel_pe1 = con_buff.headers[0].encoding[
                0].parallelImaging.accelerationFactor.kspace_encoding_step_1
            con_buff.meas_data.accel_pe2 = con_buff.headers[0].encoding[
                0].parallelImaging.accelerationFactor.kspace_encoding_step_2

            fov_x_enc = con_buff.meas_data.encoded_space.fieldOfView_mm.x
            fov_x_rec = con_buff.meas_data.recon_space.fieldOfView_mm.x
            os_factor = fov_x_enc / fov_x_rec

            if 'ACQ_IS_IMAGING' in con_buff.meas_data.data:

                res = ismrmrd_tools.identify_readout_type_from_acqs(con_buff.meas_data.data['ACQ_IS_IMAGING'])

                con_buff.meas_data.imaging_readout_type = res[0]
                con_buff.meas_data.is_propeller = res[2]
                con_buff.meas_data.blade_dim = res[3]

                if (con_buff.meas_data.imaging_readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN
                        and not con_buff.meas_data.is_propeller):
                    con_buff.meas_data.pf_factor_pe1 = (con_buff.meas_data.encoded_space.matrixSize.y
                                                                 / con_buff.meas_data.recon_space.matrixSize.y)
                    con_buff.meas_data.pf_factor_pe2 = (con_buff.meas_data.encoded_space.matrixSize.z
                                                                 / con_buff.meas_data.recon_space.matrixSize.z)
                else:
                    con_buff.meas_data.pf_factor_pe1 = 1.0  # Non-Cartesian or PROPELLER data
                    con_buff.meas_data.pf_factor_pe2 = 1.0  # Non-Cartesian or PROPELLER data

            con_buff.meas_data.meas_header = con_buff.headers[0]
            pf_str = str(con_buff.meas_data.pf_factor_pe1) + "x" + str(con_buff.meas_data.pf_factor_pe2)
            logging.info("gs-recon:  Partial Fourier PE1xPE2: " + pf_str)

            pi_str = str(con_buff.meas_data.accel_pe1) + "x" + str(con_buff.meas_data.accel_pe2)
            logging.info("gs-recon:  Parallel Imaging PE1xPE2 " + pi_str)

            acq_keys = list(con_buff.meas_data.data.keys())
            for acq_key in acq_keys:
                np_key = acq_key.replace('ACQ', 'NP')
                logging.info('gs-recon:  Checking readout type of ' + acq_key + " data")
                if "IS_IMAGING" in np_key:

                    res = AcquisitionConversionModule.ismrmrd_acqs_to_numpy_array(
                        list_of_acqs = con_buff.meas_data.data[acq_key],
                        encoded_space = con_buff.meas_data.encoded_space,
                        os_factor = os_factor
                    )
                    con_buff.meas_data.data[np_key] = res[0]
                    readout_type = res[1]

                    if readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:
                        book_keeper.recon_history += "_NUFFT2D"
                    elif readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:
                        book_keeper.recon_history += "_NUFFT3D"

                elif 'IS_RTFEEDBACK' not in np_key:  # No Processing for RT Feedback data for now
                    con_buff.meas_data.data[np_key], _ = AcquisitionConversionModule.ismrmrd_acqs_to_numpy_array(
                        list_of_acqs = con_buff.meas_data.data[acq_key],
                        encoded_space = None,
                        os_factor = os_factor
                    )

            if 'NP_IS_PARALLEL_CALIBRATION' in con_buff.meas_data.data:
                con_buff.meas_data.calib_reg_pe1 = (
                    min(con_buff.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1,
                    max(con_buff.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_1).idx.kspace_encode_step_1 + 1)

                con_buff.meas_data.calib_reg_pe2 = (
                    min(con_buff.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2,
                    max(con_buff.meas_data.data['ACQ_IS_PARALLEL_CALIBRATION'],
                        key=lambda acq: acq.idx.kspace_encode_step_2).idx.kspace_encode_step_2 + 1)

        return con_buff, book_keeper
