"""!
@brief Acquisition conversion module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import ismrmrd
import mrinufft
import math
from mrinufft.density import voronoi
from scipy.interpolate import interp1d
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_helpers as helpers
from typing import Tuple  

class AcquisitionConversionModule:
    """!
    @brief This module converts received ISMRMRD acquisitions into numpy array structures for further processing.
           Note: This module provides basic functionality for gammaSTAR sequences if used together with the
           sequence information from the frontend.
    """

    @staticmethod
    def ismrmrd_acqs_to_numpy_array(list_of_acqs: list[ismrmrd.Acquisition],
                                    encoded_space: object | None = None,
                                    recon_space: object | None = None,
                                    W: np.ndarray | None = None,
                                    os_factor: float = 1) -> Tuple[np.ndarray, int]:
        """!
        @brief Sort k-space data from acquisitions into numpy array structures for further processing.
        @details The function first analyzes the maximum idx indices which are available in the list of provided
                 acquisitions. Maximum indices are used to create the numpy structure and to sort the data into that
                 structure. If acquisitions provide non-Cartesian data, the data is first regridded to the Cartesian 
                 grid as defined by the encoded space. If data was sampled using ramp sampling, regridding of individual 
                 readouts is first applied.

        @param list_of_acqs: (list) List of ismrmrd.Acquisition objects.
        @param encoded_space: (ISMRMRD encoded space object) Encoded space dimensions
        @param recon_space: (ISMRMRD recon space object) Encoded space dimensions
        @param W: Noise de-correlation matrix of size (num_cha, num_cha).
        @param os_factor: (float) Readout oversampling factor.

        @return
            - (np.ndarray) 11-D Numpy array of size (number_of_samples, max_kspace_encoding_pe1, 
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
        num_compressed_cha = 10
        target_channels = num_active_channels
        if num_active_channels > num_compressed_cha:
            target_channels = num_compressed_cha

        number_of_samples = list_of_acqs[0].number_of_samples  # encoded_space.matrixSize.x
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

        for acq in list_of_acqs:
            acq_flags = ismrmrd_tools.bitmask_to_flags(acq.getHead().flags)
            data = np.transpose(acq.data, (1, 0))

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

        # --> Based on mri-nufft (see third_party_licenses.txt)

        if readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:

            acq_data_np_grid = np.zeros((encoded_space.matrixSize.x,
                                         num_active_channels,
                                         encoded_space.matrixSize.x,
                                         max_kspace_encoding_pe2,
                                         max_slice,
                                         max_set,
                                         max_phase,
                                         max_contrast,
                                         max_repetition,
                                         max_average,
                                         max_segment), dtype=complex)

            # Prepare sampling trajectory
            samples_loc = mrinufft.initialize_2D_radial(Nc=max_kspace_encoding_pe1,
                                                        Ns=list_of_acqs[0].number_of_samples)

            NufftOperator = mrinufft.get_operator("finufft")
            for i_rep in range(0, max_repetition):
                for i_ave in range(0, max_average):
                    for i_con in range(0, max_contrast):
                        for i_phs in range(0, max_phase):
                            for i_set in range(0, max_set):
                                for i_seg in range(0, max_segment):
                                    for i_slc in range(0, max_slice):
                                        for i_par in range(0, max_kspace_encoding_pe2):

                                            grid_data = acq_data_np[:, :, :,
                                            i_par, i_slc, i_set, i_phs, i_con, i_rep,
                                            i_ave, i_seg]

                                            logging.info(
                                                'gs-recon:    NUFFT for slice %d set %d phase %d contrast %d '
                                                'repetition %d average %d segment %d',
                                                i_slc, i_set, i_phs, i_con, i_rep, i_ave, i_seg)
                                            for acq in list_of_acqs:
                                                if (i_slc == acq.idx.slice and i_set == acq.idx.set and
                                                        i_phs == acq.idx.phase and i_con == acq.idx.contrast and
                                                        i_rep == acq.idx.repetition and i_ave == acq.idx.average and
                                                        i_seg == acq.idx.segment and i_par == acq.idx.kspace_encode_step_2):
                                                    samples_loc[acq.idx.kspace_encode_step_1, :, :] = acq.traj[:, 0:-1]

                                            grid_data = acq_data_np[
                                                :, :, :, i_par, i_slc, i_set, i_phs, i_con, i_rep, i_ave, i_seg]
                                            grid_data = np.transpose(grid_data, [1, 2, 0])
                                            grid_data = np.reshape(grid_data, (grid_data.shape[0], -1))

                                            if num_active_channels > num_compressed_cha:
                                                grid_data, _ = mrinufft.extras.smaps.coil_compression(
                                                    kspace_data=grid_data,
                                                    K=num_compressed_cha)

                                            samples_loc = samples_loc / np.max(samples_loc) * math.pi

                                            samples_loc = samples_loc / np.max(samples_loc) * math.pi
                                            density = voronoi(samples_loc.reshape(-1, 2))
                                            nufft = NufftOperator(
                                                samples_loc.reshape(-1, 2),
                                                shape=(encoded_space.matrixSize.x, encoded_space.matrixSize.x),
                                                density=density, n_coils=target_channels
                                            )

                                            cart_data = nufft.adj_op(grid_data)

                                            if target_channels == 1:
                                                cart_data = np.reshape(cart_data, [1,
                                                                                   cart_data.shape[0],
                                                                                   cart_data.shape[1]])

                                            cart_data = np.fft.fftshift(
                                                np.fft.ifft(np.fft.fftshift(cart_data, axes=1), axis=1), axes=1)
                                            cart_data = np.fft.fftshift(
                                                np.fft.ifft(np.fft.fftshift(cart_data, axes=2), axis=2), axes=2)

                                            cart_data = np.flip(np.transpose(cart_data, [1, 0, 2]), 2)

                                            acq_data_np_grid[:, :, :,
                                            i_par, i_slc, i_set, i_phs, i_con, i_rep,
                                            i_ave, i_seg] = cart_data

            acq_data_np = acq_data_np_grid

        elif readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:

            acq_data_np_grid = np.zeros((encoded_space.matrixSize.x,
                                         target_channels,
                                         encoded_space.matrixSize.x,
                                         encoded_space.matrixSize.x,
                                         max_slice,
                                         max_set,
                                         max_phase,
                                         max_contrast,
                                         max_repetition,
                                         max_average,
                                         max_segment), dtype=complex)

            # Prepare NUFFT Operator
            samples_loc = mrinufft.initialize_3D_golden_means_radial(Nc=max_kspace_encoding_pe1,
                                                                     Ns=list_of_acqs[0].number_of_samples)

            NufftOperator = mrinufft.get_operator("finufft")
            for i_rep in range(0, max_repetition):
                for i_ave in range(0, max_average):
                    for i_con in range(0, max_contrast):
                        for i_phs in range(0, max_phase):
                            for i_set in range(0, max_set):
                                for i_seg in range(0, max_segment):
                                    for i_slc in range(0, max_slice):

                                        logging.info(
                                            'gs-recon:    NUFFT for slice %d set %d phase %d contrast %d '
                                            'repetition %d average %d segment %d',
                                            i_slc, i_set, i_phs, i_con, i_rep, i_ave, i_seg)
                                        for acq in list_of_acqs:
                                            if (i_slc == acq.idx.slice and i_set == acq.idx.set and
                                                    i_phs == acq.idx.phase and i_con == acq.idx.contrast and
                                                    i_rep == acq.idx.repetition and i_ave == acq.idx.average and
                                                    i_seg == acq.idx.segment):
                                                samples_loc[acq.idx.kspace_encode_step_1, :, :] = acq.traj

                                        grid_data = acq_data_np[
                                            :, :, :, 0, i_slc, i_set, i_phs, i_con, i_rep, i_ave, i_seg]
                                        grid_data = np.transpose(grid_data, [1, 2, 0])
                                        grid_data = np.reshape(grid_data, (grid_data.shape[0], -1))

                                        if num_active_channels > num_compressed_cha:
                                            grid_data, _ = mrinufft.extras.smaps.coil_compression(kspace_data=grid_data,
                                                                                                  K=num_compressed_cha)

                                        samples_loc = samples_loc / np.max(samples_loc) * math.pi
                                        density = voronoi(samples_loc.reshape(-1, 3))
                                        nufft = NufftOperator(
                                            samples_loc.reshape(-1, 3),
                                            shape=(encoded_space.matrixSize.x, encoded_space.matrixSize.x,
                                                   encoded_space.matrixSize.x), density=density, n_coils=target_channels
                                        )

                                        cart_data = nufft.adj_op(grid_data)

                                        if target_channels == 1:
                                            cart_data = np.reshape(cart_data, [1,
                                                                               cart_data.shape[0],
                                                                               cart_data.shape[1],
                                                                               cart_data.shape[2]])

                                        cart_data = np.fft.fftshift(
                                            np.fft.ifft(np.fft.fftshift(cart_data, axes=1), axis=1), axes=1)
                                        cart_data = np.fft.fftshift(
                                            np.fft.ifft(np.fft.fftshift(cart_data, axes=2), axis=2), axes=2)
                                        cart_data = np.fft.fftshift(
                                            np.fft.ifft(np.fft.fftshift(cart_data, axes=3), axis=3), axes=3)

                                        cart_data = np.flip(np.transpose(cart_data, [1, 0, 2, 3]), 2)
                                        acq_data_np_grid[:, :, :,
                                        :, i_slc, i_set, i_phs, i_con, i_rep,
                                        i_ave, i_seg] = np.flipud(cart_data)

            acq_data_np = acq_data_np_grid

        # <-- Based on mri-nufft (see third_party_licenses.txt)

        elif is_ramp_sample and (readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN):

            regrid_traj = list_of_acqs[0].traj[:, 0]

            regrid_traj = regrid_traj - np.min(regrid_traj)
            regrid_traj = regrid_traj / np.max(regrid_traj)
            regrid_traj = regrid_traj * list_of_acqs[0].number_of_samples

            if regrid_traj[0] > regrid_traj[-1]:
                integer_grid = np.arange(list_of_acqs[0].number_of_samples, 0, -1)
            else:
                integer_grid = np.arange(0, list_of_acqs[0].number_of_samples)

            for i_rep in range(0, max_repetition):
                for i_ave in range(0, max_average):
                    for i_con in range(0, max_contrast):
                        for i_phs in range(0, max_phase):
                            for i_set in range(0, max_set):
                                for i_seg in range(0, max_segment):
                                    for i_slc in range(0, max_slice):
                                        for i_par in range(0, max_kspace_encoding_pe2):
                                            for i_cha in range(0, num_active_channels):
                                                for i_pe in range(0, max_kspace_encoding_pe1):
                                                    data = acq_data_np[:, i_cha, i_pe, i_par, i_slc, i_set, i_phs,
                                                    i_con, i_rep, i_ave, i_seg]

                                                    regrid_interpolator = interp1d(regrid_traj, np.squeeze(data),
                                                                                   kind='cubic',
                                                                                   fill_value='extrapolate')
                                                    resampled_data = regrid_interpolator(integer_grid)

                                                    acq_data_np[:, i_cha, i_pe, i_par, i_slc, i_set, i_phs, i_con,
                                                                i_rep, i_ave, i_seg] = resampled_data

        # Last step: We want to remove the readout oversampling
        if os_factor > 1:

            acq_data_np = helpers.remove_readout_os(acq_data_np, 0, os_factor)

            if readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:
                acq_data_np = helpers.remove_readout_os(acq_data_np, 2, os_factor)

            if readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:
                acq_data_np = helpers.remove_readout_os(acq_data_np, 2, os_factor)
                acq_data_np = helpers.remove_readout_os(acq_data_np, 3, os_factor)

        return acq_data_np, readout_type

    @staticmethod
    def __call__(con_buff: ismrmrd_tools.ConnectionBuffer,
                 book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief ()-Operator, which applies the modules functionality as defined in the "apply" method.

        @param con_buff: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (book_keeper) Object which stores calibration data etc

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            -  (book_keeper) Object which stores calibration data etc

        @author Jörn Huber
        """
        return AcquisitionConversionModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(con_buff: ismrmrd_tools.ConnectionBuffer,
              book_keeper: "BookKeeper") -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Applies the modules functionality by averaging data along the average dimension. 

        @param con_buff: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (dict) Dictionary which is used to store image processing results.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            -  (dict) Dictionary which is used to store image processing results.

        @author Jörn Huber
        """

        is_process = (any('ACQ' in acq_key for acq_key in con_buff.meas_data.data.keys())
                      and len(con_buff.headers) > 0)

        if is_process:
            logging.info("gs-recon: Processing acquisitions")

            con_buff.meas_data.encoded_space = con_buff.headers[0].encoding[0].encodedSpace
            con_buff.meas_data.recon_space = con_buff.headers[0].encoding[0].reconSpace
            con_buff.meas_data.accel_pe1 = con_buff.headers[0].encoding[
                0].parallelImaging.accelerationFactor.kspace_encoding_step_1
            con_buff.meas_data.accel_pe2 = con_buff.headers[0].encoding[
                0].parallelImaging.accelerationFactor.kspace_encoding_step_2

            fov_x = con_buff.meas_data.encoded_space.fieldOfView_mm.x
            fov_y = con_buff.meas_data.encoded_space.fieldOfView_mm.y
            os_factor = fov_x / fov_y

            if 'ACQ_IS_IMAGING' in con_buff.meas_data.data:

                res = ismrmrd_tools.identify_readout_type_from_acqs(con_buff.meas_data.data['ACQ_IS_IMAGING'])

                con_buff.meas_data.imaging_readout_type = res[0]
                con_buff.meas_data.is_propeller = res[2]
                con_buff.meas_data.blade_dim = res[3]

                if con_buff.meas_data.imaging_readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_CARTESIAN and not con_buff.meas_data.is_propeller:
                    con_buff.meas_data.pf_factor_pe1 = (con_buff.meas_data.encoded_space.matrixSize.y
                                                                 / con_buff.meas_data.recon_space.matrixSize.y)
                    con_buff.meas_data.pf_factor_pe2 = (con_buff.meas_data.encoded_space.matrixSize.z
                                                                 / con_buff.meas_data.recon_space.matrixSize.z)
                else:
                    con_buff.meas_data.pf_factor_pe1 = 1.0  # Non-Cartesian or PROPELLER data
                    con_buff.meas_data.pf_factor_pe2 = 1.0  # Non-Cartesian or PROPELLER data

            con_buff.meas_data.meas_header = con_buff.headers[0]
            logging.info("gs-recon:  Partial Fourier PE1xPE2: " + str(con_buff.meas_data.pf_factor_pe1) + "x" + str(
                con_buff.meas_data.pf_factor_pe2))
            logging.info("gs-recon:  Parallel Imaging PE1xPE2 " + str(con_buff.meas_data.accel_pe1) + "x" + str(
                con_buff.meas_data.accel_pe2))

            acq_keys = list(con_buff.meas_data.data.keys())
            for acq_key in acq_keys:
                np_key = acq_key.replace('ACQ', 'NP')
                logging.info('gs-recon:  Checking readout type of ' + acq_key + " data")
                if "IS_IMAGING" in np_key:

                    res = AcquisitionConversionModule.ismrmrd_acqs_to_numpy_array(con_buff.meas_data.data[acq_key],
                                                                                  con_buff.meas_data.encoded_space,
                                                                                  con_buff.meas_data.recon_space,
                                                                                  con_buff.meas_data.W,
                                                                                  os_factor)
                    con_buff.meas_data.data[np_key] = res[0]
                    readout_type = res[1]

                    if readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_2D:
                        book_keeper.recon_history += "_NUFFT2D"
                    elif readout_type == ismrmrd_tools.IsmrmrdConstants.READOUT_TYPE_NONCARTESIAN_3D:
                        book_keeper.recon_history += "_NUFFT3D"

                elif 'IS_RTFEEDBACK' not in np_key:  # No Processing for RT Feedback data for now
                    con_buff.meas_data.data[np_key], _ = AcquisitionConversionModule.ismrmrd_acqs_to_numpy_array(con_buff.meas_data.data[acq_key],
                                                                                                                         None,
                                                                                                                         None,
                                                                                                                          con_buff.meas_data.W,
                                                                                                                          os_factor)

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
