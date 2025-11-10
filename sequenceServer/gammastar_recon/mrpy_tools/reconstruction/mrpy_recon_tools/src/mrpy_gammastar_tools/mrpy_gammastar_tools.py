"""!
@brief Collection of tools which provide gammastar related functionality, e.g. handling of raw representations.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np

def reorder_twix_with_raw_reps(raw_adc_representations: list, twix_np_phasecor: np.ndarray, twix_np_rtfeedback: np.ndarray, twix_np_paracalib: np.ndarray, twix_np_imaging: np.ndarray) -> np.ndarray:
    """!
    @brief This function combines measured data which was loaded from a twix dataset into a single numpy array where
           individual readouts are sorted based on their occurence in the raw representations with respective header
           entries.

    @param raw_adc_representations: (list[json]) A list of json encoded raw representations
    @param twix_np_phasecor: (np.ndarray) 3D Numpy array, containing phasecorrection data from exported twix data
    @param twix_np_rtfeedback: (np.ndarray) 3D Numpy array, containing rt feedback data from exported twix data
    @param twix_np_paracalib: (np.ndarray) 3D Numpy array, containing parallel calibration data from exported twix data
    @param twix_np_imaging: (np.ndarray) 3D Numpy array, containing imaging data from exported twix data

    @return
        - (np.ndarray) 3D Numpy array with linearly ordered readouts along third dimension based on occurence in
                       gammaSTAR sequence.

    @author Jörn Huber
    """

    twix_np_reorder = np.zeros((twix_np_imaging.shape[0], twix_np_imaging.shape[1], len(raw_adc_representations)), dtype=complex)
    reorder_ind = 0
    phasecor_ind = 0
    paracalib_ind = 0
    rtfeedback_ind = 0
    imaging_ind = 0

    for raw_rep in raw_adc_representations:

        if 'ACQ_IS_PHASECORR_DATA' in raw_rep['adc_header']:
            twix_np_reorder[:, :, reorder_ind] = twix_np_phasecor[:, :, phasecor_ind]
            phasecor_ind += 1

        elif 'ACQ_IS_RTFEEDBACK_DATA' in raw_rep['adc_header']:
            twix_np_reorder[:, :, reorder_ind] = twix_np_rtfeedback[:, :, rtfeedback_ind]
            rtfeedback_ind += 1

        elif 'ACQ_IS_PARALLEL_CALIBRATION' in raw_rep['adc_header']:
            twix_np_reorder[:, :, reorder_ind] = twix_np_paracalib[:, :, paracalib_ind]
            paracalib_ind += 1

        else:
            twix_np_reorder[:, :, reorder_ind] = twix_np_imaging[:, :, imaging_ind]
            imaging_ind += 1

        reorder_ind += 1

    return twix_np_reorder
