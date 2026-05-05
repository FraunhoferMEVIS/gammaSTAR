"""!
@brief Collection of tools which provide helpful functionality.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

from typing import Tuple
import numpy as np


def remove_os(
        ksp_data: np.ndarray,
        os_factor: int,
        k_axes:Tuple
) -> np.ndarray:
    """!
    @brief Removes oversampling along k_axes.

    @param ksp_data: N-dimensional kspace data
    @param os_factor: Oversampling factor
    @param k_axes: Axes over which to remove the oversampling

    @return
        - k-space array with removed oversampling

    @author Jörn Huber
    """

    if os_factor == 1:
        return ksp_data

    # Go to image space
    im_data = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(ksp_data, axes=k_axes),axes=k_axes),axes=k_axes)

    # Remove OS along k_axes
    for axis in k_axes:
        num_samples = im_data.shape[axis]
        samples_cutoff = int((num_samples - num_samples / os_factor) / 2)
        im_data = np.take(im_data, range(samples_cutoff, int(num_samples-samples_cutoff)), axis=axis)

    # Back to k-space
    ksp_data_rem_os = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im_data,axes=k_axes),axes=k_axes),axes=k_axes)

    return ksp_data_rem_os


def filter_low_order_phase(
    ksp_data: np.ndarray,
    k_axes: Tuple[int, int, int] | Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """!
    @brief Removes low order phase shifts from blade image data, which is the result from eddy currents etc.
    @details Phase correction is needed as blades need to share a common k-space center before gridding. Therefore,
             k-space frequencies are low-pass filtered such that low order phase variation is preserved in image space
             and such phase variation is removed from individual blades by multiplication with the filtered complex
             conjugate.

    @param ksp_data: K-space data of arbitrary dimensions.
    @param k_axes: Axis along which the high-pass filter shall be applied.

    @return
        - 2D complex numpy array of phase corrected k-space data

    @author Jörn Huber, Tom Lütjen
    """

    num_col = ksp_data.shape[k_axes[0]]
    num_lin = ksp_data.shape[k_axes[1]]
    num_par = 0
    if len(k_axes) == 3:
        num_par = ksp_data.shape[k_axes[2]]

    r_col = np.arange(num_col)
    filt_col = (num_col + 1 - np.abs(r_col - r_col[::-1])) / 2

    r_lin = np.arange(num_lin)
    filt_lin = (num_lin + 1 - np.abs(r_lin - r_lin[::-1])) / 2

    filt_par = np.ones(1)
    if num_par > 0:
        r_par = np.arange(num_par)
        filt_par = (num_par + 1 - np.abs(r_par - r_par[::-1])) / 2

    fc, fl, fp = np.ix_(filt_col, filt_lin, filt_par)
    triang_filter = fc * fl * fp

    triang_filter = triang_filter / np.amax(triang_filter)

    filter_bc_shape = np.ones(len(ksp_data.shape), dtype=int)
    filter_bc_shape[k_axes[0]] = num_col
    filter_bc_shape[k_axes[1]] = num_lin
    if len(k_axes) == 3:
        filter_bc_shape[k_axes[2]] = num_par
    triang_filter = np.reshape(triang_filter, filter_bc_shape)

    for i_ax in range(0, len(ksp_data.shape)):
        if i_ax == k_axes[0] or i_ax == k_axes[1] or (len(k_axes) == 3 and i_ax == k_axes[2]):
            continue
        else:
            triang_filter = np.repeat(triang_filter, ksp_data.shape[i_ax], axis=i_ax)

    ksp_data_filt = np.multiply(ksp_data, triang_filter)

    im = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(ksp_data, axes=k_axes), axes=k_axes), axes=k_axes
    )
    im_filt = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(ksp_data_filt, axes=k_axes), axes=k_axes), axes=k_axes
    )

    if im_filt.any() == 0.0:
        return ksp_data, triang_filter

    im_filt = im_filt / np.abs(im_filt)

    im = np.multiply(im, np.conj(im_filt))

    data_corr = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(im, axes=k_axes), axes=k_axes), axes=k_axes
    )

    return data_corr, triang_filter
