"""!
@brief Image scaling module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import numpy as np
import logging

DICOM_VAL_16BIT = 65535.0-1.0  # Maximum value for 16 bit DICOM images
DICOM_VAL_4BIT = 4095.0-1.0  # Maximum value for 4 bit DICOM images


class ImageScaleModule:
    """!
    @brief This module calculates the scaling factor from the reconstructed image series.
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
        return ImageScaleModule.apply(connection_buffer, book_keeper)

    @staticmethod
    def apply(connection_buffer, book_keeper):
        """!
        @brief Calculates the scaling factor from the reconstructed image series.

        @param connection_buffer: (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                                     structures.
        @param book_keeper: (dict) Dictionary which is used to store image processing results.

        @return
            -  (ConnectionBuffer) ConnectionBuffer object, holding processed "NP_..." data
                                  structures.
            -  (dict) Dictionary which is used to store image processing results.

        @author Jörn Huber
        """

        applied_scaling_factor = 1.0
        max_val_images = np.max(np.abs(connection_buffer.meas_data.data['NP_IS_IMAGING']).flatten())

        if max_val_images == 0.0:
            logging.warning("GSTAR Recon: Maximum value of reconstructed images is 0.0")
            max_val_images = 1.0

        if book_keeper.initial_scaling_factor == 0.0:

            initial_scaling_factor = DICOM_VAL_16BIT / max_val_images
            book_keeper.initial_scaling_factor = initial_scaling_factor
            applied_scaling_factor = initial_scaling_factor
            logging.info("GSTAR Recon: Scaling factor: " + str(applied_scaling_factor))

        else:

            nominal_scaling_factor = DICOM_VAL_16BIT / max_val_images
            relative_scaling = nominal_scaling_factor / book_keeper.initial_scaling_factor
            applied_scaling_factor = relative_scaling * book_keeper.initial_scaling_factor
            book_keeper.relative_scaling_factor = relative_scaling
            logging.info("GSTAR Recon: Scaling factor: " + str(applied_scaling_factor) + " (x " + str(relative_scaling) + " from initial measurement)")

        book_keeper.last_scaling_factor = applied_scaling_factor

        return book_keeper
