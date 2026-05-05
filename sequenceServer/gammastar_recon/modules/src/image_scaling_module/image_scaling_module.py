"""!
@brief Image scaling module of gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging
import numpy as np
import mrpy_ismrmrd_tools as ismrmrd_tools

DICOM_VAL_16BIT = 65535.0-1.0  # Maximum value for 16 bit DICOM images
DICOM_VAL_4BIT = 4095.0-1.0  # Maximum value for 4 bit DICOM images


class ImageScaleModule:
    """!
    @brief This module calculates the scaling factor from the reconstructed image series.
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
        return ImageScaleModule.apply(con_buff, book_keeper)

    @staticmethod
    def apply(
            con_buff: ismrmrd_tools.ConnectionBuffer,
            book_keeper: "BookKeeper"
    ) -> tuple[ismrmrd_tools.ConnectionBuffer, "BookKeeper"]:
        """!
        @brief Calculates the scaling factor from the reconstructed image series.

        @param con_buff: ConnectionBuffer object, holding processed "NP_..." data structures.
        @param book_keeper: Object which stores calibration data etc

        @return
            -  ConnectionBuffer object, holding processed "NP_..." data structures.
            -  Object which stores calibration data etc

        @author Jörn Huber
        """

        applied_scaling_factor = 1.0
        max_val_images = np.max(np.abs(con_buff.meas_data.data['NP_IS_IMAGING']).flatten())

        if max_val_images == 0.0:
            logging.warning("gs-recon: Maximum value of reconstructed images is 0.0")
            max_val_images = 1.0

        if book_keeper.initial_scaling_factor == 0.0:

            initial_scaling_factor = DICOM_VAL_16BIT / max_val_images
            book_keeper.initial_scaling_factor = initial_scaling_factor
            applied_scaling_factor = initial_scaling_factor
            logging.info("gs-recon: Scaling factor: " + str(applied_scaling_factor))

        else:

            nominal_scaling_factor = DICOM_VAL_16BIT / max_val_images
            relative_scaling = nominal_scaling_factor / book_keeper.initial_scaling_factor
            applied_scaling_factor = relative_scaling * book_keeper.initial_scaling_factor
            book_keeper.relative_scaling_factor = relative_scaling
            logging.info("gs-recon: Scaling factor: " + str(applied_scaling_factor) +
                         " (x " + str(relative_scaling) + " from initial measurement)")

        book_keeper.last_scaling_factor = applied_scaling_factor

        book_keeper.recon_history += "_ScaledImages"

        return book_keeper
