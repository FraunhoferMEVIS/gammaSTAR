"""!
@brief Book keeping module
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import logging


class BookKeeper:
    """!
    @brief This module implements a class which is used to keep track of everything which is not directly related to
           the purely reconstructed image data. This could be e.g. fitted T1 maps, T2 maps, scaling factors etc. The
           class implements some mandatory fields which are definitely needed and offers the option to assign arbitrary
           data in form of additional keys.
    """

    def __init__(self):

        # The subject id for which the tracked data is valid
        self.subject_id = ''

        # The scaling factor which was first applied to the data
        self.initial_scaling_factor = 0.0

        # The scaling factor which was last applied to the data
        self.last_scaling_factor = 0.0

        # The relative scaling factor between first and n-th measurement
        self.relative_scaling_factor = 0.0

        # The list which contains the buffer of outgoing images
        self.outgoing_image_buffer = []

        # Keeps track of the current image series index
        self.image_series_index = 0

        # A dictionary structure which allows to keep arbitrary data from previous measurements such as T1, T2 maps...
        self.data = dict()

    def register_patient(self, connection_buffer):

        cur_subject_id = ''
        try:
            cur_subject_id = connection_buffer.headers[0].subjectInformation.patientID
        except:
            logging.warning("Could not read subject ID from measurement header")
            cur_subject_id = 'Unknown'

        if cur_subject_id != self.subject_id:
            self.__init__()
            self.subject_id = cur_subject_id
            logging.info("GSTAR Recon: Registered new patient ID: " + self.subject_id)

        self.image_series_index = 0
        self.outgoing_image_buffer = []
