"""!
@brief Main file for gammaSTAR Reconstructions
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""
import sys
import os
import logging
import time
import mrpy_ismrmrd_tools as ismrmrd_tools
from book_keeper import BookKeeper

# Reconstruction Modules
from average_combination_module import AverageCombinationModule
from segment_combination_module import SegmentCombinationModule
from parallel_imaging_module import ParallelImagingModule
from partial_fourier_module import PartialFourierModule
from phase_correction_module import PhaseCorModule
from channel_combination_module import ChannelCombinationModule
from prop_module import PropellerModule
from ksp_to_image_module import KspaceToImageModule
from ifft_module import IFFTModule
from image_scaling_module import ImageScaleModule
from finalize_out_image_module import FinalizeOutImageModule

def print_info():

    print("########################################################\n\n"
          "gammaSTAR Reconstructions v1.0.3 Release\n")

    print("The software is not qualified for use as a medical product or as part\n"
          "thereof. No bugs or restrictions are known. Delivered ‘as is’ without\n"
          "specific verification or validation.\n"
          "(C) Fraunhofer MEVIS 2025\n"
          "Contact: Joern Huber (joern.huber@mevis.fraunhofer.de)\n")

    print("List of included third-party software\n"
          "-\n"
          "MRI Nufft (BSD-3-Clause License)\n"
          "Python ISMRMRD Server (MIT License)\n"   
          "pymapvbvd (MIT License)\n"
          "pydicom (MIT-based License)\n"
          "ismrmrd (ISMRMRD SOFTWARE LICENSE JULY 2013)\n"
          "sigpy (BSD-2-Clause License)\n"
          "xmltodict (MIT License)\n"
          "numpy (NumPy License)\n"
          "scikit-learn (MIT License)\n"
          "-\n"
          "Detailed information about third-party licence conditions can\n"
          "be found in the 'third_party_licenses' folder.\n\n"
          "########################################################\n")

if __name__ == "__main__":

    print_info()
    time.sleep(0.1)

    book_keeper = BookKeeper()

    # Reconstruction Modules
    average_combination_module = AverageCombinationModule()
    segment_combination_module = SegmentCombinationModule()
    parallel_imaging_module = ParallelImagingModule()
    partial_fourier_module = PartialFourierModule()
    phase_correction_module = PhaseCorModule()
    channel_combination_module = ChannelCombinationModule()
    prop_module = PropellerModule()
    ksp_to_image_module = KspaceToImageModule()
    ifft_module = IFFTModule()
    image_scaling_module = ImageScaleModule()
    finalize_out_image_module = FinalizeOutImageModule()

    while True:

        try:
            con_buffer, server_socket = ismrmrd_tools.gstar_recon_server(host='0.0.0.0', port=9002)
            book_keeper.register_patient(con_buffer)

            ########################
            # Image Reconstruction #
            ########################

            con_buffer = (
                ksp_to_image_module(
                    prop_module(
                        channel_combination_module(
                            phase_correction_module(
                                ifft_module('PE2',
                                            partial_fourier_module(
                                                parallel_imaging_module(
                                                    segment_combination_module(
                                                        average_combination_module(con_buffer))))))))))

            ##################################################
            #     Scaling and Appending to Output Buffer     #
            ##################################################

            book_keeper = image_scaling_module(con_buffer, book_keeper)
            book_keeper = finalize_out_image_module(con_buffer, book_keeper)

            logging.info("GSTAR Recon: Sending reconstructed image series")
            for i_out_image in range(0, len(book_keeper.outgoing_image_buffer)):
                con_buffer.send_image(book_keeper.outgoing_image_buffer[i_out_image])

            con_buffer.send_close()
            con_buffer.shutdown_close()
            server_socket.close()
            time.sleep(0.1)
            print("\n")

        except KeyboardInterrupt:
            logging.info("Keyboard interruption, now exiting")
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)