"""!
@brief MRD Client Software which reads MRD files and sends the data to a gstar recon server.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import argparse
import logging
import ismrmrd
import time
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_io_tools as io_tools

defaults = {
    'address':        'localhost',
    'port':           9002, 
    'config':         'standard',
}

def main(args):

    dset = ismrmrd.Dataset(args.filename, 'dataset', False)
    groups = dset.list()

    # Extract the ismrmrd header object
    xml_header = ''
    if 'xml' in groups:
        xml_header = dset.read_xml_header()
        xml_header = xml_header.decode("utf-8")
    else:
        logging.warning("Could not find MRD metadata xml in file")
        return

    has_raw = False
    if 'data' in groups:
        has_raw = True

    if has_raw is False:
        logging.error("File does not contain properly formatted MRD raw or image data")
        return

    # Create the list of acquisition objects from hdf5
    list_of_acqs = []
    if has_raw:
        logging.info("Found %d raw data readouts", dset.number_of_acquisitions())

        for idx in range(dset.number_of_acquisitions()):
            list_of_acqs.append(dset.read_acquisition(idx))

    # We check whether no config was specified so we could try to load the config from the gui
    if args.config == "standard":
        try:
            f = open("gstar_config.txt", "r")
            args.config = f.read()
            f.close()
        except:
            print("No GUI configuration available")

    # Send ismrmrd acquisitions to server using ismrmrd_tools
    start = time.time()
    con = ismrmrd_tools.gstar_recon_emitter(args.address, args.port, list_of_acqs, xml_header, 1.0,
                                            args.config)

    # Get reconstructed images
    ismrmrd_tools.gstar_recon_injector(con)
    end = time.time()
    con.shutdown_close()

    logging.info("Socket closed (writer), Server took " + str(end-start) + " seconds for reconstruction")

    # Save a copy of the MRD XML header now that the connection thread is finished with the file
    logging.debug("Writing MRD metadata to DICOM")
    # ISMRMRD images to DICOM
    for recv_image in con.images:
        io_tools.write_dcm_from_ismrmrd_image(recv_image)

    logging.info("---------------------- Summary ----------------------")
    logging.info("Sent %5d acquisitions  |  Received %5d acquisitions", len(list_of_acqs),      0)
    logging.info("Sent %5d images        |  Received %5d images",       0,    len(con.images))
    logging.info("Sent %5d waveforms     |  Received %5d waveforms",    0, 0)
    logging.info("Session complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example client for MRD streaming format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                    help='Input file')
    parser.add_argument('-a', '--address',                             help='Address (hostname) of MRD server')
    parser.add_argument('-p', '--port',           type=int,            help='Port')
    parser.add_argument('-c', '--config',                              help='Remote configuration file')
    parser.set_defaults(**defaults)

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
    logging.root.setLevel(logging.INFO)

    main(args)
