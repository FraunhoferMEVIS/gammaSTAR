"""!
@brief Generalized client for gammaSTAR reconstruction server.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import os
import argparse
import logging
import time
import json
import ismrmrd
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_io_tools as io_tools
import mrpy_gammastar_tools as gstar_tools

default_args = {
    'address':        '127.0.0.1',
    'port':           9002
}

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='gammaSTAR Reconstructions Client',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filepath', help='Input file')
    parser.add_argument('-a', '--address', help='Address (hostname) of gammaSTAR Reconstructions server')
    parser.add_argument('-p', '--port', type=int, help='Port')
    parser.set_defaults(**default_args)
    args = parser.parse_args()

    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    list_of_acqs = []
    xml_header = ''
    if args.filepath.endswith('.dat'):

        logging.info("Loading Siemens TWIX data file")
        noise, noise_hdr, ksp_imaging, ksp_phasecor, ksp_rtfeedback, ksp_paracalib, meas_hdr = (
            io_tools.read_siemens_twix(args.filepath, unsorted = True, remove_os = False, ramp_samp_regrid = True))

        logging.info("Loading exported recon data file")
        recon_file = args.filepath[:-4] + ".json"
        try:
            with open(recon_file) as f:
                recon_data = json.load(f)
        except:
            raise FileNotFoundError(
                "Did not find gammaSTAR sequence data. Make sure to export the sequence data from the fronted which corresponds to the twix measurement and place it in the same directory as the .dat file with the same name but *.json ending.")

        logging.info("Converting raw data to acquisitions")
        adc_raw_reps = []
        try:
            for raw_rep in recon_data['raw_reps']:
                if raw_rep['has_adc']:
                    adc_raw_reps.append(raw_rep)
        except:
            adc_raw_reps = recon_data['raw_reps']

        raw_data_reorder = gstar_tools.reorder_twix_with_raw_reps(adc_raw_reps,
                                                                  ksp_phasecor,
                                                                  ksp_rtfeedback,
                                                                  ksp_paracalib,
                                                                  ksp_imaging)

        list_of_acqs = ismrmrd_tools.numpy_and_raw_rep_to_acq(raw_data_reorder,
                                                              adc_raw_reps,
                                                              recon_data['trajectory']['adc_points'],
                                                              False)
        list_of_noise_acqs = []
        if noise is not None:
            list_of_noise_acqs.append(ismrmrd_tools.noise_scan_to_acq(noise[:,:,0]))
        list_of_acqs =  list_of_noise_acqs + list_of_acqs

        logging.info("Creating ISMRMRD header")

        # Finally, we need to create the ismrmrd header object
        recon_data['protocol']['read_oversampling'] = recon_data['protocol']['read_oversampling'] + 1
        recon_data['protocol']['slice_oversampling'] = recon_data['protocol']['slice_oversampling'] + 1
        recon_data['protocol']['phase_oversampling'] = recon_data['protocol']['phase_oversampling'] + 1

        # We extract some measurement information from the twix data hdr, this will override some standard values
        meas_info = dict()
        meas_info["patientName"] = str(meas_hdr['Config']['tPatientName'])
        meas_info["patientWeight_kg"] = str(meas_hdr['Dicom']['flUsedPatientWeight'])
        meas_info["patientHeight_m"] = str(meas_hdr['Meas']['flPatientHeight']/100.0)
        meas_info["patientID"] = str(meas_hdr['Config']['PatientID'])
        meas_info["patientBirthdate"] = str(meas_hdr['Config']['PatientBirthDay'])
        meas_info["studyDate"] = str(meas_hdr['Meas']['PrepareTimestamp'][0:10])
        meas_info["studyTime"] = str(meas_hdr['Meas']['PrepareTimestamp'][11:])
        meas_info["studyDescription"] = str(meas_hdr['Dicom']['tStudyDescription'])
        meas_info["bodyPartExamined"] = str(meas_hdr['Dicom']['tBodyPartExamined'])
        meas_info["measurementID"] = str(meas_hdr['Meas']['Study'])
        meas_info["patientPosition"] = meas_hdr['Config']['PatientPosition']
        meas_info["protocolName"] = str(meas_hdr['Meas']['tProtocolName'])
        meas_info["frameOfReferenceUID"] = str(meas_hdr['Config']['FrameOfReference'])
        meas_info["seriesDescription"] = str(meas_hdr['Meas']['tProtocolName'])
        meas_info["systemVendor"] = str(meas_hdr['Dicom']['Manufacturer'])
        meas_info["systemModel"] = str(meas_hdr['Dicom']['ManufacturersModelName'])
        meas_info["systemFieldStrength_T"] = str(meas_hdr['MeasYaps'][('sProtConsistencyInfo', 'flNominalB0')])
        meas_info["institutionName"] = str(meas_hdr['Meas']['InstitutionName'])
        meas_info["H1resonanceFrequency_Hz"] = str(int(meas_hdr['Dicom']['lFrequency']))

        xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(recon_data['protocol'],
                                                        recon_data['info'],
                                                        recon_data['expo'],
                                                        recon_data['sys'],
                                                        recon_data['root'],
                                                        meas_info,
                                                        recon_data['sequence'])

    elif args.filepath.endswith('.h5'):

        dset = ismrmrd.Dataset(args.filepath, 'dataset', False)
        groups = dset.list()

        logging.info("Reading ISMRMRD header")
        xml_header = dset.read_xml_header()
        xml_header = xml_header.decode("utf-8")

        logging.info("Converting raw data to acquisitions")
        list_of_acqs = []
        for idx in range(dset.number_of_acquisitions()):
            list_of_acqs.append(dset.read_acquisition(idx))

    # Send ismrmrd acquisitions to server using ismrmrd_tools
    start = time.time()
    con = ismrmrd_tools.gstar_recon_emitter(args.address, args.port, list_of_acqs, xml_header)

    # Get reconstructed images
    ismrmrd_tools.gstar_recon_injector(con)
    end = time.time()

    logging.info("The Server took " + str(end-start) + "[s] to perform reconstruction tasks")
    con.shutdown_close()

    # ISMRMRD images to DICOM
    meta = ismrmrd.xsd.CreateFromDocument(xml_header)
    if not os.path.exists("./out/"):
        os.makedirs("./out/")
    folder_name = io_tools.get_folder_name("./out/" + meta.measurementInformation.protocolName)
    for i_im in range(0,len(con.images)):
        io_tools.write_dcm_from_ismrmrd_image(con.images[i_im], folder_name)
