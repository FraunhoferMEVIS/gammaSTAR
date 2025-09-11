"""!
@brief Example software which loads Siemens MR rawdata in twix format and in addition to gammaSTAR reconstruction
       data and stores it into an hdf5 file.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         **InsertLicense** code

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import argparse
import ismrmrd
import json
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_io_tools as io_tools
import mrpy_gammastar_tools as gstar_tools


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='gammaSTAR twix to hdf5 converter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help='Input file')
    args = parser.parse_args()

    # Load the twix raw data file using io_tools
    noise, noise_hdr, ksp_imaging, ksp_phasecor, ksp_rtfeedback, ksp_paracalib, meas_hdr = (
        io_tools.read_siemens_twix(args.filename, True, False, True))

    # Load the gammastar sequence data file
    sequence_data_filename = args.filename[:-4] + ".json"
    try:
        with open(sequence_data_filename) as f:
            seq_data = json.load(f)
    except:
        raise FileNotFoundError(
            "Did not find gammaSTAR sequence data. Make sure to export the sequence data from the fronted which corresponds to the twix measurement and place it in the same directory as the .dat file with the same name but *.json ending.")

    # Extract those raw representations which have the ADC events in it
    adc_raw_reps = []
    for raw_rep in seq_data['sequence']:
        if raw_rep['has_adc']:
            adc_raw_reps.append(raw_rep)

    # We create a single numpy array from the different twix sub arrays based on the raw rep readout structure
    raw_data_reorder = gstar_tools.reorder_twix_with_raw_reps(adc_raw_reps, ksp_phasecor, ksp_rtfeedback, ksp_paracalib, ksp_imaging)

    # Transfer loaded numpy k-space data into ismrmrd acquisitions using infos from the raw representations with trajectories
    list_of_acqs = ismrmrd_tools.numpy_and_raw_rep_to_acq(raw_data_reorder, adc_raw_reps, seq_data['trajectory']['adc_points'], False)
    list_of_noise_acqs = []
    if noise is not None:
        list_of_noise_acqs.append(ismrmrd_tools.noise_scan_to_acq(noise[:,:,0]))
    list_of_acqs =  list_of_noise_acqs + list_of_acqs

    # Finally, we need to create the ismrmrd header object
    seq_data['protocol']['read_oversampling'] = seq_data['protocol']['read_oversampling'] + 1
    seq_data['protocol']['slice_oversampling'] = seq_data['protocol']['slice_oversampling'] + 1
    seq_data['protocol']['phase_oversampling'] = seq_data['protocol']['phase_oversampling'] + 1

    # We extract some measurment information from the twix data hdr, this will override some standard values
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

    xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(seq_data['protocol'], seq_data['info'], seq_data['expo'], seq_data['sys'], seq_data['root'], meas_info)

    # Write everything to the ISMRMRD dataset
    with ismrmrd.Dataset(args.filename[:-4] + '.h5', 'dataset',
                         create_if_needed=True) as dset:
       dset.write_xml_header(xml_header)
       for i, acq in enumerate(list_of_acqs):
           dset.append_acquisition(acq)


