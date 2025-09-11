"""!
@brief Collection of tools which are used for input/output tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import os
import mapvbvd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset, FileDataset
from pydicom.sequence import Sequence
import numpy as np
from datetime import datetime

def load_from_bart_cfl(filename):
    """!
    @brief Load a BART .cfl/.hdr file pair into a numpy array.

    @param filename: (string) Base name of the .cfl and .hdr files (without extension)

    @return
        - (np.ndarray) Complex loaded data

    @author Jörn Huber
    """
    # Read the .hdr file
    hdr_file = f"{filename}.hdr"
    with open(hdr_file, 'r') as f:
        lines = f.readlines()
        if not lines[0].strip() == '# Dimensions':
            raise ValueError(f"Invalid .hdr file format: {hdr_file}")
        dims = list(map(int, lines[1].strip().split()))[::-1]

    # Read the .cfl file
    cfl_file = f"{filename}.cfl"
    with open(cfl_file, 'rb') as f:
        num_elements = np.prod(dims)
        data = np.fromfile(f, dtype=np.float32, count=2 * num_elements)
        data = data[0::2] + 1j * data[1::2]
        array = data.reshape(dims)
    return array


def save_to_bart_cfl(filename, array):
    """!
    @brief Save a numpy array to BART's .cfl/.hdr format.

    @param filename: (string) Base name for the .cfl and .hdr files (without extension)
    @param array: (np.ndarray) NumPy array to save

    @author Jörn Huber
    """
    if not np.iscomplexobj(array):
        raise ValueError("Input array must be complex (real + imaginary parts).")

    # Write the .hdr file
    hdr_file = f"{filename}.hdr"
    with open(hdr_file, 'w') as f:
        # Write the array's dimensions in reverse order (BART expects Fortran order)
        f.write('# Dimensions\n')
        f.write(' '.join(map(str, array.shape[::-1])) + '\n')

    # Write the .cfl file
    cfl_file = f"{filename}.cfl"
    with open(cfl_file, 'wb') as f:
        # Interleave real and imaginary parts and write as binary
        array.astype(np.complex64).ravel().tofile(f)


def read_siemens_twix(file_name, unsorted = False, remove_os = True, ramp_samp_regrid = True):
    """!
    @brief Reads twix rawdata as exported from the MR system into numpy files

    @param file_name: (string) File path + name of twix rawdata file

    @return
        - (np.ndarray) Numpy object with noise calibration data
        - (twix_hdr) Header data of noise scan
        - (np.ndarray) Numpy object with image data
        - (np.ndarray) Numpy object with phase correction data
        - (np.ndarray) Numpy object with real time feedback data
        - (np.ndarray) Numpy object with parallel calibration data
        - (twix_hdr) Header data of image scan

    @author Jörn Huber
    """
    noise = None
    ksp = None
    ksp_phasecor = None
    ksp_rtfeedback = None
    ksp_paracalib = None
    noise_hdr = None
    meas_hdr = None

    twix_obj = mapvbvd.mapVBVD(file_name)
    if isinstance(twix_obj, mapvbvd._attrdict.AttrDict):
        print("Single Raid File")
        twix_obj.image.flagRemoveOS = remove_os
        twix_obj.image.flagRampSampRegrid = ramp_samp_regrid
        twix_obj.image.flagDisableReflect = True
        if unsorted:
            ksp = twix_obj.image.unsorted()
        else:
            ksp = twix_obj.image['']

        if hasattr(twix_obj, 'phasecor'):
            print('Reading Phasecor Data')
            twix_obj.phasecor.flagRemoveOS = remove_os
            twix_obj.phasecor.flagRampSampRegrid = ramp_samp_regrid
            twix_obj.phasecor.flagDisableReflect = True
            if unsorted:
                ksp_phasecor = twix_obj.phasecor.unsorted()
            else:
                ksp_phasecor = twix_obj.phasecor['']

        if hasattr(twix_obj, 'refscan'):
            print('Reading Calibration Data')
            twix_obj.refscan.flagRemoveOS = remove_os
            twix_obj.refscan.flagRampSampRegrid = ramp_samp_regrid
            twix_obj.refscan.flagDisableReflect = True
            if unsorted:
                ksp_paracalib = twix_obj.refscan.unsorted()
            else:
                ksp_paracalib = twix_obj.refscan['']
                ksp_paracalib = ksp_paracalib[:, :, :, :, :, :, :, :, :, :, :, 0, 0, 0, 0, 0]

        if hasattr(twix_obj, 'rtfeedback'):
            print('Reading RTFeedback Data')
            twix_obj.rtfeedback.flagRemoveOS = remove_os
            twix_obj.rtfeedback.flagRampSampRegrid = ramp_samp_regrid
            twix_obj.rtfeedback.flagDisableReflect = True
            if unsorted:
                ksp_rtfeedback = twix_obj.rtfeedback.unsorted()
            else:
                ksp_rtfeedback = twix_obj.rtfeedback['']
        meas_hdr = twix_obj.hdr

    elif isinstance(twix_obj, list):
        print("Multi Raid File")
        twix_obj[0].noise.flagRemoveOS = remove_os
        #twix_obj[0].noise.flagRampSampRegrid = ramp_samp_regrid
        if unsorted:
            noise = twix_obj[0].noise.unsorted()
        else:
            noise = twix_obj[0].noise['']
            noise = noise[:, :, :, :, :, :, :, :, :, :, :, 0, 0, 0, 0, 0]
        noise_hdr = twix_obj[0].hdr

        print('Reading Image Data')
        twix_obj[1].image.flagRemoveOS = remove_os
        #twix_obj[1].image.flagRampSampRegrid = ramp_samp_regrid
        twix_obj[1].image.flagDisableReflect = True
        if unsorted:
            ksp = twix_obj[1].image.unsorted()
        else:
            ksp = twix_obj[1].image['']

        if hasattr(twix_obj[1], 'phasecor'):
            print('Reading Phasecor Data')
            twix_obj[1].phasecor.flagRemoveOS = remove_os
            #twix_obj[1].phasecor.flagRampSampRegrid = ramp_samp_regrid
            twix_obj[1].phasecor.flagDisableReflect = True
            if unsorted:
                ksp_phasecor = twix_obj[1].phasecor.unsorted()
            else:
                ksp_phasecor = twix_obj[1].phasecor['']
                ksp_phasecor = ksp_phasecor[:, :, :, :, :, :, :, :, :, :, :, 0, 0, 0, 0, 0]

        if hasattr(twix_obj[1], 'refscan'):
            print('Reading Calibration Data')
            twix_obj[1].refscan.flagRemoveOS = remove_os
            #twix_obj[1].refscan.flagRampSampRegrid = ramp_samp_regrid
            twix_obj[1].refscan.flagDisableReflect = True
            if unsorted:
                ksp_paracalib = twix_obj[1].refscan.unsorted()
            else:
                ksp_paracalib = twix_obj[1].refscan['']
                ksp_paracalib = ksp_paracalib[:, :, :, :, :, :, :, :, :, :, :, 0, 0, 0, 0, 0]

        if hasattr(twix_obj[1], 'rtfeedback'):
            print('Reading RTFeedback Data')
            twix_obj[1].rtfeedback.flagRemoveOS = remove_os
            #twix_obj[1].rtfeedback.flagRampSampRegrid = ramp_samp_regrid
            twix_obj[1].rtfeedback.flagDisableReflect = True
            if unsorted:
                ksp_rtfeedback = twix_obj[1].rtfeedback.unsorted()
            else:
                ksp_rtfeedback = twix_obj[1].rtfeedback['']
                ksp_rtfeedback = ksp_rtfeedback[:, :, :, :, :, :, :, :, :, :, :, 0, 0, 0, 0, 0]
        meas_hdr = twix_obj[1].hdr

    # We remove the user specific flag counters as they are not represented in the ISMRMRD format
    if not unsorted:
        ksp = ksp[:, :, :, :, :, :, :, :, :, :, :, 0, 0, 0, 0, 0]

        ksp = np.transpose(ksp, [0, 1, 2, 3, 4, 9, 6, 7, 8, 5, 10])
        if ksp_phasecor is not None:
            ksp_phasecor = np.transpose(ksp_phasecor, [0, 1, 2, 3, 4, 9, 6, 7, 8, 5, 10])
        if ksp_paracalib is not None:
            ksp_paracalib = np.transpose(ksp_paracalib, [0, 1, 2, 3, 4, 9, 6, 7, 8, 5, 10])
        if ksp_rtfeedback is not None:
            ksp_rtfeedback = np.transpose(ksp_rtfeedback, [0, 1, 2, 3, 4, 9, 6, 7, 8, 5, 10])

    return noise, noise_hdr, ksp, ksp_phasecor, ksp_rtfeedback, ksp_paracalib, meas_hdr


def write_dcm(filename,
              image_slices,
              study_description='Test Study',
              series_description='Test Series',
              mr_system_name='SYNGO_MR_XA50A',
              mr_manufacturer_name='Siemens Healthineers',
              body_part='BRAIN',
              protocol_name='Test Protocol',
              sequence_name='Test',
              acq_type='3D'):
    """!
    @brief Reads twix rawdata as exported from the MR system into numpy files

    @param filename: (string) File path + name of target dcm file
    @param image_slices: (np.ndarray) Numpy image data (num_slc, num_lin, num_col)
    @param study_description: (string) User defined string
    @param series_description: (string) User defined string
    @param mr_system_name: (string) User defined string
    @param mr_manufacturer_name: (string) User defined string
    @param body_part: (string) User defined string
    @param protocol_name: (string) User defined string
    @param sequence_name: (string) User defined string
    @param acq_type: (string) User defined string (2D or 3D)

    @author Jörn Huber
    """

    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 192
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    file_meta.MediaStorageSOPInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000121'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationClassUID = '1.3.12.2.1107.5.2'
    file_meta.ImplementationVersionName = mr_system_name

    # Main data elements
    ds = Dataset()
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    ds.InstanceCreationDate = '20240508'
    ds.InstanceCreationTime = '140642.045000'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    ds.SOPInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000121'
    ds.StudyDate = '20240508'
    ds.SeriesDate = '20240508'
    ds.ContentDate = '20240508'
    ds.AcquisitionDateTime = '20240508140642.045000'
    ds.StudyTime = '125515.000000'
    ds.SeriesTime = '140828.511000'
    ds.ContentTime = '140828.555000'
    ds.AccessionNumber = ''
    ds.Modality = 'MR'
    ds.Manufacturer = mr_manufacturer_name
    ds.InstitutionName = 'MEVIS'
    ds.InstitutionAddress = 'Max von Laue Str. 2, 28359, Bremen'
    ds.ReferringPhysicianName = ''
    ds.study_description = study_description
    ds.series_description = series_description
    ds.AdmittingDiagnosesDescription = ''
    ds.ManufacturerModelName = mr_system_name

    # Referenced Performed Procedure Step Sequence
    refd_performed_procedure_step_sequence = Sequence()
    ds.ReferencedPerformedProcedureStepSequence = refd_performed_procedure_step_sequence

    # Referenced Performed Procedure Step Sequence: Referenced Performed Procedure Step 1
    refd_performed_procedure_step1 = Dataset()
    refd_performed_procedure_step_sequence.append(refd_performed_procedure_step1)
    refd_performed_procedure_step1.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.3'
    refd_performed_procedure_step1.ReferencedSOPInstanceUID = \
        '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000122'

    ds.DerivationDescription = "Forced 'Reduced' Anonymity - Service"

    # Referenced Image Evidence Sequence
    refd_image_evidence_sequence = Sequence()
    ds.ReferencedImageEvidenceSequence = refd_image_evidence_sequence

    # Referenced Image Evidence Sequence: Referenced Image Evidence 1
    refd_image_evidence1 = Dataset()
    refd_image_evidence_sequence.append(refd_image_evidence1)

    # Referenced Series Sequence
    refd_series_sequence = Sequence()
    refd_image_evidence1.ReferencedSeriesSequence = refd_series_sequence

    # Referenced Series Sequence: Referenced Series 1
    refd_series1 = Dataset()
    refd_series_sequence.append(refd_series1)

    # Referenced SOP Sequence
    refd_sop_sequence = Sequence()
    refd_series1.ReferencedSOPSequence = refd_sop_sequence

    # Referenced SOP Sequence: Referenced SOP 1
    refd_sop1 = Dataset()
    refd_sop_sequence.append(refd_sop1)
    refd_sop1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    refd_sop1.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000123'

    # Purpose of Reference Code Sequence
    purpose_of_ref_code_sequence = Sequence()
    refd_sop1.PurposeOfReferenceCodeSequence = purpose_of_ref_code_sequence

    # Purpose of Reference Code Sequence: Purpose of Reference Code 1
    purpose_of_ref_code1 = Dataset()
    purpose_of_ref_code_sequence.append(purpose_of_ref_code1)
    purpose_of_ref_code1.CodeValue = '121311'
    purpose_of_ref_code1.CodingSchemeDesignator = 'DCM'
    purpose_of_ref_code1.CodeMeaning = 'Localizer'

    # Referenced SOP Sequence: Referenced SOP 2
    refd_sop2 = Dataset()
    refd_sop_sequence.append(refd_sop2)
    refd_sop2.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    refd_sop2.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000124'

    # Purpose of Reference Code Sequence
    purpose_of_ref_code_sequence = Sequence()
    refd_sop2.PurposeOfReferenceCodeSequence = purpose_of_ref_code_sequence

    # Purpose of Reference Code Sequence: Purpose of Reference Code 1
    purpose_of_ref_code1 = Dataset()
    purpose_of_ref_code_sequence.append(purpose_of_ref_code1)
    purpose_of_ref_code1.CodeValue = '121311'
    purpose_of_ref_code1.CodingSchemeDesignator = 'DCM'
    purpose_of_ref_code1.CodeMeaning = 'Localizer'

    refd_series1.SeriesInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000125'

    refd_image_evidence1.StudyInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000126'

    ds.PixelPresentation = 'MONOCHROME'
    ds.VolumetricProperties = 'DISTORTED'
    ds.VolumeBasedCalculationTechnique = 'NONE'
    ds.ComplexImageComponent = 'MAGNITUDE'
    ds.AcquisitionContrast = 'UNKNOWN'
    ds.PatientName = 'MR202046'
    ds.PatientID = 'MR202046'
    ds.IssuerOfPatientID = ''
    ds.PatientBirthDate = '20240508'
    ds.PatientSex = 'M'
    ds.PatientAge = '033Y'
    ds.PatientSize = '1.81'
    ds.PatientWeight = '80.0'
    ds.MedicalAlerts = ''
    ds.Allergies = ''
    ds.PatientIdentityRemoved = 'YES'
    ds.DeidentificationMethod = ''
    ds.BodyPartExamined = body_part
    ds.MRAcquisitionType = acq_type
    ds.MagneticFieldStrength = '3.0'
    ds.DeviceSerialNumber = '202046'
    ds.SoftwareVersions = ''
    ds.ProtocolName = protocol_name
    ds.B1rms = 0.0
    ds.PatientPosition = 'HFS'
    ds.ContentQualification = 'RESEARCH'
    ds.PulseSequenceName = sequence_name
    ds.EchoPulseSequence = 'SPIN'
    ds.MultipleSpinEcho = 'NO'
    ds.MultiPlanarExcitation = 'NO'
    ds.PhaseContrast = 'NO'
    ds.TimeOfFlightContrast = 'NO'
    ds.SteadyStatePulseSequence = 'NONE'
    ds.EchoPlanarPulseSequence = 'NO'
    ds.SaturationRecovery = 'NO'
    ds.SpectrallySelectedSuppression = 'NONE'
    ds.OversamplingPhase = 'NONE'
    ds.GeometryOfKSpaceTraversal = 'RECTILINEAR'
    ds.SegmentedKSpaceTraversal = 'SINGLE'
    ds.RectilinearPhaseEncodeReordering = 'LINEAR'
    ds.KSpaceFiltering = 'NONE'
    ds.AcquisitionDuration = 0.507
    ds.NumberOfKSpaceTrajectories = 1
    ds.CoverageOfKSpace = 'FULL'
    ds.ResonantNucleus = '1H'
    ds.ApplicableSafetyStandardAgency = 'IEC'
    ds.StudyInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000126'
    ds.SeriesInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000127'
    ds.StudyID = '915ce8db-15a3-4f'
    ds.SeriesNumber = '9'
    ds.AcquisitionNumber = '1'
    ds.InstanceNumber = '1'
    ds.FrameOfReferenceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000128'
    ds.NumberOfTemporalPositions = '8'
    ds.PositionReferenceIndicator = ''

    # Dimension Organization Sequence
    dimension_organization_sequence = Sequence()
    ds.DimensionOrganizationSequence = dimension_organization_sequence

    # Dimension Organization Sequence: Dimension Organization 1
    dimension_organization1 = Dataset()
    dimension_organization_sequence.append(dimension_organization1)
    dimension_organization1.DimensionOrganizationUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000129'

    # Dimension Index Sequence
    dimension_index_sequence = Sequence()
    ds.DimensionIndexSequence = dimension_index_sequence

    # Dimension Index Sequence: Dimension Index 1
    dimension_index1 = Dataset()
    dimension_index_sequence.append(dimension_index1)
    dimension_index1.DimensionOrganizationUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000129'
    dimension_index1.DimensionIndexPointer = (0x0020, 0x9056)
    dimension_index1.FunctionalGroupPointer = (0x0020, 0x9111)

    # Dimension Index Sequence: Dimension Index 2
    dimension_index2 = Dataset()
    dimension_index_sequence.append(dimension_index2)
    dimension_index2.DimensionOrganizationUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000129'
    dimension_index2.DimensionIndexPointer = (0x0020, 0x9057)
    dimension_index2.FunctionalGroupPointer = (0x0020, 0x9111)

    # Dimension Index Sequence: Dimension Index 3
    dimension_index3 = Dataset()
    dimension_index_sequence.append(dimension_index3)
    dimension_index3.DimensionOrganizationUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000129'
    dimension_index3.DimensionIndexPointer = (0x0020, 0x9128)
    dimension_index3.FunctionalGroupPointer = (0x0020, 0x9111)

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.NumberOfFrames = str(image_slices.shape[0])
    ds.Rows = image_slices.shape[2]
    ds.Columns = image_slices.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = 0
    ds.LargestImagePixelValue = 332
    ds.BurnedInAnnotation = 'NO'
    ds.LossyImageCompression = '00'
    ds.PerformedProcedureStepStartDate = '20240508'
    ds.PerformedProcedureStepStartTime = '130134.000000'
    ds.PerformedProcedureStepEndDate = '20240508'
    ds.PerformedProcedureStepID = 'SIb6d5c89433f04a'
    ds.PerformedProcedureStepDescription = ''

    # Acquisition Context Sequence
    acquisition_context_sequence = Sequence()
    ds.AcquisitionContextSequence = acquisition_context_sequence

    ds.IssueDateOfImagingServiceRequest = '20240508'
    ds.PresentationLUTShape = 'IDENTITY'

    # Shared Functional Groups Sequence
    shared_functional_groups_sequence = Sequence()
    ds.SharedFunctionalGroupsSequence = shared_functional_groups_sequence

    # Shared Functional Groups Sequence: Shared Functional Groups 1
    shared_functional_groups1 = Dataset()
    shared_functional_groups_sequence.append(shared_functional_groups1)

    # Referenced Image Sequence
    refd_image_sequence = Sequence()
    shared_functional_groups1.ReferencedImageSequence = refd_image_sequence

    # Referenced Image Sequence: Referenced Image 1
    refd_image1 = Dataset()
    refd_image_sequence.append(refd_image1)
    refd_image1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    refd_image1.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000123'
    refd_image1.ReferencedFrameNumber = [4, 9]

    # Purpose of Reference Code Sequence
    purpose_of_ref_code_sequence = Sequence()
    refd_image1.PurposeOfReferenceCodeSequence = purpose_of_ref_code_sequence

    # Purpose of Reference Code Sequence: Purpose of Reference Code 1
    purpose_of_ref_code1 = Dataset()
    purpose_of_ref_code_sequence.append(purpose_of_ref_code1)
    purpose_of_ref_code1.CodeValue = '121311'
    purpose_of_ref_code1.CodingSchemeDesignator = 'DCM'
    purpose_of_ref_code1.CodeMeaning = 'Localizer'

    # Referenced Image Sequence: Referenced Image 2
    refd_image2 = Dataset()
    refd_image_sequence.append(refd_image2)
    refd_image2.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    refd_image2.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.54.202046.30000024050814204641900000124'
    refd_image2.ReferencedFrameNumber = '3'

    # Purpose of Reference Code Sequence
    purpose_of_ref_code_sequence = Sequence()
    refd_image2.PurposeOfReferenceCodeSequence = purpose_of_ref_code_sequence

    # Purpose of Reference Code Sequence: Purpose of Reference Code 1
    purpose_of_ref_code1 = Dataset()
    purpose_of_ref_code_sequence.append(purpose_of_ref_code1)
    purpose_of_ref_code1.CodeValue = '121311'
    purpose_of_ref_code1.CodingSchemeDesignator = 'DCM'
    purpose_of_ref_code1.CodeMeaning = 'Localizer'

    # MR Imaging Modifier Sequence
    mr_imaging_modifier_sequence = Sequence()
    shared_functional_groups1.MRImagingModifierSequence = mr_imaging_modifier_sequence

    # MR Imaging Modifier Sequence: MR Imaging Modifier 1
    mr_imaging_modifier1 = Dataset()
    mr_imaging_modifier_sequence.append(mr_imaging_modifier1)
    mr_imaging_modifier1.PixelBandwidth = '2000.0'
    mr_imaging_modifier1.MagnetizationTransfer = 'NONE'
    mr_imaging_modifier1.BloodSignalNulling = 'NO'
    mr_imaging_modifier1.Tagging = 'NONE'
    mr_imaging_modifier1.TransmitterFrequency = 123.195023

    # MR Receive Coil Sequence
    mr_receive_coil_sequence = Sequence()
    shared_functional_groups1.MRReceiveCoilSequence = mr_receive_coil_sequence

    # MR Receive Coil Sequence: MR Receive Coil 1
    mr_receive_coil1 = Dataset()
    mr_receive_coil_sequence.append(mr_receive_coil1)
    mr_receive_coil1.ReceiveCoilName = 'Spine_32_RS'
    mr_receive_coil1.ReceiveCoilManufacturerName = 'NN'
    mr_receive_coil1.ReceiveCoilType = 'MULTICOIL'
    mr_receive_coil1.QuadratureReceiveCoil = 'NO'

    # Multi-Coil Definition Sequence
    multi_coil_definition_sequence = Sequence()
    mr_receive_coil1.MultiCoilDefinitionSequence = multi_coil_definition_sequence

    # Multi-Coil Definition Sequence: Multi-Coil Definition 1
    multi_coil_definition1 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition1)
    multi_coil_definition1.MultiCoilElementName = 'S31'
    multi_coil_definition1.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 2
    multi_coil_definition2 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition2)
    multi_coil_definition2.MultiCoilElementName = 'S32'
    multi_coil_definition2.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 3
    multi_coil_definition3 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition3)
    multi_coil_definition3.MultiCoilElementName = 'S33'
    multi_coil_definition3.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 4
    multi_coil_definition4 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition4)
    multi_coil_definition4.MultiCoilElementName = 'S34'
    multi_coil_definition4.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 5
    multi_coil_definition5 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition5)
    multi_coil_definition5.MultiCoilElementName = 'B21'
    multi_coil_definition5.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 6
    multi_coil_definition6 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition6)
    multi_coil_definition6.MultiCoilElementName = 'B22'
    multi_coil_definition6.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 7
    multi_coil_definition7 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition7)
    multi_coil_definition7.MultiCoilElementName = 'B23'
    multi_coil_definition7.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 8
    multi_coil_definition8 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition8)
    multi_coil_definition8.MultiCoilElementName = 'B24'
    multi_coil_definition8.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 9
    multi_coil_definition9 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition9)
    multi_coil_definition9.MultiCoilElementName = 'B25'
    multi_coil_definition9.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 10
    multi_coil_definition10 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition10)
    multi_coil_definition10.MultiCoilElementName = 'B26'
    multi_coil_definition10.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 11
    multi_coil_definition11 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition11)
    multi_coil_definition11.MultiCoilElementName = 'S21'
    multi_coil_definition11.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 12
    multi_coil_definition12 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition12)
    multi_coil_definition12.MultiCoilElementName = 'S22'
    multi_coil_definition12.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 13
    multi_coil_definition13 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition13)
    multi_coil_definition13.MultiCoilElementName = 'S23'
    multi_coil_definition13.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 14
    multi_coil_definition14 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition14)
    multi_coil_definition14.MultiCoilElementName = 'S24'
    multi_coil_definition14.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 15
    multi_coil_definition15 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition15)
    multi_coil_definition15.MultiCoilElementName = 'B11'
    multi_coil_definition15.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 16
    multi_coil_definition16 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition16)
    multi_coil_definition16.MultiCoilElementName = 'B12'
    multi_coil_definition16.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 17
    multi_coil_definition17 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition17)
    multi_coil_definition17.MultiCoilElementName = 'B13'
    multi_coil_definition17.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 18
    multi_coil_definition18 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition18)
    multi_coil_definition18.MultiCoilElementName = 'B14'
    multi_coil_definition18.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 19
    multi_coil_definition19 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition19)
    multi_coil_definition19.MultiCoilElementName = 'B15'
    multi_coil_definition19.MultiCoilElementUsed = 'YES'

    # Multi-Coil Definition Sequence: Multi-Coil Definition 20
    multi_coil_definition20 = Dataset()
    multi_coil_definition_sequence.append(multi_coil_definition20)
    multi_coil_definition20.MultiCoilElementName = 'B16'
    multi_coil_definition20.MultiCoilElementUsed = 'YES'

    # MR Transmit Coil Sequence
    mr_transmit_coil_sequence = Sequence()
    shared_functional_groups1.MRTransmitCoilSequence = mr_transmit_coil_sequence

    # MR Transmit Coil Sequence: MR Transmit Coil 1
    mr_transmit_coil1 = Dataset()
    mr_transmit_coil_sequence.append(mr_transmit_coil1)
    mr_transmit_coil1.TransmitCoilName = 'Body'
    mr_transmit_coil1.TransmitCoilManufacturerName = 'Siemens'
    mr_transmit_coil1.TransmitCoilType = 'BODY'

    # MR Spatial Saturation Sequence
    mr_spatial_saturation_sequence = Sequence()
    shared_functional_groups1.MRSpatialSaturationSequence = mr_spatial_saturation_sequence

    # MR Timing and Related Parameters Sequence
    mr_timing_and_related_parameters_sequence = Sequence()
    shared_functional_groups1.MRTimingAndRelatedParametersSequence = mr_timing_and_related_parameters_sequence

    # MR Timing and Related Parameters Sequence: MR Timing and Related Parameters 1
    mr_timing_and_related_parameters1 = Dataset()
    mr_timing_and_related_parameters_sequence.append(mr_timing_and_related_parameters1)
    mr_timing_and_related_parameters1.RepetitionTime = '15000.0'
    mr_timing_and_related_parameters1.EchoTrainLength = '0'
    mr_timing_and_related_parameters1.FlipAngle = '15.0'

    # Operating Mode Sequence
    operating_mode_sequence = Sequence()
    mr_timing_and_related_parameters1.OperatingModeSequence = operating_mode_sequence

    # Operating Mode Sequence: Operating Mode 1
    operating_mode1 = Dataset()
    operating_mode_sequence.append(operating_mode1)
    operating_mode1.OperatingModeType = 'GRADIENT'
    operating_mode1.OperatingMode = 'IEC_NORMAL'

    # Operating Mode Sequence: Operating Mode 2
    operating_mode2 = Dataset()
    operating_mode_sequence.append(operating_mode2)
    operating_mode2.OperatingModeType = 'RF'
    operating_mode2.OperatingMode = 'IEC_NORMAL'

    mr_timing_and_related_parameters1.GradientOutputType = 'PER_NERVE_STIM'
    mr_timing_and_related_parameters1.GradientOutput = 0.7093541026115417

    # Specific Absorption Rate Sequence
    specific_absorption_rate_sequence = Sequence()
    mr_timing_and_related_parameters1.SpecificAbsorptionRateSequence = specific_absorption_rate_sequence

    # Specific Absorption Rate Sequence: Specific Absorption Rate 1
    specific_absorption_rate1 = Dataset()
    specific_absorption_rate_sequence.append(specific_absorption_rate1)
    specific_absorption_rate1.SpecificAbsorptionRateDefinition = 'IEC_WHOLE_BODY'
    specific_absorption_rate1.SpecificAbsorptionRateValue = 0.0

    # Specific Absorption Rate Sequence: Specific Absorption Rate 2
    specific_absorption_rate2 = Dataset()
    specific_absorption_rate_sequence.append(specific_absorption_rate2)
    specific_absorption_rate2.SpecificAbsorptionRateDefinition = 'IEC_PARTIAL_BODY'
    specific_absorption_rate2.SpecificAbsorptionRateValue = 0.0

    # Specific Absorption Rate Sequence: Specific Absorption Rate 3
    specific_absorption_rate3 = Dataset()
    specific_absorption_rate_sequence.append(specific_absorption_rate3)
    specific_absorption_rate3.SpecificAbsorptionRateDefinition = 'IEC_HEAD'
    specific_absorption_rate3.SpecificAbsorptionRateValue = 0.0

    # Specific Absorption Rate Sequence: Specific Absorption Rate 4
    specific_absorption_rate4 = Dataset()
    specific_absorption_rate_sequence.append(specific_absorption_rate4)
    specific_absorption_rate4.SpecificAbsorptionRateDefinition = 'IEC_LOCAL'
    specific_absorption_rate4.SpecificAbsorptionRateValue = 0.0

    # Specific Absorption Rate Sequence: Specific Absorption Rate 5
    specific_absorption_rate5 = Dataset()
    specific_absorption_rate_sequence.append(specific_absorption_rate5)
    specific_absorption_rate5.SpecificAbsorptionRateDefinition = 'SMR_B1RMS'
    specific_absorption_rate5.SpecificAbsorptionRateValue = 0.0

    # Specific Absorption Rate Sequence: Specific Absorption Rate 6
    specific_absorption_rate6 = Dataset()
    specific_absorption_rate_sequence.append(specific_absorption_rate6)
    specific_absorption_rate6.SpecificAbsorptionRateDefinition = 'SMR_BORELOCAL'
    specific_absorption_rate6.SpecificAbsorptionRateValue = 0.0

    mr_timing_and_related_parameters1.RFEchoTrainLength = 0
    mr_timing_and_related_parameters1.GradientEchoTrainLength = 0

    # MR Modifier Sequence
    mr_modifier_sequence = Sequence()
    shared_functional_groups1.MRModifierSequence = mr_modifier_sequence

    # MR Modifier Sequence: MR Modifier 1
    mr_modifier1 = Dataset()
    mr_modifier_sequence.append(mr_modifier1)
    mr_modifier1.InversionRecovery = 'NO'
    mr_modifier1.FlowCompensation = 'NONE'
    mr_modifier1.T2Preparation = 'NO'
    mr_modifier1.SpectrallySelectedExcitation = 'NONE'
    mr_modifier1.SpatialPresaturation = 'NONE'
    mr_modifier1.PartialFourierDirection = 'SLICE_SELECT'
    mr_modifier1.ParallelAcquisition = 'NO'
    mr_modifier1.PartialFourier = 'YES'

    # MR FOV/Geometry Sequence
    mrfov_geometry_sequence = Sequence()
    shared_functional_groups1.MRFOVGeometrySequence = mrfov_geometry_sequence

    # MR FOV/Geometry Sequence: MR FOV/Geometry 1
    mrfov_geometry1 = Dataset()
    mrfov_geometry_sequence.append(mrfov_geometry1)
    mrfov_geometry1.PercentSampling = '100.0'
    mrfov_geometry1.PercentPhaseFieldOfView = '100.0'
    mrfov_geometry1.InPlanePhaseEncodingDirection = 'COLUMN'
    mrfov_geometry1.MRAcquisitionFrequencyEncodingSteps = 64
    mrfov_geometry1.MRAcquisitionPhaseEncodingStepsInPlane = 64
    mrfov_geometry1.MRAcquisitionPhaseEncodingStepsOutOfPlane = 12

    # Frame Anatomy Sequence
    frame_anatomy_sequence = Sequence()
    shared_functional_groups1.FrameAnatomySequence = frame_anatomy_sequence

    # Frame Anatomy Sequence: Frame Anatomy 1
    frame_anatomy1 = Dataset()
    frame_anatomy_sequence.append(frame_anatomy1)

    # Anatomic Region Sequence
    anatomic_region_sequence = Sequence()
    frame_anatomy1.AnatomicRegionSequence = anatomic_region_sequence

    # Anatomic Region Sequence: Anatomic Region 1
    anatomic_region1 = Dataset()
    anatomic_region_sequence.append(anatomic_region1)
    anatomic_region1.CodeValue = 'T-62000'
    anatomic_region1.CodingSchemeDesignator = 'SRT'
    anatomic_region1.CodeMeaning = 'Liver'

    frame_anatomy1.FrameLaterality = 'U'

    # Per-frame Functional Groups Sequence
    per_frame_functional_groups_sequence = Sequence()
    ds.PerFrameFunctionalGroupsSequence = per_frame_functional_groups_sequence

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 1
    per_frame_functional_groups1 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups1)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups1.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups1.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups1.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups1.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 1
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 1, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups1.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -35.4306]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups1.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups1.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups1.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '102.0'
    frame_voilut1.WindowWidth = '273.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups1.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 2
    per_frame_functional_groups2 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups2)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups2.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups2.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups2.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups2.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 2
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 2, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups2.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -30.4306]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups2.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups2.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups2.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '112.0'
    frame_voilut1.WindowWidth = '294.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups2.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 3
    per_frame_functional_groups3 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups3)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups3.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups3.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups3.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups3.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 3
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 3, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups3.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -25.4306]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups3.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups3.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups3.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '121.0'
    frame_voilut1.WindowWidth = '313.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups3.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 4
    per_frame_functional_groups4 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups4)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups4.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups4.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups4.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups4.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 4
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 4, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups4.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -20.4306]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups4.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups4.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups4.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '126.0'
    frame_voilut1.WindowWidth = '324.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups4.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 5
    per_frame_functional_groups5 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups5)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups5.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups5.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups5.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups5.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 5
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 5, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups5.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -15.4306]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups5.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups5.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups5.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '131.0'
    frame_voilut1.WindowWidth = '334.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups5.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 6
    per_frame_functional_groups6 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups6)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups6.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups6.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups6.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups6.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 6
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 6, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups6.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -10.4306]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups6.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups6.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups6.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '134.0'
    frame_voilut1.WindowWidth = '340.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups6.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 7
    per_frame_functional_groups7 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups7)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups7.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups7.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups7.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups7.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 7
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 7, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups7.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -5.43061]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups7.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups7.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups7.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '136.0'
    frame_voilut1.WindowWidth = '343.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups7.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 8
    per_frame_functional_groups8 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups8)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups8.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups8.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups8.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups8.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 8
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 8, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups8.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, -0.430613]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups8.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups8.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups8.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '134.0'
    frame_voilut1.WindowWidth = '339.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups8.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 9
    per_frame_functional_groups9 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups9)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups9.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups9.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups9.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups9.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 9
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 9, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups9.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 4.56939]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups9.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups9.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups9.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '131.0'
    frame_voilut1.WindowWidth = '333.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups9.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 10
    per_frame_functional_groups10 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups10)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups10.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups10.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups10.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups10.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 10
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 10, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups10.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 9.56939]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups10.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups10.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups10.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '127.0'
    frame_voilut1.WindowWidth = '324.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups10.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 11
    per_frame_functional_groups11 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups11)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups11.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups11.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups11.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups11.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 11
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 11, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups11.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 14.5694]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups11.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups11.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups11.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '122.0'
    frame_voilut1.WindowWidth = '313.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups11.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 12
    per_frame_functional_groups12 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups12)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups12.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups12.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups12.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups12.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 12
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 12, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups12.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 19.5694]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups12.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups12.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups12.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '115.0'
    frame_voilut1.WindowWidth = '300.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups12.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 13
    per_frame_functional_groups13 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups13)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups13.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups13.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups13.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups13.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 13
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 13, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups13.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 24.5694]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups13.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups13.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups13.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '110.0'
    frame_voilut1.WindowWidth = '291.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups13.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 14
    per_frame_functional_groups14 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups14)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups14.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups14.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups14.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups14.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 14
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 14, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups14.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 29.5694]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups14.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups14.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups14.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '108.0'
    frame_voilut1.WindowWidth = '286.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups14.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 15
    per_frame_functional_groups15 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups15)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups15.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups15.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups15.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups15.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 15
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 15, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups15.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 34.5694]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups15.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups15.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups15.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '99.0'
    frame_voilut1.WindowWidth = '267.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups15.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 16
    per_frame_functional_groups16 = Dataset()
    per_frame_functional_groups_sequence.append(per_frame_functional_groups16)

    # MR Echo Sequence
    mr_echo_sequence = Sequence()
    per_frame_functional_groups16.MREchoSequence = mr_echo_sequence

    # MR Echo Sequence: MR Echo 1
    mr_echo1 = Dataset()
    mr_echo_sequence.append(mr_echo1)
    mr_echo1.EffectiveEchoTime = 43.0

    # MR Averages Sequence
    mr_averages_sequence = Sequence()
    per_frame_functional_groups16.MRAveragesSequence = mr_averages_sequence

    # MR Averages Sequence: MR Averages 1
    mr_averages1 = Dataset()
    mr_averages_sequence.append(mr_averages1)
    mr_averages1.NumberOfAverages = '1.0'

    # MR Image Frame Type Sequence
    mr_image_frame_type_sequence = Sequence()
    per_frame_functional_groups16.MRImageFrameTypeSequence = mr_image_frame_type_sequence

    # MR Image Frame Type Sequence: MR Image Frame Type 1
    mr_image_frame_type1 = Dataset()
    mr_image_frame_type_sequence.append(mr_image_frame_type1)
    mr_image_frame_type1.FrameType = ['ORIGINAL', 'PRIMARY', 'M', 'NONE']
    mr_image_frame_type1.PixelPresentation = 'MONOCHROME'
    mr_image_frame_type1.VolumetricProperties = 'DISTORTED'
    mr_image_frame_type1.VolumeBasedCalculationTechnique = 'NONE'
    mr_image_frame_type1.ComplexImageComponent = 'MAGNITUDE'
    mr_image_frame_type1.AcquisitionContrast = 'UNKNOWN'

    # Frame Content Sequence
    frame_content_sequence = Sequence()
    per_frame_functional_groups16.FrameContentSequence = frame_content_sequence

    # Frame Content Sequence: Frame Content 1
    frame_content1 = Dataset()
    frame_content_sequence.append(frame_content1)
    frame_content1.FrameAcquisitionDateTime = '20240508140642.045000'
    frame_content1.FrameReferenceDateTime = '20240508140642.045000'
    frame_content1.FrameAcquisitionDuration = 507.5
    frame_content1.StackID = '1'
    frame_content1.InStackPositionNumber = 16
    frame_content1.TemporalPositionIndex = 1
    frame_content1.FrameAcquisitionNumber = 1
    frame_content1.DimensionIndexValues = [1, 16, 1]

    # Plane Position Sequence
    plane_position_sequence = Sequence()
    per_frame_functional_groups16.PlanePositionSequence = plane_position_sequence

    # Plane Position Sequence: Plane Position 1
    plane_position1 = Dataset()
    plane_position_sequence.append(plane_position1)
    plane_position1.ImagePositionPatient = [-179.635, -105.422, 39.5694]

    # Plane Orientation Sequence
    plane_orientation_sequence = Sequence()
    per_frame_functional_groups16.PlaneOrientationSequence = plane_orientation_sequence

    # Plane Orientation Sequence: Plane Orientation 1
    plane_orientation1 = Dataset()
    plane_orientation_sequence.append(plane_orientation1)
    plane_orientation1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    per_frame_functional_groups16.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures_sequence.append(pixel_measures1)
    pixel_measures1.SliceThickness = '5.0'
    pixel_measures1.PixelSpacing = [2, 2]

    # Frame VOI LUT Sequence
    frame_voilut_sequence = Sequence()
    per_frame_functional_groups16.FrameVOILUTSequence = frame_voilut_sequence

    # Frame VOI LUT Sequence: Frame VOI LUT 1
    frame_voilut1 = Dataset()
    frame_voilut_sequence.append(frame_voilut1)
    frame_voilut1.WindowCenter = '97.0'
    frame_voilut1.WindowWidth = '262.0'

    # Pixel Value Transformation Sequence
    pixel_value_transformation_sequence = Sequence()
    per_frame_functional_groups16.PixelValueTransformationSequence = pixel_value_transformation_sequence

    # Pixel Value Transformation Sequence: Pixel Value Transformation 1
    pixel_value_transformation1 = Dataset()
    pixel_value_transformation_sequence.append(pixel_value_transformation1)
    pixel_value_transformation1.RescaleIntercept = '0.0'
    pixel_value_transformation1.RescaleSlope = '1.0'
    pixel_value_transformation1.RescaleType = 'US'

    ds.PixelData = image_slices.astype(np.int16).tobytes()

    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.save_as(filename, write_like_original=False)


def write_dcm_from_ismrmrd_image(ismrmrd_image):
    """!
    @brief Writes an ismrmrd image to dicom format on the harddisk

    @param ismrmrd_image: (ismrmrd.Image) Image in ismrmrd format which shall be written to dicom

    @author Jörn Huber
    """

    img_array = ismrmrd_image.data
    if img_array.ndim == 4:  # (1, slices, height, width)
        img_array = img_array[0]

    num_slices, rows, cols = img_array.shape  # Shape: (z, y, x)

    fov = ismrmrd_image.field_of_view
    matrix = ismrmrd_image.matrix_size

    pixel_spacing = [
        fov[1] / matrix[1],  # row spacing (y)
        fov[2] / matrix[2]  # column spacing (x)
    ]
    slice_thickness = fov[0] / matrix[0]

    row_dir = ismrmrd_image.read_dir
    col_dir = ismrmrd_image.phase_dir

    image_orientation = [
        row_dir[2], row_dir[1], row_dir[0],
        col_dir[2], col_dir[1], col_dir[0]
    ]

    origin = ismrmrd_image.position
    origin_vec = [origin[2], origin[1], origin[0]]

    r = np.array([row_dir[2], row_dir[1], row_dir[0]])
    c = np.array([col_dir[2], col_dir[1], col_dir[0]])
    slice_norm = np.cross(r, c)

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.EnhancedMRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    ds = FileDataset("volume.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.PerformedProcedureStepDescription = ismrmrd_image.meta['ImageHistory'][2]
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')

    if 'referringPhysicianName' in ismrmrd_image.meta:
        ds.ReferringPhysicianName = ismrmrd_image.meta['referringPhysicianName']
    if 'studyDescription' in ismrmrd_image.meta:
        ds.StudyDescription = ismrmrd_image.meta['studyDescription']
    if 'seriesDescription' in ismrmrd_image.meta:
        ds.SeriesDescription = ismrmrd_image.meta['seriesDescription']
    if 'systemModel' in ismrmrd_image.meta:
        ds.ManufacturerModelName = ismrmrd_image.meta['systemModel']
    if 'systemVendor' in ismrmrd_image.meta:
        ds.Manufacturer = ismrmrd_image.meta['systemVendor']
    if 'institutionName' in ismrmrd_image.meta:
        ds.InstitutionName = ismrmrd_image.meta['institutionName']
    if 'patientName' in ismrmrd_image.meta:
        ds.PatientName = ismrmrd_image.meta['patientName']
    if 'patientID' in ismrmrd_image.meta:
        ds.PatientID = ismrmrd_image.meta['patientID']
    if 'patientBirthdate' in ismrmrd_image.meta:
        ds.PatientBirthDate = ismrmrd_image.meta['patientBirthdate']
    if 'patientGender' in ismrmrd_image.meta:
        ds.PatientSex = ismrmrd_image.meta['patientGender']
    if 'patientHeight_m' in ismrmrd_image.meta:
        ds.PatientSize = ismrmrd_image.meta['patientHeight_m']
    if 'patientWeight_kg' in ismrmrd_image.meta:
        ds.PatientWeight = ismrmrd_image.meta['patientWeight_kg']
    if 'bodyPartExamined' in ismrmrd_image.meta:
        ds.BodyPartExamined = ismrmrd_image.meta['bodyPartExamined']
    if 'systemFieldStrength_T' in ismrmrd_image.meta:
        ds.MagneticFieldStrength = ismrmrd_image.meta['systemFieldStrength_T']
    if 'protocolName' in ismrmrd_image.meta:
        ds.ProtocolName = ismrmrd_image.meta['protocolName']
    if 'patientPosition' in ismrmrd_image.meta:
        ds.PatientPosition = ismrmrd_image.meta['patientPosition']
    if 'studyID' in ismrmrd_image.meta:
        ds.StudyID = ismrmrd_image.meta['studyID']

    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = num_slices
    ds.PixelSpacing = pixel_spacing
    ds.SliceThickness = slice_thickness
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME']
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    # Private tag for scaling factor
    ds.add_new((0x0019, 0x0010), 'LO', 'AppliedScaling ' + str(ismrmrd_image.meta['scalingFactor']))

    # Shared Functional Groups
    shared_fg = Dataset()
    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = image_orientation
    shared_fg.PlaneOrientationSequence = Sequence([plane_orientation])
    ds.SharedFunctionalGroupsSequence = Sequence([shared_fg])

    # Per-frame Functional Groups
    per_frame = []
    for i in range(num_slices):
        fg = Dataset()

        # Calculation of position = origin + i * slice_norm * slice_thickness
        position = (np.array(origin_vec) + i * slice_thickness * slice_norm).tolist()

        # Plane Position
        plane_pos = Dataset()
        plane_pos.ImagePositionPatient = position
        fg.PlanePositionSequence = Sequence([plane_pos])

        per_frame.append(fg)

    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)

    if img_array.dtype != np.uint16:
        img_array = img_array.astype(np.uint16)

    ds.PixelData = img_array.tobytes()

    if 'protocolName' in ismrmrd_image.meta:
        prefix = ismrmrd_image.meta['protocolName'] + "/"
        if not os.path.exists(ismrmrd_image.meta['protocolName']):
            os.makedirs(ismrmrd_image.meta['protocolName'])
    else:
        prefix = ''

    ds.save_as(prefix + "series_" + str(ismrmrd_image.image_series_index) + ".dcm", write_like_original=False)
