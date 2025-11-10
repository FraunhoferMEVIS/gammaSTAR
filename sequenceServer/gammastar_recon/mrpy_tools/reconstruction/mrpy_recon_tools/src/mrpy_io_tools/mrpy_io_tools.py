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
from typing import Any, Tuple

def load_from_bart_cfl(filename: str) -> np.ndarray:
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


def save_to_bart_cfl(filename: str, array: np.ndarray) -> None:
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


def read_siemens_twix(
    file_name: str,
    unsorted: bool = False,
    remove_os: bool = True,
    ramp_samp_regrid: bool = True
) -> Tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """!
    @brief Reads twix rawdata as exported from the MR system into numpy files

    @param file_name: (string) File path + name of twix rawdata file
    @param unsorted: (bool) Boolean which indicates that data shall be read in unsorted format
    @param remove_os: (bool) Boolean indicating that oversampling shall be removed.
    @param ramp_samp_regrid: (bool) Boolean indicating that regridding of ramp sampling shall be performed.

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

def write_dcm_from_ismrmrd_image(ismrmrd_image: Any) -> None:
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
