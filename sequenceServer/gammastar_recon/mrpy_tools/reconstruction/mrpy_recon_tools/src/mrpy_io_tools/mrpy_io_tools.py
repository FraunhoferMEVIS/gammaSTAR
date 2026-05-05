"""!
@brief Collection of tools which are used for input/output tasks.
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import os
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np
from datetime import datetime
from typing import Any


def get_folder_name(target_folder: str) -> str:
    """!
    @brief Gets the output folder name based on the protocol name stored in the ismrmrd header.
    @details The function checks whether a folder with the protocol name already exists. If so, it appends an
             increasing number to the folder name until a non-existing folder name is available.

    @param target_folder: Protocol name stored in the ismrmrd header.

    @return
        - Output folder name for the dicom series

    @author Jörn Huber
    """

    os.makedirs(target_folder, exist_ok=True)
    sub_folders = [int(name) for name in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, name)) and name.isdigit()]

    if len(sub_folders) == 0:
        out_folder = target_folder + "/0"
    else:
        out_folder = target_folder + "/" + str(max(sub_folders)+1)

    return out_folder


def write_dcm_from_ismrmrd_image(ismrmrd_image: Any, folder_name = None) -> None:
    """!
    @brief Writes an ismrmrd image to dicom format on the harddisk

    @param ismrmrd_image: Image in ismrmrd format which shall be written to dicom
    @param folder_name: Optional name of the output folder. If not provided, the protocol name from the ismrmrd header
                        is used as folder name.

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
    ds.is_implicit_VR = False
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
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
    ds.add_new((0x0019, 0x0011), 'LO', 'ReconHistory' + str(ismrmrd_image.meta['ImageHistory'][2]))
    ds.add_new((0x0019, 0x0012), 'UC', ismrmrd_image.meta.get('seq', 'N/A'))
    ds.add_new((0x0019, 0x0013), 'UC', ismrmrd_image.meta.get('prot', 'N/A'))

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

    if folder_name:
        prefix = folder_name + "/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
    elif 'protocolName' in ismrmrd_image.meta:
        prefix = ismrmrd_image.meta['protocolName']

        count = 1
        temp_prefix = prefix
        while os.path.exists(temp_prefix):
            temp_prefix = f"{prefix}_{count}"
            count += 1
        prefix = temp_prefix + "/"

        if not os.path.exists(prefix):
            os.makedirs(prefix)
    else:
        prefix = ''

    ds.save_as(prefix + "series_" + str(ismrmrd_image.image_series_index) + ".dcm", write_like_original=False)
