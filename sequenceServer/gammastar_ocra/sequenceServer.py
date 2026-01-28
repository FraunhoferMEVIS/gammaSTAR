"""!
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import sys
import asyncio
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
from datetime import datetime
import random
import ismrmrd
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime
import shutil
import zipfile
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Any
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_io_tools as io_tools

def print_info_assembler():
    print(
        "########################################################\n\n"
        "gammaSTAR OCRA  v1.0 Release\n"
    )

    print(
        "The software is not qualified for use as a medical product or as part\n"
        "thereof. No bugs or restrictions are known. Delivered ‘as is’ without\n"
        "specific verification or validation.\n"
        "(C) Fraunhofer MEVIS 2025\n"
        "Contact: Juela Cufe (juela.cufe@mevis.fraunhofer.de)\n"
    )

    print(
        "List of included third-party software\n"
        "-\n"
        "ismrmrd - (ISMRMRD SOFTWARE LICENSE)\n"
        "h5py -(BSD-3-Clause license)\n"
        "pydicom - (The MIT License (MIT))\n"
        "msgpack- (MIT License)\n"
        "fastapi (MIT License)\n"
        "numpy – BSD License\n"
        "matplotlib – Matplotlib License (based on PSF)\n"
        "scipy (BSD 3-Clause)\n"
        "MaRCoS MRI - GPL-3.0 License\n"
        "MRI4ALL Console – GPL-3.0 License\n"
        "Flocra-pulseq - (MIT license)\n"
        "-\n"
        "Detailed information about third-party licence conditions can\n"
        "be found in the 'third_party_licenses' folder.\n\n"
        "########################################################\n"
    )


time.sleep(0.1)
print_info_assembler()
time.sleep(0.1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Start webserver
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


httpd = FastAPI(lifespan=lifespan)

# URL for the webserver
origins = ["http://localhost", "https://gammastar.mevis.fraunhofer.de"]

# Add CORS middleware to allow cross-origin requests
httpd.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Access-Control-Allow-Origin", "Content-Type"],
    expose_headers=["Access-Control-Allow-Origin"],
    max_age=5,
)

# Ensure /data_export is always an empty directory
if os.path.exists("/data_export"):
    shutil.rmtree("/data_export")
os.makedirs("/data_export", exist_ok=True)
httpd.mount("/data_export/", StaticFiles(directory="/data_export"), name="data_export")


@httpd.get("/")
def default():
    return {"message": "Hello World"}


# Simulate sequence for simulation
def reconstruct_image(
    sequenceData: dict[str, Any], signal: list[float], imgBufferTrajectory: BytesIO
):
    output = []
    raw_adc_representations = []
    for rawRepr in sequenceData["sequence"]:
        output.append(json.dumps(rawRepr))
        if rawRepr["has_adc"] == True:
            raw_adc_representations.append(rawRepr)

    # gammaSTAR recon
    np_arr = np.array(signal)
    sequenceData["protocol"]["read_oversampling"] = (
        sequenceData["protocol"]["read_oversampling"] + 1
    )
    sequenceData["protocol"]["slice_oversampling"] = (
        sequenceData["protocol"]["slice_oversampling"] + 1
    )
    sequenceData["protocol"]["phase_oversampling"] = (
        sequenceData["protocol"]["phase_oversampling"] + 1
    )

    read_size = (
        sequenceData["root"]["acq_size"]["1"]
        * (sequenceData["protocol"]["read_oversampling"])
        * sequenceData["protocol"]["read_partial_fourier"]
    )
    np_reshape = np_arr.reshape(
        -1, 1, sequenceData["expo"].get("adc_samples", read_size)
    ).T

    adc_points = sequenceData["trajectory"]["adc_points"]
    list_of_acqs = ismrmrd_tools.numpy_and_raw_rep_to_acq(
        np_reshape, raw_adc_representations, adc_points
    )
    xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(
        sequenceData["protocol"],
        sequenceData["info"],
        sequenceData["expo"],
        sequenceData["sys"],
        sequenceData["root"],
    )

    response = {"dicom": "", "trajectory": [], "kspace": [], "recon": [], "raw": ""}

    os.makedirs("/data_export", exist_ok=True)
    meas_id = str(random.randint(10000, 99999))
    raw_filename = os.path.join(
        "/data_export/" + sequenceData["name"].replace(" ", "_") + meas_id + "_raw.h5"
    )
    with ismrmrd.Dataset(raw_filename, "dataset", create_if_needed=True) as dset:
        dset.write_xml_header(xml_header)

        for i, acq in enumerate(list_of_acqs):
            dset.append_acquisition(acq)
        response["raw"] = raw_filename
        print(sequenceData["name"] + meas_id + "_raw.h5 written")

    if 'skipReconstruction' not in sequenceData or sequenceData['skipReconstruction'] != True:
        con = ismrmrd_tools.gstar_recon_emitter("reconServer", 9002, list_of_acqs, xml_header, 1.0, "")
        if con != False:
            ismrmrd_tools.gstar_recon_injector(con)
            con.shutdown_close()
            recon_images = []

            if len(con.images) > 0:

                for img_idx in range(len(con.images)):
                    image_data = con.images[img_idx].data[0, :, :, :]
                    for slice_index in range(len(image_data)):
                        image_slice = image_data[slice_index]

                        # Normalize image
                        image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice))

                        # save to h5
                        recon_images.append(image_slice.astype(np.float32))

                        # Encode as PNG
                        imgBufferRecon = BytesIO()
                        plt.figure(figsize=(20, 20))
                        plt.imshow(image_slice, cmap='gray')
                        plt.axis('off')
                        plt.gca().set_axis_off()
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        plt.savefig(imgBufferRecon, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
                        imgBufferRecon.seek(0)
                        plt.close()
                        b64Recon = base64.b64encode(imgBufferRecon.getvalue())
                        response['recon'].append(b64Recon.decode())

                response['recon_history'] = con.images[0].meta['ImageHistory'][2]

                logger.info(f"Sent reconstruction history: {con.images[0].meta['ImageHistory'][2]}")

                dicom_dir = '/data_export/dicom_' + sequenceData['name'].replace(' ', '_') + '_' + meas_id
                os.makedirs(dicom_dir, exist_ok=True, )
                zip_filename = os.path.join(dicom_dir, f"{sequenceData['name'].replace(' ', '_')}_{meas_id}_dicom.zip")
                for recv_image in con.images:
                    io_tools.write_dcm_from_ismrmrd_image(recv_image, dicom_dir)

                dicom_files = os.listdir(dicom_dir)

                with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    for dicom in dicom_files:
                        dicom_filename = os.path.join(dicom_dir, dicom)
                        zipf.write(dicom_filename, arcname=dicom)

                response['dicom'] = zip_filename

    if imgBufferTrajectory:
        b64Trajectory = base64.b64encode(imgBufferTrajectory.getvalue())
        response['trajectory'].append(b64Trajectory.decode())

    return JSONResponse(content=response)


########################################################################################################################
# OCRA tabletop system
########################################################################################################################
sys.path.append("/src/OCRA_system_code")
from execute_sequence import run_ocra
import shutil
import zipfile
import config as cfg


def send_receive_sequence(sequenceData):
    name = sequenceData["name"]
    rawRepr = sequenceData["sequence"]
    rxd = run_ocra(
        rawRepr,
        rf_center=cfg.LARMOR_FREQ,
        rf_max=cfg.RF_MAX,
        gx_max=cfg.GX_MAX,
        gy_max=cfg.GY_MAX,
        gz_max=cfg.GZ_MAX,
        gz2_max=cfg.GZ2_MAX,
        tx_t=123 / 122.88,
        grad_t=1229 / 122.88,
        tx_warmup=0,
        shim_x=cfg.SHIM_X,
        shim_y=cfg.SHIM_Y,
        shim_z=cfg.SHIM_Z,
        shim_z2=cfg.SHIM_Z2,
        expt=None,
        plot_instructions=False,
    )

    return rxd


async def main_ocra(sequence):
    response = None

    # create data that is send to system
    sequence_data = {"name": sequence["name"], "sequence": sequence["sequence"]}

    if sequence_data:
        k_space_data = send_receive_sequence(sequence_data)

        print("k_space_data: ", np.shape(k_space_data))
        data_full = k_space_data
        response = reconstruct_image(sequence, data_full, None)

    return response


# Endpoint for running ocra
@httpd.post("/run_ocra")
def runSequence(sequence: dict[str, Any]):
    logging.info("Run ocra sequence")
    response = asyncio.run(main_ocra(sequence))
    return response
