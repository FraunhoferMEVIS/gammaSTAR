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
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import os
import base64
from io import BytesIO
from datetime import datetime
import random
import ismrmrd
import h5py
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

def print_info_ilumr():
    print("########################################################\n\n"
          "gammaSTAR ilumr v1.0 Release\n")

    print("The software is not qualified for use as a medical product or as part\n"
          "thereof. No bugs or restrictions are known. Delivered ‘as is’ without\n"
          "specific verification or validation.\n"
          "(C) Fraunhofer MEVIS 2025\n"
          "Contact: Daniel Christopher Hoinkiss (daniel.hoinkiss@mevis.fraunhofer.de)\n")

    print("List of included third-party software\n"
          "-\n"
          "numpy (BSD License)\n"
          "matplotlib (Matplotlib License)\n"
          "ismrmrd (ISMRMRD SOFTWARE LICENSE JULY 2013)\n"
          "h5py (BSD License)\n"
          "pydicom (MIT License)\n"
          "fastapi (MIT License)\n"
          "-\n"
          "Detailed information about third-party licence conditions can\n"
          "be found in the 'third_party_licenses' folder.\n\n"
          "########################################################\n")

time.sleep(0.1)
print_info_ilumr()
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
origins = ['http://localhost', 'https://gammastar.mevis.fraunhofer.de']

# Add CORS middleware to allow cross-origin requests
httpd.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Access-Control-Allow-Origin", "Content-Type"],
    expose_headers=["Access-Control-Allow-Origin"],
    max_age=5
)

# Ensure /data_export is always an empty directory
if os.path.exists("/data_export"):
    shutil.rmtree("/data_export")
os.makedirs("/data_export", exist_ok=True)
httpd.mount("/data_export/", StaticFiles(directory="/data_export"), 
            name="data_export")

@httpd.get('/')
def default():
    return {'message': 'Hello World'}

# Sequence reconstruction function
def reconstruct_image(sequenceData: dict[str, Any], signal: list[float], imgBufferTrajectory: BytesIO, frequency_hz=None):
    output = []
    raw_adc_representations = []
    for rawRepr in sequenceData['sequence']:
        output.append(json.dumps(rawRepr))
        if rawRepr['has_adc'] == True:
            raw_adc_representations.append(rawRepr)

# gammaSTAR recon
    np_arr = np.array(signal)
    sequenceData['protocol']['read_oversampling'] = (sequenceData['protocol']['read_oversampling'] + 1)
    sequenceData['protocol']['slice_oversampling'] = sequenceData['protocol']['slice_oversampling'] + 1
    sequenceData['protocol']['phase_oversampling'] = sequenceData['protocol']['phase_oversampling'] + 1

    read_size = sequenceData['root']['acq_size']['1'] * (sequenceData['protocol']['read_oversampling']) * sequenceData['protocol']['read_partial_fourier']
    np_reshape = np_arr.reshape(-1, 1, sequenceData['expo'].get('adc_samples', read_size)).T

    adc_points = sequenceData['trajectory']['adc_points']
    list_of_acqs = ismrmrd_tools.numpy_and_raw_rep_to_acq(np_reshape, raw_adc_representations, adc_points)

    # Hardcoded DICOM header info for Resonint ilumr
    def hardcoded_resonint_ilumr_meas_info(frequency_hz: float | None = None):
        from datetime import datetime, timedelta

        # Current local time with timezone
        now = datetime.now().astimezone()
        study_date = now.strftime("%Y%m%d")
        study_time = now.strftime("%H%M%S")

        # Timezone offset (+HHMM/-HHMM)
        offset = now.utcoffset() or timedelta(0)
        offset_minutes = int(offset.total_seconds() // 60)
        sign = '+' if offset_minutes >= 0 else '-'
        hh = abs(offset_minutes) // 60
        mm = abs(offset_minutes) % 60
        tz_offset = f"{sign}{hh:02d}{mm:02d}"

        gamma_hz_per_t = 42_577_478.92  # Hz/T (1H)
        if frequency_hz and frequency_hz > 0:
            field_strength_t = float(frequency_hz) / gamma_hz_per_t
            h1_freq_hz = int(round(frequency_hz))
        else:
            field_strength_t = 0.33
            h1_freq_hz = int(round(gamma_hz_per_t * field_strength_t))

        meas_info = {
            "patientName": "Phantom",
            "patientID": "P123456",
            "studyDate": study_date,
            "studyTime": study_time,
            "timezoneOffsetFromUTC": tz_offset,
            "measurementID": "STUDY001",
            "systemVendor": "Resonint",
            "systemModel": "ilumr",
            "systemFieldStrength_T": round(field_strength_t, 5),
            "institutionName": "Fraunhofer MEVIS",
            "H1resonanceFrequency_Hz": h1_freq_hz,
        }
        return meas_info

    meas_info = hardcoded_resonint_ilumr_meas_info(frequency_hz)

    # xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(sequenceData['protocol'], sequenceData['info'], sequenceData['expo'], sequenceData['sys'], sequenceData['root'], meas_info)
    xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(sequenceData['protocol'], sequenceData['info'], sequenceData['expo'], sequenceData['sys'], sequenceData['root'])

    response = {'dicom': "", 'trajectory': [], 'kspace': [], 'recon': [], 'raw': ""}

    os.makedirs('/data_export', exist_ok=True)
    meas_id = str(random.randint(10000, 99999))
    raw_filename = os.path.join('/data_export/' + sequenceData['name'].replace(' ', '_') + meas_id + '_raw.h5')
    with ismrmrd.Dataset(raw_filename, 'dataset',
                        create_if_needed=True) as dset:
       dset.write_xml_header(xml_header)

       for i, acq in enumerate(list_of_acqs):
           dset.append_acquisition(acq)
       response['raw'] = raw_filename
       print(sequenceData['name'] + meas_id + '_raw.h5 written')

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
# ilumr tabletop system
########################################################################################################################

import socket
import struct
import traceback
from typing import Optional, Dict, Any, List
import time

# Function to receive initial data
def receive_initial_data(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """Receive initial server data and configuration"""
    try:
        message = receive_tcp_message(sock, 300.0)
        if message:
            data = json.loads(message)
            return data
        else:
            logger.error("No initial data received from server")
            return None
    except Exception as e:
        logger.error(f"Error receiving initial data: {e}")
        return None

# Enhanced TCP Message Protocol Functions
def send_tcp_message(sock: socket.socket, data: Any) -> bool:
    """Send message with 4-byte length header (big endian)"""
    try:
        data_str = json.dumps(data) if not isinstance(data, str) else data
        data_bytes = data_str.encode('utf-8')
        length = len(data_bytes)
        
        # logger.debug(f"Sending message of length: {length}")
        
        # Send length header (4 bytes, big endian)
        length_header = struct.pack('>I', length)
        sock.sendall(length_header)
        
        # Send data
        sock.sendall(data_bytes)
        logger.debug("Message sent successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error sending TCP message: {e}")
        return False

def setup_socket_keepalive(sock: socket.socket, 
                          keepalive_time: int = 30, 
                          keepalive_interval: int = 10,  
                          keepalive_probes: int = 3):  
    
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    
    if hasattr(socket, 'TCP_KEEPIDLE'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, keepalive_time)
    if hasattr(socket, 'TCP_KEEPINTVL'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, keepalive_interval)
    if hasattr(socket, 'TCP_KEEPCNT'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, keepalive_probes)

def receive_tcp_message(sock: socket.socket, timeout: float = 300.0) -> Optional[str]:
    """Receive message with 4-byte length header (big endian)"""
    try:        
        setup_socket_keepalive(sock)
        
        # Receive length header (4 bytes)
        length_data = b''
        while len(length_data) < 4:
            chunk = sock.recv(4 - len(length_data))
            if not chunk:
                logger.error("Connection closed while receiving length header")
                return None
            length_data += chunk
        
        length = struct.unpack('>I', length_data)[0]
        # logger.debug(f"Expecting message of length: {length}")
        
        if length <= 0 or length > 100 * 1024 * 1024: 
            logger.error(f"Invalid message length: {length}")
            return None
        
        # Receive message data
        message_data = b''
        while len(message_data) < length:
            chunk_size = min(8192, length - len(message_data))
            chunk = sock.recv(chunk_size)
            if not chunk:
                logger.error("Connection closed while receiving message data")
                return None
            message_data += chunk
        
        message_str = message_data.decode('utf-8')
        # logger.debug("Message received successfully")
        return message_str
        
    except Exception as e:
        logger.error(f"Error receiving TCP message: {e}")
        return None

def send_in_chunks(sock: socket.socket, data: Any, chunk_size: int = 1024) -> bool:
    try:
        data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
        total_size = len(data_str)
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            chunk = data_str[start:end]

            chunk_data = {
                "chunk_index": i,
                "total_chunks": num_chunks,
                "data": chunk
            }

            success = send_tcp_message(sock, chunk_data)
            if not success:
                logger.error(f"Failed to send chunk {i + 1} of {num_chunks}")
                return False
            
            # logger.debug(f"Sent chunk {i + 1} of {num_chunks}")

        return True
    except Exception as e:
        logger.error(f"Error sending chunked data: {e}")
        return False
    
# send sequence data
def send_sequence_data(sock: socket.socket, sequence_data: Dict[str, Any]) -> Optional[List[Any]]:
    try:
        if not send_tcp_message(sock, sequence_data):
            logger.error("Failed to send sequence data")
            return None
        
        ack_message = receive_tcp_message(sock, 30.0)
        if ack_message:
            ack_data = json.loads(ack_message)
            if ack_data.get('status') == 'received':
                # logger.debug(f"Sequence acknowledged: {ack_data.get('message', 'N/A')}")
                pass
            elif ack_data.get('status') == 'error':
                logger.error(f"Server error during acknowledgment: {ack_data.get('error', 'unknown')}")
                return None
            else:
                logger.warning(f"Unexpected acknowledgment status: {ack_data.get('status')}")
        else:
            logger.error("No acknowledgment received from server")
            return None
        
        status_message = receive_tcp_message(sock, 60.0)
        if status_message:
            status_data = json.loads(status_message)
            if status_data.get('status') == 'processing':
                # logger.debug(f"Server processing: {status_data.get('message', 'N/A')}")
                pass
            elif status_data.get('status') == 'busy':
                logger.error(f"Server busy: {status_data.get('error', 'System busy')}")
                return None
            elif status_data.get('status') == 'error':
                logger.error(f"Server error during processing: {status_data.get('error', 'unknown')}")
                return None
            else:
                logger.warning(f"Unexpected processing status: {status_data.get('status')}")
        else:
            logger.error("No processing status received from server")
            return None
        
        result_data = receive_in_chunks(sock)
        
        if result_data:
            status = result_data.get('status')
            success = result_data.get('success')
            
            if status == 'complete' and success:
                final_progress = result_data.get('final_progress', [])
                if final_progress and len(final_progress) >= 2:
                    logger.debug(f"Final progress: {final_progress[0]}/{final_progress[1]} acquisitions")
                
                measurement_data = result_data.get('data', [])
                return measurement_data
            
            elif status == 'error' or not success:
                error_msg = result_data.get('error', 'Unknown server error')
                logger.error(f"Server reported error: {error_msg}")
                logger.error(f"Run ID: {result_data.get('run_id', 'N/A')}")
                return None
                
            else:
                logger.error(f"Unexpected result status: {status}, success: {success}")
                return None
        else:
            logger.error("No result data received from server")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in send_sequence_data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in send_sequence_data: {e}")
        return None
    
def receive_in_chunks(sock: socket.socket) -> Optional[Dict[str, Any]]:
    chunks = {}
    total_chunks = None


    while True:
        message = receive_tcp_message(sock, 300.0) 
        if not message:
            logger.error("Failed to receive message chunk")
            return None
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        
        if 'chunk_index' not in data:
            return data
        
        # Handle chunked data
        chunk_index = data.get('chunk_index')
        total_chunks = data.get('total_chunks')
        chunk_data = data.get('data')
        
        if chunk_index is None or total_chunks is None or chunk_data is None:
            logger.error("Invalid chunk data received")
            return None

        chunks[chunk_index] = chunk_data
        
        if len(chunks) == total_chunks:            
            full_data_str = ''.join(chunks[i] for i in range(total_chunks))
            
            try:
                result = json.loads(full_data_str)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding assembled data: {e}")
                return None
    
# deserialize data array
def deserialize_array(serialized_arr: List[Any]) -> np.ndarray:
    """Convert serialized array back to numpy array"""
    if not serialized_arr:
        logger.warning("Empty data array received")
        return np.array([])
    
    try:
        if isinstance(serialized_arr[0], dict):
            complex_arr = np.array([complex(x['real'], x['imag']) for x in serialized_arr])
            return complex_arr.astype(np.complex64)
        else:
            return np.array(serialized_arr, dtype=np.float32).view(np.complex64)
            
    except Exception as e:
        logger.error(f"Error deserializing array: {e}")
        return np.array([])

# run sequence for ilumr (TCP version)
def run_ilumr(sequence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Main function to run NMR sequence using TCP communication"""
    host = "192.168.137.2"
    port = 8765
    
    sock = None
    try:        
        # Create TCP socket with proper configuration
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        # Connect to server
        sock.connect((host, port))
        
        # Receive initial data from server
        initial_data = receive_initial_data(sock)
        if not initial_data:
            logger.error("Failed to receive initial server data")
            return None
        
        if not initial_data.get('server_ready', False):
            logger.error("Server reports it is not ready")
            return None

        try:
            frequency_hz = float(initial_data.get('frequency')) if initial_data.get('frequency') is not None else None
        except (TypeError, ValueError):
            frequency_hz = None
        
        time.sleep(0.1)
        
        # Prepare sequence data for transmission
        sequence_data = {
            "sequence": sequence.get("name", "unknown_sequence"),
            "data": sequence.get("sequence", {})
        }
        
        k_space_data_list = send_sequence_data(sock, sequence_data)
        
        if k_space_data_list is not None:
            k_space_data = deserialize_array(k_space_data_list)

            if len(k_space_data) > 0:
                # Call the image reconstruction function
                response = reconstruct_image(sequence, k_space_data, None, frequency_hz=frequency_hz)
                return response
            else:
                logger.error("Received empty data array")
                return None
        else:
            logger.error("Failed to receive measurement data from server")
            return None
    
    except ConnectionRefusedError:
        logger.error(f"Connection refused to {host}:{port} - is the server running?")
        return None
    except socket.timeout:
        logger.error("Socket timeout - server may be unresponsive")
        return None
    except socket.error as e:
        logger.error(f"Socket error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in simulate_ilumr: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass

# Endpoint for running ilumr
@httpd.post('/run_ilumr')
def run_ilumr_sequence(sequence: dict[str, Any]):
    logging.info('Run ilumr sequence')
    response = run_ilumr(sequence)  
    return response