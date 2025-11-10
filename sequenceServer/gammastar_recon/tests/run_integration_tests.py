"""!
@brief Collection of integration tests for gammaSTAR Reconstruction
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         **InsertLicense** code

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import sys
import argparse
import os
import numpy as np
import pydicom
import shutil
import matplotlib.pyplot as plt
import subprocess

defaults = {
    'filepath':        '//gaia/foci/mrphys/rc_next_gen_imaging/Data/test_data_gstar_recon/mrzero'
}

def test_mrzero(args):
    """!
    @brief Integration tests based on MRZero test data
    @details

    @param self: Reference to object

    @author Jörn Huber
    """

    success = True
    
    if os.path.exists('build_date_time.txt'):
        with open('build_date_time.txt', 'r') as f:
            build_date_time = f.read().strip()
            print("Test suite built on: " + build_date_time)

    # Get a list of available test data files
    test_data_files = [f for f in os.listdir(args.filepath) if f.endswith('.h5')]

    for test_file in test_data_files:

        print("Processing file: " + test_file)

        # Remove the default directory if it exists
        if os.path.exists("default"):
            shutil.rmtree("default")

        # Run the client for each test file
        try:
            subprocess.run(
                ["python3", "clients/mrd_to_dicom_client.py", args.filepath + "/" + test_file, "--address", "gstar_server", "--port", "9002"],
                timeout=600, check=True
            )
        except subprocess.TimeoutExpired:
            print(f"Timeout: Command for {test_file} exceeded 10 minutes and was terminated.")
            success = False
            break

        if os.path.isdir("default"):

            dcm_files = sorted([f for f in os.listdir("default") if f.endswith('.dcm')])

            ds_list = [pydicom.dcmread(os.path.join("default", f)).pixel_array for f in dcm_files]
            ds = np.stack([arr if arr.ndim == 2 else arr[0] for arr in ds_list], axis=0)

            fig, axes = plt.subplots(1, ds.shape[0], figsize=(5 * ds.shape[0], 5))
            if ds.shape[0] == 1:
                axes = [axes]

            output_dir = "/opt/code/test_results"
            os.makedirs(output_dir, exist_ok=True)
            for i in range(ds.shape[0]):
                axes[i].imshow(ds[i, :, :], cmap='gray')
                axes[i].set_title(f"Slice {i}")
                axes[i].axis('off')
                fig.savefig(os.path.join(output_dir, f"{test_file}.png"))
            plt.close(fig)

    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integration tests for gammaSTAR Reconstructions',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filepath', help='Input filesystem path')
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    success_mrzero = test_mrzero(args)

    if success_mrzero:
        sys.exit(0)
    else:
        sys.exit(1)
