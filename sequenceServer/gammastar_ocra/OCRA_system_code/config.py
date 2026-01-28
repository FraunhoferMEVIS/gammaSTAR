#!/usr/bin/env python3
"""
Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
The software is not qualified for use as a medical product or as part
thereof. No bugs or restrictions are known.
Author: Juela Cufe
"""


import numpy as np
from larmor_freq import saved_value
from attenuation_saved import atten_db
from shims import Shim_X, Shim_Y, Shim_Z, Shim_Z2


def calc_parameters(gammaB, gFactor, attenuation_saved):
    "Calculating system configuration parameters necessary to translate sequences from gammaSTAR to MaRCoS server"

    attenuation_db = attenuation_saved
    b1Efficiency = 10 ** (attenuation_db / 10)
    rf_amp_max = (b1Efficiency / (2 * np.pi) * 1e6,)  # Hz
    gx_max = gFactor[0] * gammaB  # Hz/m
    gy_max = gFactor[1] * gammaB  # Hz/m
    gz_max = gFactor[2] * gammaB  # Hz/m
    grad_max = np.max(gFactor) * gammaB
    param_dict = {
        "b1Efficiency": b1Efficiency,
        "rf_amp_max": rf_amp_max,
        "gx_max": gx_max,
        "gy_max": gy_max,
        "gz_max": gz_max,
        "grad_max": grad_max,
    }

    return param_dict


gammaB = 42.56e6  # Hz/T
gFactor = [0.335000, 0.319000, 0.325000]


param_dict = calc_parameters(gammaB, gFactor, atten_db)
attenuation_db = atten_db
LARMOR_FREQ = saved_value  # System Larmor Frequency, in MHz

RF_MAX = param_dict["rf_amp_max"]  # System maximum RF amplitude, in Hz
RF_PI2_FRACTION = param_dict["b1Efficiency"]  # Fraction of power to expect a pi/2 pulse

GX_MAX = param_dict["gx_max"]  # System maximum X gradient strength, in Hz/m
GY_MAX = param_dict["gy_max"]  # System maximum Y gradient strength, in Hz/m
GZ_MAX = param_dict["gz_max"]  # System maximum Z gradient strength, in Hz/m
GZ2_MAX = param_dict["grad_max"]  # System maximum Z2 gradient strength, in Hz/m

SHIM_X = Shim_X
SHIM_Y = Shim_Y
SHIM_Z = Shim_Z
SHIM_Z2 = Shim_Z2
