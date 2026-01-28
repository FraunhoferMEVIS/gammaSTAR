#!/usr/bin/env python3
"""
Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
The software is not qualified for use as a medical product or as part
thereof. No bugs or restrictions are known.
Author: Juela Cufe


Portion of this software, specifically the function `apply_shim`,
was adapted from *https://github.com/mri4all/console/tree/main/external/seq/adjustments_acq/scripts.py*.
Licensed under the GNU GENERAL PUBLIC LICENSE.

"""

import time
from gammaSTAR_MaRCoS_driver_class import GSAssembler
import config as cfg
import numpy as np
from marcos_client import experiment as ex
import matplotlib.pyplot as plt


def run_ocra(
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
):

    # Convert gammaSTAR raw data representation file to machine dictionary
    psi = GSAssembler(
        raw_rep_list=rawRepr,
        rf_center=rf_center * 1e6,  # Hz
        tx_warmup=tx_warmup,
        rf_amp_max=rf_max,
        tx_t=tx_t,
        grad_t=grad_t,
        gx_max=gx_max,
        gy_max=gy_max,
        gz_max=gz_max,
        clk_t=1 / 122.88,
    )

    instructions, param_dict = psi.interpret()
    print("Sampling Time (µs)", param_dict["rx_t"])

    # Shim
    instructions = apply_shim(instructions, (shim_x, shim_y, shim_z, shim_z2))

    print("Initialize experiment class... ")
    if expt is None:
        expt = ex.Experiment(
            lo_freq=rf_center, rx_t=param_dict["rx_t"], init_gpa=True  # us
        )

    flat_delay = 10
    for buf in instructions.keys():
        instructions[buf] = (instructions[buf][0] + flat_delay, instructions[buf][1])

    print("Interpreted gammaSTAR sequence...")
    if plot_instructions:

        fig, axs = plt.subplots(
            len(instructions), 1, figsize=(8, 6), constrained_layout=True, sharex=True
        )
        fig.suptitle(
            "gammaSTAR Interpreter", fontsize=18, fontweight="bold", color="darkblue"
        )

        colors = plt.cm.viridis(np.linspace(0, 1, len(instructions)))

        for i, (key, color) in enumerate(zip(instructions.keys(), colors)):
            axs[i].step(
                instructions[key][0],
                np.real(instructions[key][1]),
                where="post",
                linewidth=1.5,
                color=color,
            )
            axs[i].plot(
                instructions[key][0],
                np.real(instructions[key][1]),
                "o",
                markersize=3,
                color="red",
                label="Data Points",
            )

            axs[i].set_title(f"{key}", fontsize=12, fontweight="bold", color=color)
            axs[i].grid(True, linestyle="--", alpha=0.6)

        axs[-1].set_xlabel("Time (µs)", fontsize=12)
        plt.show()

    print("Load Instructions to the server ...")
    time.sleep(0.15)
    expt.add_flodict(instructions)
    print("Instructions launched ...")
    time.sleep(0.15)

    print("Run experiment...")
    adcFactor = 14.395
    rxd, msgs = expt.run()
    rxd["rx0"] = adcFactor * (np.real(rxd["rx0"]) - 1j * np.imag(rxd["rx0"]))
    expt.__del__()

    return rxd["rx0"]


def apply_shim(instructions, shim):
    grad_keys = ["grad_vx", "grad_vy", "grad_vz", "grad_vz2"]

    for i, offset in enumerate(shim):
        key = grad_keys[i]
        value, updated_shims = instructions[key]

        modified = updated_shims.copy()
        modified[:-1] += offset

        if np.any(np.abs(modified) > 1):
            shims_val = modified[np.argmax(np.abs(modified))]
            raise ValueError(f"Shim {offset} too large for {key}: {shims_val}")

        instructions[key] = (value, modified)

    return instructions
