#!/usr/bin/env python3
"""
Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
The software is not qualified for use as a medical product or as part
thereof. No bugs or restrictions are known.
Author: Juela Cufe


Portions of this software, specifically the functions `_stream_block`,
`_stream_all_blocks`, and `interpret`, were adapted from *flocra_pulseq*.

Original work:
Copyright (c) 2021 Lincoln Craven-Brightman
Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import logging
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")
ROUNDING = 0


class GSAssembler:
    """
    Assembler object that can assemble a gammaSTAR file into MaRCoS server/OCRA machine code. Run GSAssembler.assemble to compile a .json file into MaRCoS array input data

    Attributes:
        tx_bytes (bytes): Transmit data bytes
        grad_bytes (list): List of grad bytes
        command_bytes (bytes): Command bytes
        readout_number (int): Expected number of readouts
    """

    def __init__(
        self,
        raw_rep_list: List[Dict[str, Any]],
        rf_center: Optional[float] = None,
        rf_amp_max: Optional[float] = None,
        gx_max: Optional[float] = None,
        gy_max: Optional[float] = None,
        gz_max: Optional[float] = None,
        grad_max: Optional[float] = None,
        tx_t: float = 123 / 122.88,
        grad_t: float = 1229 / 122.88,
        tx_warmup: int = 0,
        shim_x: Optional[float] = None,
        shim_y: Optional[float] = None,
        shim_z: Optional[float] = None,
        clk_t: float = 1 / 122.88,
        tx_zero_end: bool = True,
        grad_zero_end: bool = True,
        gamma: float = 42.575575e6,
        rf_delay_preload: bool = False,
        addresses_per_grad_sample: int = 1,
        grad_pad: int = 0,
        adc_pad: int = 0,
        rf_pad_type: str = "ext",
        grad_pad_type: str = "ext",
        log_file: str = "gs_interpreter",
        log_level: int = 20,
    ):
        self.rf_center = rf_center
        self._rf_amp_max = rf_amp_max
        self.gx_max = gx_max
        self.gy_max = gy_max
        self.gz_max = gz_max
        self.grad_max = grad_max
        self._tx_t = tx_t
        self.grad_t = grad_t
        self._tx_warmup = tx_warmup
        self.shim_x = shim_x
        self.shim_y = shim_y
        self.shim_z = shim_z
        self._clk_t = clk_t
        self._tx_zero_end = tx_zero_end
        self._grad_zero_end = grad_zero_end
        self.gamma = gamma
        self.rf_delay_preload = rf_delay_preload
        self.addresses_per_grad_sample = addresses_per_grad_sample
        self.grad_pad = grad_pad
        self.adc_pad = adc_pad
        self.rf_pad_type = rf_pad_type
        self.grad_pad_type = grad_pad_type
        self._rx_div = None
        self._rx_t = None
        self._tx_warmup_samples = int(tx_warmup / self._tx_t + ROUNDING)

        # gammaSTAR dictionary storage
        self._blocks = {}
        self._rf_events = {}
        self._grad_events = {}
        self._adc_events = {}
        self._delay_events = {}
        self._shapes = {}

        self._var_names = (
            "tx0",
            "grad_vx",
            "grad_vy",
            "grad_vz",
            "grad_vz2",
            "rx0_en",
            "tx_gate",
        )

        self._rx_div = None
        self._rx_t = None
        self._tx_warmup_samples = int(tx_warmup / self._tx_t + ROUNDING)
        self._tx_div = int(tx_t / self._clk_t + ROUNDING)  # Clock cycles per tx
        self._tx_t = tx_t  # Transmit sample period in us
        self._grad_div = int(grad_t / self._clk_t + ROUNDING)  # Clock cycles per grad
        self._grad_t = grad_t  # Gradient sample period in us
        self._rx_div = None
        self._rx_t = None
        self._tx_warmup_samples = int(tx_warmup / self._tx_t + ROUNDING)
        self._grad_pad = grad_pad
        self.gamma = gamma
        self._grad_max = {}
        if gx_max is None:
            self._grad_max["gx"] = grad_max
        else:
            self._grad_max["gx"] = gx_max
        if gx_max is None:
            self._grad_max["gy"] = grad_max
        else:
            self._grad_max["gy"] = gy_max
        if gx_max is None:
            self._grad_max["gz"] = grad_max
        else:
            self._grad_max["gz"] = gz_max

        # Defined variable names to output
        self._var_names = (
            "tx0",
            "grad_vx",
            "grad_vy",
            "grad_vz",
            "grad_vz2",
            "rx0_en",
            "tx_gate",
        )

        # Interpolated and compiled data for output
        self._tx_durations = {}  # us
        self._tx_times = {}  # us
        self._tx_data = {}  # normalized float
        self._grad_durations = {}  # us
        self._grad_times = {}  # us
        self._grad_data = {}  # normalized float

        self.out_data = {}
        self.readout_number = 0
        self.is_assembled = False

        self.tx_arr = np.zeros(0, dtype=np.complex64)
        self.grad_arr = [np.zeros(0), np.zeros(0), np.zeros(0)]  # x, y, z

        self.tx_bytes = bytes()
        self.grad_bytes = [bytes(), bytes(), bytes()]  # x, y, z
        self.command_bytes = bytes()
        self.readout_number = 0
        self.is_assembled = False
        self.sequence_start = {}
        self._sequence_duration = {}

        self.BlockDurationRaster = 1e-07
        self._total_duration = None
        self.GradientRasterTime = 1e-05
        self.RadiofrequencyRasterTime = 1e-06

        self.raw_rep_list = raw_rep_list
        self.data_objects = self._create_structure_data_list(raw_rep_list)
        self._grad_events: Dict[str, List[Any]] = {}

        self._logger = logging.getLogger(log_file)
        self._logger.setLevel(log_level)
        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def _create_structure_data_list(
        self, data_list: List[Dict[str, Any]]
    ) -> List["GSAssembler.StructureData"]:
        structure_objects = []
        for idx, item in enumerate(data_list, start=1):
            if isinstance(item, dict):
                structure_objects.append(self.StructureData(item))
            else:
                self._logger.warning(
                    f"Skipping invalid item at index {idx} (not a dict)"
                )
        return structure_objects

    class StructureData:

        def __init__(self, data: Dict[str, Any]):
            self.data = data

        def get_key(self, key: str, missing_value: Any = None) -> Any:
            return self.data.get(key, missing_value)

        def get_rf_id(self) -> Any:
            return self.get_key("rf_id")

        def get_duration(self) -> Any:
            return self.get_key("duration")

        def get_abs_start(self) -> Any:
            return self.get_key("tstart")

        def _convert_grad_dict(self, grad_dict: Any) -> List[Any]:
            if not isinstance(grad_dict, dict):
                return []
            sorted_items = sorted(grad_dict.items(), key=lambda x: int(x[0]))
            return [v for _, v in sorted_items]

        def _convert_rf_am_fm(self):
            rf_v = self.get_key("rf_v", {})
            rf_entry = rf_v.get("1", {})

            am_dict = rf_entry.get("am", {})
            fm_dict = rf_entry.get("fm", {})

            def convert(sub_dict):
                if not isinstance(sub_dict, dict):
                    return []
                sorted_items = sorted(sub_dict.items(), key=lambda x: int(x[0]))
                return [v for _, v in sorted_items]

            am_list = convert(am_dict)
            fm_list = convert(fm_dict)
            print(fm_list)

            return am_list, fm_list

        def _convert_dictADC(self, dict3: Any) -> List[Any]:
            if not isinstance(dict3, dict):
                return []
            sorted_items = sorted(dict3.items(), key=lambda x: int(x[0]))
            return [v for _, v in sorted_items]

        def get_gradients_events(self) -> Dict[str, Any]:
            return {
                "gradx_t": self._convert_grad_dict(self.get_key("gradx_t", {})),
                "grady_t": self._convert_grad_dict(self.get_key("grady_t", {})),
                "gradz_t": self._convert_grad_dict(self.get_key("gradz_t", {})),
                "gradx_v": self._convert_grad_dict(self.get_key("gradx_v", {})),
                "grady_v": self._convert_grad_dict(self.get_key("grady_v", {})),
                "gradz_v": self._convert_grad_dict(self.get_key("gradz_v", {})),
                "gradx_tstart": self.get_key("gradx_tstart", 0),
                "grady_tstart": self.get_key("grady_tstart", 0),
                "gradz_tstart": self.get_key("gradz_tstart", 0),
            }

        def get_rf_events(self) -> Dict[str, Any]:
            am_list, fm_list = self._convert_rf_am_fm()
            return {
                "rf_id": self.get_key("rf_id"),
                "rf_am": am_list,
                "rf_tstart": self.get_key("rf_tstart"),
                "rf_dur": self.get_key("rf_dur"),
                "rf_t": self.get_key("rf_t"),
                "rf_fm": fm_list,
                "rf_frq": self.get_key("rf_frq"),
                "rf_phs": self.get_key("rf_phs"),
                "rf_asym": self.get_key("rf_asym"),
                "rf_type": self.get_key("rf_type"),
            }

        def get_adc_events(self) -> Dict[str, Any]:
            adc_events = {}
            adc_bool = self.get_key("has_adc")
            rx_id = 1 if adc_bool else 0
            adc_events["rx_id"] = rx_id

            if adc_bool:
                adc_tstart = self.get_key("adc_tstart")
                if adc_tstart is not None:
                    adc_events["adc_tstart"] = adc_tstart

                adc_frq = self.get_key("adc_frq")
                if adc_frq is not None:
                    adc_events["adc_frq"] = adc_frq

                adc_phs = self.get_key("adc_phs")
                if adc_phs is not None:
                    adc_events["adc_phs"] = adc_phs

                adc_header = self.get_key("adc_header")
                if adc_header is not None:
                    sample_time_us = adc_header.get("sample_time_us")
                    if sample_time_us is not None:
                        adc_events["sample_time_us"] = sample_time_us

                    number_of_samples = adc_header.get("number_of_samples")
                    if number_of_samples is not None:
                        adc_events["number_of_samples"] = number_of_samples

                    position = self._convert_dictADC(adc_header.get("position"))
                    read_dir = self._convert_dictADC(adc_header.get("read_dir"))
                    slice_dir = self._convert_dictADC(adc_header.get("slice_dir"))
                    phase_dir = self._convert_dictADC(adc_header.get("phase_dir"))

                    adc_events["position"] = position
                    adc_events["read_dir"] = read_dir
                    adc_events["slice_dir"] = slice_dir
                    adc_events["phase_dir"] = phase_dir

            return adc_events

        def summarize(self) -> Dict[str, Any]:
            adc_bool = self.get_key("has_adc")
            if adc_bool:
                summary = {
                    "duration": self.get_duration(),
                    "rf_events": self.get_rf_events(),
                    "gradients_events": self.get_gradients_events(),
                    "adc_events": self.get_adc_events(),
                }
            else:
                summary = {
                    "duration": self.get_duration(),
                    "rf_events": self.get_rf_events(),
                    "gradients_events": self.get_gradients_events(),
                }
            return summary

    def check_rf_ids(self) -> set:
        """
        Processes the file and stores unique occurrences of rf_id values in a list.
        """
        unique_rf_ids = set()
        self._blocks = {}
        for line_number, structure_data in enumerate(self.data_objects, start=1):
            rf_dict = structure_data.get_rf_events()
            rf_id = rf_dict.get("rf_id")
            self._blocks[line_number] = rf_id
            if rf_id not in unique_rf_ids:
                unique_rf_ids.add(rf_id)
        return unique_rf_ids

    def get_rx_blocks(self) -> None:
        self._adc_blocks = {}
        for line_number, structure_data in enumerate(self.data_objects, start=1):
            adc_dict = structure_data.get_adc_events()
            rx_id = adc_dict.get("rx_id")
            self._adc_blocks[line_number] = rx_id

    def _parse_rf_entries(self) -> None:

        var_names = (
            "rf_id",
            "rf_am",
            "rf_tstart",
            "rf_dur",
            "rf_t",
            "rf_frq",
            "rf_phs",
        )
        unique_rf_ids = self.check_rf_ids()

        for structure_data in self.data_objects:
            rf_dict = structure_data.get_rf_events()
            rf_id = rf_dict.get("rf_id")
            if rf_id > 0 and rf_id not in self._rf_events:
                self._rf_events[rf_id] = rf_dict

        self._logger.info("RF: Complete")

    def _parse_grad_entries(self):

        for structure_data in self.data_objects:
            grad_dict = structure_data.get_gradients_events()
            for grad_event, value in grad_dict.items():
                if value is not None:
                    if grad_event not in self._grad_events:
                        self._grad_events[grad_event] = []
                    self._grad_events[grad_event].append(value)

        self._logger.info("GRAD: Complete")

    def _parse_adc_entries(self):

        for structure_data in self.data_objects:
            adc_dict = structure_data.get_adc_events()
            if adc_dict.get("rx_id") == 1:
                filtered_dict = {k: v for k, v in adc_dict.items() if k != "rx_id"}
                self._adc_events[adc_dict.get("rx_id")] = filtered_dict

        self._logger.info("ADC: Complete")

    def _read_abs_start(self):
        abs_start = []
        for structure_data in self.data_objects:
            abs_start.append(structure_data.get_abs_start())
        return abs_start

    def _read_duration(self):
        duration_list = []
        for structure_data in self.data_objects:
            duration_list.append(
                structure_data.get_duration() * self.BlockDurationRaster * 1e6
            )
        return duration_list

    def _error_if(self, condition, message):
        """Logs an error and raises an exception if the condition is True."""
        if condition:
            self._logger.error(message)
            raise ValueError(message)

    def get_equidistant_points_from_samples_and_duration(self, samples, duration):
        array = []
        dwell = duration / samples
        for i in range(1, samples + 1):
            array.append((i - 0.5) * dwell)
        return array

    def calculate_equipoints(self):
        samples = self._adc_events["samples"]
        rf_duration = self.self._rf_events["rf_dur"]
        equi_array = self.get_equidistant_points_from_samples_and_duration(
            samples, rf_duration
        )
        return equi_array

    def reduce_angle_to_pi_interval(self, angle, symmetric_result):
        result = angle

        if symmetric_result:
            if abs(result) > math.pi:
                number = math.floor((abs(result) + math.pi) / (2 * math.pi))
                result -= result / abs(result) * (number * 2 * math.pi)
            if result == -math.pi:
                result = math.pi
        else:
            if result >= 2 * math.pi:
                number = math.floor(result / (2 * math.pi))
                result -= number * 2 * math.pi
            if result < 0:
                number = math.ceil(abs(result) / (2 * math.pi))
                result += number * 2 * math.pi

        return result

    def process_amplitude_phase(self, amT, t, fm, duration, consider_am):

        samples = len(amT)
        if len(t) == 0:
            t = self.get_equidistant_points_from_samples_and_duration(samples, duration)

        dwell = 0.5 * (t[0] + t[1]) * 1e-6
        am, phase = [], [2 * math.pi * (fm[0] if fm else 0) * dwell]

        for i in range(1, samples - 1):
            dwell = 0.5 * (t[i + 1] - t[i - 1]) * 1e-6
            phase.append(
                phase[i - 1] + 2 * math.pi * (fm[i] if i < len(fm) else 0) * dwell
            )

        dwell = (duration - 0.5 * (t[samples - 2] + t[samples - 1])) * 1e-6
        phase.append(
            phase[samples - 2]
            + 2 * math.pi * (fm[samples - 2] if (samples - 2) < len(fm) else 0) * dwell
        )

        if consider_am:
            for i, v in enumerate(amT):
                am.append(v)
                if v < 0:
                    am[i] = am[i]
                    phase[i] += math.pi
        else:
            for v in amT:
                am.append(v)

        for i in range(samples):
            phase[i] = self.reduce_angle_to_pi_interval(phase[i], True)  #

        return am, phase

    def calculate_am_and_phase(self):
        self.amp_calc = {}
        self.phase = {}
        for tx_id, tx_event in self._rf_events.items():

            amT = tx_event.get("rf_am", [])
            t = tx_event.get("rf_t", [])
            tx_event.get("rf_t", [])
            fm = tx_event.get("rf_fm", [])
            duration = tx_event.get("rf_dur", [])
            consider_am = False

            amp, phase = self.process_amplitude_phase(amT, t, fm, duration, consider_am)
            self.amp_calc[tx_id] = amp
            self.phase[tx_id] = phase

        return self.amp_calc, self.phase

    def _assemble_transmit_data(self):

        self._logger.info("Assembling transmit data...")
        self.amp_calc, self.phase = self.calculate_am_and_phase()
        for tx_id, tx_event in self._rf_events.items():
            phase_shape = self.phase[tx_id]
            rf_am = self.amp_calc[tx_id]

            rf_am = [am * self.gamma * 1e-6 for am in rf_am]  # Hz

            self._error_if(
                len(rf_am) == 0,
                f"RF amplitude data is missing or empty for tx_id: {tx_id}",
            )

            self._error_if(
                len(rf_am) != len(phase_shape),
                f"Tx envelope of RF event {tx_id} has mismatched rf_am and phase lengths",
            )

            rf_duration = tx_event.get("rf_dur", [])

            if len(rf_am) != (rf_duration):

                event_len = rf_duration  # integer
                event_duration = event_len  # float

                rf_timings = np.linspace(0, (rf_duration * self._tx_t), num=len(rf_am))

                time_interpolated = np.linspace(
                    0, rf_duration * self._tx_t, rf_duration
                )

                interpolation_function = interp1d(rf_timings, rf_am, kind="linear")
                interpolation_phase = interp1d(rf_timings, phase_shape, kind="linear")
                amplitudes_interpolated = interpolation_function(time_interpolated)
                phase_interpolated = interpolation_phase(time_interpolated)

                mag = amplitudes_interpolated
                phase = phase_interpolated
                x = time_interpolated

            else:
                mag = rf_am
                phase = phase_shape
                event_len = len(rf_am)
                event_duration = event_len * self._tx_t
                x = np.arange(0, event_duration, self._tx_t)

            tx_env = []
            mag = np.array(mag) / self._rf_amp_max
            phase_rad = np.array(phase)

            complex_tx_env = (
                np.exp((phase_rad + tx_event.get("rf_phs") * np.pi) * 1j) * mag
            )
            tx_env.append(complex_tx_env)

            tx_env = np.array(tx_env)

            self._error_if(
                np.any(np.abs(tx_env) > 1.0),
                f"Magnitude of RF event {tx_id} is too large relative to RF max "
                f"{np.max(np.abs(tx_env))} > {self._rf_amp_max}",
            )

            if self._tx_zero_end:
                x = np.append(x, event_duration * self._tx_t)
                tx_env = np.append(tx_env, 0)

            rf_tstart = tx_event.get("rf_tstart", 0)
            self._tx_durations[tx_id] = event_duration + rf_tstart
            self._tx_times[tx_id] = x + rf_tstart
            self._tx_data[tx_id] = tx_env

        self._logger.info("Tx data compiled successfully.")

    def interpolate_grad(self, time, amp):
        """
        Interpolates the amplitude values over the given time duration.
        """

        if len(time) == 0 and len(amp) == 0:
            time_interp = np.array([])
            amp_interp = np.array([])
        else:

            time_interp = np.arange(time[0], time[-1], self._grad_t)
            amp_interp = np.interp(time_interp, time, amp)

        return (time_interp, amp_interp)

    def _assemble_gradients_data(self):
        """
        Interpolates gradient events and calculates their durations and times, considering the start time (tstart) as a list.
        Returns a dictionary with keys (grad_vx, grad_vy, grad_vz) and values as tuples containing the amplitude and times.
        """

        abs_start = self._read_abs_start()
        self.sequence_start = abs_start
        duration_list = self._read_duration()
        self._total_duration = abs_start[-1] + duration_list[-1]
        self._sequence_duration = duration_list

        gradients_events = self._grad_events

        # Convert from mT/m to Hz/m
        gradx_v_conv = [np.array(grad) for grad in gradients_events["gradx_v"]]
        grady_v_conv = [np.array(grad) for grad in gradients_events["grady_v"]]
        gradz_v_conv = [np.array(grad) for grad in gradients_events["gradz_v"]]

        if len(gradients_events["gradx_t"]) > 0:
            gradx_t = np.array(gradients_events["gradx_t"], dtype=object)
        if len(gradients_events["grady_t"]) > 0:
            grady_t = np.array(gradients_events["grady_t"], dtype=object)
        if len(gradients_events["gradz_t"]) > 0:
            gradz_t = np.array(gradients_events["gradz_t"], dtype=object)

        x_tstart = gradients_events["gradx_tstart"]
        y_tstart = gradients_events["grady_tstart"]
        z_tstart = gradients_events["gradz_tstart"]

        for event in range(len(gradx_t)):
            tvect = gradx_t[event]
            gradx_t[event] = [x_tstart[event] + tvect[i] for i in range(len(tvect))]

        for event in range(len(grady_t)):
            tvect = grady_t[event]
            grady_t[event] = [y_tstart[event] + tvect[i] for i in range(len(tvect))]

        for event in range(len(gradz_t)):
            tvect = gradz_t[event]
            gradz_t[event] = [z_tstart[event] + tvect[i] for i in range(len(tvect))]

        # removing one step from last event time to have additional space between subsequent event (20 second-10 seconds)
        for i in range(len(abs_start)):
            if len(gradx_t[i]) > 1:
                gradx_t[i][-1] -= self._grad_t
            if len(grady_t[i]) > 1:
                grady_t[i][-1] -= self._grad_t
            if len(gradz_t[i]) > 1:
                gradz_t[i][-1] -= self._grad_t

        duration = 0

        overflow_time_x = np.array([])
        overflow_time_y = np.array([])
        overflow_time_z = np.array([])
        overflow_grad_x = np.array([])
        overflow_grad_y = np.array([])
        overflow_grad_z = np.array([])

        for i in range(len(abs_start)):
            duration = abs_start[i]
            event_len = int(duration_list[i])
            event_duration = duration_list[i] / 10 * self._grad_t  # us

            time_x, amp_x = self.interpolate_grad(gradx_t[i], gradx_v_conv[i])
            time_y, amp_y = self.interpolate_grad(grady_t[i], grady_v_conv[i])
            time_z, amp_z = self.interpolate_grad(gradz_t[i], gradz_v_conv[i])

            # adding missing part from previous block that went over block duration
            time_x = np.concatenate((overflow_time_x, time_x))
            amp_x = np.concatenate((overflow_grad_x, amp_x))

            time_y = np.concatenate((overflow_time_y, time_y))
            amp_y = np.concatenate((overflow_grad_y, amp_y))

            time_z = np.concatenate((overflow_time_z, time_z))
            amp_z = np.concatenate((overflow_grad_z, amp_z))

            # If self._grad_zero_end is True and the arrays are not empty, append the end values
            if self._grad_zero_end:
                if time_x.shape[0] > 0:
                    time_x = np.append(time_x, time_x[-1] + self._grad_t)
                    amp_x = np.append(amp_x, 0)
                if time_y.shape[0] > 0:
                    time_y = np.append(time_y, time_y[-1] + self._grad_t)
                    amp_y = np.append(amp_y, 0)
                if time_z.shape[0] > 0:
                    time_z = np.append(time_z, time_z[-1] + self._grad_t)
                    amp_z = np.append(amp_z, 0)

            # removing and saving part that went over block duration
            if time_x.shape[0] > 0:
                overflow_grad_x = amp_x[time_x > event_len]
                overflow_time_x = time_x[time_x > event_len] - event_len
                amp_x = amp_x[time_x <= event_len]
                time_x = time_x[time_x <= event_len]

            if time_y.shape[0] > 0:
                overflow_grad_y = amp_y[time_y > event_len]
                overflow_time_y = time_y[time_y > event_len] - event_len
                amp_y = amp_y[time_y <= event_len]
                time_y = time_y[time_y <= event_len]

            if time_z.shape[0] > 0:
                overflow_grad_z = amp_z[time_z > event_len]
                overflow_time_z = time_z[time_z > event_len] - event_len
                amp_z = amp_z[time_z <= event_len]
                time_z = time_z[time_z <= event_len]

            # Convert lists to numpy arrays or create empty arrays if either is empty
            time_x = (np.array(time_x)) if len(time_x) > 0 else np.array([])
            amp_x = np.array(amp_x) if len(amp_x) > 0 else np.array([])
            time_y = (np.array(time_y)) if len(time_y) > 0 else np.array([])
            amp_y = np.array(amp_y) if len(amp_y) > 0 else np.array([])
            time_z = (np.array(time_z)) if len(time_z) > 0 else np.array([])
            amp_z = np.array(amp_z) if len(amp_z) > 0 else np.array([])

            # Convert from mT/m to Hz/m and scale based on maximum gradient
            amp_x = (amp_x * self.gamma * 1e-3) / self._grad_max["gx"]
            amp_y = (amp_y * self.gamma * 1e-3) / self._grad_max["gy"]
            amp_z = (amp_z * self.gamma * 1e-3) / self._grad_max["gz"]

            # Store as tuples of numpy arrays
            self._grad_times.setdefault("grad_vx", []).append(np.array(time_x))
            self._grad_times.setdefault("grad_vy", []).append(np.array(time_y))
            self._grad_times.setdefault("grad_vz", []).append(np.array(time_z))

            self._grad_data.setdefault("grad_vx", []).append(np.array(amp_x))
            self._grad_data.setdefault("grad_vy", []).append(np.array(amp_y))
            self._grad_data.setdefault("grad_vz", []).append(np.array(amp_z))

        for grad_id in ["grad_vx", "grad_vy", "grad_vz"]:
            self._grad_times[grad_id] = np.array(
                self._grad_times[grad_id], dtype=object
            )
            self._grad_data[grad_id] = np.array(self._grad_data[grad_id], dtype=object)

        self._logger.info("Grad data compiled successfully.")

    def check_constant_dwell(self):
        adc_event = self._adc_events[1]
        dwell = adc_event["sample_time_us"]  # dwell time in us
        self._rx_div = np.round(dwell / self._clk_t).astype(int)
        self._rx_t = self._clk_t * self._rx_div

    def _stream_block(self, block_id):
        """
        Encode block into sequential time updates

        Args:
            block_id (int): Block id key for block in object dict memory to be encoded

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            float: duration of the block
            int: readout count for the block
        """
        out_dict = {var: [] for var in self._var_names}
        readout_num = 0
        # duration = 0

        block = self._blocks[block_id]

        for var in self._var_names:
            out_dict[var] = (np.zeros(0, dtype=int),) * 2

        tx_id = self._blocks[block_id]
        if tx_id != 0:
            out_dict["tx0"] = (self._tx_times[tx_id], self._tx_data[tx_id])
            tx_gate_start = self._tx_times[tx_id][0] - self._tx_warmup
            self._error_if(
                tx_gate_start < 0,
                f"Tx warmup ({self._tx_warmup}) of RF event {tx_id} is longer than delay ({self._tx_times[tx_id][0]})",
            )
            out_dict["tx_gate"] = (
                np.array([tx_gate_start, self._tx_durations[tx_id]]),
                np.array([1, 0]),
            )

        for grad_id in ["grad_vx", "grad_vy", "grad_vz"]:
            time_array, amp_array = (
                self._grad_times[grad_id][block_id - 1],
                self._grad_data[grad_id][block_id - 1],
            )
            out_dict[grad_id] = (time_array, amp_array)

        rx_id = self._adc_blocks[block_id]
        if rx_id != 0:
            rx_event = self._adc_events[rx_id]
            rx_start = rx_event["adc_tstart"]
            rx_end = rx_start + rx_event["number_of_samples"] * self._rx_t
            readout_num += rx_event["number_of_samples"]
            self.oversampling = 6
            self.readout_number = int(rx_event["number_of_samples"] / self.oversampling)

            out_dict["rx0_en"] = (np.array([rx_start, rx_end]), np.array([1, 0]))

        return (out_dict, int(readout_num))

    def _stream_all_blocks(self):
        """
        Encode all blocks into sequential time updates.

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            int: number of sequence readout points
        """

        out_data = {}
        times = {var: [np.zeros(1)] for var in self._var_names}
        updates = {var: [np.zeros(1)] for var in self._var_names}
        start = 0
        readout_total = 0
        duration = self._total_duration * self.BlockDurationRaster * 1e9

        for block_id in self._blocks.keys():
            self._logger.info(f"streaming block {block_id} ...")
            var_dict, readout_num = self._stream_block(block_id)
            start = self.sequence_start[block_id - 1]
            duration = self._sequence_duration[block_id - 1]

            for var in self._var_names:
                times[var].append(var_dict[var][0] + start)
                updates[var].append(var_dict[var][1])

            start += duration
            readout_total += readout_num

        for var in self._var_names:
            time_sorted, unique_idx = np.unique(
                np.flip(np.concatenate(times[var])), return_index=True
            )
            update_sorted = np.flip(np.concatenate(updates[var]))[unique_idx]

            update_compressed_idx = np.concatenate(
                [[0], np.nonzero(update_sorted[1:] - update_sorted[:-1])[0] + 1]
            )
            update_arr = update_sorted[update_compressed_idx]
            time_arr = time_sorted[update_compressed_idx]

            time_arr = np.concatenate((time_arr, np.zeros(1) + start))
            update_arr = np.concatenate((update_arr, np.zeros(1)))

            out_data[var] = (time_arr, update_arr)

        self._logger.info(f"done")
        param_dict = {
            "readout_number": self.readout_number,
            "tx_t": self._tx_t,
            "rx_t": self._rx_t,
            "grad_t": self._grad_t,
        }

        return (out_data, param_dict)

    def _warning_if(self, condition, message):
        """Logs a warning if the condition is True."""
        if condition:
            self._logger.warning(message)

    def interpret(self):
        """
        Interpret gammaSTAR file for MaRCoS server

        Args:
            gammaSTRAR raw data (str): gammaSTAR file to compile from

        Returns:
            dict: tuple of numpy.ndarray time and update arrays, with variable name keys
            dict: parameter dictionary containing raster times, readout numbers, and any file-defined variables
        """
        self._logger.info(f"Interpreting gammaSTAR sequence")
        if self.is_assembled:
            self._logger.info("Re-initializing over old sequence...")
            self.__init__(
                rf_center=self._rf_center,
                rf_amp_max=self._rf_amp_max,
                gx_max=self._grad_max["gx"],
                gy_max=self._grad_max["gy"],
                gz_max=self._grad_max["gz"],
                clk_t=self._clk_t,
                tx_t=self._tx_t,
                grad_t=self._grad_t,
            )
        self._parse_rf_entries()
        self._parse_grad_entries()
        self._parse_adc_entries()
        self._assemble_transmit_data()
        self._assemble_gradients_data()
        self.get_rx_blocks()
        self.check_constant_dwell()
        (self.out_data, param_dict) = self._stream_all_blocks()

        self.is_assembled = True
        param_dict = {
            "readout_number": self.readout_number,
            "tx_t": self._tx_t,
            "rx_t": self._rx_t,
            "grad_t": self._grad_t,
        }

        return (self.out_data, param_dict)


if __name__ == "__main__":
    gs = GSAssembler()

    out_data, params = gs.interpret()

    flat_delay = 0
    for buf in out_data.keys():
        out_data[buf] = (out_data[buf][0] + flat_delay, out_data[buf][1])

    names = [" tx", " gx", " gy", " gz", "adc"]
    data = [
        out_data["tx0"],
        out_data["grad_vx"],
        out_data["grad_vy"],
        out_data["grad_vz"],
        out_data["tx_gate"],
    ]

    # Generate plot
    fig, axs = plt.subplots(
        len(out_data), 1, figsize=(8, 6), constrained_layout=True, sharex=True
    )
    fig.suptitle(
        "gammaSTAR Interpreter", fontsize=18, fontweight="bold", color="darkblue"
    )

    # Define a colormap for better distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(out_data)))

    for i, (key, color) in enumerate(zip(out_data.keys(), colors)):
        axs[i].step(
            out_data[key][0],
            np.real(out_data[key][1]),
            where="post",
            linewidth=1.5,
            color=color,
        )
        axs[i].plot(
            out_data[key][0],
            np.real(out_data[key][1]),
            "o",
            markersize=3,
            color="red",
            label="Data Points",
        )

        axs[i].set_title(f"{key}", fontsize=12, fontweight="bold", color=color)
        axs[i].grid(True, linestyle="--", alpha=0.6)

    axs[-1].set_xlabel("Time (µs)", fontsize=12)
    plt.show()
