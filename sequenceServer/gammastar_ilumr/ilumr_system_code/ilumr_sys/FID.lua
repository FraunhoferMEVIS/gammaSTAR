local ffi = require("ffi")

local sequence_inst = require("sequence_inst")

local function main()
	local freq = 14133814
	local a_90 = 0.255
	local t_90 = 64e-6
	local p_90 = 0
	local t_acqdelay = 100e-6
	local t_dw = 1e-6
	local n_samples = 1000
	local t_end = 0.5
	local n_scans = 1

	local t_acq = t_dw * n_samples

	for i_scan = 1, n_scans do
		coroutine.yield(ffi.new("pulse_start", ffi.C.PULSESTART, freq, p_90, a_90))
		coroutine.yield(ffi.new("delay", ffi.C.DELAY, t_90))
		coroutine.yield(ffi.new("pulse_stop", ffi.C.PULSESTOP))

		coroutine.yield(ffi.new("delay", ffi.C.DELAY, t_acqdelay))

		coroutine.yield(ffi.new("acquire", ffi.C.ACQUIRE, freq, p_90, t_dw, n_samples))
		coroutine.yield(ffi.new("delay", ffi.C.DELAY, t_acq))

		coroutine.yield(ffi.new("delay", ffi.C.DELAY, t_end))
	end
end

return main