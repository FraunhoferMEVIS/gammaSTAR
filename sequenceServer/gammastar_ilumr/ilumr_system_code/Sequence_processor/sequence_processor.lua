-- sequence_processor.lua 
local M = {}

local ffi = require("ffi")
local bit = require("bit")
package.path = package.path .. ";../ilumr_sys/?.lua;../ilumr_sys/?/init.lua"

local sequence_inst = require("sequence_inst")
local yaml = require('yaml')
local inspect = require('inspect')


-- logging
local Logging = require "logging"
local log = Logging.defaultLogger()
log:setLevel(log.DEBUG)

-- Make sure FFI constants are available
ffi.C = ffi.C or {}

-- Constants 
local DT_MIN = 1e-8
local PULSE_DEC = 1
local POST_DEC = 1
local DEG_PER_RAD = 180 / math.pi
local FREQUENCY_FILE = "/home/globals/frequency.yaml"
local RF_90_PAR_FILE = "/home/globals/hardpulse_90.yaml"
--local RF_90_PAR_FILE = "/home/globals/softpulse_90_4.0mm.yaml"
local SHIMS_FILE = "/home/globals/shims.yaml"

-- Event masks 
local PULSE_EVENT_MASK = 0x001
local GRAD_EVENT_MASK = 0x010
local ACQ_EVENT_MASK = 0x100

-- GLOBAL OUTSIDE GRADIENTS (for handling overlapping gradients between blocks)
local outside_t_x, outside_g_x = {}, {}
local outside_t_y, outside_g_y = {}, {}
local outside_t_z, outside_g_z = {}, {}

-- Performance: Try to detect LuaJIT for table.new optimization
local table_new = table.new or function(narr, nrec) return {} end

-- NaN handling
local function isnan(x)
    return x ~= x
end

local function isfinite(x)
    return x == x and x ~= math.huge and x ~= -math.huge
end

function load_yaml(filename)
    local f = io.open(filename, "rb")
    local content = f:read("*all")
    f:close()
    return yaml.eval(content)
end

-- Debug-Ausgaben aktivieren/deaktivieren
local DUMP_VALUES = false

local function dump_pulse_arrays(t_pulse, f_pulse, p_pulse, a_pulse)
    if not DUMP_VALUES then return end
    print("RF Pulse samples:")
    for i = 1, #t_pulse do
        print(string.format("  %3d: t=%.9f s, f=%.3f Hz, phs=%.6f deg, amp=%.6f T",
            i, t_pulse[i] or 0, f_pulse[i] or 0, p_pulse[i] or 0, a_pulse[i] or 0))
    end
end

local function dump_acq_arrays(t_acq, f_acq, p_acq, t_dw, n_samples)
    if not DUMP_VALUES then return end
    print("ADC events:")
    for i = 1, #t_acq do
        print(string.format("  %3d: t=%.9f s, f=%.3f Hz, phs=%.6f deg, dw=%.9f s, samples=%d",
            i, t_acq[i] or 0, f_acq[i] or 0, p_acq[i] or 0, t_dw[i] or 0, n_samples[i] or 0))
    end
end

-- Memory Pool for FFI structures
local MemoryPool = {}
MemoryPool.__index = MemoryPool

function MemoryPool.new(struct_type, pool_size)
    local pool = {
        struct_type = struct_type,
        available = table_new(pool_size, 0),
        in_use = table_new(0, pool_size),
        pool_size = pool_size,
        created_count = 0
    }
    
    -- Pre-allocate structures
    for i = 1, pool.pool_size do
        pool.available[i] = ffi.new(struct_type)
        pool.created_count = pool.created_count + 1
    end
    
    return setmetatable(pool, MemoryPool)
end

function MemoryPool:get()
    if #self.available > 0 then
        local obj = self.available[#self.available]
        self.available[#self.available] = nil
        self.in_use[obj] = true
        return obj
    else
        -- Fallback: new allocation
        local obj = ffi.new(self.struct_type)
        self.in_use[obj] = true
        self.created_count = self.created_count + 1
        return obj
    end
end

function MemoryPool:release(obj)
    if self.in_use[obj] then
        self.in_use[obj] = nil
        self.available[#self.available + 1] = obj
    end
end

local delay_pool = MemoryPool.new("delay", 1)
local gradient_pool = MemoryPool.new("gradient", 1)
local pulse_start_pool = MemoryPool.new("pulse_start", 1)
local pulse_update_pool = MemoryPool.new("pulse_update", 1)
local pulse_stop_pool = MemoryPool.new("pulse_stop", 1)
local acquire_pool = MemoryPool.new("acquire", 1)
local init_pool = MemoryPool.new("init", 1)
local end_pool = MemoryPool.new("end", 1)
local shim_pool = MemoryPool.new("shim", 1)

-- Fast array concatenation
local function concat_multiple(...)
    local arrays = {...}
    local total_len = 0
    
    -- Calculate total length
    for _, arr in ipairs(arrays) do
        total_len = total_len + #arr
    end
    
    if total_len == 0 then
        return {}
    end
    
    -- Pre-allocate result array
    local result = table_new(total_len, 0)
    local idx = 1
    
    for _, arr in ipairs(arrays) do
        for i = 1, #arr do
            result[idx] = arr[i]
            idx = idx + 1
        end
    end
    
    return result
end

-- Fast interpolation with binary search for large arrays
local function interp1d_fast(x_valid, y_valid, x_target)
    local n = #x_valid
    if n < 2 then return y_valid[1] end
    
    -- Binary search for large arrays
    if n > 50 then
        local left, right = 1, n
        while right - left > 1 do
            local mid = bit.rshift(left + right, 1)  -- Fast division by 2
            if x_valid[mid] <= x_target then
                left = mid
            else
                right = mid
            end
        end
        
        -- Interpolation between left and right
        local dx = x_valid[right] - x_valid[left]
        if dx > 0 then
            local t = (x_target - x_valid[left]) / dx
            return y_valid[left] + t * (y_valid[right] - y_valid[left])
        else
            return y_valid[left]
        end
    else
        -- Linear search for small arrays
        for i = 1, n - 1 do
            if x_valid[i] <= x_target and x_target <= x_valid[i + 1] then
                local dx = x_valid[i + 1] - x_valid[i]
                if dx > 0 then
                    local t = (x_target - x_valid[i]) / dx
                    return y_valid[i] + t * (y_valid[i + 1] - y_valid[i])
                else
                    return y_valid[i]
                end
            end
        end
    end
    
    -- Extrapolation
    return x_target < x_valid[1] and y_valid[1] or y_valid[n]
end

-- Split gradients function
local function split_gradients(timings, amplitudes, breakvalue)
    local n = #timings
    if n == 0 then
        return {{}, {}}, {{}, {}}
    end
    
    local keep_timings = table_new(n, 0)
    local keep_amplitudes = table_new(n, 0)
    local overlap_timings = table_new(n, 0)
    local overlap_amplitudes = table_new(n, 0)
    local keep_count, overlap_count = 0, 0
    
    for i = 1, n do
        if timings[i] <= breakvalue then
            keep_count = keep_count + 1
            keep_timings[keep_count] = timings[i]
            keep_amplitudes[keep_count] = amplitudes[i]
        else
            overlap_count = overlap_count + 1
            overlap_timings[overlap_count] = timings[i] - breakvalue
            overlap_amplitudes[overlap_count] = amplitudes[i]
        end
    end
    
    return {keep_timings, keep_amplitudes}, {overlap_timings, overlap_amplitudes}
end

-- Sort with cached indices for multiple arrays
local function sort_with_cache(times, ...)
    local data_arrays = {...}
    local n = #times
    if n == 0 then
        return times, ...
    end
    
    local indices = table_new(n, 0)
    for i = 1, n do 
        indices[i] = i 
    end
    
    -- Single sort operation
    table.sort(indices, function(a, b) return times[a] < times[b] end)
    
    -- Apply sorting to all arrays
    local sorted_times = table_new(n, 0)
    local sorted_arrays = table_new(#data_arrays, 0)
    
    for j = 1, #data_arrays do
        sorted_arrays[j] = table_new(#data_arrays[j], 0)
    end
    
    for i = 1, n do
        local idx = indices[i]
        sorted_times[i] = times[idx]
        for j = 1, #data_arrays do
            sorted_arrays[j][i] = data_arrays[j][idx]
        end
    end
    
    return sorted_times, unpack(sorted_arrays)
end

-- Profiling wrapper (can be disabled for production)
local function profile_function(name, func, ...)
    -- Uncomment for profiling
    -- local zmq = require("zmq")
    -- local start = zmq.stopwatch_start()
    local result = {func(...)}
    -- local elapsed = start:stop()
    -- log:debug("PROFILE: %s took %.4f ms", name, elapsed/1000)
    return unpack(result)
end

-- Reset outside gradients function
function M.reset_outside_gradients()
    outside_t_x, outside_g_x = {}, {}
    outside_t_y, outside_g_y = {}, {}
    outside_t_z, outside_g_z = {}, {}
end

-- robust numeric parser for shims
local function parse_shim_value(v)
    if type(v) == "number" then
        return v
    elseif type(v) == "string" then
        -- normalize Unicode minus
        v = v:gsub("\226\128\147", "-")
        return tonumber(v) or 0
    elseif type(v) == "table" then
        -- YAML "- 0.16965" -> {0.16965} : interpret as negative scalar
        if #v == 1 and type(v[1]) == "number" then
            return -v[1]
        end
        -- YAML split sign (rare): {"-", "0.16965"}
        if #v == 2 and v[1] == "-" and v[2] then
            local n = tonumber(v[2])
            if n then return -n end
        end
        -- fallback: concatenate parts and parse
        local s = table.concat(v)
        s = s:gsub("\226\128\147", "-")
        return tonumber(s) or 0
    else
        return 0
    end
end

local function load_shims_from_file()
    local default_shims = {
        shim_x = 0, shim_y = 0, shim_z = 0, shim_z2 = 0,
        shim_zx = 0, shim_zy = 0, shim_xy = 0, shim_x2y2 = 0
    }

    local ok, shim_data = pcall(load_yaml, SHIMS_FILE)
    if not ok or not shim_data then
        log:warn("Could not load %s, using defaults", SHIMS_FILE)
        return default_shims
    end

    local shims = {}
    for _, name in ipairs({"shim_x","shim_y","shim_z","shim_z2","shim_zx","shim_zy","shim_xy","shim_x2y2"}) do
        local raw = shim_data[name]
        local v = parse_shim_value(raw)
        if type(raw) == "table" and #raw == 1 then
            log:debug("%s parsed from one-item list -> negative: %.6f", name, v)
        end
        if v < -1 or v > 1 then
            log:warn("Shim %s out of range [-1,1]: %.6f, clamping", name, v)
            v = math.max(-1, math.min(1, v))
        end
        shims[name] = v
    end
    return shims
end

-- Create shim command
local function create_shim_command(params)
    local shim_cmd = shim_pool:get()
    shim_cmd.cmd = ffi.C.SHIM
    
    -- Set shim values in hardware order
    shim_cmd.shim[0] = params.shim_x
    shim_cmd.shim[1] = params.shim_y
    shim_cmd.shim[2] = params.shim_z
    shim_cmd.shim[3] = params.shim_z2
    shim_cmd.shim[4] = params.shim_zx
    shim_cmd.shim[5] = params.shim_zy
    shim_cmd.shim[6] = params.shim_xy
    shim_cmd.shim[7] = params.shim_x2y2
    
    return shim_cmd
end

-- Get sequence parameters
function M.get_sequence_params(sequence_data)
    local utils = require("utils")
    
    -- load frequency in Hz  
    local frequency = load_yaml(FREQUENCY_FILE).f

    -- load rf calibration in Tesla
    local rf_90_par = load_yaml(RF_90_PAR_FILE)
    local rf_cal = (0.25/(rf_90_par.a_90*rf_90_par.t_90))/42.58e6
    print("rf_cal:", rf_cal)
    
    -- load shims from YAML file
    local default_shims = load_shims_from_file()
    
    local params = {
        f = frequency,
        rf_cal = rf_cal,
        g_cal = {0.2, 0.2, 0.15},
        amp_enable = 1,
        rx_gain = 7,
        -- Shim parameters from file
        shim_x = default_shims.shim_x,
        shim_y = default_shims.shim_y,
        shim_z = default_shims.shim_z,
        shim_z2 = default_shims.shim_z2,
        shim_zx = default_shims.shim_zx,
        shim_zy = default_shims.shim_zy,
        shim_xy = default_shims.shim_xy,
        shim_x2y2 = default_shims.shim_x2y2
    }
    
    -- Override with sequence data if provided
    if sequence_data and sequence_data.params then
        for k, v in pairs(sequence_data.params) do
            if k:match("^shim_") then
                -- Additional validation for override shim parameters
                if type(v) == "number" and v >= -1 and v <= 1 then
                    params[k] = v
                else
                    log:warn("Invalid override shim value for %s: %s, keeping default %.3f", 
                            k, tostring(v), params[k])
                end
            else
                params[k] = v
            end
        end
    end
     
    return params
end

-- Validate sequence data
function M.validate_sequence_data(sequence_data)
    if not sequence_data then
        return false, "No sequence data provided"
    end
    
    local blocks = sequence_data.blocks or sequence_data
    if type(blocks) ~= "table" or #blocks == 0 then
        return false, "No valid blocks found in sequence data"
    end

    return true, "Sequence validation passed"
end

-- Preprocess acquisition events
local function preprocess_acquisition(block, f)
    if not block.has_adc then
        return {}, {}, {}, {}, {}
    end
    
    local t_acq = {1e-6 * block.adc_tstart}
    local f_acq = {f + block.adc_frq}
    local p_acq = {DEG_PER_RAD * block.adc_phs}
    local n_samples = {POST_DEC * block.adc_header.number_of_samples}
    local t_dw = {1e-6 * block.adc_header.sample_time_us / POST_DEC}
    
    -- Debug-Ausgabe
    dump_acq_arrays(t_acq, f_acq, p_acq, t_dw, n_samples)

    return t_acq, f_acq, p_acq, t_dw, n_samples
end

-- Extract times and values helper
local function extract_times_values_fast(t_data, v_data, tstart)
    local times, values = table_new(100, 0), table_new(100, 0)
    local t_count, v_count = 0, 0
    local offset = tstart or 0
    
    if type(t_data) == "table" then
        if #t_data > 0 then
            -- Array format 
            for i = 1, #t_data do
                t_count = t_count + 1
                times[t_count] = t_data[i] + offset
            end
        else
            -- Dict format - convert to array
            local keys = table_new(50, 0)
            local key_count = 0
            for k in pairs(t_data) do 
                key_count = key_count + 1
                keys[key_count] = tonumber(k)
            end
            table.sort(keys)
            for i = 1, key_count do
                t_count = t_count + 1
                times[t_count] = t_data[tostring(keys[i])] + offset
            end
        end
    end
    
    if type(v_data) == "table" then
        if #v_data > 0 then
            -- Array format
            for i = 1, #v_data do
                v_count = v_count + 1
                values[v_count] = v_data[i]
            end
        else
            -- Dict format
            local keys = table_new(50, 0)
            local key_count = 0
            for k in pairs(v_data) do 
                key_count = key_count + 1
                keys[key_count] = tonumber(k)
            end
            table.sort(keys)
            for i = 1, key_count do
                v_count = v_count + 1
                values[v_count] = v_data[tostring(keys[i])]
            end
        end
    end
    
    return times, values
end

-- Rounding function for consistent behavior
local function round(num, decimals)
    local mult = 10^(decimals or 0)
    return math.floor(num * mult + 0.5) / mult
end

-- Faster preprocess_pulse: minimal copying, direct filling, same output semantics
local function preprocess_pulse(block, f, rf_cal)
    -- Map rf_id -> rf_pulse_id (Python compatibility)
    if block.rf_id then block.rf_pulse_id = block.rf_id end
    if not block.rf_pulse_id or block.rf_pulse_id == 0 then
        return {}, {}, {}, {}
    end

    local rf = block.rf_v and block.rf_v["1"]
    if not rf or type(rf.am) ~= "table" then
        return {}, {}, {}, {}
    end

    -- Helper: copy table values (array or dict) into a numeric array in order
    local function collect_values_sorted(tbl)
        if type(tbl) ~= "table" then return {} end
        if #tbl > 0 then
            local out = table_new(#tbl, 0)
            for i = 1, #tbl do out[i] = tbl[i] end
            return out
        else
            local keys, c = {}, 0
            for k in pairs(tbl) do
                local nk = tonumber(k)
                if nk then
                    c = c + 1
                    keys[c] = nk
                end
            end
            table.sort(keys)
            local out = table_new(c, 0)
            for i = 1, c do
                out[i] = tbl[tostring(keys[i])]
            end
            return out
        end
    end

    -- Extract am/fm
    local am = collect_values_sorted(rf.am)
    local N = #am
    if N == 0 then
        return {}, {}, {}, {}
    end

    local fm = rf.fm and collect_values_sorted(rf.fm) or nil
    if not fm then
        fm = table_new(N, 0)
        for i = 1, N do fm[i] = 0 end
    elseif #fm < N then
        for i = #fm + 1, N do fm[i] = 0 end
    elseif #fm > N then
        local trimmed = table_new(N, 0)
        for i = 1, N do trimmed[i] = fm[i] end
        fm = trimmed
    end

    -- rf timing
    local rf_t = (type(block.rf_t) == "table") and collect_values_sorted(block.rf_t) or nil
    local rf_frq    = block.rf_frq    or 0
    local rf_phs    = block.rf_phs    or 0
    local rf_tstart = block.rf_tstart or 0
    local rf_dur    = block.rf_dur    or 0

    local base_freq  = f + rf_frq
    local base_phase = DEG_PER_RAD * rf_phs
    local inv_rf_cal = 1e-6 / rf_cal
    local step       = (PULSE_DEC > 1) and PULSE_DEC or 1
    local Nd         = math.floor((N + step - 1) / step) -- ceil(N/step)

    -- Uniform rf_t fallback if rf_t missing
    local uniform_start_us = 5
    local uniform_end_us   = math.max(5, rf_dur - 5)
    local uniform_step_us  = (N > 1) and ((uniform_end_us - uniform_start_us) / (N - 1)) or 0

    -- Allocate outputs (Nd samples + endpoint)
    local out_len = Nd + 1
    local t_pulse = table_new(out_len, 0)
    local f_pulse = table_new(out_len, 0)
    local p_pulse = table_new(out_len, 0)
    local a_pulse = table_new(out_len, 0)

    -- Fill decimated samples
    local oi, src = 1, 1
    while oi <= Nd do
        if src > N then src = N end

        local t_us
        if rf_t and rf_t[src] then
            t_us = rf_t[src]
        else
            t_us = uniform_start_us + (src - 1) * uniform_step_us
        end
        t_pulse[oi] = 1e-6 * (t_us + rf_tstart)
        f_pulse[oi] = base_freq + (fm[src] or 0)
        p_pulse[oi] = base_phase
        a_pulse[oi] = inv_rf_cal * (am[src] or 0)

        oi  = oi + 1
        src = src + step
    end

    -- Endpoint: stop RF
    t_pulse[out_len] = 1e-6 * (rf_tstart + rf_dur)
    f_pulse[out_len] = f
    p_pulse[out_len] = base_phase
    a_pulse[out_len] = 0

    dump_pulse_arrays(t_pulse, f_pulse, p_pulse, a_pulse)
    return t_pulse, f_pulse, p_pulse, a_pulse
end

-- Preprocess gradients
local function preprocess_gradients(block, g_cal)
    -- Keep your constants as-is (they already define 10 us)
    local GRAD_DT_MIN_US = 10        -- us, min time between gradient updates
    local GRAD_DT_INTERP_US = 10     -- us, interpolation step  


    -- Extract gradient data for each axis
    local gradx_t = block.gradx_t or {}
    local grady_t = block.grady_t or {}
    local gradz_t = block.gradz_t or {}
    local gradx_v = block.gradx_v or {}
    local grady_v = block.grady_v or {}
    local gradz_v = block.gradz_v or {}
    
    local t_x, g_x_values = extract_times_values_fast(gradx_t, gradx_v, block.gradx_tstart)
    local t_y, g_y_values = extract_times_values_fast(grady_t, grady_v, block.grady_tstart)
    local t_z, g_z_values = extract_times_values_fast(gradz_t, gradz_v, block.gradz_tstart)
       
    -- Create gradient matrices
    local function create_gradient_matrix_fast(times, values, axis_index)
        local n = #times
        if n == 0 then
            return {0, block.duration}, {{0/0, 0/0, 0/0}, {0/0, 0/0, 0/0}} 
        else
            local matrix = table_new(n, 0)
            for i = 1, n do
                matrix[i] = {0/0, 0/0, 0/0}                 -- Initialize with NaN
                matrix[i][axis_index] = values[i] or 0
            end
            return times, matrix
        end
    end
    
    local t_x_mat, g_x = create_gradient_matrix_fast(t_x, g_x_values, 1)
    local t_y_mat, g_y = create_gradient_matrix_fast(t_y, g_y_values, 2)
    local t_z_mat, g_z = create_gradient_matrix_fast(t_z, g_z_values, 3)
    
    -- Handle outside gradients
    local function ensure_matrix_dimensions(outside_g, new_g)
        if #outside_g == 0 and #new_g > 0 then
            return {}
        end
        return outside_g
    end
    
    outside_g_x = ensure_matrix_dimensions(outside_g_x, g_x)
    outside_g_y = ensure_matrix_dimensions(outside_g_y, g_y) 
    outside_g_z = ensure_matrix_dimensions(outside_g_z, g_z)
    
    -- Concatenate outside gradients
    t_x_mat = concat_multiple(outside_t_x, t_x_mat)
    g_x = concat_multiple(outside_g_x, g_x)
    t_y_mat = concat_multiple(outside_t_y, t_y_mat)
    g_y = concat_multiple(outside_g_y, g_y)
    t_z_mat = concat_multiple(outside_t_z, t_z_mat)
    g_z = concat_multiple(outside_g_z, g_z)

    -- Split gradients
    local t_event = block.duration
    local keep_x, outside_x = split_gradients(t_x_mat, g_x, t_event)
    local keep_y, outside_y = split_gradients(t_y_mat, g_y, t_event)
    local keep_z, outside_z = split_gradients(t_z_mat, g_z, t_event)
    
    -- Update outside gradients for next block
    outside_t_x, outside_g_x = outside_x[1], outside_x[2]
    outside_t_y, outside_g_y = outside_y[1], outside_y[2]
    outside_t_z, outside_g_z = outside_z[1], outside_z[2]
    
    -- Use kept gradients for this block
    t_x_mat, g_x = keep_x[1], keep_x[2]
    t_y_mat, g_y = keep_y[1], keep_y[2]
    t_z_mat, g_z = keep_z[1], keep_z[2]
    
    -- Combine gradients
    local t = concat_multiple(t_x_mat, t_y_mat, t_z_mat)
    local g = concat_multiple(g_x, g_y, g_z)
    
    -- Sort by time
    local sorted_t, sorted_g = sort_with_cache(t, g)

    -- Forward-Fill of NaNs with previous values (Start with 0)
    local function fill_nan_with_prev(curr, prev)
        if not prev then prev = {0, 0, 0} end
        for axis = 1, 3 do
            if not isfinite(curr[axis]) then
                curr[axis] = prev[axis]
            end
        end
        return curr
    end

    -- Build final arrays with forward fill and replace double values
    local final_t, final_g = table_new(#sorted_t, 0), table_new(#sorted_g, 0)
    local final_count = 0

    if #sorted_t > 0 then
        final_count = 1
        final_t[1] = sorted_t[1]
        final_g[1] = fill_nan_with_prev(sorted_g[1], {0, 0, 0})

        for i = 2, #sorted_t do
            local dt = sorted_t[i] - sorted_t[i - 1]
            local curr = fill_nan_with_prev(sorted_g[i], final_g[final_count])
            if dt < DT_MIN then
                -- same timestamp: merge
                final_g[final_count] = curr
            else
                final_count = final_count + 1
                final_t[final_count] = sorted_t[i]
                final_g[final_count] = curr
            end
        end
    end

    -- Fallback: if no actual value set all gradients to 0
    for axis = 1, 3 do
        local has_valid = false
        for i = 1, final_count do
            if isfinite(final_g[i][axis]) then
                has_valid = true
                break
            end
        end
        if not has_valid then
            for i = 1, final_count do
                final_g[i][axis] = 0
            end
        end
    end
    
    -- Gradient ramp interpolation (REPLACED)
    local t_prev = 0
    local g_prev = {0, 0, 0}
    local grad_events = table_new(final_count * 10, 0)
    local event_count = 0

    local function push_grad_event(t_us, gx, gy, gz)
        event_count = event_count + 1
        grad_events[event_count] = {
            time = 1e-6 * t_us,
            grad = {gx, gy, gz}
        }
    end

    for i = 1, final_count do
        local g_i = final_g[i]
        local t_i = final_t[i]             -- in µs
        local changed = (g_i[1] ~= g_prev[1]) or (g_i[2] ~= g_prev[2]) or (g_i[3] ~= g_prev[3])

        -- Hardware conversion helper
        local function convert_hw(gx_mTm, gy_mTm, gz_mTm)
            local gx = 1e-3 * gx_mTm / g_cal[1]
            local gy = 1e-3 * gy_mTm / g_cal[2]
            local gz = 1e-3 * gz_mTm / g_cal[3]
            if math.abs(gx) > 1 or math.abs(gy) > 1 or math.abs(gz) > 1 then
                error('Warning: Gradient too large!')
            end
            return gx, gy, gz
        end

        if changed then
            local dt_us = t_i - t_prev
            local step_us = GRAD_DT_INTERP_US
            local t_us = t_prev

            -- step points every 10 µs, strictly enforcing min spacing
            while t_us + step_us <= t_i do
                t_us = t_us + step_us
                local frac = dt_us > 0 and (t_us - t_prev) / dt_us or 1.0

                local g_interp_x = g_prev[1] + frac * (g_i[1] - g_prev[1])
                local g_interp_y = g_prev[2] + frac * (g_i[2] - g_prev[2])
                local g_interp_z = g_prev[3] + frac * (g_i[3] - g_prev[3])

                local gx, gy, gz = convert_hw(g_interp_x, g_interp_y, g_interp_z)
                push_grad_event(t_us, gx, gy, gz)
            end

            -- Endpoint handling: include t_i only if the last step is ≥ min spacing
            -- Otherwise, update the last event's grad to the final value to keep spacing ≥ 10 µs
            local gx_end, gy_end, gz_end = convert_hw(g_i[1], g_i[2], g_i[3])
            if event_count == 0 or (t_i - (grad_events[event_count].time * 1e6)) >= GRAD_DT_MIN_US - 1e-9 then
                push_grad_event(t_i, gx_end, gy_end, gz_end)
            else
                grad_events[event_count].grad[1] = gx_end
                grad_events[event_count].grad[2] = gy_end
                grad_events[event_count].grad[3] = gz_end
            end

        elseif t_prev > 0 then
            -- No change: optionally keep a single point at t_prev (plateau marker)
            local gx, gy, gz = convert_hw(g_prev[1], g_prev[2], g_prev[3])
            push_grad_event(t_prev, gx, gy, gz)
        end

        t_prev = t_i
        g_prev[1], g_prev[2], g_prev[3] = g_i[1], g_i[2], g_i[3]
    end

    -- OPTIONAL final spacing filter to guarantee ≥10 µs between any gradient commands
    local function enforce_min_spacing(events, min_us)
        if #events <= 1 then return events end
        local out = table_new(#events, 0)
        local last_t = -1e12
        for i = 1, #events do
            local t_us = events[i].time * 1e6
            if (t_us - last_t) >= (min_us - 1e-9) then
                out[#out+1] = events[i]
                last_t = t_us
            else
                -- merge: update the last event’s grad to the newer value
                local last = out[#out]
                last.grad[1] = events[i].grad[1]
                last.grad[2] = events[i].grad[2]
                last.grad[3] = events[i].grad[3]
            end
        end
        return out
    end

    grad_events = enforce_min_spacing(grad_events, GRAD_DT_MIN_US)

    return grad_events
end

-- Main block preprocessing 
local function preprocess_block(block, f, rf_cal, g_cal)
    local t_pulse, f_pulse, p_pulse, a_pulse = profile_function("preprocess_pulse", preprocess_pulse, block, f, rf_cal)
    local t_acq, f_acq, p_acq, t_dw, n_samples = profile_function("preprocess_acquisition", preprocess_acquisition, block, f)
    local grad_events = profile_function("preprocess_gradients", preprocess_gradients, block, g_cal)
    
    -- Create time ordered array with flags 
    local total_events = #t_pulse + #grad_events + #t_acq
    local all_times = table_new(total_events, 0)
    local all_flags = table_new(total_events, 0)
    local event_count = 0
    
    -- Add pulse times with flags
    for i = 1, #t_pulse do
        event_count = event_count + 1
        all_times[event_count] = t_pulse[i]
        all_flags[event_count] = PULSE_EVENT_MASK
    end
    
    -- Add gradient times with flags  
    for i = 1, #grad_events do
        event_count = event_count + 1
        all_times[event_count] = grad_events[i].time
        all_flags[event_count] = GRAD_EVENT_MASK
    end
    
    -- Add acquisition times with flags
    for i = 1, #t_acq do
        event_count = event_count + 1
        all_times[event_count] = t_acq[i]
        all_flags[event_count] = ACQ_EVENT_MASK
    end
    
    -- Sort by time 
    local sorted_times, sorted_flags = sort_with_cache(all_times, all_flags)
    
    -- Remove duplicated time values
    local final_times, final_flags = table_new(#sorted_times, 0), table_new(#sorted_flags, 0)
    local final_count = 0
    
    if #sorted_times > 0 then
        final_count = 1
        final_times[1] = sorted_times[1]
        final_flags[1] = sorted_flags[1]
        
        for i = 2, #sorted_times do
            local dt = sorted_times[i] - sorted_times[i-1]
            if dt < DT_MIN then
                -- Bitwise OR flags (matching Python)
                final_flags[final_count] = bit.bor(final_flags[final_count], sorted_flags[i])
            else
                final_count = final_count + 1
                final_times[final_count] = sorted_times[i]
                final_flags[final_count] = sorted_flags[i]
            end
        end
    end
    
    return final_times, final_flags, f_pulse, p_pulse, a_pulse, grad_events, f_acq, p_acq, t_dw, n_samples
end

-- Iterator helpers
local function create_pulse_iterator(f_pulse, p_pulse, a_pulse)
    local i = 0
    local n = #f_pulse
    return function()
        if i < n then
            i = i + 1
            return i - 1, {f_pulse[i], p_pulse[i], a_pulse[i]}
        end
        return nil
    end
end

local function create_grad_iterator(grad_events)
    local i = 0
    local n = #grad_events
    return function()
        if i < n then
            i = i + 1
            return grad_events[i].grad
        end
        return nil
    end
end

local function create_acq_iterator(f_acq, p_acq, t_dw, n_samples)
    local i = 0
    local n = #f_acq
    return function()
        if i < n then
            i = i + 1
            return {f_acq[i], p_acq[i], t_dw[i], n_samples[i]}
        end
        return nil
    end
end

-- Create sequence generator 
function M.create_generator(sequence_data)
    return coroutine.wrap(function()
        local params = M.get_sequence_params(sequence_data)
        local blocks = sequence_data.blocks or sequence_data
        
        -- Generate INIT command
        local init_cmd = init_pool:get()
        init_cmd.cmd = ffi.C.INIT
        init_cmd.amp_enable = params.amp_enable or 1
        init_cmd.rx_gain = params.rx_gain or 7
        coroutine.yield(init_cmd)
        init_pool:release(init_cmd)
        
        -- Apply shims before sequence
        local shim_cmd = create_shim_command(params)
        coroutine.yield(shim_cmd)
        shim_pool:release(shim_cmd)
        
        -- Wait after shim application
        local shim_wait_cmd = delay_pool:get()
        shim_wait_cmd.cmd = ffi.C.DELAY
        shim_wait_cmd.time = 0.01  -- 10ms
        coroutine.yield(shim_wait_cmd)
        delay_pool:release(shim_wait_cmd)
        
        local t_block_end_last = 0
        
        for i_block = 1, #blocks do
            local block = blocks[i_block]
            
            local t_block_start = 1e-6 * block.tstart
            local t_block_end = 1e-6 * (block.tstart + block.duration)
            
            -- Wait time between blocks
            if t_block_start > t_block_end_last then
                local wait_cmd = delay_pool:get()
                wait_cmd.cmd = ffi.C.DELAY
                wait_cmd.time = t_block_start - t_block_end_last
                coroutine.yield(wait_cmd)
                delay_pool:release(wait_cmd)
            end
            
            -- Process block
            local t, flags, f_pulse, p_pulse, a_pulse, grad_events, f_acq, p_acq, t_dw, n_samples = 
                preprocess_block(block, params.f, params.rf_cal, params.g_cal)
            
            -- Create iterators
            local pulse_iter = create_pulse_iterator(f_pulse, p_pulse, a_pulse)
            local grad_iter = create_grad_iterator(grad_events)
            local acq_iter = create_acq_iterator(f_acq, p_acq, t_dw, n_samples)
            
            -- Process events
            local i_pulse_last = #f_pulse - 1
            local n_events = #t
            
            -- First iteration
            if n_events > 0 and t[1] > 0 then
                local wait_cmd = delay_pool:get()
                wait_cmd.cmd = ffi.C.DELAY  
                wait_cmd.time = t[1]
                coroutine.yield(wait_cmd)
                delay_pool:release(wait_cmd)
            end
            
            if n_events > 0 then
                -- Process first event
                local first_flags = flags[1]
                
                if bit.band(first_flags, PULSE_EVENT_MASK) ~= 0 then
                    local i_pulse, pulse = pulse_iter()
                    local cmd
                    if i_pulse == 0 then
                        cmd = pulse_start_pool:get()
                        cmd.cmd = ffi.C.PULSESTART
                        cmd.freq, cmd.phase, cmd.amp = pulse[1], pulse[2], pulse[3]
                    elseif i_pulse == i_pulse_last then
                        cmd = pulse_stop_pool:get()
                        cmd.cmd = ffi.C.PULSESTOP
                    else
                        cmd = pulse_update_pool:get()
                        cmd.cmd = ffi.C.PULSEUPDATE
                        cmd.freq, cmd.phase, cmd.amp = pulse[1], pulse[2], pulse[3]
                    end
                    coroutine.yield(cmd)
                    if i_pulse == 0 then
                        pulse_start_pool:release(cmd)
                    elseif i_pulse == i_pulse_last then
                        pulse_stop_pool:release(cmd)
                    else
                        pulse_update_pool:release(cmd)
                    end
                end
                
                if bit.band(first_flags, GRAD_EVENT_MASK) ~= 0 then
                    local grad = grad_iter()
                    local cmd = gradient_pool:get()
                    cmd.cmd = ffi.C.GRADIENT
                    cmd.grad[0], cmd.grad[1], cmd.grad[2] = grad[1], grad[2], grad[3]
                    coroutine.yield(cmd)
                    gradient_pool:release(cmd)
                end
                
                if bit.band(first_flags, ACQ_EVENT_MASK) ~= 0 then
                    local acq = acq_iter()
                    local cmd = acquire_pool:get()
                    cmd.cmd = ffi.C.ACQUIRE
                    cmd.freq, cmd.phase, cmd.dw, cmd.samples = acq[1], acq[2], acq[3], acq[4]
                    coroutine.yield(cmd)
                    acquire_pool:release(cmd)
                end
                
                -- Rest of iterations  
                for i = 2, n_events do
                    local dt = t[i] - t[i-1]  
                    if dt > 0 then
                        local wait_cmd = delay_pool:get()
                        wait_cmd.cmd = ffi.C.DELAY
                        wait_cmd.time = dt
                        coroutine.yield(wait_cmd)
                        delay_pool:release(wait_cmd)
                    end
                    
                    local curr_flags = flags[i]
                    
                    -- Same event processing as first iteration
                    if bit.band(curr_flags, PULSE_EVENT_MASK) ~= 0 then
                        local i_pulse, pulse = pulse_iter()
                        local cmd
                        if i_pulse == 0 then
                            cmd = pulse_start_pool:get()
                            cmd.cmd = ffi.C.PULSESTART
                            cmd.freq, cmd.phase, cmd.amp = pulse[1], pulse[2], pulse[3]
                        elseif i_pulse == i_pulse_last then
                            cmd = pulse_stop_pool:get()
                            cmd.cmd = ffi.C.PULSESTOP
                        else
                            cmd = pulse_update_pool:get()
                            cmd.cmd = ffi.C.PULSEUPDATE
                            cmd.freq, cmd.phase, cmd.amp = pulse[1], pulse[2], pulse[3]
                        end
                        coroutine.yield(cmd)
                        if i_pulse == 0 then
                            pulse_start_pool:release(cmd)
                        elseif i_pulse == i_pulse_last then
                            pulse_stop_pool:release(cmd)
                        else
                            pulse_update_pool:release(cmd)
                        end
                    end
                    
                    if bit.band(curr_flags, GRAD_EVENT_MASK) ~= 0 then
                        local grad = grad_iter()
                        local cmd = gradient_pool:get()
                        cmd.cmd = ffi.C.GRADIENT
                        cmd.grad[0], cmd.grad[1], cmd.grad[2] = grad[1], grad[2], grad[3]
                        coroutine.yield(cmd)
                        gradient_pool:release(cmd)
                    end
                    
                    if bit.band(curr_flags, ACQ_EVENT_MASK) ~= 0 then
                        local acq = acq_iter()
                        local cmd = acquire_pool:get()
                        cmd.cmd = ffi.C.ACQUIRE
                        cmd.freq, cmd.phase, cmd.dw, cmd.samples = acq[1], acq[2], acq[3], acq[4]
                        coroutine.yield(cmd)
                        acquire_pool:release(cmd)
                    end
                end
                
                -- Wait until end of block duration
                if t_block_start + t[n_events] < t_block_end then
                    local wait_cmd = delay_pool:get()
                    wait_cmd.cmd = ffi.C.DELAY
                    wait_cmd.time = t_block_end - (t_block_start + t[n_events])
                    coroutine.yield(wait_cmd)
                    delay_pool:release(wait_cmd)
                end
            end
            
            t_block_end_last = t_block_end
        end
        
        -- Generate END command
        local end_cmd = end_pool:get()
        end_cmd.cmd = ffi.C.END
        coroutine.yield(end_cmd)
        end_pool:release(end_cmd)
    end)
end

return M