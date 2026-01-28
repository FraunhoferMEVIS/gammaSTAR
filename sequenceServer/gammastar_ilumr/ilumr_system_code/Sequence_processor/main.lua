-- main.lua - NMR Sequence Controller (with YAML command dump and per-command time)

-- imports
local Logging = require "logging"
local zmq = require("zmq")
local json = require('rapidjson')
local buffer = require("string.buffer")
local ffi = require("ffi")
local bit = require("bit")
local yaml = require("yaml")

package.path = package.path .. ";../ilumr_sys/?.lua;../ilumr_sys/?/init.lua"
local sequence_inst = require("sequence_inst")
local dmabuf = require("dma")

-- Make FFI constants available
ffi.C = ffi.C or {}

-- New modules
local tcp_server = require("tcp_server")
local sequence_processor = require("sequence_processor")
local utils = require("utils")

-- logging
local log = Logging.defaultLogger()
log:setLevel(log.DEBUG)

-- constants
local PACKET_SIZE = 10000
local EXEC_REQREP_ADDR = "tcp://driver:5005"
local EXEC_PUSHPULL_ADDR = "tcp://driver:5006"
local EXEC_PUBSUB_ADDR = "tcp://driver:5007"
local EXEC_PUSH_HWM = 2

-- run variables
local run_id = 0
local acq_count_complete = 0
local acq_count_total = 0
local run_done = false
local run_started = false

-- ZeroMQ sockets
local zmqctx = zmq.init(1)
local exec_req = zmqctx:socket(zmq.REQ)
exec_req:setopt(zmq.REQ_RELAXED, 1)
exec_req:connect(EXEC_REQREP_ADDR)
log:info("Connected exec_req to %s", EXEC_REQREP_ADDR)

local exec_push = zmqctx:socket(zmq.PUSH)
exec_push:setopt(zmq.SNDHWM, EXEC_PUSH_HWM)
exec_push:connect(EXEC_PUSHPULL_ADDR)
log:info("Connected exec_push to %s", EXEC_PUSHPULL_ADDR)

local exec_sub = zmqctx:socket(zmq.SUB)
exec_sub:setopt(zmq.SUBSCRIBE, "")
exec_sub:connect(EXEC_PUBSUB_ADDR)
log:info("Connected exec_sub to %s", EXEC_PUBSUB_ADDR)

local poller = zmq.ZMQ_Poller()
local exec_push_poll_id = poller:add(exec_push, zmq.POLLOUT)
local exec_sub_poll_id = poller:add(exec_sub, zmq.POLLIN)

-- command buffer control
local buf = buffer.new(PACKET_SIZE)

local function reset_buffer()
    buf:reset()
    buf:putcdata(ffi.new("int[1]", run_id), 4)
end

local function append_buffer(cmd)
    local cmd_size = ffi.sizeof(cmd)
    buf:putcdata(cmd, cmd_size)
end

local function send_buffer()
    exec_push:send(buf)
    reset_buffer()
end

local function force_send_buffer()
    if #buf > 4 then                -- More than just run_id
        exec_push:send(buf)
        reset_buffer()
    end
end

local function sub_receive()
    local msg = json.decode(exec_sub:recv())
    if msg.id ~= run_id then
        return
    end

    if msg.type == 'event' then
        if msg.name == 'start' then
            run_started = true
        elseif msg.name == 'dre' or msg.name == 'done' then
            acq_count_complete = msg.data[1]
            acq_count_total = msg.data[2]

            if msg.name == 'done' then
                run_done = true
            end
        end
    end
end

-- YAML fallback serializer (if yaml.encode is missing)
local function is_array(t)
    if type(t) ~= "table" then return false end
    local n = #t
    for k, _ in pairs(t) do
        if type(k) ~= "number" or k < 1 or k > n then
            return false
        end
    end
    return true
end

local function yaml_val(v)
    local tv = type(v)
    if tv == "number" or tv == "boolean" then
        return tostring(v)
    elseif tv == "string" then
        if v:match("^[%w_%.%-:/ ]+$") then
            return v
        else
            return string.format("%q", v)
        end
    elseif tv == "table" then
        if is_array(v) then
            local parts = {}
            for i = 1, #v do parts[i] = yaml_val(v[i]) end
            return "[" .. table.concat(parts, ", ") .. "]"
        else
            local parts = {}
            for k, vv in pairs(v) do
                parts[#parts+1] = string.format("%s: %s", tostring(k), yaml_val(vv))
            end
            return "{ " .. table.concat(parts, ", ") .. " }"
        end
    else
        return "null"
    end
end

local function yaml_encode_list(list)
    local out = {}
    for _, item in ipairs(list) do
        out[#out+1] = "-"
        -- stable key order first (includes absolute and duration times)
        local order = {"type","code","at_s","duration_s","freq","phase","amp","dw","samples","shim","grad","mask"}
        local printed = {}
        for _, k in ipairs(order) do
            if item[k] ~= nil then
                out[#out+1] = string.format("  %s: %s", k, yaml_val(item[k]))
                printed[k] = true
            end
        end
        -- any extra keys
        for k, v in pairs(item) do
            if not printed[k] then
                out[#out+1] = string.format("  %s: %s", tostring(k), yaml_val(v))
            end
        end
    end
    return table.concat(out, "\n") .. "\n"
end

local function write_yaml_file(filename, data)
    local ok, err = pcall(function()
        local f = assert(io.open(filename, "w"))
        if yaml and yaml.encode then
            f:write(yaml.encode(data))
        else
            f:write(yaml_encode_list(data))
        end
        f:close()
    end)
    return ok, err
end

-- Human-readable YAML: map command codes to names and copy to plain Lua table
local CMD_NAMES = {
  [tonumber(ffi.C.INIT)]            = "INIT",
  [tonumber(ffi.C.END)]             = "END",
  [tonumber(ffi.C.DELAY)]           = "DELAY",
  [tonumber(ffi.C.PULSESTART)]      = "PULSESTART",
  [tonumber(ffi.C.PULSEUPDATE)]     = "PULSEUPDATE",
  [tonumber(ffi.C.PULSESTOP)]       = "PULSESTOP",
  [tonumber(ffi.C.ACQUIRE)]         = "ACQUIRE",
  [tonumber(ffi.C.SHIM)]            = "SHIM",
  [tonumber(ffi.C.GRADIENT)]        = "GRADIENT",
  [tonumber(ffi.C.WAIT_FOR_TRIGGER)]= "WAIT_FOR_TRIGGER",
  [tonumber(ffi.C.GPO_SET)]         = "GPO_SET",
  [tonumber(ffi.C.GPO_CLEAR)]       = "GPO_CLEAR",
  [tonumber(ffi.C.PULSEUNBLANK)]    = "PULSEUNBLANK",
  [tonumber(ffi.C.PULSEBLANK)]      = "PULSEBLANK",
}

local function cmd_to_table(cmd, at_s)
    local code = tonumber(cmd.cmd)
    local t = { code = code, type = CMD_NAMES[code] or ("UNKNOWN_"..tostring(code)), at_s = at_s }

    if code == tonumber(ffi.C.INIT) then
        t.amp_enable = cmd.amp_enable
        t.rx_gain    = cmd.rx_gain

    elseif code == tonumber(ffi.C.DELAY) then
        t.duration_s = tonumber(cmd.time)

    elseif code == tonumber(ffi.C.PULSESTART) or code == tonumber(ffi.C.PULSEUPDATE) then
        t.freq  = cmd.freq
        t.phase = cmd.phase
        t.amp   = cmd.amp

    elseif code == tonumber(ffi.C.PULSESTOP) then
        -- no extra fields

    elseif code == tonumber(ffi.C.ACQUIRE) then
        t.freq    = cmd.freq
        t.phase   = cmd.phase
        t.dw      = cmd.dw
        t.samples = cmd.samples

    elseif code == tonumber(ffi.C.SHIM) then
        t.shim = {
          cmd.shim[0], cmd.shim[1], cmd.shim[2], cmd.shim[3],
          cmd.shim[4], cmd.shim[5], cmd.shim[6], cmd.shim[7]
        }

    elseif code == tonumber(ffi.C.GRADIENT) then
        t.grad = { cmd.grad[0], cmd.grad[1], cmd.grad[2] }

    elseif code == tonumber(ffi.C.GPO_SET) or code == tonumber(ffi.C.GPO_CLEAR) then
        t.mask = cmd.mask
    end

    return t
end

local function execute_sequence(sequence_data)
    -- Reset run variables
    acq_count_complete = 0
    acq_count_total = 0
    run_done = false
    run_started = false

    -- Validate sequence
    local valid, err_msg = sequence_processor.validate_sequence_data(sequence_data)
    if not valid then
        error("Sequence validation failed: " .. err_msg)
    end

    -- Send start command
    exec_req:send(json.encode({type="cmd", name="start"}))
    local reply = json.decode(exec_req:recv())
    if reply.type == "error" then
        error(reply.error)
    end
    run_id = reply.id
    log:info("Received new run_id: %d", run_id)

    reset_buffer()

    local seq_gen_timer = zmq.stopwatch_start()

    -- Create sequence generator
    local success, sequence_generator = pcall(sequence_processor.create_generator, sequence_data)
    if not success then
        error("Failed to create sequence generator: " .. sequence_generator)
    end

    local command_count = 0
    local acquire_commands = 0
    local command_types = {}
    local total_samples = 0
    local total_acquisition_time = 0

    -- Collect human-readable YAML entries
    local yaml_commands = {}
    local yaml_filename = nil

    -- Absolute time since sequence start (seconds)
    local now_s = 0.0

    for cmd in sequence_generator do
        -- Write raw bytes to driver buffer
        append_buffer(cmd)
        command_count = command_count + 1

        -- Record stats
        local cmd_val = cmd.cmd and tonumber(cmd.cmd) or nil
        if cmd_val then
            command_types[cmd_val] = (command_types[cmd_val] or 0) + 1

            if cmd_val == tonumber(ffi.C.ACQUIRE) then
                acquire_commands = acquire_commands + 1
                local samples = tonumber(cmd.samples)
                local dw = tonumber(cmd.dw)
                total_samples = total_samples + samples
                total_acquisition_time = total_acquisition_time + (samples * dw)
            end
        end

        -- Copy to YAML table with absolute time BEFORE advancing time
        yaml_commands[#yaml_commands+1] = cmd_to_table(cmd, now_s)

        -- Advance absolute time on DELAY commands
        if cmd_val == tonumber(ffi.C.DELAY) then
            now_s = now_s + tonumber(cmd.time or 0)
        end

        if #buf >= PACKET_SIZE then
            local sent = false
            repeat
                local count, err = poller:poll(-1)
                if not count then
                    error("Polling error: " .. (err or "unknown"))
                end
                for i=1,count do
                    local id, revents = poller:next_revents_idx()
                    if id==exec_push_poll_id and bit.band(revents, zmq.POLLOUT) ~= 0 then
                        send_buffer()
                        sent = true
                    elseif id==exec_sub_poll_id and bit.band(revents, zmq.POLLIN) ~= 0 then
                        sub_receive()
                    end
                end
            until sent
        end
    end

    -- Flush remaining buffered commands (END is already included by generator)
    force_send_buffer()

    -- Save YAML now that run_id is known
    -- yaml_filename = string.format("commands_run_%d.yaml", run_id)
    -- local ok_yaml, yaml_err = write_yaml_file(yaml_filename, yaml_commands)
    -- if ok_yaml then
    --     log:info("Wrote YAML commands to %s", yaml_filename)
    -- else
    --     log:error("Failed to write YAML commands: %s", yaml_err)
    -- end

    local seq_gen_elapsed = seq_gen_timer:stop()
    log:info("Sequence generation completed in %.4f sec", seq_gen_elapsed/1000000)

    -- Wait for completion
    log:info("Waiting for sequence completion...")

    local timeout_count = 0
    local max_timeouts = 60
    local last_progress = 0

    repeat
        local count, err = poller:poll(1000)  -- 1 second timeout
        if count and count > 0 then
            timeout_count = 0
            for i=1,count do
                local id, revents = poller:next_revents_idx()
                if id==exec_sub_poll_id and bit.band(revents, zmq.POLLIN) ~= 0 then
                    sub_receive()
                end
            end
        else
            timeout_count = timeout_count + 1

            local progress_info = string.format("Progress: %d/%d (%.1f%%)",
                                              acq_count_complete, acq_count_total,
                                              acq_count_total > 0 and (acq_count_complete/acq_count_total*100) or 0)

            if acq_count_complete > last_progress then
                last_progress = acq_count_complete
                timeout_count = 0
            else
                log:warn("No status update received (timeout %d/%d), current %s",
                        timeout_count, max_timeouts, progress_info)
            end

            if timeout_count >= max_timeouts then
                error(string.format("Sequence execution timeout after %d seconds - stuck at %s",
                                  max_timeouts, progress_info))
            end
        end
    until run_done

    -- Send finish command
    exec_req:send(json.encode({type="cmd", name="finish"}))
    local reply2 = json.decode(exec_req:recv())
    if reply2.type == "error" then
        error("Finish command error: " .. reply2.error)
    end

    -- Read data
    local timeout_seconds = math.max(total_acquisition_time * 2, 10.0)

    local data
    if total_samples > 0 then
        data = dmabuf.read_data(0, total_samples, timeout_seconds)
        json.dump(data, 'data.json')
    else
        log:warn("No samples expected, reading with default parameters")
        data = dmabuf.read_data(0, 1000, 10.0)
    end

    return {
        success = true,
        data = data,
        sequence_type = sequence_data.type or "unknown",
        run_id = run_id,
        generation_time = seq_gen_elapsed/1000000,
        command_count = command_count,
        acquire_count = acquire_commands,
        total_samples = total_samples,
        total_acquisition_time = total_acquisition_time,
        command_breakdown = command_types,
        final_progress = {acq_count_complete, acq_count_total},
        commands_yaml = yaml_filename
    }
end

-- Main execution function
local function main()
    -- start TCP Server
    local success, err = pcall(function()
        tcp_server.start({
            host = "0.0.0.0",
            port = 8765,
            execute_sequence = execute_sequence,
            log = log
        })
    end)

    if not success then
        log:error("Failed to start TCP server: %s", err)
        return
    end
end

-- Error handling wrapper
local status, err = pcall(main)
if not status then
    log:error("Main execution error: %s", err)

    -- Send finish command if error occurs
    local ok, reply_err = pcall(function()
        exec_req:send(json.encode({type="cmd", name="finish"}))
        local reply = json.decode(exec_req:recv())
        if reply.type == "error" then
            log:error("Finish command error: %s", reply.error)
        end
    end)

    if not ok then
        log:error("Error during cleanup: %s", reply_err)
    end

    os.exit(1)
end

-- Cleanup
log:info("Shutting down...")
exec_req:close()
exec_push:close()
exec_sub:close()
zmqctx:term()
log:info("Shutdown complete")