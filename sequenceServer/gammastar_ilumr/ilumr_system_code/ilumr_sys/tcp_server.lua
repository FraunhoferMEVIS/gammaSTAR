local M = {}

local socket = require("socket")
local json = require('rapidjson')
local cqueues = require("cqueues")
local bit = require("bit")

-- Configuration
local CHUNK_SIZE = 1024

-- Helper function to convert integer to 4 bytes (big endian)
local function int_to_bytes(num)
    return string.char(
        bit.rshift(num, 24) % 256,
        bit.rshift(num, 16) % 256,
        bit.rshift(num, 8) % 256,
        num % 256
    )
end

-- Helper function to convert 4 bytes to integer (big endian)
local function bytes_to_int(bytes)
    local b1, b2, b3, b4 = string.byte(bytes, 1, 4)
    return bit.lshift(b1, 24) + bit.lshift(b2, 16) + bit.lshift(b3, 8) + b4
end

-- TCP Message Protocol: [4 bytes length][message data]
local function send_tcp_message(client, data)
    local data_str = type(data) == "string" and data or json.encode(data)
    local length = #data_str
    
    print(string.format("DEBUG: Attempting to send message of length %d", length))
    
    -- Send length header (4 bytes, big endian)
    local length_bytes = int_to_bytes(length)
    print(string.format("DEBUG: Sending length header: %d bytes", #length_bytes))
    
    local bytes_sent, err = client:send(length_bytes)
    if not bytes_sent then
        print(string.format("DEBUG: Failed to send length header: %s", err or "unknown error"))
        return false, "Failed to send length header: " .. (err or "unknown error")
    end
    print(string.format("DEBUG: Length header sent: %d bytes", bytes_sent))
    
    -- Send message data
    print(string.format("DEBUG: Sending message data: %d bytes", length))
    bytes_sent, err = client:send(data_str)
    if not bytes_sent then
        print(string.format("DEBUG: Failed to send message data: %s", err or "unknown error"))
        return false, "Failed to send message data: " .. (err or "unknown error")
    end
    print(string.format("DEBUG: Message data sent: %d bytes", bytes_sent))
    
    return true
end

local function receive_tcp_message(client, timeout)
    timeout = timeout or 30
    client:settimeout(timeout)
    
    -- Receive length header (4 bytes)
    local length_bytes, err = client:receive(4)
    if not length_bytes then
        return nil, "Failed to receive length header: " .. (err or "unknown error")
    end
    
    local length = bytes_to_int(length_bytes)
    if length <= 0 or length > 10 * 1024 * 1024 then -- Max 10MB
        return nil, "Invalid message length: " .. length
    end
    
    -- Receive message data
    local message_data, err = client:receive(length)
    if not message_data then
        return nil, "Failed to receive message data: " .. (err or "unknown error")
    end
    
    return message_data
end

-- Utility functions for chunked data transfer (adapted for TCP)
local function send_in_chunks(client, data, chunk_size)
    chunk_size = chunk_size or CHUNK_SIZE
    local data_str = type(data) == "string" and data or json.encode(data)
    local total_size = #data_str
    local num_chunks = math.ceil(total_size / chunk_size)
    
    for i = 0, num_chunks - 1 do
        local start_idx = i * chunk_size + 1
        local end_idx = math.min(start_idx + chunk_size - 1, total_size)
        local chunk = string.sub(data_str, start_idx, end_idx)
        
        local chunk_data = {
            chunk_index = i,
            total_chunks = num_chunks,
            data = chunk
        }
        
        local success, err = send_tcp_message(client, chunk_data)
        if not success then
            print("Error sending chunk:", err)
            return false
        end
    end
    return true
end

local function receive_in_chunks(client, log)
    local chunks = {}
    local total_chunks = nil
    
    -- log:info("Starting to receive chunked data...")
    
    while true do
        -- log:info("Waiting for message chunk...")
        local message, error_msg = receive_tcp_message(client, 120) -- 120 second timeout
        
        if not message then
            log:info("Error receiving message: %s", error_msg or "unknown error")
            return nil
        end
        
        local success, data = pcall(json.decode, message)
        if not success then
            log:error("Error decoding JSON: %s", tostring(data))
            return nil
        end
        
        -- log:info("Successfully decoded message")
        
        -- Check if this is a simple message (not chunked)
        if data.sequence and not data.chunk_index then
            log:info("Received non-chunked message")
            return data
        end
        
        -- Handle chunked data
        local chunk_index = data.chunk_index
        total_chunks = data.total_chunks
        local chunk_data = data.data

        if not chunk_data then
            log:warning("Received chunk_data is nil")
            goto continue
        end

        -- log:info("Received chunk %d of %d", chunk_index + 1, total_chunks)
        chunks[chunk_index] = chunk_data
        
        -- Check if all chunks received
        local received_count = 0
        for _ in pairs(chunks) do
            received_count = received_count + 1
        end
        
        if received_count == total_chunks then
            -- log:info("All chunks received, assembling data...")
            -- Assemble full data
            local full_data_parts = {}
            for i = 0, total_chunks - 1 do
                table.insert(full_data_parts, chunks[i])
            end
            
            local full_data_str = table.concat(full_data_parts)
            local success, result = pcall(json.decode, full_data_str)
            
            if success then
                -- log:info("Data assembled successfully")
                return result
            else
                log:error("Error decoding assembled data: %s", tostring(result))
                return nil
            end
        end
        
        ::continue::
    end
end

-- Load system parameters
local function load_system_parameters()
    return {
        frequency = 10e6, -- Hz
        FOV = {0.1, 0.1, 0.1}, -- m
        gradient_calibration = {0.2, 0.2, 0.15}, -- T/m
        pulse_calibration = 0.0002621116553714017,
        RF_CAL = 0.0002621116553714017,
        G_CAL = {0.2, 0.2, 0.15},
        POST_DEC = 4
    }
end

-- Serialize array for JSON transmission
local function serialize_array(arr)
    if type(arr) ~= "table" then
        return {}
    end
    
    local result = {}
    for i, v in ipairs(arr) do
        if type(v) == "table" and v.real and v.imag then
            -- Complex number
            table.insert(result, {real = v.real, imag = v.imag})
        else
            table.insert(result, v)
        end
    end
    return result
end

-- Handle TCP client connection
local function handle_tcp_connection(client, config)
    local params = load_system_parameters()
    local sequence_counter = 0
    local log = config.log
    
    log:info("=== New TCP connection established ===")
    
    -- Socket konfigurieren
    local success, err = pcall(function()
        client:settimeout(30) -- 30 Sekunden Timeout
        client:setoption("tcp-nodelay", true) -- Disable Nagle algorithm
        client:setoption("keepalive", true)
    end)
    
    if not success then
        log:error("Failed to configure socket: %s", tostring(err))
        client:close()
        return
    end
    
    log:info("Socket configured successfully")
    
    -- Send initial data
    local initial_data = {
        message = "Welcome to the TCP server!",
        frequency = params.frequency,
        FOV = params.FOV,
        gradient_calibration = params.gradient_calibration,
        pulse_calibration = params.pulse_calibration,
        server_ready = true,
        timestamp = os.time()
    }
    
    -- log:info("Sending initial data to client...")
    -- log:info("Initial data size: %d bytes", #json.encode(initial_data))
    
    local send_success, send_err = send_tcp_message(client, initial_data)
    
    if not send_success then
        log:error("Failed to send initial data: %s", tostring(send_err))
        client:close()
        return
    end
    
    -- log:info("Initial data sent successfully")
    -- log:info("Waiting for client to send sequence data...")
    
    -- Main processing loop
    while true do
        log:info("=== Waiting for next message from client ===")
        local message = receive_in_chunks(client, log)
        
        if not message then 
            log:warn("No message received, connection may be closed")
            break 
        end
        
        local sequence_name = message.sequence
        local sequence_data = message.data
        
        -- log:info("Received sequence: %s", tostring(sequence_name))
        
        if not sequence_name then
            local error_message = "Invalid or missing sequence data"
            log:error(error_message)
            local send_success, send_err = send_tcp_message(client, {
                status = "error",
                error = error_message,
                timestamp = os.time()
            })
            if not send_success then
                log:error("Failed to send error message: %s", tostring(send_err))
                break
            end
            goto continue
        end
        
        print("====================================================================================")
        print(string.format("Sequence Nr: %d with Sequence: %s", sequence_counter, sequence_name))
        
        -- Send acknowledgment
        local ack_success, ack_err = send_tcp_message(client, {
            status = "received",
            message = "Sequence received, starting processing...",
            sequence_name = sequence_name,
            timestamp = os.time()
        })
        
        if not ack_success then
            log:error("Failed to send acknowledgment: %s", tostring(ack_err))
            break
        end
        
        -- Execute sequence
        log:info("Executing sequence...")

        -- Check if system is ready
        -- if driver.is_running() then
        --     local error_message = "System is busy, please wait and try again"
        --     log:error(error_message)
        --     local send_success, send_err = send_tcp_message(client, {
        --         status = "busy",
        --         error = error_message,
        --         timestamp = os.time()
        --     })
        --     if not send_success then
        --         log:error("Failed to send busy message: %s", tostring(send_err))
        --         break
        --     end
        --     goto continue
        -- end

        local exec_success, result = pcall(config.execute_sequence, sequence_data)
        
        if exec_success then
            sequence_counter = sequence_counter + 1
            log:info("Sequence %d completed successfully", sequence_counter)
            
            -- Send processing status
            local status_success, status_err = send_tcp_message(client, {
                status = "processing",
                sequence_number = sequence_counter,
                message = "Sequence executed, preparing data...",
                timestamp = os.time()
            })
            
            if not status_success then
                log:error("Failed to send status update: %s", tostring(status_err))
                break
            end
            
            -- Serialize and send data
            -- log:info("Serializing and sending result data...")
            local serialized_data = serialize_array(result)
            local chunk_success = send_in_chunks(client, {
                status = "complete",
                sequence_number = sequence_counter,
                data = serialized_data,
                timestamp = os.time()
            })
            
            if chunk_success then
                log:info("Data sent successfully")
                print("Data sent successfully")
            else
                log:error("Failed to send data")
                print("Failed to send data")
                break
            end
            print("====================================================================================")
        else
            log:error("Error executing sequence: %s", tostring(result))
            local send_success, send_err = send_tcp_message(client, {
                status = "error",
                error = tostring(result),
                sequence_number = sequence_counter,
                timestamp = os.time()
            })
            if not send_success then
                log:error("Failed to send error response: %s", tostring(send_err))
                break
            end
        end
        
        ::continue::
    end
    
    log:info("=== TCP connection closed ===")
    client:close()
end

-- Start TCP server
function M.start(config)
    local server, bind_err = socket.bind(config.host or "0.0.0.0", config.port or 8765)
    if not server then
        config.log:error("Failed to bind TCP server: %s", tostring(bind_err))
        return
    end
    
    -- Server-Socket konfigurieren
    server:setoption("reuseaddr", true)
    server:settimeout(1) -- Non-blocking accept mit 1 Sekunde Timeout
    
    config.log:info("TCP server started on %s:%d", 
                   config.host or "0.0.0.0", 
                   config.port or 8765)
    
    -- Erstelle einen cqueues-Kontext
    local cq = cqueues.new()
    
    cq:wrap(function()
        while true do
            local client, accept_err = server:accept()
            if client then
                -- Get client info
                local client_ip, client_port = client:getpeername()
                -- config.log:info("New client connected from %s:%s", client_ip or "unknown", client_port or "unknown")
                
                -- Handle connection in a new coroutine
                cq:wrap(function()
                    local success, connection_err = pcall(handle_tcp_connection, client, config)
                    if not success then
                        config.log:error("Error handling client connection: %s", tostring(connection_err))
                        if client then
                            client:close()
                        end
                    end
                end)
            elseif accept_err ~= "timeout" then
                config.log:warning("Accept error: %s", accept_err or "unknown")
                cqueues.sleep(0.1) -- Kurz warten bei Fehlern
            end
            cqueues.sleep(0.01) -- Kleine Pause um CPU zu schonen
        end
    end)
    
    -- Starte den cqueues Loop
    local ok, loop_err = cq:loop()
    if not ok then
        config.log:error("cqueues loop error: %s", tostring(loop_err))
    end
end

return M