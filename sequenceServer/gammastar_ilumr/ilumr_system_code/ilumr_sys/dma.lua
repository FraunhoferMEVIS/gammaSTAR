local Logging = require "logging"
local ffi = require("ffi")

local log = Logging.defaultLogger()
log:setLevel(log.DEBUG)

local DMA_DEVICE = "/dev/udmabuf0"
local DMA_SIZE_FILE = "/sys/class/u-dma-buf/udmabuf0/size"

local f = assert(io.open(DMA_SIZE_FILE, "r"))
local DMA_SIZE = f:read("*number")
f:close()
log:debug("DMA_SIZE: 0x%x", DMA_SIZE)

local SAMPLE_WORD_SIZE = 8 -- two int32 (real,imag) per sample
local DMA_INST_OFFSET = 0x00000000
local DMA_INST_SIZE = 0x01000000
local DMA_DATA_OFFSET = DMA_INST_SIZE
local DMA_DATA_SIZE = DMA_SIZE - DMA_DATA_OFFSET
log:debug("DMA_DATA_OFFSET: 0x%x", DMA_DATA_OFFSET)
log:debug("DMA_DATA_SIZE: 0x%x", DMA_DATA_SIZE)

local function reset()
	-- clear dma memory buffer with zeros
	local bs = 4096
	local count = DMA_SIZE/bs
	os.execute(string.format("dd bs=%d count=%d if=/dev/zero of=%s", bs, count, DMA_DEVICE))
end

local function get_data_dma_size_words()
    -- return size of buffer in samples
    return DMA_DATA_SIZE/SAMPLE_WORD_SIZE
end

local function read_data(offset, length, calibration)
    -- length and offset are in samples
    -- returns table of interleaved real/imag values

    bytes_length = length*SAMPLE_WORD_SIZE
    bytes_offset = DMA_DATA_OFFSET + offset*SAMPLE_WORD_SIZE
    local fd = assert(io.open(DMA_DEVICE, "rb"))
    fd:seek("set", bytes_offset)
    int_data = ffi.cast("int *", fd:read(bytes_length))
    data = {}
    for i = 1, 2*length do
    	data[i] = calibration*int_data[i]
    end
    fd:close()
    return data
end

return {
	reset = reset,
	get_data_dma_size_words = get_data_dma_size_words,
	read_data = read_data
}