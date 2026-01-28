local M = {}

-- Convert string to hex representation (from original code)
function M.stringToHex(inputString)
    local hexString = ""
    for i = 1, #inputString do
        local byte = string.byte(inputString, i)
        hexString = hexString .. string.format("%02X ", byte)
    end
    return hexString
end

-- Linear interpolation
function M.lerp(a, b, t)
    return a + (b - a) * t
end

-- Check if table is empty
function M.is_empty(t)
    return next(t) == nil
end

-- Deep copy table
function M.deep_copy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[M.deep_copy(orig_key)] = M.deep_copy(orig_value)
        end
        setmetatable(copy, M.deep_copy(getmetatable(orig)))
    else
        copy = orig
    end
    return copy
end

-- Safe JSON decode
function M.safe_json_decode(str)
    local success, result = pcall(require("cjson").decode, str)
    if success then
        return result
    else
        return nil, result
    end
end

-- Safe JSON encode
function M.safe_json_encode(obj)
    local success, result = pcall(require("cjson").encode, obj)
    if success then
        return result
    else
        return nil, result
    end
end

return M