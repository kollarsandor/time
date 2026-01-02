local C = terralib.includec("stdio.h")
local stdlib = terralib.includec("stdlib.h")
local string_h = terralib.includec("string.h")
local fcntl = terralib.includec("fcntl.h")
local unistd = terralib.includec("unistd.h")
local mman = terralib.includec("sys/mman.h")
local stat = terralib.includec("sys/stat.h")
local math_h = terralib.includec("math.h")
local pthread = terralib.includec("pthread.h")
local time_h = terralib.includec("time.h")

local CUDA_SUCCESS = 0
local NCCL_SUCCESS = 0

local cudaError_t = int32
local ncclResult_t = int32
local cudaStream_t = &opaque
local ncclComm_t = &opaque
local CUdeviceptr = uint64

struct CudaContext {
    device_id: int32
    stream: cudaStream_t
    device_memory: CUdeviceptr
    device_memory_size: uint64
    workspace: CUdeviceptr
    workspace_size: uint64
}

struct NCCLContext {
    comm: ncclComm_t
    rank: int32
    num_ranks: int32
    stream: cudaStream_t
}

struct TensorDescriptor {
    name: &int8
    data_offset: uint64
    data_size: uint64
    dtype: int32
    shape: &int64
    ndim: int32
    shard_file: &int8
}

struct SafetensorsIndex {
    weight_map: &&int8
    shard_files: &&int8
    num_weights: int32
    num_shards: int32
}

struct ShardMapping {
    shard_path: &int8
    mmap_ptr: &int8
    mmap_size: uint64
    header_size: uint64
    data_base: &int8
    fd: int32
}

struct PagedKVCache {
    k_cache: CUdeviceptr
    v_cache: CUdeviceptr
    page_size: int32
    num_pages: int32
    max_pages: int32
    page_table: &int32
    free_pages: &int32
    num_free_pages: int32
    num_layers: int32
    num_heads: int32
    head_dim: int32
    bytes_per_page: uint64
    lock: pthread.pthread_mutex_t
}

struct RequestState {
    request_id: int64
    seq_len: int32
    page_indices: &int32
    num_pages: int32
    past_tokens: &int64
    past_len: int32
    max_gen_tokens: int32
    temperature: float
    top_p: float
    rep_penalty: float
    stop_tokens: &int64
    num_stop_tokens: int32
    is_prefill_done: int32
    is_finished: int32
}

struct BatchState {
    requests: &&RequestState
    num_requests: int32
    max_batch_size: int32
    batch_seq_lens: &int32
    batch_page_tables: &&int32
    lock: pthread.pthread_mutex_t
}

struct ModelConfig {
    hidden_dim: int32
    num_layers: int32
    num_heads: int32
    num_kv_heads: int32
    head_dim: int32
    vocab_size: int32
    num_experts: int32
    top_k_experts: int32
    intermediate_size: int32
    rope_base: float
    rms_norm_eps: float
    max_seq_len: int32
}

struct EngineHandle {
    model_dir: &int8
    max_batch: int32
    max_seq: int32
    num_gpus: int32
    config: ModelConfig
    cuda_contexts: &CudaContext
    nccl_contexts: &NCCLContext
    kv_cache: &PagedKVCache
    batch_state: &BatchState
    index: SafetensorsIndex
    shard_mappings: &ShardMapping
    tensor_descriptors: &TensorDescriptor
    num_tensors: int32
    embed_weight: CUdeviceptr
    lm_head_weight: CUdeviceptr
    layer_weights: &CUdeviceptr
    num_layer_weights: int32
    futhark_ctx: &opaque
    initialized: int32
}

struct SamplingParams {
    temperature: float
    top_p: float
    top_k: int32
    rep_penalty: float
    freq_penalty: float
    presence_penalty: float
    stop_tokens: &int64
    num_stop_tokens: int32
    max_tokens: int32
}

terra cuda_device_synchronize() : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_malloc(ptr: &CUdeviceptr, size: uint64) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_free(ptr: CUdeviceptr) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_memcpy_h2d(dst: CUdeviceptr, src: &opaque, size: uint64) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_memcpy_d2h(dst: &opaque, src: CUdeviceptr, size: uint64) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_memcpy_d2d(dst: CUdeviceptr, src: CUdeviceptr, size: uint64) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_stream_create(stream: &cudaStream_t) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_stream_synchronize(stream: cudaStream_t) : cudaError_t
    return CUDA_SUCCESS
end

terra cuda_set_device(device: int32) : cudaError_t
    return CUDA_SUCCESS
end

terra nccl_comm_init_rank(comm: &ncclComm_t, num_ranks: int32, id: &opaque, rank: int32) : ncclResult_t
    return NCCL_SUCCESS
end

terra nccl_all_reduce(sendbuff: CUdeviceptr, recvbuff: CUdeviceptr, count: uint64, datatype: int32, op: int32, comm: ncclComm_t, stream: cudaStream_t) : ncclResult_t
    return NCCL_SUCCESS
end

terra nccl_all_gather(sendbuff: CUdeviceptr, recvbuff: CUdeviceptr, count: uint64, datatype: int32, comm: ncclComm_t, stream: cudaStream_t) : ncclResult_t
    return NCCL_SUCCESS
end

terra nccl_reduce_scatter(sendbuff: CUdeviceptr, recvbuff: CUdeviceptr, count: uint64, datatype: int32, op: int32, comm: ncclComm_t, stream: cudaStream_t) : ncclResult_t
    return NCCL_SUCCESS
end

terra nccl_all_to_all(sendbuff: CUdeviceptr, recvbuff: CUdeviceptr, count: uint64, datatype: int32, comm: ncclComm_t, stream: cudaStream_t) : ncclResult_t
    return NCCL_SUCCESS
end

terra read_file_contents(path: &int8, size_out: &uint64) : &int8
    var fd = fcntl.open(path, 0)
    if fd < 0 then return nil end
    var st: stat.struct_stat
    if stat.fstat(fd, &st) < 0 then
        unistd.close(fd)
        return nil
    end
    var size = st.st_size
    var buf = [&int8](stdlib.malloc(size + 1))
    if buf == nil then
        unistd.close(fd)
        return nil
    end
    var total_read: int64 = 0
    while total_read < size do
        var rd = unistd.read(fd, buf + total_read, size - total_read)
        if rd <= 0 then
            stdlib.free(buf)
            unistd.close(fd)
            return nil
        end
        total_read = total_read + rd
    end
    buf[size] = 0
    @size_out = size
    unistd.close(fd)
    return buf
end

terra mmap_file_readonly(path: &int8, size_out: &uint64) : &int8
    var fd = fcntl.open(path, 0)
    if fd < 0 then return nil end
    var st: stat.struct_stat
    if stat.fstat(fd, &st) < 0 then
        unistd.close(fd)
        return nil
    end
    var size = st.st_size
    var ptr = mman.mmap(nil, size, 1, 2, fd, 0)
    if ptr == [&opaque](-1) then
        unistd.close(fd)
        return nil
    end
    @size_out = size
    return [&int8](ptr)
end

terra parse_json_int(json: &int8, key: &int8) : int64
    var key_len = string_h.strlen(key)
    var search_key = [&int8](stdlib.malloc(key_len + 4))
    string_h.sprintf(search_key, "\"%s\"", key)
    var pos = string_h.strstr(json, search_key)
    stdlib.free(search_key)
    if pos == nil then return -1 end
    pos = pos + key_len + 2
    while @pos ~= 0 and @pos ~= [int8](':') do
        pos = pos + 1
    end
    if @pos == 0 then return -1 end
    pos = pos + 1
    while @pos ~= 0 and (@pos == [int8](' ') or @pos == [int8]('\t') or @pos == [int8]('\n')) do
        pos = pos + 1
    end
    return stdlib.strtoll(pos, nil, 10)
end

terra parse_json_float(json: &int8, key: &int8) : float
    var key_len = string_h.strlen(key)
    var search_key = [&int8](stdlib.malloc(key_len + 4))
    string_h.sprintf(search_key, "\"%s\"", key)
    var pos = string_h.strstr(json, search_key)
    stdlib.free(search_key)
    if pos == nil then return 0.0f end
    pos = pos + key_len + 2
    while @pos ~= 0 and @pos ~= [int8](':') do
        pos = pos + 1
    end
    if @pos == 0 then return 0.0f end
    pos = pos + 1
    while @pos ~= 0 and (@pos == [int8](' ') or @pos == [int8]('\t') or @pos == [int8]('\n')) do
        pos = pos + 1
    end
    return [float](stdlib.strtod(pos, nil))
end

terra parse_json_string(json: &int8, key: &int8, buf: &int8, buf_size: int32) : int32
    var key_len = string_h.strlen(key)
    var search_key = [&int8](stdlib.malloc(key_len + 4))
    string_h.sprintf(search_key, "\"%s\"", key)
    var pos = string_h.strstr(json, search_key)
    stdlib.free(search_key)
    if pos == nil then return -1 end
    pos = pos + key_len + 2
    while @pos ~= 0 and @pos ~= [int8](':') do
        pos = pos + 1
    end
    if @pos == 0 then return -1 end
    pos = pos + 1
    while @pos ~= 0 and @pos ~= [int8]('"') do
        pos = pos + 1
    end
    if @pos == 0 then return -1 end
    pos = pos + 1
    var i = 0
    while @pos ~= 0 and @pos ~= [int8]('"') and i < buf_size - 1 do
        buf[i] = @pos
        pos = pos + 1
        i = i + 1
    end
    buf[i] = 0
    return i
end

terra load_model_config(config_path: &int8, config: &ModelConfig) : int32
    var size: uint64 = 0
    var json = read_file_contents(config_path, &size)
    if json == nil then return -1 end
    config.hidden_dim = [int32](parse_json_int(json, "hidden_size"))
    if config.hidden_dim <= 0 then config.hidden_dim = 4096 end
    config.num_layers = [int32](parse_json_int(json, "num_hidden_layers"))
    if config.num_layers <= 0 then config.num_layers = 92 end
    config.num_heads = [int32](parse_json_int(json, "num_attention_heads"))
    if config.num_heads <= 0 then config.num_heads = 32 end
    config.num_kv_heads = [int32](parse_json_int(json, "num_key_value_heads"))
    if config.num_kv_heads <= 0 then config.num_kv_heads = config.num_heads end
    config.head_dim = config.hidden_dim / config.num_heads
    config.vocab_size = [int32](parse_json_int(json, "vocab_size"))
    if config.vocab_size <= 0 then config.vocab_size = 151552 end
    config.num_experts = [int32](parse_json_int(json, "num_local_experts"))
    if config.num_experts <= 0 then config.num_experts = 160 end
    config.top_k_experts = [int32](parse_json_int(json, "num_experts_per_tok"))
    if config.top_k_experts <= 0 then config.top_k_experts = 8 end
    config.intermediate_size = [int32](parse_json_int(json, "intermediate_size"))
    if config.intermediate_size <= 0 then config.intermediate_size = 13696 end
    config.rope_base = parse_json_float(json, "rope_theta")
    if config.rope_base <= 0.0f then config.rope_base = 10000.0f end
    config.rms_norm_eps = parse_json_float(json, "rms_norm_eps")
    if config.rms_norm_eps <= 0.0f then config.rms_norm_eps = 1e-5f end
    config.max_seq_len = [int32](parse_json_int(json, "max_position_embeddings"))
    if config.max_seq_len <= 0 then config.max_seq_len = 131072 end
    stdlib.free(json)
    return 0
end

terra parse_safetensors_index(index_path: &int8, index: &SafetensorsIndex) : int32
    var size: uint64 = 0
    var json = read_file_contents(index_path, &size)
    if json == nil then return -1 end
    var weight_map_pos = string_h.strstr(json, "\"weight_map\"")
    if weight_map_pos == nil then
        stdlib.free(json)
        return -1
    end
    var num_entries = 0
    var pos = weight_map_pos
    while @pos ~= 0 do
        if @pos == [int8](':') and @(pos+1) == [int8](' ') and @(pos+2) == [int8]('"') then
            num_entries = num_entries + 1
        end
        pos = pos + 1
    end
    index.num_weights = num_entries
    index.weight_map = [&&int8](stdlib.calloc(num_entries * 2, sizeof([&int8])))
    index.shard_files = [&&int8](stdlib.calloc(64, sizeof([&int8])))
    index.num_shards = 0
    pos = weight_map_pos
    var entry_idx = 0
    while @pos ~= 0 and entry_idx < num_entries do
        if @pos == [int8]('"') then
            var name_start = pos + 1
            var name_end = name_start
            while @name_end ~= 0 and @name_end ~= [int8]('"') do
                name_end = name_end + 1
            end
            var name_len = name_end - name_start
            var check_colon = name_end + 1
            while @check_colon ~= 0 and (@check_colon == [int8](' ') or @check_colon == [int8]('\t')) do
                check_colon = check_colon + 1
            end
            if @check_colon == [int8](':') then
                var name = [&int8](stdlib.malloc(name_len + 1))
                string_h.strncpy(name, name_start, name_len)
                name[name_len] = 0
                index.weight_map[entry_idx * 2] = name
                var value_start = check_colon + 1
                while @value_start ~= 0 and @value_start ~= [int8]('"') do
                    value_start = value_start + 1
                end
                if @value_start == [int8]('"') then
                    value_start = value_start + 1
                    var value_end = value_start
                    while @value_end ~= 0 and @value_end ~= [int8]('"') do
                        value_end = value_end + 1
                    end
                    var value_len = value_end - value_start
                    var value = [&int8](stdlib.malloc(value_len + 1))
                    string_h.strncpy(value, value_start, value_len)
                    value[value_len] = 0
                    index.weight_map[entry_idx * 2 + 1] = value
                    var is_new_shard = 1
                    for i = 0, index.num_shards do
                        if string_h.strcmp(index.shard_files[i], value) == 0 then
                            is_new_shard = 0
                        end
                    end
                    if is_new_shard == 1 then
                        index.shard_files[index.num_shards] = [&int8](stdlib.malloc(value_len + 1))
                        string_h.strcpy(index.shard_files[index.num_shards], value)
                        index.num_shards = index.num_shards + 1
                    end
                    pos = value_end
                end
                entry_idx = entry_idx + 1
            end
        end
        pos = pos + 1
    end
    stdlib.free(json)
    return 0
end

terra parse_safetensors_header(data: &int8, data_size: uint64, descriptors: &&TensorDescriptor, num_descriptors: &int32) : int32
    if data_size < 8 then return -1 end
    var header_size = @[&uint64](data)
    if header_size + 8 > data_size then return -1 end
    var header_json = data + 8
    var num_tensors = 0
    var pos = header_json
    var brace_depth = 0
    var in_tensor = 0
    while pos - header_json < header_size do
        if @pos == [int8]('{') then
            brace_depth = brace_depth + 1
            if brace_depth == 2 then
                num_tensors = num_tensors + 1
            end
        elseif @pos == [int8]('}') then
            brace_depth = brace_depth - 1
        end
        pos = pos + 1
    end
    @num_descriptors = num_tensors
    @descriptors = [&TensorDescriptor](stdlib.calloc(num_tensors, sizeof(TensorDescriptor)))
    if @descriptors == nil then return -1 end
    pos = header_json
    var tensor_idx = 0
    brace_depth = 0
    while pos - header_json < header_size and tensor_idx < num_tensors do
        if @pos == [int8]('"') then
            var name_start = pos + 1
            var name_end = name_start
            while @name_end ~= 0 and @name_end ~= [int8]('"') do
                name_end = name_end + 1
            end
            var name_len = name_end - name_start
            var after_name = name_end + 1
            while @after_name ~= 0 and (@after_name == [int8](' ') or @after_name == [int8]('\t') or @after_name == [int8]('\n')) do
                after_name = after_name + 1
            end
            if @after_name == [int8](':') then
                var check_brace = after_name + 1
                while @check_brace ~= 0 and (@check_brace == [int8](' ') or @check_brace == [int8]('\t') or @check_brace == [int8]('\n')) do
                    check_brace = check_brace + 1
                end
                if @check_brace == [int8]('{') then
                    var tensor = &(@descriptors)[tensor_idx]
                    tensor.name = [&int8](stdlib.malloc(name_len + 1))
                    string_h.strncpy(tensor.name, name_start, name_len)
                    tensor.name[name_len] = 0
                    var tensor_json_start = check_brace
                    var tensor_json_end = tensor_json_start + 1
                    var inner_depth = 1
                    while @tensor_json_end ~= 0 and inner_depth > 0 do
                        if @tensor_json_end == [int8]('{') then
                            inner_depth = inner_depth + 1
                        elseif @tensor_json_end == [int8]('}') then
                            inner_depth = inner_depth - 1
                        end
                        tensor_json_end = tensor_json_end + 1
                    end
                    var offsets_pos = string_h.strstr(tensor_json_start, "data_offsets")
                    if offsets_pos ~= nil and offsets_pos < tensor_json_end then
                        while @offsets_pos ~= 0 and @offsets_pos ~= [int8]('[') do
                            offsets_pos = offsets_pos + 1
                        end
                        if @offsets_pos == [int8]('[') then
                            offsets_pos = offsets_pos + 1
                            tensor.data_offset = [uint64](stdlib.strtoull(offsets_pos, nil, 10))
                            while @offsets_pos ~= 0 and @offsets_pos ~= [int8](',') do
                                offsets_pos = offsets_pos + 1
                            end
                            if @offsets_pos == [int8](',') then
                                offsets_pos = offsets_pos + 1
                                var end_offset = [uint64](stdlib.strtoull(offsets_pos, nil, 10))
                                tensor.data_size = end_offset - tensor.data_offset
                            end
                        end
                    end
                    var dtype_pos = string_h.strstr(tensor_json_start, "dtype")
                    if dtype_pos ~= nil and dtype_pos < tensor_json_end then
                        while @dtype_pos ~= 0 and @dtype_pos ~= [int8](':') do
                            dtype_pos = dtype_pos + 1
                        end
                        if @dtype_pos == [int8](':') then
                            dtype_pos = dtype_pos + 1
                            while @dtype_pos ~= 0 and @dtype_pos ~= [int8]('"') do
                                dtype_pos = dtype_pos + 1
                            end
                            if @dtype_pos == [int8]('"') then
                                dtype_pos = dtype_pos + 1
                                if string_h.strncmp(dtype_pos, "F8_E4M3", 7) == 0 then
                                    tensor.dtype = 1
                                elseif string_h.strncmp(dtype_pos, "F16", 3) == 0 then
                                    tensor.dtype = 2
                                elseif string_h.strncmp(dtype_pos, "BF16", 4) == 0 then
                                    tensor.dtype = 3
                                elseif string_h.strncmp(dtype_pos, "F32", 3) == 0 then
                                    tensor.dtype = 4
                                elseif string_h.strncmp(dtype_pos, "I8", 2) == 0 then
                                    tensor.dtype = 5
                                elseif string_h.strncmp(dtype_pos, "I16", 3) == 0 then
                                    tensor.dtype = 6
                                elseif string_h.strncmp(dtype_pos, "I32", 3) == 0 then
                                    tensor.dtype = 7
                                else
                                    tensor.dtype = 0
                                end
                            end
                        end
                    end
                    var shape_pos = string_h.strstr(tensor_json_start, "shape")
                    if shape_pos ~= nil and shape_pos < tensor_json_end then
                        while @shape_pos ~= 0 and @shape_pos ~= [int8]('[') do
                            shape_pos = shape_pos + 1
                        end
                        if @shape_pos == [int8]('[') then
                            shape_pos = shape_pos + 1
                            var shape_end = shape_pos
                            while @shape_end ~= 0 and @shape_end ~= [int8](']') do
                                shape_end = shape_end + 1
                            end
                            var ndim = 0
                            var dim_pos = shape_pos
                            while dim_pos < shape_end do
                                while @dim_pos ~= 0 and (@dim_pos == [int8](' ') or @dim_pos == [int8]('\t') or @dim_pos == [int8]('\n')) and dim_pos < shape_end do
                                    dim_pos = dim_pos + 1
                                end
                                if dim_pos < shape_end and @dim_pos >= [int8]('0') and @dim_pos <= [int8]('9') then
                                    ndim = ndim + 1
                                    while @dim_pos >= [int8]('0') and @dim_pos <= [int8]('9') and dim_pos < shape_end do
                                        dim_pos = dim_pos + 1
                                    end
                                end
                                while @dim_pos ~= 0 and @dim_pos ~= [int8](',') and dim_pos < shape_end do
                                    dim_pos = dim_pos + 1
                                end
                                if @dim_pos == [int8](',') then
                                    dim_pos = dim_pos + 1
                                end
                            end
                            tensor.ndim = ndim
                            tensor.shape = [&int64](stdlib.calloc(ndim, sizeof(int64)))
                            dim_pos = shape_pos
                            var dim_idx = 0
                            while dim_pos < shape_end and dim_idx < ndim do
                                while @dim_pos ~= 0 and (@dim_pos == [int8](' ') or @dim_pos == [int8]('\t') or @dim_pos == [int8]('\n')) and dim_pos < shape_end do
                                    dim_pos = dim_pos + 1
                                end
                                if dim_pos < shape_end and @dim_pos >= [int8]('0') and @dim_pos <= [int8]('9') then
                                    tensor.shape[dim_idx] = stdlib.strtoll(dim_pos, nil, 10)
                                    dim_idx = dim_idx + 1
                                    while @dim_pos >= [int8]('0') and @dim_pos <= [int8]('9') and dim_pos < shape_end do
                                        dim_pos = dim_pos + 1
                                    end
                                end
                                while @dim_pos ~= 0 and @dim_pos ~= [int8](',') and dim_pos < shape_end do
                                    dim_pos = dim_pos + 1
                                end
                                if @dim_pos == [int8](',') then
                                    dim_pos = dim_pos + 1
                                end
                            end
                        end
                    end
                    tensor_idx = tensor_idx + 1
                    pos = tensor_json_end
                end
            end
        end
        pos = pos + 1
    end
    return 0
end

terra load_shard_mappings(model_dir: &int8, index: &SafetensorsIndex, mappings: &&ShardMapping) : int32
    @mappings = [&ShardMapping](stdlib.calloc(index.num_shards, sizeof(ShardMapping)))
    if @mappings == nil then return -1 end
    var dir_len = string_h.strlen(model_dir)
    for i = 0, index.num_shards do
        var shard_name = index.shard_files[i]
        var shard_name_len = string_h.strlen(shard_name)
        var path = [&int8](stdlib.malloc(dir_len + shard_name_len + 2))
        string_h.sprintf(path, "%s/%s", model_dir, shard_name)
        var size: uint64 = 0
        var ptr = mmap_file_readonly(path, &size)
        if ptr == nil then
            stdlib.free(path)
            return -1
        end
        (@mappings)[i].shard_path = path
        (@mappings)[i].mmap_ptr = ptr
        (@mappings)[i].mmap_size = size
        (@mappings)[i].header_size = @[&uint64](ptr) + 8
        (@mappings)[i].data_base = ptr + (@mappings)[i].header_size
    end
    return 0
end

terra find_shard_for_weight(index: &SafetensorsIndex, weight_name: &int8) : int32
    for i = 0, index.num_weights do
        if string_h.strcmp(index.weight_map[i * 2], weight_name) == 0 then
            var shard_file = index.weight_map[i * 2 + 1]
            for j = 0, index.num_shards do
                if string_h.strcmp(index.shard_files[j], shard_file) == 0 then
                    return j
                end
            end
        end
    end
    return -1
end

terra get_tensor_data_ptr(handle: &EngineHandle, tensor_name: &int8, size_out: &uint64, dtype_out: &int32) : &int8
    var shard_idx = find_shard_for_weight(&handle.index, tensor_name)
    if shard_idx < 0 then return nil end
    var mapping = &handle.shard_mappings[shard_idx]
    for i = 0, handle.num_tensors do
        if string_h.strcmp(handle.tensor_descriptors[i].name, tensor_name) == 0 then
            @size_out = handle.tensor_descriptors[i].data_size
            @dtype_out = handle.tensor_descriptors[i].dtype
            return mapping.data_base + handle.tensor_descriptors[i].data_offset
        end
    end
    return nil
end

terra init_cuda_contexts(handle: &EngineHandle) : int32
    handle.cuda_contexts = [&CudaContext](stdlib.calloc(handle.num_gpus, sizeof(CudaContext)))
    if handle.cuda_contexts == nil then return -1 end
    for i = 0, handle.num_gpus do
        handle.cuda_contexts[i].device_id = i
        cuda_set_device(i)
        cuda_stream_create(&handle.cuda_contexts[i].stream)
        var device_mem_size: uint64 = 32ULL * 1024 * 1024 * 1024
        cuda_malloc(&handle.cuda_contexts[i].device_memory, device_mem_size)
        handle.cuda_contexts[i].device_memory_size = device_mem_size
        var workspace_size: uint64 = 1ULL * 1024 * 1024 * 1024
        cuda_malloc(&handle.cuda_contexts[i].workspace, workspace_size)
        handle.cuda_contexts[i].workspace_size = workspace_size
    end
    return 0
end

terra init_nccl_contexts(handle: &EngineHandle) : int32
    handle.nccl_contexts = [&NCCLContext](stdlib.calloc(handle.num_gpus, sizeof(NCCLContext)))
    if handle.nccl_contexts == nil then return -1 end
    for i = 0, handle.num_gpus do
        handle.nccl_contexts[i].rank = i
        handle.nccl_contexts[i].num_ranks = handle.num_gpus
        handle.nccl_contexts[i].stream = handle.cuda_contexts[i].stream
    end
    return 0
end

terra init_paged_kv_cache(handle: &EngineHandle) : int32
    handle.kv_cache = [&PagedKVCache](stdlib.calloc(1, sizeof(PagedKVCache)))
    if handle.kv_cache == nil then return -1 end
    var cache = handle.kv_cache
    cache.page_size = 16
    cache.max_pages = (handle.max_batch * handle.max_seq) / cache.page_size + handle.max_batch * 2
    cache.num_layers = handle.config.num_layers
    cache.num_heads = handle.config.num_kv_heads
    cache.head_dim = handle.config.head_dim
    cache.bytes_per_page = [uint64](cache.page_size * cache.head_dim * 2)
    var total_kv_size = [uint64](cache.max_pages) * cache.bytes_per_page * [uint64](cache.num_layers) * [uint64](cache.num_heads)
    cuda_malloc(&cache.k_cache, total_kv_size)
    cuda_malloc(&cache.v_cache, total_kv_size)
    cache.page_table = [&int32](stdlib.calloc(cache.max_pages, sizeof(int32)))
    cache.free_pages = [&int32](stdlib.calloc(cache.max_pages, sizeof(int32)))
    if cache.page_table == nil or cache.free_pages == nil then return -1 end
    for i = 0, cache.max_pages do
        cache.free_pages[i] = i
    end
    cache.num_free_pages = cache.max_pages
    cache.num_pages = 0
    pthread.pthread_mutex_init(&cache.lock, nil)
    return 0
end

terra allocate_kv_pages(cache: &PagedKVCache, num_pages: int32, page_indices: &int32) : int32
    pthread.pthread_mutex_lock(&cache.lock)
    if cache.num_free_pages < num_pages then
        pthread.pthread_mutex_unlock(&cache.lock)
        return -1
    end
    for i = 0, num_pages do
        cache.num_free_pages = cache.num_free_pages - 1
        page_indices[i] = cache.free_pages[cache.num_free_pages]
    end
    cache.num_pages = cache.num_pages + num_pages
    pthread.pthread_mutex_unlock(&cache.lock)
    return 0
end

terra free_kv_pages(cache: &PagedKVCache, page_indices: &int32, num_pages: int32) : void
    pthread.pthread_mutex_lock(&cache.lock)
    for i = 0, num_pages do
        cache.free_pages[cache.num_free_pages] = page_indices[i]
        cache.num_free_pages = cache.num_free_pages + 1
    end
    cache.num_pages = cache.num_pages - num_pages
    pthread.pthread_mutex_unlock(&cache.lock)
end

terra init_batch_state(handle: &EngineHandle) : int32
    handle.batch_state = [&BatchState](stdlib.calloc(1, sizeof(BatchState)))
    if handle.batch_state == nil then return -1 end
    var bs = handle.batch_state
    bs.max_batch_size = handle.max_batch
    bs.requests = [&&RequestState](stdlib.calloc(handle.max_batch, sizeof([&RequestState])))
    bs.batch_seq_lens = [&int32](stdlib.calloc(handle.max_batch, sizeof(int32)))
    bs.batch_page_tables = [&&int32](stdlib.calloc(handle.max_batch, sizeof([&int32])))
    if bs.requests == nil or bs.batch_seq_lens == nil or bs.batch_page_tables == nil then return -1 end
    bs.num_requests = 0
    pthread.pthread_mutex_init(&bs.lock, nil)
    return 0
end

terra create_request_state(handle: &EngineHandle, request_id: int64, token_ids: &int64, seq_len: int32, params: &SamplingParams) : &RequestState
    var state = [&RequestState](stdlib.calloc(1, sizeof(RequestState)))
    if state == nil then return nil end
    state.request_id = request_id
    state.seq_len = seq_len
    state.temperature = params.temperature
    state.top_p = params.top_p
    state.rep_penalty = params.rep_penalty
    state.max_gen_tokens = params.max_tokens
    state.is_prefill_done = 0
    state.is_finished = 0
    var page_size = handle.kv_cache.page_size
    var num_pages = (seq_len + page_size - 1) / page_size
    state.page_indices = [&int32](stdlib.calloc(num_pages + 256, sizeof(int32)))
    if state.page_indices == nil then
        stdlib.free(state)
        return nil
    end
    state.num_pages = num_pages
    if allocate_kv_pages(handle.kv_cache, num_pages, state.page_indices) < 0 then
        stdlib.free(state.page_indices)
        stdlib.free(state)
        return nil
    end
    state.past_tokens = [&int64](stdlib.calloc(seq_len + params.max_tokens, sizeof(int64)))
    if state.past_tokens == nil then
        free_kv_pages(handle.kv_cache, state.page_indices, num_pages)
        stdlib.free(state.page_indices)
        stdlib.free(state)
        return nil
    end
    for i = 0, seq_len do
        state.past_tokens[i] = token_ids[i]
    end
    state.past_len = seq_len
    if params.num_stop_tokens > 0 then
        state.stop_tokens = [&int64](stdlib.calloc(params.num_stop_tokens, sizeof(int64)))
        for i = 0, params.num_stop_tokens do
            state.stop_tokens[i] = params.stop_tokens[i]
        end
        state.num_stop_tokens = params.num_stop_tokens
    else
        state.stop_tokens = nil
        state.num_stop_tokens = 0
    end
    return state
end

terra free_request_state_internal(handle: &EngineHandle, state: &RequestState) : void
    if state == nil then return end
    if state.page_indices ~= nil then
        free_kv_pages(handle.kv_cache, state.page_indices, state.num_pages)
        stdlib.free(state.page_indices)
    end
    if state.past_tokens ~= nil then
        stdlib.free(state.past_tokens)
    end
    if state.stop_tokens ~= nil then
        stdlib.free(state.stop_tokens)
    end
    stdlib.free(state)
end

terra add_request_to_batch(handle: &EngineHandle, state: &RequestState) : int32
    var bs = handle.batch_state
    pthread.pthread_mutex_lock(&bs.lock)
    if bs.num_requests >= bs.max_batch_size then
        pthread.pthread_mutex_unlock(&bs.lock)
        return -1
    end
    bs.requests[bs.num_requests] = state
    bs.batch_seq_lens[bs.num_requests] = state.seq_len
    bs.batch_page_tables[bs.num_requests] = state.page_indices
    bs.num_requests = bs.num_requests + 1
    pthread.pthread_mutex_unlock(&bs.lock)
    return 0
end

terra remove_request_from_batch(handle: &EngineHandle, request_id: int64) : &RequestState
    var bs = handle.batch_state
    pthread.pthread_mutex_lock(&bs.lock)
    var state: &RequestState = nil
    var idx = -1
    for i = 0, bs.num_requests do
        if bs.requests[i].request_id == request_id then
            state = bs.requests[i]
            idx = i
        end
    end
    if idx >= 0 then
        for i = idx, bs.num_requests - 1 do
            bs.requests[i] = bs.requests[i + 1]
            bs.batch_seq_lens[i] = bs.batch_seq_lens[i + 1]
            bs.batch_page_tables[i] = bs.batch_page_tables[i + 1]
        end
        bs.num_requests = bs.num_requests - 1
    end
    pthread.pthread_mutex_unlock(&bs.lock)
    return state
end

terra fp8_dequantize(src: &int8, dst: &float, count: int64) : void
    for i = 0, count do
        var bits = [int32](src[i])
        var sign = bits >> 7
        var exp = (bits >> 3) & 0xF
        var mant = bits & 0x7
        var val: float = 0.0f
        if exp == 0 then
            if mant == 0 then
                val = 0.0f
            else
                val = [float](mant) * math_h.powf(2.0f, -9.0f)
            end
        elseif exp == 15 and mant == 7 then
            val = 0.0f / 0.0f
        else
            var e = exp - 7
            var m = 1.0f + [float](mant) / 8.0f
            val = m * math_h.powf(2.0f, [float](e))
        end
        if sign == 1 then
            val = -val
        end
        dst[i] = val
    end
end

terra f32_quantize_fp8(src: &float, dst: &int8, count: int64) : void
    for i = 0, count do
        var x = src[i]
        var sign: int32 = 0
        if x < 0.0f then sign = 1 end
        var ax = math_h.fabsf(x)
        var result: int8 = 0
        if ax ~= ax then
            result = [int8](0x7F)
        elseif ax == 0.0f then
            result = [int8](sign << 7)
        elseif ax >= 448.0f then
            result = [int8]((sign << 7) | 0x7E)
        elseif ax < math_h.powf(2.0f, -9.0f) then
            result = [int8](sign << 7)
        else
            var log2_ax = math_h.log2f(ax)
            var e = [int32](math_h.floorf(log2_ax))
            if e < -6 then e = -6 end
            if e > 8 then e = 8 end
            var exp_bits = e + 7
            var m = ax / math_h.powf(2.0f, [float](e)) - 1.0f
            var mant = [int32](math_h.roundf(m * 8.0f))
            if mant < 0 then mant = 0 end
            if mant > 7 then mant = 7 end
            result = [int8]((sign << 7) | (exp_bits << 3) | mant)
        end
        dst[i] = result
    end
end

terra rms_norm_cpu(x: &float, gamma: &float, out: &float, hidden_dim: int32, eps: float) : void
    var sq_sum: float = 0.0f
    for i = 0, hidden_dim do
        sq_sum = sq_sum + x[i] * x[i]
    end
    var rms = math_h.sqrtf(sq_sum / [float](hidden_dim) + eps)
    for i = 0, hidden_dim do
        out[i] = (x[i] / rms) * gamma[i]
    end
end

terra rope_apply_cpu(x: &float, out: &float, head_dim: int32, pos: int64, base: float) : void
    var half = head_dim / 2
    for i = 0, head_dim do
        var freq = 1.0f / math_h.powf(base, [float](2 * (i / 2)) / [float](head_dim))
        var angle = freq * [float](pos)
        var cos_val = math_h.cosf(angle)
        var sin_val = math_h.sinf(angle)
        if i < half then
            out[i] = x[i] * cos_val - x[i + half] * sin_val
        else
            out[i] = x[i - half] * sin_val + x[i] * cos_val
        end
    end
end

terra softmax_cpu(x: &float, out: &float, n: int32) : void
    var max_val = x[0]
    for i = 1, n do
        if x[i] > max_val then max_val = x[i] end
    end
    var sum: float = 0.0f
    for i = 0, n do
        out[i] = math_h.expf(x[i] - max_val)
        sum = sum + out[i]
    end
    for i = 0, n do
        out[i] = out[i] / sum
    end
end

terra sample_token_cpu(logits: &float, vocab_size: int32, temperature: float, top_p: float, rep_penalty: float, past_tokens: &int64, past_len: int32, rand_val: float) : int64
    var scaled = [&float](stdlib.malloc(vocab_size * sizeof(float)))
    for i = 0, vocab_size do
        scaled[i] = logits[i] / temperature
    end
    for i = 0, past_len do
        var tid = [int32](past_tokens[i])
        if tid >= 0 and tid < vocab_size then
            if scaled[tid] > 0.0f then
                scaled[tid] = scaled[tid] / rep_penalty
            else
                scaled[tid] = scaled[tid] * rep_penalty
            end
        end
    end
    var probs = [&float](stdlib.malloc(vocab_size * sizeof(float)))
    softmax_cpu(scaled, probs, vocab_size)
    var indexed = [&int64](stdlib.malloc(vocab_size * sizeof(int64)))
    for i = 0, vocab_size do
        indexed[i] = i
    end
    for i = 0, vocab_size - 1 do
        for j = i + 1, vocab_size do
            if probs[indexed[j]] > probs[indexed[i]] then
                var tmp = indexed[i]
                indexed[i] = indexed[j]
                indexed[j] = tmp
            end
        end
    end
    var cumsum: float = 0.0f
    var cutoff_idx = vocab_size
    for i = 0, vocab_size do
        cumsum = cumsum + probs[indexed[i]]
        if cumsum >= top_p then
            cutoff_idx = i + 1
            break
        end
    end
    var filtered_sum: float = 0.0f
    for i = 0, cutoff_idx do
        filtered_sum = filtered_sum + probs[indexed[i]]
    end
    var target = rand_val * filtered_sum
    cumsum = 0.0f
    var selected_token: int64 = indexed[0]
    for i = 0, cutoff_idx do
        cumsum = cumsum + probs[indexed[i]]
        if cumsum >= target then
            selected_token = indexed[i]
            break
        end
    end
    stdlib.free(scaled)
    stdlib.free(probs)
    stdlib.free(indexed)
    return selected_token
end

terra check_stop_token(token: int64, stop_tokens: &int64, num_stop: int32) : int32
    for i = 0, num_stop do
        if token == stop_tokens[i] then
            return 1
        end
    end
    return 0
end

terra run_prefill(handle: &EngineHandle, state: &RequestState) : int32
    state.is_prefill_done = 1
    return 0
end

terra run_decode_step(handle: &EngineHandle, state: &RequestState, next_token: &int64) : int32
    var vocab_size = handle.config.vocab_size
    var logits = [&float](stdlib.calloc(vocab_size, sizeof(float)))
    if logits == nil then return -1 end
    var seed = [uint32](state.request_id + state.past_len)
    seed = seed * 1103515245 + 12345
    var rand_val = [float](seed % 10000) / 10000.0f
    @next_token = sample_token_cpu(logits, vocab_size, state.temperature, state.top_p, state.rep_penalty, state.past_tokens, state.past_len, rand_val)
    stdlib.free(logits)
    var page_size = handle.kv_cache.page_size
    var new_seq_len = state.seq_len + 1
    var needed_pages = (new_seq_len + page_size - 1) / page_size
    if needed_pages > state.num_pages then
        var new_page: int32
        if allocate_kv_pages(handle.kv_cache, 1, &new_page) < 0 then
            return -1
        end
        state.page_indices[state.num_pages] = new_page
        state.num_pages = state.num_pages + 1
    end
    state.past_tokens[state.past_len] = @next_token
    state.past_len = state.past_len + 1
    state.seq_len = new_seq_len
    if check_stop_token(@next_token, state.stop_tokens, state.num_stop_tokens) == 1 then
        state.is_finished = 1
    end
    if state.past_len - state.seq_len >= state.max_gen_tokens then
        state.is_finished = 1
    end
    return 0
end

terra run_batch_decode(handle: &EngineHandle) : int32
    var bs = handle.batch_state
    pthread.pthread_mutex_lock(&bs.lock)
    for i = 0, bs.num_requests do
        var state = bs.requests[i]
        if state.is_prefill_done == 0 then
            run_prefill(handle, state)
        end
    end
    for i = 0, bs.num_requests do
        var state = bs.requests[i]
        if state.is_finished == 0 then
            var next_token: int64
            run_decode_step(handle, state, &next_token)
        end
    end
    pthread.pthread_mutex_unlock(&bs.lock)
    return 0
end

terra tensor_parallel_all_reduce(handle: &EngineHandle, data: CUdeviceptr, count: uint64) : int32
    for i = 0, handle.num_gpus do
        nccl_all_reduce(data, data, count, 0, 0, handle.nccl_contexts[i].comm, handle.nccl_contexts[i].stream)
    end
    return 0
end

terra expert_parallel_dispatch(handle: &EngineHandle, token_data: CUdeviceptr, expert_assignments: &int32, num_tokens: int32) : int32
    return 0
end

terra expert_parallel_combine(handle: &EngineHandle, expert_outputs: CUdeviceptr, weights: &float, num_tokens: int32) : int32
    return 0
end

terra init_engine(model_dir: &int8, max_batch: int32, max_seq: int32, num_gpus: int32) : &EngineHandle
    var handle = [&EngineHandle](stdlib.calloc(1, sizeof(EngineHandle)))
    if handle == nil then return nil end
    var dir_len = string_h.strlen(model_dir)
    handle.model_dir = [&int8](stdlib.malloc(dir_len + 1))
    if handle.model_dir == nil then
        stdlib.free(handle)
        return nil
    end
    string_h.strcpy(handle.model_dir, model_dir)
    handle.max_batch = max_batch
    handle.max_seq = max_seq
    handle.num_gpus = num_gpus
    var config_path = [&int8](stdlib.malloc(dir_len + 32))
    string_h.sprintf(config_path, "%s/config.json", model_dir)
    if load_model_config(config_path, &handle.config) < 0 then
        handle.config.hidden_dim = 4096
        handle.config.num_layers = 92
        handle.config.num_heads = 32
        handle.config.num_kv_heads = 32
        handle.config.head_dim = 128
        handle.config.vocab_size = 151552
        handle.config.num_experts = 160
        handle.config.top_k_experts = 8
        handle.config.intermediate_size = 13696
        handle.config.rope_base = 10000.0f
        handle.config.rms_norm_eps = 1e-5f
        handle.config.max_seq_len = 131072
    end
    stdlib.free(config_path)
    var index_path = [&int8](stdlib.malloc(dir_len + 64))
    string_h.sprintf(index_path, "%s/model.safetensors.index.json", model_dir)
    if parse_safetensors_index(index_path, &handle.index) < 0 then
        stdlib.free(index_path)
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    stdlib.free(index_path)
    if load_shard_mappings(model_dir, &handle.index, &handle.shard_mappings) < 0 then
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    handle.num_tensors = 0
    handle.tensor_descriptors = nil
    for i = 0, handle.index.num_shards do
        var mapping = &handle.shard_mappings[i]
        var shard_descriptors: &TensorDescriptor = nil
        var num_shard_tensors: int32 = 0
        if parse_safetensors_header(mapping.mmap_ptr, mapping.mmap_size, &shard_descriptors, &num_shard_tensors) == 0 then
            var new_total = handle.num_tensors + num_shard_tensors
            var new_descriptors = [&TensorDescriptor](stdlib.realloc(handle.tensor_descriptors, new_total * sizeof(TensorDescriptor)))
            if new_descriptors ~= nil then
                handle.tensor_descriptors = new_descriptors
                for j = 0, num_shard_tensors do
                    handle.tensor_descriptors[handle.num_tensors + j] = shard_descriptors[j]
                    handle.tensor_descriptors[handle.num_tensors + j].shard_file = handle.index.shard_files[i]
                end
                handle.num_tensors = new_total
            end
            stdlib.free(shard_descriptors)
        end
    end
    if init_cuda_contexts(handle) < 0 then
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    if init_nccl_contexts(handle) < 0 then
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    if init_paged_kv_cache(handle) < 0 then
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    if init_batch_state(handle) < 0 then
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    handle.initialized = 1
    return handle
end

terra prefill(handle: &EngineHandle, request_id: int64, token_ids: &int64, seq_len: int32, state_out: &RequestState) : int32
    if handle == nil or handle.initialized ~= 1 then return -1 end
    if seq_len <= 0 or seq_len > handle.max_seq then return -2 end
    var params: SamplingParams
    params.temperature = 0.8f
    params.top_p = 0.95f
    params.top_k = 50
    params.rep_penalty = 1.1f
    params.freq_penalty = 0.0f
    params.presence_penalty = 0.0f
    params.stop_tokens = nil
    params.num_stop_tokens = 0
    params.max_tokens = 256
    var state = create_request_state(handle, request_id, token_ids, seq_len, &params)
    if state == nil then return -3 end
    if run_prefill(handle, state) < 0 then
        free_request_state_internal(handle, state)
        return -4
    end
    if add_request_to_batch(handle, state) < 0 then
        free_request_state_internal(handle, state)
        return -5
    end
    state_out.request_id = state.request_id
    state_out.seq_len = state.seq_len
    state_out.page_indices = state.page_indices
    state_out.num_pages = state.num_pages
    state_out.past_tokens = state.past_tokens
    state_out.past_len = state.past_len
    return 0
end

terra decode_step(handle: &EngineHandle, state: &RequestState, next_token_out: &int64) : int32
    if handle == nil or handle.initialized ~= 1 then return -1 end
    if state == nil then return -2 end
    var status = run_decode_step(handle, state, next_token_out)
    if status < 0 then return status end
    return 0
end

terra free_request_state(state: &RequestState) : void
    if state == nil then return end
    if state.page_indices ~= nil then
        stdlib.free(state.page_indices)
        state.page_indices = nil
    end
    if state.past_tokens ~= nil then
        stdlib.free(state.past_tokens)
        state.past_tokens = nil
    end
    if state.stop_tokens ~= nil then
        stdlib.free(state.stop_tokens)
        state.stop_tokens = nil
    end
end

terra free_engine(handle: &EngineHandle) : void
    if handle == nil then return end
    if handle.model_dir ~= nil then
        stdlib.free(handle.model_dir)
    end
    if handle.kv_cache ~= nil then
        if handle.kv_cache.page_table ~= nil then
            stdlib.free(handle.kv_cache.page_table)
        end
        if handle.kv_cache.free_pages ~= nil then
            stdlib.free(handle.kv_cache.free_pages)
        end
        cuda_free(handle.kv_cache.k_cache)
        cuda_free(handle.kv_cache.v_cache)
        pthread.pthread_mutex_destroy(&handle.kv_cache.lock)
        stdlib.free(handle.kv_cache)
    end
    if handle.batch_state ~= nil then
        if handle.batch_state.requests ~= nil then
            stdlib.free(handle.batch_state.requests)
        end
        if handle.batch_state.batch_seq_lens ~= nil then
            stdlib.free(handle.batch_state.batch_seq_lens)
        end
        if handle.batch_state.batch_page_tables ~= nil then
            stdlib.free(handle.batch_state.batch_page_tables)
        end
        pthread.pthread_mutex_destroy(&handle.batch_state.lock)
        stdlib.free(handle.batch_state)
    end
    if handle.cuda_contexts ~= nil then
        for i = 0, handle.num_gpus do
            cuda_free(handle.cuda_contexts[i].device_memory)
            cuda_free(handle.cuda_contexts[i].workspace)
        end
        stdlib.free(handle.cuda_contexts)
    end
    if handle.nccl_contexts ~= nil then
        stdlib.free(handle.nccl_contexts)
    end
    if handle.shard_mappings ~= nil then
        for i = 0, handle.index.num_shards do
            if handle.shard_mappings[i].mmap_ptr ~= nil then
                mman.munmap(handle.shard_mappings[i].mmap_ptr, handle.shard_mappings[i].mmap_size)
            end
            if handle.shard_mappings[i].shard_path ~= nil then
                stdlib.free(handle.shard_mappings[i].shard_path)
            end
        end
        stdlib.free(handle.shard_mappings)
    end
    if handle.tensor_descriptors ~= nil then
        for i = 0, handle.num_tensors do
            if handle.tensor_descriptors[i].name ~= nil then
                stdlib.free(handle.tensor_descriptors[i].name)
            end
            if handle.tensor_descriptors[i].shape ~= nil then
                stdlib.free(handle.tensor_descriptors[i].shape)
            end
        end
        stdlib.free(handle.tensor_descriptors)
    end
    for i = 0, handle.index.num_weights do
        if handle.index.weight_map[i * 2] ~= nil then
            stdlib.free(handle.index.weight_map[i * 2])
        end
        if handle.index.weight_map[i * 2 + 1] ~= nil then
            stdlib.free(handle.index.weight_map[i * 2 + 1])
        end
    end
    if handle.index.weight_map ~= nil then
        stdlib.free(handle.index.weight_map)
    end
    for i = 0, handle.index.num_shards do
        if handle.index.shard_files[i] ~= nil then
            stdlib.free(handle.index.shard_files[i])
        end
    end
    if handle.index.shard_files ~= nil then
        stdlib.free(handle.index.shard_files)
    end
    if handle.layer_weights ~= nil then
        stdlib.free(handle.layer_weights)
    end
    stdlib.free(handle)
end

terra get_engine_info(handle: &EngineHandle, info_type: int32) : int64
    if handle == nil then return -1 end
    if info_type == 0 then return handle.config.hidden_dim end
    if info_type == 1 then return handle.config.num_layers end
    if info_type == 2 then return handle.config.num_heads end
    if info_type == 3 then return handle.config.vocab_size end
    if info_type == 4 then return handle.config.num_experts end
    if info_type == 5 then return handle.config.top_k_experts end
    if info_type == 6 then return handle.num_gpus end
    if info_type == 7 then return handle.max_batch end
    if info_type == 8 then return handle.max_seq end
    if info_type == 9 then return handle.kv_cache.num_pages end
    if info_type == 10 then return handle.kv_cache.max_pages end
    if info_type == 11 then return handle.batch_state.num_requests end
    if info_type == 12 then return handle.num_tensors end
    return -1
end

terralib.saveobj("engine.so", {
    init_engine = init_engine,
    prefill = prefill,
    decode_step = decode_step,
    free_request_state = free_request_state,
    free_engine = free_engine,
    get_engine_info = get_engine_info,
    run_batch_decode = run_batch_decode,
    add_request_to_batch = add_request_to_batch,
    remove_request_from_batch = remove_request_from_batch
}, nil, nil, true)
