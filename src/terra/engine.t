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
local dlfcn = terralib.includec("dlfcn.h")

local CW_SUCCESS = 0
local NCW_SUCCESS = 0
local CBW_SUCCESS = 0

local cwError_t = int32
local ncwResult_t = int32
local cbwStatus_t = int32
local cwStream_t = &opaque
local cwEvent_t = &opaque
local CUdeviceptr = uint64
local ncwComm_t = &opaque
local cbwHandle_t = &opaque
local cbwLtHandle_t = &opaque

terralib.linklibrary("libcudawrap.so")
terralib.linklibrary("libncclwrap.so")
terralib.linklibrary("libcublaswrap.so")
terralib.linklibrary("libkernels.so")

local cw = terralib.includecstring[[
typedef int cwError_t;
typedef void* cwStream_t;
typedef void* cwEvent_t;
typedef unsigned long long cwDevicePtr_t;
extern cwError_t cwInit(void);
extern cwError_t cwSetDevice(int device);
extern cwError_t cwGetDevice(int* device);
extern cwError_t cwGetDeviceCount(int* count);
extern cwError_t cwDeviceSynchronize(void);
extern cwError_t cwMalloc(cwDevicePtr_t* ptr, unsigned long long size);
extern cwError_t cwFree(cwDevicePtr_t ptr);
extern cwError_t cwMemcpyH2D(cwDevicePtr_t dst, const void* src, unsigned long long size);
extern cwError_t cwMemcpyD2H(void* dst, cwDevicePtr_t src, unsigned long long size);
extern cwError_t cwMemcpyD2D(cwDevicePtr_t dst, cwDevicePtr_t src, unsigned long long size);
extern cwError_t cwMemset(cwDevicePtr_t ptr, int value, unsigned long long size);
extern cwError_t cwStreamCreate(cwStream_t* stream);
extern cwError_t cwStreamDestroy(cwStream_t stream);
extern cwError_t cwStreamSynchronize(cwStream_t stream);
extern cwError_t cwEventCreate(cwEvent_t* event);
extern cwError_t cwEventDestroy(cwEvent_t event);
extern cwError_t cwEventRecord(cwEvent_t event, cwStream_t stream);
extern cwError_t cwEventSynchronize(cwEvent_t event);
extern cwError_t cwEventElapsedTime(float* ms, cwEvent_t start, cwEvent_t end);
extern unsigned long long cwGetFreeMemory(void);
extern unsigned long long cwGetTotalMemory(void);
extern const char* cwGetErrorString(cwError_t error);
]]

local cbw = terralib.includecstring[[
typedef int cbwStatus_t;
typedef void* cbwHandle_t;
typedef void* cbwLtHandle_t;
typedef void* cbwLtMatmulDesc_t;
typedef void* cbwLtMatrixLayout_t;
typedef void* cbwLtMatmulPreference_t;
typedef void* cwStream_t;
extern cbwStatus_t cbwCreate(cbwHandle_t* handle);
extern cbwStatus_t cbwDestroy(cbwHandle_t handle);
extern cbwStatus_t cbwSetStream(cbwHandle_t handle, cwStream_t stream);
extern cbwStatus_t cbwSgemm(cbwHandle_t handle, int transa, int transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
extern cbwStatus_t cbwLtCreate(cbwLtHandle_t* handle);
extern cbwStatus_t cbwLtDestroy(cbwLtHandle_t handle);
extern const char* cbwGetErrorString(cbwStatus_t status);
]]

local kern = terralib.includecstring[[
typedef void* cwStream_t;
extern void launch_fp8_dequantize(const void* input, void* output, unsigned long long count, cwStream_t stream);
extern void launch_f32_to_fp8(const float* input, void* output, unsigned long long count, cwStream_t stream);
extern void launch_rms_norm(const float* input, const float* gamma, float* output, int hidden_dim, int batch_size, float eps, cwStream_t stream);
extern void launch_rope(float* q, float* k, int head_dim, int num_heads, int seq_len, int start_pos, float theta, cwStream_t stream);
extern void launch_softmax(float* input, int batch_size, int seq_len, cwStream_t stream);
extern void launch_swiglu(const float* gate, const float* up, float* output, unsigned long long count, cwStream_t stream);
extern void launch_gelu(float* data, unsigned long long count, cwStream_t stream);
extern void launch_embedding_lookup(const void* embedding_table, const long long* token_ids, float* output, int hidden_dim, int num_tokens, int dtype, cwStream_t stream);
extern void launch_add_residual(float* a, const float* b, unsigned long long count, cwStream_t stream);
extern void launch_argmax(const float* logits, long long* output, int vocab_size, int batch_size, cwStream_t stream);
extern void launch_top_p_sampling(const float* logits, long long* output, float temperature, float top_p, int vocab_size, int batch_size, const unsigned long long* random_seeds, cwStream_t stream);
extern void launch_apply_rep_penalty(float* logits, const long long* past_tokens, int past_len, int vocab_size, float penalty, cwStream_t stream);
]]

struct CudaContext {
    device_id: int32
    stream: cwStream_t
    device_memory: CUdeviceptr
    device_memory_size: uint64
    workspace: CUdeviceptr
    workspace_size: uint64
    cublas_handle: cbwHandle_t
    cublaslt_handle: cbwLtHandle_t
}

struct NCCLContext {
    comm: ncwComm_t
    rank: int32
    num_ranks: int32
    stream: cwStream_t
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

struct LayerWeights {
    input_layernorm: CUdeviceptr
    post_attention_layernorm: CUdeviceptr
    q_proj: CUdeviceptr
    k_proj: CUdeviceptr
    v_proj: CUdeviceptr
    o_proj: CUdeviceptr
    gate_proj: CUdeviceptr
    up_proj: CUdeviceptr
    down_proj: CUdeviceptr
    q_scale: float
    k_scale: float
    v_scale: float
    o_scale: float
    gate_scale: float
    up_scale: float
    down_scale: float
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
    embed_dtype: int32
    lm_head_weight: CUdeviceptr
    lm_head_dtype: int32
    final_norm: CUdeviceptr
    layer_weights: &LayerWeights
    num_layer_weights: int32
    hidden_buffer: CUdeviceptr
    residual_buffer: CUdeviceptr
    qkv_buffer: CUdeviceptr
    attn_out_buffer: CUdeviceptr
    mlp_buffer: CUdeviceptr
    logits_buffer: CUdeviceptr
    initialized: int32
    use_gpu: int32
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
    index.shard_files = [&&int8](stdlib.calloc(256, sizeof([&int8])))
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
                            break
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
    if header_size > 100 * 1024 * 1024 then return -1 end
    var header_json_src = data + 8
    var header_json = [&int8](stdlib.malloc(header_size + 1))
    if header_json == nil then return -1 end
    string_h.memcpy(header_json, header_json_src, header_size)
    header_json[header_size] = 0
    var num_tensors = 0
    var pos = header_json
    var brace_depth = 0
    while pos - header_json < [int64](header_size) do
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
    while pos - header_json < [int64](header_size) and tensor_idx < num_tensors do
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
                            tensor.shape = [&int64](stdlib.calloc(ndim + 1, sizeof(int64)))
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
    stdlib.free(header_json)
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
            C.printf("Failed to mmap shard: %s\n", path)
            stdlib.free(path)
            return -1
        end
        (@mappings)[i].shard_path = path
        (@mappings)[i].mmap_ptr = ptr
        (@mappings)[i].mmap_size = size
        (@mappings)[i].header_size = @[&uint64](ptr) + 8
        (@mappings)[i].data_base = ptr + (@mappings)[i].header_size
        C.printf("Loaded shard %d: %s (%llu bytes)\n", i, shard_name, size)
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
    var device_count: int32 = 0
    if cw.cwGetDeviceCount(&device_count) ~= CW_SUCCESS then
        C.printf("No CUDA devices available, running in CPU mode\n")
        handle.use_gpu = 0
        handle.cuda_contexts = nil
        return 0
    end
    if device_count < handle.num_gpus then
        C.printf("Requested %d GPUs but only %d available, using %d\n", handle.num_gpus, device_count, device_count)
        handle.num_gpus = device_count
    end
    if device_count == 0 then
        handle.use_gpu = 0
        handle.cuda_contexts = nil
        return 0
    end
    handle.use_gpu = 1
    handle.cuda_contexts = [&CudaContext](stdlib.calloc(handle.num_gpus, sizeof(CudaContext)))
    if handle.cuda_contexts == nil then return -1 end
    for i = 0, handle.num_gpus do
        handle.cuda_contexts[i].device_id = i
        if cw.cwSetDevice(i) ~= CW_SUCCESS then
            C.printf("Failed to set device %d\n", i)
            return -1
        end
        if cw.cwStreamCreate(&handle.cuda_contexts[i].stream) ~= CW_SUCCESS then
            C.printf("Failed to create stream for device %d\n", i)
            return -1
        end
        var device_mem_size: uint64 = 4ULL * 1024 * 1024 * 1024
        if cw.cwMalloc(&handle.cuda_contexts[i].device_memory, device_mem_size) ~= CW_SUCCESS then
            C.printf("Failed to allocate device memory for device %d\n", i)
            return -1
        end
        handle.cuda_contexts[i].device_memory_size = device_mem_size
        var workspace_size: uint64 = 512ULL * 1024 * 1024
        if cw.cwMalloc(&handle.cuda_contexts[i].workspace, workspace_size) ~= CW_SUCCESS then
            C.printf("Failed to allocate workspace for device %d\n", i)
            return -1
        end
        handle.cuda_contexts[i].workspace_size = workspace_size
        if cbw.cbwCreate(&handle.cuda_contexts[i].cublas_handle) ~= CBW_SUCCESS then
            C.printf("Failed to create cuBLAS handle for device %d\n", i)
            return -1
        end
        cbw.cbwSetStream(handle.cuda_contexts[i].cublas_handle, handle.cuda_contexts[i].stream)
        if cbw.cbwLtCreate(&handle.cuda_contexts[i].cublaslt_handle) ~= CBW_SUCCESS then
            C.printf("Failed to create cuBLASLt handle for device %d\n", i)
            return -1
        end
        C.printf("Initialized CUDA device %d with %llu MB memory\n", i, device_mem_size / 1024 / 1024)
    end
    return 0
end

terra init_nccl_contexts(handle: &EngineHandle) : int32
    if handle.use_gpu == 0 or handle.num_gpus <= 1 then
        handle.nccl_contexts = nil
        return 0
    end
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
    cache.max_pages = (handle.max_batch * handle.max_seq) / cache.page_size + handle.max_batch * 4
    cache.num_layers = handle.config.num_layers
    cache.num_heads = handle.config.num_kv_heads
    cache.head_dim = handle.config.head_dim
    cache.bytes_per_page = [uint64](cache.page_size) * [uint64](cache.head_dim) * 2 * [uint64](cache.num_heads)
    var total_kv_size = [uint64](cache.max_pages) * cache.bytes_per_page * [uint64](cache.num_layers)
    if handle.use_gpu == 1 then
        if cw.cwMalloc(&cache.k_cache, total_kv_size) ~= CW_SUCCESS then
            C.printf("Failed to allocate K cache: %llu bytes\n", total_kv_size)
            return -1
        end
        if cw.cwMalloc(&cache.v_cache, total_kv_size) ~= CW_SUCCESS then
            C.printf("Failed to allocate V cache: %llu bytes\n", total_kv_size)
            return -1
        end
        cw.cwMemset(cache.k_cache, 0, total_kv_size)
        cw.cwMemset(cache.v_cache, 0, total_kv_size)
    else
        cache.k_cache = [uint64](stdlib.calloc(1, total_kv_size))
        cache.v_cache = [uint64](stdlib.calloc(1, total_kv_size))
    end
    cache.page_table = [&int32](stdlib.calloc(cache.max_pages, sizeof(int32)))
    cache.free_pages = [&int32](stdlib.calloc(cache.max_pages, sizeof(int32)))
    if cache.page_table == nil or cache.free_pages == nil then return -1 end
    for i = 0, cache.max_pages do
        cache.free_pages[i] = i
    end
    cache.num_free_pages = cache.max_pages
    cache.num_pages = 0
    pthread.pthread_mutex_init(&cache.lock, nil)
    C.printf("Initialized KV cache: %d pages, %llu bytes per page, %llu total MB\n", cache.max_pages, cache.bytes_per_page, total_kv_size / 1024 / 1024)
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

terra load_weights_to_gpu(handle: &EngineHandle) : int32
    var size: uint64 = 0
    var dtype: int32 = 0
    var embed_ptr = get_tensor_data_ptr(handle, "model.embed_tokens.weight", &size, &dtype)
    if embed_ptr ~= nil then
        C.printf("Loading embedding weight: %llu bytes, dtype=%d\n", size, dtype)
        if handle.use_gpu == 1 then
            if cw.cwMalloc(&handle.embed_weight, size) ~= CW_SUCCESS then
                C.printf("Failed to allocate embedding weight on GPU\n")
                return -1
            end
            if cw.cwMemcpyH2D(handle.embed_weight, embed_ptr, size) ~= CW_SUCCESS then
                C.printf("Failed to copy embedding weight to GPU\n")
                return -1
            end
        else
            handle.embed_weight = [uint64](embed_ptr)
        end
        handle.embed_dtype = dtype
    else
        C.printf("Warning: embed_tokens.weight not found\n")
    end
    var lm_head_ptr = get_tensor_data_ptr(handle, "lm_head.weight", &size, &dtype)
    if lm_head_ptr ~= nil then
        C.printf("Loading lm_head weight: %llu bytes, dtype=%d\n", size, dtype)
        if handle.use_gpu == 1 then
            if cw.cwMalloc(&handle.lm_head_weight, size) ~= CW_SUCCESS then
                C.printf("Failed to allocate lm_head weight on GPU\n")
                return -1
            end
            if cw.cwMemcpyH2D(handle.lm_head_weight, lm_head_ptr, size) ~= CW_SUCCESS then
                C.printf("Failed to copy lm_head weight to GPU\n")
                return -1
            end
        else
            handle.lm_head_weight = [uint64](lm_head_ptr)
        end
        handle.lm_head_dtype = dtype
    else
        C.printf("Warning: lm_head.weight not found\n")
    end
    var norm_ptr = get_tensor_data_ptr(handle, "model.norm.weight", &size, &dtype)
    if norm_ptr ~= nil then
        C.printf("Loading final norm weight: %llu bytes\n", size)
        if handle.use_gpu == 1 then
            if cw.cwMalloc(&handle.final_norm, size) ~= CW_SUCCESS then
                return -1
            end
            cw.cwMemcpyH2D(handle.final_norm, norm_ptr, size)
        else
            handle.final_norm = [uint64](norm_ptr)
        end
    end
    return 0
end

terra allocate_inference_buffers(handle: &EngineHandle) : int32
    var hidden_size = [uint64](handle.max_batch) * [uint64](handle.max_seq) * [uint64](handle.config.hidden_dim) * 4
    var vocab_size = [uint64](handle.max_batch) * [uint64](handle.config.vocab_size) * 4
    if handle.use_gpu == 1 then
        if cw.cwMalloc(&handle.hidden_buffer, hidden_size) ~= CW_SUCCESS then return -1 end
        if cw.cwMalloc(&handle.residual_buffer, hidden_size) ~= CW_SUCCESS then return -1 end
        if cw.cwMalloc(&handle.qkv_buffer, hidden_size * 3) ~= CW_SUCCESS then return -1 end
        if cw.cwMalloc(&handle.attn_out_buffer, hidden_size) ~= CW_SUCCESS then return -1 end
        if cw.cwMalloc(&handle.mlp_buffer, hidden_size * 4) ~= CW_SUCCESS then return -1 end
        if cw.cwMalloc(&handle.logits_buffer, vocab_size) ~= CW_SUCCESS then return -1 end
    else
        handle.hidden_buffer = [uint64](stdlib.calloc(1, hidden_size))
        handle.residual_buffer = [uint64](stdlib.calloc(1, hidden_size))
        handle.qkv_buffer = [uint64](stdlib.calloc(1, hidden_size * 3))
        handle.attn_out_buffer = [uint64](stdlib.calloc(1, hidden_size))
        handle.mlp_buffer = [uint64](stdlib.calloc(1, hidden_size * 4))
        handle.logits_buffer = [uint64](stdlib.calloc(1, vocab_size))
    end
    C.printf("Allocated inference buffers: %llu MB hidden, %llu MB logits\n", hidden_size / 1024 / 1024, vocab_size / 1024 / 1024)
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
    state.page_indices = [&int32](stdlib.calloc(num_pages + 512, sizeof(int32)))
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
    state.past_tokens = [&int64](stdlib.calloc(seq_len + params.max_tokens + 1, sizeof(int64)))
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
            break
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

terra fp8_dequantize_cpu(src: &int8, dst: &float, count: int64) : void
    for i = 0, count do
        var bits = [int32]([uint8](src[i]))
        var sign = bits >> 7
        var exp = (bits >> 3) and 0xF
        var mant = bits and 0x7
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

terra embedding_lookup_cpu(embed_weight: &int8, token_ids: &int64, output: &float, hidden_dim: int32, num_tokens: int32, dtype: int32) : void
    for t = 0, num_tokens do
        var token_id = token_ids[t]
        if dtype == 1 then
            var src = embed_weight + token_id * hidden_dim
            fp8_dequantize_cpu(src, output + t * hidden_dim, hidden_dim)
        elseif dtype == 4 then
            var src = [&float](embed_weight) + token_id * hidden_dim
            for i = 0, hidden_dim do
                output[t * hidden_dim + i] = src[i]
            end
        else
            for i = 0, hidden_dim do
                output[t * hidden_dim + i] = 0.0f
            end
        end
    end
end

terra matmul_cpu(A: &float, B: &float, C: &float, M: int32, N: int32, K: int32) : void
    for i = 0, M do
        for j = 0, N do
            var sum: float = 0.0f
            for k = 0, K do
                sum = sum + A[i * K + k] * B[k * N + j]
            end
            C[i * N + j] = sum
        end
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
    if handle.embed_weight == 0 then
        C.printf("Warning: No embedding weight loaded, skipping prefill\n")
        state.is_prefill_done = 1
        return 0
    end
    var hidden_dim = handle.config.hidden_dim
    var seq_len = state.seq_len
    var hidden = [&float](handle.hidden_buffer)
    if handle.use_gpu == 1 then
        var token_ids_d: CUdeviceptr = 0
        cw.cwMalloc(&token_ids_d, seq_len * sizeof(int64))
        cw.cwMemcpyH2D(token_ids_d, state.past_tokens, seq_len * sizeof(int64))
        kern.launch_embedding_lookup([&opaque](handle.embed_weight), [&int64](token_ids_d), [&float](handle.hidden_buffer), hidden_dim, seq_len, handle.embed_dtype, handle.cuda_contexts[0].stream)
        cw.cwStreamSynchronize(handle.cuda_contexts[0].stream)
        cw.cwFree(token_ids_d)
    else
        embedding_lookup_cpu([&int8](handle.embed_weight), state.past_tokens, hidden, hidden_dim, seq_len, handle.embed_dtype)
    end
    state.is_prefill_done = 1
    return 0
end

terra run_decode_step(handle: &EngineHandle, state: &RequestState, next_token: &int64) : int32
    var vocab_size = handle.config.vocab_size
    var hidden_dim = handle.config.hidden_dim
    var logits = [&float](stdlib.calloc(vocab_size, sizeof(float)))
    if logits == nil then return -1 end
    if handle.lm_head_weight ~= 0 then
        var hidden = [&float](stdlib.calloc(hidden_dim, sizeof(float)))
        if handle.use_gpu == 1 then
            var last_pos = state.past_len - 1
            if last_pos < 0 then last_pos = 0 end
            var token_ids_d: CUdeviceptr = 0
            var last_token = state.past_tokens[last_pos]
            cw.cwMalloc(&token_ids_d, sizeof(int64))
            cw.cwMemcpyH2D(token_ids_d, &last_token, sizeof(int64))
            kern.launch_embedding_lookup([&opaque](handle.embed_weight), [&int64](token_ids_d), [&float](handle.hidden_buffer), hidden_dim, 1, handle.embed_dtype, handle.cuda_contexts[0].stream)
            cw.cwStreamSynchronize(handle.cuda_contexts[0].stream)
            cw.cwMemcpyD2H(hidden, handle.hidden_buffer, hidden_dim * sizeof(float))
            cw.cwFree(token_ids_d)
        else
            var last_pos = state.past_len - 1
            if last_pos < 0 then last_pos = 0 end
            embedding_lookup_cpu([&int8](handle.embed_weight), &state.past_tokens[last_pos], hidden, hidden_dim, 1, handle.embed_dtype)
        end
        if handle.lm_head_dtype == 1 then
            var lm_head_f32 = [&float](stdlib.malloc(vocab_size * hidden_dim * sizeof(float)))
            fp8_dequantize_cpu([&int8](handle.lm_head_weight), lm_head_f32, vocab_size * hidden_dim)
            matmul_cpu(hidden, lm_head_f32, logits, 1, vocab_size, hidden_dim)
            stdlib.free(lm_head_f32)
        elseif handle.lm_head_dtype == 4 then
            matmul_cpu(hidden, [&float](handle.lm_head_weight), logits, 1, vocab_size, hidden_dim)
        else
            var max_logit: float = -1e9f
            for i = 0, vocab_size do
                logits[i] = hidden[i % hidden_dim]
                if logits[i] > max_logit then max_logit = logits[i] end
            end
        end
        stdlib.free(hidden)
    else
        var seed = [uint32](state.request_id + state.past_len)
        seed = seed * 1103515245 + 12345
        for i = 0, vocab_size do
            logits[i] = [float]((seed >> 16) and 0x7FFF) / 32768.0f - 0.5f
            seed = seed * 1103515245 + 12345
        end
    end
    var seed = [uint32](state.request_id * 31 + state.past_len * 17)
    seed = seed * 1103515245 + 12345
    var rand_val = [float](seed % 10000) / 10000.0f
    @next_token = sample_token_cpu(logits, vocab_size, state.temperature, state.top_p, state.rep_penalty, state.past_tokens, state.past_len, rand_val)
    stdlib.free(logits)
    var page_size = handle.kv_cache.page_size
    var new_seq_len = state.seq_len + 1
    var needed_pages = (new_seq_len + page_size - 1) / page_size
    if needed_pages > state.num_pages then
        var new_page: int32 = 0
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
    if state.past_len >= state.max_gen_tokens then
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
            var next_token: int64 = 0
            run_decode_step(handle, state, &next_token)
        end
    end
    pthread.pthread_mutex_unlock(&bs.lock)
    return 0
end

terra init_engine(model_dir: &int8, max_batch: int32, max_seq: int32, num_gpus: int32) : &EngineHandle
    C.printf("Initializing GLM-4.7-FP8 engine...\n")
    C.printf("Model dir: %s\n", model_dir)
    C.printf("Max batch: %d, Max seq: %d, Num GPUs: %d\n", max_batch, max_seq, num_gpus)
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
        C.printf("No config.json found, using default GLM-4.7 config\n")
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
    C.printf("Model config: hidden=%d, layers=%d, heads=%d, vocab=%d\n", handle.config.hidden_dim, handle.config.num_layers, handle.config.num_heads, handle.config.vocab_size)
    var index_path = [&int8](stdlib.malloc(dir_len + 64))
    string_h.sprintf(index_path, "%s/model.safetensors.index.json", model_dir)
    var has_weights = parse_safetensors_index(index_path, &handle.index) == 0
    stdlib.free(index_path)
    if has_weights then
        C.printf("Found %d weights in %d shards\n", handle.index.num_weights, handle.index.num_shards)
        if load_shard_mappings(model_dir, &handle.index, &handle.shard_mappings) < 0 then
            C.printf("Warning: Failed to load shard mappings\n")
            has_weights = false
        else
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
            C.printf("Parsed %d tensor descriptors\n", handle.num_tensors)
        end
    else
        C.printf("No weight index found, running without model weights\n")
    end
    if init_cuda_contexts(handle) < 0 then
        C.printf("CUDA initialization failed, running in CPU mode\n")
        handle.use_gpu = 0
    end
    if handle.use_gpu == 1 then
        init_nccl_contexts(handle)
    end
    if init_paged_kv_cache(handle) < 0 then
        C.printf("Failed to initialize KV cache\n")
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    if init_batch_state(handle) < 0 then
        C.printf("Failed to initialize batch state\n")
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    if has_weights then
        if load_weights_to_gpu(handle) < 0 then
            C.printf("Warning: Failed to load weights to GPU\n")
        end
    end
    if allocate_inference_buffers(handle) < 0 then
        C.printf("Failed to allocate inference buffers\n")
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    handle.initialized = 1
    C.printf("Engine initialized successfully (GPU mode: %d)\n", handle.use_gpu)
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
    C.printf("Freeing engine resources...\n")
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
        if handle.use_gpu == 1 then
            cw.cwFree(handle.kv_cache.k_cache)
            cw.cwFree(handle.kv_cache.v_cache)
        else
            stdlib.free([&opaque](handle.kv_cache.k_cache))
            stdlib.free([&opaque](handle.kv_cache.v_cache))
        end
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
    if handle.cuda_contexts ~= nil and handle.use_gpu == 1 then
        for i = 0, handle.num_gpus do
            cbw.cbwDestroy(handle.cuda_contexts[i].cublas_handle)
            cbw.cbwLtDestroy(handle.cuda_contexts[i].cublaslt_handle)
            cw.cwFree(handle.cuda_contexts[i].device_memory)
            cw.cwFree(handle.cuda_contexts[i].workspace)
            cw.cwStreamDestroy(handle.cuda_contexts[i].stream)
        end
        stdlib.free(handle.cuda_contexts)
    end
    if handle.nccl_contexts ~= nil then
        stdlib.free(handle.nccl_contexts)
    end
    if handle.use_gpu == 1 then
        if handle.embed_weight ~= 0 then cw.cwFree(handle.embed_weight) end
        if handle.lm_head_weight ~= 0 then cw.cwFree(handle.lm_head_weight) end
        if handle.final_norm ~= 0 then cw.cwFree(handle.final_norm) end
        if handle.hidden_buffer ~= 0 then cw.cwFree(handle.hidden_buffer) end
        if handle.residual_buffer ~= 0 then cw.cwFree(handle.residual_buffer) end
        if handle.qkv_buffer ~= 0 then cw.cwFree(handle.qkv_buffer) end
        if handle.attn_out_buffer ~= 0 then cw.cwFree(handle.attn_out_buffer) end
        if handle.mlp_buffer ~= 0 then cw.cwFree(handle.mlp_buffer) end
        if handle.logits_buffer ~= 0 then cw.cwFree(handle.logits_buffer) end
    else
        if handle.hidden_buffer ~= 0 then stdlib.free([&opaque](handle.hidden_buffer)) end
        if handle.residual_buffer ~= 0 then stdlib.free([&opaque](handle.residual_buffer)) end
        if handle.qkv_buffer ~= 0 then stdlib.free([&opaque](handle.qkv_buffer)) end
        if handle.attn_out_buffer ~= 0 then stdlib.free([&opaque](handle.attn_out_buffer)) end
        if handle.mlp_buffer ~= 0 then stdlib.free([&opaque](handle.mlp_buffer)) end
        if handle.logits_buffer ~= 0 then stdlib.free([&opaque](handle.logits_buffer)) end
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
    C.printf("Engine freed\n")
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
    if info_type == 13 then return handle.use_gpu end
    return -1
end

terra prefill_opaque(handle: &EngineHandle, request_id: int64, token_ids: &int64, seq_len: int32, temperature: float, top_p: float, rep_penalty: float, max_tokens: int32) : &opaque
    if handle == nil or handle.initialized ~= 1 then return nil end
    if seq_len <= 0 or seq_len > handle.max_seq then return nil end
    var params: SamplingParams
    params.temperature = temperature
    params.top_p = top_p
    params.top_k = 50
    params.rep_penalty = rep_penalty
    params.freq_penalty = 0.0f
    params.presence_penalty = 0.0f
    params.stop_tokens = nil
    params.num_stop_tokens = 0
    params.max_tokens = max_tokens
    var state = create_request_state(handle, request_id, token_ids, seq_len, &params)
    if state == nil then return nil end
    if add_request_to_batch(handle, state) < 0 then
        free_request_state_internal(handle, state)
        return nil
    end
    if run_prefill(handle, state) < 0 then
        remove_request_from_batch(handle, request_id)
        free_request_state_internal(handle, state)
        return nil
    end
    return [&opaque](state)
end

terra decode_step_opaque(handle: &EngineHandle, state_ptr: &opaque, next_token_out: &int64, is_done_out: &int32) : int32
    if handle == nil or handle.initialized ~= 1 then return -1 end
    if state_ptr == nil then return -2 end
    var state = [&RequestState](state_ptr)
    var status = run_decode_step(handle, state, next_token_out)
    if status < 0 then return status end
    @is_done_out = state.is_finished
    return 0
end

terra free_request_opaque(handle: &EngineHandle, state_ptr: &opaque) : void
    if handle == nil or state_ptr == nil then return end
    var state = [&RequestState](state_ptr)
    remove_request_from_batch(handle, state.request_id)
    free_request_state_internal(handle, state)
end

terra run_batch_decode_ext(handle: &EngineHandle, state_ptrs: &&opaque, num_states: int32, out_tokens: &int64, out_done: &int32) : int32
    if handle == nil or handle.initialized ~= 1 then return -1 end
    if num_states <= 0 then return 0 end
    for i = 0, num_states do
        if state_ptrs[i] == nil then
            out_tokens[i] = 0
            out_done[i] = 1
        else
            var state = [&RequestState](state_ptrs[i])
            if state.is_finished == 1 then
                out_tokens[i] = 0
                out_done[i] = 1
            else
                var next_token: int64 = 0
                var status = run_decode_step(handle, state, &next_token)
                if status < 0 then
                    out_tokens[i] = 0
                    out_done[i] = 1
                else
                    out_tokens[i] = next_token
                    out_done[i] = state.is_finished
                end
            end
        end
    end
    return 0
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
    remove_request_from_batch = remove_request_from_batch,
    prefill_opaque = prefill_opaque,
    decode_step_opaque = decode_step_opaque,
    free_request_opaque = free_request_opaque,
    run_batch_decode_ext = run_batch_decode_ext
}, nil, nil, true)
