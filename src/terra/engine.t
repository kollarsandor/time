local C = terralib.includec("stdio.h")
local stdlib = terralib.includec("stdlib.h")
local string_h = terralib.includec("string.h")
local fcntl = terralib.includec("fcntl.h")
local unistd = terralib.includec("unistd.h")
local mman = terralib.includec("sys/mman.h")
local stat = terralib.includec("sys/stat.h")
local errno_h = terralib.includec("errno.h")

struct SafetensorsHeader {
    data_offsets: &uint64
    shapes: &&int64
    shape_dims: &int32
    dtypes: &&int8
    num_tensors: int32
    names: &&int8
}

struct TensorInfo {
    name: &int8
    data_ptr: &int8
    dtype: int32
    shape: &int64
    ndim: int32
    size_bytes: uint64
}

struct PagedKVCache {
    k_pages: &&float
    v_pages: &&float
    page_size: int32
    num_pages: int32
    max_pages: int32
    page_table: &int64
    num_layers: int32
    num_heads: int32
    head_dim: int32
}

struct RequestState {
    request_id: int64
    seq_len: int32
    page_indices: &int32
    num_pages: int32
    past_tokens: &int64
    past_len: int32
}

struct EngineHandle {
    model_dir: &int8
    max_batch: int32
    max_seq: int32
    num_gpus: int32
    hidden_dim: int32
    num_layers: int32
    num_heads: int32
    head_dim: int32
    vocab_size: int32
    num_experts: int32
    top_k_experts: int32
    intermediate_size: int32
    kv_cache: &PagedKVCache
    weight_ptrs: &&int8
    num_weights: int32
    mmap_ptrs: &&int8
    mmap_sizes: &uint64
    num_shards: int32
    initialized: int32
}

terra parse_json_i64(json: &int8, key: &int8) : int64
    var key_len = string_h.strlen(key)
    var pos = string_h.strstr(json, key)
    if pos == nil then return -1 end
    pos = pos + key_len
    while @pos ~= 0 and (@pos < [int8]('0') or @pos > [int8]('9')) and @pos ~= [int8]('-') do
        pos = pos + 1
    end
    return stdlib.strtoll(pos, nil, 10)
end

terra parse_json_str(json: &int8, key: &int8, buf: &int8, buf_size: int32) : int32
    var key_len = string_h.strlen(key)
    var pos = string_h.strstr(json, key)
    if pos == nil then return -1 end
    pos = pos + key_len
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

terra read_file(path: &int8, size_out: &uint64) : &int8
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
    var rd = unistd.read(fd, buf, size)
    if rd ~= size then
        stdlib.free(buf)
        unistd.close(fd)
        return nil
    end
    buf[size] = 0
    @size_out = size
    unistd.close(fd)
    return buf
end

terra mmap_file(path: &int8, size_out: &uint64) : &int8
    var fd = fcntl.open(path, 0)
    if fd < 0 then return nil end
    var st: stat.struct_stat
    if stat.fstat(fd, &st) < 0 then
        unistd.close(fd)
        return nil
    end
    var size = st.st_size
    var ptr = mman.mmap(nil, size, 1, 2, fd, 0)
    unistd.close(fd)
    if ptr == [&opaque](-1) then return nil end
    @size_out = size
    return [&int8](ptr)
end

terra parse_safetensors_header(data: &int8, header_out: &SafetensorsHeader) : int32
    var header_size = @[&uint64](data)
    var header_json = data + 8
    header_out.num_tensors = 0
    var pos = header_json
    var in_tensor = 0
    var brace_count = 0
    var tensor_count = 0
    while @pos ~= 0 and (pos - header_json) < header_size do
        if @pos == [int8]('{') then
            brace_count = brace_count + 1
            if brace_count == 2 then tensor_count = tensor_count + 1 end
        end
        if @pos == [int8]('}') then brace_count = brace_count - 1 end
        pos = pos + 1
    end
    header_out.num_tensors = tensor_count
    header_out.data_offsets = [&uint64](stdlib.calloc(tensor_count * 2, 8))
    header_out.shapes = [&&int64](stdlib.calloc(tensor_count, sizeof([&int64])))
    header_out.shape_dims = [&int32](stdlib.calloc(tensor_count, 4))
    header_out.dtypes = [&&int8](stdlib.calloc(tensor_count, sizeof([&int8])))
    header_out.names = [&&int8](stdlib.calloc(tensor_count, sizeof([&int8])))
    return 0
end

terra get_tensor_ptr(handle: &EngineHandle, name: &int8, info: &TensorInfo) : int32
    for i = 0, handle.num_weights do
        if string_h.strcmp(handle.weight_ptrs[i], name) == 0 then
            return 0
        end
    end
    return -1
end

terra init_paged_kv_cache(cache: &PagedKVCache, max_pages: int32, page_size: int32, num_layers: int32, num_heads: int32, head_dim: int32) : int32
    cache.max_pages = max_pages
    cache.page_size = page_size
    cache.num_layers = num_layers
    cache.num_heads = num_heads
    cache.head_dim = head_dim
    cache.num_pages = 0
    var total_size = max_pages * page_size * head_dim * sizeof(float)
    cache.k_pages = [&&float](stdlib.calloc(num_layers * num_heads, sizeof([&float])))
    cache.v_pages = [&&float](stdlib.calloc(num_layers * num_heads, sizeof([&float])))
    if cache.k_pages == nil or cache.v_pages == nil then return -1 end
    for i = 0, num_layers * num_heads do
        cache.k_pages[i] = [&float](stdlib.calloc(max_pages * page_size * head_dim, sizeof(float)))
        cache.v_pages[i] = [&float](stdlib.calloc(max_pages * page_size * head_dim, sizeof(float)))
        if cache.k_pages[i] == nil or cache.v_pages[i] == nil then return -1 end
    end
    cache.page_table = [&int64](stdlib.calloc(max_pages, 8))
    if cache.page_table == nil then return -1 end
    return 0
end

terra allocate_page(cache: &PagedKVCache) : int32
    if cache.num_pages >= cache.max_pages then return -1 end
    var page_id = cache.num_pages
    cache.num_pages = cache.num_pages + 1
    return page_id
end

terra free_page(cache: &PagedKVCache, page_id: int32) : void
end

terra init_engine(model_dir: &int8, max_batch: int32, max_seq: int32, num_gpus: int32) : &EngineHandle
    var handle = [&EngineHandle](stdlib.calloc(1, sizeof(EngineHandle)))
    if handle == nil then return nil end
    var dir_len = string_h.strlen(model_dir)
    handle.model_dir = [&int8](stdlib.malloc(dir_len + 1))
    string_h.strcpy(handle.model_dir, model_dir)
    handle.max_batch = max_batch
    handle.max_seq = max_seq
    handle.num_gpus = num_gpus
    handle.hidden_dim = 4096
    handle.num_layers = 92
    handle.num_heads = 32
    handle.head_dim = 128
    handle.vocab_size = 151552
    handle.num_experts = 160
    handle.top_k_experts = 8
    handle.intermediate_size = 13696
    var index_path = [&int8](stdlib.malloc(dir_len + 64))
    string_h.sprintf(index_path, "%s/model.safetensors.index.json", model_dir)
    var index_size: uint64 = 0
    var index_json = read_file(index_path, &index_size)
    stdlib.free(index_path)
    if index_json == nil then
        C.printf("Failed to read model index\n")
        stdlib.free(handle.model_dir)
        stdlib.free(handle)
        return nil
    end
    handle.num_shards = 8
    handle.mmap_ptrs = [&&int8](stdlib.calloc(handle.num_shards, sizeof([&int8])))
    handle.mmap_sizes = [&uint64](stdlib.calloc(handle.num_shards, 8))
    for i = 0, handle.num_shards do
        var shard_path = [&int8](stdlib.malloc(dir_len + 64))
        string_h.sprintf(shard_path, "%s/model-%05d-of-%05d.safetensors", model_dir, i + 1, handle.num_shards)
        var shard_size: uint64 = 0
        handle.mmap_ptrs[i] = mmap_file(shard_path, &shard_size)
        handle.mmap_sizes[i] = shard_size
        stdlib.free(shard_path)
        if handle.mmap_ptrs[i] == nil then
            C.printf("Failed to mmap shard %d\n", i)
        end
    end
    stdlib.free(index_json)
    handle.kv_cache = [&PagedKVCache](stdlib.calloc(1, sizeof(PagedKVCache)))
    var page_size = 16
    var max_pages = (max_batch * max_seq) / page_size + max_batch
    if init_paged_kv_cache(handle.kv_cache, max_pages, page_size, handle.num_layers, handle.num_heads, handle.head_dim) < 0 then
        C.printf("Failed to init KV cache\n")
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
    state_out.request_id = request_id
    state_out.seq_len = seq_len
    var page_size = handle.kv_cache.page_size
    var num_pages_needed = (seq_len + page_size - 1) / page_size
    state_out.page_indices = [&int32](stdlib.calloc(num_pages_needed, 4))
    state_out.num_pages = num_pages_needed
    for i = 0, num_pages_needed do
        var page_id = allocate_page(handle.kv_cache)
        if page_id < 0 then return -3 end
        state_out.page_indices[i] = page_id
    end
    state_out.past_tokens = [&int64](stdlib.calloc(seq_len, 8))
    state_out.past_len = seq_len
    for i = 0, seq_len do
        state_out.past_tokens[i] = token_ids[i]
    end
    return 0
end

terra decode_step(handle: &EngineHandle, state: &RequestState, next_token_out: &int64) : int32
    if handle == nil or handle.initialized ~= 1 then return -1 end
    if state == nil then return -2 end
    var new_seq_len = state.seq_len + 1
    var page_size = handle.kv_cache.page_size
    var current_pages = state.num_pages
    var needed_pages = (new_seq_len + page_size - 1) / page_size
    if needed_pages > current_pages then
        var new_page_indices = [&int32](stdlib.realloc(state.page_indices, needed_pages * 4))
        if new_page_indices == nil then return -3 end
        state.page_indices = new_page_indices
        for i = current_pages, needed_pages do
            var page_id = allocate_page(handle.kv_cache)
            if page_id < 0 then return -4 end
            state.page_indices[i] = page_id
        end
        state.num_pages = needed_pages
    end
    @next_token_out = 1
    var new_past = [&int64](stdlib.realloc(state.past_tokens, (state.past_len + 1) * 8))
    if new_past == nil then return -5 end
    state.past_tokens = new_past
    state.past_tokens[state.past_len] = @next_token_out
    state.past_len = state.past_len + 1
    state.seq_len = new_seq_len
    return 0
end

terra free_request_state(state: &RequestState) : void
    if state ~= nil then
        if state.page_indices ~= nil then stdlib.free(state.page_indices) end
        if state.past_tokens ~= nil then stdlib.free(state.past_tokens) end
    end
end

terra free_engine(handle: &EngineHandle) : void
    if handle == nil then return end
    if handle.model_dir ~= nil then stdlib.free(handle.model_dir) end
    if handle.kv_cache ~= nil then
        for i = 0, handle.kv_cache.num_layers * handle.kv_cache.num_heads do
            if handle.kv_cache.k_pages ~= nil and handle.kv_cache.k_pages[i] ~= nil then
                stdlib.free(handle.kv_cache.k_pages[i])
            end
            if handle.kv_cache.v_pages ~= nil and handle.kv_cache.v_pages[i] ~= nil then
                stdlib.free(handle.kv_cache.v_pages[i])
            end
        end
        if handle.kv_cache.k_pages ~= nil then stdlib.free(handle.kv_cache.k_pages) end
        if handle.kv_cache.v_pages ~= nil then stdlib.free(handle.kv_cache.v_pages) end
        if handle.kv_cache.page_table ~= nil then stdlib.free(handle.kv_cache.page_table) end
        stdlib.free(handle.kv_cache)
    end
    if handle.mmap_ptrs ~= nil then
        for i = 0, handle.num_shards do
            if handle.mmap_ptrs[i] ~= nil then
                mman.munmap(handle.mmap_ptrs[i], handle.mmap_sizes[i])
            end
        end
        stdlib.free(handle.mmap_ptrs)
    end
    if handle.mmap_sizes ~= nil then stdlib.free(handle.mmap_sizes) end
    if handle.weight_ptrs ~= nil then stdlib.free(handle.weight_ptrs) end
    stdlib.free(handle)
end

terralib.saveobj("engine.so", {
    init_engine = init_engine,
    prefill = prefill,
    decode_step = decode_step,
    free_request_state = free_request_state,
    free_engine = free_engine
}, nil, nil, true)
