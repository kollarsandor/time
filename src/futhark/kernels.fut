type fp8_e4m3 = i8

def fp8_to_f32 (x: fp8_e4m3) : f32 =
  let bits = i32.i8 x
  let sign = bits >> 7
  let exp = (bits >> 3) & 0xF
  let mant = bits & 0x7
  in if exp == 0 then
       if mant == 0 then
         if sign == 1 then -0.0f32 else 0.0f32
       else
         let denorm = f32.i32 mant * (2.0f32 ** (-9.0f32))
         in if sign == 1 then -denorm else denorm
     else if exp == 15 && mant == 7 then
       f32.nan
     else
       let e = exp - 7
       let m = 1.0f32 + (f32.i32 mant / 8.0f32)
       let val = m * (2.0f32 ** f32.i32 e)
       in if sign == 1 then -val else val

def f32_to_fp8 (x: f32) : fp8_e4m3 =
  let sign = if x < 0.0f32 then 1i32 else 0i32
  let ax = f32.abs x
  in if f32.isnan x then
       i8.i32 (0x7F)
     else if ax == 0.0f32 then
       i8.i32 (sign << 7)
     else if ax >= 448.0f32 then
       i8.i32 ((sign << 7) | 0x7E)
     else if ax < (2.0f32 ** (-9.0f32)) then
       i8.i32 (sign << 7)
     else
       let log2_ax = f32.log2 ax
       let e = i32.f32 (f32.floor log2_ax)
       let e_clamped = i32.max (-6) (i32.min 8 e)
       let exp_bits = e_clamped + 7
       let m = ax / (2.0f32 ** f32.i32 e_clamped) - 1.0f32
       let mant = i32.f32 (f32.round (m * 8.0f32))
       let mant_clamped = i32.max 0 (i32.min 7 mant)
       in i8.i32 ((sign << 7) | (exp_bits << 3) | mant_clamped)

def fp8_to_f16 (x: fp8_e4m3) : f16 =
  f16.f32 (fp8_to_f32 x)

def f16_to_fp8 (x: f16) : fp8_e4m3 =
  f32_to_fp8 (f32.f16 x)

def fp8_arr_to_f32 [n] (arr: [n]fp8_e4m3) : [n]f32 =
  map fp8_to_f32 arr

def f32_arr_to_fp8 [n] (arr: [n]f32) : [n]fp8_e4m3 =
  map f32_to_fp8 arr

def fp8_arr_to_f16 [n] (arr: [n]fp8_e4m3) : [n]f16 =
  map fp8_to_f16 arr

def f16_arr_to_fp8 [n] (arr: [n]f16) : [n]fp8_e4m3 =
  map f16_to_fp8 arr

def rms_norm [n] (x: [n]f32) (gamma: [n]f32) (eps: f32) : [n]f32 =
  let sq_sum = reduce (+) 0.0f32 (map (\v -> v * v) x)
  let rms = f32.sqrt (sq_sum / f32.i64 n + eps)
  in map2 (\xi gi -> (xi / rms) * gi) x gamma

def rms_norm_fp8 [n] (x: [n]fp8_e4m3) (gamma: [n]f32) (eps: f32) : [n]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let result = rms_norm xf gamma eps
  in f32_arr_to_fp8 result

def rms_norm_f16 [n] (x: [n]f16) (gamma: [n]f16) (eps: f32) : [n]f16 =
  let xf = map f32.f16 x
  let gf = map f32.f16 gamma
  let sq_sum = reduce (+) 0.0f32 (map (\v -> v * v) xf)
  let rms = f32.sqrt (sq_sum / f32.i64 n + eps)
  let result = map2 (\xi gi -> (xi / rms) * gi) xf gf
  in map f16.f32 result

def swiglu [n] (x: [n]f32) (gate: [n]f32) : [n]f32 =
  map2 (\xi gi -> xi * (gi / (1.0f32 + f32.exp (-gi)))) x gate

def swiglu_f16 [n] (x: [n]f16) (gate: [n]f16) : [n]f16 =
  map2 (\xi gi -> 
    let xf = f32.f16 xi
    let gf = f32.f16 gi
    let result = xf * (gf / (1.0f32 + f32.exp (-gf)))
    in f16.f32 result
  ) x gate

def geglu [n] (x: [n]f32) (gate: [n]f32) : [n]f32 =
  map2 (\xi gi ->
    let gelu = 0.5f32 * gi * (1.0f32 + f32.tanh (0.7978845608f32 * (gi + 0.044715f32 * gi * gi * gi)))
    in xi * gelu
  ) x gate

def geglu_f16 [n] (x: [n]f16) (gate: [n]f16) : [n]f16 =
  map2 (\xi gi ->
    let xf = f32.f16 xi
    let gf = f32.f16 gi
    let gelu = 0.5f32 * gf * (1.0f32 + f32.tanh (0.7978845608f32 * (gf + 0.044715f32 * gf * gf * gf)))
    in f16.f32 (xf * gelu)
  ) x gate

def matmul_f32 [m][n][k] (a: [m][k]f32) (b: [k][n]f32) : [m][n]f32 =
  let bt = transpose b
  in map (\a_row -> map (\b_col -> reduce (+) 0.0f32 (map2 (*) a_row b_col)) bt) a

def matmul_f16 [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : [m][n]f16 =
  let bt = transpose b
  in map (\a_row -> 
    map (\b_col -> 
      let products = map2 (\ai bi -> f32.f16 ai * f32.f16 bi) a_row b_col
      in f16.f32 (reduce (+) 0.0f32 products)
    ) bt
  ) a

def matmul_fp8 [m][n][k] (a: [m][k]fp8_e4m3) (b: [k][n]fp8_e4m3) : [m][n]fp8_e4m3 =
  let af = map fp8_arr_to_f32 a
  let bf = map fp8_arr_to_f32 b
  let cf = matmul_f32 af bf
  in map f32_arr_to_fp8 cf

def matmul_fp8_to_f16 [m][n][k] (a: [m][k]fp8_e4m3) (b: [k][n]fp8_e4m3) : [m][n]f16 =
  let af = map fp8_arr_to_f32 a
  let bf = map fp8_arr_to_f32 b
  let cf = matmul_f32 af bf
  in map (\row -> map f16.f32 row) cf

def matvec_f32 [m][n] (mat: [m][n]f32) (vec: [n]f32) : [m]f32 =
  map (\row -> reduce (+) 0.0f32 (map2 (*) row vec)) mat

def matvec_fp8 [m][n] (mat: [m][n]fp8_e4m3) (vec: [n]fp8_e4m3) : [m]fp8_e4m3 =
  let matf = map fp8_arr_to_f32 mat
  let vecf = fp8_arr_to_f32 vec
  let result = matvec_f32 matf vecf
  in f32_arr_to_fp8 result

def rope_freqs (dim: i64) (base: f32) : [dim]f32 =
  tabulate dim (\i -> 1.0f32 / (base ** (f32.i64 (2 * (i / 2)) / f32.i64 dim)))

def apply_rope [n] (x: [n]f32) (freqs: [n]f32) (pos: i64) : [n]f32 =
  let half = n / 2
  in tabulate n (\i ->
    let freq = freqs[i] * f32.i64 pos
    let cos_f = f32.cos freq
    let sin_f = f32.sin freq
    in if i < half then
         x[i] * cos_f - x[i + half] * sin_f
       else
         x[i - half] * sin_f + x[i] * cos_f
  )

def rope_fp8 [n] (x: [n]fp8_e4m3) (freqs: [n]f32) (pos: i64) : [n]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let result = apply_rope xf freqs pos
  in f32_arr_to_fp8 result

def rope_f16 [n] (x: [n]f16) (freqs: [n]f32) (pos: i64) : [n]f16 =
  let xf = map f32.f16 x
  let result = apply_rope xf freqs pos
  in map f16.f32 result

def apply_rope_2d [seq_len][head_dim] (x: [seq_len][head_dim]f32) (base: f32) (start_pos: i64) : [seq_len][head_dim]f32 =
  let freqs = rope_freqs head_dim base
  in tabulate seq_len (\s ->
    apply_rope x[s] freqs (start_pos + s)
  )

def softmax [n] (x: [n]f32) : [n]f32 =
  let max_x = reduce f32.max f32.lowest x
  let exp_x = map (\xi -> f32.exp (xi - max_x)) x
  let sum_exp = reduce (+) 0.0f32 exp_x
  in map (\e -> e / sum_exp) exp_x

def softmax_f16 [n] (x: [n]f16) : [n]f16 =
  let xf = map f32.f16 x
  let result = softmax xf
  in map f16.f32 result

def online_softmax_step (m: f32) (d: f32) (x: f32) : (f32, f32) =
  let new_m = f32.max m x
  let old_factor = f32.exp (m - new_m)
  let new_factor = f32.exp (x - new_m)
  let new_d = d * old_factor + new_factor
  in (new_m, new_d)

def flash_attention_tile [tile_size][head_dim]
    (q_tile: [tile_size][head_dim]f32)
    (k_tile: [tile_size][head_dim]f32)
    (v_tile: [tile_size][head_dim]f32)
    (scale: f32)
    (m_prev: [tile_size]f32)
    (l_prev: [tile_size]f32)
    (o_prev: [tile_size][head_dim]f32) : ([tile_size]f32, [tile_size]f32, [tile_size][head_dim]f32) =
  let scores = map (\qi ->
    map (\ki -> reduce (+) 0.0f32 (map2 (*) qi ki) * scale) k_tile
  ) q_tile
  let (m_new, l_new, o_new) = 
    loop (m, l, o) = (m_prev, l_prev, o_prev) for j < tile_size do
      let score_col = map (\i -> scores[i, j]) (iota tile_size)
      let m_temp = map2 f32.max m score_col
      let exp_old = map2 (\mi mti -> f32.exp (mi - mti)) m m_temp
      let exp_new = map2 (\sci mti -> f32.exp (sci - mti)) score_col m_temp
      let l_temp = map2 (\li eoi -> li * eoi) l exp_old
      let l_next = map2 (+) l_temp exp_new
      let o_scaled = map2 (\oi eoi -> map (\oij -> oij * eoi) oi) o exp_old
      let v_row = v_tile[j]
      let o_contrib = map2 (\eni _ -> map (\vj -> eni * vj) v_row) exp_new (iota tile_size)
      let o_next = map2 (\osi oci -> map2 (+) osi oci) o_scaled o_contrib
      in (m_temp, l_next, o_next)
  in (m_new, l_new, o_new)

def paged_attention_prefill [seq_len][head_dim][page_size]
    (q: [seq_len][head_dim]f32)
    (k: [seq_len][head_dim]f32)
    (v: [seq_len][head_dim]f32)
    (page_table: []i64)
    (kv_cache_k: [][page_size][head_dim]f32)
    (kv_cache_v: [][page_size][head_dim]f32)
    (scale: f32) : ([seq_len][head_dim]f32, [][page_size][head_dim]f32, [][page_size][head_dim]f32) =
  let scores = map (\qi ->
    map (\ki -> reduce (+) 0.0f32 (map2 (*) qi ki) * scale) k
  ) q
  let attn = map softmax scores
  let out = map (\a_row ->
    map (\d ->
      reduce (+) 0.0f32 (map2 (\a vi -> a * vi[d]) a_row v)
    ) (iota head_dim)
  ) attn
  let num_pages = length page_table
  let new_k = tabulate num_pages (\p ->
    tabulate page_size (\s ->
      let global_idx = p * page_size + s
      in if global_idx < seq_len then k[global_idx] else replicate head_dim 0.0f32
    )
  )
  let new_v = tabulate num_pages (\p ->
    tabulate page_size (\s ->
      let global_idx = p * page_size + s
      in if global_idx < seq_len then v[global_idx] else replicate head_dim 0.0f32
    )
  )
  in (out, new_k, new_v)

def paged_attention_decode [head_dim][page_size]
    (q: [head_dim]f32)
    (page_table: []i64)
    (kv_cache_k: [][page_size][head_dim]f32)
    (kv_cache_v: [][page_size][head_dim]f32)
    (seq_len: i64)
    (scale: f32) : [head_dim]f32 =
  let num_pages = length page_table
  let max_tokens = num_pages * page_size
  let all_scores = tabulate max_tokens (\idx ->
    let p = idx / page_size
    let s = idx % page_size
    let page_idx = if p < num_pages then page_table[p] else 0
    in if idx < seq_len && p < num_pages then
         reduce (+) 0.0f32 (map2 (*) q kv_cache_k[page_idx, s]) * scale
       else
         f32.lowest
  )
  let scores_valid = take seq_len all_scores
  let attn = softmax scores_valid
  let attn_padded = attn ++ replicate (max_tokens - seq_len) 0.0f32
  in tabulate head_dim (\d ->
    reduce (+) 0.0f32 (
      tabulate max_tokens (\idx ->
        let p = idx / page_size
        let s = idx % page_size
        let page_idx = if p < num_pages then page_table[p] else 0
        in if idx < seq_len && p < num_pages then
             attn_padded[idx] * kv_cache_v[page_idx, s, d]
           else
             0.0f32
      )
    )
  )

def paged_attention_decode_tiled [head_dim][page_size]
    (q: [head_dim]f32)
    (page_table: []i64)
    (kv_cache_k: [][page_size][head_dim]f32)
    (kv_cache_v: [][page_size][head_dim]f32)
    (seq_len: i64)
    (scale: f32)
    (tile_size: i64) : [head_dim]f32 =
  let num_pages = length page_table
  let num_tiles = (seq_len + tile_size - 1) / tile_size
  let (final_m, final_l, final_o) = 
    loop (m_acc, l_acc, o_acc) = (f32.lowest, 0.0f32, replicate head_dim 0.0f32) for tile_idx < num_tiles do
      let tile_start = tile_idx * tile_size
      let tile_end = i64.min (tile_start + tile_size) seq_len
      let actual_tile_size = tile_end - tile_start
      let tile_scores = tabulate actual_tile_size (\ti ->
        let global_idx = tile_start + ti
        let p = global_idx / page_size
        let s = global_idx % page_size
        let page_idx = if p < num_pages then page_table[p] else 0
        in reduce (+) 0.0f32 (map2 (*) q kv_cache_k[page_idx, s]) * scale
      )
      let tile_max = reduce f32.max f32.lowest tile_scores
      let new_m = f32.max m_acc tile_max
      let old_factor = f32.exp (m_acc - new_m)
      let tile_exp = map (\s -> f32.exp (s - new_m)) tile_scores
      let tile_sum = reduce (+) 0.0f32 tile_exp
      let new_l = l_acc * old_factor + tile_sum
      let o_scaled = map (\oi -> oi * old_factor) o_acc
      let tile_contrib = tabulate head_dim (\d ->
        reduce (+) 0.0f32 (
          tabulate actual_tile_size (\ti ->
            let global_idx = tile_start + ti
            let p = global_idx / page_size
            let s = global_idx % page_size
            let page_idx = if p < num_pages then page_table[p] else 0
            in tile_exp[ti] * kv_cache_v[page_idx, s, d]
          )
        )
      )
      let o_new = map2 (+) o_scaled tile_contrib
      in (new_m, new_l, o_new)
  in map (\oi -> oi / final_l) final_o

def top_k_indices [n] (arr: [n]f32) (k: i64) : [k]i64 =
  let indexed = zip arr (iota n)
  let sorted = merge_sort (\(a, _) (b, _) -> a >= b) indexed
  in map (\i -> sorted[i].1) (iota k)

def moe_router [num_experts][hidden] (x: [hidden]f32) (router_w: [num_experts][hidden]f32) (top_k: i64) : ([top_k]i64, [top_k]f32) =
  let logits = map (\w -> reduce (+) 0.0f32 (map2 (*) x w)) router_w
  let probs = softmax logits
  let topk_idx = top_k_indices probs top_k
  let topk_probs = map (\i -> probs[i]) topk_idx
  let sum_probs = reduce (+) 0.0f32 topk_probs
  let norm_probs = map (\p -> p / sum_probs) topk_probs
  in (topk_idx, norm_probs)

def moe_router_fp8 [num_experts][hidden] (x: [hidden]fp8_e4m3) (router_w: [num_experts][hidden]fp8_e4m3) (top_k: i64) : ([top_k]i64, [top_k]f32) =
  let xf = fp8_arr_to_f32 x
  let wf = map fp8_arr_to_f32 router_w
  in moe_router xf wf top_k

def moe_expert_mlp [hidden][intermediate] (x: [hidden]f32) (w1: [intermediate][hidden]f32) (w2: [hidden][intermediate]f32) (w_gate: [intermediate][hidden]f32) : [hidden]f32 =
  let h1 = map (\row -> reduce (+) 0.0f32 (map2 (*) x row)) w1
  let gate = map (\row -> reduce (+) 0.0f32 (map2 (*) x row)) w_gate
  let h_act = swiglu h1 gate
  in map (\row -> reduce (+) 0.0f32 (map2 (*) h_act row)) w2

def moe_expert_mlp_fp8 [hidden][intermediate] (x: [hidden]fp8_e4m3) (w1: [intermediate][hidden]fp8_e4m3) (w2: [hidden][intermediate]fp8_e4m3) (w_gate: [intermediate][hidden]fp8_e4m3) : [hidden]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let w1f = map fp8_arr_to_f32 w1
  let w2f = map fp8_arr_to_f32 w2
  let wgf = map fp8_arr_to_f32 w_gate
  let result = moe_expert_mlp xf w1f w2f wgf
  in f32_arr_to_fp8 result

def moe_forward [num_experts][hidden][intermediate]
    (x: [hidden]f32)
    (router_w: [num_experts][hidden]f32)
    (expert_w1: [num_experts][intermediate][hidden]f32)
    (expert_w2: [num_experts][hidden][intermediate]f32)
    (expert_wg: [num_experts][intermediate][hidden]f32)
    (k: i64) : [hidden]f32 =
  let (indices, weights) = moe_router x router_w k
  let expert_outputs = map (\i -> moe_expert_mlp x expert_w1[i] expert_w2[i] expert_wg[i]) indices
  in map (\h ->
    reduce (+) 0.0f32 (map2 (\out w -> out[h] * w) expert_outputs weights)
  ) (iota hidden)

def moe_forward_fp8 [num_experts][hidden][intermediate]
    (x: [hidden]fp8_e4m3)
    (router_w: [num_experts][hidden]fp8_e4m3)
    (expert_w1: [num_experts][intermediate][hidden]fp8_e4m3)
    (expert_w2: [num_experts][hidden][intermediate]fp8_e4m3)
    (expert_wg: [num_experts][intermediate][hidden]fp8_e4m3)
    (k: i64) : [hidden]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let rwf = map fp8_arr_to_f32 router_w
  let w1f = map (map fp8_arr_to_f32) expert_w1
  let w2f = map (map fp8_arr_to_f32) expert_w2
  let wgf = map (map fp8_arr_to_f32) expert_wg
  let result = moe_forward xf rwf w1f w2f wgf k
  in f32_arr_to_fp8 result

def moe_dispatch [batch][hidden][num_experts]
    (tokens: [batch][hidden]f32)
    (assignments: [batch][]i64)
    (weights: [batch][]f32) : [num_experts][batch][hidden]f32 =
  tabulate num_experts (\e ->
    map2 (\token assign ->
      let weight = reduce (+) 0.0f32 (
        map2 (\a w -> if a == e then w else 0.0f32) assign weights[0]
      )
      in map (\t -> t * weight) token
    ) tokens assignments
  )

def moe_combine [batch][hidden][num_experts]
    (expert_outputs: [num_experts][batch][hidden]f32)
    (assignments: [batch][]i64)
    (weights: [batch][]f32) : [batch][hidden]f32 =
  map2 (\assigns ws ->
    map (\h ->
      reduce (+) 0.0f32 (
        map2 (\e w -> expert_outputs[assigns[0], 0, h] * w) assigns ws
      )
    ) (iota hidden)
  ) assignments weights

def embedding_lookup [vocab_size][hidden] (token_id: i64) (embed_table: [vocab_size][hidden]f32) : [hidden]f32 =
  embed_table[token_id]

def embedding_lookup_fp8 [vocab_size][hidden] (token_id: i64) (embed_table: [vocab_size][hidden]fp8_e4m3) : [hidden]fp8_e4m3 =
  embed_table[token_id]

def embedding_batch [batch][vocab_size][hidden] (token_ids: [batch]i64) (embed_table: [vocab_size][hidden]f32) : [batch][hidden]f32 =
  map (\tid -> embed_table[tid]) token_ids

def residual_add [n] (x: [n]f32) (residual: [n]f32) : [n]f32 =
  map2 (+) x residual

def residual_add_fp8 [n] (x: [n]fp8_e4m3) (residual: [n]fp8_e4m3) : [n]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let rf = fp8_arr_to_f32 residual
  let result = map2 (+) xf rf
  in f32_arr_to_fp8 result

def bias_add [n] (x: [n]f32) (bias: [n]f32) : [n]f32 =
  map2 (+) x bias

def bias_add_fp8 [n] (x: [n]fp8_e4m3) (bias: [n]f32) : [n]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let result = map2 (+) xf bias
  in f32_arr_to_fp8 result

def temperature_scale [n] (logits: [n]f32) (temp: f32) : [n]f32 =
  map (\l -> l / temp) logits

def top_p_filter [n] (logits: [n]f32) (p: f32) : [n]f32 =
  let probs = softmax logits
  let indexed = zip probs (iota n)
  let sorted = merge_sort (\(a, _) (b, _) -> a >= b) indexed
  let cumsum = scan (+) 0.0f32 (map (.0) sorted)
  let mask = map2 (\cs (_, idx) -> (cs, idx)) cumsum sorted
  let valid_indices = map (\(cs, idx) -> if cs <= p || cs == (map (.0) sorted)[0] then idx else -1) mask
  in tabulate n (\i ->
    if any (\vi -> vi == i) valid_indices && valid_indices[0] != -1 then logits[i] else f32.lowest
  )

def repetition_penalty_apply [n] (logits: [n]f32) (past_tokens: []i64) (penalty: f32) : [n]f32 =
  tabulate n (\i ->
    if any (\t -> t == i) past_tokens then
      if logits[i] > 0.0f32 then logits[i] / penalty else logits[i] * penalty
    else
      logits[i]
  )

def frequency_penalty_apply [n] (logits: [n]f32) (past_tokens: []i64) (penalty: f32) : [n]f32 =
  tabulate n (\i ->
    let count = reduce (+) 0i64 (map (\t -> if t == i then 1 else 0) past_tokens)
    in logits[i] - penalty * f32.i64 count
  )

def presence_penalty_apply [n] (logits: [n]f32) (past_tokens: []i64) (penalty: f32) : [n]f32 =
  tabulate n (\i ->
    let present = any (\t -> t == i) past_tokens
    in if present then logits[i] - penalty else logits[i]
  )

def sample_argmax [n] (logits: [n]f32) : i64 =
  let (_, idx) = reduce (\(a, ai) (b, bi) -> if a >= b then (a, ai) else (b, bi)) (f32.lowest, 0i64) (zip logits (iota n))
  in idx

def sample_multinomial [n] (probs: [n]f32) (rand: f32) : i64 =
  let cumsum = scan (+) 0.0f32 probs
  in reduce_comm (\acc i -> if cumsum[i] >= rand && acc == n then i else acc) n (iota n)

def sample_with_temperature [n] (logits: [n]f32) (temperature: f32) (rand: f32) : i64 =
  let scaled = temperature_scale logits temperature
  let probs = softmax scaled
  in sample_multinomial probs rand

def sample_top_p [n] (logits: [n]f32) (p: f32) (temperature: f32) (rand: f32) : i64 =
  let scaled = temperature_scale logits temperature
  let filtered = top_p_filter scaled p
  let probs = softmax filtered
  in sample_multinomial probs rand

def sample_full [n] (logits: [n]f32) (temperature: f32) (top_p: f32) (rep_penalty: f32) (past_tokens: []i64) (rand: f32) : i64 =
  let scaled = temperature_scale logits temperature
  let penalized = repetition_penalty_apply scaled past_tokens rep_penalty
  let filtered = top_p_filter penalized top_p
  let probs = softmax filtered
  in sample_multinomial probs rand

def check_stop_token (token: i64) (stop_tokens: []i64) : bool =
  any (\st -> st == token) stop_tokens

def transformer_attention_prefill [batch][seq_len][num_heads][head_dim][page_size]
    (q: [batch][seq_len][num_heads][head_dim]f32)
    (k: [batch][seq_len][num_heads][head_dim]f32)
    (v: [batch][seq_len][num_heads][head_dim]f32)
    (page_tables: [batch][]i64)
    (kv_k: [batch][num_heads][][page_size][head_dim]f32)
    (kv_v: [batch][num_heads][][page_size][head_dim]f32)
    (scale: f32) : ([batch][seq_len][num_heads][head_dim]f32, [batch][num_heads][][page_size][head_dim]f32, [batch][num_heads][][page_size][head_dim]f32) =
  let results = map4 (\qi ki vi (pt, kvki, kvvi) ->
    let qh = transpose qi
    let kh = transpose ki
    let vh = transpose vi
    in map3 (\qhead khead vhead ->
      paged_attention_prefill qhead khead vhead pt (head kvki) (head kvvi) scale
    ) qh kh vh
  ) q k v (zip3 page_tables kv_k kv_v)
  let out = map (\r -> transpose (map (\(o, _, _) -> o) r)) results
  let new_k = map (\r -> map (\(_, nk, _) -> nk) r) results
  let new_v = map (\r -> map (\(_, _, nv) -> nv) r) results
  in (out, new_k, new_v)

def transformer_attention_decode [batch][num_heads][head_dim][page_size]
    (q: [batch][num_heads][head_dim]f32)
    (page_tables: [batch][]i64)
    (kv_k: [batch][num_heads][][page_size][head_dim]f32)
    (kv_v: [batch][num_heads][][page_size][head_dim]f32)
    (seq_lens: [batch]i64)
    (scale: f32) : [batch][num_heads][head_dim]f32 =
  map4 (\qi (pt, kvki, kvvi) sl _ ->
    map3 (\qh kh vh ->
      paged_attention_decode qh pt kh vh sl scale
    ) qi kvki kvvi
  ) q (zip3 page_tables kv_k kv_v) seq_lens (iota batch)

def glm4_qkv_proj [batch][seq_len][hidden][num_heads][head_dim]
    (x: [batch][seq_len][hidden]fp8_e4m3)
    (wq: [num_heads * head_dim][hidden]fp8_e4m3)
    (wk: [num_heads * head_dim][hidden]fp8_e4m3)
    (wv: [num_heads * head_dim][hidden]fp8_e4m3) : ([batch][seq_len][num_heads][head_dim]f32, [batch][seq_len][num_heads][head_dim]f32, [batch][seq_len][num_heads][head_dim]f32) =
  let wqf = map fp8_arr_to_f32 wq
  let wkf = map fp8_arr_to_f32 wk
  let wvf = map fp8_arr_to_f32 wv
  let result = map (\batch_x ->
    map (\token_x ->
      let xf = fp8_arr_to_f32 token_x
      let q_flat = matvec_f32 wqf xf
      let k_flat = matvec_f32 wkf xf
      let v_flat = matvec_f32 wvf xf
      let q_heads = unflatten num_heads head_dim q_flat
      let k_heads = unflatten num_heads head_dim k_flat
      let v_heads = unflatten num_heads head_dim v_flat
      in (q_heads, k_heads, v_heads)
    ) batch_x
  ) x
  let q = map (map (\(q, _, _) -> q)) result
  let k = map (map (\(_, k, _) -> k)) result
  let v = map (map (\(_, _, v) -> v)) result
  in (q, k, v)

def glm4_attention_output [batch][seq_len][num_heads][head_dim][hidden]
    (attn_out: [batch][seq_len][num_heads][head_dim]f32)
    (wo: [hidden][num_heads * head_dim]fp8_e4m3) : [batch][seq_len][hidden]fp8_e4m3 =
  let wof = map fp8_arr_to_f32 wo
  in map (\batch_attn ->
    map (\token_attn ->
      let flat = flatten token_attn
      let out = matvec_f32 wof flat
      in f32_arr_to_fp8 out
    ) batch_attn
  ) attn_out

def glm4_mlp_block [batch][seq_len][hidden][intermediate]
    (x: [batch][seq_len][hidden]fp8_e4m3)
    (w1: [intermediate][hidden]fp8_e4m3)
    (w2: [hidden][intermediate]fp8_e4m3)
    (wg: [intermediate][hidden]fp8_e4m3) : [batch][seq_len][hidden]fp8_e4m3 =
  let w1f = map fp8_arr_to_f32 w1
  let w2f = map fp8_arr_to_f32 w2
  let wgf = map fp8_arr_to_f32 wg
  in map (\batch_x ->
    map (\token_x ->
      let xf = fp8_arr_to_f32 token_x
      let h1 = matvec_f32 w1f xf
      let gate = matvec_f32 wgf xf
      let h_act = swiglu h1 gate
      let out = matvec_f32 w2f h_act
      in f32_arr_to_fp8 out
    ) batch_x
  ) x

def glm4_moe_block [batch][seq_len][hidden][num_experts][intermediate]
    (x: [batch][seq_len][hidden]fp8_e4m3)
    (router_w: [num_experts][hidden]fp8_e4m3)
    (expert_w1: [num_experts][intermediate][hidden]fp8_e4m3)
    (expert_w2: [num_experts][hidden][intermediate]fp8_e4m3)
    (expert_wg: [num_experts][intermediate][hidden]fp8_e4m3)
    (top_k: i64) : [batch][seq_len][hidden]fp8_e4m3 =
  map (\batch_x ->
    map (\token_x ->
      moe_forward_fp8 token_x router_w expert_w1 expert_w2 expert_wg top_k
    ) batch_x
  ) x

def reduce_sum_shard [n][num_shards] (shards: [num_shards][n]f32) : [n]f32 =
  reduce (map2 (+)) (replicate n 0.0f32) shards

def all_reduce_ring [n][num_gpus] (data: [num_gpus][n]f32) : [num_gpus][n]f32 =
  let summed = reduce_sum_shard data
  in replicate num_gpus summed

def all_gather [n][num_gpus] (shards: [num_gpus][n]f32) : [n * num_gpus]f32 =
  flatten shards

def all_to_all [num_gpus][n] (data: [num_gpus][num_gpus][n]f32) : [num_gpus][num_gpus][n]f32 =
  tabulate num_gpus (\dst ->
    tabulate num_gpus (\src ->
      data[src, dst]
    )
  )

def shard_heads [batch][seq_len][num_heads][head_dim][num_gpus]
    (x: [batch][seq_len][num_heads][head_dim]f32)
    (gpu_id: i64) : [batch][seq_len][num_heads / num_gpus][head_dim]f32 =
  let heads_per_gpu = num_heads / num_gpus
  let start_head = gpu_id * heads_per_gpu
  in map (\bi ->
    map (\si ->
      tabulate heads_per_gpu (\hi ->
        x[bi, si, start_head + hi]
      )
    ) (iota seq_len)
  ) (iota batch)

def gather_heads [batch][seq_len][heads_per_gpu][head_dim][num_gpus]
    (shards: [num_gpus][batch][seq_len][heads_per_gpu][head_dim]f32) : [batch][seq_len][heads_per_gpu * num_gpus][head_dim]f32 =
  map (\bi ->
    map (\si ->
      flatten (map (\shard -> shard[bi, si]) shards)
    ) (iota seq_len)
  ) (iota batch)

def shard_experts [num_experts][intermediate][hidden][num_gpus]
    (expert_w1: [num_experts][intermediate][hidden]f32)
    (gpu_id: i64) : [num_experts / num_gpus][intermediate][hidden]f32 =
  let experts_per_gpu = num_experts / num_gpus
  let start_expert = gpu_id * experts_per_gpu
  in tabulate experts_per_gpu (\ei ->
    expert_w1[start_expert + ei]
  )

entry prefill_attention [batch][seq_len][num_heads][head_dim][page_size]
    (q: [batch][seq_len][num_heads][head_dim]f32)
    (k: [batch][seq_len][num_heads][head_dim]f32)
    (v: [batch][seq_len][num_heads][head_dim]f32)
    (page_tables: [batch][]i64)
    (kv_k: [batch][num_heads][][page_size][head_dim]f32)
    (kv_v: [batch][num_heads][][page_size][head_dim]f32)
    (scale: f32) : ([batch][seq_len][num_heads][head_dim]f32, [batch][num_heads][][page_size][head_dim]f32, [batch][num_heads][][page_size][head_dim]f32) =
  transformer_attention_prefill q k v page_tables kv_k kv_v scale

entry decode_step_attention [batch][num_heads][head_dim][page_size]
    (q: [batch][num_heads][head_dim]f32)
    (page_tables: [batch][]i64)
    (kv_k: [batch][num_heads][][page_size][head_dim]f32)
    (kv_v: [batch][num_heads][][page_size][head_dim]f32)
    (seq_lens: [batch]i64)
    (scale: f32) : [batch][num_heads][head_dim]f32 =
  transformer_attention_decode q page_tables kv_k kv_v seq_lens scale

entry moe_layer [batch][hidden][num_experts][intermediate]
    (x: [batch][hidden]f32)
    (router: [num_experts][hidden]f32)
    (w1: [num_experts][intermediate][hidden]f32)
    (w2: [num_experts][hidden][intermediate]f32)
    (wg: [num_experts][intermediate][hidden]f32)
    (top_k: i64) : [batch][hidden]f32 =
  map (\xi -> moe_forward xi router w1 w2 wg top_k) x

entry moe_layer_fp8 [batch][hidden][num_experts][intermediate]
    (x: [batch][hidden]i8)
    (router: [num_experts][hidden]i8)
    (w1: [num_experts][intermediate][hidden]i8)
    (w2: [num_experts][hidden][intermediate]i8)
    (wg: [num_experts][intermediate][hidden]i8)
    (top_k: i64) : [batch][hidden]i8 =
  map (\xi -> moe_forward_fp8 xi router w1 w2 wg top_k) x

entry rms_norm_layer [batch][hidden]
    (x: [batch][hidden]f32)
    (gamma: [hidden]f32)
    (eps: f32) : [batch][hidden]f32 =
  map (\xi -> rms_norm xi gamma eps) x

entry rms_norm_layer_fp8 [batch][hidden]
    (x: [batch][hidden]i8)
    (gamma: [hidden]f32)
    (eps: f32) : [batch][hidden]i8 =
  map (\xi -> rms_norm_fp8 xi gamma eps) x

entry sample_tokens [batch][vocab]
    (logits: [batch][vocab]f32)
    (temperature: f32)
    (top_p: f32)
    (rep_penalty: f32)
    (past_tokens: [batch][]i64)
    (random_vals: [batch]f32) : [batch]i64 =
  map3 (\l pt rand ->
    sample_full l temperature top_p rep_penalty pt rand
  ) logits past_tokens random_vals

entry sample_tokens_argmax [batch][vocab]
    (logits: [batch][vocab]f32) : [batch]i64 =
  map sample_argmax logits

entry fp8_gemm [m][n][k]
    (a: [m][k]i8)
    (b: [k][n]i8) : [m][n]i8 =
  matmul_fp8 a b

entry fp8_gemm_to_f16 [m][n][k]
    (a: [m][k]i8)
    (b: [k][n]i8) : [m][n]f16 =
  matmul_fp8_to_f16 a b

entry rope_transform [batch][seq_len][hidden]
    (x: [batch][seq_len][hidden]i8)
    (positions: [batch]i64)
    (rope_base: f32) : [batch][seq_len][hidden]i8 =
  let freqs = rope_freqs hidden rope_base
  in map2 (\xi pos ->
    map (\s ->
      rope_fp8 xi[s] freqs (pos + s)
    ) (iota seq_len)
  ) x positions

entry rope_transform_f32 [batch][seq_len][hidden]
    (x: [batch][seq_len][hidden]f32)
    (positions: [batch]i64)
    (rope_base: f32) : [batch][seq_len][hidden]f32 =
  let freqs = rope_freqs hidden rope_base
  in map2 (\xi pos ->
    map (\s ->
      apply_rope xi[s] freqs (pos + s)
    ) (iota seq_len)
  ) x positions

entry embedding_forward [batch][vocab_size][hidden]
    (token_ids: [batch]i64)
    (embed_table: [vocab_size][hidden]f32) : [batch][hidden]f32 =
  embedding_batch token_ids embed_table

entry embedding_forward_fp8 [batch][vocab_size][hidden]
    (token_ids: [batch]i64)
    (embed_table: [vocab_size][hidden]i8) : [batch][hidden]i8 =
  map (\tid -> embed_table[tid]) token_ids

entry residual_forward [batch][hidden]
    (x: [batch][hidden]f32)
    (residual: [batch][hidden]f32) : [batch][hidden]f32 =
  map2 (\xi ri -> residual_add xi ri) x residual

entry residual_forward_fp8 [batch][hidden]
    (x: [batch][hidden]i8)
    (residual: [batch][hidden]i8) : [batch][hidden]i8 =
  map2 (\xi ri -> residual_add_fp8 xi ri) x residual

entry swiglu_forward [batch][hidden]
    (x: [batch][hidden]f32)
    (gate: [batch][hidden]f32) : [batch][hidden]f32 =
  map2 swiglu x gate

entry geglu_forward [batch][hidden]
    (x: [batch][hidden]f32)
    (gate: [batch][hidden]f32) : [batch][hidden]f32 =
  map2 geglu x gate

entry softmax_forward [batch][n]
    (x: [batch][n]f32) : [batch][n]f32 =
  map softmax x

entry all_reduce_sum [num_gpus][n]
    (data: [num_gpus][n]f32) : [num_gpus][n]f32 =
  all_reduce_ring data

entry logits_to_probs [batch][vocab]
    (logits: [batch][vocab]f32)
    (temperature: f32) : [batch][vocab]f32 =
  map (\l -> softmax (temperature_scale l temperature)) logits
