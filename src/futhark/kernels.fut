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
       if sign == 1 then f32.nan else f32.nan
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

def fp8_arr_to_f32 [n] (arr: [n]fp8_e4m3) : [n]f32 =
  map fp8_to_f32 arr

def f32_arr_to_fp8 [n] (arr: [n]f32) : [n]fp8_e4m3 =
  map f32_to_fp8 arr

def rms_norm [n] (x: [n]f32) (gamma: [n]f32) (eps: f32) : [n]f32 =
  let sq_sum = reduce (+) 0.0f32 (map (\v -> v * v) x)
  let rms = f32.sqrt (sq_sum / f32.i64 n + eps)
  in map2 (\xi gi -> (xi / rms) * gi) x gamma

def rms_norm_fp8 [n] (x: [n]fp8_e4m3) (gamma: [n]f32) (eps: f32) : [n]fp8_e4m3 =
  let xf = fp8_arr_to_f32 x
  let result = rms_norm xf gamma eps
  in f32_arr_to_fp8 result

def swiglu [n] (x: [n]f32) (gate: [n]f32) : [n]f32 =
  map2 (\xi gi -> xi * (gi / (1.0f32 + f32.exp (-gi)))) x gate

def geglu [n] (x: [n]f32) (gate: [n]f32) : [n]f32 =
  let sqrt2 = 1.41421356f32
  in map2 (\xi gi ->
    let gelu = 0.5f32 * gi * (1.0f32 + f32.tanh (0.7978845608f32 * (gi + 0.044715f32 * gi * gi * gi)))
    in xi * gelu
  ) x gate

def matmul_f32 [m][n][k] (a: [m][k]f32) (b: [k][n]f32) : [m][n]f32 =
  map (\a_row -> map (\b_col -> reduce (+) 0.0f32 (map2 (*) a_row b_col)) (transpose b)) a

def matmul_fp8 [m][n][k] (a: [m][k]fp8_e4m3) (b: [k][n]fp8_e4m3) : [m][n]fp8_e4m3 =
  let af = map fp8_arr_to_f32 a
  let bf = map fp8_arr_to_f32 b
  let cf = matmul_f32 af bf
  in map f32_arr_to_fp8 cf

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

def softmax [n] (x: [n]f32) : [n]f32 =
  let max_x = reduce f32.max f32.lowest x
  let exp_x = map (\xi -> f32.exp (xi - max_x)) x
  let sum_exp = reduce (+) 0.0f32 exp_x
  in map (\e -> e / sum_exp) exp_x

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
  let num_pages = (seq_len + page_size - 1) / page_size
  let new_k = tabulate num_pages (\p ->
    let page_idx = page_table[p]
    in tabulate page_size (\s ->
      let global_idx = p * page_size + s
      in if global_idx < seq_len then k[global_idx] else replicate head_dim 0.0f32
    )
  )
  let new_v = tabulate num_pages (\p ->
    let page_idx = page_table[p]
    in tabulate page_size (\s ->
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
  let all_scores = flatten (
    map (\p ->
      let page_idx = page_table[p]
      in map (\s ->
        let global_idx = p * page_size + s
        in if global_idx < seq_len then
             reduce (+) 0.0f32 (map2 (*) q kv_cache_k[page_idx, s]) * scale
           else
             f32.lowest
      ) (iota page_size)
    ) (iota num_pages)
  )
  let scores_trimmed = take seq_len all_scores
  let attn = softmax scores_trimmed
  in map (\d ->
    reduce (+) 0.0f32 (
      flatten (
        map2 (\p a_start ->
          let page_idx = page_table[p]
          in map2 (\s local_a ->
            let global_idx = p * page_size + s
            in if global_idx < seq_len then
                 local_a * kv_cache_v[page_idx, s, d]
               else
                 0.0f32
          ) (iota page_size) (take page_size (drop a_start attn ++ replicate page_size 0.0f32))
        ) (iota num_pages) (map (\p -> p * page_size) (iota num_pages))
      )
    )
  ) (iota head_dim)

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

def moe_expert_mlp [hidden][intermediate] (x: [hidden]f32) (w1: [intermediate][hidden]f32) (w2: [hidden][intermediate]f32) (w_gate: [intermediate][hidden]f32) : [hidden]f32 =
  let h1 = map (\row -> reduce (+) 0.0f32 (map2 (*) x row)) w1
  let gate = map (\row -> reduce (+) 0.0f32 (map2 (*) x row)) w_gate
  let h_act = swiglu h1 gate
  in map (\row -> reduce (+) 0.0f32 (map2 (*) h_act row)) w2

def moe_forward [num_experts][hidden][intermediate][top_k]
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

def embedding_lookup [vocab_size][hidden] (token_id: i64) (embed_table: [vocab_size][hidden]f32) : [hidden]f32 =
  embed_table[token_id]

def residual_add [n] (x: [n]f32) (residual: [n]f32) : [n]f32 =
  map2 (+) x residual

def temperature_scale [n] (logits: [n]f32) (temp: f32) : [n]f32 =
  map (\l -> l / temp) logits

def top_p_mask [n] (logits: [n]f32) (p: f32) : [n]f32 =
  let probs = softmax logits
  let indexed = zip probs (iota n)
  let sorted = merge_sort (\(a, _) (b, _) -> a >= b) indexed
  let cumsum = scan (+) 0.0f32 (map (.0) sorted)
  let mask = map2 (\cs idx -> if cs <= p then sorted[idx].1 else -1i64) cumsum (iota n)
  in map (\i ->
    if any (\m -> m == i) mask then logits[i] else f32.lowest
  ) (iota n)

def repetition_penalty [n] (logits: [n]f32) (past_tokens: []i64) (penalty: f32) : [n]f32 =
  map (\i ->
    if any (\t -> t == i) past_tokens then
      if logits[i] > 0.0f32 then logits[i] / penalty else logits[i] * penalty
    else
      logits[i]
  ) (iota n)

def sample_argmax [n] (logits: [n]f32) : i64 =
  let (_, idx) = reduce (\(a, ai) (b, bi) -> if a >= b then (a, ai) else (b, bi)) (f32.lowest, 0i64) (zip logits (iota n))
  in idx

entry prefill_attention [batch][seq_len][num_heads][head_dim][page_size]
    (q: [batch][seq_len][num_heads][head_dim]f32)
    (k: [batch][seq_len][num_heads][head_dim]f32)
    (v: [batch][seq_len][num_heads][head_dim]f32)
    (page_tables: [batch][]i64)
    (kv_k: [batch][num_heads][][page_size][head_dim]f32)
    (kv_v: [batch][num_heads][][page_size][head_dim]f32)
    (scale: f32) : ([batch][seq_len][num_heads][head_dim]f32, [batch][num_heads][][page_size][head_dim]f32, [batch][num_heads][][page_size][head_dim]f32) =
  let results = map4 (\qi ki vi (pt, kvki, kvvi) ->
    map3 (\qh kh vh ->
      paged_attention_prefill qh kh vh pt (head kvki) (head kvvi) scale
    ) (transpose qi) (transpose ki) (transpose vi)
  ) q k v (zip3 page_tables (transpose kv_k) (transpose kv_v))
  let out = map (\r -> transpose (map (\(o, _, _) -> o) r)) results
  let new_k = map (\r -> map (\(_, nk, _) -> nk) r) results
  let new_v = map (\r -> map (\(_, _, nv) -> nv) r) results
  in (out, new_k, new_v)

entry decode_step_attention [batch][num_heads][head_dim][page_size]
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

entry moe_layer [batch][hidden][num_experts][intermediate]
    (x: [batch][hidden]f32)
    (router: [num_experts][hidden]f32)
    (w1: [num_experts][intermediate][hidden]f32)
    (w2: [num_experts][hidden][intermediate]f32)
    (wg: [num_experts][intermediate][hidden]f32)
    (top_k: i64) : [batch][hidden]f32 =
  map (\xi -> moe_forward xi router w1 w2 wg top_k) x

entry rms_norm_layer [batch][hidden]
    (x: [batch][hidden]f32)
    (gamma: [hidden]f32)
    (eps: f32) : [batch][hidden]f32 =
  map (\xi -> rms_norm xi gamma eps) x

entry sample_tokens [batch][vocab]
    (logits: [batch][vocab]f32)
    (temperature: f32)
    (top_p: f32)
    (rep_penalty: f32)
    (past_tokens: [batch][]i64) : [batch]i64 =
  map2 (\l pt ->
    let scaled = temperature_scale l temperature
    let penalized = repetition_penalty scaled pt rep_penalty
    let masked = top_p_mask penalized top_p
    in sample_argmax masked
  ) logits past_tokens

entry fp8_gemm [m][n][k]
    (a: [m][k]i8)
    (b: [k][n]i8) : [m][n]i8 =
  matmul_fp8 a b

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
