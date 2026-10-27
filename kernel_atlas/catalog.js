/* Kernel Atlas index content as data: models, kernel families, launch order.
   index.html renders it; kernel pages link back by file name. */
window.CATALOG = (() => {
  const BASE = '';
  const ENGINES = {
    gptoss: { name: 'gpt-oss-20b', short: 'gpt-oss', page: 'gptoss.html' },
    gemma3: { name: 'Gemma 3 12B', short: 'Gemma 3', page: 'gemma3.html' },
    emb2: { name: 'EmbeddingGemma 2', short: 'EmbGemma 2', page: 'gemma_emb2.html' },
    qwen35: { name: 'Qwen 3.5 9B', short: 'Qwen 3.5', page: 'qwen35.html' },
    llama31: { name: 'Llama 3.1 8B', short: 'Llama 3.1', page: 'llama31.html' },
  };
  const MODELS = [
    { id: 'gptoss', fig: 'figures/gptoss.svg', desc: '24 layers alternating local (window 128) and global attention, each with a 32-expert MoE feed-forward (top 4).',
      shows: 'The GGUF graph with each step named by its kernel; decode vs prefill; launch order; every tensor and its engine format.',
      ptx: 'gptoss.ptx · 15 kernels',
      spec: [['layers', '24'], ['width', '2,880'], ['q / kv heads', '64 / 8'], ['feed-forward', '32 experts, top 4'], ['weights', 'MXFP4 experts, Q8_0']] },
    { id: 'gemma3', fig: 'figures/gemma3.svg', desc: '48 layers, five local (window 1024) to one global, dense GELU feed-forward, tied output head.',
      shows: 'The same views; add_rmsnorm fusing the residual add with the next norm.',
      ptx: 'gemma3.ptx · 13 kernels · Triton port: kernels.py · 15 kernels',
      spec: [['layers', '48'], ['width', '3,840'], ['q / kv heads', '16 / 8'], ['feed-forward', '15,360, GEGLU'], ['weights', 'Q5_0 FFN, Q8_0']] },
    { id: 'emb2', fig: 'figures/gemma_emb2.svg', desc: '271M encoder: 24 bidirectional layers (five ±512 local to one whole-text global), per-layer inputs, one 768-dim vector per text.',
      shows: 'Packed texts in one batch, 256- and 512-dim heads, the per-layer-input branch, mean pooling on the host.',
      ptx: 'emb2.ptx · 9 kernels, 4 shared with gemma3.ptx',
      spec: [['layers', '24'], ['width', '512 → 768 out'], ['q / kv heads', '4 / 2 or 1'], ['feed-forward', '2,048, GEGLU'], ['weights', 'Q8_0']] },
    { id: 'qwen35', fig: 'figures/qwen35.svg', desc: 'Hybrid: 8 blocks of three Gated DeltaNet layers (linear attention, a 128 × 128 state per head) and one gated full-attention layer; SwiGLU feed-forward.',
      shows: 'The recurrent state in place of a KV cache, the conv state, four quantized weight types in one file, the int8 tensor-core prompt GEMM.',
      ptx: 'qwen35.ptx · 17 kernels',
      spec: [['layers', '32 (24 DeltaNet + 8 attention)'], ['width', '4,096'], ['heads', 'DeltaNet 32 × 128² · attention 16 / 4'], ['feed-forward', '12,288, SwiGLU'], ['weights', 'Q4_1, Q5_K, Q6_K, Q8_0']] },
    { id: 'llama31', fig: 'figures/llama31.svg', desc: 'The plain pre-norm decoder: 32 identical layers of grouped-query attention (RoPE with Llama 3 frequency scaling) and a SwiGLU feed-forward, all Q8_0.',
      shows: 'The reference transformer with nothing fused away: one matrix kernel for 9 weight formats, adjacent-pair RoPE, a 128-dim attention and a KV cache reused across chat turns.',
      ptx: 'llama31.ptx · 10 kernels, 6 shared with qwen35.ptx',
      spec: [['layers', '32'], ['width', '4,096'], ['q / kv heads', '32 / 8 × 128'], ['feed-forward', '14,336, SwiGLU'], ['weights', 'Q8_0 (any of 9 formats)']] },
  ];
  const FAMILIES = [
    { id: 'steps', name: 'Steps across engines', kernels: [
      ['qkv', ['gptoss', 'gemma3', 'emb2'], 'The ×3 q, k, v projections that start every attention block, compared across the three engines.', 'To-scale weight shapes, grouped-query sharing, a 4-token attention toy.', '3 launches per layer'],
      ['gate_up', ['gptoss', 'gemma3', 'emb2'], 'The gate and up projections that start every feed-forward block; stacked into one launch in the Gemma engines.', 'The stacked matrix, the [gate | up] row, GEGLU vs clamped SwiGLU on a toy.', '1 launch (Gemma) or 2 (gpt-oss)'],
    ] },
    { id: 'norm', name: 'Norms and activations', kernels: [
      ['rmsnorm', ['gptoss', 'gemma3', 'emb2', 'qwen35', 'llama31'], 'Scales a hidden vector to unit RMS and applies per-dimension weights.', 'Strided loop, warp butterfly, shared-memory partials, one barrier.', 'grid (tokens) · 1024 threads'],
      ['add_rmsnorm', ['gemma3', 'emb2'], 'Post-norm, residual add and the next pre-norm of a row in one launch, twice per layer.', 'Three passes, two reductions, both call sites, the memory saved by fusing.', 'grid (tokens) · 1024 threads'],
      ['norm_add', ['emb2'], "Post-norm and residual add with each layer's learned output scale folded in, twice per layer.", "One reduction, two passes, the 24 real layer scales, why add_rmsnorm's third pass is dropped.", 'grid (tokens) · 256 threads'],
      ['qk_norm_rope', ['gemma3'], 'Per-head RMSNorm and rotary embedding for queries and keys, plus the f16 KV-cache write.', "Why lanes own dims 4l and 128 + 4l, every pair's rotation angle, the ring slot.", 'grid (32 heads, tokens) · 32 threads'],
      ['qk_norm_rope_qwen', ['qwen35'], "Qwen 3.5's rewrite of qk_norm_rope: per-head RMSNorm, RoPE on dims 0 … 63 only, k and v to the f16 cache.", 'Lanes owning 8 consecutive dims, the l ^ 4 partner shuffle, q heads read past their gates.', 'grid (24 heads, tokens) · 32 threads'],
      ['head_norm_rope', ['emb2'], 'Per-head RMSNorm and RoPE for 256- and 512-dim heads, value heads normalized without weight, positions per token.', 'Pair ownership with a 128-stride, a 4-warp reduction, packed positions, the k_norm constants that replace 1/√d.', 'grid (8 or 6 heads, tokens) · 128 threads'],
      ['swiglu_oai', ['gptoss'], "OpenAI's clamped SwiGLU, combining gate and up projections of each expert.", 'Every register for one element; grid-stride laps in decode vs prefill.', '≤ 448 blocks · 256 threads · grid-stride'],
      ['geglu', ['gemma3', 'emb2'], 'Tanh-approximated GELU on the gate half times the up half.', 'tanh from one ex2, the tanh vs erf GELU curves, flat-index arithmetic.', '≤ 448 blocks · 256 threads · grid-stride'],
      ['gated_norm', ['qwen35'], "Per-head RMSNorm of the DeltaNet read-out, times a shared weight and the output gate SiLU(z).", 'Four warp butterflies and a 4-entry shared merge, z read in place from the qkvz buffer.', 'grid (32 heads, tokens) · 128 threads'],
      ['swiglu', ['qwen35', 'llama31'], "SiLU of the gate half times the up half, the feed-forward activation of Qwen 3.5 and Llama 3.1.", 'SiLU as g / (1 + e^−g) with one ex2 and one division; the flat-index arithmetic.', '≤ 448 blocks · 256 threads · grid-stride'],
      ['gelu_mul', ['emb2'], "GELU of the per-layer gate times that layer's slice of the per-layer inputs, in place.", 'A strided view into the [T][24][512] buffer, every register of one element.', '≤ 448 blocks · 256 threads · grid-stride'],
    ] },
    { id: 'mm', name: 'Matrix products', kernels: [
      ['mv_q8', ['gptoss', 'gemma3'], 'Decode-time matrix × vector with 8-bit weights: one warp per output row.', 'Contiguous kilobyte loads per round, dot4_s8, the per-row reduction.', 'grid ⌈rows / 8⌉ · 256 threads'],
      ['gemm_q8', ['gptoss', 'gemma3', 'emb2'], 'Prefill matrix × matrix: 64 rows × 32 tokens per block through shared memory.', 'Tile staging, thread roles from the index bits, bank-conflict padding.', 'grid (rows / 64, ⌈T / 32⌉) · 256 threads'],
      ['quant_q8', ['gptoss', 'gemma3'], 'Turns a float vector into int8 blocks of 32 with one scale each, for the integer dot products.', 'The warp-wide max, rounding, all six call sites, the in-place mode used in prefill.', '≤ 448 blocks · 256 threads · grid-stride over blocks'],
      ['quant_q8_qwen', ['qwen35', 'llama31'], "Qwen 3.5's quant_q8 (Llama 3.1 runs it too): the same int8 blocks plus d · Σq per block, which the Q4_1 and Q5_K offsets need.", 'A second, integer butterfly for Σq, the {d, d·Σq} float2, the offset term worked both ways.', '≤ 448 blocks · 256 threads · grid-stride over blocks'],
      ['mv_q5', ['gemma3'], 'Decode matrix × vector with Q5_0 feed-forward weights and int8 activations.', 'Rebuilding 5-bit weights with one multiply, dp4a, the −16 correction; gate ‖ up and down.', 'grid ⌈rows / 8⌉ · 256 threads'],
      ['mv_q4_1', ['qwen35'], 'Decode matrix × vector with Q4_1 weights (scale and offset per 32): 4 of the 5 GB a Qwen 3.5 token reads.', 'Nibbles straight into dp4a, the offset folded into one FMA with quant_q8\'s dₓΣx.', 'grid ⌈rows / 8⌉ · 256 threads'],
      ['mv_q8_0', ['qwen35'], 'Decode matrix × vector with Q8_0 weights: the DeltaNet gates, or every matrix of a BF16-loaded model.', 'Eight signed dp4a and one FMA per block; a 64-row launch that is over in microseconds.', 'grid ⌈rows / 8⌉ · 256 threads'],
      ['mv_q5_k', ['qwen35'], 'Decode matrix × vector with Q5_K super-blocks: the DeltaNet output projection.', 'The 176-byte layout, 6-bit scales and mins with bfe and selp, the fifth bit from a shared array.', 'grid ⌈rows / 8⌉ · 256 threads'],
      ['mv_q', ['llama31'], 'Decode matrix × vector for any of 9 ggml formats, from Q8_0 to IQ3_S: one decoder, unpack16, turns 16 weights into int8 and a scale.', 'Each format\'s bytes becoming the same 4 words, two dp4a halves, the generic {s₀, s₁, m} FMA, the 7.95 GB a Llama token reads.', 'grid ⌈rows / 8⌉ · 256 threads'],
      ['gemm_q', ['llama31', 'qwen35'], 'The prompt GEMM: 64 rows × 64 tokens per block on int8 tensor cores, two mma.m16n8k16 per 32-value block, any weight format.', 'Loader roles from the thread index, the 80-byte padded rows, A and B fragments, the per-half scales in the epilogue math; Qwen 3.5\'s three-format version.', 'grid (⌈T / 64⌉, ⌈rows / 64⌉) · 256 threads'],
      ['mv_mxfp4', ['gptoss'], 'Decode matrix × vector for the 4 chosen experts, 4-bit E2M1 weights decoded with byte permutes.', 'Reading 4 of 32 experts, fp4x4 via prmt, the e8m0 scale; gate, up and down.', 'grid (360, 4 slots) · 256 threads'],
    ] },
    { id: 'att', name: 'Attention', kernels: [
      ['rope_kv', ['gptoss'], 'Rotary embedding for 64 query and 8 key heads, plus the f16 KV-cache write.', 'One warp per head, NeoX pairs, the YaRN angle blend, the ring-buffer slot.', 'grid (11, tokens) · 256 threads'],
      ['attention', ['gemma3', 'qwen35'], 'Grouped-query attention over the f16 cache with a sliding window, one pass, no stored scores; Qwen 3.5 adds a q-head stride and the output gate.', '8 warps × 2 keys per step, online softmax per warp, the final merge, the Qwen additions.', 'grid (16 q heads, tokens) · 256 threads'],
      ['rope_kv_llama', ['llama31'], "Llama 3.1's rope_kv: RoPE on adjacent pairs (2i, 2i + 1) with Llama 3's frequency factors baked into the table, then k and v to the f16 cache.", 'A warp per head, 4 dims per lane and no shuffles, the 64 angles with their ÷1 … ÷8 factors, the K/V cache row.', 'grid (48 heads, tokens) · 32 threads'],
      ['attention_llama', ['llama31'], "Llama 3.1's attention: the Gemma 3 kernel cut to 128-dim heads, 4 per lane, 32 q heads over 8 kv heads, no window.", 'The same 8 warps × 2 keys and online softmax, 512 bytes per key, half the block idle in the final write.', 'grid (32 q heads, tokens) · 256 threads'],
      ['attention_oss', ['gptoss'], "gpt-oss's attention: 64-dim heads, an exact two-pass softmax with the scores in shared memory, and a learned sink logit per head.", '4 warps × 1 key, the two passes, the sink\'s share of the weight, the 256-slot ring.', 'grid (64 q heads, tokens) · 128 threads · 4 B per key'],
      ['attention_enc', ['emb2'], 'Bidirectional attention within each packed text, ±512 on local layers, with an exact two-pass softmax in shared memory.', "The batch's attention pattern, the key range from seg and window, scores → max → exp → sum, two accumulators.", 'grid (4 q heads, tokens) · 256 threads · 4·keys B'],
    ] },
    { id: 'lin', name: 'Linear attention', kernels: [
      ['gdn_conv', ['qwen35'], "The DeltaNet's causal conv1d: each of 8192 channels mixes its last 4 inputs, then SiLU, then the q and k heads are L2-normalized.", 'The conv state rows before the chunk, the 4 taps, the fused head norm, conv_save.', 'grid (64 heads, tokens) · 128 threads'],
      ['gdn_recur', ['qwen35'], "The gated delta rule of Qwen 3.5's DeltaNet layers: a 128 × 128 state per head, decayed, corrected toward each new (k, v) and read with q, token by token.", 'The state slice in shared memory, the two passes per token, the gates, a float64 check of the recurrence.', 'grid (32 heads, 4) · 32 threads'],
    ] },
    { id: 'rs', name: 'Routing and sampling', kernels: [
      ['mv_f32', ['gptoss'], 'The MoE router: 32 expert scores per token from a float32 32 × 2880 matrix.', 'One warp per row, 90 coalesced loads per lane, the logits handed to router_topk.', 'grid (4, tokens) · 256 threads'],
      ['router_topk', ['gptoss'], "Picks each token's 4 experts from 32 logits and softmaxes their weights.", 'Four rounds of warp argmax with tie-breaking, a softmax over the picks.', 'grid (tokens) · 32 threads'],
      ['moe_sort', ['gptoss'], "Groups (token, expert) pairs by expert so each expert's weights are read once per chunk.", 'Histogram, serial scan, scatter with atomics; arbitrary but harmless order.', '1 block · 1024 threads'],
      ['moe_combine', ['gptoss'], 'Mixes the 4 expert outputs with the router weights and adds them to the residual stream.', "The [token][slot][2880] layout, one element's four FMAs, grid-stride over tokens.", '≤ 448 blocks · 256 threads · grid-stride'],
      ['argmax', ['gptoss', 'gemma3', 'qwen35', 'llama31'], 'Finds the highest logit of 201,088 with deterministic tie-breaking.', 'Thread, warp and block levels; a planted tie resolved to the smaller index.', '1 block · 1024 threads'],
      ['compact_logits', ['gptoss', 'gemma3', 'qwen35', 'llama31'], 'Keeps only logits near the maximum so the host sorts hundreds of candidates instead of the vocabulary.', "The threshold moving with temperature, atomic append, the host's fallback ladder.", '≤ 448 blocks · 256 threads · grid-stride'],
    ] },
  ].map(f => ({ ...f, kernels: f.kernels.map(([name, engines, desc, shows, launch]) => ({ name, engines, desc, shows, launch, href: BASE + name + '.html' })) }));
  // one layer, in launch order: [kernel, note, note page]; a kernel without a page is marked with page: false
  const FLOWS = [
    { id: 'gptoss', note: 'decode (prefill: mv_* become gemm_*, and moe_sort runs after router_topk)', steps: [
      ['rmsnorm'], ['mv_q8', '×3 q, k, v', 'qkv'], ['rope_kv'], ['attention_oss'], ['mv_q8', '+ residual'], ['rmsnorm'], ['mv_f32'], ['router_topk'],
      ['quant_q8'], ['mv_mxfp4', '×2 gate, up', 'gate_up'], ['swiglu_oai'], ['quant_q8'], ['mv_mxfp4', 'down'], ['moe_combine']] },
    { id: 'gemma3', note: 'decode (prefill: gemm_q8 and gemm_q5; rmsnorm runs once before layer 0)', steps: [
      ['mv_q8', '×3 q, k, v', 'qkv'], ['qk_norm_rope'], ['attention'], ['mv_q8'], ['add_rmsnorm'], ['quant_q8'], ['mv_q5', 'gate ‖ up', 'gate_up'],
      ['geglu'], ['quant_q8'], ['mv_q5'], ['add_rmsnorm']] },
    { id: 'emb2', note: 'all T tokens of a batch at once (every product is gemm_q8)', steps: [
      ['gemm_q8', '×3 q, k, v', 'qkv'], ['head_norm_rope'], ['attention_enc'], ['gemm_q8'], ['add_rmsnorm'], ['gemm_q8', 'gate ‖ up', 'gate_up'], ['geglu'],
      ['gemm_q8'], ['norm_add'], ['gemm_q8', 'inp_gate'], ['gelu_mul'], ['gemm_q8', 'proj'], ['norm_add', '× layer scale'], ['rmsnorm']] },
    { id: 'qwen35', note: 'decode, a DeltaNet layer (prefill: every mv_* becomes gemm_q; attention layers run mv_q4_1 ×3 q|gate, k, v → qk_norm_rope → attention → mv_q4_1 in place of steps 4–11)', steps: [
      ['rmsnorm'], ['quant_q8_qwen'], ['mv_q4_1', 'qkv ‖ z'], ['mv_q8_0', 'β ‖ α'], ['gdn_conv'], ['conv_save', 'on the gdn_conv page', 'gdn_conv', false],
      ['gdn_recur', 'state 32 × 128²'], ['gated_norm'], ['quant_q8_qwen'], ['mv_q5_k', '+ residual'], ['rmsnorm'], ['quant_q8_qwen'],
      ['mv_q4_1', 'gate ‖ up'], ['swiglu'], ['quant_q8_qwen'], ['mv_q4_1', 'down + residual']] },
    { id: 'llama31', note: 'decode (prefill: every mv_q becomes gemm_q)', steps: [
      ['rmsnorm'], ['quant_q8_qwen'], ['mv_q', 'q ‖ k ‖ v'], ['rope_kv_llama', 'k, v → cache'], ['attention_llama'], ['quant_q8_qwen'], ['mv_q', 'o + residual'],
      ['rmsnorm'], ['quant_q8_qwen'], ['mv_q', 'gate ‖ up'], ['swiglu'], ['quant_q8_qwen'], ['mv_q', 'down + residual']] },
  ];
  const TAIL = { note: 'After the last layer, the four generators (Qwen 3.5: rmsnorm, quant_q8, then mv_q6_k as the head; Llama 3.1: rmsnorm, quant_q8, then mv_q; EmbeddingGemma 2 instead: gemm_q8 to 768 values per token, then mean pooling on the CPU)',
    steps: [['mv_q8', 'output head'], ['argmax'], ['sample_logits', null, null, false], ['compact_logits', 'top-k / top-p only']] };
  const HOW = [
    ['Step through it', "Play, the arrow buttons, the slider, or your keyboard's ← and → keys move one step of the kernel at a time."],
    ['Point at the data', 'Hovering over a cell names the element, the thread that owns it and the values involved. Clicking selects that thread so every panel follows it.'],
    ['Read the code', 'The source panel highlights the lines for the current step, in hand-written PTX or the equivalent CUDA C++, and on Gemma 3 kernels in the Triton port too. The tab you pick is kept from page to page.'],
    ['Ask why', 'Each page ends with the design reasons: the access pattern, the reduction scheme, what the kernel avoids.'],
  ];
  const NAMES = [
    ['Kernels', 'The PTX <code>.entry</code> name, e.g. <code>swiglu_oai</code>; its page is <code>swiglu_oai.html</code>.'],
    ['Engines', 'Dots: <i class="dot d-gptoss"></i> gpt-oss, <i class="dot d-gemma3"></i> Gemma 3, <i class="dot d-emb2"></i> EmbeddingGemma 2, <i class="dot d-qwen35"></i> Qwen 3.5, <i class="dot d-llama31"></i> Llama 3.1. Several dots mean the same PTX runs in each: emb2.ptx copies rmsnorm, add_rmsnorm, geglu and gemm_q8 from gemma3.ptx unchanged, qwen35.ptx copies rmsnorm, argmax and compact_logits, llama31.ptx copies rmsnorm, quant_q8, swiglu, argmax and compact_logits from qwen35.ptx. Where two engines have different code under one name, the second page gets a suffix: <code>qk_norm_rope_qwen</code>, <code>quant_q8_qwen</code>, <code>attention_oss</code>, <code>rope_kv_llama</code>, <code>attention_llama</code>; tracks show the <code>.entry</code> name.'],
    ['Tensors', 'GGUF names without <code>blk.N.</code>; shapes are rows × cols (output × input); types F32, F16, BF16, Q8_0, Q5_0, Q4_1, the K-quants, MXFP4.'],
    ['Layers and buffers', '<i>local</i> = sliding-window layer, <i>global</i> = full context. Buffers use the engine\'s names: x, xn, qkv, att, y, gu, h, logits.'],
  ];
  const INTRO = "Interactive walkthroughs of the GPU kernels behind the five inference engines in this project: four text generators, among them the hybrid linear-attention Qwen 3.5 and the plain Llama 3.1, and one text-embedding encoder. Each page steps through one kernel the way the hardware runs it: which thread touches which bytes, where threads cooperate, and the matching lines of the hand-written PTX and of an equivalent CUDA C++ version. Every number on the pages is computed in your browser by re-running the kernel's float32 arithmetic on generated data. Gemma 3 also has a Triton rewrite; its kernels appear on the same pages as a third listing, tagged with the same steps, with a section on how each one differs from the PTX.";
  const FOOT = 'Sources: <code>gptoss/gptoss.ptx</code>, <code>gemma3/gemma3.ptx</code>, <code>gemma_emb2/emb2.ptx</code>, <code>qwen35/qwen35.ptx</code>, <code>llama31/llama31.ptx</code> and their host code; the Qwen 3.5 and Llama 3.1 model pages are drawn from <code>forward()</code> in <code>qwen35/qwen35.c</code> and <code>llama31/llama31.c</code> and the GGUF headers; Qwen 3.5\'s one kernel without a page yet is mv_q6_k (the output head); the Triton port <code>gemma3_triton/kernels.py</code> (compiler figures from Triton 3.8 on sm_89). Model graphs: <code>model_htmls/gptoss_graph.html</code> and <code>gemma3_graph.html</code> (regenerated for the 12B). Kernels without a page yet: gemm_q5, gemm_mxfp4, sample_logits and EmbeddingGemma 2\'s one-line scale.<br>Copyright (c) 2026 Edward Jiang, all rights reserved.';
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const page = name => BASE + name + '.html';
  // pages whose id is not the PTX .entry name (two engines, different code, one name)
  const ENTRY = { qk_norm_rope_qwen: 'qk_norm_rope', quant_q8_qwen: 'quant_q8', attention_oss: 'attention', rope_kv_llama: 'rope_kv', attention_llama: 'attention' };
  // kernels whose page carries the Triton port of gemma3_triton/kernels.py
  const TRITON = ['rmsnorm', 'add_rmsnorm', 'qk_norm_rope', 'geglu', 'mv_q8', 'gemm_q8', 'quant_q8', 'mv_q5', 'attention', 'argmax', 'compact_logits'];
  return { TRITON, ENTRY, BASE, ENGINES, MODELS, FAMILIES, FLOWS, TAIL, HOW, NAMES, INTRO, FOOT, esc, page };
})();
