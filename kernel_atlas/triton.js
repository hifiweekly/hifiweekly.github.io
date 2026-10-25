// Kernel Atlas - the Triton port (gemma3_triton/kernels.py) of the Gemma 3 kernels.
// One entry per kernel page: the @triton.jit source with the page's step tags, launch facts,
// what Triton 3.8 compiled it to on the RTX 4070 SUPER (sm_89), and how it differs from the PTX.
// atlas.js adds the listing as a code tab and the rest as a "Triton version" section.
(function () {
'use strict';
const A = window.Atlas;
const SRC = 'gemma3_triton/kernels.py';

A.TRITON = {
rmsnorm: {
  fns: ['_rmsnorm', 'rmsnorm'],
  intro: 'Gemma 3 runs this kernel once per forward pass, before layer 0 (n = 3840, ε = 1e-6); everything after that goes through add_rmsnorm. The Triton version is three lines: load the row, reduce it, store it.',
  facts: [['(T,)', 'grid: one program per row'], ['8 warps', '256 threads'], ['4096', 'BLOCK: the whole row, masked'], ['40', 'registers'], ['32 B', 'shared memory'], ['8 + 1', 'shfl.sync.bfly + bar.sync']],
  src: `A|@triton.jit
A|def _rmsnorm(x, w, y, n, eps, BLOCK: tl.constexpr):
A|    row = tl.program_id(0).to(tl.int64)            # one program per row (token)
A|    i = tl.arange(0, BLOCK)                        # BLOCK = 4096 >= n: the whole row at once
A|    m = i < n
S|    v = tl.load(x + row * n + i, mask=m, other=0.0)
S,B,R,C|    s = 1.0 / tl.sqrt_rn(tl.sum(v * v, 0) / n + eps)
O|    tl.store(y + row * n + i, v * s * tl.load(w + i, mask=m), mask=m)
A|
A|def rmsnorm(x, w, y, n, eps, T):
A|    _rmsnorm[(T,)](x, w, y, n, eps, BLOCK=triton.next_power_of_2(n), num_warps=8)`,
  notes: [
    ['No strided loop', 'The program sees the row as one 4096-wide tensor. The compiler splits it over 256 threads, 16 elements each, loaded as four <code>ld.global.v4</code> of x and four of w. The 256 masked elements past n = 3840 play the part of the hatched cells above. x is read once and stays in registers, so the second read of x in the PTX disappears.'],
    ['<code>tl.sum</code> writes the reduction', 'The compiled kernel has the same shape as the hand-written one: five butterfly rounds inside each warp, the 8 warp totals through 32 bytes of shared memory and one <code>bar.sync</code>, then three more rounds over the 8 partials. Only 8 warps take part instead of 32, because the block is 256 threads, not 1024.'],
    ['Close, not bit-identical', 'Both divisions (<code>/ n</code> and <code>1.0 / …</code>) compile to <code>div.full.f32</code>, Triton\'s default division (up to 2 ulp), where the PTX uses <code>div.rn</code> and <code>rcp.rn</code>. The squares are also added in a different order. The scale can differ in its last bits, which is why the port is checked end to end: same greedy text, logits within KL 2.8e-6.'],
  ],
},

add_rmsnorm: {
  fns: ['_add_rmsnorm', 'add_rmsnorm'],
  intro: 'The same fusion as the PTX, written as two reductions over a row held in registers.',
  facts: [['(T,)', 'grid: one program per row'], ['8 warps', '256 threads'], ['4096', 'BLOCK (n = 3840)'], ['60', 'registers'], ['32 B', 'shared memory, both reductions'], ['16 + 3', 'shfl.sync.bfly + bar.sync']],
  src: `R|@triton.jit
R|def _add_rmsnorm(y, w1, x, w2, xn, n, eps, BLOCK: tl.constexpr):
R|    row = tl.program_id(0).to(tl.int64)
R|    i = tl.arange(0, BLOCK)                        # the whole row; 16 values per thread
R|    m = i < n
P1|    a = tl.load(y + row * n + i, mask=m, other=0.0)
P1,B1,S1|    s1 = 1.0 / tl.sqrt_rn(tl.sum(a * a, 0) / n + eps)
P2|    v = tl.load(x + row * n + i, mask=m, other=0.0) + a * s1 * tl.load(w1 + i, mask=m)
P2|    tl.store(x + row * n + i, v, mask=m)           # post-norm + residual
P2,B2,S2|    s2 = 1.0 / tl.sqrt_rn(tl.sum(v * v, 0) / n + eps)
P3|    tl.store(xn + row * n + i, v * s2 * tl.load(w2 + i, mask=m), mask=m)   # v is still in registers
R|
R|def add_rmsnorm(y, w1, x, w2, xn, n, eps, T):
R|    _add_rmsnorm[(T,)](y, w1, x, w2, xn, n, eps, BLOCK=triton.next_power_of_2(n), num_warps=8)`,
  notes: [
    ['Pass 3 reads registers, not memory', 'The PTX re-reads the x values each thread wrote in pass 2. Here the new row <code>v</code> never leaves registers between the store and the second norm, so per token the kernel reads y, x, w1 and w2 once and writes x and xn once: 6 × 15 KB.'],
    ['One scratch buffer, three barriers', 'Each <code>tl.sum</code> becomes 5 + 3 butterfly rounds with the 8 warp totals in shared memory. The compiler reuses the same 32 bytes for both reductions, so the kernel has three <code>bar.sync</code> where the PTX, with separate <code>arn_red1</code> and <code>arn_red2</code>, has two.'],
    ['Same numbers, different last bits', 'As in rmsnorm, the divisions are <code>div.full.f32</code> and the summation order differs, so the scales agree to a few ulp rather than bit for bit.'],
  ],
},

quant_q8: {
  fns: ['_quant_q8', 'quant_q8'],
  intro: 'One program quantizes 32 Q8 blocks at once as a [32 blocks × 32 values] tile. The IEEE divisions are requested explicitly, so the result matches the PTX bit for bit.',
  facts: [['⌈nblk / 32⌉', 'grid, no grid-stride loop'], ['4 warps', '128 threads, 8 values each'], ['29', 'registers'], ['2', 'shfl.sync.bfly rounds per max'], ['div.rn ×2', 'd and 1/d'], ['cvt.rni', 'round, ties to even']],
  src: `I|@triton.jit(do_not_specialize=["nblk"])
I|def _quant_q8(x, q, d, nblk, NB: tl.constexpr, FAKE: tl.constexpr):
I|    b = tl.program_id(0) * NB + tl.arange(0, NB)        # NB = 32 Q8 blocks per program
I|    j = tl.arange(0, 32)
I|    off = b[:, None].to(tl.int64) * 32 + j[None, :]     # a [32 blocks, 32 values] tile
I|    m = (b < nblk)[:, None]
L|    v = tl.load(x + off, mask=m, other=0.0)
A,B,D|    dd = tl.math.div_rn(tl.max(tl.abs(v), 1), 127.0)   # max |x| along each block, / 127
D|    inv = tl.where(dd > 0, tl.math.div_rn(tl.full(dd.shape, 1.0, tl.float32), dd), 0.0)
Q|    r = libdevice.rint(v * inv[:, None])
S|    if FAKE:                                            # prefill: x <- q * d in place
S|        tl.store(x + off, r * dd[:, None], mask=m)
S|    else:
S|        tl.store(q + off, r.to(tl.int8), mask=m)
S|        tl.store(d + b, dd, mask=b < nblk)
N|
N|def quant_q8(x, xq, xd, n):
N|    nblk = n // 32
N|    _quant_q8[(triton.cdiv(nblk, 32),)](x, xq if xq is not None else x, xd if xd is not None else x,
N|                                        nblk, NB=32, FAKE=xq is None, num_warps=4)`,
  notes: [
    ['A block spans 4 lanes, not 32', 'The compiler gives each of the 128 threads 8 consecutive values (two <code>ld.global.v4</code>), so one Q8 block lives in 4 neighboring lanes. <code>tl.max</code> takes the max of 8 in registers, then two butterfly rounds (xor 2, xor 1) instead of five. The only shared memory, 128 bytes behind one barrier, rearranges the 32 scales for their store.'],
    ['Bit for bit with the PTX', '<code>tl.math.div_rn</code> forces IEEE division where plain <code>/</code> would give <code>div.full.f32</code>; 1 / d is correctly rounded either way (<code>rcp.rn</code> in the PTX). <code>libdevice.rint</code> compiles to <code>cvt.rni</code>, round to nearest even, as <code>cvt.rni.s32.f32</code> does. Max is exact in any order. So xq and xd match the hand-written kernel exactly.'],
    ['No grid-stride loop', 'The grid simply has one program per 32 blocks: 4 programs for a 3840-wide token, 15 for the 15,360-wide feed-forward hidden state. Decode launches are replayed from a CUDA graph, so small grids cost little.'],
  ],
},

mv_q8: {
  fns: ['_mv_q8', 'mv_q8'],
  intro: 'Decode-time matrix × vector, written as a [4 rows × 1024 columns] tile per step instead of one warp per row.',
  facts: [['⌈rows / 4⌉', 'grid: 4 rows per program'], ['4 warps', '128 threads'], ['1024', 'BK: columns per step (masked)'], ['56', 'registers'], ['4 KB', 'shared memory'], ['tl.sum', 'per step, 2 axes']],
  src: `A|@triton.jit
A|def _mv_q8(wq, wd, x, bias, resid, y, rows, cols, HAS_BIAS: tl.constexpr, HAS_RESID: tl.constexpr,
A|           BR: tl.constexpr, BK: tl.constexpr):
A|    NB: tl.constexpr = BK // 32                       # 32 Q8 blocks per step
A|    r = tl.program_id(0) * BR + tl.arange(0, BR)      # BR = 4 rows per program
A|    rm = r < rows
A|    r64 = r.to(tl.int64)
A|    k = tl.arange(0, BK)                              # BK = 1024 columns per step
A|    kb = tl.arange(0, NB)
A|    acc = tl.zeros([BR], tl.float32)
L|    for k0 in range(0, cols, BK):                     # 3840 = 3 full steps + a masked one
L|        km = k0 + k < cols
L|        q = tl.load(wq + r64[:, None] * cols + k0 + k[None, :], mask=rm[:, None] & km[None, :], other=0)
L|        xv = tl.load(x + k0 + k, mask=km, other=0.0)
D|        p = tl.reshape(q.to(tl.float32) * xv[None, :], [BR, NB, 32])
D|        bm = rm[:, None] & (k0 + kb * 32 < cols)[None, :]
D|        dd = tl.load(wd + r64[:, None] * (cols // 32) + k0 // 32 + kb[None, :], mask=bm, other=0.0)
D,B|        acc += tl.sum(tl.sum(p, 2) * dd.to(tl.float32), 1)   # block dot, then x scale
E|    if HAS_BIAS:
E|        acc += tl.load(bias + r, mask=rm, other=0.0)
E|    if HAS_RESID:
E|        acc += tl.load(resid + r, mask=rm, other=0.0)
E|    tl.store(y + r, acc, mask=rm)
E|
E|def mv_q8(w, x, bias, resid, y):
E|    _mv_q8[(triton.cdiv(w.rows, 4),)](w.q, w.d, x, ..., w.rows, w.cols, BR=4, BK=1024, num_warps=4)`,
  notes: [
    ['Same order of operations, different owner', 'Like the PTX, each 32-value block\'s products are summed before its f16 scale is applied (<code>tl.sum(p, 2) * dd</code>). The difference is who holds what: four rows share one program and every step covers 1024 columns of all four, where the PTX gives each row a warp and each lane three blocks of it.'],
    ['A reduction every step', 'The row sum <code>tl.sum(…, 1)</code> sits inside the loop, so the compiled kernel reduces across threads (shuffles plus 4 KB of shared memory and barriers) on every 1024-column step. The PTX keeps one partial per lane and reduces once at the end. At 3840 columns that is four reductions instead of one, still small next to reading the weights.'],
    ['Tile size measured, not derived', 'BR = 4, BK = 1024 came out best on the 4070 SUPER. Benchmarks use two copies of the weights, because a single 15 MB matrix fits in the 48 MB L2 cache and would report a bandwidth DRAM cannot give.'],
  ],
},

mv_q5: {
  fns: ['_mv_q5', 'mv_q5'],
  intro: 'The Q5_0 × Q8 product in Triton. The bit tricks of the PTX (bfe, the 0x00204081 multiply, dp4a) become plain shifts and masks on int32 tensors.',
  facts: [['⌈rows / 4⌉', 'grid: 4 rows per program'], ['4 warps', '128 threads'], ['1024', 'BK: columns per step'], ['115', 'registers'], ['0', 'dp4a: int32 mul / mad instead'], ['2 KB', 'shared memory']],
  src: `I|@triton.jit
I|def _mv_q5(wqs, wqh, wd, xq, xd, y, rows, cols, BR: tl.constexpr, BK: tl.constexpr):
I|    NB: tl.constexpr = BK // 32
I|    r = tl.program_id(0) * BR + tl.arange(0, BR)          # 4 rows per program
I|    rm = r < rows
I|    r64 = r.to(tl.int64)
I|    j = tl.arange(0, 16)
I|    kb = tl.arange(0, NB)
I|    nbc = cols // 32
X|    acc = tl.zeros([BR], tl.float32)
X|    for k0 in range(0, cols, BK):
L|        bk = k0 + kb * 32 < cols                         # blocks inside the row
L|        bm = rm[:, None] & bk[None, :]
L|        b = tl.load(wqs + r64[:, None] * (cols // 2) + k0 // 2 + tl.arange(0, BK // 2)[None, :],
L|                    mask=rm[:, None] & (k0 + 2 * tl.arange(0, BK // 2) < cols)[None, :], other=0)
N|        b = tl.reshape(b.to(tl.int32), [BR, NB, 16])       # 16 nibble bytes per block
L|        qh = tl.load(wqh + r64[:, None] * nbc + k0 // 32 + kb[None, :], mask=bm, other=0)
H|        qh = qh[:, :, None]
N,H,M|        lo = ((b & 15) | (((qh >> j[None, None, :]) & 1) << 4)) - 16          # values 0..15
N,H,M|        hi = ((b >> 4) | (((qh >> (j[None, None, :] + 16)) & 1) << 4)) - 16   # values 16..31
L|        xo = k0 + kb[:, None] * 32 + j[None, :]
L|        xl = tl.load(xq + xo, mask=bk[:, None], other=0).to(tl.int32)
L|        xh = tl.load(xq + xo + 16, mask=bk[:, None], other=0).to(tl.int32)
D|        isum = tl.sum(lo * xl[None, :, :] + hi * xh[None, :, :], 2)            # exact int32
X|        dd = tl.load(wd + r64[:, None] * nbc + k0 // 32 + kb[None, :], mask=bm, other=0.0)
X|        xs = tl.load(xd + k0 // 32 + kb, mask=bk, other=0.0)
X,R|        acc += tl.sum(isum.to(tl.float32) * dd.to(tl.float32) * xs[None, :], 1)
R|    tl.store(y + r, acc, mask=rm)`,
  notes: [
    ['−16 per value, not −16 · Σx', 'The PTX multiplies unsigned q (0…31) with dp4a and subtracts 16 · Σx afterwards. Here each value is rebuilt as q − 16 before the multiply. Both integer sums are exact, so both kernels get the same int32 per block; the float steps after it (× d · xd, then the row sum) are where their last bits can part.'],
    ['No dp4a', 'Triton has no 4-way byte dot product in its language, and the compiler does not find one in <code>lo * xl + hi * xh</code>: the PTX it emits has 16 <code>mul.lo.s32</code> and 16 <code>mad.lo.s32</code> in its loop body and no <code>dp4a</code>. Decode is bound by reading weights, so the extra integer work is mostly hidden.'],
    ['The prompt version uses tensor cores', 'For prefill, <code>_gemm_q5</code> rebuilds the same int8 (q − 16) tile and multiplies it with the int8 activations by <code>tl.dot(…, out_dtype=tl.int32)</code>, which compiles to <code>mma.sync.m16n8k32.s8.s8.s32</code>. It is exact, like dp4a, and is most of why the Triton engine reads prompts at 998 tok/s against the PTX engine\'s 367. See the Gemma 3 model page.'],
  ],
},

gemm_q8: {
  fns: ['_gemm_q8', 'gemm_q8'],
  intro: 'In Gemma 3 this kernel runs the prefill q, k, v and output projections. The Triton tile is close to the PTX one, 32 tokens × 64 rows × 64 columns per step, but the product runs on tensor cores and the compiler does the shared-memory staging that the PTX writes by hand.',
  facts: [['(rows / 64, ⌈T / 32⌉)', 'grid (BT = 16 when T ≤ 16)'], ['4 warps', '128 threads'], ['198', 'registers'], ['36 KB', 'shared memory, 2 stages'], ['cp.async', 'tile loads'], ['mma tf32 ×3', 'm16n8k8, ~f32 accuracy']],
  src: `W,X,F,E|@triton.jit(do_not_specialize=["T"])
W,X,F,E|def _gemm_q8(wq, wd, X, bias, resid, Y, rows, cols, T, ldy, HAS_BIAS: tl.constexpr,
W,X,F,E|             HAS_RESID: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
W|    NB: tl.constexpr = BK // 32
W|    r = tl.program_id(0) * BR + tl.arange(0, BR)       # BR = 64 rows
X|    t = tl.program_id(1) * BT + tl.arange(0, BT)       # BT = 32 tokens
W,X|    rm, tm = r < rows, t < T
W,X|    r64, t64 = r.to(tl.int64), t.to(tl.int64)
W,X|    k = tl.arange(0, BK)                               # BK = 64 columns per step
F|    acc = tl.zeros([BT, BR], tl.float32)
1,2|    for k0 in range(0, cols, BK):                      # staging and barriers: the compiler's
X|        xv = tl.load(X + t64[:, None] * cols + k0 + k[None, :], mask=tm[:, None], other=0.0)
W|        q = tl.load(wq + r64[:, None] * cols + k0 + k[None, :], mask=rm[:, None], other=0)
W|        dd = tl.load(wd + r64[:, None] * (cols // 32) + k0 // 32 + tl.arange(0, NB)[None, :],
W|                     mask=rm[:, None], other=0.0)
W|        w = tl.reshape(tl.reshape(q.to(tl.float32), [BR, NB, 32]) * dd.to(tl.float32)[:, :, None], [BR, BK])
F|        acc = tl.dot(xv, tl.trans(w), acc, input_precision="tf32x3")
E|    out = t64[:, None] * ldy + r[None, :]
E|    m = tm[:, None] & rm[None, :]
E|    if HAS_BIAS:
E|        acc += tl.load(bias + r, mask=rm, other=0.0)[None, :]
E|    if HAS_RESID:
E|        acc += tl.load(resid + out, mask=m, other=0.0)
E|    tl.store(Y + out, acc, mask=m)`,
  notes: [
    ['The weights are dequantized before the product', 'The PTX multiplies int8 weights by f32 activations in FMAs and applies each block\'s scale at the end of the block. Triton\'s <code>tl.dot</code> wants one element type, so each 64 × 64 weight tile is turned into f32 (q · d) in registers first, then multiplied.'],
    ['tf32x3: tensor cores at f32 accuracy', 'A plain tf32 product keeps 10 mantissa bits. <code>input_precision="tf32x3"</code> splits each operand into a high and a low tf32 part and adds three products (hi·hi + hi·lo + lo·hi), which lands within about 1e-6 of f32. That costs three <code>mma.sync.m16n8k8.tf32</code> per product, which is why this kernel, unlike gemm_q5, is compute-bound and uses 16-token tiles for short chunks to get more programs.'],
    ['What the compiler writes for you', 'The page above shows the PTX staging both tiles through padded shared memory with two barriers per step. In Triton those steps are implied: with <code>num_stages=2</code> the compiler issues <code>cp.async</code> loads into 36 KB of double-buffered shared memory, reads tensor-core fragments with <code>ldmatrix</code>, and places the barriers itself.'],
  ],
},

qk_norm_rope: {
  fns: ['_qk_norm_rope', 'qk_norm_rope'],
  intro: 'One program per (head, token) as in the PTX, but with 4 warps and one pair per thread instead of one warp with 4 pairs per lane. The position is read from device memory so a decode step can be replayed as a CUDA graph.',
  facts: [['(32 heads, T)', 'grid: 16 q + 8 k + 8 v'], ['4 warps', '128 threads = 128 pairs'], ['23', 'registers'], ['16 B', 'shared memory'], ['7 + 7', 'shfl rounds, two tl.sum'], ['pos_ptr', 'position on the GPU']],
  src: `A|@triton.jit
A|def _qk_norm_rope(qkv, qw, kw, cs, kc, vc, pos_ptr, slots, nh, nkv, eps, D: tl.constexpr):
A|    h = tl.program_id(0)                           # q heads, then k heads, then v heads
A|    t = tl.program_id(1)
A|    H: tl.constexpr = D // 2
A|    i = tl.arange(0, H)                            # pair i: dims i and i + 128
A|    pos = tl.load(pos_ptr) + t                     # read on the GPU (CUDA-graph replay)
A|    base = qkv + t.to(tl.int64) * (nh + 2 * nkv) * D + h * D
L,V|    a = tl.load(base + i)
L,V|    b = tl.load(base + H + i)
A,V,K|    slot = (pos % slots).to(tl.int64) * nkv * D
A|    if h < nh + nkv:
Q,B,N|        s = 1.0 / tl.sqrt_rn((tl.sum(a * a, 0) + tl.sum(b * b, 0)) / D + eps)
N|        if h < nh:
N|            wa, wb = tl.load(qw + i), tl.load(qw + H + i)
N|        else:
N|            wa, wb = tl.load(kw + i), tl.load(kw + H + i)
N|        a = a * s * wa
N|        b = b * s * wb
R|        c = tl.load(cs + pos.to(tl.int64) * D + 2 * i)          # cos
R|        sn = tl.load(cs + pos.to(tl.int64) * D + 2 * i + 1)     # sin
R|        ra = a * c - b * sn
R|        rb = a * sn + b * c
W|        tl.store(base + i, ra)
W|        tl.store(base + H + i, rb)
K|        if h >= nh:                                  # key head: also into the f16 cache
K|            kd = kc + slot + (h - nh) * D
K|            tl.store(kd + i, ra.to(tl.float16))
K|            tl.store(kd + H + i, rb.to(tl.float16))
V|    else:                                            # value head: f16 copy only
V|        vd = vc + slot + (h - nh - nkv) * D
V|        tl.store(vd + i, a.to(tl.float16))
V|        tl.store(vd + H + i, b.to(tl.float16))`,
  notes: [
    ['Pair i is the natural index', 'The PTX gives lane l dims 4l…4l+3 and 128+4l…128+4l+3 so that one warp covers a head with float4 loads. In Triton the program just names the pairs: <code>a</code> = dims 0…127, <code>b</code> = dims 128…255, element i of each forms a NeoX pair. The compiler places one pair per thread over 4 warps.'],
    ['Four warps need shared memory', 'With 128 threads across 4 warps, each <code>tl.sum</code> is 5 butterfly rounds plus 2 more over the 4 warp totals, which go through 16 bytes of shared memory. Two sums (one for <code>a</code>, one for <code>b</code>) give 14 shuffles and 3 barriers; the PTX\'s single warp needs 5 shuffles and none.'],
    ['The cos/sin loads are strided', 'The table stores {cos, sin} pairs, so <code>2 * i</code> and <code>2 * i + 1</code> are two loads with stride 8 bytes. The PTX reads the same bytes as two float4. The table row is 1 KB per position and stays in cache, so this costs little.'],
  ],
},

attention: {
  fns: ['_attention', '_attention_merge', 'attention'],
  intro: 'This is where the port differs most. Instead of one block per (q head, token) with 8 warps each walking keys, a program takes a kv head and both q heads that share it, and walks 32-key blocks with tensor-core products (flash attention). When decoding, the keys are split over 32 programs and merged by a second kernel (flash decoding), the job of the shared-memory merge on this page.',
  facts: [['(8 kv heads, 1, 32)', 'decode grid: 32 key splits'], ['(8, ⌈T / 16⌉, 1)', 'prefill grid: 16 tokens × 2 heads'], ['32', 'BN: keys per block'], ['255', 'registers'], ['48 / 80 KB', 'shared memory, decode / prefill'], ['mma tf32 ×3', 'q·kᵀ and p·v']],
  src: `A|@triton.jit(do_not_specialize=["T"])
A|def _attention(qkv, q_stride, kc, vc, out, part, pml, pos_ptr, T, window, slots, scale,
A|               NH: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
A|               BN: tl.constexpr, SPLIT: tl.constexpr, NS: tl.constexpr):
A|    G: tl.constexpr = NH // NKV                    # 2 q heads per kv head
A|    BM: tl.constexpr = BT * G                      # query rows: (token, q head)
A|    g = tl.program_id(0)                           # kv head
A|    mb = tl.program_id(1)                          # block of BT tokens
A|    sp = tl.program_id(2)                          # key split (decode)
A|    rr = tl.arange(0, BM)
A|    t = mb * BT + rr // G
A|    h = g * G + rr % G
A|    rok = t < T
A|    pos0 = tl.load(pos_ptr)
A|    p = pos0 + t
A|    lo = tl.where(window > 0, tl.maximum(p - window + 1, 0), 0)
A|    d = tl.arange(0, D)
A|    q = tl.load(qkv + t.to(tl.int64)[:, None] * q_stride + h[:, None] * D + d[None, :],
A|                mask=rok[:, None], other=0.0) * (scale * 1.4426950408889634)
A|    t_last = tl.minimum(mb * BT + BT, T) - 1       # keys any row of this program needs
A|    j_lo = pos0 + mb * BT - window + 1
A|    j_lo = tl.where(window > 0, tl.maximum(j_lo, 0), 0) // BN * BN
A|    j_hi = pos0 + t_last + 1
A|    if SPLIT:                                      # NS stretches of whole BN-key blocks
A|        kps = ((j_hi - j_lo + NS - 1) // NS + BN - 1) // BN * BN
A|        j_lo = j_lo + sp * kps
A|        j_hi = tl.minimum(j_hi, j_lo + kps)
A|    m_i = tl.full([BM], float("-inf"), tl.float32)
A|    l_i = tl.zeros([BM], tl.float32)
A|    acc = tl.zeros([BM, D], tl.float32)
A|    n = tl.arange(0, BN)
K|    for j0 in range(j_lo, j_hi, BN):
K|        j = j0 + n
K|        jok = j < j_hi
K|        kv_off = (j % slots).to(tl.int64)[:, None] * (NKV * D) + g * D + d[None, :]
K|        k = tl.load(kc + kv_off, mask=jok[:, None], other=0.0).to(tl.float32)
K|        s = tl.dot(q, tl.trans(k), input_precision="tf32x3")      # [BM, 32] scores
K|        ok = jok[None, :] & (j[None, :] >= lo[:, None]) & (j[None, :] <= p[:, None]) & rok[:, None]
K|        s = tl.where(ok, s, float("-inf"))
U|        m_new = tl.maximum(m_i, tl.max(s, 1))
U|        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
U|        alpha = tl.exp2(m_i - m_use)
U|        e = tl.exp2(s - m_use[:, None])
U|        l_i = l_i * alpha + tl.sum(e, 1)
U|        v = tl.load(vc + kv_off, mask=jok[:, None], other=0.0).to(tl.float32)
U|        acc = tl.dot(e, v, acc * alpha[:, None], input_precision="tf32x3")
U|        m_i = m_new
M|    if SPLIT:                                      # decode: leave (m, l, acc) for the merge
M|        row = (sp * T + t).to(tl.int64) * NH + h
M|        tl.store(part + row[:, None] * D + d[None, :], acc, mask=rok[:, None])
M|        tl.store(pml + 2 * row, m_i, mask=rok)
M|        tl.store(pml + 2 * row + 1, l_i, mask=rok)
O|    else:                                          # prefill: done
O|        o = acc / l_i[:, None]
O|        tl.store(out + t.to(tl.int64)[:, None] * (NH * D) + h[:, None] * D + d[None, :], o,
O|                 mask=rok[:, None])
C|
C|@triton.jit(do_not_specialize=["T"])
C|def _attention_merge(part, pml, out, T, NH: tl.constexpr, D: tl.constexpr, NS: tl.constexpr):
C|    h = tl.program_id(0)                           # one program per (q head, token)
C|    t = tl.program_id(1)
C|    s = tl.arange(0, NS)
C|    sm = s < NS
C|    row = (s * T + t).to(tl.int64) * NH + h
C|    m = tl.load(pml + 2 * row, mask=sm, other=float("-inf"))
C|    l = tl.load(pml + 2 * row + 1, mask=sm, other=0.0)
C|    M = tl.max(m, 0)
C|    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - M))   # empty splits weigh 0
O|    d = tl.arange(0, D)
O|    a = tl.load(part + row[:, None] * D + d[None, :], mask=sm[:, None], other=0.0)
O|    o = tl.sum(a * w[:, None], 0) / tl.sum(l * w, 0)
O|    tl.store(out + (t * NH + h).to(tl.int64) * D + d, o)`,
  notes: [
    ['Both q heads of a kv head in one tile', 'Grouped-query attention lets the 2 q heads that share a kv head read each key once. A program\'s query rows are (token, q head) pairs, so <code>tl.dot(q, kᵀ)</code> scores all of them against a 32-key block at once. In prefill a program holds 16 tokens × 2 heads = 32 rows; in decode one token gives 2 real rows, padded to 16 because <code>tl.dot</code> needs at least 16.'],
    ['Flash decoding replaces the warp merge', 'The PTX splits one query\'s keys over 8 warps and merges their (m, l, acc) through shared memory. Here the split is across 32 programs, which matters when one token must fill the GPU: 8 kv heads alone would be 8 programs. Each leaves (m, l, acc) in global scratch, and <code>_attention_merge</code> rescales them by 2^(m − M) and divides, the same arithmetic as the merge step above.'],
    ['Key blocks aligned to absolute positions', 'Blocks start at multiples of 32 counted from position 0, and a block masked out for a row leaves its (m, l, acc) untouched. A row\'s arithmetic therefore does not depend on which other rows share its program, so prefill gives a token the same bits whatever chunk it lands in; that is what makes KV-cache reuse bit-exact (<code>gemma3.py cache-test</code>).'],
    ['Shared memory sets the pipeline depth', 'With 256-dim heads and tf32x3, the decode kernel needs 48 KB of shared memory at one stage. Two stages would gain at most 5% and exceed the shared-memory limit under Triton 3.8, so it runs with <code>num_stages=1</code>.'],
  ],
},

geglu: {
  fns: ['_geglu', 'geglu'],
  intro: 'Element-wise, so the port is almost a transcription. The flat index and the gate/up offsets are the same; the grid has one program per 1024 outputs instead of a grid-stride loop.',
  facts: [['⌈n · T / 1024⌉', 'grid: 15 programs per token'], ['4 warps', '128 threads, 8 outputs each'], ['40', 'registers'], ['0', 'shared memory'], ['ex2.approx', 'exp2'], ['div.full', 'the tanh division']],
  src: `I|@triton.jit(do_not_specialize=["total"])
I|def _geglu(gu, out, n, total, BLOCK: tl.constexpr):
I|    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)   # 1024 outputs
I|    m = i < total
I|    t = i // n                                       # token
I|    j = i - t * n
L|    g = tl.load(gu + t * 2 * n + j, mask=m, other=0.0)
L|    u = tl.load(gu + t * 2 * n + n + j, mask=m, other=0.0)
P|    y = 0.7978845608028654 * g * (1.0 + 0.044715 * g * g)
T|    th = 1.0 - 2.0 / (tl.exp2(2.0 * 1.4426950408889634 * tl.abs(y)) + 1.0)
G|    th = tl.where(y < 0, -th, th)
G,S|    tl.store(out + i, 0.5 * g * (1.0 + th) * u, mask=m)
S|
S|def geglu(gu, out, n, T):
S|    _geglu[(triton.cdiv(n * T, 1024),)](gu, out, n, n * T, BLOCK=1024, num_warps=4)`,
  notes: [
    ['Same formula, same instructions, one exception', 'tanh comes from one <code>ex2.approx</code> as in the PTX, and the sign is put back with a select instead of <code>copysign</code> (equal except at y = −0, where both give 0). The division <code>2.0 / (…)</code> compiles to <code>div.full.f32</code>; the PTX uses <code>div.rn.f32</code>. Wrapping it in <code>tl.math.div_rn</code> would match it bit for bit, as quant_q8 does.'],
    ['<code>//</code> on a tensor', '<code>t = i // n</code> is an integer division per element, the <code>div.u32</code> of the PTX. The 64-bit index (<code>.to(tl.int64)</code>) keeps the offsets safe for long prompts: the input offset t · 2n passes 2³¹ at about 70,000 tokens.'],
  ],
},

argmax: {
  fns: ['_argmax', 'argmax'],
  intro: 'One program, as in the PTX, but it walks the logits in chunks of 4096 and reduces each chunk to (max, smallest index of the max) with two reductions.',
  facts: [['(1,)', 'grid: one program'], ['16 warps', '512 threads'], ['4096', 'BLOCK: logits per iteration'], ['65', 'iterations for 262,208 logits'], ['29', 'registers'], ['13 + 4', 'shfl.sync.bfly + bar.sync']],
  src: `A|@triton.jit
A|def _argmax(x, n, out, BLOCK: tl.constexpr):
A|    i = tl.arange(0, BLOCK)
S|    best_v = tl.load(x)
S|    best_i = 0
S|    for i0 in range(0, n, BLOCK):
S|        v = tl.load(x + i0 + i, mask=i0 + i < n, other=float("-inf"))
B,R,F|        m = tl.max(v, 0)                                  # chunk max
B,R,F|        k = tl.min(tl.where(v == m, i0 + i, n), 0)        # its smallest index
S|        take = m > best_v                                 # strictly greater: keep the first
S|        best_i = tl.where(take, k, best_i)
S|        best_v = tl.where(take, m, best_v)
O|    # ties: k is the smallest index within a block, and blocks go in order
O|    # with a strict >, so an earlier block's equal maximum is kept
O|    tl.store(out, best_i)
O|
O|def argmax(x, n, out):
O|    _argmax[(1,)](x, n, out, BLOCK=4096, num_warps=16)`,
  notes: [
    ['Two plain reductions instead of a pair reduction', 'The PTX shuffles (value, index) pairs and resolves ties inside its <code>pick</code> step. Triton\'s <code>tl.max</code> and <code>tl.min</code> work on one tensor at a time, so each chunk is reduced twice: the max, then the smallest index whose value equals it. Each one compiles to a warp butterfly plus a pass over the 16 warp results in shared memory.'],
    ['The same answer', 'Ties go to the smaller index, as in the PTX: within a chunk the second reduction picks the smallest index of the max, and chunks are visited in order, replacing the best only when strictly greater, so an equal maximum in a later chunk never wins. The planted tie in the demo above resolves to the same index in both.'],
    ['One pass over the logits', 'An earlier version followed the loop with a second pass that took the minimum index among all values equal to the final maximum. It could not change the result, since the first pass already keeps the first maximum, and it read the 1 MB of logits a second time. On the 4070 SUPER, removing it took the launch from about 72 µs to 40 µs: with one program, the kernel runs on a single SM and is bound by how fast that SM can pull in the logits.'],
  ],
},

compact_logits: {
  fns: ['_compact_logits', 'compact_logits'],
  intro: 'A direct port: compare each logit with the threshold and append survivors through one atomic counter. The order of the list is arbitrary in both, and the host sorts it.',
  facts: [['⌈n / 1024⌉', 'grid: 257 programs'], ['4 warps', '128 threads, 8 logits each'], ['40', 'registers'], ['8', 'atom.global.add per thread'], ['2 KB', 'shared memory'], ['no laps', 'one program per 1024']],
  src: `A|@triton.jit
A|def _compact_logits(x, n, best, margin, out, cap, BLOCK: tl.constexpr):
L|    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)     # 1024 logits per program
L|    v = tl.load(x + i, mask=i < n, other=float("-inf"))
T,L|    keep = (i < n) & (v >= tl.load(x + tl.load(best)) - margin)
A|    slot = tl.atomic_add(out + i * 0, 1, mask=keep)          # every element points at out[0]
A|    st = keep & (slot < cap)
A|    tl.store(out + 2 + 2 * slot, i, mask=st)
A|    tl.store(out + 3 + 2 * slot, v.to(tl.int32, bitcast=True), mask=st)
H|
H|def compact_logits(x, n, best, margin, out, cap):
H|    out[:1].zero_()
H|    _compact_logits[(triton.cdiv(n, 1024),)](x, n, best, margin, out, cap, BLOCK=1024, num_warps=4)`,
  notes: [
    ['<code>out + i * 0</code>', 'A Triton atomic acts on a tensor of pointers, one per element. Adding <code>i * 0</code> broadcasts the address of the counter to all 1024 elements, so each element that passes the mask gets its own old value, its slot, as each thread\'s <code>atomicAdd</code> does in the PTX. The compiled kernel has 8 predicated <code>atom.global.add</code> per thread, one per element it holds.'],
    ['Survivors are rare, so atomics are cheap', 'Only logits within the margin of the maximum pass, typically hundreds of 262,208, so almost every predicated atomic is skipped. Neither version aggregates per warp first; neither needs to.'],
    ['Sampling without the host', 'When neither top-k nor top-p is set, the Triton engine skips this kernel and draws on the GPU with <code>_sample_logits</code> (an inverse CDF over the whole vocabulary with <code>tl.cumsum</code>), so a sampled decode step still fits in one CUDA graph.'],
  ],
},
};


// Triton kernels without a walkthrough page, listed on the Gemma 3 model page (tag-free listings)
const raw = src => src.split('\n').map(l => ' |' + l).join('\n');
A.TRITON_EXTRA = {
  gemm_q5: { label: '_gemm_q5', lang: 'py', src: raw(`@triton.jit(do_not_specialize=["T"])
def _gemm_q5(wqs, wqh, wd, XQ, XD, bias, resid, Y, rows, cols, T, ldy, HAS_BIAS: tl.constexpr,
             HAS_RESID: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)           # 64 weight rows
    t = tl.program_id(1) * BT + tl.arange(0, BT)           # up to 128 tokens
    rm, tm = r < rows, t < T
    r64, t64 = r.to(tl.int64), t.to(tl.int64)
    j = tl.arange(0, 16)
    j32 = tl.arange(0, 32)
    nbc = cols // 32
    acc = tl.zeros([BT, BR], tl.float32)
    for kb in range(0, nbc):                               # one 32-value block per step
        xq = tl.load(XQ + t64[:, None] * cols + kb * 32 + j32[None, :], mask=tm[:, None], other=0)
        xd = tl.load(XD + t64 * nbc + kb, mask=tm, other=0.0)
        b = tl.load(wqs + r64[:, None] * (cols // 2) + kb * 16 + j[None, :], mask=rm[:, None], other=0)
        b = b.to(tl.int32)
        qh = tl.load(wqh + r64 * nbc + kb, mask=rm, other=0)[:, None]
        lo = ((b & 15) | (((qh >> j[None, :]) & 1) << 4)) - 16
        hi = ((b >> 4) | (((qh >> (j[None, :] + 16)) & 1) << 4)) - 16
        w = tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), [BR, 32]).to(tl.int8)
        isum = tl.dot(xq, tl.trans(w), out_dtype=tl.int32)  # int8 tensor cores, exact
        dd = tl.load(wd + r64 * nbc + kb, mask=rm, other=0.0).to(tl.float32)
        acc += isum.to(tl.float32) * xd[:, None] * dd[None, :]
    out = t64[:, None] * ldy + r[None, :]
    m = tm[:, None] & rm[None, :]
    if HAS_BIAS:
        acc += tl.load(bias + r, mask=rm, other=0.0)[None, :]
    if HAS_RESID:
        acc += tl.load(resid + out, mask=m, other=0.0)
    tl.store(Y + out, acc, mask=m)`) },
  embed: { label: '_embed', lang: 'py', src: raw(`@triton.jit
def _embed(ids, wq, wd, x, cols, scale, BLOCK: tl.constexpr):
    t = tl.program_id(0)                                   # one program per token
    tok = tl.load(ids + t).to(tl.int64)                    # ids live on the GPU (CUDA graph)
    c = tl.arange(0, BLOCK)
    m = c < cols
    q = tl.load(wq + tok * cols + c, mask=m, other=0).to(tl.float32)
    d = tl.load(wd + tok * (cols // 32) + c // 32, mask=m, other=0.0).to(tl.float32)
    tl.store(x + t.to(tl.int64) * cols + c, q * d * scale, mask=m)   # scale = sqrt(3840)`) },
  sample_logits: { label: '_sample_logits', lang: 'py', src: raw(`@triton.jit
def _sample_logits(x, n, best, inv_temp, min_w, u, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    xb = tl.load(x + tl.load(best))                        # the argmax result, already on the GPU
    c = inv_temp * 1.4426950408889634
    z = 0.0
    for i0 in range(0, n, BLOCK):                          # pass 1: Z = sum of the kept weights
        v = tl.load(x + i0 + i, mask=i0 + i < n, other=float("-inf"))
        w = tl.exp2((v - xb) * c)
        z += tl.sum(tl.where(w >= min_w, w, 0.0), 0)
    target = u * z
    run = 0.0
    pick = n
    last = tl.load(best)
    for i0 in range(0, n, BLOCK):                          # pass 2: first i with running sum > u * Z
        v = tl.load(x + i0 + i, mask=i0 + i < n, other=float("-inf"))
        w = tl.exp2((v - xb) * c)
        w = tl.where(w >= min_w, w, 0.0)
        cum = tl.cumsum(w, 0) + run
        pick = tl.minimum(pick, tl.min(tl.where(cum > target, i0 + i, n), 0))
        last = tl.maximum(last, tl.max(tl.where(w > 0, i0 + i, -1), 0))
        run += tl.sum(w, 0)
    pick = tl.where(pick < n, pick, last)                  # u * Z past the rounded total
    tl.store(out, pick)
    tl.store(out + 1, z.to(tl.int32, bitcast=True))`) },
};

A.TRITON_SRC = SRC;

// the section under "Why it is built this way" on a page whose kernel has a Triton port
A.tritonSection = function (id) {
  const T = A.TRITON[id], foot = A.$('foot');
  if (!T || !foot || A.$('s-triton')) return;
  const sec = document.createElement('section');
  sec.setAttribute('aria-labelledby', 's-triton');
  sec.innerHTML = `<div class="sechead"><h2 id="s-triton">The Triton version</h2><p>${T.intro} The listing is in the code panel above, under <b>Triton</b>: ${T.fns.map(f => `<code>${f}</code>`).join(', ')} in <code>${SRC}</code>. Compiler figures are from Triton 3.8 for the RTX 4070 SUPER (sm_89).</p></div>
    <div class="facts">${T.facts.map(([b, s]) => `<div class="fact"><b>${b}</b><span>${s}</span></div>`).join('')}</div>
    <div class="two">${T.notes.map(([h, p]) => `<div class="panel"><h3>${h}</h3><p>${p}</p></div>`).join('')}</div>`;
  foot.parentNode.insertBefore(sec, foot);
};
})();
