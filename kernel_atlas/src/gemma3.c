/*
 * gemma3.c - Gemma 3 (text) inference on the GPU with hand-written PTX
 * kernels (gemma3.ptx) driven through the CUDA Driver API. No other libraries.
 *
 * The model is read from a GGUF file (gguf.c), text is tokenized with the
 * SentencePiece vocabulary stored in it (tokenizer.c), and every layer runs
 * in the PTX kernels; the host only looks up token embeddings and launches.
 *
 * Weights on the GPU (12B: 9.6 GiB with an 8192-token context):
 *   feed-forward gate/up/down     Q5_0 (5 bits + f16 scale per 32 values)
 *   attention q/k/v/o, output     Q8_0 (int8 + f16 scale per 32 values);
 *                                 the output head is the token embedding
 *                                 matrix (tied weights)
 *   norms                         F32 as stored
 *   token embeddings              stay in host memory (one row per token)
 * The weights are quantized at load time from a BF16 GGUF, or read as is
 * from a GGUF that `gemma3_demo quantize` wrote (same formats, loads ~3x
 * faster, and llama.cpp can run that file too, for comparisons).
 *
 * The prompt is processed in chunks of up to -b tokens (by default the
 * largest of 128, 64, 32, ... that fits in GPU memory) with tiled GEMM
 * kernels that read each weight once per chunk; generated tokens then go one
 * at a time through matrix-vector kernels. Keys and values live in an f16
 * KV cache: the global-attention layers (every 6th) keep every position, the
 * sliding-window layers a ring of the last sliding_window + batch positions.
 *
 * Usage (see usage() for every option):
 *   ./gemma3_demo [options] PROMPT        continuation of PROMPT as plain text
 *                                         (after <bos>; control tokens such as
 *                                         "<start_of_turn>" are recognised)
 *   ./gemma3_demo [options] -chat MESSAGE one reply in the chat format
 *   ./gemma3_demo [options] -i            interactive multi-turn chat
 *   ./gemma3_demo quantize [IN] [OUT]     BF16 GGUF -> Q8_0/Q5_0 GGUF
 *   ./gemma3_demo test                    kernel self-tests against the CPU
 *   ./gemma3_demo cache-test              KV-cache reuse is bit-exact (needs the model)
 *   ./gemma3_demo tokenize TEXT           show the tokens of TEXT
 *   ./gemma3_demo tokenize-ids            one line of ids per stdin line
 * Sampling defaults to Gemma's generation config (temperature 1, top-k 64,
 * top-p 0.95); --temp 0 is greedy. A chat keeps its KV cache between turns
 * and recomputes only what changed.
 * Environment (debugging): GEMMA3_DUMP=1 prints a summary of intermediate
 * tensors (named like llama.cpp's llama-eval-callback) and the top logits;
 * GEMMA3_LOGITS=FILE writes the logits after a PROMPT as raw f32;
 * GEMMA3_NO_REUSE=1 recomputes every chat turn from scratch.
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "chat.h"
#include "gguf.h"
#include "quant.h"
#include "sampler.h"
#include "tokenizer.h"

/* Abort with the driver's error name/message if a CUDA Driver API call fails. */
#define CHECK(call)                                                          \
    do {                                                                     \
        CUresult err_ = (call);                                              \
        if (err_ != CUDA_SUCCESS) {                                          \
            const char *name_ = NULL, *msg_ = NULL;                          \
            cuGetErrorName(err_, &name_);                                    \
            cuGetErrorString(err_, &msg_);                                   \
            fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", __FILE__,         \
                    __LINE__, #call, name_ ? name_ : "?", msg_ ? msg_ : ""); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define DEFAULT_MODEL "gemma-3-12b-it-Q5_0.gguf"
#define BF16_MODEL "gemma-3-12b-it-BF16.gguf"
#define DEFAULT_CUBIN "gemma3.cubin"
#define HEAD_DIM 256         /* qk_norm_rope and attention assume 256-dim heads */
#define CAND_CAP 16384       /* sampling candidates gathered on the GPU */
#define AUTO_MAX_BATCH 128   /* default prompt chunk: the largest power of 2 up to */
#define AUTO_SPARE_BYTES (64u << 20) /* this that leaves this much GPU memory free */
#define MAX_BATCH 256

/* Host wall-clock time in milliseconds. */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* xorshift64* - small deterministic RNG for the self-tests */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rng_u32(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}
static float rng_uniform(float lo, float hi) /* [lo, hi) */
{
    return lo + (hi - lo) * (rng_u32() >> 8) * (1.0f / 16777216.0f);
}

/* ------------------------------------------------------------------------ */
/* Kernel handles and launch helpers                                         */

static CUfunction k_rmsnorm, k_add_rmsnorm, k_quant, k_mv_q8, k_mv_q5, k_qk_rope, k_attn, k_geglu,
    k_gemm_q8, k_gemm_q5, k_argmax, k_compact, k_sample;
static unsigned max_blocks; /* cap for grid-stride kernels: 8 blocks per SM */

static void load_kernels(const char *cubin)
{
    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, cubin);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cannot load %s (run `make` in the gemma3 directory)\n", cubin);
        exit(1);
    }
    struct { CUfunction *fn; const char *name; } k[] = {
        {&k_rmsnorm, "rmsnorm"}, {&k_add_rmsnorm, "add_rmsnorm"}, {&k_quant, "quant_q8"},
        {&k_mv_q8, "mv_q8"}, {&k_mv_q5, "mv_q5"}, {&k_qk_rope, "qk_norm_rope"},
        {&k_attn, "attention"}, {&k_geglu, "geglu"}, {&k_gemm_q8, "gemm_q8"},
        {&k_gemm_q5, "gemm_q5"}, {&k_argmax, "argmax"}, {&k_compact, "compact_logits"},
        {&k_sample, "sample_logits"},
    };
    for (size_t i = 0; i < sizeof(k) / sizeof(k[0]); i++)
        CHECK(cuModuleGetFunction(k[i].fn, mod, k[i].name));
    CUdevice dev;
    int sms;
    CHECK(cuCtxGetDevice(&dev));
    CHECK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    max_blocks = 8 * sms;
}

/* 1-D launch of a grid-stride kernel over `total` items. */
static void launch_1d(CUfunction fn, unsigned total, void **args)
{
    const unsigned threads = 256;
    unsigned blocks = (total + threads - 1) / threads;
    if (blocks > max_blocks)
        blocks = max_blocks;
    if (blocks == 0)
        blocks = 1;
    CHECK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL));
}

/* Q8 matrix on the GPU: q int8 [rows][cols], d f16 [rows][cols/32] */
typedef struct {
    CUdeviceptr q, d;
    unsigned rows, cols;
} Q8W;

/* Q5 matrix on the GPU: qs [rows][cols/2], qh u32 [rows][cols/32], d f16 [rows][cols/32] */
typedef struct {
    CUdeviceptr qs, qh, d;
    unsigned rows, cols;
} Q5W;

/* each of the T rows of x [T][n] */
static void rmsnorm(CUdeviceptr x, CUdeviceptr w, CUdeviceptr y, unsigned n, float eps,
                    unsigned T)
{
    void *args[] = {&x, &w, &y, &n, &eps};
    CHECK(cuLaunchKernel(k_rmsnorm, T, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
}

/* per row of T: x += rmsnorm(y) * w1, xn = rmsnorm(x) * w2 */
static void add_rmsnorm(CUdeviceptr y, CUdeviceptr w1, CUdeviceptr x, CUdeviceptr w2,
                        CUdeviceptr xn, unsigned n, float eps, unsigned T)
{
    void *args[] = {&y, &w1, &x, &w2, &xn, &n, &eps};
    CHECK(cuLaunchKernel(k_add_rmsnorm, T, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
}

/* x f32 [n] -> xq int8 [n], xd f32 [n/32]; with xq = 0, x is replaced in
 * place by its Q8-rounded values ("fake quantization") */
static void quant_q8(CUdeviceptr x, CUdeviceptr xq, CUdeviceptr xd, unsigned n)
{
    unsigned nblk = n / 32;
    void *args[] = {&x, &xq, &xd, &nblk};
    launch_1d(k_quant, nblk * 32, args);
}

/* y = W . x + bias + resid (x f32; bias / resid may be 0) */
static void mv_q8(const Q8W *w, CUdeviceptr x, CUdeviceptr bias, CUdeviceptr resid,
                  CUdeviceptr y)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->q, (void *)&w->d, &x, &bias, &resid, &y, &rows, &cols};
    CHECK(cuLaunchKernel(k_mv_q8, (rows + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* y = W . x with x in Q8 (xq, xd from quant_q8) */
static void mv_q5(const Q5W *w, CUdeviceptr xq, CUdeviceptr xd, CUdeviceptr y)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->qs, (void *)&w->qh, (void *)&w->d, &xq, &xd, &y, &rows, &cols};
    CHECK(cuLaunchKernel(k_mv_q5, (rows + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* Y[t] (row stride ldy) = W . X[t] + bias + resid[t] for T tokens */
static void gemm_q8(const Q8W *w, CUdeviceptr X, CUdeviceptr bias, CUdeviceptr resid,
                    CUdeviceptr Y, unsigned ldy, unsigned T)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->q, (void *)&w->d, &X, &bias, &resid, &Y, &rows, &cols, &T, &ldy};
    CHECK(cuLaunchKernel(k_gemm_q8, rows / 64, (T + 31) / 32, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static void gemm_q5(const Q5W *w, CUdeviceptr X, CUdeviceptr bias, CUdeviceptr resid,
                    CUdeviceptr Y, unsigned ldy, unsigned T)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->qs, (void *)&w->qh, (void *)&w->d, &X, &bias, &resid, &Y,
                    &rows, &cols, &T, &ldy};
    CHECK(cuLaunchKernel(k_gemm_q5, rows / 64, (T + 31) / 32, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* q/k head norms and RoPE for tokens at positions pos0 .. pos0+T-1; k and v
 * are cached in slot pos % slots */
static void qk_norm_rope(CUdeviceptr qkv, CUdeviceptr qw, CUdeviceptr kw, CUdeviceptr cs,
                         CUdeviceptr kc, CUdeviceptr vc, unsigned pos0, unsigned slots,
                         unsigned nh, unsigned nkv, float eps, unsigned T)
{
    void *args[] = {&qkv, &qw, &kw, &cs, &kc, &vc, &pos0, &slots, &nh, &nkv, &eps};
    CHECK(cuLaunchKernel(k_qk_rope, nh + 2 * nkv, T, 1, 32, 1, 1, 0, NULL, args, NULL));
}

/* causal attention for T query tokens at pos0..; window = 0 for global layers */
static void attention(CUdeviceptr q, unsigned q_stride, CUdeviceptr kc, CUdeviceptr vc,
                      CUdeviceptr out, unsigned pos0, unsigned window, unsigned slots,
                      unsigned nh, unsigned nkv, float scale, unsigned T)
{
    void *args[] = {&q, &q_stride, &kc, &vc, &out, &pos0, &window, &slots, &nh, &nkv, &scale};
    CHECK(cuLaunchKernel(k_attn, nh, T, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* out[t] = gelu(gu[t][0..n)) * gu[t][n..2n) for T tokens */
static void geglu(CUdeviceptr gu, CUdeviceptr out, unsigned n, unsigned T)
{
    void *args[] = {&gu, &out, &n, &T};
    launch_1d(k_geglu, n * T, args);
}

static void argmax(CUdeviceptr x, unsigned n, CUdeviceptr out)
{
    void *args[] = {&x, &n, &out};
    CHECK(cuLaunchKernel(k_argmax, 1, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
}

/* out = {count, 0, (index, logit) x min(count, cap)} for every logit within
 * margin of x[*best] */
static void compact_logits(CUdeviceptr x, unsigned n, CUdeviceptr best, float margin,
                           CUdeviceptr out, unsigned cap)
{
    CHECK(cuMemsetD32(out, 0, 1));
    void *args[] = {&x, &n, &best, &margin, &out, &cap};
    launch_1d(k_compact, n, args);
}

/* out = {token drawn from softmax(x / T) (weights below min_w dropped),
 * normalizer Z as f32} using the uniform number u */
static void sample_logits(CUdeviceptr x, unsigned n, CUdeviceptr best, float inv_temp,
                          float min_w, float u, CUdeviceptr out)
{
    void *args[] = {&x, &n, &best, &inv_temp, &min_w, &u, &out};
    CHECK(cuLaunchKernel(k_sample, 1, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
}

/* ------------------------------------------------------------------------ */
/* Model                                                                     */

typedef struct {
    unsigned n_layer, n_embd, n_head, n_kv_head, head_dim, n_ff, n_vocab, max_ctx;
    unsigned sliding_window, swa_pattern;
    float eps, rope_base, rope_scale, rope_base_local, attn_scale;
} Config;

typedef struct {
    CUdeviceptr attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffw_norm;
    Q8W wq, wk, wv, wo;
    Q5W gate_up;        /* gate rows then up rows, [2 n_ff][n_embd] */
    Q5W down;
    CUdeviceptr kc, vc; /* f16 [slots][n_kv_head][256] */
    unsigned slots;
    int swa;            /* sliding-window layer */
} Layer;

typedef struct {
    Config c;
    unsigned ctx;            /* max positions */
    unsigned batch;          /* max tokens per forward() call */
    Layer *layer;
    CUdeviceptr out_norm;
    Q8W out;                 /* output head [n_vocab][n_embd] */
    const GgufTensor *embd;  /* token embeddings in the mmap'ed file (Q8_0 or BF16) */
    CUdeviceptr rope_global, rope_local; /* f32 [ctx][head_dim/2][2] */
    /* activations, [batch][...] (decoding uses row 0) */
    CUdeviceptr x, xn, y, qkv, att, gu, h, xq, xd, logits, next;
    CUdeviceptr cand;        /* sampling candidates, see compact_logits */
    CUdeviceptr draw;        /* sample_logits result */
    float *h_x;
    /* what the KV cache holds: tokens of positions 0 .. n_cache-1 */
    int32_t *cache_tok;
    unsigned n_cache;
    /* copy of the sliding-window rings, taken when the cache held snap_pos
     * tokens, so a chat can rewind into a reply that overran the rings */
    void *snap;
    unsigned snap_pos;
    int has_snap;
} Model;

static int has_kv(const Gguf *g, const char *key)
{
    return gguf_kv(g, key) != NULL;
}

static void read_config(const Gguf *g, Config *c)
{
    const GgufKV *arch = gguf_kv(g, "general.architecture");
    uint64_t len;
    const uint8_t *p = arch ? arch->val : NULL;
    if (!arch || strncmp(gguf_str(&p, &len), "gemma3", 6) || len != 6) {
        fprintf(stderr, "model architecture is not gemma3\n");
        exit(1);
    }
    c->n_layer = gguf_u32(g, "gemma3.block_count");
    c->n_embd = gguf_u32(g, "gemma3.embedding_length");
    c->n_head = gguf_u32(g, "gemma3.attention.head_count");
    c->n_kv_head = gguf_u32(g, "gemma3.attention.head_count_kv");
    c->head_dim = gguf_u32(g, "gemma3.attention.key_length");
    c->n_ff = gguf_u32(g, "gemma3.feed_forward_length");
    c->max_ctx = gguf_u32(g, "gemma3.context_length");
    c->sliding_window = gguf_u32(g, "gemma3.attention.sliding_window");
    c->swa_pattern = has_kv(g, "gemma3.attention.sliding_window_pattern")
                         ? gguf_u32(g, "gemma3.attention.sliding_window_pattern") : 6;
    c->eps = gguf_f32(g, "gemma3.attention.layer_norm_rms_epsilon");
    c->rope_base = gguf_f32(g, "gemma3.rope.freq_base");
    /* linear RoPE scaling (factor 8 on 4B+) applies to the global layers only */
    c->rope_scale = has_kv(g, "gemma3.rope.scaling.factor")
                        ? 1.0f / gguf_f32(g, "gemma3.rope.scaling.factor") : 1.0f;
    c->rope_base_local = has_kv(g, "gemma3.rope.freq_base_swa")
                             ? gguf_f32(g, "gemma3.rope.freq_base_swa") : 10000.0f;
    /* query_pre_attn_scalar: head_dim, except n_embd / n_head on the 27B (62 layers) */
    c->attn_scale = c->n_layer == 62 ? 1.0f / sqrtf((float)(c->n_embd / c->n_head))
                                     : 1.0f / sqrtf((float)c->head_dim);
    c->n_vocab = (unsigned)gguf_kv(g, "tokenizer.ggml.tokens")->arr_n;
    if (c->head_dim != HEAD_DIM || gguf_u32(g, "gemma3.attention.value_length") != HEAD_DIM ||
        c->n_head % c->n_kv_head || c->n_embd % 64 || c->n_ff % 64 ||
        c->n_head * c->head_dim % 64 || c->swa_pattern == 0) {
        fprintf(stderr, "unsupported gemma3 hyperparameters\n");
        exit(1);
    }
}

/* llama.cpp: layers il % pattern < pattern - 1 use sliding-window attention */
static int is_swa(const Config *c, unsigned l)
{
    return c->sliding_window && l % c->swa_pattern < c->swa_pattern - 1;
}

/* cos/sin for every position and rotary pair, as ggml's rope cache:
 * theta = pos * freq_scale * base^(-2i/d), the power taken by repeated
 * multiplication. */
static void rope_table(unsigned n_dims, unsigned ctx, float base, float freq_scale, float *cs)
{
    const float theta_scale = powf(base, -2.0f / n_dims);
    for (unsigned p = 0; p < ctx; p++) {
        float theta = (float)p;
        for (unsigned i0 = 0; i0 < n_dims; i0 += 2) {
            float t = freq_scale * theta;
            cs[(size_t)p * n_dims + i0] = cosf(t);
            cs[(size_t)p * n_dims + i0 + 1] = sinf(t);
            theta *= theta_scale;
        }
    }
}

/* --- loading --- */

static void *staging;                    /* pinned host buffer for uploads */
static uint8_t *quant_buf;               /* quantized blocks of a chunk of rows */
static const size_t STAGING_BYTES = 64 << 20;
/* All model memory is carved out of one cuMemAlloc'ed arena: hundreds of
 * separate allocations would waste memory this card can't spare. */
static CUdeviceptr arena;
static size_t arena_size, dev_bytes;     /* dev_bytes = arena bytes in use */

static CUdeviceptr dalloc(size_t n)
{
    size_t off = (dev_bytes + 255) & ~(size_t)255;
    if (off + n > arena_size) {
        fprintf(stderr, "internal error: GPU arena of %zu bytes too small\n", arena_size);
        exit(1);
    }
    dev_bytes = off + n;
    return arena + off;
}

/* The tensor `name` of shape [ne0, ne1] (any type), or exit. */
static const GgufTensor *find_tensor(const Gguf *g, const char *name, uint64_t ne0, uint64_t ne1)
{
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (strcmp(t->name, name))
            continue;
        if (!t->nbytes || t->ne[0] != ne0 || t->ne[1] != ne1) {
            fprintf(stderr, "%s: unexpected type %u / shape [%lu, %lu]\n", name, t->type,
                    (unsigned long)t->ne[0], (unsigned long)t->ne[1]);
            exit(1);
        }
        return t;
    }
    fprintf(stderr, "%s: tensor not found\n", name);
    exit(1);
}

static CUdeviceptr upload_f32(const Gguf *g, const char *name, uint64_t n)
{
    const GgufTensor *t = gguf_tensor(g, name, GGML_F32, n, 1, 0);
    CUdeviceptr p = dalloc(t->nbytes);
    CHECK(cuMemcpyHtoD(p, t->data, t->nbytes));
    return p;
}

/* Rows [r0, r0+nr) of t as `type` blocks: straight from the file when t is
 * stored that way, else quantized into quant_buf. */
static const uint8_t *blocks(const GgufTensor *t, uint32_t type, size_t r0, size_t nr)
{
    size_t row = gguf_type_bytes(type, t->ne[0]);
    if (t->type == type)
        return t->data + r0 * row;
    if (t->type != GGML_BF16 && t->type != GGML_F16 && t->type != GGML_F32) {
        fprintf(stderr, "%s: tensor type %u is not supported\n", t->name, t->type);
        exit(1);
    }
    quantize_rows(type, t, r0, nr, quant_buf);
    return quant_buf;
}

/* matrix [rows][cols] -> Q8 on the GPU, in staging-sized row chunks */
static Q8W upload_q8(const Gguf *g, const char *name, unsigned cols, unsigned rows)
{
    const GgufTensor *t = find_tensor(g, name, cols, rows);
    unsigned nb = cols / 32;
    Q8W w = {dalloc((size_t)rows * cols), dalloc((size_t)rows * nb * 2), rows, cols};
    size_t chunk = STAGING_BYTES / (nb * 34);
    for (size_t r0 = 0; r0 < rows; r0 += chunk) {
        size_t nr = rows - r0 < chunk ? rows - r0 : chunk;
        const uint8_t *src = blocks(t, GGML_Q8_0, r0, nr);
        int8_t *q = staging;
        uint16_t *d = (uint16_t *)(q + nr * cols);
        for (size_t b = 0; b < nr * nb; b++) {
            memcpy(&d[b], src + b * 34, 2);
            memcpy(q + b * 32, src + b * 34 + 2, 32);
        }
        CHECK(cuMemcpyHtoD(w.q + r0 * cols, q, nr * cols));
        CHECK(cuMemcpyHtoD(w.d + r0 * nb * 2, d, nr * nb * 2));
    }
    return w;
}

/* matrix [rows][cols] -> rows row0 .. row0+rows-1 of the Q5 matrix w */
static void upload_q5(const Gguf *g, const char *name, unsigned cols, unsigned rows, Q5W *w,
                      unsigned row0)
{
    const GgufTensor *t = find_tensor(g, name, cols, rows);
    size_t nb = cols / 32, base = (size_t)row0 * nb;
    size_t chunk = STAGING_BYTES / (nb * 22);
    for (size_t r0 = 0; r0 < rows; r0 += chunk) {
        size_t nr = rows - r0 < chunk ? rows - r0 : chunk, n = nr * nb;
        const uint8_t *src = blocks(t, GGML_Q5_0, r0, nr);
        uint8_t *qs = staging;
        uint32_t *qh = (uint32_t *)(qs + n * 16);
        uint16_t *d = (uint16_t *)(qh + n);
        for (size_t b = 0; b < n; b++) {
            memcpy(&d[b], src + b * 22, 2);
            memcpy(&qh[b], src + b * 22 + 2, 4);
            memcpy(qs + b * 16, src + b * 22 + 6, 16);
        }
        size_t o = base + r0 * nb;
        CHECK(cuMemcpyHtoD(w->qs + o * 16, qs, n * 16));
        CHECK(cuMemcpyHtoD(w->qh + o * 4, qh, n * 4));
        CHECK(cuMemcpyHtoD(w->d + o * 2, d, n * 2));
    }
}

static Q5W alloc_q5(unsigned rows, unsigned cols)
{
    size_t n = (size_t)rows * (cols / 32);
    return (Q5W){dalloc(n * 16), dalloc(n * 4), dalloc(n * 2), rows, cols};
}

/* KV cache slots of a layer. A sliding-window layer keeps a ring of the
 * last window + batch positions: a chunk writes all its keys before its
 * attention runs, and its first token must still find the window before it. */
static unsigned cache_slots(const Config *c, unsigned l, unsigned ctx, unsigned batch)
{
    unsigned ring = c->sliding_window + batch;
    return is_swa(c, l) && ring < ctx ? ring : ctx;
}

/* Bytes of activation buffers for `batch` tokens. */
static size_t activation_bytes(const Config *c, unsigned batch)
{
    size_t E = c->n_embd, Q = (size_t)c->n_head * c->head_dim, KV = (size_t)c->n_kv_head * c->head_dim;
    size_t F = c->n_ff, B = batch;
    size_t per_token = 3 * E + Q + 2 * KV + Q + 2 * F + F;
    return B * per_token * 4 + F + F / 32 * 4 + (size_t)c->n_vocab * 4 + (2 + 2 * CAND_CAP) * 4 +
           8 + 16 * 256;
}

/* Expected device bytes for weights + cache + activations. */
static size_t model_bytes(const Config *c, unsigned ctx, unsigned batch)
{
    size_t E = c->n_embd, Q = (size_t)c->n_head * c->head_dim, KV = (size_t)c->n_kv_head * c->head_dim;
    size_t q8 = (Q * E + 2 * KV * E + E * Q) * 34 / 32;
    size_t q5 = 3 * (size_t)c->n_ff * E * 22 / 32;
    size_t small = (4 * E + 2 * c->head_dim) * 4;
    size_t cache = 0;
    for (unsigned l = 0; l < c->n_layer; l++)
        cache += (size_t)cache_slots(c, l, ctx, batch) * KV * 4;
    return c->n_layer * (q8 + q5 + small) + cache + (size_t)c->n_vocab * E * 34 / 32 + E * 4 +
           2 * (size_t)ctx * c->head_dim * 4 + activation_bytes(c, batch) +
           (c->n_layer * 16 + 64) * 256 + (8u << 20);
}

static void model_load(Model *m, const Gguf *g, unsigned ctx, unsigned batch)
{
    Config *c = &m->c;
    read_config(g, c);
    unsigned E = c->n_embd, Q = c->n_head * c->head_dim, KV = c->n_kv_head * c->head_dim,
             F = c->n_ff, D = c->head_dim;

    size_t free_b, total_b;
    CHECK(cuMemGetInfo(&free_b, &total_b));
    if (!batch) /* the largest prompt chunk that fits next to the weights */
        for (batch = AUTO_MAX_BATCH;
             batch > 1 && model_bytes(c, ctx, batch) + AUTO_SPARE_BYTES > free_b; batch /= 2)
            ;
    m->ctx = ctx;
    m->batch = batch;
    size_t need = model_bytes(c, ctx, batch);
    fprintf(stderr, "GPU memory: %.2f GiB needed, %.2f GiB free of %.2f GiB\n", need / 1073741824.0,
            free_b / 1073741824.0, total_b / 1073741824.0);
    if (cuMemAlloc(&arena, need) != CUDA_SUCCESS) {
        fprintf(stderr, "not enough GPU memory; close other GPU programs or use a smaller -c "
                        "or -b\n");
        exit(1);
    }
    arena_size = need;
    CHECK(cuMemHostAlloc(&staging, STAGING_BYTES, 0));
    quant_buf = malloc(STAGING_BYTES);

    double t0 = now_ms();
    m->layer = calloc(c->n_layer, sizeof(Layer));
    char name[128];
#define T(fmt) (snprintf(name, sizeof(name), "blk.%u." fmt, l), name)
    for (unsigned l = 0; l < c->n_layer; l++) {
        Layer *L = &m->layer[l];
        L->attn_norm = upload_f32(g, T("attn_norm.weight"), E);
        L->q_norm = upload_f32(g, T("attn_q_norm.weight"), D);
        L->k_norm = upload_f32(g, T("attn_k_norm.weight"), D);
        L->post_attn_norm = upload_f32(g, T("post_attention_norm.weight"), E);
        L->ffn_norm = upload_f32(g, T("ffn_norm.weight"), E);
        L->post_ffw_norm = upload_f32(g, T("post_ffw_norm.weight"), E);
        L->wq = upload_q8(g, T("attn_q.weight"), E, Q);
        L->wk = upload_q8(g, T("attn_k.weight"), E, KV);
        L->wv = upload_q8(g, T("attn_v.weight"), E, KV);
        L->wo = upload_q8(g, T("attn_output.weight"), Q, E);
        L->gate_up = alloc_q5(2 * F, E);
        upload_q5(g, T("ffn_gate.weight"), E, F, &L->gate_up, 0);
        upload_q5(g, T("ffn_up.weight"), E, F, &L->gate_up, F);
        L->down = alloc_q5(E, F);
        upload_q5(g, T("ffn_down.weight"), F, E, &L->down, 0);
        L->swa = is_swa(c, l);
        L->slots = cache_slots(c, l, ctx, batch);
        L->kc = dalloc((size_t)L->slots * KV * 2);
        L->vc = dalloc((size_t)L->slots * KV * 2);
        fprintf(stderr, "\rloading layer %u/%u  (%.2f GiB on GPU, %.1f s)", l + 1, c->n_layer,
                dev_bytes / 1073741824.0, (now_ms() - t0) / 1e3);
    }
#undef T
    m->out_norm = upload_f32(g, "output_norm.weight", E);
    int tied = 1;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        tied &= strcmp(g->tensors[i].name, "output.weight") != 0;
    m->out = upload_q8(g, tied ? "token_embd.weight" : "output.weight", E, c->n_vocab);
    m->embd = find_tensor(g, "token_embd.weight", E, c->n_vocab);
    if (m->embd->type != GGML_Q8_0 && m->embd->type != GGML_BF16) {
        fprintf(stderr, "token_embd.weight: type %u is not supported\n", m->embd->type);
        exit(1);
    }

    /* RoPE: global layers base rope_base with linear scaling, local ones base 10000 */
    float *cs = malloc((size_t)ctx * D * sizeof(float));
    m->rope_global = dalloc((size_t)ctx * D * 4);
    m->rope_local = dalloc((size_t)ctx * D * 4);
    rope_table(D, ctx, c->rope_base, c->rope_scale, cs);
    CHECK(cuMemcpyHtoD(m->rope_global, cs, (size_t)ctx * D * 4));
    rope_table(D, ctx, c->rope_base_local, 1.0f, cs);
    CHECK(cuMemcpyHtoD(m->rope_local, cs, (size_t)ctx * D * 4));
    free(cs);

    size_t B = batch;
    m->x = dalloc(B * E * 4);
    m->xn = dalloc(B * E * 4);
    m->y = dalloc(B * E * 4);
    m->qkv = dalloc(B * (Q + 2 * KV) * 4);
    m->att = dalloc(B * Q * 4);
    m->gu = dalloc(B * 2 * F * 4);
    m->h = dalloc(B * F * 4);
    m->xq = dalloc(F);          /* decoding: int8 input of a Q5 matrix */
    m->xd = dalloc(F / 32 * 4);
    m->logits = dalloc((size_t)c->n_vocab * 4);
    m->next = dalloc(4);
    m->cand = dalloc((2 + 2 * CAND_CAP) * 4);
    m->draw = dalloc(8);
    m->h_x = malloc(B * E * sizeof(float));
    m->cache_tok = malloc(ctx * sizeof(int32_t));
    CHECK(cuMemFreeHost(staging));
    free(quant_buf);
    fprintf(stderr, "\rloaded %u layers in %.1f s: %.2f GiB on the GPU, context %u, batch %u        \n",
            c->n_layer, (now_ms() - t0) / 1e3, dev_bytes / 1073741824.0, ctx, batch);
}

/* Embedding of a token: its row of the token matrix as Q8_0 values (what the
 * tied output head uses; a BF16 file is rounded to Q8_0 on the fly) times
 * sqrt(n_embd). */
static void embed_token(const Model *m, int32_t id, float *out)
{
    unsigned E = m->c.n_embd;
    uint8_t blk[(8192 / 32) * 34];
    const uint8_t *row;
    if (m->embd->type == GGML_Q8_0) {
        row = m->embd->data + (size_t)id * (E / 32) * 34;
    } else {
        const uint16_t *src = (const uint16_t *)m->embd->data + (size_t)id * E;
        for (unsigned i = 0; i < E; i++)
            out[i] = bf16_to_float(src[i]);
        quantize_q8_0(out, blk, E);
        row = blk;
    }
    dequantize_q8_0(row, out, E);
    float s = sqrtf((float)E);
    for (unsigned i = 0; i < E; i++)
        out[i] *= s;
}

/* --- debug dumps (GEMMA3_DUMP=1) --- */

static int dump_enabled;

static void dump(const char *name, int il, CUdeviceptr p, unsigned n)
{
    if (!dump_enabled)
        return;
    float *h = malloc(n * sizeof(float));
    CHECK(cuMemcpyDtoH(h, p, n * sizeof(float)));
    double sum = 0;
    for (unsigned i = 0; i < n; i++)
        sum += h[i];
    char full[64];
    if (il >= 0)
        snprintf(full, sizeof(full), "%s-%d", name, il);
    else
        snprintf(full, sizeof(full), "%s", name);
    printf("  %-22s [%6u] sum = %14.6f  [%10.4f %10.4f %10.4f ... %10.4f %10.4f %10.4f]\n",
           full, n, sum, h[0], h[1], h[2], h[n - 3], h[n - 2], h[n - 1]);
    free(h);
}

/* --- forward pass --- */

/* Runs T (<= m->batch) tokens at positions pos0 .. pos0+T-1; with
 * want_logits, fills m->logits and m->next (argmax) for the last one.
 * A single token goes through the matrix-vector kernels; a chunk of several
 * goes through the tiled GEMMs, which read every weight once per chunk. Both
 * paths feed the Q5 matrices the same Q8-rounded activations. Dumps show the
 * last token. */
static int force_gemm; /* cache-test: single tokens through the GEMMs too, for exact comparisons */

static void forward(Model *m, const int32_t *tokens, unsigned T, unsigned pos0, int want_logits)
{
    const Config *c = &m->c;
    unsigned E = c->n_embd, Q = c->n_head * c->head_dim, KV = c->n_kv_head * c->head_dim;
    unsigned QKV = Q + 2 * KV, F = c->n_ff, last = T - 1;

    for (unsigned t = 0; t < T; t++)
        embed_token(m, tokens[t], m->h_x + (size_t)t * E);
    CHECK(cuMemcpyHtoD(m->x, m->h_x, (size_t)T * E * 4));
    dump("inp_scaled", -1, m->x + (size_t)last * E * 4, E);
    rmsnorm(m->x, m->layer[0].attn_norm, m->xn, E, c->eps, T);

    for (unsigned l = 0; l < c->n_layer; l++) {
        const Layer *L = &m->layer[l];
        /* attention */
        dump("attn_norm", l, m->xn + (size_t)last * E * 4, E);
        int mv = T == 1 && !force_gemm;
        if (mv) {
            mv_q8(&L->wq, m->xn, 0, 0, m->qkv);
            mv_q8(&L->wk, m->xn, 0, 0, m->qkv + Q * 4);
            mv_q8(&L->wv, m->xn, 0, 0, m->qkv + (Q + KV) * 4);
        } else {
            gemm_q8(&L->wq, m->xn, 0, 0, m->qkv, QKV, T);
            gemm_q8(&L->wk, m->xn, 0, 0, m->qkv + Q * 4, QKV, T);
            gemm_q8(&L->wv, m->xn, 0, 0, m->qkv + (Q + KV) * 4, QKV, T);
        }
        qk_norm_rope(m->qkv, L->q_norm, L->k_norm, L->swa ? m->rope_local : m->rope_global,
                     L->kc, L->vc, pos0, L->slots, c->n_head, c->n_kv_head, c->eps, T);
        CUdeviceptr qkv_last = m->qkv + (size_t)last * QKV * 4;
        dump("Qcur", l, qkv_last, Q);
        dump("Kcur", l, qkv_last + Q * 4, KV);
        dump("Vcur", l, qkv_last + (Q + KV) * 4, KV);
        attention(m->qkv, QKV, L->kc, L->vc, m->att, pos0, L->swa ? c->sliding_window : 0,
                  L->slots, c->n_head, c->n_kv_head, c->attn_scale, T);
        dump("kqv_out", l, m->att + (size_t)last * Q * 4, Q);
        if (mv)
            mv_q8(&L->wo, m->att, 0, 0, m->y);
        else
            gemm_q8(&L->wo, m->att, 0, 0, m->y, E, T);
        dump("attn_out", l, m->y + (size_t)last * E * 4, E);
        /* x += post_attn_norm(y); xn = ffn_norm(x) */
        add_rmsnorm(m->y, L->post_attn_norm, m->x, L->ffn_norm, m->xn, E, c->eps, T);
        dump("sa_out", l, m->x + (size_t)last * E * 4, E);
        dump("ffn_norm", l, m->xn + (size_t)last * E * 4, E);

        /* feed-forward: down(gelu(gate . xn) * (up . xn)) */
        if (mv) {
            quant_q8(m->xn, m->xq, m->xd, E);
            mv_q5(&L->gate_up, m->xq, m->xd, m->gu);
            geglu(m->gu, m->h, F, 1);
            quant_q8(m->h, m->xq, m->xd, F);
            mv_q5(&L->down, m->xq, m->xd, m->y);
        } else {
            quant_q8(m->xn, 0, 0, T * E); /* round in place, as the int8 path does */
            gemm_q5(&L->gate_up, m->xn, 0, 0, m->gu, 2 * F, T);
            geglu(m->gu, m->h, F, T);
            quant_q8(m->h, 0, 0, T * F);
            gemm_q5(&L->down, m->h, 0, 0, m->y, E, T);
        }
        dump("ffn_out", l, m->y + (size_t)last * E * 4, E);
        /* x += post_ffw_norm(y); xn = next layer's attn_norm(x), or output_norm */
        CUdeviceptr next_norm = l + 1 < c->n_layer ? m->layer[l + 1].attn_norm : m->out_norm;
        add_rmsnorm(m->y, L->post_ffw_norm, m->x, next_norm, m->xn, E, c->eps, T);
        dump("l_out", l, m->x + (size_t)last * E * 4, E);
    }
    if (!want_logits)
        return;
    CUdeviceptr xn_last = m->xn + (size_t)last * E * 4;
    dump("result_norm", -1, xn_last, E);
    mv_q8(&m->out, xn_last, 0, 0, m->logits);
    dump("result_output", -1, m->logits, c->n_vocab);
    argmax(m->logits, c->n_vocab, m->next);
}

/* Prints the 5 highest logits (debug). */
static void print_top_logits(const Model *m, const Tokenizer *tok)
{
    unsigned n = m->c.n_vocab;
    float *h = malloc(n * sizeof(float));
    CHECK(cuMemcpyDtoH(h, m->logits, n * sizeof(float)));
    int top[5] = {-1, -1, -1, -1, -1};
    for (unsigned i = 0; i < n; i++)
        for (int k = 0; k < 5; k++)
            if (top[k] < 0 || h[i] > h[top[k]]) {
                memmove(top + k + 1, top + k, (4 - k) * sizeof(int));
                top[k] = i;
                break;
            }
    printf("  top logits:");
    for (int k = 0; k < 5; k++) {
        size_t len;
        const char *s = tok_piece(tok, top[k], &len);
        printf("  %d '", top[k]);
        for (size_t i = 0; i < len; i++)
            printf(s[i] == '\n' ? "\\n" : s[i] == '\t' ? "\\t" : "%c", s[i]);
        printf("' %.4f", h[top[k]]);
    }
    printf("\n");
    free(h);
}

/* ------------------------------------------------------------------------ */
/* Generation                                                                */

/* Next token from m->logits. Greedy takes argmax's result. Otherwise the
 * GPU draws from the whole vocabulary (temperature, min-p); with top-k /
 * top-p, it gathers the tokens within a logit margin of the best instead and
 * the host sorts and filters those, widening the margin (finally to the
 * whole vocabulary) until they cover what the filters keep. */
static int32_t next_token(Model *m, Sampler *s)
{
    static const float margins[] = {8, 16}; /* in units of the temperature */
    static uint32_t *h_cand, *ci;
    static float *cv, *h_logits;
    int32_t best;
    if (s->temperature <= 0) {
        CHECK(cuMemcpyDtoH(&best, m->next, 4));
        return best;
    }
    unsigned nv = m->c.n_vocab, n;
    int sorted = sampler_needs_sort(s);
    uint32_t draw[2];
    sample_logits(m->logits, nv, m->next, 1.0f / s->temperature, sorted ? 0 : s->min_p,
                  (float)sampler_uniform(s), m->draw);
    CHECK(cuMemcpyDtoH(draw, m->draw, 8));
    if (!sorted)
        return (int32_t)draw[0];

    float z;
    memcpy(&z, &draw[1], 4);
    if (!h_cand) {
        h_cand = malloc(2 * CAND_CAP * 4);
        ci = malloc(nv * 4);
        cv = malloc(nv * 4);
    }
    for (size_t k = 0; k < sizeof(margins) / sizeof(margins[0]); k++) {
        uint32_t count;
        compact_logits(m->logits, nv, m->next, margins[k] * s->temperature, m->cand, CAND_CAP);
        CHECK(cuMemcpyDtoH(&count, m->cand, 4));
        if (count > CAND_CAP)
            break;
        CHECK(cuMemcpyDtoH(h_cand, m->cand + 8, count * 8));
        for (n = 0; n < count; n++) {
            ci[n] = h_cand[2 * n];
            memcpy(&cv[n], &h_cand[2 * n + 1], 4);
        }
        int32_t t = sampler_pick(s, ci, cv, count, z);
        if (t >= 0)
            return t;
    }
    if (!h_logits)
        h_logits = malloc(nv * 4);
    CHECK(cuMemcpyDtoH(h_logits, m->logits, nv * 4));
    for (n = 0; n < nv; n++) {
        ci[n] = n;
        cv[n] = h_logits[n];
    }
    return sampler_pick(s, ci, cv, nv, 0);
}

/* Bytes of the sliding-window rings (0 when every layer keeps all positions). */
static size_t ring_bytes(const Model *m)
{
    size_t kv = (size_t)m->c.n_kv_head * m->c.head_dim * 2, total = 0;
    for (unsigned l = 0; l < m->c.n_layer; l++)
        if (m->layer[l].slots < m->ctx)
            total += 2 * m->layer[l].slots * kv;
    return total;
}

/* Copies the rings to / from host memory (dir 0: save, 1: restore). */
static void ring_copy(Model *m, int restore)
{
    size_t kv = (size_t)m->c.n_kv_head * m->c.head_dim * 2, off = 0;
    for (unsigned l = 0; l < m->c.n_layer; l++) {
        const Layer *L = &m->layer[l];
        if (L->slots == m->ctx)
            continue;
        size_t n = L->slots * kv;
        char *h = (char *)m->snap + off;
        if (restore) {
            CHECK(cuMemcpyHtoD(L->kc, h, n));
            CHECK(cuMemcpyHtoD(L->vc, h + n, n));
        } else {
            CHECK(cuMemcpyDtoH(h, L->kc, n));
            CHECK(cuMemcpyDtoH(h + n, L->vc, n));
        }
        off += 2 * n;
    }
}

/* Remembers the current cache so prefill() can return to it later. */
static void cache_snapshot(Model *m)
{
    size_t bytes = ring_bytes(m);
    if (!bytes)
        return;
    if (!m->snap)
        CHECK(cuMemHostAlloc(&m->snap, bytes, 0));
    ring_copy(m, 0);
    m->snap_pos = m->n_cache;
    m->has_snap = 1;
}

/* Brings the cache to hold toks[0..n) and computes the logits after the
 * last token, reusing the longest cached prefix. Global layers stay valid
 * for any prefix; a sliding-window ring (window + batch slots) only when
 * cutting back at most `batch` positions, or from the snapshot. */
static int no_reuse; /* GEMMA3_NO_REUSE=1: always recompute from position 0 */

static void prefill(Model *m, const int32_t *toks, unsigned n)
{
    unsigned L = 0;
    while (!no_reuse && L + 1 < n && L < m->n_cache && m->cache_tok[L] == toks[L])
        L++; /* the last token always runs, to produce logits */
    int ring = ring_bytes(m) > 0;
    unsigned start = 0;
    if (!ring || L + m->batch >= m->n_cache) {
        start = L;
    } else if (m->has_snap && L + m->batch >= m->snap_pos) {
        ring_copy(m, 1);
        start = L < m->snap_pos ? L : m->snap_pos;
    }
    for (unsigned i = start; i < n; i += m->batch) {
        unsigned T = n - i < m->batch ? n - i : m->batch;
        if (dump_enabled)
            printf("-- tokens %u..%u (last %d)\n", i, i + T - 1, toks[i + T - 1]);
        forward(m, toks + i, T, i, i + T == n);
    }
    memcpy(m->cache_tok + start, toks + start, (n - start) * sizeof(int32_t));
    m->n_cache = n;
}

/* Appends one token and computes the logits after it. */
static void step(Model *m, int32_t token)
{
    unsigned pos = m->n_cache;
    forward(m, &token, 1, pos, 1);
    m->cache_tok[pos] = token;
    m->n_cache = pos + 1;
}

typedef struct {
    Sampler sampler;
    int max_new;          /* < 0: until the model stops or the context is full */
    const char *system;   /* chat system prompt, or NULL */
} GenOptions;

/* The model's end-of-turn / end-of-text tokens. */
static int is_stop(const Tokenizer *tok, int32_t id)
{
    static int32_t eot = -2, eos = -2;
    if (eot == -2) {
        eot = tok_special(tok, "<end_of_turn>");
        eos = tok_special(tok, "<eos>");
    }
    return id == eot || id == eos;
}

/* Writes a generated token's text (control tokens are not shown). */
static void emit(const Tokenizer *tok, int32_t id)
{
    if (tok_type(tok, id) == TOK_CONTROL)
        return;
    size_t len;
    const char *s = tok_piece(tok, id, &len);
    fwrite(s, 1, len, stdout);
    fflush(stdout);
}

/* Plain continuation of `prompt` (no chat template, but after <bos>). */
static void run_completion(Model *m, const Tokenizer *tok, const char *prompt, GenOptions *o)
{
    int32_t *body;
    unsigned nb = tok_encode(tok, prompt, 1, &body), n = nb + 1;
    int32_t *ids = malloc(n * sizeof(int32_t));
    ids[0] = tok_special(tok, "<bos>");
    memcpy(ids + 1, body, nb * sizeof(int32_t));
    free(body);
    if (n >= m->ctx) {
        fprintf(stderr, "prompt must have 1..%u tokens (has %u)\n", m->ctx - 2, nb);
        exit(1);
    }
    fprintf(stderr, "prompt: %u tokens\n", n);
    double t0 = now_ms();
    prefill(m, ids, n);
    CHECK(cuCtxSynchronize());
    double t_prompt = now_ms() - t0;
    if (dump_enabled)
        print_top_logits(m, tok);
    const char *logits_path = getenv("GEMMA3_LOGITS");
    if (logits_path) { /* the logits after the prompt, raw f32 (for comparisons) */
        float *h = malloc((size_t)m->c.n_vocab * 4);
        CHECK(cuMemcpyDtoH(h, m->logits, (size_t)m->c.n_vocab * 4));
        FILE *f = fopen(logits_path, "wb");
        if (!f || fwrite(h, 4, m->c.n_vocab, f) != m->c.n_vocab || fclose(f))
            fprintf(stderr, "cannot write %s\n", logits_path);
        free(h);
    }

    unsigned limit = o->max_new < 0 ? 64 : (unsigned)o->max_new, n_gen = 0;
    printf("%s", prompt);
    fflush(stdout);
    t0 = now_ms();
    while (n_gen < limit) {
        int32_t next = next_token(m, &o->sampler);
        emit(tok, next);
        if (++n_gen == limit || is_stop(tok, next) || m->n_cache >= m->ctx)
            break;
        if (dump_enabled)
            printf("\n-- token %d (pos %u)\n", next, m->n_cache);
        step(m, next);
        if (dump_enabled)
            print_top_logits(m, tok);
    }
    CHECK(cuCtxSynchronize());
    double t_gen = now_ms() - t0;
    printf("\n");
    fprintf(stderr, "\nprompt: %u tokens in %.0f ms (%.1f tok/s)   generated: %u tokens in %.0f ms "
                    "(%.1f tok/s)\n", n, t_prompt, n * 1e3 / t_prompt, n_gen, t_gen,
            n_gen * 1e3 / t_gen);
    free(ids);
}

/* One model reply to the conversation in turns[0..n). Streams it to stdout
 * and returns its text (malloc'ed), or NULL if the conversation does not fit
 * in the context. */
static char *chat_reply(Model *m, const Tokenizer *tok, const ChatTurn *turns, int n,
                        GenOptions *o)
{
    int32_t *ids;
    unsigned n_ids = chat_render(tok, o->system, turns, n, &ids);
    if (n_ids >= m->ctx) {
        fprintf(stderr, "the conversation (%u tokens) does not fit in the context of %u; "
                        "use /reset or a larger -c\n", n_ids, m->ctx);
        free(ids);
        return NULL;
    }
    unsigned reused = 0;
    while (!no_reuse && reused + 1 < n_ids && reused < m->n_cache && m->cache_tok[reused] == ids[reused])
        reused++;
    double t0 = now_ms();
    prefill(m, ids, n_ids);
    cache_snapshot(m);
    CHECK(cuCtxSynchronize());
    double t_prompt = now_ms() - t0;
    free(ids);

    size_t cap = 256, len = 0;
    char *reply = malloc(cap);
    unsigned n_gen = 0, limit = o->max_new < 0 ? m->ctx : (unsigned)o->max_new;
    const char *stop = NULL;
    t0 = now_ms();
    for (;;) {
        int32_t next = next_token(m, &o->sampler);
        if (is_stop(tok, next))
            break;
        emit(tok, next);
        size_t pl;
        const char *piece = tok_piece(tok, next, &pl);
        if (tok_type(tok, next) == TOK_CONTROL)
            pl = 0;
        if (len + pl + 1 > cap) {
            cap = (len + pl + 1) * 2;
            reply = realloc(reply, cap);
        }
        memcpy(reply + len, piece, pl);
        len += pl;
        if (++n_gen >= limit || m->n_cache >= m->ctx) {
            stop = m->n_cache >= m->ctx ? "context full" : "reply length limit (-n)";
            break;
        }
        step(m, next);
    }
    reply[len] = 0;
    CHECK(cuCtxSynchronize());
    double t_gen = now_ms() - t0;
    printf("\n");
    fflush(stdout);
    if (stop)
        fprintf(stderr, "[stopped: %s]\n", stop);
    fprintf(stderr, "[prompt %u tokens, %u new in %.0f ms | reply %u tokens, %.1f tok/s]\n",
            n_ids, n_ids - reused, t_prompt, n_gen, n_gen * 1e3 / t_gen);
    return reply;
}

/* Multi-turn chat on stdin: one user message per line; /reset, /quit. */
static void run_interactive(Model *m, const Tokenizer *tok, GenOptions *o)
{
    ChatTurn *turns = NULL;
    int n = 0, cap = 0, prompt_tty = isatty(0);
    char *line = NULL;
    size_t line_cap = 0;
    fprintf(stderr, "chat: type a message; /reset starts over, /quit (or EOF) exits\n");
    for (;;) {
        if (prompt_tty) {
            printf("\n> ");
            fflush(stdout);
        }
        ssize_t len = getline(&line, &line_cap, stdin);
        if (len < 0)
            break;
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (len == 0)
            continue;
        if (!strcmp(line, "/quit") || !strcmp(line, "/exit"))
            break;
        if (!strcmp(line, "/reset")) {
            for (int i = 0; i < n; i++)
                free((char *)turns[i].content);
            n = 0;
            fprintf(stderr, "[conversation cleared]\n");
            continue;
        }
        if (n + 2 > cap) {
            cap = cap ? cap * 2 : 16;
            turns = realloc(turns, cap * sizeof(ChatTurn));
        }
        turns[n++] = (ChatTurn){1, strdup(line)};
        char *reply = chat_reply(m, tok, turns, n, o);
        if (!reply) {
            free((char *)turns[--n].content);
            continue;
        }
        turns[n++] = (ChatTurn){0, reply};
    }
    for (int i = 0; i < n; i++)
        free((char *)turns[i].content);
    free(turns);
    free(line);
}

/* ------------------------------------------------------------------------ */
/* Kernel self-tests against CPU references                                  */

static CUdeviceptr upload(const void *p, size_t n)
{
    CUdeviceptr d;
    CHECK(cuMemAlloc(&d, n));
    CHECK(cuMemcpyHtoD(d, p, n));
    return d;
}

static int report(const char *name, double err, double tol)
{
    int ok = err < tol;
    printf("   %-48s max rel err = %.2e  %s\n", name, err, ok ? "PASS" : "FAIL");
    return !ok;
}

/* Error of got vs ref relative to the largest |ref|. */
static double rel_err(const float *got, const double *ref, unsigned n)
{
    double max_err = 0, max_ref = 1e-30;
    for (unsigned i = 0; i < n; i++) {
        max_err = fmax(max_err, fabs(got[i] - ref[i]));
        max_ref = fmax(max_ref, fabs(ref[i]));
    }
    return max_err / max_ref;
}

/* CPU Q8 quantization of activations, mirroring quant_q8. */
static void cpu_quant_q8(const float *x, unsigned n, int8_t *q, float *d)
{
    for (unsigned b = 0; b < n / 32; b++) {
        float amax = 0;
        for (int j = 0; j < 32; j++)
            amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        d[b] = amax / 127;
        float inv = d[b] ? 1 / d[b] : 0;
        for (int j = 0; j < 32; j++)
            q[b * 32 + j] = (int8_t)rintf(x[b * 32 + j] * inv);
    }
}

/* Random [rows][cols] matrix (with some outliers), its Q8_0 and Q5_0 blocks,
 * and the GPU matrices made from them the way model_load makes them. */
static float *random_matrix(unsigned rows, unsigned cols)
{
    float *w = malloc((size_t)rows * cols * 4);
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        w[i] = rng_uniform(-1, 1) * (i % 97 == 0 ? 8 : 1);
    return w;
}

static Q8W gpu_q8(const uint8_t *blk, unsigned rows, unsigned cols)
{
    size_t nb = (size_t)rows * cols / 32;
    int8_t *q = malloc(nb * 32);
    uint16_t *d = malloc(nb * 2);
    for (size_t b = 0; b < nb; b++) {
        memcpy(&d[b], blk + b * 34, 2);
        memcpy(q + b * 32, blk + b * 34 + 2, 32);
    }
    Q8W w = {upload(q, nb * 32), upload(d, nb * 2), rows, cols};
    free(q), free(d);
    return w;
}

static Q5W gpu_q5(const uint8_t *blk, unsigned rows, unsigned cols)
{
    size_t nb = (size_t)rows * cols / 32;
    uint8_t *qs = malloc(nb * 16);
    uint32_t *qh = malloc(nb * 4);
    uint16_t *d = malloc(nb * 2);
    for (size_t b = 0; b < nb; b++) {
        memcpy(&d[b], blk + b * 22, 2);
        memcpy(&qh[b], blk + b * 22 + 2, 4);
        memcpy(qs + b * 16, blk + b * 22 + 6, 16);
    }
    Q5W w = {upload(qs, nb * 16), upload(qh, nb * 4), upload(d, nb * 2), rows, cols};
    free(qs), free(qh), free(d);
    return w;
}

static int test_quant_formats(void)
{
    /* dequantized values must be within half a step (d) of the input, plus the
     * f16 rounding of d; Q5_0 maps the largest magnitude to -16, so values of
     * the opposite sign clamp at +15 and can be off by up to one step */
    const unsigned n = 4096;
    float *x = random_matrix(1, n), *y = malloc(n * 4);
    uint8_t *b = malloc(n / 32 * 34);
    double e8 = 0, e5 = 0;
    quantize_q8_0(x, b, n);
    dequantize_q8_0(b, y, n);
    for (unsigned i = 0; i < n; i++) {
        float amax = 0;
        for (unsigned j = i / 32 * 32; j < i / 32 * 32 + 32; j++)
            amax = fmaxf(amax, fabsf(x[j]));
        e8 = fmax(e8, fabs(y[i] - x[i]) / (amax / 127));
    }
    quantize_q5_0(x, b, n);
    dequantize_q5_0(b, y, n);
    for (unsigned i = 0; i < n; i++) {
        float amax = 0;
        for (unsigned j = i / 32 * 32; j < i / 32 * 32 + 32; j++)
            amax = fmaxf(amax, fabsf(x[j]));
        e5 = fmax(e5, fabs(y[i] - x[i]) / (amax / 16));
    }
    int ok = e8 < 0.6 && e5 < 1.05;
    printf("   %-48s %.3f / %.3f steps  %s\n", "Q8_0 / Q5_0 round trip (max error)", e8, e5,
           ok ? "PASS" : "FAIL");
    free(x), free(y), free(b);
    return !ok;
}

static int test_mv_q8(void)
{
    const unsigned rows = 37, cols = 4096;
    float *w = random_matrix(rows, cols), *wq = malloc(rows * cols * 4);
    float *x = malloc(cols * 4), *bias = malloc(rows * 4), *resid = malloc(rows * 4);
    uint8_t *blk = malloc(rows * cols / 32 * 34);
    for (unsigned i = 0; i < cols; i++)
        x[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < rows; i++) {
        bias[i] = rng_uniform(-1, 1);
        resid[i] = rng_uniform(-1, 1);
    }
    quantize_q8_0(w, blk, (size_t)rows * cols);
    dequantize_q8_0(blk, wq, (size_t)rows * cols);
    double *ref = malloc(rows * sizeof(double)), *ref_fp = malloc(rows * sizeof(double));
    for (unsigned r = 0; r < rows; r++) {
        double acc = 0, acc_fp = 0;
        for (unsigned k = 0; k < cols; k++) {
            acc += (double)wq[(size_t)r * cols + k] * x[k];
            acc_fp += (double)w[(size_t)r * cols + k] * x[k];
        }
        ref[r] = acc + bias[r] + resid[r];
        ref_fp[r] = acc_fp + bias[r] + resid[r];
    }
    Q8W g = gpu_q8(blk, rows, cols);
    CUdeviceptr d_x = upload(x, cols * 4), d_b = upload(bias, rows * 4),
                d_r = upload(resid, rows * 4), d_y;
    CHECK(cuMemAlloc(&d_y, rows * 4));
    mv_q8(&g, d_x, d_b, d_r, d_y);
    float *y = malloc(rows * 4);
    CHECK(cuMemcpyDtoH(y, d_y, rows * 4));
    int fails = report("mv_q8 vs Q8-weight reference", rel_err(y, ref, rows), 1e-5);
    fails += report("mv_q8 vs f32 (quantization error)", rel_err(y, ref_fp, rows), 2e-2);
    cuMemFree(g.q), cuMemFree(g.d), cuMemFree(d_x), cuMemFree(d_b), cuMemFree(d_r), cuMemFree(d_y);
    free(w), free(wq), free(x), free(bias), free(resid), free(blk), free(ref), free(ref_fp), free(y);
    return fails;
}

static int test_mv_q5(void)
{
    const unsigned rows = 45, cols = 3840, nb = cols / 32;
    float *w = random_matrix(rows, cols), *wq = malloc(rows * cols * 4), *x = malloc(cols * 4);
    uint8_t *blk = malloc(rows * nb * 22);
    for (unsigned i = 0; i < cols; i++)
        x[i] = rng_uniform(-3, 3);
    quantize_q5_0(w, blk, (size_t)rows * cols);
    dequantize_q5_0(blk, wq, (size_t)rows * cols);
    int8_t *xq = malloc(cols);
    float *xd = malloc(nb * 4);
    cpu_quant_q8(x, cols, xq, xd);
    /* exact reference of what the kernel computes: Q5 weights . Q8 activations */
    double *ref = malloc(rows * sizeof(double)), *ref_fp = malloc(rows * sizeof(double));
    for (unsigned r = 0; r < rows; r++) {
        double acc = 0, acc_fp = 0;
        for (unsigned b = 0; b < nb; b++)
            for (unsigned j = 0; j < 32; j++) {
                size_t k = (size_t)b * 32 + j;
                acc += (double)wq[(size_t)r * cols + k] * xq[k] * xd[b];
                acc_fp += (double)w[(size_t)r * cols + k] * x[k];
            }
        ref[r] = acc;
        ref_fp[r] = acc_fp;
    }
    Q5W g = gpu_q5(blk, rows, cols);
    CUdeviceptr d_x = upload(x, cols * 4), d_xq, d_xd, d_y;
    CHECK(cuMemAlloc(&d_xq, cols));
    CHECK(cuMemAlloc(&d_xd, nb * 4));
    CHECK(cuMemAlloc(&d_y, rows * 4));
    quant_q8(d_x, d_xq, d_xd, cols);
    mv_q5(&g, d_xq, d_xd, d_y);
    float *y = malloc(rows * 4);
    CHECK(cuMemcpyDtoH(y, d_y, rows * 4));
    int fails = report("quant_q8 + mv_q5 vs Q5 x Q8 reference", rel_err(y, ref, rows), 1e-5);
    fails += report("quant_q8 + mv_q5 vs f32 (quantization error)", rel_err(y, ref_fp, rows), 1e-1);
    CUdeviceptr bufs[] = {g.qs, g.qh, g.d, d_x, d_xq, d_xd, d_y};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        cuMemFree(bufs[i]);
    free(w), free(wq), free(x), free(blk), free(xq), free(xd), free(ref), free(ref_fp), free(y);
    return fails;
}

/* gemm_q8 / gemm_q5 (37 tokens, strided output, in-place residual) vs the
 * dequantized weights in double precision */
static int test_gemm(int q5)
{
    const unsigned rows = 128, cols = 384, T = 37, off = 32, ldy = rows + 64;
    float *w = random_matrix(rows, cols), *wq = malloc(rows * cols * 4);
    float *x = malloc(T * cols * 4), *bias = malloc(rows * 4), *y = malloc(T * ldy * 4);
    uint8_t *blk = malloc(rows * cols / 32 * 34);
    for (unsigned i = 0; i < T * cols; i++)
        x[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < rows; i++)
        bias[i] = rng_uniform(-1, 1);
    for (unsigned i = 0; i < T * ldy; i++)
        y[i] = rng_uniform(-1, 1); /* residual, updated in place */
    if (q5) {
        quantize_q5_0(w, blk, (size_t)rows * cols);
        dequantize_q5_0(blk, wq, (size_t)rows * cols);
    } else {
        quantize_q8_0(w, blk, (size_t)rows * cols);
        dequantize_q8_0(blk, wq, (size_t)rows * cols);
    }
    double *ref = malloc(T * ldy * sizeof(double));
    for (unsigned i = 0; i < T * ldy; i++)
        ref[i] = y[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned r = 0; r < rows; r++) {
            double acc = 0;
            for (unsigned k = 0; k < cols; k++)
                acc += (double)wq[r * cols + k] * x[t * cols + k];
            ref[t * ldy + off + r] += acc + bias[r];
        }
    CUdeviceptr d_x = upload(x, T * cols * 4), d_b = upload(bias, rows * 4),
                d_y = upload(y, T * ldy * 4);
    CUdeviceptr y_slice = d_y + off * 4;
    Q8W g8 = {0};
    Q5W g5 = {0};
    if (q5) {
        g5 = gpu_q5(blk, rows, cols);
        gemm_q5(&g5, d_x, d_b, y_slice, y_slice, ldy, T);
    } else {
        g8 = gpu_q8(blk, rows, cols);
        gemm_q8(&g8, d_x, d_b, y_slice, y_slice, ldy, T);
    }
    CHECK(cuMemcpyDtoH(y, d_y, T * ldy * 4));
    int fails = report(q5 ? "gemm_q5 (37 tokens, strided, in place)"
                          : "gemm_q8 (37 tokens, strided, in place)", rel_err(y, ref, T * ldy), 1e-5);
    CUdeviceptr bufs[] = {g8.q, g8.d, g5.qs, g5.qh, g5.d, d_x, d_b, d_y};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        if (bufs[i])
            cuMemFree(bufs[i]);
    free(w), free(wq), free(x), free(bias), free(y), free(blk), free(ref);
    return fails;
}

/* q/k head norm + RoPE and the K/V cache writes vs a CPU reference */
static int test_qk_norm_rope(void)
{
    const unsigned nh = 4, nkv = 2, D = HEAD_DIM, T = 3, pos0 = 70, slots = 32, ctx = 80;
    const unsigned QKV = (nh + 2 * nkv) * D;
    const float eps = 1e-6f;
    float *qkv = malloc(T * QKV * 4), *qw = malloc(D * 4), *kw = malloc(D * 4);
    float *cs = malloc((size_t)ctx * D * 4), *out = malloc(T * QKV * 4);
    for (unsigned i = 0; i < T * QKV; i++)
        qkv[i] = rng_uniform(-3, 3);
    for (unsigned i = 0; i < D; i++) {
        qw[i] = rng_uniform(0.5f, 2);
        kw[i] = rng_uniform(0.5f, 2);
    }
    rope_table(D, ctx, 1e6f, 0.125f, cs);
    double *ref = malloc(T * QKV * sizeof(double));
    for (unsigned t = 0; t < T; t++)
        for (unsigned h = 0; h < nh + 2 * nkv; h++) {
            const float *x = qkv + t * QKV + h * D;
            double *r = ref + t * QKV + h * D;
            if (h >= nh + nkv) {
                for (unsigned i = 0; i < D; i++)
                    r[i] = x[i];
                continue;
            }
            double ss = 0;
            for (unsigned i = 0; i < D; i++)
                ss += (double)x[i] * x[i];
            double s = 1 / sqrt(ss / D + eps);
            const float *w = h < nh ? qw : kw;
            for (unsigned i = 0; i < D / 2; i++) {
                double a = x[i] * s * w[i], b = x[i + D / 2] * s * w[i + D / 2];
                double c = cs[(size_t)(pos0 + t) * D + 2 * i], sn = cs[(size_t)(pos0 + t) * D + 2 * i + 1];
                r[i] = a * c - b * sn;
                r[i + D / 2] = a * sn + b * c;
            }
        }
    CUdeviceptr d_qkv = upload(qkv, T * QKV * 4), d_qw = upload(qw, D * 4), d_kw = upload(kw, D * 4),
                d_cs = upload(cs, (size_t)ctx * D * 4), d_kc, d_vc;
    CHECK(cuMemAlloc(&d_kc, slots * nkv * D * 2));
    CHECK(cuMemAlloc(&d_vc, slots * nkv * D * 2));
    qk_norm_rope(d_qkv, d_qw, d_kw, d_cs, d_kc, d_vc, pos0, slots, nh, nkv, eps, T);
    CHECK(cuMemcpyDtoH(out, d_qkv, T * QKV * 4));
    /* v is not rewritten in qkv: compare the q and k heads there, k and v in the cache */
    for (unsigned t = 0; t < T; t++)
        for (unsigned i = (nh + nkv) * D; i < QKV; i++)
            out[t * QKV + i] = qkv[t * QKV + i];
    int fails = report("qk_norm_rope (q, k in place)", rel_err(out, ref, T * QKV), 1e-5);
    uint16_t *kc = malloc(slots * nkv * D * 2), *vc = malloc(slots * nkv * D * 2);
    CHECK(cuMemcpyDtoH(kc, d_kc, slots * nkv * D * 2));
    CHECK(cuMemcpyDtoH(vc, d_vc, slots * nkv * D * 2));
    float *got = malloc(T * 2 * nkv * D * 4);
    double *want = malloc(T * 2 * nkv * D * sizeof(double));
    for (unsigned t = 0; t < T; t++)
        for (unsigned i = 0; i < nkv * D; i++) {
            unsigned slot = (pos0 + t) % slots;
            got[(t * 2) * nkv * D + i] = half_to_float(kc[slot * nkv * D + i]);
            got[(t * 2 + 1) * nkv * D + i] = half_to_float(vc[slot * nkv * D + i]);
            want[(t * 2) * nkv * D + i] = ref[t * QKV + nh * D + i];
            want[(t * 2 + 1) * nkv * D + i] = ref[t * QKV + (nh + nkv) * D + i];
        }
    fails += report("qk_norm_rope (f16 K/V cache slots)", rel_err(got, want, T * 2 * nkv * D), 1e-3);
    cuMemFree(d_qkv), cuMemFree(d_qw), cuMemFree(d_kw), cuMemFree(d_cs), cuMemFree(d_kc), cuMemFree(d_vc);
    free(qkv), free(qw), free(kw), free(cs), free(out), free(ref), free(kc), free(vc), free(got), free(want);
    return fails;
}

/* Causal attention for T queries at pos0.. over a (possibly ring-buffer)
 * cache, vs a CPU reference that reads keys by position. */
static int test_attention_case(unsigned pos0, unsigned T, unsigned window, unsigned slots)
{
    const unsigned nh = 4, nkv = 2, D = HEAD_DIM, q_stride = nh * D + 16, npos = pos0 + T;
    const float scale = 0.0625f;
    float *q = malloc(T * q_stride * 4), *out = malloc(T * nh * D * 4);
    uint16_t *kpos = malloc(npos * nkv * D * 2), *vpos = malloc(npos * nkv * D * 2);
    uint16_t *kc = calloc(slots * nkv * D, 2), *vc = calloc(slots * nkv * D, 2);
    for (unsigned i = 0; i < T * q_stride; i++)
        q[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < npos * nkv * D; i++) {
        kpos[i] = float_to_half(rng_uniform(-2, 2));
        vpos[i] = float_to_half(rng_uniform(-1, 1));
    }
    for (unsigned p = 0; p < npos; p++) /* later positions overwrite their ring slot */
        for (unsigned i = 0; i < nkv * D; i++) {
            kc[(p % slots) * nkv * D + i] = kpos[p * nkv * D + i];
            vc[(p % slots) * nkv * D + i] = vpos[p * nkv * D + i];
        }
    double *ref = malloc(T * nh * D * sizeof(double)), *s = malloc(npos * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        unsigned p = pos0 + t, lo = window && p >= window ? p - window + 1 : 0;
        for (unsigned h = 0; h < nh; h++) {
            unsigned kh = h / (nh / nkv);
            double m = -1e300, sum = 0;
            for (unsigned j = lo; j <= p; j++) {
                double dot = 0;
                for (unsigned d = 0; d < D; d++)
                    dot += q[t * q_stride + h * D + d] * scale * half_to_float(kpos[(j * nkv + kh) * D + d]);
                s[j] = dot;
                m = fmax(m, dot);
            }
            for (unsigned j = lo; j <= p; j++)
                sum += exp(s[j] - m);
            for (unsigned d = 0; d < D; d++) {
                double acc = 0;
                for (unsigned j = lo; j <= p; j++)
                    acc += exp(s[j] - m) * half_to_float(vpos[(j * nkv + kh) * D + d]);
                ref[(t * nh + h) * D + d] = acc / sum;
            }
        }
    }
    CUdeviceptr d_q = upload(q, T * q_stride * 4), d_k = upload(kc, slots * nkv * D * 2),
                d_v = upload(vc, slots * nkv * D * 2), d_o;
    CHECK(cuMemAlloc(&d_o, T * nh * D * 4));
    attention(d_q, q_stride, d_k, d_v, d_o, pos0, window, slots, nh, nkv, scale, T);
    CHECK(cuMemcpyDtoH(out, d_o, T * nh * D * 4));
    char name[64];
    snprintf(name, sizeof(name), "attention T=%u pos0=%u window=%u ring=%u", T, pos0, window, slots);
    int fails = report(name, rel_err(out, ref, T * nh * D), 1e-4);
    cuMemFree(d_q), cuMemFree(d_k), cuMemFree(d_v), cuMemFree(d_o);
    free(q), free(out), free(kpos), free(vpos), free(kc), free(vc), free(ref), free(s);
    return fails;
}

static int test_attention(void)
{
    return test_attention_case(0, 1, 0, 16) +     /* first token: one key */
           test_attention_case(76, 1, 0, 100) +   /* decoding, global layer */
           test_attention_case(40, 1, 16, 20) +   /* decoding, sliding window in a ring */
           test_attention_case(300, 1, 0, 400) +  /* many keys per warp */
           test_attention_case(0, 9, 0, 64) +     /* first chunk of a prompt */
           test_attention_case(3, 5, 16, 24) +    /* window not yet full */
           test_attention_case(40, 7, 16, 24);    /* chunk wrapping around the ring */
}

static int test_add_rmsnorm_geglu(void)
{
    const unsigned n = 3840, T = 3, F = 1000;
    const float eps = 1e-6f;
    float *y = malloc(T * n * 4), *x = malloc(T * n * 4), *w1 = malloc(n * 4), *w2 = malloc(n * 4);
    for (unsigned i = 0; i < T * n; i++) {
        y[i] = rng_uniform(-5, 5);
        x[i] = rng_uniform(-50, 50);
    }
    for (unsigned i = 0; i < n; i++) {
        w1[i] = rng_uniform(0, 3);
        w2[i] = rng_uniform(0, 3);
    }
    double *rx = malloc(T * n * sizeof(double)), *rn = malloc(T * n * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        double s1 = 0, s2 = 0;
        for (unsigned i = 0; i < n; i++)
            s1 += (double)y[t * n + i] * y[t * n + i];
        s1 = 1 / sqrt(s1 / n + eps);
        for (unsigned i = 0; i < n; i++) {
            rx[t * n + i] = x[t * n + i] + y[t * n + i] * s1 * w1[i];
            s2 += rx[t * n + i] * rx[t * n + i];
        }
        s2 = 1 / sqrt(s2 / n + eps);
        for (unsigned i = 0; i < n; i++)
            rn[t * n + i] = rx[t * n + i] * s2 * w2[i];
    }
    CUdeviceptr d_y = upload(y, T * n * 4), d_x = upload(x, T * n * 4), d_w1 = upload(w1, n * 4),
                d_w2 = upload(w2, n * 4), d_xn;
    CHECK(cuMemAlloc(&d_xn, T * n * 4));
    add_rmsnorm(d_y, d_w1, d_x, d_w2, d_xn, n, eps, T);
    float *gx = malloc(T * n * 4), *gn = malloc(T * n * 4);
    CHECK(cuMemcpyDtoH(gx, d_x, T * n * 4));
    CHECK(cuMemcpyDtoH(gn, d_xn, T * n * 4));
    int fails = report("add_rmsnorm (residual)", rel_err(gx, rx, T * n), 1e-6);
    fails += report("add_rmsnorm (normed)", rel_err(gn, rn, T * n), 1e-5);
    cuMemFree(d_y), cuMemFree(d_x), cuMemFree(d_w1), cuMemFree(d_w2), cuMemFree(d_xn);

    float *gu = malloc(T * 2 * F * 4), *h = malloc(T * F * 4);
    double *rh = malloc(T * F * sizeof(double));
    for (unsigned i = 0; i < T * 2 * F; i++)
        gu[i] = rng_uniform(-6, 6);
    for (unsigned t = 0; t < T; t++)
        for (unsigned j = 0; j < F; j++) {
            double g = gu[t * 2 * F + j], u = gu[t * 2 * F + F + j];
            rh[t * F + j] = 0.5 * g * (1 + tanh(sqrt(2 / M_PI) * g * (1 + 0.044715 * g * g))) * u;
        }
    CUdeviceptr d_gu = upload(gu, T * 2 * F * 4), d_h;
    CHECK(cuMemAlloc(&d_h, T * F * 4));
    geglu(d_gu, d_h, F, T);
    CHECK(cuMemcpyDtoH(h, d_h, T * F * 4));
    fails += report("geglu", rel_err(h, rh, T * F), 1e-6);
    cuMemFree(d_gu), cuMemFree(d_h);
    free(y), free(x), free(w1), free(w2), free(rx), free(rn), free(gx), free(gn);
    free(gu), free(h), free(rh);
    return fails;
}

/* compact_logits must gather exactly the logits within the margin of the
 * maximum; the sampler's frequencies must follow the filtered softmax. */
static int test_sampling(void)
{
    const unsigned n = 50000, cap = 4096;
    const float margin = 4.0f;
    float *x = malloc(n * 4);
    for (unsigned i = 0; i < n; i++)
        x[i] = rng_uniform(-10, 10);
    unsigned best = 0, want = 0;
    for (unsigned i = 1; i < n; i++)
        if (x[i] > x[best])
            best = i;
    for (unsigned i = 0; i < n; i++)
        want += x[i] >= x[best] - margin;
    CUdeviceptr d_x = upload(x, n * 4), d_best = upload(&best, 4), d_out;
    CHECK(cuMemAlloc(&d_out, (2 + 2 * cap) * 4));
    compact_logits(d_x, n, d_best, margin, d_out, cap);
    uint32_t *out = malloc((2 + 2 * cap) * 4);
    CHECK(cuMemcpyDtoH(out, d_out, (2 + 2 * cap) * 4));
    int ok = out[0] == want;
    for (unsigned j = 0; ok && j < out[0] && j < cap; j++) {
        float v;
        memcpy(&v, &out[3 + 2 * j], 4);
        ok = out[2 + 2 * j] < n && x[out[2 + 2 * j]] == v && v >= x[best] - margin;
    }
    printf("   %-48s %u of %u  %s\n", "compact_logits candidates", out[0], want, ok ? "PASS" : "FAIL");
    int fails = !ok;
    cuMemFree(d_x), cuMemFree(d_best), cuMemFree(d_out);
    free(x), free(out);

    /* sampler frequencies over 5 tokens */
    static const float logits[5] = {2.0f, 1.5f, 1.0f, 0.0f, -1.0f};
    struct { float temp, top_p, min_p; int top_k; const char *name; } cfg[] = {
        {0.7f, 1.0f, 0.0f, 0, "sampler temp 0.7"},
        {1.0f, 1.0f, 0.0f, 2, "sampler top-k 2"},
        {1.0f, 0.6f, 0.0f, 0, "sampler top-p 0.6"},
        {1.0f, 1.0f, 0.3f, 0, "sampler min-p 0.3"},
    };
    for (size_t c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        Sampler s = {cfg[c].temp, cfg[c].top_k, cfg[c].top_p, cfg[c].min_p, 0x1234567 + c};
        /* expected: softmax at temp over the survivors of the filters */
        double p[5], sum = 0, cum = 0;
        for (int i = 0; i < 5; i++)
            sum += p[i] = exp((logits[i] - logits[0]) / cfg[c].temp);
        int keep = cfg[c].top_k ? cfg[c].top_k : 5;
        if (cfg[c].top_p < 1)
            for (int i = 0; i < keep; i++)
                if ((cum += p[i] / sum) >= cfg[c].top_p) {
                    keep = i + 1;
                    break;
                }
        while (keep > 1 && p[keep - 1] < cfg[c].min_p * p[0])
            keep--;
        sum = 0;
        for (int i = 0; i < keep; i++)
            sum += p[i];
        const int draws = 200000;
        int hits[5] = {0};
        for (int d = 0; d < draws; d++) {
            uint32_t idx[5] = {3, 0, 4, 1, 2}; /* shuffled: the sampler sorts */
            float val[5];
            for (int i = 0; i < 5; i++)
                val[i] = logits[idx[i]];
            hits[sampler_pick(&s, idx, val, 5, 0)]++;
        }
        double worst = 0;
        for (int i = 0; i < 5; i++)
            worst = fmax(worst, fabs(hits[i] / (double)draws - (i < keep ? p[i] / sum : 0)));
        printf("   %-48s max |freq - p| = %.4f  %s\n", cfg[c].name, worst,
               worst < 0.005 ? "PASS" : "FAIL");
        fails += worst >= 0.005;
    }

    /* GPU whole-vocabulary draws: 5 likely tokens in a long tail */
    const unsigned nv = 50000, draws = 20000;
    static const float top[5] = {3.0f, 2.5f, 2.0f, 1.0f, 0.0f};
    static const unsigned pos[5] = {31337, 7, 49999, 20000, 123};
    float *v = malloc(nv * 4);
    for (unsigned i = 0; i < nv; i++)
        v[i] = -6.0f;
    for (int i = 0; i < 5; i++)
        v[pos[i]] = top[i];
    unsigned best0 = pos[0];
    CUdeviceptr d_v = upload(v, nv * 4), d_b = upload(&best0, 4), d_draw;
    CHECK(cuMemAlloc(&d_draw, 8));
    for (int mp = 0; mp < 2; mp++) { /* plain, then min-p 0.1 (drops the tail) */
        float min_w = mp ? 0.1f : 0.0f;
        double w[6], z = 0;
        for (int i = 0; i < 5; i++)
            z += w[i] = exp(top[i] - top[0]) >= min_w ? exp(top[i] - top[0]) : 0;
        z += w[5] = exp(-6.0 - top[0]) >= min_w ? (nv - 5) * exp(-6.0 - top[0]) : 0;
        unsigned hits[6] = {0};
        Sampler sm = {1.0f, 0, 1.0f, 0, 99};
        for (unsigned d = 0; d < draws; d++) {
            uint32_t out2[2];
            sample_logits(d_v, nv, d_b, 1.0f, min_w, (float)sampler_uniform(&sm), d_draw);
            CHECK(cuMemcpyDtoH(out2, d_draw, 8));
            int k = 5;
            for (int i = 0; i < 5; i++)
                if (out2[0] == pos[i])
                    k = i;
            hits[k]++;
        }
        double worst = 0;
        for (int i = 0; i < 6; i++)
            worst = fmax(worst, fabs(hits[i] / (double)draws - w[i] / z));
        printf("   %-48s max |freq - p| = %.4f  %s\n",
               mp ? "sample_logits min-p 0.1 (GPU)" : "sample_logits whole vocabulary (GPU)",
               worst, worst < 0.012 ? "PASS" : "FAIL");
        fails += worst >= 0.012;
    }
    cuMemFree(d_v), cuMemFree(d_b), cuMemFree(d_draw);
    free(v);
    return fails;
}

static int self_test(void)
{
    printf("== kernel self-tests against CPU references\n");
    int fails = test_quant_formats() + test_mv_q8() + test_mv_q5() + test_gemm(0) + test_gemm(1) +
                test_qk_norm_rope() + test_attention() + test_add_rmsnorm_geglu() + test_sampling();
    printf("%s\n", fails ? "SELF-TEST FAILED" : "all self-tests passed");
    return fails;
}

/* ------------------------------------------------------------------------ */
/* Command line                                                              */

static void init_cuda(const char *cubin)
{
    CUdevice dev;
    CUcontext ctx;
    char name[256];
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
    /* the kernels use no local memory or device printf/malloc: shrink the
     * per-context reservations to leave room for the weights */
    CHECK(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 0));
    CHECK(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 0));
    CHECK(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 0));
    fprintf(stderr, "device: %s\n", name);
    load_kernels(cubin);
}

/* The default model: the quantized file if it exists, else the BF16 one. */
static const char *default_model(void)
{
    if (access(DEFAULT_MODEL, R_OK) && !access(BF16_MODEL, R_OK)) {
        fprintf(stderr, "note: %s not found, loading %s (run `make quantize` once for faster "
                        "loading)\n", DEFAULT_MODEL, BF16_MODEL);
        return BF16_MODEL;
    }
    return DEFAULT_MODEL;
}

static int cmd_tokenize(int argc, char **argv)
{
    Gguf g;
    gguf_open(&g, argc >= 4 ? argv[3] : default_model());
    Tokenizer *tok = tok_load(&g);
    int32_t *ids;
    int n = tok_encode(tok, argv[2], 1, &ids);
    for (int i = 0; i < n; i++) {
        size_t len;
        const char *s = tok_piece(tok, ids[i], &len);
        printf("%6d -> '%.*s'\n", ids[i], (int)len, s);
    }
    free(ids);
    tok_free(tok);
    gguf_close(&g);
    return 0;
}

/* each stdin line ("\n" and "\\" unescaped) -> one line of token ids
 * (no BOS; control-token text is parsed only with --special) */
static int cmd_tokenize_ids(int argc, char **argv)
{
    int special = 0;
    const char *path = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--special"))
            special = 1;
        else
            path = argv[i];
    }
    Gguf g;
    gguf_open(&g, path ? path : default_model());
    Tokenizer *tok = tok_load(&g);
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    while ((n = getline(&line, &cap, stdin)) > 0) {
        if (line[n - 1] == '\n')
            line[--n] = 0;
        char *w = line;
        for (char *r = line; *r; r++) {
            if (r[0] == '\\' && (r[1] == 'n' || r[1] == '\\'))
                *w++ = *++r == 'n' ? '\n' : '\\';
            else
                *w++ = *r;
        }
        *w = 0;
        int32_t *ids;
        int k = tok_encode(tok, line, special, &ids);
        for (int i = 0; i < k; i++)
            printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
        free(ids);
    }
    free(line);
    tok_free(tok);
    gguf_close(&g);
    return 0;
}

/* Whole file as a NUL-terminated string (exits if unreadable). */
static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc(n + 1);
    if (fread(s, 1, n, f) != (size_t)n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    s[n] = 0;
    fclose(f);
    return s;
}

/* `cache-test [model.gguf]`: prefill() must give the same logits, bit for
 * bit, as a fresh computation when it reuses the cache: after a full
 * recompute (no snapshot), from the ring snapshot, and after a short rewind
 * inside the rings. Uses tale.txt (twice) as text and the GEMM path for
 * every chunk, so all runs do identical arithmetic per token. */
static int cmd_cache_test(const char *model_path, const char *cubin)
{
    init_cuda(cubin);
    Gguf g;
    gguf_open(&g, model_path);
    Tokenizer *tok = tok_load(&g);
    Model m = {0};
    const unsigned batch = 64;
    model_load(&m, &g, 4096, batch);
    force_gemm = 1;
    char *tale = read_file("tale.txt");
    size_t tl = strlen(tale);
    char *text = malloc(2 * tl + 1);
    memcpy(text, tale, tl);
    memcpy(text + tl, tale, tl + 1);
    int32_t *body;
    unsigned nb = tok_encode(tok, text, 0, &body), nv = m.c.n_vocab;
    const unsigned P = 1300, R = 300; /* "prompt" past the 1024 window, "reply" */
    if (nb < P + R + 1) {
        fprintf(stderr, "tale.txt is too short\n");
        return 1;
    }
    int32_t *a = malloc((P + R) * sizeof(int32_t)), *b = malloc((P + R) * sizeof(int32_t));
    a[0] = tok_special(tok, "<bos>");
    memcpy(a + 1, body, (P + R - 1) * sizeof(int32_t));
    memcpy(b, a, (P + R) * sizeof(int32_t));
    for (unsigned i = P + 5; i < P + R; i++) /* b: a with the reply changed from P + 5 on */
        b[i] = body[(i * 7) % nb];
    float *ref = malloc(nv * 4), *got = malloc(nv * 4);
    int fails = 0;
#define FRESH(seq, n)                                                         \
    do {                                                                      \
        m.n_cache = 0, m.has_snap = 0;                                        \
        prefill(&m, seq, n);                                                  \
        CHECK(cuMemcpyDtoH(ref, m.logits, nv * 4));                           \
    } while (0)
#define CHECK_SAME(name)                                                      \
    do {                                                                      \
        CHECK(cuMemcpyDtoH(got, m.logits, nv * 4));                           \
        int same = !memcmp(ref, got, nv * 4);                                 \
        printf("   %-56s %s\n", name, same ? "PASS" : "FAIL");                \
        fails += !same;                                                       \
    } while (0)
    /* 1. rewind far back without a snapshot: recompute from position 0 */
    FRESH(b, P + R);
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P + R);
    prefill(&m, b, P + R);
    CHECK_SAME("rewind 295 positions, no snapshot (recompute)");
    /* 2. same rewind with a snapshot taken after the prompt */
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P);
    cache_snapshot(&m);
    prefill(&m, a, P + R);
    prefill(&m, b, P + R);
    CHECK_SAME("rewind 295 positions to the ring snapshot");
    /* 3. short rewind (< batch) inside the rings */
    unsigned cut = P + R - 40;
    memcpy(b, a, (P + R) * sizeof(int32_t));
    for (unsigned i = cut; i < P + R; i++)
        b[i] = body[(i * 11) % nb];
    FRESH(b, P + R);
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P + R);
    prefill(&m, b, P + R);
    CHECK_SAME("rewind 40 positions inside the rings");
    /* 4. plain extension of the cache */
    FRESH(a, P + R);
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P);
    prefill(&m, a, P + R);
    CHECK_SAME("extend a cached prompt by 300 tokens");
#undef FRESH
#undef CHECK_SAME
    printf("%s\n", fails ? "CACHE TEST FAILED" : "cache reuse is exact");
    return fails;
}

/* `render-ids [--sys TEXT] USER [MODEL USER ...]`: the token ids of a
 * conversation in the chat format, ready for the model's reply (tests) */
static int cmd_render_ids(int argc, char **argv)
{
    const char *system = NULL;
    int i = 2;
    if (argc > 3 && !strcmp(argv[2], "--sys")) {
        system = argv[3];
        i = 4;
    }
    int n = argc - i;
    ChatTurn *turns = malloc((n > 0 ? n : 1) * sizeof(ChatTurn));
    for (int k = 0; k < n; k++)
        turns[k] = (ChatTurn){k % 2 == 0, argv[i + k]};
    Gguf g;
    gguf_open(&g, default_model());
    Tokenizer *tok = tok_load(&g);
    int32_t *ids;
    int m = chat_render(tok, system, turns, n, &ids);
    for (int k = 0; k < m; k++)
        printf(k ? " %d" : "%d", ids[k]);
    printf("\n");
    free(ids), free(turns);
    tok_free(tok);
    gguf_close(&g);
    return 0;
}

static int cmd_quantize(int argc, char **argv)
{
    const char *in = argc >= 3 ? argv[2] : BF16_MODEL, *out = argc >= 4 ? argv[3] : DEFAULT_MODEL;
    Gguf g;
    gguf_open(&g, in);
    fprintf(stderr, "quantizing %s -> %s (Q5_0 feed-forward, Q8_0 attention and embeddings)\n",
            in, out);
    double t0 = now_ms();
    int r = quantize_model(&g, out);
    if (!r)
        fprintf(stderr, "done in %.0f s\n", (now_ms() - t0) / 1e3);
    gguf_close(&g);
    return r;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [options] PROMPT        continue PROMPT as plain text (-f FILE: from FILE)\n"
            "       %s [options] -chat MESSAGE  one chat reply\n"
            "       %s [options] -i             interactive chat (one message per line)\n"
            "       %s quantize [IN] [OUT]      BF16 GGUF -> Q8_0/Q5_0 GGUF (default\n"
            "                                   " BF16_MODEL " -> " DEFAULT_MODEL ")\n"
            "       %s test | cache-test        kernel self-tests | KV-cache reuse test\n"
            "       %s tokenize TEXT [model.gguf] | tokenize-ids [--special] [model.gguf] < lines\n"
            "       %s render-ids [--sys TEXT] USER [MODEL USER ...]   chat-format token ids\n"
            "options:\n"
            "  -m FILE        model (default " DEFAULT_MODEL ", else " BF16_MODEL ")\n"
            "  -n N           max new tokens (default: 64 for PROMPT, until done for chat)\n"
            "  -c N           context length (default 8192)\n"
            "  -b N           prompt chunk size (default: largest that fits, up to 128)\n"
            "  -k FILE        kernels (default " DEFAULT_CUBIN ")\n"
            "  --temp T       temperature (default 1.0; 0 = greedy)\n"
            "  --top-k K      keep the K most likely tokens (default 64; 0 = off)\n"
            "  --top-p P      nucleus sampling (default 0.95; 1 = off)\n"
            "  --min-p P      drop tokens below P x the most likely (default 0 = off)\n"
            "  --seed S       sampling seed (default: time)\n"
            "  --sys TEXT     system prompt for chat\n"
            "  --print-prompt with -chat: print the rendered prompt and its tokens, then exit\n",
            prog, prog, prog, prog, prog, prog, prog);
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "tokenize"))
        return cmd_tokenize(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "tokenize-ids"))
        return cmd_tokenize_ids(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "quantize"))
        return cmd_quantize(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "render-ids"))
        return cmd_render_ids(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "cache-test"))
        return cmd_cache_test(argc >= 3 ? argv[2] : default_model(), DEFAULT_CUBIN);

    const char *model_path = NULL, *cubin = DEFAULT_CUBIN, *prompt = NULL, *chat = NULL;
    unsigned ctx = 8192, batch = 0; /* 0 = pick from free memory */
    uint64_t seed = 0;
    int test = 0, interactive = 0, print_prompt = 0;
    GenOptions o = {.sampler = {.temperature = 1.0f, .top_k = 64, .top_p = 0.95f}, .max_new = -1};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
#define OPT(name) (!strcmp(a, name) && v && ++i)
        if (OPT("-m"))
            model_path = v;
        else if (OPT("-n"))
            o.max_new = atoi(v);
        else if (OPT("-c"))
            ctx = atoi(v);
        else if (OPT("-b"))
            batch = atoi(v);
        else if (OPT("-k"))
            cubin = v;
        else if (OPT("--temp"))
            o.sampler.temperature = atof(v);
        else if (OPT("--top-k"))
            o.sampler.top_k = atoi(v);
        else if (OPT("--top-p"))
            o.sampler.top_p = atof(v);
        else if (OPT("--min-p"))
            o.sampler.min_p = atof(v);
        else if (OPT("--seed"))
            seed = strtoull(v, NULL, 10);
        else if (OPT("--sys"))
            o.system = v;
        else if (OPT("-chat"))
            chat = v;
        else if (OPT("-f"))
            prompt = read_file(v);
        else if (!strcmp(a, "-i"))
            interactive = 1;
        else if (!strcmp(a, "--print-prompt"))
            print_prompt = 1;
        else if (!strcmp(a, "test"))
            test = 1;
        else if (a[0] == '-' || prompt)
            usage(argv[0]);
        else
            prompt = a;
#undef OPT
    }
    if (test) {
        init_cuda(cubin);
        return self_test() ? 1 : 0;
    }
    if ((!!prompt + !!chat + interactive) != 1 || ctx < 16 || batch > MAX_BATCH)
        usage(argv[0]);
    if (!model_path)
        model_path = default_model();
    if (print_prompt && chat) { /* no GPU needed */
        Gguf g;
        gguf_open(&g, model_path);
        Tokenizer *tok = tok_load(&g);
        ChatTurn turn = {1, chat};
        int32_t *ids;
        int n = chat_render(tok, o.system, &turn, 1, &ids);
        for (int i = 0; i < n; i++) {
            size_t len;
            const char *s = tok_piece(tok, ids[i], &len);
            fwrite(s, 1, len, stdout);
        }
        printf("\n");
        for (int i = 0; i < n; i++)
            printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
        free(ids);
        tok_free(tok);
        gguf_close(&g);
        return 0;
    }
    if (!seed)
        seed = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
    o.sampler.rng = seed | 1;
    if (o.sampler.temperature > 0)
        fprintf(stderr, "sampling: temp %.2f, top-k %d, top-p %.2f, min-p %.2f, seed %llu\n",
                o.sampler.temperature, o.sampler.top_k, o.sampler.top_p, o.sampler.min_p,
                (unsigned long long)seed);
    dump_enabled = getenv("GEMMA3_DUMP") && atoi(getenv("GEMMA3_DUMP"));
    no_reuse = getenv("GEMMA3_NO_REUSE") && atoi(getenv("GEMMA3_NO_REUSE"));

    init_cuda(cubin);
    Gguf g;
    gguf_open(&g, model_path);
    Tokenizer *tok = tok_load(&g);
    Model m = {0};
    read_config(&g, &m.c);
    if (ctx > m.c.max_ctx) {
        fprintf(stderr, "context %u is larger than the model's %u\n", ctx, m.c.max_ctx);
        return 1;
    }
    model_load(&m, &g, ctx, batch);

    if (prompt) {
        run_completion(&m, tok, prompt, &o);
    } else if (chat) {
        ChatTurn turn = {1, chat};
        free(chat_reply(&m, tok, &turn, 1, &o));
    } else {
        run_interactive(&m, tok, &o);
    }
    tok_free(tok);
    gguf_close(&g);
    return 0;
}
