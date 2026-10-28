/*
 * gptoss.c - gpt-oss inference on the GPU with hand-written PTX kernels
 * (gptoss.ptx) driven through the CUDA Driver API. No other libraries.
 *
 * The model is read from a GGUF file (gguf.c), text is tokenized with the
 * o200k byte-level BPE stored in it (tokenizer.c), and every layer runs in
 * the PTX kernels; the host only looks up token embeddings and launches.
 *
 * Weights on the GPU:
 *   experts (gate/up/down)      MXFP4 as stored, repacked into separate
 *                               nibble and scale arrays (9.5 GiB)
 *   attention q/k/v/o, output   F16 in the file, converted to Q8 (int8 +
 *                               f16 scale per 32) at load time (1.2 GiB)
 *   norms, biases, router, sinks  F32 as stored
 *   token embeddings            stay in host memory (one row per token)
 * so the 20B model fits in a 12 GB card next to a desktop.
 *
 * The prompt is processed in chunks of up to -b tokens (by default the
 * largest of 128, 64, 32, ... that fits in GPU memory) with
 * tiled GEMM kernels that read each weight once per chunk; generated tokens
 * then go one at a time through matrix-vector kernels. Keys and values live
 * in an f16 KV cache: full-attention layers keep every position,
 * sliding-window layers a ring of the last sliding_window + batch positions.
 *
 * Usage (see usage() for every option):
 *   ./gptoss_demo [options] PROMPT        continuation of PROMPT as plain text;
 *                                         control tokens such as "<|start|>"
 *                                         in PROMPT are recognised
 *   ./gptoss_demo [options] -chat MESSAGE one reply in the harmony chat format
 *   ./gptoss_demo [options] -i            interactive multi-turn chat
 *   ./gptoss_demo test                    kernel self-tests against the CPU
 *   ./gptoss_demo tokenize TEXT           show the tokens of TEXT
 *   ./gptoss_demo tokenize-ids            one line of ids per stdin line
 * Sampling defaults to OpenAI's recommendation for gpt-oss (temperature 1,
 * no truncation); --temp 0 is greedy and reproduces llama.cpp exactly.
 * A chat keeps its KV cache between turns and recomputes only what changed
 * (earlier replies are re-rendered without their reasoning).
 * Environment (debugging): GPTOSS_DUMP=1 prints a summary of every
 * intermediate tensor (named like llama.cpp's llama-eval-callback) and the
 * top logits; GPTOSS_NO_REUSE=1 recomputes every chat turn from scratch
 * instead of reusing the KV cache.
 */
#include <cuda.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "gguf.h"
#include "harmony.h"
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

#define DEFAULT_MODEL "gpt-oss-20b-F16.gguf"
#define DEFAULT_CUBIN "gptoss.cubin"
#define HEAD_DIM 64          /* rope_kv and attention assume 64-dim heads */
#define MAX_EXPERTS 32       /* router_topk uses one lane per expert */
#define CAND_CAP 16384       /* sampling candidates gathered on the GPU */
#define AUTO_MAX_BATCH 128   /* default prompt chunk: the largest power of 2 up to */
#define AUTO_SPARE_BYTES (8u << 20) /* this that leaves this much GPU memory free */
#define MAX_BATCH 256
#define SWIGLU_ALPHA 1.702f  /* gpt-oss clamped SwiGLU constants */
#define SWIGLU_LIMIT 7.0f
#define YARN_BETA_FAST 32.0f
#define YARN_BETA_SLOW 1.0f

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
/* IEEE half precision on the host (the CPU has no F16C)                     */

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    float f;
    if (exp == 0x1f) {
        bits = sign | 0x7f800000 | man << 13;
    } else if (exp) {
        bits = sign | (exp + 112) << 23 | man << 13;
    } else { /* zero / subnormal: man * 2^-24 */
        f = man * (1.0f / 16777216.0f);
        memcpy(&bits, &f, 4);
        bits |= sign;
    }
    memcpy(&f, &bits, 4);
    return f;
}

/* Round to nearest even. */
static uint16_t float_to_half(float f)
{
    uint32_t x, ax;
    memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    ax = x & 0x7fffffff;
    if (ax > 0x7f800000)
        return sign | 0x7e00; /* NaN */
    if (ax >= 0x477ff000)
        return sign | 0x7c00; /* rounds to inf */
    if (ax < 0x38800000) {    /* below 2^-14: subnormal, value * 2^24 */
        float v;
        memcpy(&v, &ax, 4);
        return sign | (uint16_t)lrintf(v * 16777216.0f);
    }
    return sign | (uint16_t)((ax + 0xc8000fffu + ((ax >> 13) & 1)) >> 13);
}

static float half_lut[65536];

/* ------------------------------------------------------------------------ */
/* Host-side weight formats                                                  */

/* F16 rows -> Q8: per block of 32, d = max|w| / 127, q = round(w / d), the
 * scale stored as f16 (same as ggml's quantize_row_q8_0). */
static void quantize_rows_q8(const uint16_t *src, size_t nrows, unsigned cols, int8_t *q,
                             uint16_t *d)
{
    for (size_t i = 0; i < nrows * cols / 32; i++) {
        float v[32], amax = 0;
        for (int j = 0; j < 32; j++) {
            v[j] = half_lut[src[i * 32 + j]];
            amax = fmaxf(amax, fabsf(v[j]));
        }
        float s = amax / 127, inv = s ? 1 / s : 0;
        for (int j = 0; j < 32; j++)
            q[i * 32 + j] = (int8_t)roundf(v[j] * inv);
        d[i] = float_to_half(s);
    }
}

typedef struct {
    const uint16_t *src;
    size_t nrows;
    unsigned cols;
    int8_t *q;
    uint16_t *d;
} QuantJob;

static void *quant_worker(void *arg)
{
    QuantJob *j = arg;
    quantize_rows_q8(j->src, j->nrows, j->cols, j->q, j->d);
    return NULL;
}

/* quantize_rows_q8 split over all CPU cores */
static void quantize_rows_q8_mt(const uint16_t *src, size_t nrows, unsigned cols, int8_t *q,
                                uint16_t *d)
{
    enum { MAX_THREADS = 64 };
    long nt = sysconf(_SC_NPROCESSORS_ONLN);
    nt = nt < 1 ? 1 : nt > MAX_THREADS ? MAX_THREADS : nt;
    pthread_t th[MAX_THREADS];
    QuantJob jobs[MAX_THREADS];
    size_t per = (nrows + nt - 1) / nt;
    int started = 0;
    for (long t = 0; t < nt; t++) {
        size_t r0 = t * per, nr = r0 >= nrows ? 0 : nrows - r0 < per ? nrows - r0 : per;
        if (!nr)
            break;
        jobs[t] = (QuantJob){src + r0 * cols, nr, cols, q + r0 * cols, d + r0 * cols / 32};
        pthread_create(&th[t], NULL, quant_worker, &jobs[t]);
        started++;
    }
    for (int t = 0; t < started; t++)
        pthread_join(th[t], NULL);
}

/* E2M1 code -> 2x value, as in ggml's kvalues_mxfp4 */
static const int8_t fp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/* E8M0 scale byte -> 2^(e-128), i.e. half the scale to undo the doubled codes */
static float e8m0_half(uint8_t e)
{
    return ldexpf(1.0f, (int)e - 128);
}

/* ------------------------------------------------------------------------ */
/* Kernel handles and launch helpers                                         */

static CUfunction k_rmsnorm, k_quant, k_mv_q8, k_mv_mxfp4, k_mv_f32, k_topk, k_rope, k_attn,
    k_swiglu, k_combine, k_argmax, k_gemm_q8, k_gemm_mxfp4, k_moe_sort, k_compact, k_sample;
static unsigned max_blocks; /* cap for grid-stride kernels: 8 blocks per SM */

static void load_kernels(const char *cubin)
{
    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, cubin);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cannot load %s (run `make` in the gptoss directory)\n", cubin);
        exit(1);
    }
    struct { CUfunction *fn; const char *name; } k[] = {
        {&k_rmsnorm, "rmsnorm"}, {&k_quant, "quant_q8"}, {&k_mv_q8, "mv_q8"},
        {&k_mv_mxfp4, "mv_mxfp4"}, {&k_mv_f32, "mv_f32"}, {&k_topk, "router_topk"},
        {&k_rope, "rope_kv"}, {&k_attn, "attention"}, {&k_swiglu, "swiglu_oai"},
        {&k_combine, "moe_combine"}, {&k_argmax, "argmax"}, {&k_gemm_q8, "gemm_q8"},
        {&k_gemm_mxfp4, "gemm_mxfp4"}, {&k_moe_sort, "moe_sort"}, {&k_compact, "compact_logits"},
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

/* MXFP4 expert matrices: qs [E][rows][cols/2], sc [E][rows][cols/32], bias f32 [E][rows] */
typedef struct {
    CUdeviceptr qs, sc, bias;
    unsigned rows, cols, n_expert;
} MxW;

/* each of the T rows of x [T][n] */
static void rmsnorm(CUdeviceptr x, CUdeviceptr w, CUdeviceptr y, unsigned n, float eps,
                    unsigned T)
{
    void *args[] = {&x, &w, &y, &n, &eps};
    CHECK(cuLaunchKernel(k_rmsnorm, T, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
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

/* y[s] = W[ids[s]] . x_s + bias[ids[s]] for s < nslots */
static void mv_mxfp4(const MxW *w, CUdeviceptr ids, unsigned nslots, CUdeviceptr xq,
                     CUdeviceptr xd, unsigned xstride, CUdeviceptr y)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->qs, (void *)&w->sc, &ids, &xq, &xd, &xstride,
                    (void *)&w->bias, &y, &rows, &cols};
    CHECK(cuLaunchKernel(k_mv_mxfp4, (rows + 7) / 8, nslots, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* y[t] = W . x[t] + bias for each of T tokens */
static void mv_f32(CUdeviceptr W, CUdeviceptr x, CUdeviceptr bias, CUdeviceptr y, unsigned rows,
                   unsigned cols, unsigned T)
{
    void *args[] = {&W, &x, &bias, &y, &rows, &cols};
    CHECK(cuLaunchKernel(k_mv_f32, (rows + 7) / 8, T, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static void router_topk(CUdeviceptr logits, CUdeviceptr ids, CUdeviceptr w, unsigned n,
                        unsigned k, unsigned T)
{
    void *args[] = {&logits, &ids, &w, &n, &k};
    CHECK(cuLaunchKernel(k_topk, T, 1, 1, 32, 1, 1, 0, NULL, args, NULL));
}

/* tokens at positions pos0 .. pos0+T-1, cached in slot pos % slots */
static void rope_kv(CUdeviceptr qkv, CUdeviceptr cs, CUdeviceptr kc, CUdeviceptr vc,
                    unsigned pos0, unsigned slots, unsigned nh, unsigned nkv, unsigned T)
{
    void *args[] = {&qkv, &cs, &kc, &vc, &pos0, &slots, &nh, &nkv};
    unsigned total = (nh + nkv) * 32 + nkv * HEAD_DIM;
    CHECK(cuLaunchKernel(k_rope, (total + 255) / 256, T, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* causal attention for T query tokens at pos0..; window = 0 for full layers */
static void attention(CUdeviceptr q, unsigned q_stride, CUdeviceptr kc, CUdeviceptr vc,
                      CUdeviceptr sinks, CUdeviceptr out, unsigned pos0, unsigned window,
                      unsigned slots, unsigned nh, unsigned nkv, float scale, unsigned T)
{
    void *args[] = {&q, &q_stride, &kc, &vc, &sinks, &out, &pos0, &window, &slots,
                    &nh, &nkv, &scale};
    unsigned max_keys = pos0 + T;
    if (window && max_keys > window)
        max_keys = window;
    CHECK(cuLaunchKernel(k_attn, nh, T, 1, 128, 1, 1, max_keys * 4, NULL, args, NULL));
}

static void swiglu(CUdeviceptr g, CUdeviceptr u, CUdeviceptr out, unsigned n)
{
    float alpha = SWIGLU_ALPHA, limit = SWIGLU_LIMIT;
    void *args[] = {&g, &u, &out, &n, &alpha, &limit};
    launch_1d(k_swiglu, n, args);
}

/* y[t] = resid[t] + sum_j w[t][j] d[t][j] for T tokens of k experts */
static void moe_combine(CUdeviceptr d, CUdeviceptr w, CUdeviceptr resid, CUdeviceptr y,
                        unsigned n, unsigned k, unsigned T)
{
    void *args[] = {&d, &w, &resid, &y, &n, &k, &T};
    launch_1d(k_combine, n * T, args);
}

/* Y[t] (row stride ldy) = W . X[t] + bias + resid[t] for T tokens */
static void gemm_q8(const Q8W *w, CUdeviceptr X, CUdeviceptr bias, CUdeviceptr resid,
                    CUdeviceptr Y, unsigned ldy, unsigned T)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->q, (void *)&w->d, &X, &bias, &resid, &Y, &rows, &cols, &T, &ldy};
    CHECK(cuLaunchKernel(k_gemm_q8, rows / 64, (T + 31) / 32, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* buckets the n routed (token, slot) pairs by expert into offs / list */
static void moe_sort(CUdeviceptr ids, unsigned n, unsigned n_expert, CUdeviceptr offs,
                     CUdeviceptr list)
{
    void *args[] = {&ids, &n, &n_expert, &offs, &list};
    CHECK(cuLaunchKernel(k_moe_sort, 1, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
}

/* Y[p] = W[expert of p] . X[p / xdiv] + bias for every routed pair p of T
 * tokens (each expert gets at most T pairs) */
static void gemm_mxfp4(const MxW *w, CUdeviceptr X, unsigned xdiv, CUdeviceptr list,
                       CUdeviceptr offs, CUdeviceptr Y, unsigned T)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->qs, (void *)&w->sc, (void *)&w->bias, &X, &xdiv, &list,
                    &offs, &Y, &rows, &cols};
    CHECK(cuLaunchKernel(k_gemm_mxfp4, rows / 64, (T + 31) / 32, w->n_expert, 256, 1, 1, 0,
                         NULL, args, NULL));
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
    unsigned n_layer, n_embd, n_head, n_kv_head, head_dim, n_expert, n_expert_used, n_ff;
    unsigned n_vocab, sliding_window, rope_orig_ctx;
    float eps, rope_base, rope_factor;
} Config;

typedef struct {
    CUdeviceptr attn_norm, post_norm, sinks;
    Q8W wq, wk, wv, wo;
    CUdeviceptr bq, bk, bv, bo;
    CUdeviceptr router_w, router_b; /* f32 [n_expert][n_embd], [n_expert] */
    MxW gate, up, down;
    CUdeviceptr kc, vc; /* f16 [slots][n_kv_head][64] */
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
    const uint16_t *embd;    /* f16 [n_vocab][n_embd] in the mmap'ed file */
    CUdeviceptr rope_cs;     /* f32 [ctx][head_dim/2][2] */
    /* activations, [batch][...] (decoding uses row 0) */
    CUdeviceptr x, xn, xq, xd, qkv, att, rlogits, ids, ew, gate, up, down, logits, next;
    CUdeviceptr offs, list;  /* expert buckets of the routed (token, slot) pairs */
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

static void read_config(const Gguf *g, Config *c)
{
    const GgufKV *arch = gguf_kv(g, "general.architecture");
    uint64_t len;
    const uint8_t *p = arch ? arch->val : NULL;
    if (!arch || strncmp(gguf_str(&p, &len), "gpt-oss", 7)) {
        fprintf(stderr, "model architecture is not gpt-oss\n");
        exit(1);
    }
    c->n_layer = gguf_u32(g, "gpt-oss.block_count");
    c->n_embd = gguf_u32(g, "gpt-oss.embedding_length");
    c->n_head = gguf_u32(g, "gpt-oss.attention.head_count");
    c->n_kv_head = gguf_u32(g, "gpt-oss.attention.head_count_kv");
    c->head_dim = gguf_u32(g, "gpt-oss.attention.key_length");
    c->n_expert = gguf_u32(g, "gpt-oss.expert_count");
    c->n_expert_used = gguf_u32(g, "gpt-oss.expert_used_count");
    c->n_ff = gguf_u32(g, "gpt-oss.expert_feed_forward_length");
    c->sliding_window = gguf_u32(g, "gpt-oss.attention.sliding_window");
    c->rope_orig_ctx = gguf_u32(g, "gpt-oss.rope.scaling.original_context_length");
    c->eps = gguf_f32(g, "gpt-oss.attention.layer_norm_rms_epsilon");
    c->rope_base = gguf_f32(g, "gpt-oss.rope.freq_base");
    c->rope_factor = gguf_f32(g, "gpt-oss.rope.scaling.factor");
    c->n_vocab = (unsigned)gguf_kv(g, "tokenizer.ggml.tokens")->arr_n;
    if (c->head_dim != HEAD_DIM || c->n_expert > MAX_EXPERTS || c->n_head % c->n_kv_head ||
        c->n_embd % 32 || c->n_ff % 32 || c->n_head * c->head_dim % 32 ||
        c->n_expert_used > c->n_expert) {
        fprintf(stderr, "unsupported gpt-oss hyperparameters\n");
        exit(1);
    }
}

/* cos/sin for every position and rotary pair, as ggml's rope with YaRN
 * (ext_factor 1, beta_fast 32, beta_slow 1) and llama.cpp's attn_factor. */
static void rope_table(const Config *c, unsigned ctx, float *cs)
{
    const int n_dims = c->head_dim;
    const float factor = c->rope_factor, freq_scale = 1.0f / factor;
    /* llama.cpp: attn_factor = get_mscale(factor) / (1 + 0.1 ln factor), and
     * rope_yarn then multiplies by (1 + 0.1 ln factor) again */
    const float attn_factor = (0.1f * logf(factor) + 1.0f) / (1.0f + 0.1f * logf(factor));
    const float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    float corr[2];
    for (int i = 0; i < 2; i++) {
        float beta = i ? YARN_BETA_SLOW : YARN_BETA_FAST;
        float dim = n_dims * logf(c->rope_orig_ctx / (beta * 2 * (float)M_PI)) /
                    (2 * logf(c->rope_base));
        corr[i] = i ? fminf(n_dims - 1, ceilf(dim)) : fmaxf(0, floorf(dim));
    }
    const float theta_scale = powf(c->rope_base, -2.0f / n_dims);
    for (unsigned p = 0; p < ctx; p++) {
        float theta_extrap = (float)p;
        for (int i0 = 0; i0 < n_dims; i0 += 2) {
            float theta_interp = freq_scale * theta_extrap;
            float y = (i0 / 2 - corr[0]) / fmaxf(0.001f, corr[1] - corr[0]);
            float ramp_mix = 1 - fminf(1, fmaxf(0, y)); /* times ext_factor = 1 */
            float theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            cs[(size_t)p * n_dims + i0] = cosf(theta) * mscale;
            cs[(size_t)p * n_dims + i0 + 1] = sinf(theta) * mscale;
            theta_extrap *= theta_scale;
        }
    }
}

/* --- loading --- */

static void *staging;                    /* pinned host buffer for uploads */
static const size_t STAGING_BYTES = 64 << 20;
/* All model memory is carved out of one cuMemAlloc'ed arena: hundreds of
 * separate allocations would waste ~150 MiB, which this card can't spare. */
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

static CUdeviceptr upload_f32(const Gguf *g, const char *name, uint64_t ne0, uint64_t ne1)
{
    const GgufTensor *t = gguf_tensor(g, name, GGML_F32, ne0, ne1, 0);
    CUdeviceptr p = dalloc(t->nbytes);
    CHECK(cuMemcpyHtoD(p, t->data, t->nbytes));
    return p;
}

/* F16 [rows][cols] -> Q8 on the GPU, converted in staging-sized row chunks */
static Q8W upload_q8(const Gguf *g, const char *name, unsigned cols, unsigned rows)
{
    const GgufTensor *t = gguf_tensor(g, name, GGML_F16, cols, rows, 0);
    const uint16_t *src = (const uint16_t *)t->data;
    unsigned nb = cols / 32;
    Q8W w = {dalloc((size_t)rows * cols), dalloc((size_t)rows * nb * 2), rows, cols};
    size_t chunk = STAGING_BYTES / (cols + nb * 2);
    for (size_t r0 = 0; r0 < rows; r0 += chunk) {
        size_t nr = rows - r0 < chunk ? rows - r0 : chunk;
        int8_t *q = staging;
        uint16_t *d = (uint16_t *)(q + nr * cols);
        quantize_rows_q8_mt(src + r0 * cols, nr, cols, q, d);
        CHECK(cuMemcpyHtoD(w.q + r0 * cols, q, nr * cols));
        CHECK(cuMemcpyHtoD(w.d + r0 * nb * 2, d, nr * nb * 2));
    }
    return w;
}

/* MXFP4 [E][rows][cols] (17-byte blocks: scale, 16 code bytes) -> separate
 * code and scale arrays so the kernel can use aligned 16-byte loads. */
static MxW upload_mxfp4(const Gguf *g, const char *wname, const char *bname, unsigned cols,
                        unsigned rows, unsigned n_expert)
{
    const GgufTensor *t = gguf_tensor(g, wname, GGML_MXFP4, cols, rows, n_expert);
    size_t nb = cols / 32, total_rows = (size_t)rows * n_expert;
    MxW w = {dalloc(total_rows * nb * 16), dalloc(total_rows * nb), 0, rows, cols, n_expert};
    w.bias = upload_f32(g, bname, rows, n_expert);
    size_t chunk = STAGING_BYTES / (nb * 17);
    for (size_t r0 = 0; r0 < total_rows; r0 += chunk) {
        size_t nr = total_rows - r0 < chunk ? total_rows - r0 : chunk;
        uint8_t *qs = staging, *sc = qs + nr * nb * 16;
        const uint8_t *src = t->data + r0 * nb * 17;
        for (size_t b = 0; b < nr * nb; b++) {
            sc[b] = src[b * 17];
            memcpy(qs + b * 16, src + b * 17 + 1, 16);
        }
        CHECK(cuMemcpyHtoD(w.qs + r0 * nb * 16, qs, nr * nb * 16));
        CHECK(cuMemcpyHtoD(w.sc + r0 * nb, sc, nr * nb));
    }
    return w;
}

/* KV cache slots of a layer. A sliding-window layer keeps a ring of the
 * last window + batch positions: a chunk writes all its keys before its
 * attention runs, and its first token must still find the window before it. */
static unsigned cache_slots(const Config *c, unsigned l, unsigned ctx, unsigned batch)
{
    unsigned ring = c->sliding_window + batch;
    return l % 2 == 0 && ring < ctx ? ring : ctx;
}

/* Bytes of activation buffers for `batch` tokens. */
static size_t activation_bytes(const Config *c, unsigned batch)
{
    size_t E = c->n_embd, Q = (size_t)c->n_head * c->head_dim, KV = (size_t)c->n_kv_head * c->head_dim;
    size_t K = c->n_expert_used, B = batch;
    size_t per_token = 2 * E + Q + 2 * KV + Q + c->n_expert + 3 * K + 2 * K * c->n_ff + K * E;
    size_t decode_q8 = K * c->n_ff * 2; /* int8 values + scales, generously */
    return B * per_token * 4 + decode_q8 + (c->n_expert + 1) * 4 + (size_t)c->n_vocab * 4 +
           (2 + 2 * CAND_CAP) * 4 + 8 + 64 * 256; /* alignment of the ~16 buffers */
}

/* Expected device bytes for weights + cache + activations. */
static size_t model_bytes(const Config *c, unsigned ctx, unsigned batch)
{
    size_t E = c->n_embd, Q = (size_t)c->n_head * c->head_dim, KV = (size_t)c->n_kv_head * c->head_dim;
    size_t q8 = (Q * E + 2 * KV * E + E * Q) * 34 / 32;
    size_t mx = 3 * (size_t)c->n_expert * c->n_ff * E * 17 / 32;
    size_t small = (4 * E + Q + 2 * KV + c->n_head + c->n_expert * (E + 1 + 2 * c->n_ff + E)) * 4;
    size_t cache = 0;
    for (unsigned l = 0; l < c->n_layer; l++)
        cache += (size_t)cache_slots(c, l, ctx, batch) * KV * 4;
    return c->n_layer * (q8 + mx + small) + cache + (size_t)c->n_vocab * E * 34 / 32 +
           (size_t)ctx * c->head_dim * 4 + activation_bytes(c, batch) + (8u << 20);
}

static void model_load(Model *m, const Gguf *g, unsigned ctx, unsigned batch)
{
    Config *c = &m->c;
    read_config(g, c);
    unsigned E = c->n_embd, Q = c->n_head * c->head_dim, KV = c->n_kv_head * c->head_dim;

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

    for (int i = 0; i < 65536; i++)
        half_lut[i] = half_to_float((uint16_t)i);
    CHECK(cuMemHostAlloc(&staging, STAGING_BYTES, 0));

    double t0 = now_ms();
    m->layer = calloc(c->n_layer, sizeof(Layer));
    char name[128], bname[128];
#define T(fmt) (snprintf(name, sizeof(name), "blk.%u." fmt, l), name)
#define TB(fmt) (snprintf(bname, sizeof(bname), "blk.%u." fmt, l), bname)
    for (unsigned l = 0; l < c->n_layer; l++) {
        Layer *L = &m->layer[l];
        L->attn_norm = upload_f32(g, T("attn_norm.weight"), E, 1);
        L->post_norm = upload_f32(g, T("post_attention_norm.weight"), E, 1);
        L->sinks = upload_f32(g, T("attn_sinks.weight"), c->n_head, 1);
        L->wq = upload_q8(g, T("attn_q.weight"), E, Q);
        L->wk = upload_q8(g, T("attn_k.weight"), E, KV);
        L->wv = upload_q8(g, T("attn_v.weight"), E, KV);
        L->wo = upload_q8(g, T("attn_output.weight"), Q, E);
        L->bq = upload_f32(g, T("attn_q.bias"), Q, 1);
        L->bk = upload_f32(g, T("attn_k.bias"), KV, 1);
        L->bv = upload_f32(g, T("attn_v.bias"), KV, 1);
        L->bo = upload_f32(g, T("attn_output.bias"), E, 1);
        L->router_w = upload_f32(g, T("ffn_gate_inp.weight"), E, c->n_expert);
        L->router_b = upload_f32(g, T("ffn_gate_inp.bias"), c->n_expert, 1);
        L->gate = upload_mxfp4(g, T("ffn_gate_exps.weight"), TB("ffn_gate_exps.bias"), E,
                               c->n_ff, c->n_expert);
        L->up = upload_mxfp4(g, T("ffn_up_exps.weight"), TB("ffn_up_exps.bias"), E, c->n_ff,
                             c->n_expert);
        L->down = upload_mxfp4(g, T("ffn_down_exps.weight"), TB("ffn_down_exps.bias"), c->n_ff,
                               E, c->n_expert);
        /* llama.cpp's gpt-oss uses sliding-window attention on even layers */
        L->swa = l % 2 == 0;
        L->slots = cache_slots(c, l, ctx, batch);
        L->kc = dalloc((size_t)L->slots * KV * 2);
        L->vc = dalloc((size_t)L->slots * KV * 2);
        fprintf(stderr, "\rloading layer %u/%u  (%.2f GiB on GPU, %.1f s)", l + 1, c->n_layer,
                dev_bytes / 1073741824.0, (now_ms() - t0) / 1e3);
    }
#undef T
#undef TB
    m->out_norm = upload_f32(g, "output_norm.weight", E, 1);
    m->out = upload_q8(g, "output.weight", E, c->n_vocab);
    m->embd = (const uint16_t *)gguf_tensor(g, "token_embd.weight", GGML_F16, E, c->n_vocab, 0)->data;

    float *cs = malloc((size_t)ctx * c->head_dim * sizeof(float));
    rope_table(c, ctx, cs);
    m->rope_cs = dalloc((size_t)ctx * c->head_dim * 4);
    CHECK(cuMemcpyHtoD(m->rope_cs, cs, (size_t)ctx * c->head_dim * 4));
    free(cs);

    size_t B = batch, K = c->n_expert_used;
    m->x = dalloc(B * E * 4);
    m->xn = dalloc(B * E * 4);
    m->xq = dalloc(K * c->n_ff);          /* decoding: int8 input of an expert matrix */
    m->xd = dalloc(K * c->n_ff / 32 * 4);
    m->qkv = dalloc(B * (Q + 2 * KV) * 4);
    m->att = dalloc(B * Q * 4);
    m->rlogits = dalloc(B * c->n_expert * 4);
    m->ids = dalloc(B * K * 4);
    m->ew = dalloc(B * K * 4);
    m->gate = dalloc(B * K * c->n_ff * 4);
    m->up = dalloc(B * K * c->n_ff * 4);
    m->down = dalloc(B * K * E * 4);
    m->offs = dalloc((c->n_expert + 1) * 4);
    m->list = dalloc(B * K * 4);
    m->logits = dalloc((size_t)c->n_vocab * 4);
    m->next = dalloc(4);
    m->cand = dalloc((2 + 2 * CAND_CAP) * 4);
    m->draw = dalloc(8);
    m->h_x = malloc(B * E * sizeof(float));
    m->cache_tok = malloc(ctx * sizeof(int32_t));
    CHECK(cuMemFreeHost(staging));
    fprintf(stderr, "\rloaded %u layers in %.1f s: %.2f GiB on the GPU, context %u, batch %u        \n",
           c->n_layer, (now_ms() - t0) / 1e3, dev_bytes / 1073741824.0, ctx, batch);
}

/* --- debug dumps (GPTOSS_DUMP=1) --- */

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

/* When set (the `routes` command), forward() stores each token's chosen
 * experts and gate weights here, as [position][layer][k]. */
static uint8_t *route_ids;
static float *route_w;
static uint32_t *route_tmp_ids;
static float *route_tmp_w;

/* Runs T (<= m->batch) tokens at positions pos0 .. pos0+T-1; with
 * want_logits, fills m->logits and m->next (argmax) for the last one.
 * A single token goes through the matrix-vector kernels; a chunk of several
 * goes through the tiled GEMMs, which read every weight once per chunk. Both
 * paths feed the experts the same Q8-rounded activations. Dumps show the
 * last token. */
static void forward(Model *m, const int32_t *tokens, unsigned T, unsigned pos0, int want_logits)
{
    const Config *c = &m->c;
    unsigned E = c->n_embd, Q = c->n_head * c->head_dim, KV = c->n_kv_head * c->head_dim;
    unsigned QKV = Q + 2 * KV, K = c->n_expert_used, F = c->n_ff, last = T - 1;
    float scale = 1.0f / sqrtf((float)c->head_dim);

    for (unsigned t = 0; t < T; t++) {
        const uint16_t *row = m->embd + (size_t)tokens[t] * E;
        for (unsigned i = 0; i < E; i++)
            m->h_x[(size_t)t * E + i] = half_lut[row[i]];
    }
    CHECK(cuMemcpyHtoD(m->x, m->h_x, (size_t)T * E * 4));

    for (unsigned l = 0; l < c->n_layer; l++) {
        const Layer *L = &m->layer[l];
        /* attention */
        rmsnorm(m->x, L->attn_norm, m->xn, E, c->eps, T);
        dump("attn_norm", l, m->xn + (size_t)last * E * 4, E);
        if (T == 1) {
            mv_q8(&L->wq, m->xn, L->bq, 0, m->qkv);
            mv_q8(&L->wk, m->xn, L->bk, 0, m->qkv + Q * 4);
            mv_q8(&L->wv, m->xn, L->bv, 0, m->qkv + (Q + KV) * 4);
        } else {
            gemm_q8(&L->wq, m->xn, L->bq, 0, m->qkv, QKV, T);
            gemm_q8(&L->wk, m->xn, L->bk, 0, m->qkv + Q * 4, QKV, T);
            gemm_q8(&L->wv, m->xn, L->bv, 0, m->qkv + (Q + KV) * 4, QKV, T);
        }
        rope_kv(m->qkv, m->rope_cs, L->kc, L->vc, pos0, L->slots, c->n_head, c->n_kv_head, T);
        CUdeviceptr qkv_last = m->qkv + (size_t)last * QKV * 4;
        dump("Qcur", l, qkv_last, Q);
        dump("Kcur", l, qkv_last + Q * 4, KV);
        dump("Vcur", l, qkv_last + (Q + KV) * 4, KV);
        attention(m->qkv, QKV, L->kc, L->vc, L->sinks, m->att, pos0,
                  L->swa ? c->sliding_window : 0, L->slots, c->n_head, c->n_kv_head, scale, T);
        dump("kqv_out", l, m->att + (size_t)last * Q * 4, Q);
        if (T == 1) /* x += Wo . att + bo */
            mv_q8(&L->wo, m->att, L->bo, m->x, m->x);
        else
            gemm_q8(&L->wo, m->att, L->bo, m->x, m->x, E, T);
        dump("ffn_inp", l, m->x + (size_t)last * E * 4, E);

        /* mixture of experts */
        rmsnorm(m->x, L->post_norm, m->xn, E, c->eps, T);
        dump("attn_post_norm", l, m->xn + (size_t)last * E * 4, E);
        mv_f32(L->router_w, m->xn, L->router_b, m->rlogits, c->n_expert, E, T);
        dump("ffn_moe_logits", l, m->rlogits + (size_t)last * c->n_expert * 4, c->n_expert);
        router_topk(m->rlogits, m->ids, m->ew, c->n_expert, K, T);
        if (route_ids) { /* recording expert routing: [position][layer][k] */
            CHECK(cuMemcpyDtoH(route_tmp_ids, m->ids, (size_t)T * K * 4));
            CHECK(cuMemcpyDtoH(route_tmp_w, m->ew, (size_t)T * K * 4));
            for (unsigned t = 0; t < T; t++)
                for (unsigned k = 0; k < K; k++) {
                    size_t o = ((size_t)(pos0 + t) * c->n_layer + l) * K + k;
                    route_ids[o] = (uint8_t)route_tmp_ids[t * K + k];
                    route_w[o] = route_tmp_w[t * K + k];
                }
        }
        if (dump_enabled) {
            uint32_t ids[MAX_EXPERTS];
            float w[MAX_EXPERTS];
            CHECK(cuMemcpyDtoH(ids, m->ids + (size_t)last * K * 4, K * 4));
            CHECK(cuMemcpyDtoH(w, m->ew + (size_t)last * K * 4, K * 4));
            printf("  %-22s", "experts");
            for (unsigned k = 0; k < K; k++)
                printf(" %2u (%.4f)", ids[k], w[k]);
            printf("\n");
        }
        if (T == 1) {
            quant_q8(m->xn, m->xq, m->xd, E);
            mv_mxfp4(&L->gate, m->ids, K, m->xq, m->xd, 0, m->gate);
            mv_mxfp4(&L->up, m->ids, K, m->xq, m->xd, 0, m->up);
            swiglu(m->gate, m->up, m->gate, K * F);
            quant_q8(m->gate, m->xq, m->xd, K * F);
            mv_mxfp4(&L->down, m->ids, K, m->xq, m->xd, F, m->down);
        } else {
            moe_sort(m->ids, T * K, c->n_expert, m->offs, m->list);
            quant_q8(m->xn, 0, 0, T * E); /* round in place, as the int8 path does */
            gemm_mxfp4(&L->gate, m->xn, K, m->list, m->offs, m->gate, T);
            gemm_mxfp4(&L->up, m->xn, K, m->list, m->offs, m->up, T);
            swiglu(m->gate, m->up, m->gate, T * K * F);
            quant_q8(m->gate, 0, 0, T * K * F);
            gemm_mxfp4(&L->down, m->gate, 1, m->list, m->offs, m->down, T);
        }
        moe_combine(m->down, m->ew, m->x, m->x, E, K, T); /* x += sum_k w_k down_k */
        dump("l_out", l, m->x + (size_t)last * E * 4, E);
    }
    if (!want_logits)
        return;
    rmsnorm(m->x + (size_t)last * E * 4, m->out_norm, m->xn, E, c->eps, 1);
    dump("result_norm", -1, m->xn, E);
    mv_q8(&m->out, m->xn, 0, 0, m->logits);
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
        printf("  %d '%.*s' %.4f", top[k], (int)len, s, h[top[k]]);
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
 * last token, reusing the longest cached prefix. Full-attention layers stay
 * valid for any prefix; a sliding-window ring (window + batch slots) only
 * when cutting back at most `batch` positions, or from the snapshot. */
static int no_reuse; /* GPTOSS_NO_REUSE=1: always recompute from position 0 */

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
    int show_analysis;
    int tty;              /* stdout is a terminal: style the analysis channel */
    ChatConfig chat;
} GenOptions;

/* Plain continuation of `prompt` (no chat template). */
static void run_completion(Model *m, const Tokenizer *tok, const char *prompt, GenOptions *o,
                           int32_t eos)
{
    int32_t *ids;
    unsigned n = tok_encode(tok, prompt, 1, &ids);
    if (n == 0 || n >= m->ctx) {
        fprintf(stderr, "prompt must have 1..%u tokens (has %u)\n", m->ctx - 1, n);
        exit(1);
    }
    fprintf(stderr, "prompt: %u tokens\n", n);
    double t0 = now_ms();
    prefill(m, ids, n);
    CHECK(cuCtxSynchronize());
    double t_prompt = now_ms() - t0;
    if (dump_enabled)
        print_top_logits(m, tok);

    int32_t eot = tok_special(tok, "<|endoftext|>");
    unsigned limit = o->max_new < 0 ? 64 : (unsigned)o->max_new, n_gen = 0;
    printf("%s", prompt);
    fflush(stdout);
    t0 = now_ms();
    for (; n_gen < limit; n_gen++) {
        int32_t next = next_token(m, &o->sampler);
        size_t len;
        const char *s = tok_piece(tok, next, &len);
        fwrite(s, 1, len, stdout);
        fflush(stdout);
        if (next == eos || next == eot || m->n_cache >= m->ctx)
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

/* One assistant reply to the conversation in turns[0..n). Streams it to
 * stdout and returns its final-channel text (malloc'ed), or NULL if the
 * conversation does not fit in the context. */
static char *chat_reply(Model *m, const Tokenizer *tok, const ChatTurn *turns, int n,
                        GenOptions *o)
{
    int32_t *ids;
    unsigned n_ids = harmony_render(tok, &o->chat, turns, n, &ids);
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

    HarmonyParser hp;
    harmony_parser_init(&hp, tok);
    int32_t eot = tok_special(tok, "<|endoftext|>");
    int showing = 0; /* inside a displayed analysis message */
    unsigned n_gen = 0, limit = o->max_new < 0 ? m->ctx : (unsigned)o->max_new;
    const char *stop = NULL;
    t0 = now_ms();
    for (;;) {
        int32_t next = next_token(m, &o->sampler);
        const char *text;
        size_t len;
        int ev = harmony_parser_feed(&hp, tok, next, &text, &len);
        int analysis = strcmp(hp.channel, "final") != 0;
        if (ev == HARMONY_MESSAGE_START && analysis && o->show_analysis) {
            printf(o->tty ? "\033[2m" : "[%s] ", hp.channel);
            showing = 1;
        } else if (ev == HARMONY_TEXT && (!analysis || o->show_analysis)) {
            fwrite(text, 1, len, stdout);
        } else if (ev == HARMONY_MESSAGE_END && showing) {
            printf(o->tty ? "\033[0m\n\n" : "\n\n");
            showing = 0;
        } else if (ev == HARMONY_DONE) {
            break;
        } else if (ev == HARMONY_TOOL_CALL) {
            stop = "the model asked for a tool call, which is not supported";
            break;
        }
        fflush(stdout);
        if (next == eot) {
            stop = "end of text";
            break;
        }
        if (++n_gen >= limit || m->n_cache >= m->ctx) {
            stop = m->n_cache >= m->ctx ? "context full" : "reply length limit (-n)";
            break;
        }
        step(m, next);
    }
    CHECK(cuCtxSynchronize());
    double t_gen = now_ms() - t0;
    if (showing && o->tty)
        printf("\033[0m");
    printf("\n");
    fflush(stdout);
    if (stop)
        fprintf(stderr, "[stopped: %s]\n", stop);
    fprintf(stderr, "[prompt %u tokens, %u new in %.0f ms | reply %u tokens, %.1f tok/s]\n",
            n_ids, n_ids - reused, t_prompt, n_gen, n_gen * 1e3 / t_gen);
    char *final = strdup(hp.final);
    harmony_parser_free(&hp);
    return final;
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

static int test_mv_q8(void)
{
    const unsigned rows = 37, cols = 4096;
    uint16_t *w16 = malloc(rows * cols * 2);
    float *x = malloc(cols * 4), *bias = malloc(rows * 4), *resid = malloc(rows * 4);
    for (unsigned i = 0; i < rows * cols; i++)
        w16[i] = float_to_half(rng_uniform(-1, 1) * (i % 97 == 0 ? 8 : 1));
    for (unsigned i = 0; i < cols; i++)
        x[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < rows; i++) {
        bias[i] = rng_uniform(-1, 1);
        resid[i] = rng_uniform(-1, 1);
    }
    int8_t *wq = malloc(rows * cols);
    uint16_t *wd = malloc(rows * cols / 32 * 2);
    quantize_rows_q8(w16, rows, cols, wq, wd);

    /* exact reference of what the kernel computes (Q8 weights, f32 x) and
     * the full-precision product, to see the weight quantization error */
    double *ref = malloc(rows * sizeof(double)), *ref_fp = malloc(rows * sizeof(double));
    for (unsigned r = 0; r < rows; r++) {
        double acc = 0, acc_fp = 0;
        for (unsigned b = 0; b < cols / 32; b++) {
            double bsum = 0;
            for (int j = 0; j < 32; j++) {
                size_t k = (size_t)b * 32 + j;
                bsum += (double)wq[(size_t)r * cols + k] * x[k];
                acc_fp += (double)half_lut[w16[(size_t)r * cols + k]] * x[k];
            }
            acc += bsum * half_lut[wd[(size_t)r * cols / 32 + b]];
        }
        ref[r] = acc + bias[r] + resid[r];
        ref_fp[r] = acc_fp + bias[r] + resid[r];
    }

    Q8W w = {upload(wq, rows * cols), upload(wd, rows * cols / 32 * 2), rows, cols};
    CUdeviceptr d_x = upload(x, cols * 4), d_b = upload(bias, rows * 4),
                d_r = upload(resid, rows * 4), d_y;
    CHECK(cuMemAlloc(&d_y, rows * 4));
    mv_q8(&w, d_x, d_b, d_r, d_y);
    float *y = malloc(rows * 4);
    CHECK(cuMemcpyDtoH(y, d_y, rows * 4));
    int fails = report("mv_q8 vs Q8-weight reference", rel_err(y, ref, rows), 1e-5);
    fails += report("mv_q8 vs f32 (quantization error)", rel_err(y, ref_fp, rows), 2e-2);

    CUdeviceptr bufs[] = {w.q, w.d, d_x, d_b, d_r, d_y};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        cuMemFree(bufs[i]);
    free(w16), free(x), free(bias), free(resid), free(wq), free(wd);
    free(ref), free(ref_fp), free(y);
    return fails;
}

static int test_mv_mxfp4(void)
{
    const unsigned E = 5, rows = 41, cols = 2880, nb = cols / 32, S = 3;
    const uint32_t ids[3] = {4, 0, 2};
    size_t nblk = (size_t)E * rows * nb;
    uint8_t *qs = malloc(nblk * 16), *sc = malloc(nblk);
    float *bias = malloc(E * rows * 4), *x = malloc(S * cols * 4);
    for (size_t i = 0; i < nblk * 16; i++)
        qs[i] = (uint8_t)rng_u32();
    for (size_t i = 0; i < nblk; i++)
        sc[i] = (uint8_t)(118 + rng_u32() % 12); /* scales 2^-10 .. 2^1 */
    sc[0] = 0, sc[1] = 1, sc[2] = 255 - 128;      /* subnormal-scale paths */
    for (unsigned i = 0; i < E * rows; i++)
        bias[i] = rng_uniform(-1, 1);
    for (unsigned i = 0; i < S * cols; i++)
        x[i] = rng_uniform(-3, 3);

    int fails = 0;
    for (int shared = 1; shared >= 0; shared--) {
        unsigned xstride = shared ? 0 : cols;
        int8_t *xq = malloc(S * cols);
        float *xd = malloc(S * nb * 4);
        for (unsigned s = 0; s < S; s++)
            cpu_quant_q8(x + s * xstride, cols, xq + s * cols, xd + s * nb);
        double *ref = malloc(S * rows * sizeof(double));
        for (unsigned s = 0; s < S; s++)
            for (unsigned r = 0; r < rows; r++) {
                double acc = 0;
                size_t grow = (size_t)ids[s] * rows + r;
                for (unsigned b = 0; b < nb; b++) {
                    const uint8_t *blk = qs + (grow * nb + b) * 16;
                    long isum = 0;
                    for (int j = 0; j < 16; j++) {
                        isum += fp4_values[blk[j] & 15] * xq[s * cols + b * 32 + j];
                        isum += fp4_values[blk[j] >> 4] * xq[s * cols + b * 32 + j + 16];
                    }
                    acc += (double)isum * e8m0_half(sc[grow * nb + b]) * xd[s * nb + b];
                }
                ref[s * rows + r] = acc + bias[ids[s] * rows + r];
            }

        MxW w = {upload(qs, nblk * 16), upload(sc, nblk), upload(bias, E * rows * 4), rows, cols, E};
        CUdeviceptr d_ids = upload(ids, sizeof(ids)), d_x = upload(x, S * cols * 4), d_xq, d_xd, d_y;
        CHECK(cuMemAlloc(&d_xq, S * cols));
        CHECK(cuMemAlloc(&d_xd, S * nb * 4));
        CHECK(cuMemAlloc(&d_y, S * rows * 4));
        quant_q8(d_x, d_xq, d_xd, shared ? cols : S * cols);
        mv_mxfp4(&w, d_ids, S, d_xq, d_xd, xstride, d_y);
        float *y = malloc(S * rows * 4);
        CHECK(cuMemcpyDtoH(y, d_y, S * rows * 4));
        fails += report(shared ? "mv_mxfp4 (shared input)" : "mv_mxfp4 (per-expert input)",
                        rel_err(y, ref, S * rows), 1e-5);
        CUdeviceptr bufs[] = {w.qs, w.sc, w.bias, d_ids, d_x, d_xq, d_xd, d_y};
        for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
            cuMemFree(bufs[i]);
        free(xq), free(xd), free(ref), free(y);
    }
    free(qs), free(sc), free(bias), free(x);
    return fails;
}

/* Causal attention for T queries at pos0.. over a (possibly ring-buffer)
 * cache, vs a CPU reference that reads keys by position. */
static int test_attention_case(unsigned pos0, unsigned T, unsigned window, unsigned slots)
{
    const unsigned nh = 8, nkv = 2, D = HEAD_DIM, q_stride = nh * D + 16, npos = pos0 + T;
    const float scale = 0.125f;
    float *q = malloc(T * q_stride * 4), *sinks = malloc(nh * 4), *out = malloc(T * nh * D * 4);
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
    for (unsigned h = 0; h < nh; h++)
        sinks[h] = rng_uniform(-1, 3);

    double *ref = malloc(T * nh * D * sizeof(double)), *s = malloc(npos * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        unsigned p = pos0 + t, lo = window && p >= window ? p - window + 1 : 0;
        for (unsigned h = 0; h < nh; h++) {
            unsigned kh = h / (nh / nkv);
            double m = sinks[h], sum = 0;
            for (unsigned j = lo; j <= p; j++) {
                double dot = 0;
                for (unsigned d = 0; d < D; d++)
                    dot += q[t * q_stride + h * D + d] * scale * half_lut[kpos[(j * nkv + kh) * D + d]];
                s[j] = dot;
                m = fmax(m, dot);
            }
            for (unsigned j = lo; j <= p; j++)
                sum += exp(s[j] - m);
            sum += exp(sinks[h] - m);
            for (unsigned d = 0; d < D; d++) {
                double acc = 0;
                for (unsigned j = lo; j <= p; j++)
                    acc += exp(s[j] - m) * half_lut[vpos[(j * nkv + kh) * D + d]];
                ref[(t * nh + h) * D + d] = acc / sum;
            }
        }
    }
    CUdeviceptr d_q = upload(q, T * q_stride * 4), d_k = upload(kc, slots * nkv * D * 2),
                d_v = upload(vc, slots * nkv * D * 2), d_s = upload(sinks, nh * 4), d_o;
    CHECK(cuMemAlloc(&d_o, T * nh * D * 4));
    attention(d_q, q_stride, d_k, d_v, d_s, d_o, pos0, window, slots, nh, nkv, scale, T);
    CHECK(cuMemcpyDtoH(out, d_o, T * nh * D * 4));
    char name[64];
    snprintf(name, sizeof(name), "attention T=%u pos0=%u window=%u ring=%u", T, pos0, window, slots);
    int fails = report(name, rel_err(out, ref, T * nh * D), 1e-4);
    cuMemFree(d_q), cuMemFree(d_k), cuMemFree(d_v), cuMemFree(d_s), cuMemFree(d_o);
    free(q), free(sinks), free(out), free(kpos), free(vpos), free(kc), free(vc), free(ref), free(s);
    return fails;
}

static int test_attention(void)
{
    return test_attention_case(76, 1, 0, 100) +  /* decoding, full layer */
           test_attention_case(40, 1, 16, 20) +  /* decoding, sliding window in a ring */
           test_attention_case(0, 9, 0, 64) +    /* first chunk of a prompt */
           test_attention_case(3, 5, 16, 24) +   /* window not yet full */
           test_attention_case(40, 7, 16, 24);   /* chunk wrapping around the ring */
}

static int test_gemm_q8(void)
{
    const unsigned rows = 128, cols = 384, T = 37, off = 32, ldy = rows + 64;
    uint16_t *w16 = malloc(rows * cols * 2);
    int8_t *wq = malloc(rows * cols);
    uint16_t *wd = malloc(rows * cols / 32 * 2);
    float *x = malloc(T * cols * 4), *bias = malloc(rows * 4), *y = malloc(T * ldy * 4);
    for (unsigned i = 0; i < rows * cols; i++)
        w16[i] = float_to_half(rng_uniform(-1, 1));
    for (unsigned i = 0; i < T * cols; i++)
        x[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < rows; i++)
        bias[i] = rng_uniform(-1, 1);
    for (unsigned i = 0; i < T * ldy; i++)
        y[i] = rng_uniform(-1, 1); /* residual, updated in place */
    quantize_rows_q8(w16, rows, cols, wq, wd);

    double *ref = malloc(T * ldy * sizeof(double));
    for (unsigned i = 0; i < T * ldy; i++)
        ref[i] = y[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned r = 0; r < rows; r++) {
            double acc = 0;
            for (unsigned k = 0; k < cols; k++)
                acc += (double)wq[r * cols + k] * half_lut[wd[(r * cols + k) / 32]] * x[t * cols + k];
            ref[t * ldy + off + r] += acc + bias[r];
        }
    Q8W w = {upload(wq, rows * cols), upload(wd, rows * cols / 32 * 2), rows, cols};
    CUdeviceptr d_x = upload(x, T * cols * 4), d_b = upload(bias, rows * 4),
                d_y = upload(y, T * ldy * 4);
    CUdeviceptr y_slice = d_y + off * 4;
    gemm_q8(&w, d_x, d_b, y_slice, y_slice, ldy, T);
    CHECK(cuMemcpyDtoH(y, d_y, T * ldy * 4));
    int fails = report("gemm_q8 (37 tokens, strided, in place)", rel_err(y, ref, T * ldy), 1e-5);
    cuMemFree(w.q), cuMemFree(w.d), cuMemFree(d_x), cuMemFree(d_b), cuMemFree(d_y);
    free(w16), free(wq), free(wd), free(x), free(bias), free(y), free(ref);
    return fails;
}

static int test_gemm_mxfp4(void)
{
    const unsigned E = 6, rows = 128, cols = 384, nb = cols / 32, T = 45, K = 4;
    size_t nblk = (size_t)E * rows * nb;
    uint8_t *qs = malloc(nblk * 16), *sc = malloc(nblk);
    float *bias = malloc(E * rows * 4), *x = malloc(T * K * cols * 4), *y = malloc(T * K * rows * 4);
    uint32_t *ids = malloc(T * K * 4);
    for (size_t i = 0; i < nblk * 16; i++)
        qs[i] = (uint8_t)rng_u32();
    for (size_t i = 0; i < nblk; i++)
        sc[i] = (uint8_t)(118 + rng_u32() % 12);
    for (unsigned i = 0; i < E * rows; i++)
        bias[i] = rng_uniform(-1, 1);
    for (unsigned i = 0; i < T * K * cols; i++)
        x[i] = rng_uniform(-3, 3);
    for (unsigned t = 0; t < T; t++) /* K distinct experts per token */
        for (unsigned k = 0; k < K; k++) {
            uint32_t e;
            int dup;
            do {
                e = rng_u32() % E;
                dup = 0;
                for (unsigned j = 0; j < k; j++)
                    dup |= ids[t * K + j] == e;
            } while (dup);
            ids[t * K + k] = e;
        }

    MxW w = {upload(qs, nblk * 16), upload(sc, nblk), upload(bias, E * rows * 4), rows, cols, E};
    CUdeviceptr d_ids = upload(ids, T * K * 4), d_x = upload(x, T * K * cols * 4), d_offs, d_list, d_y;
    CHECK(cuMemAlloc(&d_offs, (E + 1) * 4));
    CHECK(cuMemAlloc(&d_list, T * K * 4));
    CHECK(cuMemAlloc(&d_y, T * K * rows * 4));
    moe_sort(d_ids, T * K, E, d_offs, d_list);

    int fails = 0;
    double *ref = malloc(T * K * rows * sizeof(double));
    for (unsigned xdiv = K; xdiv >= 1; xdiv = xdiv == K ? 1 : 0) {
        for (unsigned p = 0; p < T * K; p++)
            for (unsigned r = 0; r < rows; r++) {
                size_t g = (size_t)ids[p] * rows + r;
                double acc = 0;
                for (unsigned b = 0; b < nb; b++)
                    for (int j = 0; j < 16; j++) {
                        uint8_t v = qs[(g * nb + b) * 16 + j];
                        double s = e8m0_half(sc[g * nb + b]);
                        acc += fp4_values[v & 15] * s * x[(p / xdiv) * cols + b * 32 + j];
                        acc += fp4_values[v >> 4] * s * x[(p / xdiv) * cols + b * 32 + j + 16];
                    }
                ref[p * rows + r] = acc + bias[ids[p] * rows + r];
            }
        CHECK(cuMemsetD32(d_y, 0, T * K * rows));
        gemm_mxfp4(&w, d_x, xdiv, d_list, d_offs, d_y, T);
        CHECK(cuMemcpyDtoH(y, d_y, T * K * rows * 4));
        fails += report(xdiv == K ? "moe_sort + gemm_mxfp4 (token input)"
                                  : "moe_sort + gemm_mxfp4 (per-pair input)",
                        rel_err(y, ref, T * K * rows), 1e-5);
    }
    CUdeviceptr bufs[] = {w.qs, w.sc, w.bias, d_ids, d_x, d_offs, d_list, d_y};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        cuMemFree(bufs[i]);
    free(qs), free(sc), free(bias), free(x), free(y), free(ids), free(ref);
    return fails;
}

static int test_router(void)
{
    const unsigned n = 32, k = 4;
    float logits[32], w[4];
    uint32_t ids[4];
    for (unsigned i = 0; i < n; i++)
        logits[i] = rng_uniform(-4, 4);
    logits[7] = logits[3]; /* a tie: the lower index must win */
    int order[32];
    for (unsigned i = 0; i < n; i++)
        order[i] = i;
    for (unsigned i = 0; i < k; i++) /* partial selection sort, stable */
        for (unsigned j = i + 1; j < n; j++)
            if (logits[order[j]] > logits[order[i]]) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
    double ref[4], sum = 0;
    for (unsigned i = 0; i < k; i++)
        sum += ref[i] = exp(logits[order[i]] - logits[order[0]]);
    CUdeviceptr d_l = upload(logits, sizeof(logits)), d_i, d_w;
    CHECK(cuMemAlloc(&d_i, k * 4));
    CHECK(cuMemAlloc(&d_w, k * 4));
    router_topk(d_l, d_i, d_w, n, k, 1);
    CHECK(cuMemcpyDtoH(ids, d_i, k * 4));
    CHECK(cuMemcpyDtoH(w, d_w, k * 4));
    int ids_ok = 1;
    for (unsigned i = 0; i < k; i++) {
        ids_ok &= ids[i] == (uint32_t)order[i];
        ref[i] /= sum;
    }
    printf("   %-48s %s\n", "router_topk ids", ids_ok ? "PASS" : "FAIL");
    int fails = !ids_ok + report("router_topk weights", rel_err(w, ref, k), 1e-5);
    cuMemFree(d_l), cuMemFree(d_i), cuMemFree(d_w);
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
    for (int i = 0; i < 65536; i++)
        half_lut[i] = half_to_float((uint16_t)i);
    printf("== kernel self-tests against CPU references\n");
    int fails = test_mv_q8() + test_mv_mxfp4() + test_attention() + test_router() +
                test_gemm_q8() + test_gemm_mxfp4() + test_sampling();
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

static int cmd_tokenize(int argc, char **argv)
{
    Gguf g;
    gguf_open(&g, argc >= 4 ? argv[3] : DEFAULT_MODEL);
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

/* each stdin line ("\n" and "\\" unescaped) -> one line of token ids */
static int cmd_tokenize_ids(int argc, char **argv)
{
    Gguf g;
    gguf_open(&g, argc >= 3 ? argv[2] : DEFAULT_MODEL);
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
        int k = tok_encode(tok, line, 0, &ids);
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

/* `routes CORPUS OUT`: runs every document of CORPUS through the model and
 * writes, per token, the experts the router chose in every layer. CORPUS is
 * text where a line "@@@ category" starts a new document; documents are cut
 * to the first ROUTES_MAX_TOKENS tokens. OUT gets one record per token:
 *   i32 category, i32 document, i32 position, i32 token,
 *   u8 expert[n_layer][k], f32 weight[n_layer][k]
 * and OUT.cats the category names, one per line, in index order. */
#define ROUTES_MAX_TOKENS 1024

static int cmd_routes(const char *corpus, const char *out_path, const char *model_path,
                      const char *cubin)
{
    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", corpus);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc(size + 1);
    if (fread(text, 1, size, f) != (size_t)size)
        return 1;
    text[size] = 0;
    fclose(f);

    init_cuda(cubin);
    Gguf g;
    gguf_open(&g, model_path);
    Tokenizer *tok = tok_load(&g);
    Model m = {0};
    model_load(&m, &g, ROUTES_MAX_TOKENS + 16, 0);
    unsigned L = m.c.n_layer, K = m.c.n_expert_used;
    route_ids = calloc((size_t)ROUTES_MAX_TOKENS * L * K, 1);
    route_w = malloc((size_t)ROUTES_MAX_TOKENS * L * K * 4);
    route_tmp_ids = malloc((size_t)MAX_BATCH * K * 4);
    route_tmp_w = malloc((size_t)MAX_BATCH * K * 4);

    char cats_path[1024];
    snprintf(cats_path, sizeof(cats_path), "%s.cats", out_path);
    FILE *out = fopen(out_path, "wb"), *cats = fopen(cats_path, "w");
    if (!out || !cats) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    char cat_names[256][64];
    int n_cats = 0, n_docs = 0;
    size_t n_tok_total = 0;
    double t0 = now_ms();
    for (char *p = strstr(text, "@@@ "); p;) {
        char *eol = strchr(p, '\n');
        if (!eol)
            break;
        char name[64];
        snprintf(name, sizeof(name), "%.*s", (int)(eol - p - 4), p + 4);
        char *next = strstr(eol + 1, "\n@@@ ");
        char *end = next ? next : text + size;
        char saved = *end;
        *end = 0;
        int32_t cat = -1;
        for (int i = 0; i < n_cats; i++)
            if (!strcmp(cat_names[i], name))
                cat = i;
        if (cat < 0 && n_cats < 256) {
            cat = n_cats;
            snprintf(cat_names[n_cats++], 64, "%s", name);
            fprintf(cats, "%s\n", name);
        }
        int32_t *ids;
        unsigned n = tok_encode(tok, eol + 1, 1, &ids);
        if (n > ROUTES_MAX_TOKENS)
            n = ROUTES_MAX_TOKENS;
        if (n >= 2) {
            m.n_cache = 0; /* every document starts from an empty cache */
            prefill(&m, ids, n);
            for (unsigned t = 0; t < n; t++) {
                int32_t hdr[4] = {cat, n_docs, (int32_t)t, ids[t]};
                fwrite(hdr, 4, 4, out);
                fwrite(route_ids + (size_t)t * L * K, 1, L * K, out);
                fwrite(route_w + (size_t)t * L * K, 4, L * K, out);
            }
            n_tok_total += n;
            n_docs++;
            fprintf(stderr, "\r%d documents, %zu tokens (%.0f tok/s)", n_docs, n_tok_total,
                    n_tok_total * 1e3 / (now_ms() - t0));
        }
        free(ids);
        *end = saved;
        p = next ? next + 1 : NULL;
    }
    fprintf(stderr, "\n");
    fclose(out);
    fclose(cats);
    free(text);
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [options] PROMPT        continue PROMPT as plain text\n"
            "       %s [options] -chat MESSAGE  one chat reply (harmony format)\n"
            "       %s [options] -i             interactive chat (one message per line)\n"
            "       %s test                     kernel self-tests\n"
            "       %s tokenize TEXT [model.gguf] | tokenize-ids [model.gguf] < lines\n"
            "       %s routes CORPUS OUT [model.gguf]  record expert routing (see cmd_routes)\n"
            "options:\n"
            "  -m FILE        model (default " DEFAULT_MODEL ")\n"
            "  -n N           max new tokens (default: 64 for PROMPT, until done for chat)\n"
            "  -c N           context length (default 4096, max 12000)\n"
            "  -b N           prompt chunk size (default: largest that fits, up to 128)\n"
            "  -k FILE        kernels (default " DEFAULT_CUBIN ")\n"
            "  --temp T       temperature (default 1.0; 0 = greedy)\n"
            "  --top-k K      keep the K most likely tokens (default 0 = off)\n"
            "  --top-p P      nucleus sampling (default 1.0 = off)\n"
            "  --min-p P      drop tokens below P x the most likely (default 0 = off)\n"
            "  --seed S       sampling seed (default: time)\n"
            "  --sys TEXT     developer instructions (system prompt) for chat\n"
            "  --effort E     reasoning effort for chat: low, medium (default), high\n"
            "  --date D       date in the chat system message (default today, YYYY-MM-DD)\n"
            "  --no-analysis  hide the model's reasoning (analysis channel) in chat\n"
            "  --print-prompt with -chat: print the rendered prompt and its tokens, then exit\n",
            prog, prog, prog, prog, prog, prog);
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "tokenize"))
        return cmd_tokenize(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "tokenize-ids"))
        return cmd_tokenize_ids(argc, argv);
    if (argc >= 4 && !strcmp(argv[1], "routes"))
        return cmd_routes(argv[2], argv[3], argc >= 5 ? argv[4] : DEFAULT_MODEL, DEFAULT_CUBIN);

    const char *model_path = DEFAULT_MODEL, *cubin = DEFAULT_CUBIN, *prompt = NULL, *chat = NULL;
    unsigned ctx = 4096, batch = 0; /* 0 = pick from free memory */
    uint64_t seed = 0;
    int test = 0, interactive = 0, print_prompt = 0;
    GenOptions o = {.sampler = {.temperature = 1.0f, .top_p = 1.0f}, .max_new = -1,
                    .show_analysis = 1, .tty = isatty(1)};
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
            o.chat.developer = v;
        else if (OPT("--effort"))
            o.chat.effort = v;
        else if (OPT("--date"))
            snprintf(o.chat.date, sizeof(o.chat.date), "%s", v);
        else if (OPT("-chat"))
            chat = v;
        else if (!strcmp(a, "-i"))
            interactive = 1;
        else if (!strcmp(a, "--no-analysis"))
            o.show_analysis = 0;
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
    /* attention keeps up to ctx scores in 48 KB of shared memory */
    if ((!!prompt + !!chat + interactive) != 1 || ctx < 16 || ctx > 12000 || batch > MAX_BATCH)
        usage(argv[0]);
    if (o.chat.effort && strcmp(o.chat.effort, "low") && strcmp(o.chat.effort, "medium") &&
        strcmp(o.chat.effort, "high"))
        usage(argv[0]);
    if (print_prompt && chat) { /* no GPU needed */
        Gguf g;
        gguf_open(&g, model_path);
        Tokenizer *tok = tok_load(&g);
        ChatTurn turn = {1, chat};
        int32_t *ids;
        int n = harmony_render(tok, &o.chat, &turn, 1, &ids);
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
    dump_enabled = getenv("GPTOSS_DUMP") && atoi(getenv("GPTOSS_DUMP"));
    no_reuse = getenv("GPTOSS_NO_REUSE") && atoi(getenv("GPTOSS_NO_REUSE"));

    init_cuda(cubin);
    Gguf g;
    gguf_open(&g, model_path);
    Tokenizer *tok = tok_load(&g);
    Model m = {0};
    model_load(&m, &g, ctx, batch);

    if (prompt) {
        const GgufKV *eos_kv = gguf_kv(&g, "tokenizer.ggml.eos_token_id");
        int32_t eos = -1;
        if (eos_kv)
            memcpy(&eos, eos_kv->val, 4);
        run_completion(&m, tok, prompt, &o, eos);
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
