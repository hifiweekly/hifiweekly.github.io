/*
 * qwen35.c - Qwen 3.5 (dense, text) inference on the GPU with hand-written PTX
 * kernels (qwen35.ptx) driven through the CUDA Driver API. No other libraries.
 *
 * The model is read from a GGUF file (gguf.c), text is tokenized with the
 * byte-level BPE vocabulary stored in it (tokenizer.c), and every layer runs
 * in the PTX kernels; the host only looks up token embeddings and launches.
 *
 * Qwen 3.5 is a hybrid: of every 4 layers 3 are Gated DeltaNet layers (linear
 * attention: a causal conv1d over the last 4 tokens, then a 128 x 128
 * recurrent state per head updated by the gated delta rule) and the 4th is a
 * gated full-attention layer (16 query heads, 4 kv heads, RoPE on 64 of the
 * 256 dims, output * sigmoid(gate)). The feed-forward is SwiGLU.
 *
 * Weights stay in the file's ggml formats (see quant.h): Q4_1, Q5_K, Q6_K
 * and Q8_0 matrices are used as stored (the Unsloth Q4_1 file mixes them);
 * a BF16 / F16 / F32 file is quantized to Q8_0 at load time (or once with
 * `qwen35_demo quantize`). Token embeddings stay in host memory (one row per
 * token). Activations entering a matrix are quantized to int8 blocks, as
 * llama.cpp does, so every product is an exact integer dot product.
 *
 * The prompt is processed in chunks of up to -b tokens (by default the
 * largest of 512, 256, ... that fits in GPU memory) with an int8 tensor-core
 * GEMM; generated tokens then go one at a time through matrix-vector
 * kernels. The full-attention layers keep an f16 KV cache of every position;
 * the DeltaNet layers keep their conv and recurrent state, which can't be
 * rewound: a chat snapshots it where the next turn's prompt will diverge
 * (before the reply's thinking block) and recomputes from there.
 *
 * Usage (see usage() for every option):
 *   ./qwen35_demo [options] PROMPT        continuation of PROMPT as plain text
 *                                         (control tokens such as
 *                                         "<|im_start|>" are recognised)
 *   ./qwen35_demo [options] -chat MESSAGE one reply in the chat format
 *   ./qwen35_demo [options] -i            interactive multi-turn chat
 *   ./qwen35_demo quantize [IN] [OUT]     BF16 GGUF -> Q8_0 GGUF
 *   ./qwen35_demo test                    kernel self-tests against the CPU
 *   ./qwen35_demo cache-test              state reuse is bit-exact (needs the model)
 *   ./qwen35_demo tokenize TEXT           show the tokens of TEXT
 *   ./qwen35_demo tokenize-ids            one line of ids per stdin line
 * Sampling defaults to Qwen's recommended settings for non-thinking mode
 * (temperature 0.7, top-k 20, top-p 0.8; with --think: 1.0, 20, 0.95);
 * --temp 0 is greedy.
 * Environment (debugging): QWEN35_DUMP=1 prints a summary of intermediate
 * tensors (named like llama.cpp's llama-eval-callback) and the top logits;
 * QWEN35_LOGITS=FILE writes the logits after a PROMPT as raw f32;
 * QWEN35_NO_REUSE=1 recomputes every chat turn from scratch.
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

#define DEFAULT_MODEL "Qwen3.5-9B-Q4_1.gguf"
#define Q8_MODEL "Qwen3.5-9B-Q8_0.gguf"
#define BF16_MODEL "Qwen3.5-9B-BF16.gguf"
#define DEFAULT_CUBIN "qwen35.cubin"
#define HEAD_DIM 256         /* qk_norm_rope and attention assume 256-dim heads */
#define ROPE_DIMS 64         /* qk_norm_rope rotates the first 64 dims */
#define SSM_DIM 128          /* gdn kernels: 128-dim heads, 16 k / 32 v heads */
#define SSM_KH 16
#define SSM_VH 32
#define CONV_CH (2 * SSM_KH * SSM_DIM + SSM_VH * SSM_DIM) /* 8192 */
#define CONV_K 4
#define CAND_CAP 16384       /* sampling candidates gathered on the GPU */
#define AUTO_MAX_BATCH 512   /* default prompt chunk: the largest power of 2 up to */
#define AUTO_SPARE_BYTES (64u << 20) /* this that leaves this much GPU memory free */
#define MAX_BATCH 512

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

static CUfunction k_rmsnorm, k_quant, k_mv_q4_1, k_mv_q8_0, k_mv_q5_k, k_mv_q6_k, k_gemm,
    k_conv, k_conv_save, k_recur, k_gnorm, k_swiglu, k_qk_rope, k_attn, k_argmax, k_compact,
    k_sample;
static unsigned max_blocks; /* cap for grid-stride kernels: 8 blocks per SM */

static void load_kernels(const char *cubin)
{
    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, cubin);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cannot load %s (run `make` in the qwen35 directory)\n", cubin);
        exit(1);
    }
    struct { CUfunction *fn; const char *name; } k[] = {
        {&k_rmsnorm, "rmsnorm"}, {&k_quant, "quant_q8"}, {&k_mv_q4_1, "mv_q4_1"},
        {&k_mv_q8_0, "mv_q8_0"}, {&k_mv_q5_k, "mv_q5_k"}, {&k_mv_q6_k, "mv_q6_k"},
        {&k_gemm, "gemm_q"}, {&k_conv, "gdn_conv"}, {&k_conv_save, "conv_save"},
        {&k_recur, "gdn_recur"}, {&k_gnorm, "gated_norm"}, {&k_swiglu, "swiglu"},
        {&k_qk_rope, "qk_norm_rope"}, {&k_attn, "attention"}, {&k_argmax, "argmax"},
        {&k_compact, "compact_logits"}, {&k_sample, "sample_logits"},
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

/* A quantized matrix [rows][cols] on the GPU (layouts in qwen35.ptx):
 *   Q8_0  q = int8 [rows][cols],               s = f16 d [rows][cols/32]
 *   Q4_1  q = qs [rows][cols/32][16],          s = f16x2 {d, m} [rows][cols/32]
 *   Q5_K  q = the 176-byte blocks,             s unused
 *   Q6_K  q = ql|qh|scales [rows][cols/256][208], s = f16 d [rows][cols/256] */
typedef struct {
    uint32_t type; /* GGML_* */
    CUdeviceptr q, s;
    unsigned rows, cols;
} QW;

static const char *type_name(uint32_t t)
{
    switch (t) {
    case GGML_F32: return "F32";
    case GGML_F16: return "F16";
    case GGML_BF16: return "BF16";
    case GGML_Q8_0: return "Q8_0";
    case GGML_Q4_1: return "Q4_1";
    case GGML_Q5_K: return "Q5_K";
    case GGML_Q6_K: return "Q6_K";
    }
    return "?";
}

/* y = W . x + resid (x as xq / xds from quant_q8; resid may be 0) */
static void mv(const QW *w, CUdeviceptr xq, CUdeviceptr xds, CUdeviceptr resid, CUdeviceptr y)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->q, (void *)&w->s, &xq, &xds, &resid, &y, &rows, &cols};
    CUfunction fn = w->type == GGML_Q4_1 ? k_mv_q4_1 : w->type == GGML_Q8_0 ? k_mv_q8_0
                  : w->type == GGML_Q5_K ? k_mv_q5_k : k_mv_q6_k;
    CHECK(cuLaunchKernel(fn, (rows + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* Y[t] (row stride ldy) = W . X[t] + resid[t] for T tokens (X as Xq / Xds) */
static void gemm(const QW *w, CUdeviceptr X, CUdeviceptr XS, CUdeviceptr resid, CUdeviceptr Y,
                 unsigned ldy, unsigned T)
{
    unsigned rows = w->rows, cols = w->cols;
    unsigned type = w->type == GGML_Q8_0 ? 0 : w->type == GGML_Q4_1 ? 1 : 2;
    if (w->type == GGML_Q6_K) {
        fprintf(stderr, "internal error: no Q6_K GEMM (only the output head may be Q6_K)\n");
        exit(1);
    }
    void *args[] = {(void *)&w->q, (void *)&w->s, &X, &XS, &resid, &Y, &rows, &cols, &T, &ldy, &type};
    CHECK(cuLaunchKernel(k_gemm, (T + 63) / 64, rows / 64, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* each of the T rows of x [T][n] */
static void rmsnorm(CUdeviceptr x, CUdeviceptr w, CUdeviceptr y, unsigned n, float eps,
                    unsigned T)
{
    void *args[] = {&x, &w, &y, &n, &eps};
    CHECK(cuLaunchKernel(k_rmsnorm, T, 1, 1, 1024, 1, 1, 0, NULL, args, NULL));
}

/* x f32 [n] -> xq int8 [n], xds f32x2 [n/32] */
static void quant_q8(CUdeviceptr x, CUdeviceptr xq, CUdeviceptr xds, unsigned n)
{
    unsigned nblk = n / 32;
    void *args[] = {&x, &xq, &xds, &nblk};
    launch_1d(k_quant, nblk * 32, args);
}

/* causal conv1d + silu (+ L2 norm of the first nnorm heads) for T tokens */
static void gdn_conv(CUdeviceptr in, unsigned ld, CUdeviceptr state, CUdeviceptr w,
                     CUdeviceptr out, unsigned C, unsigned nnorm, float eps, unsigned T)
{
    void *args[] = {&in, &ld, &state, &w, &out, &C, &nnorm, &eps};
    CHECK(cuLaunchKernel(k_conv, C / 128, T, 1, 128, 1, 1, 0, NULL, args, NULL));
}

static void conv_save(CUdeviceptr in, unsigned ld, CUdeviceptr state, unsigned C, unsigned T)
{
    void *args[] = {&in, &ld, &state, &C, &T};
    launch_1d(k_conv_save, C, args);
}

static void gdn_recur(CUdeviceptr qkv, CUdeviceptr ba, CUdeviceptr dt, CUdeviceptr a,
                      CUdeviceptr state, CUdeviceptr out, unsigned T)
{
    float scale = 1.0f / sqrtf((float)SSM_DIM);
    void *args[] = {&qkv, &ba, &dt, &a, &state, &out, &T, &scale};
    CHECK(cuLaunchKernel(k_recur, SSM_VH, SSM_DIM / 32, 1, 32, 1, 1, 0, NULL, args, NULL));
}

static void gated_norm(CUdeviceptr o, CUdeviceptr z, unsigned ldz, CUdeviceptr w, CUdeviceptr out,
                       unsigned heads, float eps, unsigned T)
{
    void *args[] = {&o, &z, &ldz, &w, &out, &eps};
    CHECK(cuLaunchKernel(k_gnorm, heads, T, 1, 128, 1, 1, 0, NULL, args, NULL));
}

/* out[t] = silu(gu[t][0..n)) * gu[t][n..2n) for T tokens */
static void swiglu(CUdeviceptr gu, CUdeviceptr out, unsigned n, unsigned T)
{
    void *args[] = {&gu, &out, &n, &T};
    launch_1d(k_swiglu, n * T, args);
}

/* q/k head norms and RoPE for tokens at positions pos0 .. pos0+T-1; k and v
 * go to the cache at their position */
static void qk_norm_rope(CUdeviceptr qkv, unsigned stride, CUdeviceptr qw, CUdeviceptr kw,
                         CUdeviceptr cs, CUdeviceptr kc, CUdeviceptr vc, unsigned pos0,
                         unsigned nh, unsigned nkv, float eps, unsigned T)
{
    void *args[] = {&qkv, &stride, &qw, &kw, &cs, &kc, &vc, &pos0, &nh, &nkv, &eps};
    CHECK(cuLaunchKernel(k_qk_rope, nh + 2 * nkv, T, 1, 32, 1, 1, 0, NULL, args, NULL));
}

/* causal attention for T query tokens at pos0.. (q head h at q + h * qhs in
 * a row of q_stride floats; gate: multiply by sigmoid of the 256 values after it) */
static void attention(CUdeviceptr q, unsigned q_stride, unsigned qhs, unsigned gate,
                      CUdeviceptr kc, CUdeviceptr vc, CUdeviceptr out, unsigned pos0,
                      unsigned slots, unsigned nh, unsigned nkv, float scale, unsigned T)
{
    unsigned window = 0;
    void *args[] = {&q, &q_stride, &qhs, &gate, &kc, &vc, &out, &pos0, &window, &slots, &nh,
                    &nkv, &scale};
    CHECK(cuLaunchKernel(k_attn, nh, T, 1, 256, 1, 1, 0, NULL, args, NULL));
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
    unsigned rope_dims, attn_interval, ssm_inner;
    float eps, rope_base, attn_scale;
} Config;

typedef struct {
    int recurrent;                  /* Gated DeltaNet layer, else full attention */
    CUdeviceptr attn_norm, post_norm;
    QW gate_up;                     /* gate rows then up rows, [2 n_ff][n_embd] */
    QW down;
    /* DeltaNet */
    QW qkvz;                        /* qkv rows (8192) then z rows (ssm_inner) */
    QW ba;                          /* beta rows (32) then alpha rows (32) */
    QW ssm_out;
    CUdeviceptr conv_w, dt_bias, ssm_a, ssm_norm;
    CUdeviceptr conv_state, ssm_state; /* f32 [3][8192], f32 [32][128][128] */
    /* full attention */
    QW wqkv;                        /* q|gate rows (2 n_head 256), k rows, v rows */
    QW wo;
    CUdeviceptr q_norm, k_norm;
    CUdeviceptr kc, vc;             /* f16 [ctx][n_kv_head][256] */
} Layer;

typedef struct {
    Config c;
    unsigned ctx;            /* max positions */
    unsigned batch;          /* max tokens per forward() call */
    Layer *layer;
    CUdeviceptr out_norm;
    QW out;                  /* output head [n_vocab][n_embd] */
    const GgufTensor *embd;  /* token embeddings in the mmap'ed file */
    CUdeviceptr rope;        /* f32 [ctx][32][2] */
    /* activations, [batch][...] (decoding uses row 0) */
    CUdeviceptr x, xn, big, cv, ro, ba, xq, xds, logits, next;
    CUdeviceptr cand;        /* sampling candidates, see compact_logits */
    CUdeviceptr draw;        /* sample_logits result */
    float *h_x;
    /* the DeltaNet layers' conv and recurrent states, back to back */
    CUdeviceptr rstate;
    size_t rstate_bytes;
    /* what the caches hold: tokens of positions 0 .. n_cache-1 */
    int32_t *cache_tok;
    unsigned n_cache;
    /* copy of the recurrent states taken when the cache held snap_tok[0..snap_pos) */
    CUdeviceptr snap;
    int32_t *snap_tok;
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
    if (!arch || strncmp(gguf_str(&p, &len), "qwen35", 6) || len != 6) {
        fprintf(stderr, "model architecture is not qwen35\n");
        exit(1);
    }
    c->n_layer = gguf_u32(g, "qwen35.block_count");
    c->n_embd = gguf_u32(g, "qwen35.embedding_length");
    c->n_head = gguf_u32(g, "qwen35.attention.head_count");
    c->n_kv_head = gguf_u32(g, "qwen35.attention.head_count_kv");
    c->head_dim = gguf_u32(g, "qwen35.attention.key_length");
    c->n_ff = gguf_u32(g, "qwen35.feed_forward_length");
    c->max_ctx = gguf_u32(g, "qwen35.context_length");
    c->rope_dims = gguf_u32(g, "qwen35.rope.dimension_count");
    c->rope_base = gguf_f32(g, "qwen35.rope.freq_base");
    c->eps = gguf_f32(g, "qwen35.attention.layer_norm_rms_epsilon");
    c->attn_interval = has_kv(g, "qwen35.full_attention_interval")
                           ? gguf_u32(g, "qwen35.full_attention_interval") : 4;
    c->ssm_inner = gguf_u32(g, "qwen35.ssm.inner_size");
    c->attn_scale = 1.0f / sqrtf((float)c->head_dim);
    c->n_vocab = (unsigned)gguf_kv(g, "tokenizer.ggml.tokens")->arr_n;
    /* the kernels' fixed shapes (the 9B's) */
    if (c->head_dim != HEAD_DIM || gguf_u32(g, "qwen35.attention.value_length") != HEAD_DIM ||
        c->rope_dims != ROPE_DIMS || c->n_head % c->n_kv_head || c->n_embd % 256 ||
        c->n_ff % 256 || c->attn_interval == 0 ||
        gguf_u32(g, "qwen35.ssm.conv_kernel") != CONV_K ||
        gguf_u32(g, "qwen35.ssm.state_size") != SSM_DIM ||
        gguf_u32(g, "qwen35.ssm.group_count") != SSM_KH ||
        gguf_u32(g, "qwen35.ssm.time_step_rank") != SSM_VH || c->ssm_inner != SSM_VH * SSM_DIM) {
        fprintf(stderr, "unsupported qwen35 hyperparameters\n");
        exit(1);
    }
}

/* llama.cpp: every attn_interval-th layer (il + 1) is full attention */
static int is_recurrent(const Config *c, unsigned l)
{
    return (l + 1) % c->attn_interval != 0;
}

/* cos/sin for every position and rotary pair, as ggml's rope cache:
 * theta = pos * base^(-2i/d), the power taken by repeated multiplication. */
static void rope_table(unsigned n_dims, unsigned ctx, float base, float *cs)
{
    const float theta_scale = powf(base, -2.0f / n_dims);
    for (unsigned p = 0; p < ctx; p++) {
        float theta = (float)p;
        for (unsigned i0 = 0; i0 < n_dims; i0 += 2) {
            cs[(size_t)p * n_dims + i0] = cosf(theta);
            cs[(size_t)p * n_dims + i0 + 1] = sinf(theta);
            theta *= theta_scale;
        }
    }
}

/* --- loading --- */

static void *staging;                    /* pinned host buffer for uploads */
static uint8_t *quant_buf;               /* quantized blocks of a chunk of rows */
static const size_t STAGING_BYTES = 64 << 20;
/* All model memory is carved out of one cuMemAlloc'ed arena. With dry_run,
 * dalloc only counts: model_load then measures what it would need. */
static CUdeviceptr arena;
static size_t arena_size, dev_bytes;     /* dev_bytes = arena bytes in use */
static int dry_run;

static CUdeviceptr dalloc(size_t n)
{
    size_t off = (dev_bytes + 255) & ~(size_t)255;
    if (!dry_run && off + n > arena_size) {
        fprintf(stderr, "internal error: GPU arena of %zu bytes too small\n", arena_size);
        exit(1);
    }
    dev_bytes = off + n;
    return dry_run ? 0 : arena + off;
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

static int has_tensor(const Gguf *g, const char *name)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, name))
            return 1;
    return 0;
}

static CUdeviceptr upload_f32(const Gguf *g, const char *name, uint64_t ne0, uint64_t ne1)
{
    const GgufTensor *t = gguf_tensor(g, name, GGML_F32, ne0, ne1, 0);
    CUdeviceptr p = dalloc(t->nbytes);
    if (!dry_run)
        CHECK(cuMemcpyHtoD(p, t->data, t->nbytes));
    return p;
}

/* The GPU type of a matrix stored as `type`: quantized types as they are,
 * float ones as Q8_0. */
static uint32_t gpu_type(const GgufTensor *t)
{
    switch (t->type) {
    case GGML_Q8_0:
    case GGML_Q4_1:
    case GGML_Q5_K:
    case GGML_Q6_K: return t->type;
    case GGML_F32:
    case GGML_F16:
    case GGML_BF16: return GGML_Q8_0;
    }
    fprintf(stderr, "%s: tensor type %u is not supported\n", t->name, t->type);
    exit(1);
}

/* Bytes of the q and s arrays of a [rows][cols] matrix of `type`. */
static void qw_sizes(uint32_t type, size_t rows, size_t cols, size_t *q, size_t *s)
{
    switch (type) {
    case GGML_Q8_0: *q = rows * cols, *s = rows * (cols / 32) * 2; return;
    case GGML_Q4_1: *q = rows * (cols / 32) * 16, *s = rows * (cols / 32) * 4; return;
    case GGML_Q5_K: *q = rows * (cols / 256) * 176, *s = 0; return;
    case GGML_Q6_K: *q = rows * (cols / 256) * 208, *s = rows * (cols / 256) * 2; return;
    }
    *q = *s = 0;
}

/* nb ggml blocks of `type` -> the GPU layout's q and s arrays */
static void split_blocks(uint32_t type, const uint8_t *src, size_t nb, uint8_t *q, uint8_t *s)
{
    for (size_t b = 0; b < nb; b++)
        switch (type) {
        case GGML_Q8_0:
            memcpy(s + b * 2, src + b * 34, 2);
            memcpy(q + b * 32, src + b * 34 + 2, 32);
            break;
        case GGML_Q4_1:
            memcpy(s + b * 4, src + b * 20, 4);
            memcpy(q + b * 16, src + b * 20 + 4, 16);
            break;
        case GGML_Q5_K:
            memcpy(q + b * 176, src + b * 176, 176);
            break;
        case GGML_Q6_K:
            memcpy(q + b * 208, src + b * 210, 208);
            memcpy(s + b * 2, src + b * 210 + 208, 2);
            break;
        }
}

/* ggml blocks per row of `cols` values */
static size_t blocks_per_row(uint32_t type, size_t cols)
{
    return cols / (type == GGML_Q5_K || type == GGML_Q6_K ? 256 : 32);
}

/* A matrix made of the tensors names[0..n) stacked by rows (all [cols][rows_i]),
 * in the GPU layout of their (common) type. */
static QW upload_matrix(const Gguf *g, const char *const *names, const unsigned *rows, int n,
                        unsigned cols)
{
    const GgufTensor *t[4];
    unsigned total = 0;
    for (int i = 0; i < n; i++) {
        t[i] = find_tensor(g, names[i], cols, rows[i]);
        total += rows[i];
    }
    QW w = {gpu_type(t[0]), 0, 0, total, cols};
    for (int i = 1; i < n; i++)
        if (gpu_type(t[i]) != w.type) {
            fprintf(stderr, "%s and %s have different types (%s, %s): not supported\n",
                    names[0], names[i], type_name(t[0]->type), type_name(t[i]->type));
            exit(1);
        }
    if (cols % (w.type == GGML_Q5_K || w.type == GGML_Q6_K ? 256 : 64) ||
        (w.type != GGML_Q6_K && total % 64)) {
        fprintf(stderr, "%s: shape [%u, %u] is not supported\n", names[0], cols, total);
        exit(1);
    }
    size_t qb, sb;
    qw_sizes(w.type, total, cols, &qb, &sb);
    w.q = dalloc(qb);
    w.s = sb ? dalloc(sb) : 0;
    if (dry_run)
        return w;
    size_t row_q, row_s, file_row = gguf_type_bytes(w.type, cols), base = 0;
    size_t chunk = STAGING_BYTES / file_row;
    qw_sizes(w.type, 1, cols, &row_q, &row_s);
    for (int i = 0; i < n; base += rows[i++]) {
        for (size_t r0 = 0; r0 < rows[i]; r0 += chunk) {
            size_t nr = rows[i] - r0 < chunk ? rows[i] - r0 : chunk;
            const uint8_t *src;
            if (t[i]->type == w.type) {
                src = t[i]->data + r0 * file_row;
            } else {
                quantize_rows_q8_0(t[i], r0, nr, quant_buf);
                src = quant_buf;
            }
            uint8_t *q = staging, *s = q + nr * row_q;
            split_blocks(w.type, src, nr * blocks_per_row(w.type, cols), q, s);
            CHECK(cuMemcpyHtoD(w.q + (base + r0) * row_q, q, nr * row_q));
            if (row_s)
                CHECK(cuMemcpyHtoD(w.s + (base + r0) * row_s, s, nr * row_s));
        }
    }
    return w;
}

static QW upload_one(const Gguf *g, const char *name, unsigned cols, unsigned rows)
{
    return upload_matrix(g, &name, &rows, 1, cols);
}

static void model_free_host(Model *m)
{
    free(m->layer);
    m->layer = NULL;
}

static void model_load(Model *m, const Gguf *g, unsigned ctx, unsigned batch)
{
    Config *c = &m->c;
    read_config(g, c);
    unsigned E = c->n_embd, F = c->n_ff, KV = c->n_kv_head * HEAD_DIM,
             QG = 2 * c->n_head * HEAD_DIM, QH = c->n_head * HEAD_DIM, DI = c->ssm_inner;
    m->ctx = ctx;
    m->batch = batch;
    if (!dry_run) {
        CHECK(cuMemHostAlloc(&staging, STAGING_BYTES, 0));
        quant_buf = malloc(STAGING_BYTES);
    }

    double t0 = now_ms();
    m->layer = calloc(c->n_layer, sizeof(Layer));
    unsigned n_recr = 0;
    for (unsigned l = 0; l < c->n_layer; l++)
        n_recr += is_recurrent(c, l);
    size_t conv_bytes = (size_t)(CONV_K - 1) * CONV_CH * 4,
           ssm_bytes = (size_t)SSM_VH * SSM_DIM * SSM_DIM * 4;
    m->rstate_bytes = n_recr * (conv_bytes + ssm_bytes);
    m->rstate = dalloc(m->rstate_bytes);
    m->snap = dalloc(m->rstate_bytes);
    char name[4][128];
#define T(i, fmt) (snprintf(name[i], sizeof(name[i]), "blk.%u." fmt, l), name[i])
    unsigned ri = 0;
    for (unsigned l = 0; l < c->n_layer; l++) {
        Layer *L = &m->layer[l];
        L->recurrent = is_recurrent(c, l);
        L->attn_norm = upload_f32(g, T(0, "attn_norm.weight"), E, 1);
        L->post_norm = upload_f32(g, T(0, "post_attention_norm.weight"), E, 1);
        if (L->recurrent) {
            const char *qz[] = {T(0, "attn_qkv.weight"), T(1, "attn_gate.weight")};
            unsigned qz_rows[] = {CONV_CH, DI};
            L->qkvz = upload_matrix(g, qz, qz_rows, 2, E);
            const char *ba[] = {T(2, "ssm_beta.weight"), T(3, "ssm_alpha.weight")};
            unsigned ba_rows[] = {SSM_VH, SSM_VH};
            L->ba = upload_matrix(g, ba, ba_rows, 2, E);
            L->ssm_out = upload_one(g, T(0, "ssm_out.weight"), DI, E);
            L->conv_w = upload_f32(g, T(0, "ssm_conv1d.weight"), CONV_K, CONV_CH);
            L->dt_bias = upload_f32(g, T(0, "ssm_dt.bias"), SSM_VH, 1);
            L->ssm_a = upload_f32(g, T(0, "ssm_a"), SSM_VH, 1);
            L->ssm_norm = upload_f32(g, T(0, "ssm_norm.weight"), SSM_DIM, 1);
            L->conv_state = m->rstate + ri * (conv_bytes + ssm_bytes);
            L->ssm_state = L->conv_state + conv_bytes;
            ri++;
        } else {
            const char *qkv[] = {T(0, "attn_q.weight"), T(1, "attn_k.weight"), T(2, "attn_v.weight")};
            unsigned qkv_rows[] = {QG, KV, KV};
            L->wqkv = upload_matrix(g, qkv, qkv_rows, 3, E);
            L->wo = upload_one(g, T(0, "attn_output.weight"), QH, E);
            L->q_norm = upload_f32(g, T(0, "attn_q_norm.weight"), HEAD_DIM, 1);
            L->k_norm = upload_f32(g, T(0, "attn_k_norm.weight"), HEAD_DIM, 1);
            L->kc = dalloc((size_t)ctx * KV * 2);
            L->vc = dalloc((size_t)ctx * KV * 2);
        }
        const char *gu[] = {T(0, "ffn_gate.weight"), T(1, "ffn_up.weight")};
        unsigned gu_rows[] = {F, F};
        L->gate_up = upload_matrix(g, gu, gu_rows, 2, E);
        L->down = upload_one(g, T(0, "ffn_down.weight"), F, E);
        if (!dry_run)
            fprintf(stderr, "\rloading layer %u/%u  (%.2f GiB on GPU, %.1f s)", l + 1, c->n_layer,
                    dev_bytes / 1073741824.0, (now_ms() - t0) / 1e3);
    }
#undef T
    m->out_norm = upload_f32(g, "output_norm.weight", E, 1);
    m->out = upload_one(g, has_tensor(g, "output.weight") ? "output.weight" : "token_embd.weight",
                        E, c->n_vocab);
    m->embd = find_tensor(g, "token_embd.weight", E, c->n_vocab);
    if (!gguf_type_bytes(m->embd->type, E)) {
        fprintf(stderr, "token_embd.weight: type %u is not supported\n", m->embd->type);
        exit(1);
    }

    m->rope = dalloc((size_t)ctx * ROPE_DIMS * 4);
    size_t B = batch;
    size_t wide = 2 * (size_t)F, in = E > F ? E : F;
    if (CONV_CH + DI > wide)
        wide = CONV_CH + DI;
    if (QG + 2 * KV > wide)
        wide = QG + 2 * KV;
    unsigned ro = DI > QH ? DI : QH;
    if (ro > in)
        in = ro;
    m->x = dalloc(B * E * 4);
    m->xn = dalloc(B * in * 4);
    m->big = dalloc(B * wide * 4);
    m->cv = dalloc(B * CONV_CH * 4);
    m->ro = dalloc(B * ro * 4);
    m->ba = dalloc(B * 64 * 4);
    m->xq = dalloc(B * in);
    m->xds = dalloc(B * in / 32 * 8);
    m->logits = dalloc((size_t)c->n_vocab * 4);
    m->next = dalloc(4);
    m->cand = dalloc((2 + 2 * CAND_CAP) * 4);
    m->draw = dalloc(8);
    if (dry_run)
        return;

    float *cs = malloc((size_t)ctx * ROPE_DIMS * sizeof(float));
    rope_table(ROPE_DIMS, ctx, c->rope_base, cs);
    CHECK(cuMemcpyHtoD(m->rope, cs, (size_t)ctx * ROPE_DIMS * 4));
    free(cs);
    m->h_x = malloc(B * E * sizeof(float));
    m->cache_tok = malloc(ctx * sizeof(int32_t));
    m->snap_tok = malloc(ctx * sizeof(int32_t));
    CHECK(cuMemFreeHost(staging));
    free(quant_buf);
    fprintf(stderr, "\rloaded %u layers in %.1f s: %.2f GiB on the GPU (%s weights), context %u, "
                    "batch %u        \n", c->n_layer, (now_ms() - t0) / 1e3,
            dev_bytes / 1073741824.0, type_name(m->layer[0].gate_up.type), ctx, batch);
}

/* Measures, allocates and loads (batch 0: the largest that fits). */
static void model_open(Model *m, const Gguf *g, unsigned ctx, unsigned batch)
{
    size_t free_b, total_b;
    CHECK(cuMemGetInfo(&free_b, &total_b));
    size_t need = 0;
    int fixed = batch != 0;
    for (batch = fixed ? batch : AUTO_MAX_BATCH;; batch /= 2) {
        Model probe = {0};
        dry_run = 1, dev_bytes = 0;
        model_load(&probe, g, ctx, batch);
        model_free_host(&probe);
        need = dev_bytes + (8u << 20);
        if (fixed || batch == 1 || need + AUTO_SPARE_BYTES <= free_b)
            break;
    }
    dry_run = 0, dev_bytes = 0;
    fprintf(stderr, "GPU memory: %.2f GiB needed, %.2f GiB free of %.2f GiB\n", need / 1073741824.0,
            free_b / 1073741824.0, total_b / 1073741824.0);
    if (cuMemAlloc(&arena, need) != CUDA_SUCCESS) {
        fprintf(stderr, "not enough GPU memory; close other GPU programs or use a smaller -c "
                        "or -b\n");
        exit(1);
    }
    arena_size = need;
    model_load(m, g, ctx, batch);
}

/* Embedding of a token: its row of the token matrix (any stored type). */
static void embed_token(const Model *m, int32_t id, float *out)
{
    unsigned E = m->c.n_embd;
    size_t row = gguf_type_bytes(m->embd->type, E);
    dequantize_row(m->embd->type, m->embd->data + (size_t)id * row, out, E);
}

/* --- debug dumps (QWEN35_DUMP=1) --- */

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

/* Runs T (<= m->batch) tokens at positions pos0 .. pos0+T-1 (the recurrent
 * state must hold exactly the tokens before pos0); with want_logits, fills
 * m->logits and m->next (argmax) for the last one. A single token goes
 * through the matrix-vector kernels, a chunk of several through the GEMM,
 * which reads every weight once per chunk; both see the same int8
 * activations. Dumps show the last token. */
static int force_gemm; /* cache-test: single tokens through the GEMM too, for exact comparisons */

/* y (row stride ldy) = W . xq + resid for T tokens */
static void matmul(Model *m, const QW *w, unsigned T, CUdeviceptr resid, CUdeviceptr y, unsigned ldy)
{
    if (T == 1 && !force_gemm)
        mv(w, m->xq, m->xds, resid, y);
    else
        gemm(w, m->xq, m->xds, resid, y, ldy, T);
}

static void forward(Model *m, const int32_t *tokens, unsigned T, unsigned pos0, int want_logits)
{
    const Config *c = &m->c;
    unsigned E = c->n_embd, F = c->n_ff, KV = c->n_kv_head * HEAD_DIM, QG = 2 * c->n_head * HEAD_DIM,
             QH = c->n_head * HEAD_DIM, DI = c->ssm_inner, QKVZ = CONV_CH + DI, QKV = QG + 2 * KV,
             last = T - 1;

    for (unsigned t = 0; t < T; t++)
        embed_token(m, tokens[t], m->h_x + (size_t)t * E);
    CHECK(cuMemcpyHtoD(m->x, m->h_x, (size_t)T * E * 4));
    dump("model.input_embed", -1, m->x + (size_t)last * E * 4, E);

    for (unsigned l = 0; l < c->n_layer; l++) {
        const Layer *L = &m->layer[l];
        rmsnorm(m->x, L->attn_norm, m->xn, E, c->eps, T);
        dump("attn_norm", l, m->xn + (size_t)last * E * 4, E);
        quant_q8(m->xn, m->xq, m->xds, T * E);
        if (L->recurrent) {
            matmul(m, &L->qkvz, T, 0, m->big, QKVZ);
            matmul(m, &L->ba, T, 0, m->ba, 2 * SSM_VH);
            dump("linear_attn_qkv_mixed", l, m->big + (size_t)last * QKVZ * 4, CONV_CH);
            dump("z", l, m->big + ((size_t)last * QKVZ + CONV_CH) * 4, DI);
            gdn_conv(m->big, QKVZ, L->conv_state, L->conv_w, m->cv, CONV_CH, 2 * SSM_KH, c->eps, T);
            conv_save(m->big, QKVZ, L->conv_state, CONV_CH, T);
            gdn_recur(m->cv, m->ba, L->dt_bias, L->ssm_a, L->ssm_state, m->ro, T);
            dump("attn_output", l, m->ro + (size_t)last * DI * 4, DI);
            gated_norm(m->ro, m->big + CONV_CH * 4, QKVZ, L->ssm_norm, m->xn, SSM_VH, c->eps, T);
            dump("final_output", l, m->xn + (size_t)last * DI * 4, DI);
            quant_q8(m->xn, m->xq, m->xds, T * DI);
            matmul(m, &L->ssm_out, T, m->x, m->x, E);
        } else {
            matmul(m, &L->wqkv, T, 0, m->big, QKV);
            qk_norm_rope(m->big, QKV, L->q_norm, L->k_norm, m->rope, L->kc, L->vc, pos0,
                         c->n_head, c->n_kv_head, c->eps, T);
            CUdeviceptr qkv_last = m->big + (size_t)last * QKV * 4;
            dump("Kcur", l, qkv_last + QG * 4, KV);
            dump("Vcur", l, qkv_last + (QG + KV) * 4, KV);
            attention(m->big, QKV, 2 * HEAD_DIM, 1, L->kc, L->vc, m->ro, pos0, m->ctx, c->n_head,
                      c->n_kv_head, c->attn_scale, T);
            dump("attn_gated", l, m->ro + (size_t)last * QH * 4, QH);
            quant_q8(m->ro, m->xq, m->xds, T * QH);
            matmul(m, &L->wo, T, m->x, m->x, E);
        }
        dump("attn_residual", l, m->x + (size_t)last * E * 4, E);

        /* feed-forward: x += down(silu(gate . xn) * (up . xn)), xn = post_norm(x) */
        rmsnorm(m->x, L->post_norm, m->xn, E, c->eps, T);
        dump("attn_post_norm", l, m->xn + (size_t)last * E * 4, E);
        quant_q8(m->xn, m->xq, m->xds, T * E);
        matmul(m, &L->gate_up, T, 0, m->big, 2 * F);
        swiglu(m->big, m->xn, F, T);
        quant_q8(m->xn, m->xq, m->xds, T * F);
        matmul(m, &L->down, T, m->x, m->x, E);
        dump("l_out", l, m->x + (size_t)last * E * 4, E);
    }
    if (!want_logits)
        return;
    CUdeviceptr x_last = m->x + (size_t)last * E * 4;
    rmsnorm(x_last, m->out_norm, m->xn, E, c->eps, 1);
    dump("result_norm", -1, m->xn, E);
    quant_q8(m->xn, m->xq, m->xds, E);
    mv(&m->out, m->xq, m->xds, 0, m->logits);
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

/* Remembers the recurrent states (they hold toks[0..pos)) so prefill() can
 * return to this position later. */
static void cache_snapshot(Model *m, const int32_t *toks, unsigned pos)
{
    CHECK(cuMemcpyDtoD(m->snap, m->rstate, m->rstate_bytes));
    memcpy(m->snap_tok, toks, pos * sizeof(int32_t));
    m->snap_pos = pos;
    m->has_snap = 1;
}

/* Brings the caches to hold toks[0..n) and computes the logits after the
 * last token, reusing what it can. The KV cache is valid for any common
 * prefix, the recurrent states only for exactly the tokens they have seen:
 * so this extends the cache when it holds a prefix of toks, else restarts
 * from the snapshot when that is a prefix of toks, else from position 0.
 * With 0 < snap_at < n, it snapshots the states after toks[0..snap_at). */
static int no_reuse; /* QWEN35_NO_REUSE=1: always recompute from position 0 */

static void prefill(Model *m, const int32_t *toks, unsigned n, unsigned snap_at)
{
    unsigned L = 0, start = 0;
    while (!no_reuse && L + 1 < n && L < m->n_cache && m->cache_tok[L] == toks[L])
        L++; /* the last token always runs, to produce logits */
    if (L == m->n_cache && L > 0) {
        start = L;
    } else if (!no_reuse && m->has_snap && m->snap_pos <= L &&
               !memcmp(m->snap_tok, toks, m->snap_pos * sizeof(int32_t))) {
        CHECK(cuMemcpyDtoD(m->rstate, m->snap, m->rstate_bytes));
        start = m->snap_pos;
    }
    if (start == 0)
        CHECK(cuMemsetD8(m->rstate, 0, m->rstate_bytes));
    for (unsigned i = start; i < n;) {
        if (i == snap_at && snap_at > 0)
            cache_snapshot(m, toks, i);
        unsigned end = n - i < m->batch ? n : i + m->batch;
        if (snap_at > i && snap_at < end)
            end = snap_at;
        if (dump_enabled)
            printf("-- tokens %u..%u (last %d)\n", i, end - 1, toks[end - 1]);
        forward(m, toks + i, end - i, i, end == n);
        i = end;
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
    int think;            /* chat: let the model reason first (thinking mode) */
} GenOptions;

/* The model's end-of-turn / end-of-text tokens. */
static int is_stop(const Tokenizer *tok, int32_t id)
{
    static int32_t eot = -2, eos = -2;
    if (eot == -2) {
        eot = tok_special(tok, "<|im_end|>");
        eos = tok_special(tok, "<|endoftext|>");
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

/* Plain continuation of `prompt` (no chat template; no BOS, as llama.cpp). */
static void run_completion(Model *m, const Tokenizer *tok, const char *prompt, GenOptions *o)
{
    int32_t *ids;
    unsigned n = tok_encode(tok, prompt, 1, &ids);
    if (n == 0 || n >= m->ctx) {
        fprintf(stderr, "prompt must have 1..%u tokens (has %u)\n", m->ctx - 1, n);
        exit(1);
    }
    fprintf(stderr, "prompt: %u tokens\n", n);
    double t0 = now_ms();
    prefill(m, ids, n, 0);
    CHECK(cuCtxSynchronize());
    double t_prompt = now_ms() - t0;
    if (dump_enabled)
        print_top_logits(m, tok);
    const char *logits_path = getenv("QWEN35_LOGITS");
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
    int stable;
    unsigned n_ids = chat_render(tok, o->system, turns, n, o->think, &ids, &stable);
    if (n_ids >= m->ctx) {
        fprintf(stderr, "the conversation (%u tokens) does not fit in the context of %u; "
                        "use /reset or a larger -c\n", n_ids, m->ctx);
        free(ids);
        return NULL;
    }
    /* what prefill will recompute: from the end of the cache, the snapshot, or 0 */
    unsigned L = 0, start = 0;
    while (!no_reuse && L + 1 < n_ids && L < m->n_cache && m->cache_tok[L] == ids[L])
        L++;
    if (L == m->n_cache && L > 0)
        start = L;
    else if (!no_reuse && m->has_snap && m->snap_pos <= L &&
             !memcmp(m->snap_tok, ids, m->snap_pos * sizeof(int32_t)))
        start = m->snap_pos;
    double t0 = now_ms();
    prefill(m, ids, n_ids, (unsigned)stable);
    CHECK(cuCtxSynchronize());
    double t_prompt = now_ms() - t0;
    free(ids);
    if (o->think) { /* the template's "<think>\n", which the model continues */
        printf("<think>\n");
        fflush(stdout);
    }

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
            n_ids, n_ids - start, t_prompt, n_gen, n_gen * 1e3 / t_gen);
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

static CUdeviceptr dzeros(size_t n)
{
    CUdeviceptr d;
    CHECK(cuMemAlloc(&d, n));
    CHECK(cuMemsetD8(d, 0, n));
    return d;
}

static float *download(CUdeviceptr d, size_t n_floats)
{
    float *h = malloc(n_floats * 4);
    CHECK(cuMemcpyDtoH(h, d, n_floats * 4));
    return h;
}

static int report(const char *name, double err, double tol)
{
    int ok = err < tol;
    printf("   %-52s max rel err = %.2e  %s\n", name, err, ok ? "PASS" : "FAIL");
    return !ok;
}

/* Error of got vs ref relative to the largest |ref|. */
static double rel_err(const float *got, const double *ref, size_t n)
{
    double max_err = 0, max_ref = 1e-30;
    for (size_t i = 0; i < n; i++) {
        max_err = fmax(max_err, fabs(got[i] - ref[i]));
        max_ref = fmax(max_ref, fabs(ref[i]));
    }
    return max_err / max_ref;
}

/* CPU int8 quantization of activations, mirroring quant_q8: q [n], d [n/32] */
static void cpu_quant_q8(const float *x, size_t n, int8_t *q, float *d)
{
    for (size_t b = 0; b < n / 32; b++) {
        float amax = 0;
        for (int j = 0; j < 32; j++)
            amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        d[b] = amax / 127;
        float inv = d[b] ? 1 / d[b] : 0;
        for (int j = 0; j < 32; j++)
            q[b * 32 + j] = (int8_t)rintf(x[b * 32 + j] * inv);
    }
}

static void put_half(uint8_t *p, float v)
{
    uint16_t h = float_to_half(v);
    memcpy(p, &h, 2);
}

/* A random [rows][cols] matrix as `type` blocks: random bits, with sane f16
 * scales so the values stay O(1). */
static uint8_t *random_blocks(uint32_t type, unsigned rows, unsigned cols)
{
    size_t bytes = gguf_type_bytes(type, (size_t)rows * cols), nb = rows * blocks_per_row(type, cols);
    uint8_t *b = malloc(bytes);
    for (size_t i = 0; i < bytes; i++)
        b[i] = (uint8_t)rng_u32();
    for (size_t i = 0; i < nb; i++)
        switch (type) {
        case GGML_Q8_0: put_half(b + i * 34, rng_uniform(0.002f, 0.01f)); break;
        case GGML_Q4_1:
            put_half(b + i * 20, rng_uniform(0.01f, 0.1f));
            put_half(b + i * 20 + 2, rng_uniform(-0.8f, 0.0f));
            break;
        case GGML_Q5_K:
            put_half(b + i * 176, rng_uniform(0.0005f, 0.002f));
            put_half(b + i * 176 + 2, rng_uniform(0.0005f, 0.002f));
            break;
        case GGML_Q6_K: put_half(b + i * 210 + 208, rng_uniform(0.0005f, 0.002f)); break;
        }
    return b;
}

/* The GPU matrix made from blocks the way model_load makes it. */
static QW gpu_matrix(uint32_t type, const uint8_t *blk, unsigned rows, unsigned cols)
{
    size_t qb, sb, nb = rows * blocks_per_row(type, cols);
    qw_sizes(type, rows, cols, &qb, &sb);
    uint8_t *q = malloc(qb), *s = malloc(sb + 1);
    split_blocks(type, blk, nb, q, s);
    QW w = {type, upload(q, qb), sb ? upload(s, sb) : 0, rows, cols};
    free(q), free(s);
    return w;
}

static void free_matrix(QW *w)
{
    cuMemFree(w->q);
    if (w->s)
        cuMemFree(w->s);
}

static int test_quant_formats(void)
{
    /* round trips within half a step (plus f16 rounding of the scales);
     * quant_q8 on the GPU must match the CPU bit for bit */
    const unsigned n = 4096;
    float *x = malloc(n * 4), *y = malloc(n * 4);
    for (unsigned i = 0; i < n; i++)
        x[i] = rng_uniform(-1, 1) * (i % 97 == 0 ? 8 : 1);
    uint8_t *b = malloc(n / 32 * 34);
    double e8 = 0, e4 = 0;
    quantize_q8_0(x, b, n);
    dequantize_q8_0(b, y, n);
    for (unsigned i = 0; i < n; i++) {
        float amax = 0;
        for (unsigned j = i / 32 * 32; j < i / 32 * 32 + 32; j++)
            amax = fmaxf(amax, fabsf(x[j]));
        e8 = fmax(e8, fabs(y[i] - x[i]) / (amax / 127));
    }
    quantize_q4_1(x, b, n);
    dequantize_q4_1(b, y, n);
    for (unsigned i = 0; i < n; i++) {
        float lo = INFINITY, hi = -INFINITY;
        for (unsigned j = i / 32 * 32; j < i / 32 * 32 + 32; j++)
            lo = fminf(lo, x[j]), hi = fmaxf(hi, x[j]);
        e4 = fmax(e4, fabs(y[i] - x[i]) / ((hi - lo) / 15));
    }
    int fails = 0, ok = e8 < 0.6 && e4 < 0.6;
    printf("   %-52s %.3f / %.3f steps  %s\n", "Q8_0 / Q4_1 round trip (max error)", e8, e4,
           ok ? "PASS" : "FAIL");
    fails += !ok;

    int8_t *q = malloc(n), *gq = malloc(n);
    float *d = malloc(n / 32 * 4), *gds = malloc(n / 32 * 8);
    cpu_quant_q8(x, n, q, d);
    CUdeviceptr d_x = upload(x, n * 4), d_q = dzeros(n), d_ds = dzeros(n / 32 * 8);
    quant_q8(d_x, d_q, d_ds, n);
    CHECK(cuMemcpyDtoH(gq, d_q, n));
    CHECK(cuMemcpyDtoH(gds, d_ds, n / 32 * 8));
    ok = !memcmp(q, gq, n);
    for (unsigned blk = 0; blk < n / 32; blk++) {
        int sum = 0;
        for (int j = 0; j < 32; j++)
            sum += q[blk * 32 + j];
        ok &= gds[2 * blk] == d[blk] && gds[2 * blk + 1] == (float)sum * d[blk];
    }
    printf("   %-52s %s\n", "quant_q8 (int8 blocks, d, d * sum) vs CPU", ok ? "PASS" : "FAIL");
    fails += !ok;
    cuMemFree(d_x), cuMemFree(d_q), cuMemFree(d_ds);
    free(x), free(y), free(b), free(q), free(gq), free(d), free(gds);
    return fails;
}

/* x [T][cols] -> its int8 blocks on the GPU and their values on the CPU */
static void test_activations(unsigned T, unsigned cols, CUdeviceptr *d_q, CUdeviceptr *d_ds,
                             float **xv)
{
    size_t n = (size_t)T * cols;
    float *x = malloc(n * 4), *d = malloc(n / 32 * 4), *ds = malloc(n / 32 * 8);
    int8_t *q = malloc(n);
    for (size_t i = 0; i < n; i++)
        x[i] = rng_uniform(-2, 2) * (i % 61 == 0 ? 6 : 1);
    cpu_quant_q8(x, n, q, d);
    for (size_t b = 0; b < n / 32; b++) {
        int sum = 0;
        for (int j = 0; j < 32; j++)
            sum += q[b * 32 + j];
        ds[2 * b] = d[b];
        ds[2 * b + 1] = (float)sum * d[b];
    }
    for (size_t i = 0; i < n; i++)
        x[i] = q[i] * d[i / 32];
    *d_q = upload(q, n);
    *d_ds = upload(ds, n / 32 * 8);
    *xv = x;
    free(d), free(ds), free(q);
}

/* mv_<type> (37 rows, with a residual) vs the dequantized weights in double */
static int test_mv(uint32_t type)
{
    const unsigned rows = 37, cols = 4096;
    uint8_t *blk = random_blocks(type, rows, cols);
    float *w = malloc((size_t)rows * cols * 4), *resid = malloc(rows * 4), *x;
    dequantize_row(type, blk, w, (size_t)rows * cols);
    for (unsigned i = 0; i < rows; i++)
        resid[i] = rng_uniform(-1, 1);
    CUdeviceptr d_q, d_ds, d_r = upload(resid, rows * 4), d_y = dzeros(rows * 4);
    test_activations(1, cols, &d_q, &d_ds, &x);
    double *ref = malloc(rows * sizeof(double));
    for (unsigned r = 0; r < rows; r++) {
        double acc = resid[r];
        for (unsigned k = 0; k < cols; k++)
            acc += (double)w[(size_t)r * cols + k] * x[k];
        ref[r] = acc;
    }
    QW g = gpu_matrix(type, blk, rows, cols);
    mv(&g, d_q, d_ds, d_r, d_y);
    float *y = download(d_y, rows);
    char name[64];
    snprintf(name, sizeof(name), "mv %s vs dequantized reference", type_name(type));
    int fails = report(name, rel_err(y, ref, rows), 2e-5);
    free_matrix(&g);
    cuMemFree(d_q), cuMemFree(d_ds), cuMemFree(d_r), cuMemFree(d_y);
    free(blk), free(w), free(resid), free(x), free(ref), free(y);
    return fails;
}

/* gemm_q (T tokens, strided output, in-place residual) vs the dequantized
 * weights in double */
static int test_gemm(uint32_t type, unsigned T)
{
    const unsigned rows = 128, cols = 512, off = 32, ldy = rows + 64;
    uint8_t *blk = random_blocks(type, rows, cols);
    float *w = malloc((size_t)rows * cols * 4), *y = malloc(T * ldy * 4), *x;
    dequantize_row(type, blk, w, (size_t)rows * cols);
    for (unsigned i = 0; i < T * ldy; i++)
        y[i] = rng_uniform(-1, 1); /* residual, updated in place */
    CUdeviceptr d_q, d_ds, d_y = upload(y, T * ldy * 4);
    test_activations(T, cols, &d_q, &d_ds, &x);
    double *ref = malloc(T * ldy * sizeof(double));
    for (unsigned i = 0; i < T * ldy; i++)
        ref[i] = y[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned r = 0; r < rows; r++) {
            double acc = 0;
            for (unsigned k = 0; k < cols; k++)
                acc += (double)w[r * cols + k] * x[t * cols + k];
            ref[t * ldy + off + r] += acc;
        }
    QW g = gpu_matrix(type, blk, rows, cols);
    CUdeviceptr y_slice = d_y + off * 4;
    gemm(&g, d_q, d_ds, y_slice, y_slice, ldy, T);
    CHECK(cuMemcpyDtoH(y, d_y, T * ldy * 4));
    char name[64];
    snprintf(name, sizeof(name), "gemm %s (%u tokens, strided, in place)", type_name(type), T);
    int fails = report(name, rel_err(y, ref, T * ldy), 2e-5);
    free_matrix(&g);
    cuMemFree(d_q), cuMemFree(d_ds), cuMemFree(d_y);
    free(blk), free(w), free(y), free(x), free(ref);
    return fails;
}

static double silu_d(double v)
{
    return v / (1 + exp(-v));
}

/* gdn_conv + conv_save over a chunk of T tokens after an existing state */
static int test_gdn_conv(unsigned T)
{
    const unsigned C = 512, nnorm = 2, ld = C + 64;
    const float eps = 1e-6f;
    float *in = malloc(T * ld * 4), *state = malloc(3 * C * 4), *w = malloc(C * 4 * 4);
    for (unsigned i = 0; i < T * ld; i++)
        in[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < 3 * C; i++)
        state[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < 4 * C; i++)
        w[i] = rng_uniform(-0.6f, 0.6f);
    double *ref = malloc(T * C * sizeof(double)), *ref_state = malloc(3 * C * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        for (unsigned c = 0; c < C; c++) {
            double acc = 0;
            for (unsigned s = 0; s < 4; s++) {
                int u = (int)t + (int)s - 3;
                acc += (double)w[c * 4 + s] * (u >= 0 ? in[u * ld + c] : state[(3 + u) * C + c]);
            }
            ref[t * C + c] = silu_d(acc);
        }
        for (unsigned h = 0; h < nnorm; h++) {
            double ss = 0;
            for (unsigned i = 0; i < 128; i++)
                ss += ref[t * C + h * 128 + i] * ref[t * C + h * 128 + i];
            for (unsigned i = 0; i < 128; i++)
                ref[t * C + h * 128 + i] /= sqrt(ss + eps);
        }
    }
    for (unsigned r = 0; r < 3; r++)
        for (unsigned c = 0; c < C; c++) {
            unsigned src = T + r;
            ref_state[r * C + c] = src < 3 ? state[src * C + c] : in[(src - 3) * ld + c];
        }
    CUdeviceptr d_in = upload(in, T * ld * 4), d_st = upload(state, 3 * C * 4),
                d_w = upload(w, C * 16), d_out = dzeros(T * C * 4);
    gdn_conv(d_in, ld, d_st, d_w, d_out, C, nnorm, eps, T);
    conv_save(d_in, ld, d_st, C, T);
    float *out = download(d_out, T * C), *st = download(d_st, 3 * C);
    char name[64];
    snprintf(name, sizeof(name), "gdn_conv (%u tokens, L2-normed q/k heads)", T);
    int fails = report(name, rel_err(out, ref, T * C), 1e-5);
    snprintf(name, sizeof(name), "conv_save (%u tokens)", T);
    fails += report(name, rel_err(st, ref_state, 3 * C), 1e-12);
    cuMemFree(d_in), cuMemFree(d_st), cuMemFree(d_w), cuMemFree(d_out);
    free(in), free(state), free(w), free(ref), free(ref_state), free(out), free(st);
    return fails;
}

/* gdn_recur over T tokens from a random state vs a CPU reference */
static int test_gdn_recur(unsigned T)
{
    const unsigned D = SSM_DIM, H = SSM_VH, NS = H * D * D;
    float *qkv = malloc(T * CONV_CH * 4), *ba = malloc(T * 64 * 4), *dt = malloc(H * 4),
          *a = malloc(H * 4), *state = malloc(NS * 4);
    for (unsigned t = 0; t < T; t++) {
        for (unsigned i = 0; i < CONV_CH; i++)
            qkv[t * CONV_CH + i] = rng_uniform(-1, 1);
        for (unsigned h = 0; h < 2 * SSM_KH; h++) { /* q, k heads L2-normalized as in the model */
            double ss = 0;
            float *v = qkv + t * CONV_CH + h * D;
            for (unsigned i = 0; i < D; i++)
                ss += v[i] * v[i];
            for (unsigned i = 0; i < D; i++)
                v[i] /= (float)sqrt(ss);
        }
        for (unsigned i = 0; i < 64; i++)
            ba[t * 64 + i] = rng_uniform(-4, 4) + (i == 32 ? 30 : 0); /* one alpha past softplus' cutoff */
    }
    for (unsigned h = 0; h < H; h++) {
        dt[h] = rng_uniform(-3, 3);
        a[h] = -rng_uniform(0.01f, 0.15f);
    }
    for (unsigned i = 0; i < NS; i++)
        state[i] = rng_uniform(-0.5f, 0.5f);
    double *S = malloc(NS * sizeof(double)), *ref = malloc(T * H * D * sizeof(double));
    for (unsigned i = 0; i < NS; i++)
        S[i] = state[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned h = 0; h < H; h++) {
            const float *q = qkv + t * CONV_CH + (h % SSM_KH) * D, *k = q + SSM_KH * D,
                        *v = qkv + t * CONV_CH + 2 * SSM_KH * D + h * D;
            double b = 1 / (1 + exp(-(double)ba[t * 64 + h])), x = (double)ba[t * 64 + 32 + h] + dt[h];
            double g = exp((x > 20 ? x : log(1 + exp(x))) * a[h]);
            double *Sh = S + (size_t)h * D * D;
            for (unsigned j = 0; j < D; j++) {
                double kv = 0, o = 0;
                for (unsigned i = 0; i < D; i++)
                    kv += Sh[i * D + j] * k[i];
                double delta = (v[j] - g * kv) * b;
                for (unsigned i = 0; i < D; i++) {
                    Sh[i * D + j] = g * Sh[i * D + j] + k[i] * delta;
                    o += Sh[i * D + j] * q[i];
                }
                ref[(t * H + h) * D + j] = o / sqrt((double)D);
            }
        }
    CUdeviceptr d_qkv = upload(qkv, T * CONV_CH * 4), d_ba = upload(ba, T * 64 * 4),
                d_dt = upload(dt, H * 4), d_a = upload(a, H * 4), d_st = upload(state, NS * 4),
                d_out = dzeros(T * H * D * 4);
    gdn_recur(d_qkv, d_ba, d_dt, d_a, d_st, d_out, T);
    float *out = download(d_out, T * H * D), *st = download(d_st, NS);
    char name[64];
    snprintf(name, sizeof(name), "gdn_recur (%u tokens): outputs", T);
    int fails = report(name, rel_err(out, ref, T * H * D), 1e-4);
    snprintf(name, sizeof(name), "gdn_recur (%u tokens): final state", T);
    fails += report(name, rel_err(st, S, NS), 1e-4);
    cuMemFree(d_qkv), cuMemFree(d_ba), cuMemFree(d_dt), cuMemFree(d_a), cuMemFree(d_st),
        cuMemFree(d_out);
    free(qkv), free(ba), free(dt), free(a), free(state), free(S), free(ref), free(out), free(st);
    return fails;
}

/* rmsnorm, gated_norm, swiglu vs CPU references */
static int test_norms(void)
{
    const unsigned n = 4096, T = 3, heads = 32, ldz = 4096 + 128, F = 1000;
    const float eps = 1e-6f;
    float *x = malloc(T * n * 4), *w = malloc(n * 4), *z = malloc(T * ldz * 4), *gw = malloc(128 * 4);
    for (unsigned i = 0; i < T * n; i++)
        x[i] = rng_uniform(-5, 5);
    for (unsigned i = 0; i < n; i++)
        w[i] = rng_uniform(0, 3);
    for (unsigned i = 0; i < T * ldz; i++)
        z[i] = rng_uniform(-6, 6);
    for (unsigned i = 0; i < 128; i++)
        gw[i] = rng_uniform(0.5f, 1.5f);
    double *rn = malloc(T * n * sizeof(double)), *rg = malloc(T * n * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        double ss = 0;
        for (unsigned i = 0; i < n; i++)
            ss += (double)x[t * n + i] * x[t * n + i];
        for (unsigned i = 0; i < n; i++)
            rn[t * n + i] = x[t * n + i] / sqrt(ss / n + eps) * w[i];
        for (unsigned h = 0; h < heads; h++) {
            double s2 = 0;
            for (unsigned j = 0; j < 128; j++)
                s2 += (double)x[t * n + h * 128 + j] * x[t * n + h * 128 + j];
            for (unsigned j = 0; j < 128; j++)
                rg[t * n + h * 128 + j] = x[t * n + h * 128 + j] / sqrt(s2 / 128 + eps) * gw[j] *
                                          silu_d(z[t * ldz + h * 128 + j]);
        }
    }
    CUdeviceptr d_x = upload(x, T * n * 4), d_w = upload(w, n * 4), d_z = upload(z, T * ldz * 4),
                d_gw = upload(gw, 128 * 4), d_y = dzeros(T * n * 4);
    rmsnorm(d_x, d_w, d_y, n, eps, T);
    float *y = download(d_y, T * n);
    int fails = report("rmsnorm", rel_err(y, rn, T * n), 1e-5);
    free(y);
    gated_norm(d_x, d_z, ldz, d_gw, d_y, heads, eps, T);
    y = download(d_y, T * n);
    fails += report("gated_norm (rmsnorm * w * silu(z), strided z)", rel_err(y, rg, T * n), 1e-5);
    free(y);

    float *gu = malloc(T * 2 * F * 4);
    double *rh = malloc(T * F * sizeof(double));
    for (unsigned i = 0; i < T * 2 * F; i++)
        gu[i] = rng_uniform(-6, 6);
    for (unsigned t = 0; t < T; t++)
        for (unsigned j = 0; j < F; j++)
            rh[t * F + j] = silu_d(gu[t * 2 * F + j]) * gu[t * 2 * F + F + j];
    CUdeviceptr d_gu = upload(gu, T * 2 * F * 4), d_h = dzeros(T * F * 4);
    swiglu(d_gu, d_h, F, T);
    y = download(d_h, T * F);
    fails += report("swiglu", rel_err(y, rh, T * F), 1e-6);
    cuMemFree(d_x), cuMemFree(d_w), cuMemFree(d_z), cuMemFree(d_gw), cuMemFree(d_y),
        cuMemFree(d_gu), cuMemFree(d_h);
    free(x), free(w), free(z), free(gw), free(rn), free(rg), free(gu), free(rh), free(y);
    return fails;
}

/* q/k head norm + partial RoPE and the K/V cache writes vs a CPU reference */
static int test_qk_norm_rope(void)
{
    const unsigned nh = 4, nkv = 2, D = HEAD_DIM, T = 3, pos0 = 70, ctx = 80;
    const unsigned stride = nh * 2 * D + 2 * nkv * D + 16;
    const float eps = 1e-6f;
    float *qkv = malloc(T * stride * 4), *qw = malloc(D * 4), *kw = malloc(D * 4);
    float *cs = malloc((size_t)ctx * ROPE_DIMS * 4);
    for (unsigned i = 0; i < T * stride; i++)
        qkv[i] = rng_uniform(-3, 3);
    for (unsigned i = 0; i < D; i++) {
        qw[i] = rng_uniform(0.5f, 2);
        kw[i] = rng_uniform(0.5f, 2);
    }
    rope_table(ROPE_DIMS, ctx, 1e7f, cs);
    double *ref = malloc(T * stride * sizeof(double));
    for (unsigned i = 0; i < T * stride; i++)
        ref[i] = qkv[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned h = 0; h < nh + nkv; h++) {
            unsigned off = t * stride + (h < nh ? h * 2 * D : nh * 2 * D + (h - nh) * D);
            const float *x = qkv + off, *w = h < nh ? qw : kw;
            double *r = ref + off, ss = 0, y[HEAD_DIM];
            for (unsigned i = 0; i < D; i++)
                ss += (double)x[i] * x[i];
            for (unsigned i = 0; i < D; i++)
                y[i] = x[i] / sqrt(ss / D + eps) * w[i];
            for (unsigned i = 0; i < D; i++)
                r[i] = y[i];
            for (unsigned i = 0; i < ROPE_DIMS / 2; i++) {
                double c = cs[(size_t)(pos0 + t) * ROPE_DIMS + 2 * i],
                       sn = cs[(size_t)(pos0 + t) * ROPE_DIMS + 2 * i + 1];
                r[i] = y[i] * c - y[i + ROPE_DIMS / 2] * sn;
                r[i + ROPE_DIMS / 2] = y[i] * sn + y[i + ROPE_DIMS / 2] * c;
            }
        }
    CUdeviceptr d_qkv = upload(qkv, T * stride * 4), d_qw = upload(qw, D * 4), d_kw = upload(kw, D * 4),
                d_cs = upload(cs, (size_t)ctx * ROPE_DIMS * 4), d_kc = dzeros(ctx * nkv * D * 2),
                d_vc = dzeros(ctx * nkv * D * 2);
    qk_norm_rope(d_qkv, stride, d_qw, d_kw, d_cs, d_kc, d_vc, pos0, nh, nkv, eps, T);
    float *out = download(d_qkv, T * stride);
    int fails = report("qk_norm_rope (q, k in place; gates, v untouched)",
                       rel_err(out, ref, T * stride), 1e-5);
    uint16_t *kc = malloc(ctx * nkv * D * 2), *vc = malloc(ctx * nkv * D * 2);
    CHECK(cuMemcpyDtoH(kc, d_kc, ctx * nkv * D * 2));
    CHECK(cuMemcpyDtoH(vc, d_vc, ctx * nkv * D * 2));
    float *got = malloc(T * 2 * nkv * D * 4);
    double *want = malloc(T * 2 * nkv * D * sizeof(double));
    for (unsigned t = 0; t < T; t++)
        for (unsigned i = 0; i < nkv * D; i++) {
            unsigned pos = pos0 + t;
            got[(t * 2) * nkv * D + i] = half_to_float(kc[pos * nkv * D + i]);
            got[(t * 2 + 1) * nkv * D + i] = half_to_float(vc[pos * nkv * D + i]);
            want[(t * 2) * nkv * D + i] = ref[t * stride + nh * 2 * D + i];
            want[(t * 2 + 1) * nkv * D + i] = ref[t * stride + nh * 2 * D + nkv * D + i];
        }
    fails += report("qk_norm_rope (f16 K/V cache)", rel_err(got, want, T * 2 * nkv * D), 1e-3);
    cuMemFree(d_qkv), cuMemFree(d_qw), cuMemFree(d_kw), cuMemFree(d_cs), cuMemFree(d_kc),
        cuMemFree(d_vc);
    free(qkv), free(qw), free(kw), free(cs), free(out), free(ref), free(kc), free(vc), free(got),
        free(want);
    return fails;
}

/* Causal attention for T queries at pos0.. with the q|gate head layout and
 * the output gate, vs a CPU reference. */
static int test_attention_case(unsigned pos0, unsigned T)
{
    const unsigned nh = 8, nkv = 2, D = HEAD_DIM, qhs = 2 * D, q_stride = nh * qhs + 16,
                   npos = pos0 + T, slots = npos + 3;
    const float scale = 0.0625f;
    float *q = malloc(T * q_stride * 4), *out = malloc(T * nh * D * 4);
    uint16_t *kc = calloc(slots * nkv * D, 2), *vc = calloc(slots * nkv * D, 2);
    for (unsigned i = 0; i < T * q_stride; i++)
        q[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < npos * nkv * D; i++) {
        kc[i] = float_to_half(rng_uniform(-2, 2));
        vc[i] = float_to_half(rng_uniform(-1, 1));
    }
    double *ref = malloc(T * nh * D * sizeof(double)), *s = malloc(npos * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        unsigned p = pos0 + t;
        for (unsigned h = 0; h < nh; h++) {
            unsigned kh = h / (nh / nkv);
            const float *qh = q + t * q_stride + h * qhs;
            double m = -1e300, sum = 0;
            for (unsigned j = 0; j <= p; j++) {
                double dot = 0;
                for (unsigned d = 0; d < D; d++)
                    dot += qh[d] * scale * half_to_float(kc[(j * nkv + kh) * D + d]);
                s[j] = dot;
                m = fmax(m, dot);
            }
            for (unsigned j = 0; j <= p; j++)
                sum += exp(s[j] - m);
            for (unsigned d = 0; d < D; d++) {
                double acc = 0;
                for (unsigned j = 0; j <= p; j++)
                    acc += exp(s[j] - m) * half_to_float(vc[(j * nkv + kh) * D + d]);
                ref[(t * nh + h) * D + d] = acc / sum / (1 + exp(-(double)qh[D + d]));
            }
        }
    }
    CUdeviceptr d_q = upload(q, T * q_stride * 4), d_k = upload(kc, slots * nkv * D * 2),
                d_v = upload(vc, slots * nkv * D * 2), d_o = dzeros(T * nh * D * 4);
    attention(d_q, q_stride, qhs, 1, d_k, d_v, d_o, pos0, slots, nh, nkv, scale, T);
    CHECK(cuMemcpyDtoH(out, d_o, T * nh * D * 4));
    char name[64];
    snprintf(name, sizeof(name), "attention + gate T=%u pos0=%u", T, pos0);
    int fails = report(name, rel_err(out, ref, T * nh * D), 1e-4);
    cuMemFree(d_q), cuMemFree(d_k), cuMemFree(d_v), cuMemFree(d_o);
    free(q), free(out), free(kc), free(vc), free(ref), free(s);
    return fails;
}

static int test_attention(void)
{
    return test_attention_case(0, 1) +     /* first token: one key */
           test_attention_case(76, 1) +    /* decoding */
           test_attention_case(300, 1) +   /* many keys per warp */
           test_attention_case(0, 9) +     /* first chunk of a prompt */
           test_attention_case(40, 7);     /* a later chunk */
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
    printf("   %-52s %u of %u  %s\n", "compact_logits candidates", out[0], want, ok ? "PASS" : "FAIL");
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
        printf("   %-52s max |freq - p| = %.4f  %s\n", cfg[c].name, worst,
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
        printf("   %-52s max |freq - p| = %.4f  %s\n",
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
    int fails = test_quant_formats();
    uint32_t types[] = {GGML_Q4_1, GGML_Q8_0, GGML_Q5_K, GGML_Q6_K};
    for (int i = 0; i < 4; i++)
        fails += test_mv(types[i]);
    for (int i = 0; i < 3; i++)
        fails += test_gemm(types[i], 37) + test_gemm(types[i], 100);
    fails += test_gdn_conv(5) + test_gdn_conv(2) + test_gdn_conv(1) + test_gdn_recur(1) +
             test_gdn_recur(4) + test_norms() + test_qk_norm_rope() + test_attention() +
             test_sampling();
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

/* The default model: the Q4_1 file, else the Q8_0 one `quantize` writes, else BF16. */
static const char *default_model(void)
{
    if (!access(DEFAULT_MODEL, R_OK))
        return DEFAULT_MODEL;
    if (!access(Q8_MODEL, R_OK))
        return Q8_MODEL;
    if (!access(BF16_MODEL, R_OK)) {
        fprintf(stderr, "note: %s not found, loading %s (quantized to Q8_0 while loading; "
                        "`make quantize` writes %s once)\n", DEFAULT_MODEL, BF16_MODEL, Q8_MODEL);
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
 * (control-token text is parsed only with --special) */
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
 * bit, as a fresh computation when it reuses the caches: extending a cached
 * prompt, rewinding to the recurrent-state snapshot, and rewinding without
 * one (recompute). Uses tale.txt as text and the GEMM path for every chunk,
 * so all runs do identical arithmetic per token; chunk boundaries differ. */
static int cmd_cache_test(const char *model_path, const char *cubin)
{
    init_cuda(cubin);
    Gguf g;
    gguf_open(&g, model_path);
    Tokenizer *tok = tok_load(&g);
    Model m = {0};
    model_open(&m, &g, 2048, 64);
    force_gemm = 1;
    char *tale = read_file("tale.txt");
    int32_t *body;
    unsigned nb = tok_encode(tok, tale, 0, &body), nv = m.c.n_vocab;
    const unsigned P = 601, R = 300; /* "prompt", "reply" */
    if (nb < P + R + 1) {
        fprintf(stderr, "tale.txt is too short\n");
        return 1;
    }
    int32_t *a = malloc((P + R) * sizeof(int32_t)), *b = malloc((P + R) * sizeof(int32_t));
    memcpy(a, body, (P + R) * sizeof(int32_t));
    memcpy(b, a, (P + R) * sizeof(int32_t));
    for (unsigned i = P + 5; i < P + R; i++) /* b: a with the reply changed from P + 5 on */
        b[i] = body[(i * 7) % nb];
    float *ref = malloc(nv * 4), *got = malloc(nv * 4);
    int fails = 0;
#define FRESH(seq, n)                                                         \
    do {                                                                      \
        m.n_cache = 0, m.has_snap = 0;                                        \
        prefill(&m, seq, n, 0);                                               \
        CHECK(cuMemcpyDtoH(ref, m.logits, nv * 4));                           \
    } while (0)
#define CHECK_SAME(name)                                                      \
    do {                                                                      \
        CHECK(cuMemcpyDtoH(got, m.logits, nv * 4));                           \
        int same = !memcmp(ref, got, nv * 4);                                 \
        printf("   %-60s %s\n", name, same ? "PASS" : "FAIL");                \
        fails += !same;                                                       \
    } while (0)
    /* 1. plain extension of the cache */
    FRESH(a, P + R);
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P, 0);
    prefill(&m, a, P + R, 0);
    CHECK_SAME("extend a cached prompt by 300 tokens");
    /* 2. rewind to the snapshot taken at the end of the prompt */
    FRESH(b, P + R);
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P + R, P);
    prefill(&m, b, P + R, 0);
    CHECK_SAME("rewind 295 positions to the recurrent-state snapshot");
    /* 3. rewind without a snapshot: recompute from position 0 */
    m.n_cache = 0, m.has_snap = 0;
    prefill(&m, a, P + R, 0);
    prefill(&m, b, P + R, 0);
    CHECK_SAME("rewind 295 positions without a snapshot (recompute)");
#undef FRESH
#undef CHECK_SAME
    printf("%s\n", fails ? "CACHE TEST FAILED" : "cache reuse is exact");
    return fails;
}

/* `render-ids [--sys TEXT] [--think] USER [ASSISTANT USER ...]`: the token ids
 * of a conversation in the chat format, ready for the model's reply (tests) */
static int cmd_render_ids(int argc, char **argv)
{
    const char *system = NULL;
    int i = 2, think = 0;
    for (;;) {
        if (i + 1 < argc && !strcmp(argv[i], "--sys")) {
            system = argv[i + 1];
            i += 2;
        } else if (i < argc && !strcmp(argv[i], "--think")) {
            think = 1;
            i++;
        } else {
            break;
        }
    }
    int n = argc - i;
    ChatTurn *turns = malloc((n > 0 ? n : 1) * sizeof(ChatTurn));
    for (int k = 0; k < n; k++)
        turns[k] = (ChatTurn){k % 2 == 0, argv[i + k]};
    Gguf g;
    gguf_open(&g, default_model());
    Tokenizer *tok = tok_load(&g);
    int32_t *ids;
    int m = chat_render(tok, system, turns, n, think, &ids, NULL);
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
    const char *in = argc >= 3 ? argv[2] : BF16_MODEL, *out = argc >= 4 ? argv[3] : Q8_MODEL;
    Gguf g;
    gguf_open(&g, in);
    fprintf(stderr, "quantizing %s -> %s (Q8_0 matrices)\n", in, out);
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
            "       %s quantize [IN] [OUT]      BF16 GGUF -> Q8_0 GGUF (default\n"
            "                                   " BF16_MODEL " -> " Q8_MODEL ")\n"
            "       %s test | cache-test        kernel self-tests | cache reuse test\n"
            "       %s tokenize TEXT [model.gguf] | tokenize-ids [--special] [model.gguf] < lines\n"
            "       %s render-ids [--sys TEXT] [--think] USER [ASSISTANT USER ...]   chat token ids\n"
            "options:\n"
            "  -m FILE        model (default " DEFAULT_MODEL ", else " Q8_MODEL ", else " BF16_MODEL ")\n"
            "  -n N           max new tokens (default: 64 for PROMPT, until done for chat)\n"
            "  -c N           context length (default 8192)\n"
            "  -b N           prompt chunk size (default: largest that fits, up to 512)\n"
            "  -k FILE        kernels (default " DEFAULT_CUBIN ")\n"
            "  --think        chat: thinking mode (the model reasons inside <think> first)\n"
            "  --temp T       temperature (default 0.7; 1.0 with --think; 0 = greedy)\n"
            "  --top-k K      keep the K most likely tokens (default 20; 0 = off)\n"
            "  --top-p P      nucleus sampling (default 0.8; 0.95 with --think; 1 = off)\n"
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
    int test = 0, interactive = 0, print_prompt = 0, temp_set = 0, top_p_set = 0;
    GenOptions o = {.sampler = {.temperature = 0.7f, .top_k = 20, .top_p = 0.8f}, .max_new = -1};
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
            o.sampler.temperature = atof(v), temp_set = 1;
        else if (OPT("--top-k"))
            o.sampler.top_k = atoi(v);
        else if (OPT("--top-p"))
            o.sampler.top_p = atof(v), top_p_set = 1;
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
        else if (!strcmp(a, "--think"))
            o.think = 1;
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
    if (o.think) { /* Qwen's recommended sampling for thinking mode */
        if (!temp_set)
            o.sampler.temperature = 1.0f;
        if (!top_p_set)
            o.sampler.top_p = 0.95f;
    }
    if (!model_path)
        model_path = default_model();
    if (print_prompt && chat) { /* no GPU needed */
        Gguf g;
        gguf_open(&g, model_path);
        Tokenizer *tok = tok_load(&g);
        ChatTurn turn = {1, chat};
        int32_t *ids;
        int n = chat_render(tok, o.system, &turn, 1, o.think, &ids, NULL);
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
    dump_enabled = getenv("QWEN35_DUMP") && atoi(getenv("QWEN35_DUMP"));
    no_reuse = getenv("QWEN35_NO_REUSE") && atoi(getenv("QWEN35_NO_REUSE"));

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
    model_open(&m, &g, ctx, batch);

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
