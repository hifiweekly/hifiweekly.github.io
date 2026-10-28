/*
 * emb2.c - EmbeddingGemma 2 (text) sentence embeddings on the GPU with
 * hand-written PTX kernels (emb2.ptx) driven through the CUDA Driver API. No
 * other libraries. Built on the Gemma 3 engine in ../gemma3: same GGUF
 * reader and Q8_0 code (gguf.c, quant.c, copied), same launch style, and its
 * rmsnorm / add_rmsnorm / geglu / gemm_q8 kernels.
 *
 * The model ("gemma-embedding2", 271M parameters, 24 layers) is a
 * bidirectional Gemma 4 style encoder:
 *   - 512-wide hidden state, 4 query heads; 5 of every 6 layers attend
 *     within +-512 positions with 256-dim heads (2 KV heads, RoPE base 1e4),
 *     the 6th attends to the whole text with 512-dim heads (1 KV head, RoPE
 *     base 1e6); q and k get RMS head norms, v an RMS norm without weight,
 *     and the scores are not scaled by 1/sqrt(d)
 *   - Gemma's sandwich norms around attention and the GeGLU feed-forward
 *   - per-layer inputs: 512 values per layer projected from the token
 *     embedding, gated into the residual stream after every feed-forward,
 *     and a learned scalar per layer
 *   - a final 512 -> 768 projection, mean-pooled over the tokens (<bos> and
 *     <eos> included) and L2-normalized: the 768-dim text embedding
 * The mmproj file next to the model holds the image and audio encoders,
 * which produce input embeddings for the same text encoder; see README.md
 * (not implemented here).
 *
 * Weights on the GPU (Q8_0: int8 + f16 scale per 32 values, ~0.15 GiB);
 * norms in F32; the token embedding matrix (262144 x 512, 70% of the
 * parameters) stays in host memory and is read one row per token. A BF16
 * GGUF is quantized to Q8_0 at load time (except the token embeddings,
 * which are used exactly).
 *
 * Several texts are packed into one batch of up to -b tokens: every token
 * carries its position within its text and the [start, end) of its text,
 * and attention never crosses texts.
 *
 * Usage (see usage() for every option):
 *   ./emb2 [options] TEXT...           embed each TEXT, print a summary
 *   ./emb2 [options] -f FILE           embed each non-empty line of FILE
 *   ./emb2 sim [options] TEXT...       cosine similarity matrix
 *   ./emb2 test                        kernel self-tests against the CPU
 *   ./emb2 tokenize TEXT [model]       show the tokens of TEXT
 * Environment (debugging): EMB2_DUMP=1 prints a summary of intermediate
 * tensors of the last token, named like llama.cpp's llama-eval-callback.
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "gguf.h"
#include "quant.h"
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

#define DEFAULT_MODEL "Q8_0/embeddinggemma-2-Q8_0.gguf"
#define BF16_MODEL "embeddinggemma-2-BF16.gguf"
#define DEFAULT_CUBIN "emb2.cubin"
#define MAX_LAYERS 64
#define MAX_HEAD_DIM 512     /* attention_enc keeps q in 512 floats of shared memory */
#define DEFAULT_BATCH 2048   /* tokens per forward pass (also the longest text) */
#define MAX_BATCH 8192       /* attention_enc keeps the scores of a row in shared memory */

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

static CUfunction k_rmsnorm, k_add_rmsnorm, k_geglu, k_gemm_q8, k_hnr, k_attn, k_gelu_mul,
    k_norm_add, k_scale;
static unsigned max_blocks; /* cap for grid-stride kernels: 8 blocks per SM */

static void load_kernels(const char *cubin)
{
    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, cubin);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cannot load %s (run `make` in the gemma_emb2 directory)\n", cubin);
        exit(1);
    }
    struct { CUfunction *fn; const char *name; } k[] = {
        {&k_rmsnorm, "rmsnorm"}, {&k_add_rmsnorm, "add_rmsnorm"}, {&k_geglu, "geglu"},
        {&k_gemm_q8, "gemm_q8"}, {&k_hnr, "head_norm_rope"}, {&k_attn, "attention_enc"},
        {&k_gelu_mul, "gelu_mul"}, {&k_norm_add, "norm_add"}, {&k_scale, "scale"},
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

/* each of the T rows of x [T][n] */
static void rmsnorm(CUdeviceptr x, CUdeviceptr w, CUdeviceptr y, unsigned n, float eps,
                    unsigned T)
{
    void *args[] = {&x, &w, &y, &n, &eps};
    CHECK(cuLaunchKernel(k_rmsnorm, T, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* per row of T: x += rmsnorm(y) * w1, xn = rmsnorm(x) * w2 */
static void add_rmsnorm(CUdeviceptr y, CUdeviceptr w1, CUdeviceptr x, CUdeviceptr w2,
                        CUdeviceptr xn, unsigned n, float eps, unsigned T)
{
    void *args[] = {&y, &w1, &x, &w2, &xn, &n, &eps};
    CHECK(cuLaunchKernel(k_add_rmsnorm, T, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* per row of T: x = (x + rmsnorm(y) * w) * s */
static void norm_add(CUdeviceptr y, CUdeviceptr w, CUdeviceptr x, unsigned n, float eps, float s,
                     unsigned T)
{
    void *args[] = {&y, &w, &x, &n, &eps, &s};
    CHECK(cuLaunchKernel(k_norm_add, T, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* Y[t] (row stride ldy) = W . X[t] + bias + resid[t] for T tokens */
static void gemm_q8(const Q8W *w, CUdeviceptr X, CUdeviceptr bias, CUdeviceptr resid,
                    CUdeviceptr Y, unsigned ldy, unsigned T)
{
    unsigned rows = w->rows, cols = w->cols;
    void *args[] = {(void *)&w->q, (void *)&w->d, &X, &bias, &resid, &Y, &rows, &cols, &T, &ldy};
    CHECK(cuLaunchKernel(k_gemm_q8, rows / 64, (T + 31) / 32, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* q/k head norms + RoPE and v norms of qkv rows [T][stride] */
static void head_norm_rope(CUdeviceptr qkv, unsigned stride, CUdeviceptr qw, CUdeviceptr kw,
                           CUdeviceptr cs, CUdeviceptr pos, unsigned nh, unsigned nkv,
                           unsigned D, float eps, unsigned T)
{
    void *args[] = {&qkv, &stride, &qw, &kw, &cs, &pos, &nh, &nkv, &D, &eps};
    CHECK(cuLaunchKernel(k_hnr, nh + 2 * nkv, T, 1, 128, 1, 1, 0, NULL, args, NULL));
}

/* bidirectional attention within each token's text (seg), +-half_window
 * around the query when half_window != 0; `keys` bounds the keys per row */
static void attention_enc(CUdeviceptr qkv, unsigned stride, CUdeviceptr seg, CUdeviceptr out,
                          unsigned nh, unsigned nkv, unsigned D, unsigned half_window,
                          float scale, unsigned T, unsigned keys)
{
    void *args[] = {&qkv, &stride, &seg, &out, &nh, &nkv, &D, &half_window, &scale};
    CHECK(cuLaunchKernel(k_attn, nh, T, 1, 256, 1, 1, keys * 4, NULL, args, NULL));
}

/* out[t] = gelu(gu[t][0..n)) * gu[t][n..2n) for T tokens */
static void geglu(CUdeviceptr gu, CUdeviceptr out, unsigned n, unsigned T)
{
    void *args[] = {&gu, &out, &n, &T};
    launch_1d(k_geglu, n * T, args);
}

/* g[t][i] = gelu(g[t][i]) * p[t * pstride + i] */
static void gelu_mul(CUdeviceptr g, CUdeviceptr p, unsigned n, unsigned pstride, unsigned T)
{
    void *args[] = {&g, &p, &n, &pstride, &T};
    launch_1d(k_gelu_mul, n * T, args);
}

static void scale(CUdeviceptr x, unsigned n, float s)
{
    void *args[] = {&x, &n, &s};
    launch_1d(k_scale, n, args);
}

/* ------------------------------------------------------------------------ */
/* Model                                                                     */

enum { POOL_NONE = 0, POOL_MEAN = 1, POOL_CLS = 2, POOL_LAST = 3 };

typedef struct {
    unsigned n_layer, n_embd, n_head, n_ff, n_vocab, n_ple, n_out, max_ctx;
    unsigned n_kv[MAX_LAYERS], head_dim[MAX_LAYERS];
    int swa[MAX_LAYERS];
    unsigned head_dim_global, head_dim_swa, sliding_window, pooling;
    float eps, rope_base, rope_base_swa;
    int add_bos, add_eos;
} Config;

typedef struct {
    CUdeviceptr attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffw_norm, post_norm;
    Q8W wq, wk, wv, wo;
    Q8W gate_up;        /* gate rows then up rows, [2 n_ff][n_embd] */
    Q8W down;
    Q8W inp_gate, proj; /* per-layer input: [n_ple][n_embd], [n_embd][n_ple] */
    float out_scale;
} Layer;

typedef struct {
    Config c;
    unsigned batch;          /* max tokens per forward() call */
    Layer *layer;
    Q8W ple_proj;            /* [n_layer * n_ple][n_embd] */
    CUdeviceptr ple_norm, out_norm;
    Q8W out;                 /* [n_out][n_embd] */
    const GgufTensor *embd;  /* token embeddings in the mmap'ed file */
    CUdeviceptr rope_global, rope_swa; /* f32 [batch][head_dim/2][2] */
    /* activations, [batch][...] */
    CUdeviceptr x, xn, y, g, qkv, att, gu, h, ple, out_buf, pos, seg;
    float *h_x;
    uint32_t *h_pos, *h_seg;
} Model;

static int has_kv(const Gguf *g, const char *key)
{
    return gguf_kv(g, key) != NULL;
}

/* Element i of a per-layer metadata value: an array of u32/i32/bool, or a
 * scalar that applies to every layer. */
static uint32_t per_layer(const Gguf *g, const char *key, unsigned i)
{
    const GgufKV *kv = gguf_kv(g, key);
    if (!kv) {
        fprintf(stderr, "%s: missing metadata\n", key);
        exit(1);
    }
    if (kv->type != GGUF_ARRAY)
        return kv->type == GGUF_BOOL ? kv->val[0] : gguf_u32(g, key);
    if (i >= kv->arr_n) {
        fprintf(stderr, "%s: %lu values for %u layers\n", key, (unsigned long)kv->arr_n, i + 1);
        exit(1);
    }
    if (kv->arr_type == GGUF_BOOL || kv->arr_type == GGUF_U8 || kv->arr_type == GGUF_I8)
        return kv->val[i];
    uint32_t v;
    memcpy(&v, kv->val + 4 * i, 4);
    return v;
}

static int kv_bool(const Gguf *g, const char *key, int def)
{
    const GgufKV *kv = gguf_kv(g, key);
    return kv && kv->type == GGUF_BOOL ? kv->val[0] : def;
}

#define ARCH "gemma-embedding2"
#define K(name) ARCH "." name

static void read_config(const Gguf *g, Config *c)
{
    const GgufKV *arch = gguf_kv(g, "general.architecture");
    uint64_t len = 0;
    const uint8_t *p = arch ? arch->val : NULL;
    const char *name = p ? gguf_str(&p, &len) : NULL;
    if (!name || len != strlen(ARCH) || strncmp(name, ARCH, len)) {
        fprintf(stderr, "model architecture is not " ARCH "\n");
        exit(1);
    }
    c->n_layer = gguf_u32(g, K("block_count"));
    c->n_embd = gguf_u32(g, K("embedding_length"));
    c->n_ff = gguf_u32(g, K("feed_forward_length"));
    c->n_head = gguf_u32(g, K("attention.head_count"));
    c->n_ple = gguf_u32(g, K("embedding_length_per_layer_input"));
    c->n_out = gguf_u32(g, K("embedding_length_out"));
    c->max_ctx = gguf_u32(g, K("context_length"));
    c->head_dim_global = gguf_u32(g, K("attention.key_length"));
    c->head_dim_swa = gguf_u32(g, K("attention.key_length_swa"));
    c->sliding_window = gguf_u32(g, K("attention.sliding_window"));
    c->pooling = has_kv(g, K("pooling_type")) ? gguf_u32(g, K("pooling_type")) : POOL_MEAN;
    c->eps = gguf_f32(g, K("attention.layer_norm_rms_epsilon"));
    c->rope_base = gguf_f32(g, K("rope.freq_base"));
    c->rope_base_swa = gguf_f32(g, K("rope.freq_base_swa"));
    c->n_vocab = (unsigned)gguf_kv(g, "tokenizer.ggml.tokens")->arr_n;
    c->add_bos = kv_bool(g, "tokenizer.ggml.add_bos_token", 1);
    c->add_eos = kv_bool(g, "tokenizer.ggml.add_eos_token", 1);
    int ok = c->n_layer <= MAX_LAYERS && c->n_embd % 64 == 0 && c->n_ff % 64 == 0 &&
             c->n_ple % 64 == 0 && c->n_out % 64 == 0 && !kv_bool(g, K("attention.causal"), 0) &&
             c->pooling >= POOL_MEAN && c->pooling <= POOL_LAST &&
             gguf_u32(g, K("attention.value_length")) == c->head_dim_global &&
             gguf_u32(g, K("attention.value_length_swa")) == c->head_dim_swa &&
             gguf_u32(g, K("rope.dimension_count")) == c->head_dim_global &&
             gguf_u32(g, K("rope.dimension_count_swa")) == c->head_dim_swa;
    for (unsigned l = 0; ok && l < c->n_layer; l++) {
        c->swa[l] = per_layer(g, K("attention.sliding_window_pattern"), l) != 0;
        c->n_kv[l] = per_layer(g, K("attention.head_count_kv"), l);
        c->head_dim[l] = c->swa[l] ? c->head_dim_swa : c->head_dim_global;
        unsigned D = c->head_dim[l];
        ok = c->n_kv[l] && c->n_head % c->n_kv[l] == 0 && D % 256 == 0 && D <= MAX_HEAD_DIM &&
             c->n_head * D % 64 == 0 && c->n_kv[l] * D % 64 == 0;
    }
    if (!ok) {
        fprintf(stderr, "unsupported " ARCH " hyperparameters\n");
        exit(1);
    }
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
static const size_t STAGING_BYTES = 16 << 20;
/* All model memory is carved out of one cuMemAlloc'ed arena. */
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

/* Rows [r0, r0+nr) of t as Q8_0 blocks: straight from the file when t is
 * stored that way, else quantized into quant_buf. */
static const uint8_t *q8_blocks(const GgufTensor *t, size_t r0, size_t nr)
{
    size_t row = gguf_type_bytes(GGML_Q8_0, t->ne[0]);
    if (t->type == GGML_Q8_0)
        return t->data + r0 * row;
    if (t->type != GGML_BF16 && t->type != GGML_F16 && t->type != GGML_F32) {
        fprintf(stderr, "%s: tensor type %u is not supported\n", t->name, t->type);
        exit(1);
    }
    quantize_rows(GGML_Q8_0, t, r0, nr, quant_buf);
    return quant_buf;
}

static Q8W alloc_q8(unsigned rows, unsigned cols)
{
    return (Q8W){dalloc((size_t)rows * cols), dalloc((size_t)rows * (cols / 32) * 2), rows, cols};
}

/* matrix [rows][cols] -> rows row0 .. row0+rows-1 of the Q8 matrix w, in
 * staging-sized row chunks */
static void upload_q8(const Gguf *g, const char *name, unsigned cols, unsigned rows, Q8W *w,
                      unsigned row0)
{
    const GgufTensor *t = find_tensor(g, name, cols, rows);
    unsigned nb = cols / 32;
    size_t chunk = STAGING_BYTES / (nb * 34);
    for (size_t r0 = 0; r0 < rows; r0 += chunk) {
        size_t nr = rows - r0 < chunk ? rows - r0 : chunk;
        const uint8_t *src = q8_blocks(t, r0, nr);
        int8_t *q = staging;
        uint16_t *d = (uint16_t *)(q + nr * cols);
        for (size_t b = 0; b < nr * nb; b++) {
            memcpy(&d[b], src + b * 34, 2);
            memcpy(q + b * 32, src + b * 34 + 2, 32);
        }
        size_t r = row0 + r0;
        CHECK(cuMemcpyHtoD(w->q + r * cols, q, nr * cols));
        CHECK(cuMemcpyHtoD(w->d + r * nb * 2, d, nr * nb * 2));
    }
}

static Q8W load_q8(const Gguf *g, const char *name, unsigned cols, unsigned rows)
{
    Q8W w = alloc_q8(rows, cols);
    upload_q8(g, name, cols, rows, &w, 0);
    return w;
}

static size_t q8_bytes(size_t rows, size_t cols)
{
    return rows * cols + rows * (cols / 32) * 2 + 512; /* + alignment of the two arrays */
}

/* Floats per token of the activation buffers. */
static size_t token_floats(const Config *c)
{
    size_t qkv = 0, att = 0;
    for (unsigned l = 0; l < c->n_layer; l++) {
        size_t D = c->head_dim[l], q = c->n_head * D, kv = c->n_kv[l] * D;
        qkv = q + 2 * kv > qkv ? q + 2 * kv : qkv;
        att = q > att ? q : att;
    }
    return 4 * (size_t)c->n_embd + qkv + att + 3 * (size_t)c->n_ff +
           (size_t)c->n_layer * c->n_ple + c->n_out;
}

/* Expected device bytes for weights + activations. */
static size_t model_bytes(const Config *c, unsigned batch)
{
    size_t E = c->n_embd, F = c->n_ff, P = c->n_ple, sum = 0;
    for (unsigned l = 0; l < c->n_layer; l++) {
        size_t D = c->head_dim[l], Q = c->n_head * D, KV = c->n_kv[l] * D;
        sum += q8_bytes(Q, E) + 2 * q8_bytes(KV, E) + q8_bytes(E, Q) + q8_bytes(2 * F, E) +
               q8_bytes(E, F) + q8_bytes(P, E) + q8_bytes(E, P) + (7 * E + 2 * D) * 4 + 7 * 256;
    }
    sum += q8_bytes((size_t)c->n_layer * P, E) + q8_bytes(c->n_out, E) + (P + E) * 4 + 512;
    sum += (size_t)batch * (c->head_dim_global + c->head_dim_swa) * 4 + 512;
    sum += (size_t)batch * (token_floats(c) * 4 + 12) + 16 * 256;
    return sum + (1u << 20);
}

static void model_load(Model *m, const Gguf *g, unsigned batch)
{
    Config *c = &m->c;
    read_config(g, c);
    unsigned E = c->n_embd, F = c->n_ff, P = c->n_ple, L = c->n_layer;
    m->batch = batch;
    size_t need = model_bytes(c, batch), free_b, total_b;
    CHECK(cuMemGetInfo(&free_b, &total_b));
    if (cuMemAlloc(&arena, need) != CUDA_SUCCESS) {
        fprintf(stderr, "not enough GPU memory (%.2f GiB needed, %.2f GiB free); use a smaller "
                        "-b\n", need / 1073741824.0, free_b / 1073741824.0);
        exit(1);
    }
    arena_size = need;
    CHECK(cuMemHostAlloc(&staging, STAGING_BYTES, 0));
    quant_buf = malloc(STAGING_BYTES);

    double t0 = now_ms();
    m->layer = calloc(L, sizeof(Layer));
    char name[128];
#define T(fmt) (snprintf(name, sizeof(name), "blk.%u." fmt, l), name)
    for (unsigned l = 0; l < L; l++) {
        Layer *Ly = &m->layer[l];
        unsigned D = c->head_dim[l], Q = c->n_head * D, KV = c->n_kv[l] * D;
        Ly->attn_norm = upload_f32(g, T("attn_norm.weight"), E);
        Ly->q_norm = upload_f32(g, T("attn_q_norm.weight"), D);
        Ly->k_norm = upload_f32(g, T("attn_k_norm.weight"), D);
        Ly->post_attn_norm = upload_f32(g, T("post_attention_norm.weight"), E);
        Ly->ffn_norm = upload_f32(g, T("ffn_norm.weight"), E);
        Ly->post_ffw_norm = upload_f32(g, T("post_ffw_norm.weight"), E);
        Ly->post_norm = upload_f32(g, T("post_norm.weight"), E);
        Ly->wq = load_q8(g, T("attn_q.weight"), E, Q);
        Ly->wk = load_q8(g, T("attn_k.weight"), E, KV);
        Ly->wv = load_q8(g, T("attn_v.weight"), E, KV);
        Ly->wo = load_q8(g, T("attn_output.weight"), Q, E);
        Ly->gate_up = alloc_q8(2 * F, E);
        upload_q8(g, T("ffn_gate.weight"), E, F, &Ly->gate_up, 0);
        upload_q8(g, T("ffn_up.weight"), E, F, &Ly->gate_up, F);
        Ly->down = load_q8(g, T("ffn_down.weight"), F, E);
        Ly->inp_gate = load_q8(g, T("inp_gate.weight"), E, P);
        Ly->proj = load_q8(g, T("proj.weight"), P, E);
        const GgufTensor *s = gguf_tensor(g, T("layer_output_scale.weight"), GGML_F32, 1, 0, 0);
        memcpy(&Ly->out_scale, s->data, 4);
    }
#undef T
    m->ple_proj = load_q8(g, "per_layer_model_proj.weight", E, L * P);
    m->ple_norm = upload_f32(g, "per_layer_proj_norm.weight", P);
    m->out_norm = upload_f32(g, "output_norm.weight", E);
    m->out = load_q8(g, "output.weight", E, c->n_out);
    m->embd = find_tensor(g, "token_embd.weight", E, c->n_vocab);
    if (m->embd->type != GGML_Q8_0 && m->embd->type != GGML_BF16 &&
        m->embd->type != GGML_F16 && m->embd->type != GGML_F32) {
        fprintf(stderr, "token_embd.weight: type %u is not supported\n", m->embd->type);
        exit(1);
    }

    unsigned Dg = c->head_dim_global, Ds = c->head_dim_swa;
    float *cs = malloc((size_t)batch * (Dg > Ds ? Dg : Ds) * sizeof(float));
    m->rope_global = dalloc((size_t)batch * Dg * 4);
    m->rope_swa = dalloc((size_t)batch * Ds * 4);
    rope_table(Dg, batch, c->rope_base, cs);
    CHECK(cuMemcpyHtoD(m->rope_global, cs, (size_t)batch * Dg * 4));
    rope_table(Ds, batch, c->rope_base_swa, cs);
    CHECK(cuMemcpyHtoD(m->rope_swa, cs, (size_t)batch * Ds * 4));
    free(cs);

    size_t B = batch, qkv = 0, att = 0;
    for (unsigned l = 0; l < L; l++) {
        size_t D = c->head_dim[l], q = c->n_head * D, kv = c->n_kv[l] * D;
        qkv = q + 2 * kv > qkv ? q + 2 * kv : qkv;
        att = q > att ? q : att;
    }
    m->x = dalloc(B * E * 4);
    m->xn = dalloc(B * E * 4);
    m->y = dalloc(B * E * 4);
    m->g = dalloc(B * P * 4);
    m->qkv = dalloc(B * qkv * 4);
    m->att = dalloc(B * att * 4);
    m->gu = dalloc(B * 2 * F * 4);
    m->h = dalloc(B * F * 4);
    m->ple = dalloc(B * L * P * 4);
    m->out_buf = dalloc(B * c->n_out * 4);
    m->pos = dalloc(B * 4);
    m->seg = dalloc(B * 8);
    m->h_x = malloc(B * E * sizeof(float));
    m->h_pos = malloc(B * sizeof(uint32_t));
    m->h_seg = malloc(B * 2 * sizeof(uint32_t));
    CHECK(cuMemFreeHost(staging));
    free(quant_buf);
    fprintf(stderr, "loaded %u layers in %.1f s: %.2f GiB on the GPU (of %.2f GiB free), batch %u\n",
            L, (now_ms() - t0) / 1e3, dev_bytes / 1073741824.0, free_b / 1073741824.0, batch);
}

/* Embedding of a token: its row of the token matrix times sqrt(n_embd). */
static void embed_token(const Model *m, int32_t id, float *out)
{
    unsigned E = m->c.n_embd;
    const GgufTensor *t = m->embd;
    size_t row = gguf_type_bytes(t->type, E);
    const uint8_t *src = t->data + (size_t)id * row;
    if (t->type == GGML_Q8_0) {
        dequantize_q8_0(src, out, E);
    } else if (t->type == GGML_BF16) {
        for (unsigned i = 0; i < E; i++)
            out[i] = bf16_to_float(((const uint16_t *)src)[i]);
    } else if (t->type == GGML_F16) {
        for (unsigned i = 0; i < E; i++)
            out[i] = half_to_float(((const uint16_t *)src)[i]);
    } else {
        memcpy(out, src, E * 4);
    }
    float s = sqrtf((float)E);
    for (unsigned i = 0; i < E; i++)
        out[i] *= s;
}

/* --- debug dumps (EMB2_DUMP=1) --- */

static int dump_enabled;

static void dump(const char *name, int il, CUdeviceptr p, unsigned n)
{
    if (!dump_enabled)
        return;
    CHECK(cuCtxSynchronize());
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

/* Runs T (<= m->batch) tokens; m->h_pos / m->h_seg hold each token's
 * position in its text and its text's [start, end) in the batch. Fills
 * m->out_buf [T][n_out] (per-token output, before pooling). Dumps show the
 * last token. */
static void forward(Model *m, const int32_t *tokens, unsigned T)
{
    const Config *c = &m->c;
    unsigned E = c->n_embd, F = c->n_ff, P = c->n_ple, L = c->n_layer, last = T - 1;

    for (unsigned t = 0; t < T; t++)
        embed_token(m, tokens[t], m->h_x + (size_t)t * E);
    CHECK(cuMemcpyHtoD(m->x, m->h_x, (size_t)T * E * 4));
    CHECK(cuMemcpyHtoD(m->pos, m->h_pos, T * 4));
    CHECK(cuMemcpyHtoD(m->seg, m->h_seg, T * 8));
    unsigned keys = 0; /* longest text: bounds the keys of an attention row */
    for (unsigned t = 0; t < T; t++)
        if (m->h_seg[2 * t + 1] - m->h_seg[2 * t] > keys)
            keys = m->h_seg[2 * t + 1] - m->h_seg[2 * t];
    dump("inp_scaled", -1, m->x + (size_t)last * E * 4, E);

    /* per-layer inputs: rmsnorm(Wple . x / sqrt(E)) * ple_norm, [T][L][P] */
    gemm_q8(&m->ple_proj, m->x, 0, 0, m->ple, L * P, T);
    scale(m->ple, T * L * P, 1.0f / sqrtf((float)E));
    rmsnorm(m->ple, m->ple_norm, m->ple, P, c->eps, T * L);
    dump("inp_per_layer", -1, m->ple + (size_t)last * L * P * 4, L * P);

    rmsnorm(m->x, m->layer[0].attn_norm, m->xn, E, c->eps, T);
    for (unsigned l = 0; l < L; l++) {
        const Layer *Ly = &m->layer[l];
        unsigned D = c->head_dim[l], nkv = c->n_kv[l], Q = c->n_head * D, KV = nkv * D;
        unsigned QKV = Q + 2 * KV;
        /* attention */
        dump("attn_norm", l, m->xn + (size_t)last * E * 4, E);
        gemm_q8(&Ly->wq, m->xn, 0, 0, m->qkv, QKV, T);
        gemm_q8(&Ly->wk, m->xn, 0, 0, m->qkv + Q * 4, QKV, T);
        gemm_q8(&Ly->wv, m->xn, 0, 0, m->qkv + (Q + KV) * 4, QKV, T);
        head_norm_rope(m->qkv, QKV, Ly->q_norm, Ly->k_norm, c->swa[l] ? m->rope_swa : m->rope_global,
                       m->pos, c->n_head, nkv, D, c->eps, T);
        CUdeviceptr qkv_last = m->qkv + (size_t)last * QKV * 4;
        dump("Qcur_pos", l, qkv_last, Q);
        dump("Kcur_pos", l, qkv_last + Q * 4, KV);
        dump("Vcur_normed", l, qkv_last + (Q + KV) * 4, KV);
        unsigned hw = c->swa[l] ? c->sliding_window / 2 : 0;
        unsigned k = hw && 2 * hw + 1 < keys ? 2 * hw + 1 : keys;
        attention_enc(m->qkv, QKV, m->seg, m->att, c->n_head, nkv, D, hw, 1.0f, T, k);
        dump("kqv_out", l, m->att + (size_t)last * Q * 4, Q);
        gemm_q8(&Ly->wo, m->att, 0, 0, m->y, E, T);
        /* x += post_attn_norm(y); xn = ffn_norm(x) */
        add_rmsnorm(m->y, Ly->post_attn_norm, m->x, Ly->ffn_norm, m->xn, E, c->eps, T);
        dump("attn_out", l, m->x + (size_t)last * E * 4, E);
        dump("ffn_norm", l, m->xn + (size_t)last * E * 4, E);

        /* feed-forward: x += post_ffw_norm(down(gelu(gate . xn) * (up . xn))) */
        gemm_q8(&Ly->gate_up, m->xn, 0, 0, m->gu, 2 * F, T);
        geglu(m->gu, m->h, F, T);
        gemm_q8(&Ly->down, m->h, 0, 0, m->y, E, T);
        dump("ffn_out", l, m->y + (size_t)last * E * 4, E);
        norm_add(m->y, Ly->post_ffw_norm, m->x, E, c->eps, 1.0f, T);
        dump("pe_in", l, m->x + (size_t)last * E * 4, E);

        /* per-layer input: x = (x + post_norm(proj(gelu(inp_gate . x) * ple[l]))) * scale */
        gemm_q8(&Ly->inp_gate, m->x, 0, 0, m->g, P, T);
        gelu_mul(m->g, m->ple + (size_t)l * P * 4, P, L * P, T);
        gemm_q8(&Ly->proj, m->g, 0, 0, m->y, E, T);
        norm_add(m->y, Ly->post_norm, m->x, E, c->eps, Ly->out_scale, T);
        dump("l_out", l, m->x + (size_t)last * E * 4, E);

        CUdeviceptr next_norm = l + 1 < L ? m->layer[l + 1].attn_norm : m->out_norm;
        rmsnorm(m->x, next_norm, m->xn, E, c->eps, T);
    }
    dump("result_norm", -1, m->xn + (size_t)last * E * 4, E);
    gemm_q8(&m->out, m->xn, 0, 0, m->out_buf, c->n_out, T);
    dump("result_embd", -1, m->out_buf + (size_t)last * c->n_out * 4, c->n_out);
}

/* ------------------------------------------------------------------------ */
/* Embedding texts                                                           */

typedef struct {
    const char *prefix;  /* prepended to every text */
    unsigned dim;        /* keep the first dim values (0 = all) */
    int normalize;       /* L2-normalize the pooled vector */
} EmbedOpts;

/* Tokens of one text as the model sees it: <bos> text <eos>, at most max. */
static int text_tokens(const Model *m, const Tokenizer *tok, const char *prefix, const char *text,
                       unsigned max, int32_t **out)
{
    size_t pl = prefix ? strlen(prefix) : 0, tl = strlen(text);
    char *s = malloc(pl + tl + 1);
    memcpy(s, prefix, pl);
    memcpy(s + pl, text, tl + 1);
    int32_t *body;
    int n = tok_encode(tok, s, 1, &body);
    free(s);
    int32_t *ids = malloc((n + 2) * sizeof(int32_t));
    int k = 0;
    if (m->c.add_bos)
        ids[k++] = tok_special(tok, "<bos>");
    memcpy(ids + k, body, n * sizeof(int32_t));
    k += n;
    if (m->c.add_eos)
        ids[k++] = tok_special(tok, "<eos>");
    free(body);
    if ((unsigned)k > max) {
        fprintf(stderr, "warning: a text of %d tokens was cut to %u (raise -b)\n", k, max);
        if (m->c.add_eos)
            ids[max - 1] = ids[k - 1];
        k = max;
    }
    *out = ids;
    return k;
}

/* Embeds n texts into out [n][dim]; texts are packed into batches. Returns
 * the dimension and the number of tokens in *n_tokens. */
static unsigned embed_texts(Model *m, const Tokenizer *tok, const char *const *texts, int n,
                            const EmbedOpts *o, float **out, size_t *n_tokens)
{
    const Config *c = &m->c;
    unsigned N = c->n_out, dim = o->dim && o->dim < N ? o->dim : N;
    float *res = calloc((size_t)n * dim, sizeof(float));
    float *tok_out = malloc((size_t)m->batch * N * sizeof(float));
    int32_t *batch = malloc(m->batch * sizeof(int32_t));
    int32_t **ids = malloc(n * sizeof(int32_t *));
    int *len = malloc(n * sizeof(int));
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        len[i] = text_tokens(m, tok, o->prefix, texts[i], m->batch, &ids[i]);
        total += len[i];
    }
    for (int i0 = 0; i0 < n;) {
        /* pack texts i0 .. i1-1 */
        unsigned T = 0;
        int i1 = i0;
        while (i1 < n && T + len[i1] <= m->batch) {
            for (int k = 0; k < len[i1]; k++) {
                batch[T + k] = ids[i1][k];
                m->h_pos[T + k] = k;
                m->h_seg[2 * (T + k)] = T;
                m->h_seg[2 * (T + k) + 1] = T + len[i1];
            }
            T += len[i1++];
        }
        forward(m, batch, T);
        CHECK(cuMemcpyDtoH(tok_out, m->out_buf, (size_t)T * N * 4));
        for (int i = i0, start = 0; i < i1; start += len[i++]) {
            float *v = res + (size_t)i * dim;
            unsigned a = start, b = start + len[i];
            if (c->pooling == POOL_CLS)
                b = a + 1;
            else if (c->pooling == POOL_LAST)
                a = b - 1;
            for (unsigned t = a; t < b; t++)
                for (unsigned d = 0; d < dim; d++)
                    v[d] += tok_out[(size_t)t * N + d];
            for (unsigned d = 0; d < dim; d++)
                v[d] /= (float)(b - a);
            if (o->normalize) {
                double ss = 0;
                for (unsigned d = 0; d < dim; d++)
                    ss += (double)v[d] * v[d];
                float inv = ss > 0 ? (float)(1 / sqrt(ss)) : 0;
                for (unsigned d = 0; d < dim; d++)
                    v[d] *= inv;
            }
        }
        i0 = i1;
    }
    for (int i = 0; i < n; i++)
        free(ids[i]);
    free(ids), free(len), free(batch), free(tok_out);
    *out = res;
    if (n_tokens)
        *n_tokens = total;
    return dim;
}

/* ------------------------------------------------------------------------ */
/* Self-tests (CPU references; no model needed)                              */

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

static double gelu_ref(double g)
{
    return 0.5 * g * (1 + tanh(sqrt(2 / M_PI) * g * (1 + 0.044715 * g * g)));
}

static int test_gemm(void)
{
    const unsigned rows = 128, cols = 384, T = 37, off = 32, ldy = rows + 64;
    float *w = malloc(rows * cols * 4), *wq = malloc(rows * cols * 4);
    float *x = malloc(T * cols * 4), *y = malloc(T * ldy * 4);
    uint8_t *blk = malloc(rows * cols / 32 * 34);
    for (unsigned i = 0; i < rows * cols; i++)
        w[i] = rng_uniform(-1, 1) * (i % 97 == 0 ? 8 : 1);
    for (unsigned i = 0; i < T * cols; i++)
        x[i] = rng_uniform(-2, 2);
    for (unsigned i = 0; i < T * ldy; i++)
        y[i] = rng_uniform(-1, 1);
    quantize_q8_0(w, blk, (size_t)rows * cols);
    dequantize_q8_0(blk, wq, (size_t)rows * cols);
    double *ref = malloc(T * ldy * sizeof(double));
    for (unsigned i = 0; i < T * ldy; i++)
        ref[i] = y[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned r = 0; r < rows; r++) {
            double acc = 0;
            for (unsigned k = 0; k < cols; k++)
                acc += (double)wq[r * cols + k] * x[t * cols + k];
            ref[t * ldy + off + r] = acc; /* Y = W . X, no bias / residual */
        }
    int8_t *q = malloc(rows * cols);
    uint16_t *d = malloc(rows * cols / 32 * 2);
    for (unsigned b = 0; b < rows * cols / 32; b++) {
        memcpy(&d[b], blk + b * 34, 2);
        memcpy(q + b * 32, blk + b * 34 + 2, 32);
    }
    Q8W g8 = {upload(q, rows * cols), upload(d, rows * cols / 32 * 2), rows, cols};
    CUdeviceptr d_x = upload(x, T * cols * 4), d_y = upload(y, T * ldy * 4);
    gemm_q8(&g8, d_x, 0, 0, d_y + off * 4, ldy, T);
    CHECK(cuMemcpyDtoH(y, d_y, T * ldy * 4));
    int fails = report("gemm_q8 (37 tokens, column slice)", rel_err(y, ref, T * ldy), 1e-5);
    cuMemFree(g8.q), cuMemFree(g8.d), cuMemFree(d_x), cuMemFree(d_y);
    free(w), free(wq), free(x), free(y), free(blk), free(ref), free(q), free(d);
    return fails;
}

/* head norms + RoPE for both head sizes, with q, k and v heads */
static int test_head_norm_rope(unsigned D)
{
    const unsigned nh = 4, nkv = D == 512 ? 1 : 2, T = 5, stride = (nh + 2 * nkv) * D + 64;
    const float eps = 1e-6f, base = D == 512 ? 1e6f : 1e4f;
    const unsigned npos = 300;
    float *qkv = malloc(T * stride * 4), *qw = malloc(D * 4), *kw = malloc(D * 4);
    float *cs = malloc((size_t)npos * D * 4);
    uint32_t pos[5] = {0, 1, 7, 150, 299};
    for (unsigned i = 0; i < T * stride; i++)
        qkv[i] = rng_uniform(-3, 3);
    for (unsigned i = 0; i < D; i++) {
        qw[i] = rng_uniform(0, 2);
        kw[i] = rng_uniform(0, 2);
    }
    rope_table(D, npos, base, cs);
    double *ref = malloc(T * stride * sizeof(double));
    for (unsigned i = 0; i < T * stride; i++)
        ref[i] = qkv[i];
    for (unsigned t = 0; t < T; t++)
        for (unsigned h = 0; h < nh + 2 * nkv; h++) {
            double *x = ref + t * stride + h * D, ss = 0;
            for (unsigned i = 0; i < D; i++)
                ss += x[i] * x[i];
            double inv = 1 / sqrt(ss / D + eps);
            const float *w = h < nh ? qw : kw;
            for (unsigned i = 0; i < D; i++)
                x[i] *= inv * (h < nh + nkv ? w[i] : 1);
            if (h >= nh + nkv)
                continue;
            /* the same f32 cos/sin table (ggml's rope cache), checked below */
            const float *r = cs + (size_t)pos[t] * D;
            for (unsigned i = 0; i < D / 2; i++) {
                double a = x[i], b = x[i + D / 2];
                x[i] = a * r[2 * i] - b * r[2 * i + 1];
                x[i + D / 2] = a * r[2 * i + 1] + b * r[2 * i];
            }
        }
    CUdeviceptr d_qkv = upload(qkv, T * stride * 4), d_qw = upload(qw, D * 4),
                d_kw = upload(kw, D * 4), d_cs = upload(cs, (size_t)npos * D * 4),
                d_pos = upload(pos, sizeof(pos));
    head_norm_rope(d_qkv, stride, d_qw, d_kw, d_cs, d_pos, nh, nkv, D, eps, T);
    CHECK(cuMemcpyDtoH(qkv, d_qkv, T * stride * 4));
    char name[64];
    snprintf(name, sizeof(name), "head_norm_rope (%u-dim heads, q/k/v)", D);
    int fails = report(name, rel_err(qkv, ref, T * stride), 1e-6);
    /* the table itself vs exact angles: f32 theta at position ~300 rad */
    double terr = 0;
    for (unsigned p = 0; p < npos; p++)
        for (unsigned i = 0; i < D / 2; i++) {
            double th = p * pow(base, -2.0 * i / D);
            terr = fmax(terr, fabs(cs[(size_t)p * D + 2 * i] - cos(th)));
            terr = fmax(terr, fabs(cs[(size_t)p * D + 2 * i + 1] - sin(th)));
        }
    snprintf(name, sizeof(name), "rope table (%u dims, base %.0e, 300 positions)", D, base);
    fails += report(name, terr, 1e-4);
    cuMemFree(d_qkv), cuMemFree(d_qw), cuMemFree(d_kw), cuMemFree(d_cs), cuMemFree(d_pos);
    free(qkv), free(qw), free(kw), free(cs), free(ref);
    return fails;
}

/* two texts packed in one batch, windowed and full, grouped-query */
static int test_attention(unsigned D, unsigned hw)
{
    const unsigned nh = 4, nkv = D == 512 ? 1 : 2, len0 = 23, len1 = 41, T = len0 + len1;
    const unsigned stride = (nh + 2 * nkv) * D;
    const float sc = 0.05f;
    float *qkv = malloc(T * stride * 4), *out = malloc(T * nh * D * 4);
    uint32_t *seg = malloc(T * 8);
    for (unsigned i = 0; i < T * stride; i++)
        qkv[i] = rng_uniform(-2, 2);
    for (unsigned t = 0; t < T; t++) {
        seg[2 * t] = t < len0 ? 0 : len0;
        seg[2 * t + 1] = t < len0 ? len0 : T;
    }
    double *ref = malloc(T * nh * D * sizeof(double)), *s = malloc(T * sizeof(double));
    for (unsigned t = 0; t < T; t++)
        for (unsigned h = 0; h < nh; h++) {
            unsigned kvh = h / (nh / nkv), lo = seg[2 * t], hi = seg[2 * t + 1];
            if (hw) {
                lo = t >= lo + hw ? t - hw : lo;
                hi = t + hw + 1 < hi ? t + hw + 1 : hi;
            }
            const float *q = qkv + t * stride + h * D;
            double mx = -1e300, sum = 0;
            for (unsigned j = lo; j < hi; j++) {
                const float *k = qkv + j * stride + nh * D + kvh * D;
                double dot = 0;
                for (unsigned i = 0; i < D; i++)
                    dot += (double)q[i] * k[i];
                s[j] = dot * sc;
                mx = fmax(mx, s[j]);
            }
            for (unsigned j = lo; j < hi; j++)
                sum += s[j] = exp(s[j] - mx);
            for (unsigned i = 0; i < D; i++) {
                double acc = 0;
                for (unsigned j = lo; j < hi; j++)
                    acc += s[j] * qkv[j * stride + (nh + nkv) * D + kvh * D + i];
                ref[(t * nh + h) * D + i] = acc / sum;
            }
        }
    CUdeviceptr d_qkv = upload(qkv, T * stride * 4), d_seg = upload(seg, T * 8), d_out;
    CHECK(cuMemAlloc(&d_out, T * nh * D * 4));
    attention_enc(d_qkv, stride, d_seg, d_out, nh, nkv, D, hw, sc, T, len1);
    CHECK(cuMemcpyDtoH(out, d_out, T * nh * D * 4));
    char name[80];
    snprintf(name, sizeof(name), "attention_enc (%u-dim heads, 2 texts, window %s)", D,
             hw ? "+-5" : "off");
    int fails = report(name, rel_err(out, ref, T * nh * D), 1e-5);
    cuMemFree(d_qkv), cuMemFree(d_seg), cuMemFree(d_out);
    free(qkv), free(out), free(seg), free(ref), free(s);
    return fails;
}

static int test_elementwise(void)
{
    const unsigned n = 512, T = 3, L = 4, pstride = L * n;
    const float eps = 1e-6f, s = 0.75f;
    float *y = malloc(T * n * 4), *x = malloc(T * n * 4), *w = malloc(n * 4);
    for (unsigned i = 0; i < T * n; i++) {
        y[i] = rng_uniform(-5, 5);
        x[i] = rng_uniform(-50, 50);
    }
    for (unsigned i = 0; i < n; i++)
        w[i] = rng_uniform(0, 3);
    double *rx = malloc(T * n * sizeof(double));
    for (unsigned t = 0; t < T; t++) {
        double ss = 0;
        for (unsigned i = 0; i < n; i++)
            ss += (double)y[t * n + i] * y[t * n + i];
        ss = 1 / sqrt(ss / n + eps);
        for (unsigned i = 0; i < n; i++)
            rx[t * n + i] = (x[t * n + i] + y[t * n + i] * ss * w[i]) * s;
    }
    CUdeviceptr d_y = upload(y, T * n * 4), d_x = upload(x, T * n * 4), d_w = upload(w, n * 4);
    norm_add(d_y, d_w, d_x, n, eps, s, T);
    CHECK(cuMemcpyDtoH(x, d_x, T * n * 4));
    int fails = report("norm_add (residual, scaled)", rel_err(x, rx, T * n), 1e-6);
    cuMemFree(d_y), cuMemFree(d_x), cuMemFree(d_w);

    float *g = malloc(T * n * 4), *p = malloc(T * pstride * 4);
    double *rg = malloc(T * n * sizeof(double));
    for (unsigned i = 0; i < T * n; i++)
        g[i] = rng_uniform(-6, 6);
    for (unsigned i = 0; i < T * pstride; i++)
        p[i] = rng_uniform(-2, 2);
    const unsigned layer = 2;
    for (unsigned t = 0; t < T; t++)
        for (unsigned i = 0; i < n; i++)
            rg[t * n + i] = gelu_ref(g[t * n + i]) * p[t * pstride + layer * n + i];
    CUdeviceptr d_g = upload(g, T * n * 4), d_p = upload(p, T * pstride * 4);
    gelu_mul(d_g, d_p + layer * n * 4, n, pstride, T);
    CHECK(cuMemcpyDtoH(g, d_g, T * n * 4));
    fails += report("gelu_mul (per-layer input slice)", rel_err(g, rg, T * n), 1e-6);
    scale(d_p, T * pstride, s);
    float *ps = malloc(T * pstride * 4);
    double *rp = malloc(T * pstride * sizeof(double));
    for (unsigned i = 0; i < T * pstride; i++)
        rp[i] = (double)p[i] * s;
    CHECK(cuMemcpyDtoH(ps, d_p, T * pstride * 4));
    fails += report("scale", rel_err(ps, rp, T * pstride), 1e-7);
    cuMemFree(d_g), cuMemFree(d_p);
    free(y), free(x), free(w), free(rx), free(g), free(p), free(rg), free(ps), free(rp);
    return fails;
}

static int self_test(void)
{
    printf("== kernel self-tests against CPU references\n");
    int fails = test_gemm() + test_head_norm_rope(256) + test_head_norm_rope(512) +
                test_attention(256, 5) + test_attention(512, 0) + test_elementwise();
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
    CHECK(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 0));
    CHECK(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 0));
    CHECK(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 0));
    fprintf(stderr, "device: %s\n", name);
    load_kernels(cubin);
}

/* The default model: the Q8_0 file if it exists, else the BF16 one. */
static const char *default_model(void)
{
    return access(DEFAULT_MODEL, R_OK) && !access(BF16_MODEL, R_OK) ? BF16_MODEL : DEFAULT_MODEL;
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
 * (no <bos> / <eos>; control-token text is parsed only with --special) */
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

/* Non-empty lines of a file (exits if unreadable); the file text is kept. */
static int read_lines(const char *path, char ***lines)
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
    int k = 0, cap = 64;
    char **v = malloc(cap * sizeof(char *));
    for (char *p = s; *p;) {
        char *e = strchr(p, '\n');
        if (e)
            *e = 0;
        if (*p) {
            if (k == cap)
                v = realloc(v, (cap *= 2) * sizeof(char *));
            v[k++] = p;
        }
        if (!e)
            break;
        p = e + 1;
    }
    *lines = v;
    return k;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [options] TEXT...           embed each TEXT\n"
            "       %s [options] -f FILE           embed each non-empty line of FILE\n"
            "       %s sim [options] TEXT...       cosine similarity matrix of the TEXTs\n"
            "       %s test                        kernel self-tests\n"
            "       %s tokenize TEXT [model.gguf] | tokenize-ids [--special] [model.gguf] < lines\n"
            "options:\n"
            "  -m FILE        model (default " DEFAULT_MODEL ", else " BF16_MODEL ")\n"
            "  -k FILE        kernels (default " DEFAULT_CUBIN ")\n"
            "  -b N           tokens per batch and longest text (default %u, max %u)\n"
            "  --prefix STR   text put before every input (e.g. a task prompt)\n"
            "  --dim N        keep the first N values (Matryoshka truncation), then normalize\n"
            "  --no-norm      do not L2-normalize the embeddings\n"
            "  -o FILE        write the embeddings as raw f32 [texts][dim]\n"
            "  --full         print every value (default: the first 8 and the norm)\n",
            prog, prog, prog, prog, prog, DEFAULT_BATCH, MAX_BATCH);
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "tokenize"))
        return cmd_tokenize(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "tokenize-ids"))
        return cmd_tokenize_ids(argc, argv);

    const char *model_path = NULL, *cubin = DEFAULT_CUBIN, *out_path = NULL;
    unsigned batch = DEFAULT_BATCH;
    int test = 0, sim = 0, full = 0, n = 0;
    EmbedOpts o = {.normalize = 1};
    const char **texts = malloc(argc * sizeof(char *));
    char **lines = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
#define OPT(name) (!strcmp(a, name) && v && ++i)
        if (OPT("-m"))
            model_path = v;
        else if (OPT("-k"))
            cubin = v;
        else if (OPT("-b"))
            batch = atoi(v);
        else if (OPT("--prefix"))
            o.prefix = v;
        else if (OPT("--dim"))
            o.dim = atoi(v);
        else if (OPT("-o"))
            out_path = v;
        else if (OPT("-f")) {
            int k = read_lines(v, &lines);
            texts = realloc(texts, (argc + k) * sizeof(char *));
            for (int j = 0; j < k; j++)
                texts[n++] = lines[j];
        } else if (!strcmp(a, "--no-norm"))
            o.normalize = 0;
        else if (!strcmp(a, "--full"))
            full = 1;
        else if (i == 1 && !strcmp(a, "test"))
            test = 1;
        else if (i == 1 && !strcmp(a, "sim"))
            sim = 1;
        else if (a[0] == '-' && a[1])
            usage(argv[0]);
        else
            texts[n++] = a;
#undef OPT
    }
    if (test) {
        init_cuda(cubin);
        return self_test() ? 1 : 0;
    }
    if (!n || batch < 16 || batch > MAX_BATCH)
        usage(argv[0]);
    dump_enabled = getenv("EMB2_DUMP") && atoi(getenv("EMB2_DUMP"));

    init_cuda(cubin);
    Gguf g;
    gguf_open(&g, model_path ? model_path : default_model());
    Tokenizer *tok = tok_load(&g);
    Model m = {0};
    model_load(&m, &g, batch);

    float *e;
    size_t n_tok;
    double t0 = now_ms();
    unsigned dim = embed_texts(&m, tok, texts, n, &o, &e, &n_tok);
    double ms = now_ms() - t0;
    fprintf(stderr, "%d texts, %zu tokens in %.1f ms (%.0f tokens/s)\n", n, n_tok, ms,
            n_tok / (ms / 1e3));

    if (out_path) {
        FILE *f = fopen(out_path, "wb");
        if (!f || fwrite(e, sizeof(float), (size_t)n * dim, f) != (size_t)n * dim || fclose(f)) {
            fprintf(stderr, "cannot write %s\n", out_path);
            return 1;
        }
        fprintf(stderr, "wrote %d x %u floats to %s\n", n, dim, out_path);
    }
    if (sim) {
        printf("cosine similarity:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0, a = 0, b = 0;
                for (unsigned d = 0; d < dim; d++) {
                    dot += (double)e[(size_t)i * dim + d] * e[(size_t)j * dim + d];
                    a += (double)e[(size_t)i * dim + d] * e[(size_t)i * dim + d];
                    b += (double)e[(size_t)j * dim + d] * e[(size_t)j * dim + d];
                }
                printf(" %6.3f", a > 0 && b > 0 ? dot / sqrt(a * b) : 0);
            }
            printf("   %.60s\n", texts[i]);
        }
    } else if (!out_path || full) {
        for (int i = 0; i < n; i++) {
            const float *v = e + (size_t)i * dim;
            double ss = 0;
            for (unsigned d = 0; d < dim; d++)
                ss += (double)v[d] * v[d];
            printf("embedding %d (%u values, norm %.4f): ", i, dim, sqrt(ss));
            unsigned shown = full ? dim : 8;
            for (unsigned d = 0; d < shown; d++)
                printf(d ? " %.6f" : "%.6f", v[d]);
            printf(full ? "\n" : " ...\n");
        }
    }
    free(e), free(texts), free(lines);
    tok_free(tok);
    gguf_close(&g);
    return 0;
}
